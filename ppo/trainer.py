import torch
import re
import collections
import time
import gc
import math
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, GenerationConfig
from sentence_transformers import SentenceTransformer

from trl import PPOTrainer
from trl.trainer.utils import (
    empty_cache, 
    pad_to_length, 
    forward, 
    selective_log_softmax, 
    first_true_indices,
    truncate_response,
)
from trl.models.utils import unwrap_model_for_generation
from trl.core import masked_mean, masked_whiten

# 定义一个在官方代码中使用的常量
INVALID_LOGPROB = 1.0
MAX_INTERACTION_STEPS = 7

class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens):
        self.tokenizer = tokenizer
        self.stop_token_ids = []
        for token in stop_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                self.stop_token_ids.append(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] > 0 and input_ids[0, -1].item() in self.stop_token_ids:
            return True
        return False

class E5VectorRetriever:
    def __init__(self, paragraphs: List[dict], model: SentenceTransformer):
        self.raw_paragraphs = paragraphs
        self.model = model
        self.corpus = [f'passage: {p.get("paragraph_text", "")}' for p in paragraphs]
        if not self.corpus: self.corpus_embeddings = None; return
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=False)
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query: return []
        query_with_prefix = f'query: {query}'
        query_embedding = self.model.encode(query_with_prefix, convert_to_tensor=True)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        corpus_embeddings_norm = torch.nn.functional.normalize(self.corpus_embeddings, p=2, dim=1)
        cos_scores = torch.mm(query_embedding.unsqueeze(0), corpus_embeddings_norm.transpose(0, 1))[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus)))
        retrieved_docs_with_info = []
        for score, idx in zip(top_results[0], top_results[1]):
            paragraph = self.raw_paragraphs[idx]
            retrieved_docs_with_info.append(f"Document {paragraph['idx']} (Title: {paragraph['title']}): {paragraph['paragraph_text']}")
        return retrieved_docs_with_info

def calculate_f1_score(prediction: str, ground_truth_list: list) -> float:
    prediction_tokens = prediction.lower().split()
    best_f1 = 0.0
    for ground_truth in ground_truth_list:
        ground_truth_tokens = ground_truth.lower().split()
        if not prediction_tokens or not ground_truth_tokens: continue
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1: best_f1 = f1
    return best_f1


class RAGEnv:
    def __init__(self, raw_data_item: dict, similarity_model: SentenceTransformer):
        self.question = raw_data_item["question"]
        self.final_answer_list = [raw_data_item.get("answer", "")] + raw_data_item.get("answer_aliases", [])
        self.retriever = E5VectorRetriever(raw_data_item["paragraphs"], similarity_model)
        
        self.state = f"<|im_start|>user\n{self.question}<|im_end|>\n<|im_start|>assistant\n"
        self.steps = 0
        self.is_done = False

    def step(self, response_text: str) -> tuple[str, bool]:
        self.steps += 1
        self.state += response_text
        
        if self.state.strip().endswith("<retrieval>"):
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", response_text, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = self.retriever.retrieve(subquery, top_k=3)
                retrieved_docs_text = "\n".join(retrieved_docs)
                self.state += f"\n{retrieved_docs_text}\n</retrieval>\n"
        
        if (self.state.strip().endswith("<|im_end|>") and "<answer>" in response_text) or self.steps >= MAX_INTERACTION_STEPS:
            self.is_done = True
            
        return self.state, self.is_done

class RAGPPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        self.reward_tokenizer = kwargs.pop('reward_tokenizer')
        self.retrieval_model = kwargs.pop('retrieval_model')
        super().__init__(*args, **kwargs)
        self.model = self.policy_model
        if hasattr(self, 'ref_model') and self.ref_model is not None:
            if self.ref_model is not self.policy_model:
                del self.ref_model
                gc.collect()
                torch.cuda.empty_cache()
            
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        tokenizer = self.processing_class
        model = self.model
        rm_tokenizer = self.reward_tokenizer
        reward_model = self.reward_model
        retrieval_model = self.retrieval_model
        dataloader = self.dataloader
        device = accelerator.device
        model.train()

        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches

        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        
        for update, data in enumerate(dataloader, 1):
            self.state.episode += args.batch_size
            # print("="*50)
            # print(f"--- Starting Update Step {update} ---")
            # torch.cuda.empty_cache()
            # gc.collect()
            # print(f"Memory after models loaded (before rollout): {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

            # --- 1. Rollout Phase ---
            with torch.no_grad():
                envs = [RAGEnv({key: data[key][i] for key in data}, retrieval_model) for i in range(args.local_batch_size)]

                all_step_responses = [[] for _ in envs]
                all_step_rewards = [[] for _ in envs]
                query_tensors = [tokenizer.encode(env.state, return_tensors="pt").to(device)[0] for env in envs]

                with unwrap_model_for_generation(self.policy_model, self.accelerator) as unwrapped_model:
                    for _ in range(MAX_INTERACTION_STEPS):
                        active_envs_indices = [i for i, env in enumerate(envs) if not env.is_done]
                        if not active_envs_indices: 
                            break

                        active_query_tensors = [query_tensors[i] for i in active_envs_indices]
                        padded_queries = tokenizer.pad({"input_ids": active_query_tensors}, padding=True, return_tensors="pt").to(device)
                        
                        gen_config = GenerationConfig(
                            max_new_tokens=128, 
                            pad_token_id=tokenizer.pad_token_id, 
                            do_sample=True, 
                            temperature=0.9, 
                            top_k=50
                        )
                        response_outputs = unwrapped_model.generate(
                            **padded_queries, 
                            generation_config=gen_config, 
                            stopping_criteria=StoppingCriteriaList([StopOnKeywords(tokenizer, ["<retrieval>", "<|im_end|>"])])
                        )

                        for i, env_idx in enumerate(active_envs_indices):
                            env = envs[env_idx]
                            query_len = padded_queries['input_ids'][i].ne(tokenizer.pad_token_id).sum()
                            response_tensor = response_outputs[i][query_len:]
                            response_text = tokenizer.decode(response_tensor, skip_special_tokens=False)

                            rm_inputs = rm_tokenizer([env.state + response_text], return_tensors='pt', padding=True, truncation=True).to(device)
                            reward = reward_model(**rm_inputs).logits[0]
                            
                            new_state_text, _ = env.step(response_text)
                            query_tensors[env_idx] = tokenizer.encode(new_state_text, return_tensors="pt").to(device)[0]

                            all_step_responses[env_idx].append(response_tensor)
                            all_step_rewards[env_idx].append(reward)
                
                del query_tensors, unwrapped_model, padded_queries, response_outputs
                gc.collect()
                torch.cuda.empty_cache()

                # --- 2. Post-processing and Reward Calculation ---
                queries, responses, rewards_list = [], [], []
                for k in range(len(envs)):
                    if not all_step_responses[k]: 
                        continue
                    
                    initial_query_text = f"<|im_start|>user\n{envs[k].question}<|im_end|>\n<|im_start|>assistant\n"
                    queries.append(tokenizer.encode(initial_query_text, return_tensors="pt").to(device)[0])
                    
                    full_response = torch.cat(all_step_responses[k])
                    responses.append(full_response)
                    
                    rewards = torch.zeros_like(full_response, dtype=torch.float, device=device)
                    current_idx = 0
                    for i, resp_step in enumerate(all_step_responses[k]):
                        step_len = len(resp_step)
                        if step_len > 0:
                            rewards[current_idx + step_len - 1] = all_step_rewards[k][i]
                            current_idx += step_len
                    
                    final_match = re.search(r"<answer>(.*?)</answer>", envs[k].state, re.DOTALL)
                    rewards[-1] = 5.0 * calculate_f1_score(final_match.group(1).strip(), envs[k].final_answer_list) if final_match else -2.0
                    rewards_list.append(rewards)

                print(f"============== Rollout Data for Update {update} ==============")
                for i in range(len(envs)):
                    if not all_step_responses[i]: 
                        continue
                    
                    print(f"----- Sample {i+1} -----")
                    # 完整的对话历史
                    print(f"Full Conversation History:\n{envs[i].state}")
                    
                    # 分步的响应和奖励
                    for j, resp_tensor in enumerate(all_step_responses[i]):
                        response_text = tokenizer.decode(resp_tensor, skip_special_tokens=False)
                        reward_value = all_step_rewards[i][j].item()
                        print(f"  Step {j+1} Response: {response_text.strip()} | Reward: {reward_value:.4f}")
                    
                    # 最终的奖励
                    final_reward = rewards_list[i][-1].item()
                    print(f"  Final Reward: {final_reward:.4f}")
                    print("-" * 20)
                print("=" * 60 + "\n")
                # ******************************************************************

                # --- 3. Prepare Tensors for PPO Update ---
                max_len = max(len(r) for r in responses)
                if max_len == 0:
                    log.warning(f"Skipping update {update} due to no valid responses.")
                    continue
                responses = torch.stack([pad_to_length(r, max_len, tokenizer.pad_token_id) for r in responses])
                queries = tokenizer.pad({"input_ids": queries}, padding=True, return_tensors="pt")["input_ids"].to(device)
                rewards = torch.stack([pad_to_length(rw, max_len, 0) for rw in rewards_list])
                
                query_responses = torch.cat((queries, responses), dim=1)
                context_length = queries.shape[1]

                all_forward_outputs = forward(self.model, query_responses, tokenizer.pad_token_id)
                logits, vpred_temp = all_forward_outputs[0], all_forward_outputs[2]
                logprobs = selective_log_softmax(logits[:, context_length - 1 : -1], responses)
                values = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                
                ref_logprobs = logprobs.detach()

                sequence_lengths = first_true_indices((responses == tokenizer.pad_token_id)) - 1
                padding_mask = torch.arange(responses.shape[1], device=device)[None, :] > sequence_lengths[:, None]
                
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards += non_score_reward

                # --- 4. GAE (Generalized Advantage Estimation) ---
                lastgaelam = 0
                advantages_reversed = []
                for t in reversed(range(responses.shape[1])):
                    nextvalues = values[:, t + 1] if t < responses.shape[1] - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)

                del all_forward_outputs, logits, vpred_temp, ref_logprobs, non_score_reward, delta
                gc.collect()
                torch.cuda.empty_cache()

            # --- 5. PPO Optimization Phase ---
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    
                    with accelerator.accumulate(self.model):
                        mb_advantage = advantages[mini_batch_inds]
                        mb_responses = responses[mini_batch_inds]
                        mb_query_responses = query_responses[mini_batch_inds]
                        mb_logprobs = logprobs[mini_batch_inds]
                        mb_return = returns[mini_batch_inds]

                        # final_sequence_length = mb_query_responses.shape[1]
                        # print(f"!!!!!!!! [Process {accelerator.process_index}] PRE-FORWARD CHECK: Final sequence length is {final_sequence_length} tokens. !!!!!!!!!!")
                        # print(f"--- [Process {accelerator.process_index}] Detailed Memory Summary BEFORE forward: ---")
                        # print(torch.cuda.memory_summary(device=accelerator.device))
                        # print(f"---------------------------------------------------------------------------------")

                        all_forward_outputs = forward(self.model, mb_query_responses, tokenizer.pad_token_id)
                        logits, vpred_temp = all_forward_outputs[0], all_forward_outputs[2]
                        new_logprobs = selective_log_softmax(logits[:, context_length - 1 : -1], mb_responses)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), ~padding_mask[mini_batch_inds])

                        vpredclipped = torch.clamp(vpred, values[mini_batch_inds] - args.cliprange_value, values[mini_batch_inds] + args.cliprange_value)
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), ~padding_mask[mini_batch_inds])

                        loss = pg_loss + args.vf_coef * vf_loss
                        
                        accelerator.backward(loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        del all_forward_outputs, logits, vpred_temp, new_logprobs, vpred, logprobs_diff, ratio
                        del pg_losses, pg_losses2, vpredclipped, vf_losses1, vf_losses2, loss
                        del mb_advantage, mb_responses, mb_query_responses, mb_logprobs, mb_return, mb_values, mb_mask
            
            # --- 6. Logging and Cleanup ---
            self.lr_scheduler.step()
            self.state.global_step += 1

            mean_kl = masked_mean(kl, ~padding_mask).item()
            mean_f1 = np.mean(final_f1_scores)
            mean_reward_final = masked_mean(rewards, ~padding_mask).item()
            
            metrics = {
                "ppo/loss": loss.item(),
                "ppo/pg_loss": pg_loss.item(),
                "ppo/vf_loss": vf_loss.item(),
                "objective/kl": mean_kl,
                "objective/f1_score": mean_f1,
                "objective/reward": mean_reward_final,
                "time/update": time.time() - start_time,
                "lr": self.lr_scheduler.get_last_lr()[0],
            }
            self.log(metrics)

            del advantages, returns, query_responses, queries, responses, rewards_list, rewards, padding_mask
            del logprobs, values, kl
            gc.collect()
            torch.cuda.empty_cache()

        print("PPO training finished.")