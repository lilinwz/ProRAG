import torch
import re
import collections
import time
import gc
import math
import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, GenerationConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
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
        query_embedding = self.model.encode(query_with_prefix, convert_to_tensor=True, show_progress_bar=False)
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
                retrieved_docs = self.retriever.retrieve(subquery, top_k=1)
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
            
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        tokenizer = self.processing_class
        rm_tokenizer = self.reward_tokenizer
        model = self.model
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
            start_time = time.time()
            # print("="*50)
            # print(f"--- Starting Update Step {update} ---")
            # torch.cuda.empty_cache()
            # gc.collect()
            # print(f"Memory after models loaded (before rollout): {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

            # --- 1. Rollout Phase ---
            reward_model.to(device)
            retrieval_model.to(device)
            with torch.no_grad():
                envs = [RAGEnv({key: data[key][i] for key in data}, retrieval_model) for i in range(args.local_batch_size)]

                all_step_responses = [[] for _ in envs]
                all_step_rewards = [[] for _ in envs]
                
                with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                    gen_config = GenerationConfig(
                        max_new_tokens=512, 
                        temperature=0.7, 
                        top_k=0.0,
                        top_p=1.0,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    stopping_criteria = StoppingCriteriaList([StopOnKeywords(tokenizer, ["<retrieval>", "<|im_end|>"])])

                    for _ in range(MAX_INTERACTION_STEPS):
                        active_envs_indices = [i for i, env in enumerate(envs) if not env.is_done]
                        if not active_envs_indices: 
                            break

                        for env_idx in active_envs_indices:
                            env = envs[env_idx]
                            query_tensor = tokenizer.encode(env.state, return_tensors="pt").to(device)
                            response_outputs = unwrapped_model.policy.generate(
                                input_ids=query_tensor, 
                                generation_config=gen_config, 
                                stopping_criteria=stopping_criteria
                            )

                            query_len = query_tensor.shape[1]
                            response_tensor = response_outputs[0, query_len:]
                            response_text = tokenizer.decode(response_tensor, skip_special_tokens=False)

                            rm_inputs = rm_tokenizer([env.state + response_text], return_tensors='pt', padding=True, truncation=True).to(device)
                            reward = reward_model(**rm_inputs).logits[0]
                            
                            new_state_text, _ = env.step(response_text)

                            accelerator.print(f"Full response length for env {env_idx}: {len(response_text)} tokens.")
                            accelerator.print(response_text)
                            accelerator.print("+==============================+")
                            
                            all_step_responses[env_idx].append(response_tensor)
                            all_step_rewards[env_idx].append(reward)
                
                del response_outputs
                reward_model.to('cpu')
                retrieval_model.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()

                # --- 2. Post-processing and Reward Calculation ---
                queries, responses, rewards_list = [], [], []
                final_scores = []
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
                    final_scores.append(rewards[-1].item())
                    rewards_list.append(rewards)

                # --- 3. Prepare Tensors for PPO Update ---
                # max_len = max(len(r) for r in responses)
                # if max_len == 0:
                #     log.warning(f"Skipping update {update} due to no valid responses.")
                #     continue
                # responses = torch.stack([pad_to_length(r, max_len, tokenizer.pad_token_id) for r in responses])
                # queries = tokenizer.pad({"input_ids": queries}, padding=True, return_tensors="pt")["input_ids"].to(device)
                # rewards = torch.stack([pad_to_length(rw, max_len, 0) for rw in rewards_list])
                
                # query_responses = torch.cat((queries, responses), dim=1)
                # context_length = queries.shape[1]

                local_max_len = max(len(r) for r in responses) if responses else 0
                local_max_len_tensor = torch.tensor(local_max_len, device=accelerator.device)
                all_max_lens = [torch.tensor(0, device=accelerator.device) for _ in range(accelerator.num_processes)]
                torch.distributed.all_gather(all_max_lens, local_max_len_tensor)
                global_max_len = max(t.item() for t in all_max_lens)

                if global_max_len == 0:
                    accelerator.print(f"Skipping update {update} due to no valid responses across all processes.")
                    continue
                
                responses = torch.stack([pad_to_length(r, global_max_len, tokenizer.pad_token_id) for r in responses])
                queries = tokenizer.pad({"input_ids": queries}, padding=True, return_tensors="pt")["input_ids"].to(device)
                rewards = torch.stack([pad_to_length(rw, global_max_len, 0) for rw in rewards_list])
                
                query_responses = torch.cat((queries, responses), dim=1)
                context_length = queries.shape[1]

                policy_output, vpred_temp = forward(model, query_responses, tokenizer.pad_token_id)
                logits = policy_output.logits
                logprobs = selective_log_softmax(logits[:, context_length - 1 : -1], responses)
                values = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                
                with torch.no_grad():
                    with self.null_ref_context(): 
                        ref_output, _ = forward(model, query_responses, tokenizer.pad_token_id)
                    ref_logits = ref_output.logits
                    ref_logprobs = selective_log_softmax(ref_logits[:, context_length - 1 : -1], responses)

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

                mean_kl = masked_mean(kl, ~padding_mask).item()
                mean_reward_final = masked_mean(rewards, ~padding_mask).item()

                del policy_output, vpred_temp, ref_output, ref_logits, non_score_reward, kl, rewards
                del ref_logprobs, advantages_reversed, lastgaelam, delta, all_step_responses, all_step_rewards
                gc.collect()
                torch.cuda.empty_cache()

            # --- 5. PPO Optimization Phase ---
            accelerator.print(f"Starting PPO optimization.")
            total_pg_loss = 0
            total_vf_loss = 0
            mini_batch_count = 0
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    
                    with accelerator.accumulate(model):
                        mb_advantage = advantages[mini_batch_inds]
                        mb_responses = responses[mini_batch_inds]
                        mb_query_responses = query_responses[mini_batch_inds]
                        mb_logprobs = logprobs[mini_batch_inds]
                        mb_return = returns[mini_batch_inds]
                        mb_values = values[mini_batch_inds] # <-- 添加 mb_values
                        mb_mask = ~padding_mask[mini_batch_inds] # <-- 添加 mb_mask

                        # final_sequence_length = mb_query_responses.shape[1]
                        # print(f"!!!!!!!! [Process {accelerator.process_index}] PRE-FORWARD CHECK: Final sequence length is {final_sequence_length} tokens. !!!!!!!!!!")
                        # print(f"--- [Process {accelerator.process_index}] Detailed Memory Summary BEFORE forward: ---")
                        # print(torch.cuda.memory_summary(device=accelerator.device))
                        # print(f"---------------------------------------------------------------------------------")

                        policy_output, vpred_temp = forward(model, mb_query_responses, tokenizer.pad_token_id)
                        logits = policy_output.logits
                        new_logprobs = selective_log_softmax(logits[:, context_length - 1 : -1], mb_responses)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mb_mask)

                        vpredclipped = torch.clamp(vpred, mb_values - args.cliprange_value, mb_values + args.cliprange_value)
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mb_mask)

                        loss = pg_loss + args.vf_coef * vf_loss

                        total_pg_loss += pg_loss.item()
                        total_vf_loss += vf_loss.item()
                        mini_batch_count += 1
                        
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                        del policy_output, vpred_temp, logits, new_logprobs, vpred
                        del logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss
                        del vpredclipped, vf_losses1, vf_losses2, vf_loss, loss
                        del mb_advantage, mb_responses, mb_query_responses, mb_logprobs, mb_return, mb_values, mb_mask
            
            # --- 6. Logging and Cleanup ---
            mean_pg_loss = total_pg_loss / mini_batch_count if mini_batch_count > 0 else 0
            mean_vf_loss = total_vf_loss / mini_batch_count if mini_batch_count > 0 else 0
            mean_ppo_loss = mean_pg_loss + args.vf_coef * mean_vf_loss
            mean_f1 = np.mean(final_scores)
            
            metrics = {
                "ppo/loss": mean_ppo_loss,
                "ppo/pg_loss": mean_pg_loss,
                "ppo/vf_loss": mean_vf_loss,
                "objective/kl": mean_kl,
                "objective/f1_score": mean_f1,
                "objective/reward": mean_reward_final,
                "time/update": time.time() - start_time,
                "lr": self.lr_scheduler.get_last_lr()[0],
            }
            self.log(metrics)
            self.state.global_step += 1

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.co