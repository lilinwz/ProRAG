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

# --- 辅助类和函数 (保持不变) ---
# ... (此处省略 StopOnKeywords, E5VectorRetriever, calculate_f1_score, RAGEnv 的代码)
class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_token_ids = []
        for token in stop_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                self.stop_token_ids.extend(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] > 0:
            for stop_id in self.stop_token_ids:
                if input_ids[0, -1].item() == stop_id:
                    return True
        return False

class E5VectorRetriever:
    def __init__(self, paragraphs: List[dict], model: SentenceTransformer, device):
        self.model = model
        self.device = device
        self.raw_paragraphs = paragraphs
        self.corpus = [f'passage: {p.get("paragraph_text", "")}' for p in paragraphs]
        if not self.corpus: self.corpus_embeddings = None; return
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        self.corpus_embeddings = torch.nn.functional.normalize(self.corpus_embeddings, p=2, dim=1)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query: return []
        query_with_prefix = f'query: {query}'
        query_embedding = self.model.encode(query_with_prefix, convert_to_tensor=True, device=self.device)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        cos_scores = torch.mm(query_embedding.unsqueeze(0), self.corpus_embeddings.transpose(0, 1))[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus)))
        retrieved_docs_with_info = []
        for score, idx in zip(top_results[0], top_results[1]):
            paragraph = self.raw_paragraphs[idx.item()]
            retrieved_docs_with_info.append(f"Document {paragraph.get('idx', '')} (Title: {paragraph.get('title', '')}): {paragraph.get('paragraph_text', '')}")
        return retrieved_docs_with_info

def calculate_f1_score(prediction: str, ground_truth_list: List[str]) -> float:
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
    return f1

class RAGEnv:
    def __init__(self, data_item: dict, retriever: E5VectorRetriever):
        self.question = data_item["question"]
        self.final_answer_list = [data_item.get("answer", "")] + data_item.get("answer_aliases", [])
        self.retriever = retriever
        self.state_text = f"<|im_start|>user\n{self.question}<|im_end|>\n<|im_start|>assistant\n"
        self.steps = 0
        self.is_done = False
        self.max_steps = 7

    def step(self, response_text: str) -> Tuple[str, bool]:
        self.steps += 1
        self.state_text += response_text
        if self.state_text.strip().endswith("<retrieval>"):
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", response_text, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = self.retriever.retrieve(subquery, top_k=3)
                retrieved_docs_text = "\n".join(retrieved_docs)
                self.state_text += f"\n{retrieved_docs_text}\n</retrieval>\n"
        if (self.state_text.strip().endswith("<|im_end|>") and "<answer>" in response_text) or self.steps >= self.max_steps:
            self.is_done = True
        return self.state_text, self.is_done
# --- 主训练器类 ---

class RAGPPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        self.reward_tokenizer = kwargs.pop('reward_tokenizer')
        self.retrieval_model = kwargs.pop('retrieval_model')
        super().__init__(*args, **kwargs)

    def train(self):
        # --- 1. 初始化 (大部分从官方PPOTrainer.train复制) ---
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device
        iter_dataloader = iter(self.dataloader)
        
        # trainer state initialization
        # ... (这部分与官方代码一致)
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # --- 2. 主训练循环 ---
        for update in tqdm(range(1, args.num_total_batches + 1), desc="PPO Steps"):
            self.state.episode += args.batch_size
            
            # === Rollout 阶段 (这是我们自定义的部分) ===
            with torch.no_grad():
                # 2.1. 通过多步交互生成轨迹
                batch = next(iter_dataloader)
                
                envs = [RAGEnv(
                    {key: batch[key][i] for key in batch},
                    E5VectorRetriever(batch["paragraphs"][i], self.retrieval_model, self.accelerator.device)
                ) for i in range(args.local_batch_size)]

                all_step_responses = [[] for _ in envs]
                all_step_rewards = [[] for _ in envs]
                query_tensors = [self.tokenizer.encode(env.state_text, return_tensors="pt").to(device)[0] for env in envs]

                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    # ... (这部分多步批量生成逻辑保持不变)
                    for _ in range(7):
                        active_envs_indices = [i for i, env in enumerate(envs) if not env.is_done]
                        if not active_envs_indices: break

                        active_query_tensors = [query_tensors[i] for i in active_envs_indices]
                        padded_queries = self.tokenizer.pad({"input_ids": active_query_tensors}, padding=True, return_tensors="pt").to(device)
                        
                        gen_config = GenerationConfig(max_new_tokens=128, pad_token_id=self.tokenizer.pad_token_id, do_sample=True, temperature=0.9, top_k=50, stopping_criteria=StoppingCriteriaList([StopOnKeywords(self.tokenizer, ["<retrieval>", "<|im_end|>"])]))
                        response_outputs = unwrapped_model.generate(**padded_queries, generation_config=gen_config)

                        for i, env_idx in enumerate(active_envs_indices):
                            env = envs[env_idx]
                            query_len = padded_queries['input_ids'][i].ne(self.tokenizer.pad_token_id).sum()
                            response_tensor = response_outputs[i][query_len:]
                            response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                            reward = self.reward_model(self.reward_tokenizer([env.state_text + response_text], return_tensors='pt', padding=True, truncation=True).to(device))[0].logits[0]
                            new_state_text, _ = env.step(response_text)
                            query_tensors[env_idx] = self.tokenizer.encode(new_state_text, return_tensors="pt").to(device)[0]
                            all_step_responses[env_idx].append(response_tensor)
                            all_step_rewards[env_idx].append(reward)
                
                # 2.2. 整理轨迹数据 (queries, responses, rewards)
                queries, responses, rewards_list = [], [], []
                for k in range(len(envs)):
                    if not all_step_responses[k]: continue
                    
                    initial_query_text = f"<|im_start|>user\n{envs[k].question}<|im_end|>\n<|im_start|>assistant\n"
                    queries.append(self.tokenizer.encode(initial_query_text, return_tensors="pt").to(device)[0])
                    
                    full_response = torch.cat(all_step_responses[k])
                    responses.append(full_response)
                    
                    rewards = torch.zeros_like(full_response, dtype=torch.float, device=device)
                    current_idx = 0
                    for i, resp_step in enumerate(all_step_responses[k]):
                        step_len = len(resp_step)
                        if step_len > 0:
                            rewards[current_idx + step_len - 1] = all_step_rewards[k][i]
                            current_idx += step_len
                    
                    final_match = re.search(r"<answer>(.*?)</answer>", envs[k].state_text, re.DOTALL)
                    rewards[-1] = 5.0 * calculate_f1_score(final_match.group(1).strip(), envs[k].final_answer_list) if final_match else -2.0
                    rewards_list.append(rewards)

                # 2.3. 将数据 padding 成矩形张量
                max_len = max(len(r) for r in responses)
                responses = torch.stack([pad_to_length(r, max_len, self.tokenizer.pad_token_id) for r in responses])
                queries = self.tokenizer.pad({"input_ids": queries}, padding=True, return_tensors="pt")["input_ids"]
                rewards = torch.stack([pad_to_length(rw, max_len, 0) for rw in rewards_list])
                
                # 2.4. 计算 logprobs, ref_logprobs, values (复制自官方代码)
                query_responses = torch.cat((queries, responses), dim=1)
                context_length = queries.shape[1]

                # Policy model forward pass
                output, vpred_temp = forward(self.model, query_responses, self.tokenizer.pad_token_id)
                logits = output.logits[:, context_length - 1 : -1]
                logprobs = selective_log_softmax(logits, responses)
                values = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                
                # Ref model forward pass
                with self.null_ref_context():
                    ref_output, _ = forward(self.model, query_responses, self.tokenizer.pad_token_id)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logprobs = selective_log_softmax(ref_logits, responses)

                # 2.5. 计算最终奖励和优势 (复制自官方代码)
                sequence_lengths = first_true_indices((responses == self.tokenizer.pad_token_id)) - 1
                padding_mask = torch.arange(responses.shape[1], device=device)[None, :] > sequence_lengths[:, None]
                
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards += non_score_reward

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

            # === 优化阶段 (从官方PPOTrainer.train原封不动地复制) ===
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    
                    with accelerator.accumulate(self.model):
                        # ... 此处是完整的 PPO 损失计算和优化步骤 ...
                        # (直接从您贴出的 PPOTrainer 源码中复制 'for ppo_epoch_idx' 循环内部的所有内容)
                        mb_advantage = advantages[mini_batch_inds]
                        mb_responses = responses[mini_batch_inds]
                        mb_query_responses = query_responses[mini_batch_inds]
                        mb_logprobs = logprobs[mini_batch_inds]
                        mb_return = returns[mini_batch_inds]

                        output, vpred_temp = forward(self.model, mb_query_responses, self.tokenizer.pad_token_id)
                        logits = output.logits[:, context_length - 1 : -1]
                        new_logprobs = selective_log_softmax(logits, mb_responses)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        
                        # ... (省略vf_loss, pg_loss, loss = ..., accelerator.backward(loss)等计算)
                        # ... 这些都从官方代码复制 ...
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
            
            # --- 日志和收尾 (从官方代码复制) ---
            self.lr_scheduler.step()
            self.log({"ppo/loss": loss.item(), "ppo/pg_loss": pg_loss.item(), "ppo/vf_loss": vf_loss.item()})
            # ... (其他日志)

        print("PPO training finished.")