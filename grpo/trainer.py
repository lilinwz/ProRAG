import torch
import re
from typing import List, Dict, Any
from transformers import GenerationConfig, StoppingCriteriaList
from trl import GRPOTrainer

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

class RAGTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self.retrieval_model = kwargs.pop('retrieval_model')
        self.rm_tokenizer = kwargs.pop('rm_tokenizer')
        self.reward_model = kwargs.pop('reward_model')
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"--- Starting Interactive Rollout for {len(inputs)} prompts ---")

        all_prompts = []
        all_completions = []
        all_rewards = []

        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )
        stop_criteria = StoppingCriteriaList([StopOnKeywords(self.processing_class, ["<retrieval>", "<|im_end|>"])])

        for data_item in inputs:
            env = RAGEnv(data_item, self.retrieval_model)
            trajectory = []

            for _ in range(MAX_INTERACTION_STEPS):
                query_tensor = self.processing_class.encode(env.state, return_tensors="pt").to(self.accelerator.device)

                with torch.no_grad():
                    response_tensor = self.accelerator.unwrap_model(self.model).generate(
                        query_tensor,
                        generation_config=gen_config,
                        stopping_criteria=stop_criteria,
                    )
                
                response_tensor = response_tensor[0, query_tensor.shape[1]:]
                response_text = self.processing_class.decode(response_tensor, skip_special_tokens=False)
                
                state_for_reward = env.state + response_text
                inputs = rm_tokenizer(state_for_reward, return_tensors='pt', padding=True, max_length=4096, truncation=True).to(reward_model.device)
                with torch.no_grad():
                    step_reward = self.reward_model(**rm_inputs).logits.squeeze().item()
                
                trajectory.append({"prompt": env.state, "response": response_tensor, "reward": step_reward})
                
                new_state, is_done = env.step(response_text)
                if is_done:
                    final_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
                    if final_match:
                        final_f1_reward = 5.0 * calculate_f1_score(final_match.group(1).strip(), env.final_answer_list)
                    else:
                        final_f1_reward = -2.0
                    step_reward += final_f1_reward
                    break

            
            all_prompts_text.append(initial_prompt)
            all_completions_text.append(full_completion_text)

        # 4. 调用父类的原始方法来处理评分和后续步骤
        # 这是一个技巧：我们用交互式生成的结果 "欺骗" 原始的 GRPOTrainer，
        # 让它以为这些 completions 是用简单 generate 生成的。
        # 为此，我们需要重新构建 `inputs` 字典，因为原始方法需要它来计算奖励
        
        # 重新构造 `inputs` 列表，其长度现在是 batch_size * num_generations
        # 这是必要的，因为奖励函数可能需要原始数据项中的其他信息（例如 `answer`）
        extended_inputs = []
        for item in inputs:
            for _ in range(self.num_generations):
                 extended_inputs.append(item)

        # 准备好所有需要的信息后，调用父类的 `_calculate_rewards_and_prepare_for_loss`
        # （这是一个假设的方法名，我们需要查看 GRPOTrainer 源码找到正确的方法）
        # 经过查阅 trl 源码，后续逻辑都在 _generate_and_score_completions 方法内部，
        # 所以我们不能简单调用父类方法，而是需要复制其后续逻辑。

        # --- 从这里开始，我们复制 GRPOTrainer 的标准流程 ---
        
        # 将 prompts 和 completions 编码为 tensor
        prompt_inputs = self.processing_class(text=all_prompts_text, return_tensors="pt", padding=True, padding_side="left").to(self.accelerator.device)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        completion_ids = self.processing_class(text=all_completions_text, return_tensors="pt", padding=True, padding_side="right").to(self.accelerator.device)["input_ids"]
        
        # 截断到 max_completion_length
        if completion_ids.shape[1] > self.max_completion_length:
            completion_ids = completion_ids[:, :self.max_completion_length]

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        # 创建 completion mask
        is_eos = (completion_ids == self.eos_token_id)
        eos_indices = torch.argmax(is_eos.int(), dim=1)
        eos_indices[~is_eos.any(dim=1)] = completion_ids.shape[1]
        completion_mask = torch.arange(completion_ids.shape[1], device=self.accelerator.device)[None, :] < eos_indices[:, None]
        
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # 计算奖励
        rewards_per_func = self._calculate_rewards(extended_inputs, all_prompts_text, all_completions_text, [])
        rewards = (rewards_per_func * self.reward_weights.to(self.accelerator.device).unsqueeze(0)).nansum(dim=1)
        rewards = self.accelerator.gather(rewards)
        
        # 计算优势
        mean_rewards = rewards.view(-1, self.num_generations).mean(dim=1, keepdim=True)
        advantages = rewards.view(-1, self.num_generations) - mean_rewards
        advantages = advantages.view(-1)
        
        # 切片以获取当前进程的数据
        process_slice = slice(self.accelerator.process_index * len(all_prompts_text), (self.accelerator.process_index + 1) * len(all_prompts_text))
        advantages = advantages[process_slice]
        
        # 计算参考模型的 logprobs
        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(self.ref_model, prompt_completion_ids, attention_mask, completion_ids.shape[1])
                else: # 使用 PEFT disable_adapter
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(self.model, prompt_completion_ids, attention_mask, completion_ids.shape[1])

        # 返回给 compute_loss 方法所需的所有数据
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "ref_per_token_logps": ref_per_token_logps
        }