import re
import collections
from typing import List, Dict, Any
import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from trl import GRPOTrainer

MAX_INTERACTION_STEPS = 7
NUM_ROLLOUTS = 2

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
        if not self.corpus: 
            self.corpus_embeddings = None
            return
        self.corpus_embeddings = self.model.encode(
            self.corpus, convert_to_tensor=True, show_progress_bar=False
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query: 
            return []
        query_with_prefix = f'query: {query}'
        query_embedding = self.model.encode(query_with_prefix, convert_to_tensor=True, show_progress_bar=False)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        corpus_embeddings_norm = torch.nn.functional.normalize(self.corpus_embeddings, p=2, dim=1)
        cos_scores = torch.mm(query_embedding.unsqueeze(0), corpus_embeddings_norm.transpose(0, 1))[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus)))
        retrieved_docs_with_info = []
        for score, idx in zip(top_results[0], top_results[1]):
            paragraph = self.raw_paragraphs[idx]
            retrieved_docs_with_info.append(
                f"Document {paragraph['idx']} (Title: {paragraph['title']}): {paragraph['paragraph_text']}"
            )
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
        device = self.accelerator.device

        all_prompts_texts: List[str] = []
        completion_texts: List[str] = []  # length = batch * NUM_ROLLOUTS
        completion_step_deltas: List[List[str]] = []  # per completion, list of step deltas
        completion_step_rewards: List[List[float]] = []  # per completion, per-step rewards

        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )
        stop_criteria = StoppingCriteriaList([StopOnKeywords(self.processing_class, ["<retrieval>", "<|im_end|>"])])

        # 1) Rollouts: collect step deltas and step rewards
        for data_item in inputs:
            prompt_text = None
            for k in range(NUM_ROLLOUTS):
                env = RAGEnv(data_item, self.retrieval_model)
                if prompt_text is None:
                    prompt_text = env.state

                # For GRPO grouping we append the prompt per rollout (same prompt repeated)
                all_prompts_texts.append(env.state)

                step_deltas: List[str] = []
                step_rewards: List[float] = []

                for step_idx in range(MAX_INTERACTION_STEPS):
                    # encode current state
                    query_tensor = self.processing_class.encode(env.state, return_tensors="pt").to(device)

                    with torch.no_grad():
                        response_tensor = self.accelerator.unwrap_model(self.model).generate(
                            query_tensor,
                            generation_config=gen_config,
                            stopping_criteria=stop_criteria,
                        )

                    # generated tokens relative to prompt
                    response_slice = response_tensor[0, query_tensor.shape[1]:]
                    response_text = self.processing_class.decode(response_slice, skip_special_tokens=False)

                    # advance environment -> get delta (response + retrieval docs if any), new_state includes retrieval
                    delta_text, new_state, done = env.step(response_text)

                    # compute step reward based on post-action new_state (includes retrieval docs)
                    rm_inputs = self.rm_tokenizer(new_state, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
                    with torch.no_grad():
                        rm_outputs = self.reward_model(**rm_inputs)
                        step_reward = float(rm_outputs.logits.squeeze().cpu().numpy().item())

                    # if done and final answer exists, add final F1 bonus/penalty
                    if done:
                        final_match = re.search(r"<answer>(.*?)</answer>", new_state, re.DOTALL)
                        if final_match:
                            final_f1 = 5.0 * calculate_f1_score(final_match.group(1).strip(), env.final_answer_list)
                        else:
                            final_f1 = -2.0
                        step_reward += final_f1

                    step_deltas.append(delta_text)
                    step_rewards.append(step_reward)

                    if done:
                        break

                completion_text = "".join(step_deltas)  # full completion including retrieval docs
                completion_texts.append(completion_text)
                completion_step_deltas.append(step_deltas)
                completion_step_rewards.append(step_rewards)

        total_completions = len(completion_texts)
        if total_completions == 0:
            raise RuntimeError("No completions generated in rollouts.")

        # 2) Compute per-step group baselines and advantages
        # find max steps among completions
        max_steps = max(len(sr) for sr in completion_step_rewards)
        device = self.accelerator.device

        # build rewards matrix with NaN where step missing
        rewards_matrix = torch.full((total_completions, max_steps), float('nan'), device=device, dtype=torch.float32)
        for i, step_rewards in enumerate(completion_step_rewards):
            for t, r in enumerate(step_rewards):
                rewards_matrix[i, t] = float(r)

        # comp_step_adv_matrix: same shape, filled with advantages (0 where NaN)
        comp_step_adv_matrix = torch.zeros_like(rewards_matrix)
        comp_step_mask = ~torch.isnan(rewards_matrix)
        for t in range(max_steps):
            col = rewards_matrix[:, t]
            mask = ~torch.isnan(col)
            if mask.sum() == 0:
                comp_step_adv_matrix[:, t] = 0.0
                continue
            vals = col[mask]
            mean = vals.mean()
            std = vals.std(unbiased=False) + 1e-8
            advs = (vals - mean) / std
            col_adv = torch.zeros_like(col)
            col_adv[mask] = advs
            comp_step_adv_matrix[:, t] = col_adv

        # 3) Tokenize prompts and completions (use add_special_tokens=False for per-step tokenization consistency)
        prompt_inputs = self.processing_class(text=all_prompts_texts, return_tensors="pt", padding=True, padding_side="left").to(device)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        completion_inputs = self.processing_class(text=completion_texts, return_tensors="pt", padding=True, padding_side="right").to(device)
        completion_ids = completion_inputs["input_ids"]

        # truncate completions if too long
        if completion_ids.shape[1] > self.max_completion_length:
            completion_ids = completion_ids[:, : self.max_completion_length]

        # 4) Map per-step advantages to per-token advantages by tokenizing each delta with add_special_tokens=False
        per_token_advantages = torch.zeros_like(completion_ids, dtype=torch.float32, device=device)
        per_token_mask = torch.zeros_like(completion_ids, dtype=torch.bool, device=device)

        for i, step_deltas in enumerate(completion_step_deltas):
            offset = 0
            for t, delta_text in enumerate(step_deltas):
                # tokenize delta_text without special tokens to get token count
                tok = self.processing_class(delta_text, return_tensors="pt", add_special_tokens=False)
                tok_ids = tok["input_ids"][0]
                L = min(len(tok_ids), completion_ids.shape[1] - offset)
                if L <= 0:
                    break
                # assign this step's advantage to these tokens
                per_token_advantages[i, offset:offset+L] = comp_step_adv_matrix[i, t]
                per_token_mask[i, offset:offset+L] = True
                offset += L
                if offset >= completion_ids.shape[1]:
                    break

        # 5) Build prompt+completion concat ids & attention mask for later per-token logprobs
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, (completion_ids != self.processing_class.pad_token_id).long()], dim=1)

        # 6) compute reference per-token logprobs if needed (same as before)
        ref_per_token_logps = None
        if getattr(self, "beta", 0.0) != 0.0:
            with torch.no_grad():
                if getattr(self, "ref_model", None) is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, completion_ids.shape[1]
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask, completion_ids.shape[1]
                        )

        # Flatten for multi-process gather if needed
        advantages_flat = per_token_advantages.view(-1)
        token_mask_flat = per_token_mask.view(-1)
        advantages_flat = self.accelerator.gather(advantages_flat)
        token_mask_flat = self.accelerator.gather(token_mask_flat)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": per_token_mask,
            "advantages_per_token": per_token_advantages,
            "advantages_flat": advantages_flat,
            "token_mask_flat": token_mask_flat,
            "ref_per_token_logps": ref_per_token_logps,
            "prompt_completion_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
        }