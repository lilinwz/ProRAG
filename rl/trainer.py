import re
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Dict, Any
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from trl import GRPOTrainer

MAX_INTERACTION_STEPS = 10

class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens: List[str], lookback_tokens: int=3):
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens
        self.lookback_tokens = lookback_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] == 0:
            return False
        last_ids = input_ids[0, -self.lookback_tokens:].tolist()
        decoded = self.tokenizer.decode(last_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for tok in self.stop_tokens:
            if decoded.endswith(tok):
                return True
        return False

class E5VectorRetriever:
    def __init__(self, paragraphs: Dict[str, List], model: SentenceTransformer):
        self.model = model
        self.titles = [item["title"] for item in paragraphs]
        self.passages = [item["paragraph_text"] for item in paragraphs]
        self.corpus = [f"Title: {t}\n{p}" for t, p in zip(self.titles, self.passages)]

        if not self.corpus:
            self.corpus_embeddings = None
            return

        emb = self.model.encode(self.corpus, convert_to_tensor=False, show_progress_bar=False)
        self.corpus_embeddings = np.asarray(emb).astype(np.float32)
        self.corpus_embeddings /= np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-12

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query:
            return []
        
        q_emb = self.model.encode(f"query: {query}", convert_to_tensor=False, show_progress_bar=False)
        q_emb = np.asarray(q_emb).astype(np.float32)
        q_emb /= np.linalg.norm(q_emb) + 1e-12

        scores = np.dot(self.corpus_embeddings, q_emb)
        k = min(top_k, len(scores))
        idxs = np.argsort(-scores)[:k]
        return [f"Title: {self.titles[idx]}\n{self.passages[idx]}" for idx in idxs]

class RAGEnv:
    def __init__(self, data_item: dict, similarity_model: SentenceTransformer):
        self.data_item = data_item
        self.state = data_item["init_prompt"]
        self.retriever = E5VectorRetriever(data_item.get("paragraphs", []), similarity_model)
        self.is_done = False
        self.steps = 0
        self.generated_text_history = ""

    def step(self, new_generated_text: str) -> tuple[str, bool]:
        self.steps += 1
        self.generated_text_history += new_generated_text
        self.state += new_generated_text
        
        sq_match = re.search(r"<subquery>(.*?)</subquery>", new_generated_text, re.DOTALL)
        if sq_match:
            query_content = sq_match.group(1).strip()
            retrieved_doc = self.retriever.retrieve(query_content, top_k=1)
            
            retrieval_block = f"\n{retrieved_doc}\n</retrieval>\n"
            self.state += retrieval_block
            self.generated_text_history += retrieval_block
            
            return self.state, False

        if "<answer>" in new_generated_text or "<|im_end|>" in new_generated_text:
            self.is_done = True
        
        if self.steps >= MAX_INTERACTION_STEPS:
            self.is_done = True
            
        return self.state, self.is_done

def format_reward(completion):
    TAG_MAP = {
        "<step>":      ("</step>",      "S"),
        "<subquery>":  ("</subquery>",  "Q"),
        "<retrieval>": ("</retrieval>", "R"),
        "<subanswer>": ("</subanswer>", "A"),
        "<answer>":    ("</answer>",    "F")
    }
    CYCLE_PATTERN = ["S", "Q", "R", "S", "A"]
    END_PATTERN = ["S", "F"]
    
    tags = re.findall(r"</?[a-zA-Z]+>", completion)

    symbols = []
    is_valid_pairing = True
    i = 0
    while i < len(tags):
        open_tag = tags[i]
        if open_tag not in TAG_MAP:
            is_valid_pairing = False
            break
            
        expected_close, symbol = TAG_MAP[open_tag]
        if i + 1 >= len(tags) or tags[i+1] != expected_close:
            is_valid_pairing = False
            break

        symbols.append(symbol)
        i += 2

    if not is_valid_pairing:
        return 0.0
        
    if len(symbols) < len(END_PATTERN):
        return 0.0
        
    if symbols[-len(END_PATTERN):] != END_PATTERN:
        return 0.0
        
    remaining = symbols[:-len(END_PATTERN)]
    
    if len(remaining) == 0:
        return 1.0
        
    if len(remaining) % len(CYCLE_PATTERN) != 0:
        return 0.0
        
    is_structure_valid = True
    for k in range(0, len(remaining), len(CYCLE_PATTERN)):
        chunk = remaining[k : k + len(CYCLE_PATTERN)]
        if chunk != CYCLE_PATTERN:
            is_structure_valid = False
            break
    
    if is_structure_valid:
        return 1.0
    return 0.0

class RAGTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self.retrieval_model = kwargs.pop('retrieval_model')
        self.rm_tokenizer = kwargs.pop('rm_tokenizer')
        self.reward_model = kwargs.pop('reward_model')
        self.prm_beta = kwargs.pop('prm_beta', 0.5)
        super().__init__(*args, **kwargs)
        
        try:
            self.reward_model.to(self.accelerator.device)
            self.reward_model.eval()
        except:
            pass

    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        核心逻辑: Rollout -> Outcome Reward -> GRPO Norm -> PRM Scoring -> Advantage Mixing
        """
        device = self.accelerator.device
        all_groups_data = []

        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )
        
        stop_criteria = StoppingCriteriaList([
            StopOnSpecificTokens(self.processing_class, ["<retrieval>", "</subanswer>", "<|im_end|>", "<|endoftext|>"])
        ])

        # --- 1. Rollout Phase ---
        for data_item in inputs:
            group_rollouts = []
            
            for _ in range(self.args.num_generations):
                env = RAGEnv(data_item, self.retrieval_model)
                trajectory_steps = []
                
                for step_i in range(MAX_INTERACTION_STEPS):
                    current_prompt = env.state
                    
                    enc = self.processing_class(
                        current_prompt, 
                        return_tensors="pt", 
                        padding=False, 
                        truncation=True, 
                        max_length=self.args.max_prompt_length
                    )
                    input_ids = enc.input_ids.to(device)
                    attn_mask = enc.attention_mask.to(device)
                    
                    model_unwrapped = self.accelerator.unwrap_model(self.model)
                    with torch.no_grad():
                        out = model_unwrapped.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            generation_config=gen_config,
                            stopping_criteria=stop_criteria
                        )
                    
                    prompt_len = input_ids.shape[1]
                    new_ids = out[0, prompt_len:]
                    new_text = self.processing_class.decode(new_ids, skip_special_tokens=False)
                    
                    prm_text = f"Question: {data_item['question']}\nHistory:\n{env.generated_text_history}\n{new_generated_text}"
                    prm_inputs = self.rm_tokenizer(
                        prm_text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=self.args.max_prompt_length,
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        prm_out = self.reward_model(**prm_inputs)
                        prm_score = prm_out.logits.view(-1).cpu().item()

                    _, done = env.step(new_text)
                    
                    trajectory_steps.append({
                        "prompt": current_prompt,
                        "completion": new_text,
                        "prm_score": prm_score,
                        "done": done
                    })
                    
                    if done:
                        break
                
                full_completion = env.generated_text_history
                ground_truth = data_item["answer"]
                
                acc = 0.0
                match = re.search(r"<answer>(.*?)</answer>", full_completion, re.DOTALL)
                if match:
                    pred = match.group(1).strip().lower()
                    if ground_truth.strip().lower() in pred:
                        acc = 1.0
                
                fmt_scores = format_reward(full_completion)
                outcome_reward = acc * 2.0 + fmt
             
                group_rollouts.append({
                    "steps": trajectory_steps,
                    "outcome_reward": outcome_reward
                })
            
            all_groups_data.append(group_rollouts)

        # --- 2. Advantage Calculation & Mixing ---
        flat_prompts = []
        flat_completions = []
        flat_advantages = []
        
        # 统计数据用于 Log
        log_outcome_rewards = []
        log_prm_scores = []
        
        for group in all_groups_data:
            outcomes = torch.tensor([g["outcome_reward"] for g in group], dtype=torch.float32, device=device)
            log_outcome_rewards.extend(outcomes.tolist())
            
            if len(outcomes) > 1:
                g_mean = outcomes.mean()
                g_std = outcomes.std(unbiased=False) + 1e-8
                outcome_advs = (outcomes - g_mean) / g_std
            else:
                outcome_advs = outcomes - outcomes.mean()
            
            all_prm_scores_in_group = []
            for rollout in group:
                for step in rollout["steps"]:
                    all_prm_scores_in_group.append(step["prm_score"])
            
            prm_tensor = torch.tensor(all_prm_scores_in_group, dtype=torch.float32, device=device)
            if len(prm_tensor) > 1:
                prm_mean = prm_tensor.mean()
                prm_std = prm_tensor.std(unbiased=False) + 1e-8
            else:
                prm_mean = prm_tensor.mean()
                prm_std = 1.0

            prm_counter = 0 
            for idx, rollout in enumerate(group):
                base_adv = outcome_advs[idx].item()
                for step in rollout["steps"]:
                    raw_prm_score = step["prm_score"]
                    log_prm_scores.append(raw_prm_score)
                    
                    normalized_prm = (raw_prm_score - prm_mean) / prm_std
                    final_adv = base_adv + self.prm_beta * normalized_prm
                    
                    flat_prompts.append(step["prompt"])
                    flat_completions.append(step["completion"])
                    flat_advantages.append(final_adv)

        # --- 3. Batch Preparation ---
        if not flat_prompts:
            return {}

        adv_tensor = torch.tensor(flat_advantages, dtype=torch.float32, device=device)
        prompt_inputs = self.processing_class(
            flat_prompts, return_tensors="pt", padding=True, truncation=True, padding_side="left"
        ).to(device)
        
        completion_inputs = self.processing_class(
            flat_completions, return_tensors="pt", padding=True, truncation=True, padding_side="right"
        ).to(device)

        mode = "train" if self.model.training else "eval"
        if log_outcome_rewards:
            self._metrics[mode].setdefault("rollout/outcome_reward_mean", []).append(np.mean(log_outcome_rewards))
        if log_prm_scores:
            self._metrics[mode].setdefault("rollout/prm_score_mean", []).append(np.mean(log_prm_scores))
        self._metrics[mode].setdefault("rollout/advantage_mean", []).append(adv_tensor.mean().item())

        return {
            "prompt_ids": prompt_inputs["input_ids"],
            "prompt_mask": prompt_inputs["attention_mask"],
            "completion_ids": completion_inputs["input_ids"],
            "completion_mask": completion_inputs["attention_mask"],
            "advantages": adv_tensor
        }

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        advantages = inputs["advantages"]
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # 只计算 completion 部分的 logprobs
        logits_to_keep = completion_ids.size(1)
        
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True
        )
        
        completion_mask = completion_mask.float()
        
        # 计算每个 Step (Sequence) 的平均 Logprob
        # sum(log_prob * mask) / sum(mask)
        per_seq_logp = (per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
        
        # Loss = - Advantage * Logprob
        loss = - (advantages * per_seq_logp).mean()
        
        # Metrics
        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("loss", []).append(loss.item())
        self._metrics[mode].setdefault("entropy", []).append(entropies.mean().item())
        
        return loss