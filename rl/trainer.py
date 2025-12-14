import re
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Dict, Any
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from trl import GRPOTrainer
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

MAX_INTERACTION_STEPS = 9
TAG_MAP = {
"<step>":      ("</step>",      "S"),
"<subquery>":  ("</subquery>",  "Q"),
"<retrieval>": ("</retrieval>", "R"),
"<subanswer>": ("</subanswer>", "A"),
"<answer>":    ("</answer>",    "F")
}
CYCLE_PATTERN = ["S", "Q", "R", "S", "A"]
END_PATTERN = ["S", "F"]

class E5VectorRetriever:
    def __init__(self, paragraphs: Dict[str, List], model: SentenceTransformer, device="cuda"):
        self.model = model
        self.device = device
        self.titles = [item["title"] for item in paragraphs]
        self.passages = [item["paragraph_text"] for item in paragraphs]
        self.corpus = [f"Title: {t}\n{p}" for t, p in zip(self.titles, self.passages)]
        self.corpus_embeddings = None
        self._encoded = False

    def _encode_corpus(self):
        if not self.corpus: return
        with torch.no_grad():
            emb = self.model.encode(
                self.corpus, 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
        self.corpus_embeddings = emb
        self._encoded = True

    def retrieve(self, query: str, top_k: int = 1) -> List[str]:
        if not self.corpus or not query: return []
        if not self._encoded: self._encode_corpus()
        if self.corpus_embeddings is None: return []
        
        with torch.no_grad():
            q_emb = self.model.encode(
                f"query: {query}", 
                convert_to_tensor=True, 
                device=self.device, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
            scores = torch.matmul(self.corpus_embeddings, q_emb)
            top_indices = torch.topk(scores, min(top_k, len(scores))).indices.cpu().tolist()
        
        return [f"Title: {self.titles[idx]}\n{self.passages[idx]}" for idx in top_indices]

class RAGEnv:
    def __init__(self, data_item: dict, retriever: E5VectorRetriever):
        self.data_item = data_item
        self.current_prompt = data_item["init_prompt"]
        self.history_text = "<step>\n"
        self.retriever = retriever
        self.is_done = False
        self.trajectory = []

    def update(self, new_text: str, prm_score: float, format_score: float):
        retrieval_text = ""
        if "<answer>" in new_text:
            self.is_done = True
        else:
            retrieval_text += "<step>\n"
    
        self.trajectory.append({
            "prompt": self.current_prompt,
            "completion": new_text,
            "prm_score": prm_score,
            "format_score": format_score
        })
        self.history_text += (new_text + retrieval_text)
        self.current_prompt += (new_text + retrieval_text)   

def format_reward_step(completion):
    tags = re.findall(r"</?[a-zA-Z]+>", completion)

    if tags == ["</step>", "<subquery>", "</subquery>", "<retrieval>", "</retrieval>"]:
        return 1.0

    if tags == ["</step>", "<subanswer>", "</subanswer>"]:
        return 1.0

    if tags == ["</step>", "<answer>", "</answer>"]:
        return 1.0

    return -2.0

def format_reward(completion):
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
        return -1.0
        
    if len(symbols) < len(END_PATTERN):
        return -1.0
        
    if symbols[-len(END_PATTERN):] != END_PATTERN:
        return -1.0
        
    remaining = symbols[:-len(END_PATTERN)]

    if len(remaining) == 0:
        return 1.0
        
    if len(remaining) % len(CYCLE_PATTERN) != 0:
        return -1.0
        
    is_structure_valid = True
    for k in range(0, len(remaining), len(CYCLE_PATTERN)):
        chunk = remaining[k : k + len(CYCLE_PATTERN)]
        if chunk != CYCLE_PATTERN:
            is_structure_valid = False
            break

    if is_structure_valid:
        return 1.0
    return -1.0

class RAGTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self.retrieval_model = kwargs.pop('retrieval_model')
        self.rm_tokenizer = kwargs.pop('rm_tokenizer')
        self.reward_model = kwargs.pop('reward_model')
        self.prm_beta = kwargs.pop('prm_beta', 0.5)
        self.clip_range = kwargs.pop('clip_range', 0.2)
        self.kl_beta = kwargs.pop('kl_beta', 0.04)
        super().__init__(*args, **kwargs)

        if hasattr(self.reward_model, "to"):
                self.reward_model.to(self.accelerator.device)
                self.reward_model.eval()

    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        核心逻辑: Rollout -> Outcome Reward -> GRPO Norm -> PRM Scoring -> Advantage Mixing
        """
        torch.cuda.empty_cache()

        model_unwrapped = self.accelerator.unwrap_model(self.model)
        model_unwrapped.eval()

        device = self.accelerator.device
        num_gens = self.args.num_generations

        active_envs = []
        for item in inputs:
            item_retriever = E5VectorRetriever(item.get("paragraphs", []), self.retrieval_model, device=device)
            for _ in range(num_gens):
                active_envs.append(RAGEnv(item, item_retriever))

        active_indices = list(range(len(active_envs)))
        
        stop_words = ["</subquery>", "</subanswer>", "<|im_end|>", "</answer>"]
        stop_ids = [self.processing_class.convert_tokens_to_ids(tok) for tok in stop_words]

        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=stop_ids,
        )

        # --- 1. Rollout Phase ---
        for step in range(MAX_INTERACTION_STEPS):
            if not active_indices: break
            
            current_prompts = [active_envs[i].current_prompt for i in active_indices]
            enc = self.processing_class(
                current_prompts, return_tensors="pt", padding=True, truncation=True, padding_side="left"
            ).to(device)
            
            with torch.no_grad():
                outputs = model_unwrapped.generate(
                    **enc,
                    generation_config=gen_config
                )
            
            input_len = enc.input_ids.shape[1]
            new_ids = outputs[:, input_len:]
            raw_texts = self.processing_class.batch_decode(new_ids, skip_special_tokens=False)

            new_texts = []
            for i, text in zip(active_indices, raw_texts):
                retriever = active_envs[i].retriever
                text = text.replace(self.processing_class.eos_token, "")
                text = text.replace(self.processing_class.pad_token, "")
                
                sq_match = re.search(r"<subquery>(.*?)</subquery>", text, re.DOTALL)
                if sq_match:
                    query = sq_match.group(1).strip()
                    docs = retriever.retrieve(query, top_k=1)
                    if docs:
                        retrieval_text = f"\n<retrieval>\n{docs[0]}\n</retrieval>\n"
                else:
                    retrieval_text = "\n"
                
                new_text = text + retrieval_text
                new_texts.append(new_text)
            
            del enc, outputs, new_ids
            torch.cuda.empty_cache()

            prm_texts = []
            for i, text in zip(active_indices, new_texts):
                env = active_envs[i]
                prm_texts.append(f"Question: {env.data_item['question']}\nHistory:\n{env.history_text}\n{text}")
            
            prm_enc = self.rm_tokenizer(
                prm_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(device)
            
            with torch.no_grad():
                prm_out = self.reward_model(**prm_enc)
                if prm_out.logits.shape[-1] == 1:
                    prm_scores = prm_out.logits.view(-1).tolist()
                else:
                    prm_scores = prm_out.logits[:, 1].tolist()
            
            del prm_enc, prm_out
            torch.cuda.empty_cache()

            next_indices = []
            for i, text, score in zip(active_indices, new_texts, prm_scores):
                env = active_envs[i]
                fmt_score = format_reward_step(text)
                env.update(text, score, fmt_score)
                if not env.is_done:
                    next_indices.append(i)
            active_indices = next_indices

        if self.accelerator.is_main_process:
            print("\n" + "="*50)
            print(" [DEBUG ROLLOUT SNAPSHOT] ")
            print("="*50)

            debug_env = active_envs[0]            
            debug_text = debug_env.history_text
            debug_traj = debug_env.trajectory
            debug_fmt = format_reward(debug_text)
 
            print(f">>> Total Interaction Steps: {len(debug_traj)}")
            print("-" * 20)
            for idx, step in enumerate(debug_traj):
                print(f" Step {idx+1}:")
                print(f">>> Prompt (Partial): {step['prompt'][-100:]}")
                print(f">>> Completion: {step['completion']}")
                print(f">>> PRM Score: {step['prm_score']:.4f}, Format Score: {step['format_score']:.4f}")
                print("-" * 20)

            print(f">>> Calculated Format Reward: {debug_fmt}")
            print("="*50 + "\n")
        
        # --- 2. Advantage Calculation & Mixing ---
        flat_prompts, flat_completions, flat_advantages = [], [], []
        stats_outcome_rewards, stats_prm_scores, stats_step_format_scores, stats_accuracies, stats_format_rewards = [], [], [], [], []

        for i in range(len(inputs)):
            group_envs = active_envs[i*num_gens : (i+1)*num_gens]
            gt_ans = inputs[i]["answer"].strip().lower()
            
            outcomes = []
            for env in group_envs:
                full_text = env.history_text
                acc = 0.0
                match = re.search(r"<answer>\n(.*?)</answer>", full_text, re.DOTALL)
                if match:
                    pred = match.group(1).strip().lower()
                    if gt_ans == pred: acc = 1.0

                fmt = format_reward(full_text)
                total_reward = fmt + acc * 2.0
    
                outcomes.append(total_reward)
                
                stats_outcome_rewards.append(total_reward)
                stats_accuracies.append(acc)
                stats_format_rewards.append(fmt)

            # GRPO Normalization (Group Level)
            rewards_t = torch.tensor(outcomes, device=device, dtype=torch.float32)
            if len(rewards_t) > 1:
                adv_outcome = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            else:
                adv_outcome = rewards_t - rewards_t.mean()

            # PRM Stats
            all_prm = [s["prm_score"] for env in group_envs for s in env.trajectory]
            all_step_fmt = [s["format_score"] for env in group_envs for s in env.trajectory]
            
            stats_prm_scores.extend(all_prm)
            stats_step_format_scores.extend(all_step_fmt)
                        
            prm_t = torch.tensor(all_prm, device=device)
            prm_mean, prm_std = (prm_t.mean().item(), prm_t.std().item() + 1e-8) if len(prm_t) > 0 else (0, 1)
            
            # Mixing Advantage
            for idx, env in enumerate(group_envs):
                base_adv = adv_outcome[idx].item()
                for step in env.trajectory:
                    # PRM Normalized
                    norm_prm = (step["prm_score"] - prm_mean) / prm_std
                    step_adv = (norm_prm) if step["format_score"] > 0 else step["format_score"]
                    final_adv = base_adv + self.prm_beta * step_adv

                    p_ids = self.processing_class.encode(step["prompt"], add_special_tokens=False, return_tensors="pt")[0]
                    c_ids = self.processing_class.encode(step["completion"], add_special_tokens=False, return_tensors="pt")[0]
                    
                    flat_prompts.append(p_ids)
                    flat_completions.append(c_ids)
                    flat_advantages.append(final_adv)
        
        if not flat_prompts: return {}
        
        prompt_ids_reversed = [t.flip(0) for t in flat_prompts]
        prompt_ids_padded = rnn_utils.pad_sequence(
            prompt_ids_reversed, batch_first=True, padding_value=self.processing_class.pad_token_id
        ).flip(1).to(device)
        
        completion_ids_padded = rnn_utils.pad_sequence(
            flat_completions, batch_first=True, padding_value=self.processing_class.pad_token_id
        ).to(device)
        
        advantages = torch.tensor(flat_advantages, dtype=torch.float32, device=device)

        prompt_mask = (prompt_ids_padded != self.processing_class.pad_token_id).long()
        completion_mask = (completion_ids_padded != self.processing_class.pad_token_id).long()

        input_ids = torch.cat([prompt_ids_padded, completion_ids_padded], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        logits_to_keep = completion_ids_padded.shape[1] 

        def compute_logprobs_in_chunks(model_to_use, full_input_ids, full_mask, chunk_size=8):
            total_samples = full_input_ids.shape[0]
            all_logprobs = []
            
            for i in range(0, total_samples, chunk_size):
                end_i = min(i + chunk_size, total_samples)
                sub_ids = full_input_ids[i:end_i]
                sub_mask = full_mask[i:end_i]
                
                with torch.no_grad():
                    logprobs, _ = self._get_per_token_logps_and_entropies(
                        model_to_use, sub_ids, sub_mask, logits_to_keep
                    )
                    all_logprobs.append(logprobs)
                
                del sub_ids, sub_mask, logprobs
                torch.cuda.empty_cache()
                
            return torch.cat(all_logprobs, dim=0)
        
        with torch.no_grad():
            old_log_probs = compute_logprobs_in_chunks(model_unwrapped, input_ids, attention_mask)
        
        ref_log_probs = None 
        if hasattr(model_unwrapped, "disable_adapter"):
            with model_unwrapped.disable_adapter():
                with torch.no_grad():
                    ref_log_probs = compute_logprobs_in_chunks(model_unwrapped, input_ids, attention_mask)

        model_unwrapped.train()

        if flat_prompts:
            mode = "train" if self.model.training else "eval"
            if mode not in self._metrics: self._metrics[mode] = {}
            
            def log_metric(key, val_list):
                if val_list:
                    mean_val = np.mean(val_list)
                    self._metrics[mode].setdefault(key, []).append(mean_val)
            
            log_metric("rollout/outcome_reward_mean", stats_outcome_rewards)
            log_metric("rollout/accuracy_mean", stats_accuracies)
            log_metric("rollout/format_reward_mean", stats_format_rewards)
            log_metric("rollout/prm_score_mean", stats_prm_scores)
            log_metric("rollout/step_format_score_mean", stats_step_format_scores)
            log_metric("rollout/advantage_mean", flat_advantages)
            
            avg_steps = np.mean([len(e.trajectory) for e in active_envs])
            self._metrics[mode].setdefault("rollout/avg_steps", []).append(avg_steps)
        
        return {
            "prompt_ids": prompt_ids_padded,       
            "prompt_mask": prompt_mask,            
            "completion_ids": completion_ids_padded,
            "completion_mask": completion_mask,   
            "advantages": advantages,
            "old_per_token_logps": old_log_probs, 
            "ref_per_token_logps": ref_log_probs
        }

    def _compute_loss(self, model, inputs):
        torch.cuda.empty_cache() 
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        advantages = inputs["advantages"]

        old_log_probs_all = inputs["old_per_token_logps"]
        ref_log_probs_all = inputs["ref_per_token_logps"]
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        num_samples = input_ids.size(0)
        total_loss_tensor = 0.0 
        chunk_size = 4
        stats = {
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "clip_ratio": 0.0,
            "valid_token_count": 0
        }
        
        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            
            sub_input_ids = input_ids[i:end_idx]
            sub_attention_mask = attention_mask[i:end_idx]
            sub_completion_mask = completion_mask[i:end_idx]
            sub_advantages = advantages[i:end_idx]
            sub_old_log_probs = old_log_probs_all[i:end_idx]
            sub_ref_log_probs = ref_log_probs_all[i:end_idx] if ref_log_probs_all is not None else None
            
            chunk_weight = (end_idx - i) / num_samples
            logits_to_keep = sub_completion_mask.shape[1]

            per_token_logps, _ = self._get_per_token_logps_and_entropies(
                model, sub_input_ids, sub_attention_mask, logits_to_keep
            )

            ratio = torch.exp(per_token_logps - sub_old_log_probs)
            
            token_advantages = sub_advantages.unsqueeze(1).expand_as(ratio)
            surr1 = ratio * token_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * token_advantages
            policy_loss_per_token = -torch.min(surr1, surr2)
            
            kl_loss_per_token = 0.0
            if sub_ref_log_probs is not None:
                kl_val = per_token_logps - sub_ref_log_probs
                kl_loss_per_token = self.kl_beta * kl_val

            total_loss_per_token = policy_loss_per_token + kl_loss_per_token
            masked_loss = total_loss_per_token * sub_completion_mask
            
            seq_lengths = sub_completion_mask.sum(dim=1).clamp(min=1.0)
            per_sample_loss = masked_loss.sum(dim=1) / seq_lengths
            chunk_mean_loss = per_sample_loss.mean()
            
            total_loss_tensor = total_loss_tensor + (chunk_mean_loss * chunk_weight)
            
            with torch.no_grad():
                valid_mask_sum = sub_completion_mask.sum().item()
                if valid_mask_sum > 0:
                    stats["valid_token_count"] += valid_mask_sum
                    stats["policy_loss"] += (policy_loss_per_token * sub_completion_mask).sum().item()
                    stats["kl_loss"] += (kl_loss_per_token * sub_completion_mask).sum().item()
                    
                    clipped = (ratio < 1.0 - self.clip_range) | (ratio > 1.0 + self.clip_range)
                    stats["clip_ratio"] += (clipped.float() * sub_completion_mask).sum().item()
            
            del sub_input_ids, sub_attention_mask, per_token_logps, ratio
            torch.cuda.empty_cache()

        total_tokens = stats["valid_token_count"] + 1e-8
        self._log_metrics("loss/policy_loss", stats["policy_loss"] / total_tokens)
        self._log_metrics("logps/kl", stats["kl_loss"] / total_tokens)
        self._log_metrics("logps/clip_ratio", stats["clip_ratio"] / total_tokens)
        return total_loss_tensor

    def _log_metrics(self, key, value):
        mode = "train" if self.model.training else "eval"
        if mode not in self._metrics: self._metrics[mode] = {}
        self._metrics[mode].setdefault(key, []).append(value)

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, compute_entropy=False):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -logits_to_keep:, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
        del logits, outputs

        completion_input_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = torch.gather(log_probs, -1, completion_input_ids.unsqueeze(-1)).squeeze(-1)
        
        return per_token_logps, None