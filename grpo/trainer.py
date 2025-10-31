import re
import collections
from typing import List, Dict, Any
import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from trl import GRPOTrainer

MAX_INTERACTION_STEPS = 7
NUM_ROLLOUTS = 2


# ======== 自定义停止条件 ========
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


# ======== E5 向量检索器 ========
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


# ======== 环境定义 ========
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

        # 如果模型调用了 <retrieval>，则执行检索
        if self.state.strip().endswith("<retrieval>"):
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", response_text, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = self.retriever.retrieve(subquery, top_k=1)
                retrieved_docs_text = "\n".join(retrieved_docs)
                self.state += f"\n{retrieved_docs_text}\n</retrieval>\n"

        # 结束条件
        if (self.state.strip().endswith("<|im_end|>") and "<answer>" in response_text) or self.steps >= MAX_INTERACTION_STEPS:
            self.is_done = True

        return self.state, self.is_done


# ======== F1 奖励函数 ========
def calculate_f1_score(prediction: str, ground_truth_list: list) -> float:
    prediction_tokens = prediction.lower().split()
    best_f1 = 0.0
    for ground_truth in ground_truth_list:
        ground_truth_tokens = ground_truth.lower().split()
        if not prediction_tokens or not ground_truth_tokens:
            continue
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


# ======== Step-GRPO 训练器 ========
class RAGTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self.retrieval_model = kwargs.pop('retrieval_model')
        self.rm_tokenizer = kwargs.pop('rm_tokenizer')
        self.reward_model = kwargs.pop('reward_model')
        super().__init__(*args, **kwargs)

    # ---------- rollout ----------
    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        device = self.accelerator.device
        rollout_data = []

        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )
        stop_criteria = StoppingCriteriaList([StopOnKeywords(self.processing_class, ["<retrieval>", "<|im_end|>"])])

        for data_item in inputs:
            for _ in range(NUM_ROLLOUTS):
                env = RAGEnv(data_item, self.retrieval_model)
                step_traj = []

                for step_idx in range(MAX_INTERACTION_STEPS):
                    query_tensor = self.processing_class.encode(env.state, return_tensors="pt").to(device)
                    with torch.no_grad():
                        response_tensor = self.accelerator.unwrap_model(self.model).generate(
                            query_tensor,
                            generation_config=gen_config,
                            stopping_criteria=stop_criteria,
                        )
                    response_slice = response_tensor[0, query_tensor.shape[1]:]
                    response_text = self.processing_class.decode(response_slice, skip_special_tokens=False)

                    new_state, done = env.step(response_text)
                    # === 奖励 ===
                    if done:
                        final_match = re.search(r"<answer>(.*?)</answer>", new_state, re.DOTALL)
                        if final_match:
                            f1_score = 5.0 * calculate_f1_score(final_match.group(1).strip(), env.final_answer_list)
                        else:
                            f1_score = -2.0
                        step_reward = f1_score
                    else:
                        rm_inputs = self.rm_tokenizer(
                            new_state, return_tensors="pt", padding=True, truncation=True, max_length=4096
                        ).to(device)
                        with torch.no_grad():
                            rm_outputs = self.reward_model(**rm_inputs)
                            step_reward = float(rm_outputs.logits.squeeze().cpu().item())

                    step_traj.append({
                        "state": env.state,
                        "action": response_text,
                        "reward": step_reward,
                    })

                    if done:
                        break

                rollout_data.append(step_traj)

        # flatten steps
        prompts, completions, rewards = [], [], []
        for traj in rollout_data:
            for step in traj:
                prompts.append(step["state"])
                completions.append(step["action"])
                rewards.append(step["reward"])

        # Tokenize
        prompt_inputs = self.processing_class(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        completion_inputs = self.processing_class(
            text=completions, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        completion_ids = completion_inputs["input_ids"]
        completion_mask = completion_inputs["attention_mask"]

        # normalize rewards
        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        mean, std = rewards_tensor.mean(), rewards_tensor.std(unbiased=False) + 1e-8
        advantages = (rewards_tensor - mean) / std

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "step_rewards": rewards_tensor,
        }

    # ---------- step loss ----------
    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, completion_ids.size(1), compute_entropy=True
        )

        advantages = inputs["advantages"].to(per_token_logps.device)
        completion_mask = completion_mask.to(per_token_logps.device)

        logps = (per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
        per_step_loss = -advantages * logps
        loss = per_step_loss.mean()

        # === metrics ===
        mode = "train" if model.training else "eval"
        self._metrics[mode]["loss"].append(self.accelerator.gather(loss).mean().item())
        self._metrics[mode]["reward/mean"].append(self.accelerator.gather(inputs["step_rewards"].mean()).item())
        self._metrics[mode]["reward/std"].append(self.accelerator.gather(inputs["step_rewards"].std()).item())
        self._metrics[mode]["advantage/std"].append(self.accelerator.gather(advantages.std()).item())
        self._metrics[mode]["entropy"].append(self.accelerator.gather(entropies.mean()).item())

        return loss
