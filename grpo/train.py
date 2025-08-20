import torch
import re
import collections
import json
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

RAW_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/raw/train_rl.json"
POLICY_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
REWARD_MODEL_PATH = "/home/v-zhaowan/zhaowang/rag/save/rm/v1"

PPO_OUTPUT_DIR = "/home/v-zhaowan/zhaowang/rag/save/ppo/v1"
BATCH_SIZE = 4
MINI_BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1.41e-5
MAX_PPO_EPOCHS = 4

MAX_INTERACTION_STEPS = 5
MAX_GEN_LENGTH_PER_STEP = 128

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Loading SentenceTransformer model for Retriever...")
E5_MODEL_NAME = 'intfloat/e5-large-v2'
similarity_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)

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
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=False).to(DEVICE)
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query: return []
        query_with_prefix = f'query: {query}'
        query_embedding = self.model.encode(query_with_prefix, convert_to_tensor=True).to(DEVICE)
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

if __name__ == "__main__":
    print("Loading and preparing data...")
    raw_data_map = {}
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            raw_data_map[item['id']] = item
            
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data_indices = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(POLICY_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, peft_config=lora_config)
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH, torch_dtype=torch.bfloat16, num_labels=1).to(DEVICE)
    reward_model.eval()
    def get_reward_score(texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(DEVICE)
            return reward_model(**inputs).logits.squeeze(-1)

    config = PPOConfig(model_name=POLICY_MODEL_NAME, log_with="tensorboard", project_kwargs={"logging_dir": f"{PPO_OUTPUT_DIR}/logs"}, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, mini_batch_size=MINI_BATCH_SIZE, ppo_epochs=MAX_PPO_EPOCHS, remove_unused_columns=False)
    ppo_trainer = PPOTrainer(config=config, model=policy_model, ref_model=None, tokenizer=tokenizer)
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        for i in tqdm(range(0, len(data_indices), BATCH_SIZE)):
            batch_indices = data_indices[i:i+BATCH_SIZE]
            
            envs = [RAGEnv(raw_data_map[idx['id']], similarity_model) for idx in batch_indices if idx['id'] in raw_data_map]
            if not envs: continue
            
            query_tensors = [tokenizer.encode(env.state, return_tensors="pt").to(DEVICE)[0] for env in envs]
            
            all_step_responses = [[] for _ in range(len(envs))]
            all_step_rewards = [[] for _ in range(len(envs))]

            for step in range(MAX_INTERACTION_STEPS):
                if all(env.is_done for env in envs): break

                active_indices = [k for k, env in enumerate(envs) if not env.is_done]
                active_inputs = [query_tensors[k] for k in active_indices]

                stopping_criteria = StoppingCriteriaList([StopOnKeywords(tokenizer, ["<retrieval>", "<|im_end|>"])])
                response_outputs = ppo_trainer.generate(
                    active_inputs, max_new_tokens=MAX_GEN_LENGTH_PER_STEP,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria, return_prompt=False,
                )
                
                for j, response_tensor in enumerate(response_outputs):
                    env_idx = active_indices[j]
                    env = envs[env_idx]
                    response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
                    
                    reward = get_reward_score([env.state+response_text])[0]
                    new_state, is_done = env.step(response_text)
                    
                    query_tensors[env_idx] = tokenizer.encode(new_state, return_tensors="pt").to(DEVICE)[0]
                    all_step_responses[env_idx].append(response_tensor)
                    all_step_rewards[env_idx].append(reward)

            final_query_tensors, final_response_tensors, final_rewards = [], [], []
            all_rewards_for_logging = []

            for k in range(len(envs)):
                env = envs[k]
                if not all_step_responses[k]: continue
                
                initial_query = tokenizer.encode(f"<|im_start|>user\n{env.question}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(DEVICE)[0]
                full_response_tensor = torch.cat(all_step_responses[k])
                
                final_answer_match = re.search(r"<answer>(.*?)</answer>", env.state, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    f1 = calculate_f1_score(final_answer, env.final_answer_list)
                    final_reward_val = 10.0 * f1
                else:
                    final_reward_val = all_step_rewards[k][-1].item()
                
                step_rewards = torch.stack(all_step_rewards[k])
                step_rewards[-1] = final_reward_val

                final_query_tensors.append(initial_query)
                final_response_tensors.append(full_response_tensor)
                final_rewards.append(step_rewards)
                all_rewards_for_logging.extend(step_rewards.tolist())
            
            if not final_query_tensors: continue
            
            stats = ppo_trainer.step(final_query_tensors, final_response_tensors, final_rewards)
            
            mean_reward = sum(all_rewards_for_logging) / len(all_rewards_for_logging) if all_rewards_for_logging else 0
            ppo_trainer.log_stats(stats, {}, {"mean_reward": mean_reward})

    print("Training finished. Saving the final PPO model (LoRA weights)...")
    ppo_trainer.save_model(PPO_OUTPUT_DIR)