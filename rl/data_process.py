import os
import json
import random
import re
import torch
import numpy as np
import collections
from tqdm import tqdm
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

# ================= config =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

MULSIQUE_PATH = "/home/aiscuser/ds/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
HOTPOTQA_PATH = "/home/aiscuser/ds/zhaowang/rag/data/HotpotQA/train.jsonl"
OUTPUT_PATH = "/home/aiscuser/ds/zhaowang/rag/data/train_rl_tmp.jsonl"

MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft"
E5_MODEL_NAME = 'intfloat/e5-large-v2'

MAX_TOKENS = 1024
MAX_HOP = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STEP1_SEED_NUM_PER_DATASET = 500
STEP2_BATCH_SIZE = 500        
TARGET_TOTAL_SIZE = 2000        

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

class E5VectorRetriever:
    def __init__(self, paragraphs: List[Dict], model: SentenceTransformer, device="cuda"):
        self.model = model
        self.device = device
        self.titles = [item.get("title", "") for item in paragraphs]
        self.passages = [item.get("paragraph_text", "") for item in paragraphs]
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

class RequestState:
    def __init__(self, sample: Dict, retrieval_model: SentenceTransformer):
        self.id = sample.get("id", str(random.randint(0, 100000)))
        self.sample = sample
        self.question = sample["question"]
        self.user_content = INSTRUCTION_TEMPLATE.format(question=self.question)
        self.prompt = f"<|im_start|>user\n{self.user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
        self.full_trace = "<step>\n"
        self.finished = False
        self.final_answer = ""
        self.retriever = E5VectorRetriever(sample.get("paragraphs", []), retrieval_model, device=DEVICE)

def format_output_item(sample):
    user_content = INSTRUCTION_TEMPLATE.format(question=sample["question"])
    init_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
    return {
        "question": sample["question"],
        "answer": sample["answer"],
        "paragraphs": sample["paragraphs"],
        "init_prompt": init_prompt
    }

def load_all_data():
    musique_pool = []
    print(f"Loading MulSiQue from {MULSIQUE_PATH}...")
    with open(MULSIQUE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            if not item["id"].startswith("2hop"):
                musique_pool.append(item)
    print(f"MulSiQue candidates: {len(musique_pool)}")

    hotpot_pool = []
    print(f"Loading HotpotQA from {HOTPOTQA_PATH}...")
    with open(HOTPOTQA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            if item.get("level") == "hard":
                context = item["context"]
                paragraphs = []
                for title, sentences in zip(context["title"], context["sentences"]):
                    text = '\n'.join(sentences)
                    paragraphs.append({
                        "title": title,
                        "paragraph_text": text
                    })
                item["paragraphs"] = paragraphs
                hotpot_pool.append(item)
    print(f"HotpotQA candidates: {len(hotpot_pool)}")
    
    return musique_pool, hotpot_pool

def run_inference_batch(samples, llm, retrieval_model):
    if not samples: return []
    
    states = [RequestState(s, retrieval_model) for s in samples]
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        stop=["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False
    )

    print(f"  Running inference on {len(samples)} samples...")
    
    for hop in range(MAX_HOP):
        active_states = [s for s in states if not s.finished]
        if not active_states:
            break
        
        prompts = [s.prompt for s in active_states]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        for i, output in tqdm(list(enumerate(outputs)), total=len(outputs)):
            state = active_states[i]
            generated_text = output.outputs[0].text
            state.prompt += generated_text
            
            if "</answer>" in generated_text:
                state.finished = True
                match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
                if match:
                    state.final_answer = match.group(1).strip()
               
            elif "</subquery>" in generated_text:
                sq_match = re.search(r"<subquery>(.*?)</subquery>", generated_text, re.DOTALL)
                if sq_match:
                    q_str = sq_match.group(1).strip()
                    docs = state.retriever.retrieve(q_str, top_k=1)
                    
                    retrieval_block = f"\n<retrieval>\n{docs}\n</retrieval>\n<step>\n"
                    state.prompt += retrieval_block
                else:
                    state.prompt += "\n<retrieval>Error in query parsing</retrieval>\n<step>\n"

            else:
                state.prompt += "\n<step>\n"

    incorrect_samples = []
    for state in states:
        answer = state.sample["answer"]
        is_correct = (state.final_answer.lower() == answer.lower())
        if not is_correct:
            incorrect_samples.append(format_output_item(state.sample))
            
    return incorrect_samples

def main():
    musique_pool, hotpot_pool = load_all_data()
    
    random.shuffle(musique_pool)
    random.shuffle(hotpot_pool)

    final_train_data = []
    print("\n=== Step 1: Selecting 500 random samples from each dataset (Base) ===")
    base_mul = musique_pool[:STEP1_SEED_NUM_PER_DATASET]
    musique_pool = musique_pool[STEP1_SEED_NUM_PER_DATASET:]
    
    base_hot = hotpot_pool[:STEP1_SEED_NUM_PER_DATASET]
    hotpot_pool = hotpot_pool[STEP1_SEED_NUM_PER_DATASET:]
    
    for item in base_mul + base_hot:
        final_train_data.append(format_output_item(item))
        
    print(f"Added {len(base_mul)} MulSiQue and {len(base_hot)} HotpotQA samples.")
    print(f"Current Training Data Size: {len(final_train_data)}")

    print("\n=== Step 2: Mining Hard Negatives (Loop until > 2000 total) ===")
    print(f"Loading Retriever: {E5_MODEL_NAME}...")
    retrieval_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
    
    print(f"Loading LLM: {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,  
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        trust_remote_code=True
    )
    
    combined_pool = musique_pool + hotpot_pool
    random.shuffle(combined_pool)
    
    iteration = 0
    while len(final_train_data) < TARGET_TOTAL_SIZE:
        iteration += 1
        needed = TARGET_TOTAL_SIZE - len(final_train_data)
        print(f"\n--- Iteration {iteration} | Current Size: {len(final_train_data)} | Needed: >0 ---")
        
        if len(combined_pool) == 0:
            print("Warning: Run out of source data!")
            break
            
        current_batch_size = min(STEP2_BATCH_SIZE, len(combined_pool))
        batch_samples = combined_pool[:current_batch_size]
        combined_pool = combined_pool[current_batch_size:]
        
        print(f"Inferencing on batch of {len(batch_samples)} samples...")
        incorrect_items = run_inference_batch(batch_samples, llm, retrieval_model)
        
        print(f"Found {len(incorrect_items)} incorrect samples in this batch.")
        final_train_data.extend(incorrect_items)
        
        print(f"Saving checkpoint to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for item in final_train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n==========================================")
    print(f"Done! Final Dataset Size: {len(final_train_data)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("==========================================")

if __name__ == "__main__":
    main()