import os
import json
import re
import torch
import numpy as np
import collections
import ast
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, set_start_method, Queue
from typing import List, Tuple

set_start_method("spawn", force=True)

MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft/checkpoint-770"
DATA_PATH = "/home/aiscuser/ds/zhaowang/rag/data/2wiki/train.jsonl"
SAVE_PATH = "/home/aiscuser/ds/zhaowang/rag/data/train_rl.jsonl"
E5_MODEL_NAME = "intfloat/e5-large-v2"

MAX_GENERATION_LENGTH = 1024
MAX_HOP = 8

# ⚙️ 这里控制并行度
NUM_GPUS = torch.cuda.device_count()
WORKERS_PER_GPU = 3
TOTAL_PROCESSES = NUM_GPUS * WORKERS_PER_GPU

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

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
    def __init__(self, paragraphs: List[Tuple[str, List[str]]], model: SentenceTransformer):
        self.raw_paragraphs = paragraphs or []
        self.model = model
        self.corpus = []
        for p in self.raw_paragraphs:
            passage_text = "\n".join(s for s in p[1])
            self.corpus.append(f"title: {p[0]}\npassage: {passage_text}")
        if not self.corpus:
            self.corpus_embeddings = None
            return
        emb = self.model.encode(self.corpus, convert_to_tensor=False, show_progress_bar=False)
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        self.corpus_embeddings = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-12
        self.corpus_embeddings = self.corpus_embeddings / norms

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.corpus_embeddings is None or not query:
            return []
        q = f'query: {query}'
        q_emb = self.model.encode(q, convert_to_tensor=False, show_progress_bar=False)
        if hasattr(q_emb, "cpu"):
            q_emb = q_emb.cpu().numpy()
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        scores = np.dot(self.corpus_embeddings, q_emb)
        k = min(top_k, self.corpus_embeddings.shape[0])
        idxs = np.argsort(-scores)[:k]
        docs = []
        for idx in idxs:
            p = self.raw_paragraphs[int(idx)]
            passage_text = "\n".join(s for s in p[1])
            docs.append(f"Title: {p[0]}: {passage_text}")
        return docs

def generate_step(model, tokenizer, prompt: str):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    stop_tokens = ["<retrieval>", "<|im_end|>", "<|endoftext|>"]
    stopping_criteria = StoppingCriteriaList([StopOnKeywords(tokenizer, stop_tokens)])
    with torch.no_grad():
        gen_output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_GENERATION_LENGTH,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
    return tokenizer.decode(gen_output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)


def run_rag_inference(model, tokenizer, question: str, retriever: E5VectorRetriever):
    user_content = INSTRUCTION_TEMPLATE.format(question=question)
    prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    final_answer = "Generation failed"
    trace = ""
    num_hop = 0
    for hop in range(MAX_HOP):
        part = generate_step(model, tokenizer, prompt)
        num_hop += 1
        prompt += part
        trace += part
        if "<answer>" in part:
            match = re.search(r"<answer>(.*?)</answer>", part, re.DOTALL)
            if match:
                final_answer = match.group(1).strip()
            break
        if "<subquery>" in part:
            match = re.search(r"<subquery>(.*?)</subquery>", part, re.DOTALL)
            if match:
                subq = match.group(1).strip()
                docs = retriever.retrieve(subq)
                docs_text = "\n".join(docs) if docs else "No documents found."
                prompt += f"{docs_text}</retrieval>\n"
                trace += f"{docs_text}</retrieval>\n"
            else:
                break
        else:
            break
    return final_answer, trace, num_hop

def worker_proc(samples_slice, process_idx, gpu_id, queue):
    torch.cuda.set_device(gpu_id)
    print(f"[Proc {process_idx}] Using GPU {gpu_id} | {len(samples_slice)} samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu_id}"}
    )
    model.eval()

    retr_model = SentenceTransformer(E5_MODEL_NAME, device=f"cuda:{gpu_id}")

    results = []
    for i, sample in enumerate(tqdm(samples_slice, desc=f"Proc {process_idx}")):
        q = sample["question"]
        ground_truth = sample["answer"]
        retriever = E5VectorRetriever(json.loads(sample["context"]), retr_model)
        pred, trace, num_hop = run_rag_inference(model, tokenizer, q, retriever)
        em = 1.0 if pred.strip().lower() == ground_truth.strip().lower() else 0.0
        if em == 0.0 or num_hop > 3:
            results.append(sample)

    queue.put(results)
    print(f"[Proc {process_idx}] Done. Saved {len(results)} results")


# ============================================================
#                   主入口
# ============================================================

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    n = len(samples)
    print(f"Loaded {n} samples.")

    per_proc = (n + TOTAL_PROCESSES - 1) // TOTAL_PROCESSES
    slices = [samples[i * per_proc:(i + 1) * per_proc] for i in range(TOTAL_PROCESSES)]

    q = Queue()
    procs = []
    for i, s in enumerate(slices):
        gpu_id = i % NUM_GPUS
        p = Process(target=worker_proc, args=(s, i, gpu_id, q))
        p.start()
        procs.append(p)
    
    all_results = []
    results_collected = 0
    while results_collected < TOTAL_PROCESSES:
        batch = q.get()
        all_results.extend(batch)
        results_collected += 1

    for p in procs:
        p.join()

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
