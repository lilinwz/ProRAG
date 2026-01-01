import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import json
import re
import torch
import random
import numpy as np
import collections
import requests
from tqdm import tqdm
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/v-zhaowan/ds/zhaowang/rag/save/sft"
# TEST_DATA_PATH = "/home/v-zhaowan/ds/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_dev.jsonl"
# TEST_DATA_PATH = "/home/v-zhaowan/ds/zhaowang/rag/data/2wiki/test.jsonl"
TEST_DATA_PATH = "/home/v-zhaowan/ds/zhaowang/rag/data/bamboogle/test.jsonl"
# TEST_DATA_PATH = "/home/v-zhaowan/ds/zhaowang/rag/data/HotpotQA/test.jsonl"
OUTPUT_RESULTS_FILE = "/home/v-zhaowan/ds/zhaowang/rag/test/results/ours_sft_bamboo.jsonl"
RETRIEVAL_SERVER_URL = "http://localhost:8000/retrieve"

MAX_TOKENS = 4096
MAX_HOP = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

class RemoteRetriever:
    def __init__(self, url: str, topk: int = 3):
        self.search_url = url
        self.topk = topk

    def batch_search(self, queries: List[str]) -> List[str]:
        results = self._batch_search(queries)['result']
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True 
        }
        return requests.post(self.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

class RequestState:
    def __init__(self, sample, retriever: RemoteRetriever):
        self.id = sample.get("id", str(random.randint(0, 100000)))
        self.sample = sample
        self.question = sample["question"] if "question" in sample else sample["Question"]
        self.prompt = f"<|im_start|>user\n{INSTRUCTION_TEMPLATE.format(question=self.question)}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
        self.full_trace = "<step>\n"
        self.finished = False
        self.final_answer = ""
        self.retriever = retriever

def calculate_f1_score(prediction: str, ground_truth_list: list) -> float:
    prediction_tokens = prediction.lower().split()
    best_f1 = 0.0
    for gt in ground_truth_list:
        gt_tokens = gt.lower().split()
        if not prediction_tokens or not gt_tokens:
            continue
        common = collections.Counter(prediction_tokens) & collections.Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(prediction_tokens)
        rec = num_same / len(gt_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def main():
    print(f"Connecting to Retrieval Server at: {RETRIEVAL_SERVER_URL}")
    try:
        requests.get(RETRIEVAL_SERVER_URL.replace("/retrieve", "/docs"), timeout=5)
        print("Server connection successful.")
    except Exception as e:
        print(f"Warning: Could not connect to server ({e}). Ensure it is running.")

    retriever = RemoteRetriever(RETRIEVAL_SERVER_URL)

    print(f"Loading vLLM Model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,  
        gpu_memory_utilization=0.80,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        stop=["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False
    )

    print(f"Loading Data: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line) for line in f]
    print(f"Total samples: {len(test_samples)}")

    all_states = [RequestState(s, retriever) for s in test_samples]
    print("Starting Multi-hop RAG Inference with vLLM...")

    for hop in range(MAX_HOP):
        active_states = [s for s in all_states if not s.finished]
        if not active_states:
            break
        
        print(f"--- Hop {hop + 1}/{MAX_HOP}: Processing {len(active_states)} active requests ---")
        prompts = [s.prompt for s in active_states]

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        queries_to_search = []
        indices_to_search = []
        for i, output in tqdm(list(enumerate(outputs)), total=len(outputs)):
            state = active_states[i]
            generated_text = output.outputs[0].text
            
            state.prompt += generated_text
            state.full_trace += generated_text
            
            if "</answer>" in generated_text:
                state.finished = True
                match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
                if match:
                    state.final_answer = match.group(1).strip()
            
            elif "</subquery>" in generated_text:
                sq_match = re.search(r"<subquery>(.*?)</subquery>", generated_text, re.DOTALL)
                if sq_match:
                    q_str = sq_match.group(1).strip()
                    queries_to_search.append(q_str)
                    indices_to_search.append(i)
                else:
                    state.prompt += "\n<retrieval>Error in query parsing</retrieval>\n<step>\n"

            else:
                state.prompt += "\n<step>\n"
                state.full_trace += "\n<step>\n"
        
        if queries_to_search:
            print(f"Batch retrieving {len(queries_to_search)} queries...")
            batch_results = retriever.batch_search(queries_to_search)
            
            for idx, doc_str in zip(indices_to_search, batch_results):
                state = active_states[idx]
                if not doc_str.strip(): doc_str = "No relevant documents found."        
                retrieval_block = f"\n<retrieval>\n{doc_str}\n</retrieval>\n<step>\n"
                state.prompt += retrieval_block
                state.full_trace += retrieval_block

    results, all_f1, all_em = [], [], []
    print("Evaluating results...")
    for state in all_states:
        if not state.final_answer:
            state.final_answer = "Max hops reached"
        
        answer = state.sample["answer"] if "answer" in state.sample else state.sample["Answer"]
        gt_list = [answer] + state.sample.get("answer_aliases", [])
        
        em = 1.0 if any(state.final_answer.lower() == g.lower() for g in gt_list) else 0.0
        f1 = calculate_f1_score(state.final_answer, gt_list)
        
        all_em.append(em)
        all_f1.append(f1)
        
        results.append({
            "id": state.id,
            "question": state.question,
            "prediction": state.final_answer,
            "gold": gt_list,
            "em": em,
            "f1": f1,
            "trace": state.full_trace,
        })

    avg_em = np.mean(all_em) if all_em else 0
    avg_f1 = np.mean(all_f1) if all_f1 else 0
    
    print(f"\n==========================================")
    print(f"Final Results (vLLM Accelerated)")
    print(f"Samples: {len(test_samples)}")
    print(f"EM Score: {avg_em:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"==========================================")

    print(f"Saving to {OUTPUT_RESULTS_FILE}...")
    with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

if __name__ == "__main__":
    main()
