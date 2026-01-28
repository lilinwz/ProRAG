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
from prorag.utils.prompts import build_user_prompt
from prorag.utils.retriever import RemoteRetriever

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

class RequestState:
    def __init__(self, sample, retriever: RemoteRetriever):
        self.id = sample.get("id", str(random.randint(0, 100000)))
        self.sample = sample
        self.question = sample["question"] if "question" in sample else sample["Question"]
        self.prompt = f"<|im_start|>user\n{build_user_prompt(self.question)}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
        self.full_trace = "<step>\n"
        self.step_history = []
        self.finished = False
        self.final_answer = ""
        self.retriever = retriever

def main():
    print(f"Connecting to Retrieval Server")
    retriever = RemoteRetriever()

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

if __name__ == "__main__":
    main()
