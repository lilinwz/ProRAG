import os
import json
import re
import torch
import random
import numpy as np
import collections
from tqdm import tqdm
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from prorag.utils.prompts import build_user_prompt
from prorag.utils.retriever import RemoteRetriever
import argparse

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

def main(args):
    print(f"Connecting to Retrieval Server")
    retriever = RemoteRetriever()

    print(f"Loading vLLM Model: {args.model_path}")
    llm = LLM(
        model=args.model_path, 
        gpu_memory_utilization=0.80,
        tensor_parallel_size=2,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_seq_length,
        stop=["</subquery>", "</subanswer>", "</answer>", "<|im_end|>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False
    )

    print(f"Loading Data from: {args.data_path}")
    test_samples = []
    file_paths = [p.strip() for p in args.data_path.split(',')]
    for path in file_paths:
        print(f" -> Reading file: {path} ...")
        if not os.path.exists(path):
            print(f"    [Warning] File not found: {path}, skipping.")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            current_file_count = 0
            for line in f:
                if line.strip():
                    test_samples.append(json.loads(line))
                    current_file_count += 1
            print(f"    Loaded {current_file_count} samples.")
    
    print(f"Total samples loaded from {len(file_paths)} files: {len(test_samples)}")
    
    random.shuffle(test_samples)            
    holdout_data = test_samples[:args.sample_size]
    test_samples = test_samples[args.sample_size:]

    print(f"Saving {len(holdout_data)} items to {args.holdout_path} ...")
    os.makedirs(os.path.dirname(args.holdout_path), exist_ok=True)
    with open(args.holdout_path, 'w', encoding='utf-8') as f:
        for item in holdout_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    global_em = []
    total_saved_steps = 0 
    total_batches = (len(test_samples) + args.batch_size - 1) // args.batch_size
    for batch_idx, i in enumerate(range(0, len(test_samples), args.batch_size)):
        batch_samples = test_samples[i : i + args.batch_size]
        print(f"\n=== Processing Batch {batch_idx + 1}/{total_batches} (Size: {len(batch_samples)}) ===")
        
        current_states = [RequestState(s, retriever) for s in batch_samples]

        for hop in range(args.max_iter):
            active_states = [s for s in current_states if not s.finished]
            if not active_states:
                break
            
            print(f"--- Hop {hop + 1}/{args.max_iter}: Processing {len(active_states)} active requests ---")
            prompts = [s.prompt for s in active_states]

            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            queries_to_search = []
            indices_to_search = []
            for i, output in tqdm(list(enumerate(outputs)), total=len(outputs)):
                state = active_states[i]
                generated_text = output.outputs[0].text
                step_record = {
                    "input": state.prompt,
                    "output": generated_text
                }
                state.step_history.append(step_record)
                
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

        print(f"  Evaluating Batch {batch_idx + 1}...")
        batch_save_lines = []
        
        for state in current_states:
            if not state.final_answer:
                state.final_answer = "Max hops reached"
            
            answer = state.sample["answer"] if "answer" in state.sample else state.sample["Answer"]            
            em = 1.0 if state.final_answer.lower() == answer.lower() else 0.0            
            global_em.append(em)
            
            if em == 1.0:
                for step in state.step_history:
                    record = {
                        "id": state.id, 
                        "input": step["input"],
                        "output": step["output"]
                    }
                    batch_save_lines.append(json.dumps(record, ensure_ascii=False))
        
        if batch_save_lines:
            with open(args.output_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(batch_save_lines) + '\n')
            total_saved_steps += len(batch_save_lines)
        
        print(f"  Batch {batch_idx + 1} finished. Saved {len(batch_save_lines)} interaction steps.")

    avg_em = np.mean(global_em) if global_em else 0
    
    print(f"\n==========================================")
    print(f"Final Results")
    print(f"Total Questions Processed: {len(test_samples)}")
    print(f"Total Steps Saved: {total_saved_steps}")
    print(f"Average EM Score: {avg_em:.4f}")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Inference and Filter Positive Samples.")
   
    parser.add_argument("--model_path", type=str, required=True, help="Path to the vLLM compatible model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the collected positive samples (EM=1)")
    parser.add_argument("--holdout_path", type=str, default="data/train_rl.jsonl", help="Path to save the rl training data")

    parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size")
    parser.add_argument("--sample_size", type=int, default=10000, help="Training data size for RL")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length (tokens)")
    parser.add_argument("--max_iter", type=int, default=13, help="Maximum reasoning hops (iterations)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()
    main(args)
