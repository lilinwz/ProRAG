import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import json
import re
import torch
import numpy as np
import collections
from tqdm import tqdm
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft"
TEST_DATA_PATH = "/home/aiscuser/ds/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_dev.jsonl"
OUTPUT_RESULTS_FILE = "/home/aiscuser/ds/zhaowang/rag/test/results/ours_sft.jsonl"
E5_MODEL_NAME = 'intfloat/e5-large-v2'

MAX_TOKENS = 1024
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

class RequestState:
    def __init__(self, sample, retrieval_model: E5VectorRetriever):
        self.id = sample["id"]
        self.sample = sample
        self.question = sample["question"]
        self.prompt = f"<|im_start|>user\n{INSTRUCTION_TEMPLATE.format(question=self.question)}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
        self.full_trace = "<step>\n"
        self.finished = False
        self.final_answer = ""
        self.retriever = E5VectorRetriever(sample.get("paragraphs", []), retrieval_model)

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
    print(f"Loading Retriever Model: {E5_MODEL_NAME} on {DEVICE}...")
    retrieval_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)

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

    all_states = [RequestState(s, retrieval_model) for s in test_samples]
    print("Starting Multi-hop RAG Inference with vLLM...")

    for hop in range(MAX_HOP):
        active_states = [s for s in all_states if not s.finished]
        if not active_states:
            break
        
        print(f"--- Hop {hop + 1}/{MAX_HOP}: Processing {len(active_states)} active requests ---")
        prompts = [s.prompt for s in active_states]

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

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
                    docs = state.retriever.retrieve(q_str, top_k=1)
                    
                    retrieval_block = f"\n<retrieval>\n{docs}\n</retrieval>\n<step>\n"
                    state.prompt += retrieval_block
                    state.full_trace += retrieval_block
                else:
                    state.prompt += "\n<retrieval>Error in query parsing</retrieval>\n<step>\n"

            else:
                state.prompt += "\n<step>\n"
                state.full_trace += "\n<step>\n"

    results, all_f1, all_em = [], [], []
    print("Evaluating results...")
    for state in all_states:
        if not state.final_answer:
            state.final_answer = "Max hops reached"
        
        gt_list = [state.sample["answer"]] + state.sample.get("answer_aliases", [])
        
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