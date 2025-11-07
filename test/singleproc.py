import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import numpy as np
import collections
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any

MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft/checkpoint-770"
TEST_DATA_PATH = "/home/aiscuser/ds/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_dev.jsonl"
OUTPUT_RESULTS_FILE = "/home/aiscuser/ds/zhaowang/rag/test/rag_inference_results.jsonl"
E5_MODEL_NAME = 'intfloat/e5-large-v2'

MAX_GENERATION_LENGTH = 1024
MAX_HOP = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

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
    def __init__(self, paragraphs: List[dict], model: SentenceTransformer):
        self.raw_paragraphs = paragraphs or []
        self.model = model
        self.corpus = [f'passage: {p.get("paragraph_text", "")}' for p in self.raw_paragraphs]
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
            docs.append(f"Document {p.get('idx','?')} (Title: {p.get('title','')}): {p.get('paragraph_text','')}")
        return docs

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

def generate_step(model, tokenizer, prompt: str):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    stop_tokens = ["<retrieval>", "<|im_end|>", "<|endoftext|>"]
    stopping_criteria = StoppingCriteriaList([StopOnKeywords(tokenizer, stop_tokens)])

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
    
    response = tokenizer.decode(
        gen_output_ids[0, input_ids.shape[1]:], 
        skip_special_tokens=False
    )
    return response

def run_rag_inference(model, tokenizer, question: str, retriever: E5VectorRetriever):
    user_content = INSTRUCTION_TEMPLATE.format(question=question)
    current_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    final_answer = "Generation failed"
    full_generation_trace = ""

    for hop in range(MAX_HOP):
        generated_part = generate_step(model, tokenizer, current_prompt)
        current_prompt += generated_part
        full_generation_trace += generated_part

        if "<answer>" in generated_part:
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_part, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                print(final_answer)
            break
        
        if "<subquery>" in generated_part:
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", generated_part, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = retriever.retrieve(subquery, top_k=3)
                retrieved_docs_text = "\n".join(retrieved_docs) if retrieved_docs else "No documents found."
                current_prompt += f"{retrieved_docs_text}</retrieval>\n"
                full_generation_trace += f"{retrieved_docs_text}</retrieval>\n"
            else:
                print(f"Warning: Malformed <subquery> tag in hop {hop+1}. Terminating.")
                break
        else:
            break
            
    if hop == MAX_HOP - 1 and "<answer>" not in current_prompt:
        final_answer = "Max hops reached"

    return final_answer, full_generation_trace


def main():
    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print(f"Loading retriever model: {E5_MODEL_NAME}")
    similarity_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)

    print(f"Loading test data from: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_samples)} test samples.")

    all_em_scores = []
    all_f1_scores = []
    result = []
    wrong_format_counter = 0

    max_workers = min(4, (len(test_samples) or 1))
    print(f"Prefetching retrievers with ThreadPoolExecutor, max_workers={max_workers}")
    
    print(f"\n--- Starting RAG Inference ---")
    for i, sample in tqdm(enumerate(test_samples), total=len(test_samples), desc="Running RAG inference"):
        question = sample["question"]
        golden_answers = [sample["answer"]] + sample.get("answer_aliases", [])
        
        retriever = E5VectorRetriever(sample.get("paragraphs", []), similarity_model)
        predicted_answer, full_generation_trace = run_rag_inference(model, tokenizer, question, retriever)

        if predicted_answer == "Generation failed" or predicted_answer == "Max hops reached":
            wrong_format_counter += 1
            em_score = 0.0
            f1_score = 0.0
        else:
            em_score = 1.0 if any(predicted_answer.strip().lower() == g.strip().lower() for g in golden_answers) else 0.0
            f1_score = calculate_f1_score(predicted_answer, golden_answers)
        
        result.append({
            "id": sample["id"],
            "question": question,
            "predicted_answer": predicted_answer,
            "golden_answers": golden_answers,
            "em_score": em_score,
            "f1_score": f1_score,
            "generation_trace": full_generation_trace
        })
        
        all_em_scores.append(em_score)
        all_f1_scores.append(f1_score)

        if i < 5:
            print(f"\n--- Sample {i+1} (ID: {sample['id']}) ---")
            print(f"Question: {question}")
            print(f"Full Model Generation Trace:\n{full_generation_trace}")
            print(f"\nPredicted Answer: {predicted_answer}")
            print(f"Golden Answers: {golden_answers}")
            print(f"EM Score: {em_score:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            print("-" * 40)

    if not all_em_scores:
        print("\nEvaluation could not be completed. No scores were calculated.")
        return

    average_em = np.mean(all_em_scores)
    average_f1 = np.mean(all_f1_scores)

    print("\n--- Overall RAG Performance ---")
    print(f"Total Samples Evaluated: {len(test_samples)}")
    print(f"Average EM Score: {average_em:.4f}")
    print(f"Average F1 Score: {average_f1:.4f}")

    print(f"\nSaving all results to {OUTPUT_RESULTS_FILE}...")
    with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()