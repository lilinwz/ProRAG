import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from collections import Counter

LORA_PATH = "/home/v-zhaowan/zhaowang/rag/save/731/final_adapter"
TEST_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
MAX_MODEL_INPUT_LENGTH = 2048
MAX_GENERATION_LENGTH = 512
MAX_HOP = 5

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

class SimpleBM25Retriever:
    def __init__(self, paragraphs):
        self.raw_paragraphs = paragraphs
        self.corpus = [p["paragraph_text"] for p in paragraphs]
        self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=3):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        retrieved_docs_with_info = []
        for i in top_indices:
            retrieved_docs_with_info.append(
                f"Document {self.raw_paragraphs[i]['idx']} (Title: {self.raw_paragraphs[i]['title']}): "
                f"{self.raw_paragraphs[i]['paragraph_text']}"
            )
        return retrieved_docs_with_info

def generate(model, tokenizer, prompt, max_input_len=MAX_MODEL_INPUT_LENGTH, max_gen_len=MAX_GENERATION_LENGTH):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    if input_ids.shape[1] > max_input_len:
        input_ids = input_ids[:, -max_input_len:]
        
    stop_tokens = ["<retrieval>", "<|im_end|>"]
    stopping_criteria = StoppingCriteriaList([StopOnKeywords(tokenizer, stop_tokens)])

    gen_output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_gen_len, 
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria
    )
    
    response = tokenizer.decode(
        gen_output_ids[0, input_ids.shape[1]:], 
        skip_special_tokens=False
    )
    return response

def run_rag_inference(model, tokenizer, question, retriever, max_hops=MAX_HOP):
    query_list = []
    context_list = []
    generated_responses = []
    current_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    final_answer = None

    for hop in range(max_hops):
        generated_assistant_response = generate(model, tokenizer, current_prompt)
        current_prompt += generated_assistant_response
        generated_responses.append(generated_assistant_response)

        if generated_assistant_response.strip().endswith("<|im_end|>"):
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_assistant_response, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            break
        elif generated_assistant_response.strip().endswith("<retrieval>"):
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", generated_assistant_response, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                query_list.append(subquery)
                retrieved_docs = retriever.retrieve(subquery, top_k=3)
                retrieved_docs_text = "\n".join(retrieved_docs)
                context_list.append(retrieved_docs_text)
                current_prompt += f"\n{retrieved_docs_text}\n</retrieval>\n"
            else:
                break
        else:
            print(f"Warning: Model did not generate <retrieval> or <answer> in hop {hop+1}. Generated: {generated_assistant_response}")
            final_answer = "Non-valid response."
            break

    if final_answer is None:
        print(f"Warning: Model did not generate <retrieval> or <answer> in hop {hop+1}. Generated: {generated_assistant_response}")
        final_answer = "Non-valid response."

    return final_answer, query_list, context_list, generated_responses

def calculate_f1(prediction, ground_truth_answers):
    if not isinstance(ground_truth_answers, list):
        ground_truth_answers = [ground_truth_answers]

    max_f1 = 0.0
    prediction_tokens = prediction.lower().split()

    if not prediction_tokens:
        return 0.0

    for gt_ans in ground_truth_answers:
        ground_truth_tokens = gt_ans.lower().split()
        if not ground_truth_tokens:
            continue

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            current_f1 = 0.0
        else:
            precision = num_common / len(prediction_tokens)
            recall = num_common / len(ground_truth_tokens)
            current_f1 = (2 * precision * recall) / (precision + recall)

        max_f1 = max(max_f1, current_f1)

    return max_f1

if __name__ == "__main__":
    print(f"Loading model and tokenizer from {LORA_PATH}...")
    
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    custom_special_tokens = [
        "<think>", "</think>",
        "<subquery>", "</subquery>",
        "<retrieval>", "</retrieval>",
        "<subanswer>", "</subanswer>",
        "<answer>", "</answer>"
    ]
    num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": custom_special_tokens})
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added_tokens} new special tokens to the tokenizer and resized model embeddings.")

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    
    model.eval()

    print(f"Loading test data from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_samples)} test samples.")

    all_f1_scores = []
    for i, sample in tqdm(enumerate(test_samples), total=len(test_samples), desc="Running RAG inference"):
        question = sample["question"]
        golden_answer = [sample["answer"]]
        golden_answer.extend(sample["answer_aliases"])
        
        all_paragraphs_for_retrieval = sample["paragraphs"] 
        retriever = SimpleBM25Retriever(all_paragraphs_for_retrieval)

        predicted_answer, subquery, retrieved_context, generated_text = run_rag_inference(model, tokenizer, question, retriever)
        
        f1 = calculate_f1(predicted_answer, golden_answer)
        all_f1_scores.append(f1)

        if i < 5:
            print(f"\n--- Sample {i+1} (ID: {sample['id']}) ---")
            print(f"Question: {question}")
            print(f"Full Model Generation (before retrieval): {generated_text}")
            print(f"Model's Retrieval Query: {subquery}")
            print(f"Retrieved Context:\n{retrieved_context}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Golden Answer: {golden_answer}")
            print(f"Is Answerable (Dataset): {sample['answerable']}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 30)

    if all_f1_scores:
        average_f1 = np.mean(all_f1_scores)
        print(f"\n--- Overall RAG Performance ---")
        print(f"Total Samples Evaluated: {len(all_f1_scores)}")
        print(f"Average F1 Score: {average_f1:.4f}")
    else:
        print("\nNo F1 scores calculated. Check your data and model output.")

