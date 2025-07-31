import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from collections import Counter

LORA_PATH = "/home/v-zhaowan/zhaowang/rag/save/730/final_adapter"
DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/dev.json"
OUTPUT_PATH = "/home/v-zhaowan/zhaowang/rag/sample/sampled_data.json"
Sample_K = 5
MAX_DEPTH = 5
MAX_MODEL_INPUT_LENGTH = 2048
MAX_GENERATION_LENGTH = 512

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

def MCSample(model, tokenizer, prompt, retriever, depth=0):
    if depth >= MAX_DEPTH:
        return []

    sample_list = []
    for i in range(Sample_K):
        response = generate(model, tokenizer, prompt)
        current_prompt = prompt + response

        if response.strip().endswith("<|im_end|>"):
            sample_list.append(current_prompt)
        elif response.strip().endswith("<retrieval>"):
            subquery_match = re.search(r"<subquery>(.*?)</subquery>", response, re.DOTALL)
            if subquery_match:
                subquery = subquery_match.group(1).strip()
                retrieved_docs = retriever.retrieve(subquery, top_k=3)
                retrieved_docs_text = "\n".join(retrieved_docs)
                current_prompt += f"\n{retrieved_docs_text}\n</retrieval>\n"
                follow_samples = MCSample(model, tokenizer, current_prompt, retriever, depth + 1)
                sample_list.extend(follow_samples)

    return sample_list

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

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added_tokens} new special tokens to the tokenizer and resized model embeddings.")
    
    model.eval()

    print(f"Loading test data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)[:1000]
    print(f"Loaded {len(data)} test samples.")

    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        new_data = json.load(f)

    for i, sample in tqdm(enumerate(data), total=len(data), desc="Sampling"):
        if i < len(new_data):
            continue
        idx = sample["id"]
        question = sample["question"]
        init_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        all_paragraphs_for_retrieval = sample["paragraphs"] 
        retriever = SimpleBM25Retriever(all_paragraphs_for_retrieval)

        sample_list = MCSample(model, tokenizer, init_prompt, retriever)
    
        new_data.append({
            "id": idx,
            "question": question,
            "samples": sample_list
        })

        if i % 10 == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)