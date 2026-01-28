import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main(args):
    print(f"Loading PRM from: {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()

    print(f"Reading raw data from: {args.data_path} ...")
    data_lines = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data_lines = [json.loads(line) for line in f if line.strip()]

    total_lines = len(data_lines)
    print(f"Total raw lines: {total_lines}")

    processed_hashes = set()
    existing_count = 0
    if os.path.exists(args.output_path):
        print(f"Output file detected at {args.output_path}, scanning existing data...")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                unique_key = item.get("input", "") + item.get("output", "")
                processed_hashes.add(unique_key)
                existing_count += 1
        print(f"Loaded {existing_count} existing items, these will be skipped.")
    else:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        print("No output file detected, starting from scratch.")

    filtered_data_count = 0
    batch_texts = []
    batch_indices = []
    print("Starting processing (append mode)...")
    with open(args.output_path, 'a', encoding='utf-8') as f_out:
        for idx, item in tqdm(enumerate(data_lines), total=total_lines, desc="Filtering"):            
            unique_key = item.get("input", "") + item.get("output", "")
            if unique_key in processed_hashes:
                continue

            full_text = item["input"] + item["output"] + tokenizer.eos_token
            batch_texts.append(full_text)
            batch_indices.append(idx)

            if len(batch_texts) == args.batch_size or idx == total_lines - 1:
                if not batch_texts:
                    continue

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length
                ).to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    scores = outputs.logits.squeeze(-1).float().cpu().numpy()

                for i, score in enumerate(scores):
                    original_idx = batch_indices[i]
                    original_item = data_lines[original_idx]                    
                    original_item["prm_score"] = float(score)

                    if score >= 0:
                        filtered_data_count += 1
                        f_out.write(json.dumps(original_item, ensure_ascii=False) + "\n")
                
                f_out.flush() 
                batch_texts = []
                batch_indices = []

    print("=" * 40)
    print(f"Filtering completed!")
    print(f"Newly kept this run: {filtered_data_count}")
    print(f"Total kept (including existing): {existing_count + filtered_data_count}")
    print(f"Results saved to: {args.output_path}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRM Filtering Script")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the PRM model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the filtered output JSONL")

    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for tokenizer (default: 4096)")

    args = parser.parse_args()
    main(args)