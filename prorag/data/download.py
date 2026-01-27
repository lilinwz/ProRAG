import json
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

def process_data(input_source, output_file=None, dump_file=None, repo_filename=None):
    print(f"Loading data from: {input_source}...")
    
    data_files = None
    if repo_filename:
        data_files = [f.strip() for f in repo_filename.split(',')]
        print(f"-> Target data files: {data_files}")

    if input_source.endswith(".json") or input_source.endswith(".jsonl"):
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Local file not found: {input_source}")
        print("-> Detected local file.")
        dt = load_dataset("json", data_files=input_source, split="train")
    else:
        print("-> Detected HuggingFace repository.")
        if data_files:
            dt = load_dataset(input_source, data_files=data_files, split="train")
        else:
            dt = load_dataset(input_source, split="train")

    print(f"Total items loaded: {len(dt)}")
    if dump_file:
        print(f"Saving RAW data to: {dump_file}")
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        with open(dump_file, "w", encoding='utf-8') as f:
            for row in tqdm(dt, desc="Dumping Raw"):
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"Success! Saved raw data to {dump_file}")
    else:
        print("-> No --dump_file provided, skipping raw data dump.")

    if output_file:
        data = []
        print(f"Processing items for CoT format...")
        for i, row in enumerate(tqdm(dt, desc="Processing")):
            chain = []
            q_decomp = row.get('question_decomposition', [])
            paragraphs = row.get('paragraphs', [])
            
            if not isinstance(q_decomp, list): q_decomp = []
            if not isinstance(paragraphs, list): paragraphs = []

            for item in q_decomp:
                for p in paragraphs:
                    if item.get('paragraph_support_idx') == p.get('idx'):
                        chain.append({
                            "subquery": item.get('question'), 
                            "subanswer": item.get('answer'), 
                            "paragraph": p.get('paragraph_text')
                        })
                        break
            
            data.append({
                "id": i, 
                "query": row.get("question", ""), 
                "chain_of_thought": chain, 
                "answer": row.get("answer", "")
            })

        print(f"Writing PROCESSED data to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")    
        print(f"Success! Saved {len(data)} processed items to {output_file}")
    else:
        print("-> No --output_file provided, skipping processed data save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader and Preprocessor")
    parser.add_argument("--input", type=str, required=True, help="Path to local file OR HuggingFace dataset ID")
    parser.add_argument("--repo_filename", type=str, default=None, help="Specific filenames inside HF repo, comma separated (e.g. 'a.json,b.parquet')")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the PROCESSED json (Optional)")    
    parser.add_argument("--dump_file", type=str, default=None, help="Path to save the RAW jsonl (Required if you want raw dump)")
    
    args = parser.parse_args()
    process_data(args.input, args.output_file, args.dump_file, args.repo_filename)