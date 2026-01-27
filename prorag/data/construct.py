import json
import argparse
import os
from tqdm import tqdm
from prorag.utils.prompts import build_user_prompt

def main(args):
    print(f"Reading data from: {args.input_file}")
    raw_data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    print(f"Loaded {len(raw_data)} items. Starting formatting...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    processed_count = 0
    sft_sample_count = 0
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for item in tqdm(raw_data, desc="Formatting"):
            if "response_data" in item and isinstance(item["response_data"], dict):
                thought_list = item["response_data"].get("think_process")
                chain_of_thought = item["response_data"].get("new_chain_of_thought", [])
                
            if not thought_list or not chain_of_thought:
                continue
            
            if len(thought_list) != 2 * len(chain_of_thought) + 1:
                continue

            question = item.get("query")
            answer = item.get("answer")

            user_content = build_user_prompt(question)
            history = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
            for i, subitem in enumerate(thought_list):
                uid = subitem["id"]
                thought = subitem["think"]

                if uid == "final_answer":
                    completion = f"{thought}</step>\n<answer>\n{answer}</answer>"
                else:
                    utype = uid.split("_")[0]
                    ans = int(uid.split("_")[1]) - 1
                    if utype == "subquery":
                        completion = f"{thought}</step>\n<subquery>\n{chain_of_thought[ans]['subquery']}</subquery>\n"    
                    else:
                        completion = f"{thought}</step>\n<subanswer>\n{chain_of_thought[ans]['subanswer']}</subanswer>\n"    

                out = {
                    "id": sft_sample_count,
                    "history": history,
                    "completion": completion
                }                
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                sft_sample_count += 1

                if uid.startswith("subquery"):
                    history += f"<retrieval>\n{chain_of_thought[ans]['paragraph']}</retrieval>\n"
                
                history += "<step>\n"
            processed_count += 1

    print(f"✅ Formatting Done!")
    print(f"Original items processed: {processed_count}/{len(raw_data)}")
    print(f"Total SFT samples generated: {sft_sample_count}")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input jsonl file (with thought process)")
    parser.add_argument("--output_file", type=str, required=True, help="Output SFT jsonl file")
    args = parser.parse_args()
    
    main(args)