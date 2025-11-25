import os
import json
from tqdm import tqdm

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            raise ValueError(f"{path} is not a list JSON!")
        
def load_jsonl_file(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def merge_raw_folder(raw_folder, output_path):
    all_items = []
    files = [raw_folder+".json.part0", raw_folder+".json.part1", raw_folder+".json.part2", raw_folder+".json.part3"]

    for filename in files:
        path = filename

        if filename.endswith(".json"):
            print(f"📄 Loading JSON: {filename}")
            all_items.extend(load_json_file(path))

        else:
            print(f"📄 Loading JSONL: {filename}")
            all_items.extend(load_jsonl_file(path))

    print(f"\nTotal items merged: {len(all_items)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Merged file saved to: {output_path}")


if __name__ == "__main__":
    raw_folder = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/tree_mulsique"
    output_path = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/tree_mulsique.jsonl"
    merge_raw_folder(raw_folder, output_path)
