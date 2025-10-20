import json

data = []

def read_jsonl_head(file_path, num_lines=1):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            raw_data = json.loads(line.strip())
            chain = []
            for item in raw_data['question_decomposition']:
                for p in raw_data['paragraphs']:
                    if item['paragraph_support_idx'] == p['idx']:
                        chain.append({"subquery": item['question'], "sub-answer": item['answer'], "paragraph": p['paragraph_text']})
                        break
            data.append({"id": i, "query": raw_data["question"], "chain_of_thought": chain, "answer": raw_data["answer"]})

if __name__ == "__main__":
    file_name = "/home/v-zhaowan/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
    read_jsonl_head(file_name)
    print(len(data))
    print(data[0])
    with open("/home/v-zhaowan/zhaowang/rag/data/raw/train.json", "w") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))