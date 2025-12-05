import json
import random

DATA_PATH1 = "/home/v-zhaowan/ds/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
DATA_PATH2 = "/home/v-zhaowan/ds/zhaowang/rag/data/HotpotQA/train.jsonl"
OUTPUT_PATH = "/home/v-zhaowan/ds/zhaowang/rag/data/train_rl_tmp.jsonl"

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

if __name__ == "__main__":
    data = []

    raw_data = []
    with open(DATA_PATH1, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: 
                continue
            item = json.loads(line)
            if not item["id"].startswith("2hop"):
                raw_data.append(item)
    
    sampled_mul = random.sample(raw_data, min(len(raw_data), 2000))
    
    for sample in sampled_mul:
        question = sample["question"]
        answer = sample["answer"]
        paragraphs = sample["paragraphs"]
        
        user_content = INSTRUCTION_TEMPLATE.format(question=question)
        init_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
        
        data.append({
            "question": question,
            "answer": answer,
            "paragraphs": paragraphs,
            "init_prompt": init_prompt
        })

    raw_data = []
    with open(DATA_PATH2, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: 
                continue
            item = json.loads(line)
            if item.get("level") == "hard":
                raw_data.append(item)
    
    sampled_hot = random.sample(raw_data, min(len(raw_data), 1000))

    for sample in sampled_hot:
        question = sample["question"]
        answer = sample["answer"]
        context = sample["context"]
        paragraphs = []
        for title, sentences in zip(context["title"], context["sentences"]):
            text = '\n'.join(sentences)
            paragraphs.append({
                "title": title,
                "paragraph_text": text
            })
        
        user_content = INSTRUCTION_TEMPLATE.format(question=question)
        init_prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
        
        data.append({
            "question": question,
            "answer": answer,
            "paragraphs": paragraphs,
            "init_prompt": init_prompt
        }) 


    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples.")
