import json

input_path = "/home/aiscuser/ds/zhaowang/rag/data/train_sft.jsonl"
output_path = "/home/aiscuser/ds/zhaowang/rag/data/train_sft_new.jsonl"

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

ans = 0
with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
        item = json.loads(line)
        
        idx = item["id"]
        question = item["question"]
        response_list = item["response"].split("<step>\n")
        
        user_content = INSTRUCTION_TEMPLATE.format(question=question)
        history = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"
        for i, text in enumerate(response_list[1:]):
            completion = text.split("<retrieval>")[0]
            out = {
                "id": ans,
                "history": history,
                "completion": completion
            }   

            history += text
            history += "<step>\n"
            ans += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
