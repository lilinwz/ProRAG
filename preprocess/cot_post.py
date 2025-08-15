import json
import random

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train_full.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
ans = 0

SPECIAL_TOKENS = {
    "<think>", "</think>", "<subquery>", "</subquery>",
    "<retrieval>", "</retrieval>", "<subanswer>", "</subanswer>",
    "<answer>", "</answer>"
}

for item in data:
    idx = item['id']
    query = item['query']
    chain = item['chain_of_thought']
    answer = item['answer']
    cot1 = item['cot1']
    cot2 = item['cot2']
    cot3 = item['cot3']

    is_error = False
    if not cot3 or str(cot3).isspace():
        is_error = True
    for i, sub_item in enumerate(chain):
        if not cot1[i] or str(cot1[i]).isspace():
            is_error = True
        if not cot2[i] or str(cot2[i]).isspace():
            is_error = True
        if any(token in str(cot1[i]) for token in SPECIAL_TOKENS):
            is_error = True
        if any(token in str(cot2[i]) for token in SPECIAL_TOKENS):
            is_error = True

    if is_error:
        ans += 1
        continue
    
    if len(chain) <= 1:
        ans += 1
        continue

    data_item = {"id": idx, "user": query, "assistant": []}
    info = ""
    for i, sub_item in enumerate(chain):
        subquery = sub_item['subquery']
        subanswer = sub_item['sub-answer']
        context = sub_item['paragraph']
        
        data_item["assistant"].append("<step>\n")
        data_item["assistant"].append(cot1[i])
        data_item["assistant"].append("</step>\n")

        data_item["assistant"].append("<subquery>\n")
        data_item["assistant"].append(subquery)
        data_item["assistant"].append("</subquery>\n")

        data_item["assistant"].append("<retrieval>\n")
        data_item["assistant"].append(context)
        data_item["assistant"].append("</retrieval>\n")

        data_item["assistant"].append("<step>\n")
        data_item["assistant"].append(cot2[i])
        data_item["assistant"].append("</step>\n")

        data_item["assistant"].append("<subanswer>\n")
        data_item["assistant"].append(subanswer)
        data_item["assistant"].append("</subanswer>\n")

    data_item["assistant"].append("<step>\n")
    data_item["assistant"].append(cot3)
    data_item["assistant"].append("</step>\n")

    data_item["assistant"].append("<answer>\n")
    data_item["assistant"].append(answer)
    data_item["assistant"].append("</answer>\n")

    new_data.append(data_item)

print(f"Total items: {len(data)}")
print(f"Filtered items: {len(new_data)}")
print(f"Errors found: {ans}")

partition1 = 10000
with open("/home/v-zhaowan/zhaowang/rag/data/raw/train_sft.json", "w") as f:
    f.write(json.dumps(new_data[:partition1], ensure_ascii=False, indent=4))
with open("/home/v-zhaowan/zhaowang/rag/data/raw/train_rl.json", "w") as f:
    f.write(json.dumps(new_data[partition1:], ensure_ascii=False, indent=4))

random.shuffle(new_data)
partition2 = 10000
with open("/home/v-zhaowan/zhaowang/rag/data/train_sft.json", "w") as f:
    f.write(json.dumps(new_data[:partition2], ensure_ascii=False, indent=4))