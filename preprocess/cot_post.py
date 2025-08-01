import json
import random

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train_full4.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

random.shuffle(data)
new_data = []

for item in data:
    idx = item['id']
    query = item['query']
    chain = item['chain_of_thought']
    answer = item['answer']
    cot1 = item['cot1']
    cot2 = item['cot2']
    cot3 = item['cot3']
    
    if len(chain) == 1:
        continue

    data_item = {"user": query, "assistant": []}
    info = ""
    for i, sub_item in enumerate(chain):
        subquery = sub_item['subquery']
        subanswer = sub_item['sub-answer']
        context = sub_item['paragraph']
        
        data_item["assistant"].append("<think>\n")
        data_item["assistant"].append(cot1[i])
        data_item["assistant"].append("</think>\n")

        data_item["assistant"].append("<subquery>\n")
        data_item["assistant"].append(subquery)
        data_item["assistant"].append("</subquery>\n")

        data_item["assistant"].append("<retrieval>\n")
        data_item["assistant"].append(context)
        data_item["assistant"].append("</retrieval>\n")

        data_item["assistant"].append("<think>\n")
        data_item["assistant"].append(cot2[i])
        data_item["assistant"].append("</think>\n")

        data_item["assistant"].append("<subanswer>\n")
        data_item["assistant"].append(subanswer)
        data_item["assistant"].append("</subanswer>\n")

    data_item["assistant"].append("<think>\n")
    data_item["assistant"].append(cot3)
    data_item["assistant"].append("</think>\n")

    data_item["assistant"].append("<answer>\n")
    data_item["assistant"].append(answer)
    data_item["assistant"].append("</answer>\n")

    new_data.append(data_item)

print(len(new_data))

partition = 12000
with open("/home/v-zhaowan/zhaowang/rag/data/train_sft.json", "w") as f:
    f.write(json.dumps(new_data[:partition], ensure_ascii=False, indent=4))
with open("/home/v-zhaowan/zhaowang/rag/data/train_rl.json", "w") as f:
    f.write(json.dumps(new_data[partition:], ensure_ascii=False, indent=4))

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train_sft.json", "w") as f:
    f.write(json.dumps(data[:partition], ensure_ascii=False, indent=4))
with open("/home/v-zhaowan/zhaowang/rag/data/raw/train_rl.json", "w") as f:
    f.write(json.dumps(data[partition:], ensure_ascii=False, indent=4))