import json

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train4.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
# a = {}
for item in data:
    idx = item['id']
    query = item['query']
    chain = item['chain_of_thought']
    answer = item['answer']
    cot1 = item['cot1']
    cot2 = item['cot2']
    cot3 = item['cot3']
    
    # if len(chain) in a:
    #     a[len(chain)] += 1
    # else:
    #     a[len(chain)] = 1
    if len(chain) == 1:
        continue

    data_item = {"user": query, "assistant": []}
    info = ""
    for i, sub_item in enumerate(chain):
        subquery = sub_item['subquery']
        subanswer = sub_item['sub-answer']
        context = sub_item['paragraph']
        
        info += f"<think>\n{cot1[i]}\n</think>\n<subquery>\n{subquery}\n</subquery>\n<retrieval>"
        data_item["assistant"].append(info)

        info = f"\n{context}\n</retrieval>\n"
        data_item["assistant"].append(info)
        info = f"<think>\n{cot2[i]}\n</think>\n<subanswer>\n{subanswer}\n</subanswer>\n"

    info += f"<think>\n{cot3}\n</think>\n<answer>\n{answer}\n</answer>\n"
    data_item["assistant"].append(info)

    new_data.append(data_item)

print(len(new_data))
# print(a)
with open("/home/v-zhaowan/zhaowang/rag/data/train.json", "w") as f:
    f.write(json.dumps(new_data, ensure_ascii=False, indent=4))