import json

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train4.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for item in data:
    idx = item['id']
    query = item['query']
    chain = item['chain_of_thought']
    answer = item['answer']
    cot1 = item['cot1']
    cot2 = item['cot2']
    cot3 = item['cot3']
    
    info_in = query
    info_out = ""

    data_item = [{"role": "user", "content": info_in}]
    for i, sub_item in enumerate(chain):
        subquery = sub_item['subquery']
        subanswer = sub_item['sub-answer']
        context = sub_item['paragraph']
        
        info_out += f"<think>\n{cot1[i]}\n</think>\n<retrieval>\n{subquery}\n</retrieval>\n\n"
        data_item.append({"role": "assistant", "content": info_out})

        info_in = context
        data_item.append({"role": "user", "content": info_in})
        info_out = f"<think>\n{cot2[i]}\n</think>\n<><subanswer>\n{subanswer}\n</subanswer>\n\n"

    info_out += f"<think>\n{cot3}\n</think>\n<answer>\n{answer}\n</answer>"
    data_item.append({"role": "assistant", "content": info_out})

    new_data.append(data_item)

print(len(new_data))
with open("/home/v-zhaowan/zhaowang/rag/data/train.json", "w") as f:
    f.write(json.dumps(new_data, ensure_ascii=False, indent=4))