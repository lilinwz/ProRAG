import json

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train3.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for item in data:
    for idx, i in enumerate(item['cot1']):
        start = i.find("</think>")
        if start >= 0:
            item['cot1'][idx] = i[start+10:]
        
        start = i.find("text")
        if start == 0:
            item['cot1'][idx] = i[5:]

    for idx, i in enumerate(item['cot2']):
        start = i.find("</think>")
        if start >= 0:
            item['cot2'][idx] = i[start+10:]

        start = i.find("text")
        if start == 0:
            item['cot2'][idx] = i[5:]

    start = item['cot3'].find("</think>")
    if start >= 0:
        item['cot3'] = item['cot3'][start+10:]
    
    start = i.find("text")
    if start == 0:
            item['cot3'] = item['cot3'][5:]

    new_data.append(item)

print(len(new_data))
with open("/home/v-zhaowan/zhaowang/rag/data/raw/train4.json", "w") as f:
    f.write(json.dumps(new_data, ensure_ascii=False, indent=4))