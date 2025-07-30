import json

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train1.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

ans = 0 
new_data = []
for item in data:
    try:
        start = item['chain_of_thought'].find('{')
        end = item['chain_of_thought'].rfind('}') + 1
        item['chain_of_thought'] = json.loads(item['chain_of_thought'][start:end])["new_CoT"]
        new_data.append(item)
    except Exception as e:
        # print(f"Error processing item: {item['id']}, Error: {e}")
        # print(item['chain_of_thought'])
        # break
        ans += 1

print(ans)
print(len(new_data))
with open("/home/v-zhaowan/zhaowang/rag/data/raw/train2.json", "w") as f:
    f.write(json.dumps(new_data, ensure_ascii=False, indent=4))