import json

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train4.json", "r") as f:
    data = json.load(f)

new = []
idx = []
for item in data:
    if item['id'] not in idx:
        new.append(item)
        idx.append(item['id'])

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train4.json", "w") as f:
    f.write(json.dumps(new, ensure_ascii=False, indent=4))