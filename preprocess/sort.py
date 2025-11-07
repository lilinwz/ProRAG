import json

input_path = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/mulsique_cot_v1.jsonl"
output_path = "/home/v-zhaowan/ds/zhaowang/rag/data/raw/mulsique_cot_v1.jsonl"

# 读取所有行并解析为JSON对象
lines = []
with open(input_path, "r") as fin:
    for line in fin:
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"警告：跳过无法解析的行: {line.strip()}")

# 按照id从小到大排序
# 这一步是可选的，但有助于在去重时保留具有相同id的第一个原始出现的项
lines.sort(key=lambda x: x["id"])

# 去重
deduplicated_lines = []
seen_ids = set()
for item in lines:
    idx = item.get("id")
    if idx is not None and idx not in seen_ids:
        deduplicated_lines.append(item)
        seen_ids.add(idx)

# 将排序并去重后的结果写入新文件
with open(output_path, "w") as fout:
    for item in deduplicated_lines:
        fout.write(json.dumps(item) + "\n")

print(f"文件已成功排序、去重并保存至: {output_path}")