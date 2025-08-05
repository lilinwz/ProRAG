import json

# 定义你所有的特殊tokens，以防它们出现在内容里
SPECIAL_TOKENS = {
    "<think>", "</think>", "<subquery>", "</subquery>",
    "<retrieval>", "</retrieval>", "<subanswer>", "</subanswer>",
    "<answer>", "</answer>"
}

with open("/home/v-zhaowan/zhaowang/rag/data/raw/train_full4.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Starting audit of {len(data)} raw items...")

error_count = 0
for item in data:
    item_id = item.get('id', 'N/A')

    # 嫌疑人一：检查空值或纯空格
    text_fields = {'query': item['query'], 'answer': item['answer'], 'cot3': item['cot3']}
    for name, value in text_fields.items():
        if not value or str(value).isspace():
            print(" ")
            # print(f"[ERROR] ID: {item_id} - Field '{name}' is empty or just whitespace.")
            # error_count += 1

    # 检查chain内部和对应的cot
    if 'chain_of_thought' not in item or not isinstance(item['chain_of_thought'], list):
        print(f"[ERROR] ID: {item_id} - 'chain' is missing or not a list.")
        error_count += 1
        continue
        
    if len(item['chain_of_thought']) != len(item.get('cot1', [])) or len(item['chain_of_thought']) != len(item.get('cot2', [])):
        print(f"[ERROR] ID: {item_id} - Mismatch in lengths: chain({len(item['chain_of_thought'])}), cot1({len(item.get('cot1', []))}), cot2({len(item.get('cot2', []))})")
        error_count += 1

    for i, sub_item in enumerate(item['chain_of_thought']):
        sub_fields = {
            f'chain[{i}].subquery': sub_item['subquery'],
            f'chain[{i}].sub-answer': sub_item['sub-answer'],
            f'chain[{i}].paragraph': sub_item['paragraph'],
            f'cot1[{i}]': item['cot1'][i],
            f'cot2[{i}]': item['cot2'][i]
        }
        for name, value in sub_fields.items():
            # 嫌疑人一
            if not value or str(value).isspace():
                print(" ")
                # print(f"[ERROR] ID: {item_id} - Field '{name}' is empty or just whitespace.")
                # error_count += 1
            # 嫌疑人二
            if any(token in str(value) for token in SPECIAL_TOKENS):
                print(f"[ERROR] ID: {item_id} - Field '{name}' contains a special token.")
                error_count += 1

if error_count == 0:
    print("\nAudit complete. No issues found!")
else:
    print(f"\nAudit complete. Found {error_count} potential issues.")