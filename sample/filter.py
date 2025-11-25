import json
import re
import sys
import os

def validate_cot_logic(full_text: str) -> dict:
    """
    状态机逻辑：
    1. 验证标签闭合性（允许 <retrieval> 在结尾不闭合）。
    2. 验证流转逻辑：Step -> Subquery -> Retrieval -> Step ...
    """
    result = {"valid": False, "error": ""}

    # 1. 提取所有标签
    tags_raw = re.findall(r'<(/?)(\w+)>', full_text)

    if not tags_raw:
        result["error"] = "No tags found in text."
        return result

    # 2. 验证标签闭合性 & 提取纯净序列
    clean_sequence = []
    i = 0
    while i < len(tags_raw):
        slash, name = tags_raw[i]

        # 如果遇到的是闭合标签（如 </step>）却出现在开始位置，报错
        if slash == '/':
            result["error"] = f"Unexpected closing tag </{name}> found without start tag."
            return result

        # 检查是否是列表中的最后一个标签
        is_last_tag = (i + 1 >= len(tags_raw))

        if is_last_tag:
            # === 特殊逻辑修改处 ===
            # 如果是最后一个标签，且是 retrieval，允许不闭合
            if name == "retrieval":
                clean_sequence.append(name)
                break # 验证结束，进入流转检查
            else:
                # 其他标签（如 step, subquery）在结尾必须闭合，否则报错
                result["error"] = f"Unclosed tag <{name}> at the end of text."
                return result
        
        # 如果不是最后一个，检查下一个是否是对应的闭合标签
        next_slash, next_name = tags_raw[i+1]

        if next_slash == '/':
            # 是闭合标签，检查名字是否匹配
            if name != next_name:
                result["error"] = f"Tag mismatch: <{name}> closed by </{next_name}>."
                return result
            # 匹配成功，加入序列
            clean_sequence.append(name)
            i += 2
        else:
            # 下一个不是闭合标签（即连续两个开始标签，如 <step><subquery>），视为嵌套或未闭合错误
            result["error"] = f"Missing closing tag for <{name}> before <{next_name}> starts."
            return result

    # 3. 状态机流转检查 (逻辑保持不变)
    last_tag = None
    
    for idx, curr in enumerate(clean_sequence):
        # --- Step ---
        if curr == "step":
            if last_tag is None or last_tag == "subanswer":
                pass # Outer Step
            elif last_tag == "retrieval":
                pass # Inner Step
            else:
                result["error"] = f"Flow Error: <step> cannot follow <{last_tag}>. Expected Start, <subanswer>, or <retrieval>."
                return result

        # --- Subquery ---
        elif curr == "subquery":
            if last_tag != "step":
                result["error"] = f"Flow Error: <subquery> must follow <step>, but found <{last_tag}>."
                return result
            # 检查前一个 Step 是否是 Outer Step (即 Step 前面不能是 retrieval)
            prev_prev = clean_sequence[idx-2] if idx >= 2 else None
            if prev_prev == "retrieval":
                result["error"] = "Flow Error: <subquery> cannot follow an Inner Step. Expecting <subanswer>."
                return result

        # --- Retrieval ---
        elif curr == "retrieval":
            if last_tag != "subquery":
                result["error"] = f"Flow Error: <retrieval> must follow <subquery>, but found <{last_tag}>."
                return result

        # --- Subanswer ---
        elif curr == "subanswer":
            if last_tag != "step":
                result["error"] = f"Flow Error: <subanswer> must follow <step>, but found <{last_tag}>."
                return result
            # 检查前一个 Step 是否是 Inner Step (即 Step 前面必须是 retrieval)
            prev_prev = clean_sequence[idx-2] if idx >= 2 else None
            if prev_prev != "retrieval":
                result["error"] = "Flow Error: <subanswer> must follow an Inner Step (Step after retrieval)."
                return result

        # --- Answer ---
        elif curr == "answer":
            if last_tag != "step":
                result["error"] = f"Flow Error: <answer> must follow <step>, but found <{last_tag}>."
                return result
            # 检查前一个 Step 是否是 Outer Step
            prev_prev = clean_sequence[idx-2] if idx >= 2 else None
            if prev_prev == "retrieval":
                result["error"] = "Flow Error: <answer> cannot follow an Inner Step. Expecting <subanswer>."
                return result

        else:
            result["error"] = f"Unknown tag type: <{curr}>"
            return result
        
        last_tag = curr

    result["valid"] = True
    return result

def filter_file(input_filename, output_filename):
    print(f"Filtering {input_filename}...")
    print(f"Output will be saved to {output_filename}\n")
    
    total_lines = 0
    kept_lines = 0
    skipped_lines = 0
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f_in, \
             open(output_filename, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                total_lines += 1
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Line {line_num}] [Skipped] JSON Decode Error")
                    skipped_lines += 1
                    continue
                
                # 提取字段
                history = data.get('input', {}).get('history', "")
                chosen_data = data.get('chosen', {})
                new_step = chosen_data.get('new_step', "")
                
                # 基础判空
                if not history or not new_step:
                    if not new_step:
                        print(f"[Line {line_num}] [Skipped] Missing new_step")
                        skipped_lines += 1
                        continue

                # --- 检查 1: new_step 必须以 <step> 开头 ---
                if not new_step.strip().startswith("<step>"):
                    print(f"[Line {line_num}] [Skipped] [Prefix Error] new_step does not start with <step>")
                    skipped_lines += 1
                    continue
                
                # --- 检查 2: 拼接后的逻辑流 ---
                full_text = history + new_step
                full_text = full_text.replace("<|im_end|>", "").strip()
                
                res = validate_cot_logic(full_text)
                
                if res["valid"]:
                    # === 通过检查，写入新文件 ===
                    f_out.write(line)
                    kept_lines += 1
                else:
                    # === 未通过检查，跳过 ===
                    print(f"[Line {line_num}] [Skipped] [Logic Error] {res['error']}")
                    skipped_lines += 1
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print("\n" + "="*30)
    print(f"Filter Complete.")
    print(f"Total Lines Processed: {total_lines}")
    print(f"Lines Kept (Valid):    {kept_lines}")
    print(f"Lines Skipped (Error): {skipped_lines}")
    print(f"Valid data saved to:   {output_filename}")

if __name__ == "__main__":
    # 默认输入路径
    input_path = "/home/aiscuser/ds/zhaowang/rag/data/raw/mulsique_pvm.jsonl"
    
    # 输出路径 (默认在当前目录下生成 filtered.jsonl)
    output_path = "/home/aiscuser/ds/zhaowang/rag/data/mulsique_pvm_filtered.jsonl"
    
    # 如果命令行传了参数： python filter.py [input_file] [output_file]
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    filter_file(input_path, output_path)