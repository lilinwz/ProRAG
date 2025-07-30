import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

input_file = "/home/v-zhaowan/zhaowang/rag/data/raw/train2.json"
output_file = "/home/v-zhaowan/zhaowang/rag/data/raw/train3.json"

def generate_batch(prompts, model, tokenizer):
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]

    texts = [tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for messages in messages_batch]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    generated_ids_batch = []
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids_batch = generated_ids

    outputs = []
    for i in range(len(prompts)):
        input_len = len(model_inputs.input_ids[i])
        output_ids = generated_ids_batch[i][input_len:].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip().strip('"')
        outputs.append(content)
    return outputs


def process_batched(raw_data, model, tokenizer, batch_size=4):
    with open(output_file, 'r', encoding='utf-8') as f:
        processed_results = json.load(f)

    # 阶段 1: 收集所有 query_thought=1 的 prompt
    stage1_prompts_info = []
    for idx, item in enumerate(raw_data):
        query = item.get('query')
        chain_of_thought = item.get('chain_of_thought', [])
        answer = item.get('answer')
        item_id = item.get('id')

        if not query or not chain_of_thought or answer is None:
            print(f"Skipping line {item_id}: Missing 'query', 'chain_of_thought', or 'answer'.")
            continue

        # 为每个原始item初始化其cot1和cot2列表
        raw_data[idx]['_cot1_temp'] = []
        raw_data[idx]['_cot2_temp'] = []
        raw_data[idx]['_cot3_temp'] = ""
        raw_data[idx]['_context_info_temp'] = "\n"

        for i, step in enumerate(chain_of_thought):
            subquery = step.get('subquery')

            if subquery:
                prompt_1 = f"""
You are a smart AI assistant that is good at filling in the missing thought process in multi-step reasoning chains. You will be given an original question, and a sub-question that is asked to solve it. Your task is to generate a short text to explain why this particular sub-question is asked. If this is the first sub-question in the reasoning chain, explain how it is related to the original question. If it is a subsequent sub-question, explain how it is advanced based on the previous information.

---
**Sample Input 1:**
Original question: What year saw the creation of the region where the county of Hertfordshire is located?
Contextual information:
Current sub-question: In which region is Hertfordshire located?

**Sample Output 1:**
The main question asks about the creation year of the region where Hertfordshire is located. To answer this, I first need to identify which specific region Hertfordshire belongs to.

---
**Sample Input 2:**
Original question: What year saw the creation of the region where the county of Hertfordshire is located?
Contextual information: 
<subquery>In which region is Hertfordshire located? </subquery>
<subanswer>Hertfordshire is located in the East of England.</subanswer>
Current sub-question: When was the East of England region created?

**Sample Output 2:**
Now that I know Hertfordshire is in the East of England region, the next step is to find out when this specific region was created to answer the original question.

---
**Now, please generate a thought process for the following input:**
Original question: {query}
Context information: {raw_data[idx]['_context_info_temp']}
Current sub-question: {subquery}
"""
                stage1_prompts_info.append({
                    "prompt": prompt_1,
                    "original_item_idx": idx,
                    "subquery_idx_in_chain": i
                })

    # 执行阶段 1 的批处理
    print(f"--- 正在处理 {len(stage1_prompts_info)} 个 Query_Thought=1 的请求 ---")
    stage1_prompts_list = [info["prompt"] for info in stage1_prompts_info]
    stage1_results = []
    for i in range(0, len(stage1_prompts_list), batch_size):
        batch_prompts = stage1_prompts_list[i:i + batch_size]
        print(f"正在生成 Query_Thought=1 的批次 ({i+1}-{min(i+batch_size, len(stage1_prompts_list))}/{len(stage1_prompts_list)})...")
        generated_thoughts = generate_batch(batch_prompts, model, tokenizer)
        stage1_results.extend(generated_thoughts)

    # 将阶段 1 的结果回填到 raw_data 和更新上下文
    for i, thought in enumerate(stage1_results):
        info = stage1_prompts_info[i]
        original_item_idx = info["original_item_idx"]
        subquery_idx_in_chain = info["subquery_idx_in_chain"]

        raw_data[original_item_idx]['_cot1_temp'].append(thought)
        subquery_val = raw_data[original_item_idx]['chain_of_thought'][subquery_idx_in_chain]['subquery']
        subanswer_val = raw_data[original_item_idx]['chain_of_thought'][subquery_idx_in_chain]['sub-answer']
        raw_data[original_item_idx]['_context_info_temp'] += f"<subquery>{subquery_val}</subquery>\n<subanswer>{subanswer_val}</subanswer>\n"

    # 阶段 2: 收集所有 query_thought=2 的 prompt
    stage2_prompts_info = [] # 存储 (original_question, context_info, current_sub_question, original_item_idx)
    for idx, item in enumerate(raw_data):
        chain_of_thought = item.get('chain_of_thought', [])
        for i, step in enumerate(chain_of_thought):
            subquery = step.get('subquery')
            sub_answer = step.get('sub-answer')
            paragraph = step.get('paragraph')

            if sub_answer:
                info_text = f"<text>{paragraph}</text>"
                prompt_2 = f"""
You are an intelligent AI assistant that is good at filling in the missing thought process in a multi-step reasoning chain. You will be given a pair of question and answer with context. Your task is to generate a short text, explaining how to form the answer based on the context.

Please output your text directly.

---
**Sample Input 1:**
Question: Where is the county of Hertfordshire located?
Context: Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.
Answer: Hertfordshire is located in the East of England.
---
**Now, please generate your thought process for the following input:**
Question: {subquery}
Context: {info_text}
Answer:{sub_answer}
"""
                stage2_prompts_info.append({
                    "prompt": prompt_2,
                    "original_item_idx": idx
                })

    # 执行阶段 2 的批处理
    print(f"--- 正在处理 {len(stage2_prompts_info)} 个 Query_Thought=2 的请求 ---")
    stage2_prompts_list = [info["prompt"] for info in stage2_prompts_info]
    stage2_results = []
    for i in range(0, len(stage2_prompts_list), batch_size):
        batch_prompts = stage2_prompts_list[i:i + batch_size]
        print(f"正在生成 Query_Thought=2 的批次 ({i+1}-{min(i+batch_size, len(stage2_prompts_list))}/{len(stage2_prompts_list)})...")
        generated_thoughts = generate_batch(batch_prompts, model, tokenizer)
        stage2_results.extend(generated_thoughts)

    # 将阶段 2 的结果回填到 raw_data
    current_stage2_idx = 0
    for idx, item in enumerate(raw_data):
        chain_of_thought = item.get('chain_of_thought', [])
        for step in chain_of_thought:
            if step.get('sub-answer'): # 只有有 sub_answer 的才会有 cot2
                raw_data[idx]['_cot2_temp'].append(stage2_results[current_stage2_idx])
                current_stage2_idx += 1

    # 阶段 3: 收集所有 query_thought=3 的 prompt
    stage3_prompts_info = [] # 存储 (original_question, context_info, final_answer, original_item_idx)
    for idx, item in enumerate(raw_data):
        query = item.get('query')
        answer = item.get('answer')
        # 使用之前更新的 _context_info_temp 作为 context_info
        context_info_for_final = raw_data[idx]['_context_info_temp']
        
        prompt_3 = f"""
You are an intelligent AI assistant that is good at filling in the missing thought process in a multi-step reasoning chain. You will be given an original question and all the sub-answers obtained to solve it. Your task is to generate a short text, explaining how to form the final answer based on these obtained sub-answers.

Please output your text directly.

---
**Sample Input 1:**
Original Question: What year saw the creation of the region where the county of Hertfordshire is located?
Subqueries and Subanswers:
<subquery>In which region is Hertfordshire located? </subquery>
<subanswer>Hertfordshire is located in the East of England.</subanswer>
<subquery>When was the East of England region created? </subquery>
<subanswer>The East of England region was created in 1994.</subanswer>
Final Answer: 1994

---
**Now, please generate your thought process for the following input:**
Original Question: {query}
Subqueries and Subanswers: {context_info_for_final}
Final Answer: {answer}
"""
        stage3_prompts_info.append({
            "prompt": prompt_3,
            "original_item_idx": idx
        })

    print(f"--- 正在处理 {len(stage3_prompts_info)} 个 Query_Thought=3 的请求 ---")
    stage3_prompts_list = [info["prompt"] for info in stage3_prompts_info]
    stage3_results = []
    for i in range(0, len(stage3_prompts_list), batch_size):
        batch_prompts = stage3_prompts_list[i:i + batch_size]
        print(f"正在生成 Query_Thought=3 的批次 ({i+1}-{min(i+batch_size, len(stage3_prompts_list))}/{len(stage3_prompts_list)})...")
        generated_thoughts = generate_batch(batch_prompts, model, tokenizer)
        stage3_results.extend(generated_thoughts)

    for i, thought in enumerate(stage3_results):
        raw_data[i]['_cot3_temp'] = thought

    for idx, item in enumerate(raw_data):
        processed_results.append({
            "id": item.get('id'),
            "query": item.get('query'),
            "chain_of_thought": item.get('chain_of_thought'),
            "cot1": item['_cot1_temp'],
            "cot2": item['_cot2_temp'],
            "cot3": item['_cot3_temp'],
            "answer": item.get('answer')
        })

        if idx % (batch_size * 10) == 0 and idx > 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, ensure_ascii=False, indent=4)
                print(f"已保存 {len(processed_results)} 条数据到 {output_file}")

    return processed_results


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_name}' to device: {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=4)

    with open(output_file, 'r', encoding='utf-8') as f:
        a = json.load(f)
    
    START_INDEX = len(a)
    print(f"Loaded {len(raw_data)-START_INDEX} items from {input_file}.")
    BATCH_SIZE = 32 
    output = process_batched(raw_data[START_INDEX:], model, tokenizer, batch_size=BATCH_SIZE)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)