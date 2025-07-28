import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

def generate_thought(original_question, current_sub_question, context_info, model, tokenizer, query_thought):
    if query_thought == 1:
        # Prompt for sub-query thought
        prompt = f"""
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
Original question: {original_question}
Context information: {context_info}
Current sub-question: {current_sub_question}
"""
    elif query_thought == 2:
        # Prompt for final answer thought
        prompt = f"""
You are an intelligent AI assistant that is good at filling in the missing thought process in a multi-step reasoning chain. You will be given a pair of question and answer with context. Your task is to generate a short text, explaining how to form the answer based on the context.

Please output your text directly.

---
**Sample Input 1:**
Question: Where is the county of Hertfordshire located?
Context: Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.
Answer: Hertfordshire is located in the East of England.

**Sample Output 1:**
The question asks for the location of Hertfordshire. The provided context directly states that Hertfordshire is part of the East of England region. I can use this information to directly answer the question.
---
**Now, please generate your thought process for the following input:**
Question: {original_question}
Context: {context_info}
Answer:{current_sub_question}
"""
    elif query_thought == 3:
        # Prompt for final answer thought
        prompt = f"""
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

**Sample Output 1:**
Based on the sub-answers, I have identified that Hertfordshire is in the East of England region, and that region was created in 1994. I can now form the final answer by directly stating the creation year.

---
**Now, please generate your thought process for the following input:**
Original Question: {original_question}
Subqueries and Subanswers: {context_info}
Final Answer: {current_sub_question}
"""

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids =  model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return content
    

def process(raw_data, model, tokenizer):
    data = []
    for idx, item in enumerate(raw_data):
        query = item.get('query')
        chain_of_thought = item.get('chain_of_thought', [])
        answer = item.get('answer')
        item_id = item.get('id')

        if not query or not chain_of_thought or answer is None:
            print(f"Skipping line {item_id}: Missing 'query', 'chain_of_thought', or 'answer'.")
            continue

        cot1 = []
        cot2 = []
        context_info = "\n"
        print(f"Processing: {idx}...")

        for i, step in enumerate(chain_of_thought):
            subquery = step.get('subquery')
            sub_answer = step.get('sub-answer')
            paragraph = step.get('paragraph')

            print(f"Subquery: {subquery}, Sub-answer: {sub_answer}, Paragraph: {paragraph}")

            if subquery:                
                current_thought = generate_thought(
                    query,
                    subquery,
                    context_info,
                    model,
                    tokenizer,
                    query_thought=1
                )
                cot1.append(current_thought)

                context_info += f"<subquery>{subquery}</subquery>\n<subanswer>{subquery}</subanswer>"
            
            if sub_answer:
                info = f"<text>{paragraph}</text>"
                current_thought = generate_thought(
                    subquery,
                    sub_answer,
                    info,
                    model,
                    tokenizer,
                    query_thought=2
                )
                cot2.append(current_thought)

        
        current_thought = generate_thought(
            query,
            answer,
            context_info,
            model,
            tokenizer,
            query_thought=3
        )
        cot1.append(current_thought)

        data.append({
            "id": item_id,
            "query": query,
            "chain_of_thought": chain_of_thought,
            "cot1": cot1,
            "cot2": cot2,
            "answer": answer
        })
        break
    return data

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

    input_file = "/home/v-zhaowan/zhaowang/rag/data/raw/train.json"
    output_file = "/home/v-zhaowan/zhaowang/rag/data/raw/train1.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    output = process(raw_data, model, tokenizer)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)