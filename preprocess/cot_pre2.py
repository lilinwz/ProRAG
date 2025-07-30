import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

input_file = "/home/v-zhaowan/zhaowang/rag/data/raw/train.json"
output_file = "/home/v-zhaowan/zhaowang/rag/data/raw/train1.json"

# input_file = "/home/v-zhaowan/zhaowang/rag/data/raw/dev.json"
# output_file = "/home/v-zhaowan/zhaowang/rag/data/raw/dev1.json"

def generate_thought_batch(prompts, model, tokenizer):
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for messages in messages_batch]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    while True:
        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

            outputs = []
            for i in range(len(prompts)):
                input_len = len(model_inputs.input_ids[i])
                output_ids = generated_ids[i][input_len:].tolist()
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                final = content
                outputs.append(final)
            return outputs
        except:
            pass

def process_batched(raw_data, model, tokenizer, batch_size=4):
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    current_batch_prompts = []
    current_batch_items = []

    for idx, item in enumerate(raw_data[3040:]):
        query = item.get('query')
        chain = item.get('chain_of_thought', [])
        answer = item.get('answer')
        item_id = item.get('id')

        if not query or not chain or answer is None:
            print(f"Skip line {item_id}: missing 'query', 'chain_of_thought' or 'answer'.")
            continue

        prompt = f"""
You are an intelligent AI assistant who is good at correcting problems in reasoning chains. You will receive a question-answer pair and the related thinking chain. Your task is to correct the wrong information in the thinking chain, especially the wrong information in each sub-question. For example, inexplicable aliases, wrong punctuation, and non-question, etc.
Please strictly output in the json format.

---
**Sample Input 1:**
"query": "What year saw the creation of the region where the county of Hertfordshire is located?"
"chain_of_thought": [
    {{
            "subquery": "Hertfordshire >> locate",
            "sub-answer": "East of England",
            "paragraph": "Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire."
        }},
        {{
            "subquery": "When was #1 birthed?",
            "sub-answer": "1994",
            "paragraph": "The East of England is one of nine official regions of England at the first level of NUTS for statistical purposes. It was created in 1994 and was adopted for statistics from 1999. It includes the ceremonial counties of Bedfordshire, Cambridgeshire, Essex, Hertfordshire, Norfolk and Suffolk. Essex has the highest population in the region."
        }}
    ]
"answer": "1994"

**Sample Output 1:**
{{    
    "new_CoT": [
        {{
            "subquery": "In which region is Hertfordshire located?",
            "sub-answer": "East of England",
            "paragraph": "Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire."
        }},
        {{
            "subquery": "When was The East of England birthed?",
            "sub-answer": "1994",
            "paragraph": "The East of England is one of nine official regions of England at the first level of NUTS for statistical purposes. It was created in 1994 and was adopted for statistics from 1999. It includes the ceremonial counties of Bedfordshire, Cambridgeshire, Essex, Hertfordshire, Norfolk and Suffolk. Essex has the highest population in the region."
        }}
    ]
}}

**Now, please generate a thought process for the following input:**
"query": {query}
"chain_of_thought": {chain}
"answer": {answer}
"""
        current_batch_prompts.append(prompt)
        current_batch_items.append({"id": item_id, "query": query, "answer": answer})

        if len(current_batch_prompts) == batch_size or idx == len(raw_data) - 1:
            print(f"Processing batch of {len(data)} items...")
            generated_chains = generate_thought_batch(
                current_batch_prompts,
                model,
                tokenizer,
            )

            for i, generated_chain in enumerate(generated_chains):
                item_to_update = current_batch_items[i]
                data.append({
                    "id": item_to_update["id"],
                    "query": item_to_update["query"],
                    "chain_of_thought": generated_chain,
                    "answer": item_to_update["answer"]
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            current_batch_prompts = []
            current_batch_items = []

    return data

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_name}' to device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    output = process_batched(raw_data, model, tokenizer, batch_size=32)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)