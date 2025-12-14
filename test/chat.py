import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/aiscuser/ds/zhaowang/rag/save/sft"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

question = "Who is the spouse of the Green performer?"
prompt = f"<|im_start|>user\n{INSTRUCTION_TEMPLATE.format(question=question)}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n<step>\n"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

print("=== MODEL OUTPUT ===")
print(output_text)
