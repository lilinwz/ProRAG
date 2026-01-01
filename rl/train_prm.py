import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

MODEL_NAME = "/home/aiscuser/ds/zhaowang/rag/save/sft"
OUTPUT_DIR = "/home/aiscuser/ds/zhaowang/rag/save/prm"
MAX_LENGTH = 4096
BATCH_SIZE = 8
LEARNING_RATE = 1e-5 
DATA_FILES = ["/home/aiscuser/ds/zhaowang/rag/data/train_prm_wiki.jsonl"]

SYSTEM_INSTRUCTION = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}"""

def build_full_prompt(question, history):
    user_content = SYSTEM_INSTRUCTION.format(question=question)
    prompt_head = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    full_prompt = prompt_head + history    
    return full_prompt

def load_data(file_paths, tokenizer):
    all_data = []
    for file_path in file_paths:
        print(f"正在加载文件: {file_path} ...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                item = json.loads(line)
                
                q = item["input"]["question"]
                h = item["input"]["history"]

                prompt_full = build_full_prompt(q, h)
                chosen_step = item["chosen"]["new_step"]
                rejected_step = item["rejected"]["new_step"]

                chosen_full = prompt_full + chosen_step + tokenizer.eos_token
                rejected_full = prompt_full + rejected_step + tokenizer.eos_token
                
                all_data.append({
                    "chosen": chosen_full,
                    "rejected": rejected_full
                })
            
    dataset = Dataset.from_list(all_data)
    return dataset.shuffle(seed=42)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    dataset = load_data(DATA_FILES, tokenizer)
    dataset_split = dataset.train_test_split(test_size=0.02)
    print(f"Training samples: {len(dataset_split['train'])}")

    training_args = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8, 
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        deepspeed="/home/aiscuser/ds/zhaowang/rag/sft/ds.json",
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
    )

    print("Start Training...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()