import json
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

MODEL_NAME = "Qwen/Qwen3-4B" 
OUTPUT_DIR = "/home/aiscuser/ds/zhaowang/rag/save/rm"
MAX_LENGTH = 4096
BATCH_SIZE = 8
LEARNING_RATE = 5e-6 

DATA_FILES = [
    "/home/aiscuser/ds/zhaowang/rag/data/hotpotqa_pvm_filtered.jsonl", 
    "/home/aiscuser/ds/zhaowang/rag/data/mulsique_pvm_filtered.jsonl"
]

def load_data(file_paths):
    all_data = []
    for file_path in file_paths:
        print(f"正在加载文件: {file_path} ...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                item = json.loads(line)
                user_input = item["input"]
                prompt_text = f"Question: {user_input['question']}\nHistory:\n{user_input['history']}\n"
                chosen_text = item["chosen"]["new_step"]
                rejected_text = item["rejected"]["new_step"]
                
                all_data.append({
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text
                })
            
    dataset = Dataset.from_list(all_data)
    return dataset.shuffle(seed=42)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        dtype=torch.float32
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    custom_special_tokens = [
        "<step>", "</step>",
        "<subquery>", "</subquery>",
        "<retrieval>", "</retrieval>",
        "<subanswer>", "</subanswer>",
        "<answer>", "</answer>"
    ]
    num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": custom_special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added_tokens} new special tokens to the tokenizer and resized model embeddings.")

    dataset = load_data(DATA_FILES)
    dataset_split = dataset.train_test_split(test_size=0.1)
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
        save_steps=40,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        deepspeed="/home/aiscuser/ds/zhaowang/rag/sft/ds.json",
        eval_strategy="steps",
        eval_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
    )

    print("开始全量微调...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()