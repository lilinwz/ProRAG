import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from prorag.utils.prompts import build_user_prompt
import argparse

def build_full_prompt(question, history):
    user_content = build_user_prompt(question)
    prompt_head = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    full_prompt = prompt_head + history    
    return full_prompt

def load_data(file_path, tokenizer):
    print(f"Loading file: {file_path} ...")
    all_data = []
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

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    dataset = load_data(args.train_data_path, tokenizer)
    dataset_split = dataset.train_test_split(test_size=0.02)
    print(f"Training samples: {len(dataset_split['train'])}")

    training_args = RewardConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        deepspeed=args.deepspeed,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
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

    print(f"Saving Model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete! Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Process Reward model.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Base model name from Hugging Face.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    parser.add_argument("--bf16", action='store_true', help="Use bf16")
    parser.add_argument("--fp16", action='store_true', help="Use fp16")

    parser.add_argument("--deepspeed", type=str, required=True)

    args = parser.parse_args()
    main(args)