import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler 
from deepspeed.utils.tensor_fragment import fragment_address
import argparse
import wandb

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([ZeroStageEnum])
    torch.serialization.add_safe_globals([LossScaler])
    torch.serialization.add_safe_globals([fragment_address])

MAX_LENGTH = 4096

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        use_cache=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        input_ids_list, attn_masks, label_list = [], [], []
        
        for prompt_text, response_text in zip(examples["input"], examples["output"]):
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            response_ids = tokenizer.encode(response_text, add_special_tokens=False)
            response_ids.append(tokenizer.eos_token_id)

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

            if len(input_ids) > MAX_LENGTH:
                input_ids = input_ids[:MAX_LENGTH]
                labels = labels[:MAX_LENGTH]

            attention_mask = [1] * len(input_ids)

            input_ids_list.append(input_ids)
            attn_masks.append(attention_mask)
            label_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_masks,
            "labels": label_list,
        }

    print("Loading and preparing datasets...")
    full_dataset = load_dataset("json", data_files=args.train_data_path, split="train")
    print(f"Dataset columns: {full_dataset.column_names}")
    
    split_dataset = full_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")
    
    processed_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    processed_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    sample = processed_train_dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    print("\n" + "="*20 + " DEBUG MASKING " + "="*20)
    print(f"input:{input_ids}")
    print(f"label:{labels}")
    print("="*60 + "\n")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        deepspeed=args.deepspeed,
        eval_strategy="steps",            
        eval_steps=args.eval_steps,            
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print("Start Training... Best model will be saved based on validation loss.")
    trainer.train()

    print("\n--- Training complete. Saving the best model found during training... ---")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Best model has been saved to {args.output_dir}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the RFT model.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Base model name from Hugging Face.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    parser.add_argument("--bf16", action='store_true', help="Use bf16")
    parser.add_argument("--fp16", action='store_true', help="Use fp16")

    parser.add_argument("--deepspeed", type=str, required=True)

    args = parser.parse_args()
    main(args)