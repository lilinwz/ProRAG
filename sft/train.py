import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from trainer import CustomTrainer
import json
import os
import argparse

def main(args):
    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto"
    )

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

    # --- LoRA ---
    if args.adapter_path:
        print(f"Loading existing adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        print("No adapter path provided. Creating a new LoRA config.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=json.loads(args.target_modules),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if 'embed_tokens' in name or 'lm_head' in name:
            param.requires_grad = True
    print("Manually un-froze embedding and lm_head layers.")
    
    model.print_trainable_parameters()

    # --- data ---
    print(f"Loading and processing data from: {args.train_data_path}...")

    def load_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def preprocess_function(examples):
        formatted_texts = []
        labels_full = []
        labels_spe = []

        for item in examples["conversation"]:
            full_text = ""
            current_labels_full = []
            current_labels_spe = []

            user_prefix = f"<|im_start|>user\n"
            user_suffix = f"<|im_end|>\n"
            turn_text = user_prefix + item["user"] + user_suffix        
            encoded_turn = tokenizer.encode(turn_text, add_special_tokens=False)
            full_text += turn_text
            current_labels_full.extend([-100] * len(encoded_turn))
            current_labels_spe.extend([-100] * len(encoded_turn))

            assistant_prefix = f"<|im_start|>assistant\n"
            assistant_suffix = f"<|im_end|>\n"
            
            encoded_turn = tokenizer.encode(assistant_prefix, add_special_tokens=False)
            full_text += assistant_prefix
            current_labels_full.extend([-100] * len(encoded_turn))
            current_labels_spe.extend([-100] * len(encoded_turn))
            
            for i, content in enumerate(item["assistant"]):
                encoded_turn = tokenizer.encode(content, add_special_tokens=False)
                full_text += content

                if content.strip() in custom_special_tokens:
                    current_labels_spe.extend(encoded_turn)
                else:
                    current_labels_spe.extend([-100] * len(encoded_turn))
                current_labels_full.extend(encoded_turn)

            encoded_turn = tokenizer.encode(assistant_suffix, add_special_tokens=False)
            full_text += assistant_suffix
            current_labels_full.extend(encoded_turn)
            current_labels_spe.extend(encoded_turn)

            full_text += tokenizer.eos_token
            encoded_eos = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
            current_labels_spe.extend(encoded_eos)
            current_labels_full.extend(encoded_eos)
            
            formatted_texts.append(full_text)
            labels_full.append(current_labels_full[1:] + [-100])
            labels_spe.append(current_labels_spe[1:] + [-100])

        tokenized_inputs = tokenizer(
            formatted_texts,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length"
        )

        padded_labels_full = []
        padded_labels_spe = []
        for label_full_seq, label_spe_seq in zip(labels_full, labels_spe):
            if len(label_full_seq) > args.max_seq_length:
                print(f"Warning: Label sequence longer than MAX_SEQ_LENGTH ({args.max_seq_length}), truncating.")
            padded_full = label_full_seq[:args.max_seq_length] + [-100] * (args.max_seq_length - len(label_full_seq))
            padded_spe = label_spe_seq[:args.max_seq_length] + [-100] * (args.max_seq_length - len(label_spe_seq))
            padded_labels_full.append(torch.tensor(padded_full))
            padded_labels_spe.append(torch.tensor(padded_spe))

        tokenized_inputs["labels"] = torch.stack(padded_labels_full)
        tokenized_inputs["labels_for_eval"] = torch.stack(padded_labels_spe)

        return tokenized_inputs


    raw_data = load_data(args.train_data_path)# [:2000]
    full_dataset = Dataset.from_dict({"conversation": raw_data})
    split_dataset = full_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")

    processed_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["conversation"])
    processed_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["conversation"])

    # --- Trainer Config ---
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        eval_strategy="steps",            
        eval_steps=args.eval_steps,            
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=5,
        optim="adamw_torch", 
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        remove_unused_columns=False,
        label_names=["labels", "labels_for_eval"]
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        tokenizer=tokenizer,
        special_token_weight=args.special_token_weight
    )

    # --- train ---
    print("Starting training...")
    trainer.train()

    # --- save ---
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    print(f"Saving LoRA adapter to {final_adapter_path}...")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    print("Training complete! Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Base model name from Hugging Face.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to a pre-trained adapter to continue training.")
    
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]', help="JSON string of target modules for LoRA.")
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--special_token_weight", type=int, default=10)
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    parser.add_argument("--bf16", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--fp16", type=lambda x: (str(x).lower() == 'true'), default=False)
    
    args = parser.parse_args()
    main(args)