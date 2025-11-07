import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from trainer import CustomTrainer
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler 
from deepspeed.utils.tensor_fragment import fragment_address
import json
import os
import argparse
import re

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        ZeroStageEnum,
        LossScaler,
        fragment_address
    ])

def main(args):
    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32
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

    # --- data ---
    print(f"Loading and processing data from: {args.train_data_path}...")

    def load_data(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data.append(json.loads(line))
        return data

    INSTRUCTION_TEMPLATE = """You are an assistant tasked with answering user questions by following a step-by-step reasoning process. Structure your entire response using the following special tokens and rules:
- `<step>...</step>`: Use this to explain the logical reasoning for each step in your process. Each step should bring you closer to solving the user's query.
- `<subquery>...</subquery>`: This block contains a specific question or sub-question that needs to be answered in order to progress. This is part of your reasoning, so make sure the subquery is clear and answerable.
- `<retrieval>...</retrieval>`: This block contains information retrieved from external sources (such as a search engine) that help answer the subquery. It can contain factual data or direct quotes.
- `<subanswer>...</subanswer>`: This block contains the answer to the preceding subquery. It's the most direct, concise answer that results from the retrieval.
- `<answer>...</answer>`: This is the final, conclusive answer to the user's main question, derived by combining the steps and subanswers.

Now, use this structure to answer the following user question:

User Question: {question}
"""

    def preprocess_function(examples):
        input_ids_list, attn_masks, label_list = [], [], []

        for q, r in zip(examples["question"], examples["response"]):
            user_content = INSTRUCTION_TEMPLATE.format(question=q)
            user_block = f"<|im_start|>user\n{user_content}<|im_end|>\n"
            assistant_block = f"<|im_start|>assistant\n<think>\n</think>\n{r}<|im_end|>\n"
            full_text = user_block + assistant_block
            
            retrieval_spans = [(m.start(), m.end()) for m in re.finditer(r"<retrieval>.*?</retrieval>", full_text, re.DOTALL)]
            content_start = full_text.find("<|im_start|>assistant\n<think>\n</think>\n") + len("<|im_start|>assistant\n<think>\n</think>\n")

            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
                return_offsets_mapping=True
            )

            ids = tokenized["input_ids"]
            mask = tokenized["attention_mask"]
            offsets = tokenized["offset_mapping"]

            labels = ids.copy()
            for j, (start, end) in enumerate(offsets):
                if end == 0 and start == 0:
                    continue
                if end <= content_start:
                    labels[j] = -100
                    continue
                for (r_start, r_end) in retrieval_spans:
                    if start > r_start and end < r_end:
                        labels[j] = -100
                        break

            input_ids_list.append(torch.tensor(ids))
            attn_masks.append(torch.tensor(mask))
            label_list.append(torch.tensor(labels))

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_masks,
            "labels": label_list,
        }

    raw_data = load_data(args.train_data_path)
    full_dataset = Dataset.from_dict({
        "id": [d["id"] for d in raw_data],
        "question": [d["question"] for d in raw_data],
        "response": [d["response"] for d in raw_data],
    })
    split_dataset = full_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")

    processed_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["id", "question", "response"])
    processed_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["id", "question", "response"])

    # example
    original_sample = train_dataset[0]
    q, r = original_sample["question"], original_sample["response"]
    original_full_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>\n{tokenizer.eos_token}"

    processed_sample = processed_train_dataset[0]
    input_ids = processed_sample['input_ids']
    labels = processed_sample['labels']
    
    print("\n[Original Full Text]")
    print(original_full_text)
    print("\n[Input IDs Tensor]")
    print(input_ids)
    print(f"\n[Labels Tensor]")
    print(labels)    

    # --- Trainer Config ---
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=8,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        deepspeed="/home/aiscuser/ds/zhaowang/rag/sft/ds.json",
        eval_strategy="steps",            
        eval_steps=args.eval_steps,            
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        tokenizer=tokenizer
    )

    # --- train ---
    print("Starting training...")
    trainer.train()

    # --- save ---
    save_path = args.output_dir
    print(f"Saving Model to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    print("Training complete! Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Base model name from Hugging Face.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
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