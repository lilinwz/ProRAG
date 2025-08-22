import json
import torch
from tqdm import tqdm
from typing import List, Dict
from datasets import Dataset
import evaluate
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

MCTS_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/rm/data.json"
MODEL_NAME = "Qwen/Qwen3-4B"
OUTPUT_DIR = "/home/v-zhaowan/zhaowang/rag/save/rm/v1"

NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
MAX_LENGTH = 2048
os.environ["WANDB_PROJECT"] = "RAG-MCTS"

def create_reward_samples(mcts_data_path: str) -> List[Dict]:
    print(f"Loading MCTS data from {mcts_data_path}...")
    with open(mcts_data_path, 'r', encoding='utf-8') as f:
        mcts_runs = json.load(f)

    samples = []
    
    def traverse_and_reconstruct(node: Dict, parent_state: str):
        action_value = node.get('action') or ""
        current_state = parent_state + action_value
        if node and node.get('n', 0) > 0 and len(action_value)>0:
            q = node.get('q', 0.0)
            n = node.get('n')
            target_value = q / n
            
            samples.append({
                "text": current_state,
                "label": target_value
            })

        if node and 'children' in node:
            for child in node['children']:
                traverse_and_reconstruct(child, current_state)

    print("Parsing MCTS trees and reconstructing states to create training samples...")
    for run in tqdm(mcts_runs, desc="Processing MCTS runs"):
        root_node = run.get("mcts_tree")
        question = run.get("question")
        if root_node and question:
            initial_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<step>\n"
            traverse_and_reconstruct(root_node, initial_prompt)
            
    print(f"Successfully created {len(samples)} training samples by reconstruction.")
    return samples

def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {
        "mse": mse,
        "mae": mae,
    }

if __name__ == "__main__":
    training_samples = create_reward_samples(MCTS_DATA_PATH)
    if not training_samples:
        raise ValueError("No training samples were created. Check your MCTS data file.")

    dataset = Dataset.from_list(training_samples)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['text'])
    tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['text'])

    print(f"Loading model {MODEL_NAME} for sequence classification (regression)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"rm-v1",
        fp16=False, 
        bf16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting model training...")
    trainer.train()

    print("Training finished. Saving the best model...")
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print(f"Model saved to {OUTPUT_DIR}/best_model")
    wandb.finish()