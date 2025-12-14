"""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m accelerate.commands.launch \
    --config_file /home/aiscuser/ds/zhaowang/rag/rl/ds.yaml \
    train.py 2>&1 | tee train.log
"""
import torch
import json
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
from trl import GRPOConfig
from trainer import RAGTrainer
import os

# --- 配置路径 ---
DATA_PATH = "/home/aiscuser/ds/zhaowang/rag/data/train_rl_tmp.jsonl"
MODEL_PATH = "/home/aiscuser/ds/zhaowang/rag/save/sft"
PRM_PATH = "/home/aiscuser/ds/zhaowang/rag/save/prm"
E5_MODEL_NAME = 'intfloat/e5-large-v2'
OUTPUT_DIR = "/home/aiscuser/ds/zhaowang/rag/save/rl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["WANDB_PROJECT"] = "ProRAG"

EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5 
NUM_GENERATIONS = 4
BETA_PRM = 0.5
MAX_PROMPT_LENGTH = 4096
MAX_COMPLETION_LENGTH = 1024

def load_dataset_splits(test_size=100):
    data_list = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    
    full_dataset = Dataset.from_list(data_list)
    full_dataset = full_dataset.shuffle(seed=42) 
    dataset_dict = full_dataset.train_test_split(test_size=100, seed=42)   
    return dataset_dict['train'], dataset_dict['test']

if __name__ == "__main__":
    print("Loading data...")
    train_dataset, eval_dataset = load_dataset_splits()

    print("Loading Policy Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        dtype=torch.bfloat16, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2"
    )

    print("Loading PRM Model & Tokenizer...")
    prm_tokenizer = AutoTokenizer.from_pretrained(PRM_PATH, trust_remote_code=True, padding_side='left')
    if prm_tokenizer.pad_token is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token

    prm_model = AutoModelForSequenceClassification.from_pretrained(
        PRM_PATH, 
        num_labels=1,
        dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Loading Retriever (E5)...")
    similarity_model = SentenceTransformer(E5_MODEL_NAME)

    # 展位奖励
    def dummy_reward(completions, **kwargs):
        return [0.0] * len(completions)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",     
        eval_steps=100,         
        per_device_eval_batch_size=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, 
        report_to="wandb",
        run_name="prorag",
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        log_completions=True,
        remove_unused_columns=False,
    )

    print("Initializing Trainer...")
    trainer = RAGTrainer(
        model=model,
        args=config,
        reward_funcs=[dummy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        retrieval_model=similarity_model,
        reward_model=prm_model,
        rm_tokenizer=prm_tokenizer,
        prm_beta=BETA_PRM
    )

    print("Start Training...")
    trainer.train()

    print("Training finished. Saving...")
    trainer.save_model(OUTPUT_DIR)