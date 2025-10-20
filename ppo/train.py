import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
from trainer import RAGPPOTrainer
from trl import PPOConfig
import os

RAW_DATA_PATH = "/home/aiscuser/ds/zhaowang/data/rag-dpo/raw/musique_ans_v1.0_train.jsonl"
DATA_PATH = "/home/aiscuser/ds/zhaowang/data/rag-dpo/raw/train_rl.json"
POLICY_MODEL_NAME = "/home/aiscuser/ds/zhaowang/data/rag-dpo/model/sft"
REWARD_MODEL_PATH = "/home/aiscuser/ds/zhaowang/data/rag-dpo/model/rm"
E5_MODEL_NAME = 'intfloat/e5-large-v2'

OUTPUT_DIR = "/home/aiscuser/ds/zhaowang/rag/ppo/model/ppo/v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["WANDB_PROJECT"] = "RAG-MCTS"

EPOCHS = 3.0
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

MAX_PPO_EPOCHS = 4
NUM_MINI_BATCHES = 1
LEARNING_RATE = 1.41e-5

def simple_data_collator(features):
    first = features[0]
    batch = {}
    for k in first:
        batch[k] = [f[k] for f in features]
    return batch

if __name__ == "__main__":
    print("Loading and preparing data...")
    raw_data_map = {}
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            raw_data_map[idx] = item
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data_indices = json.load(f)
    
    data = [raw_data_map[item['id']] for item in data_indices if item['id'] in raw_data_map]
    train_dataset = Dataset.from_list(data) 

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        POLICY_MODEL_NAME,
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    rm_tokenizer = AutoTokenizer.from_pretrained(
        REWARD_MODEL_PATH, 
        trust_remote_code=True,
        padding_side='left'
    )
    if rm_tokenizer.pad_token is None: 
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    
    value_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH, 
        dtype=torch.bfloat16,
        trust_remote_code=True,
        num_labels=1
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH, 
        dtype=torch.bfloat16,
        trust_remote_code=True,
        num_labels=1
    )

    policy_model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_NAME, 
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    print("Loading SentenceTransformer model for Retriever...")
    similarity_model = SentenceTransformer(E5_MODEL_NAME)

    config = PPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        num_mini_batches=NUM_MINI_BATCHES,
        num_ppo_epochs=MAX_PPO_EPOCHS,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        report_to="wandb",
        exp_name="ppo-v1",
    )

    trainer = RAGPPOTrainer(
        args=config, 
        model=policy_model, 
        processing_class=tokenizer,
        data_collator=simple_data_collator,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        reward_tokenizer=rm_tokenizer, 
        retrieval_model=similarity_model
    )

    print("Start Training...")
    trainer.train()

    print("Training finished. Saving...")
    trainer.save_model(OUTPUT_DIR)
