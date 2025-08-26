import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
from trainer import RAGPPOTrainer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig
import os

RAW_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/MulSiQue/musique_ans_v1.0_train.jsonl"
DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/raw/train_rl.json"
POLICY_MODEL_NAME = "/home/v-zhaowan/zhaowang/data/rag-dpo/model/sft"
REWARD_MODEL_PATH = "/home/v-zhaowan/zhaowang/rag/save/rm/v1/best_model"
E5_MODEL_NAME = 'intfloat/e5-large-v2'

OUTPUT_DIR = "/home/v-zhaowan/zhaowang/rag/save/ppo/v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["WANDB_PROJECT"] = "RAG-MCTS"

MAX_PPO_EPOCHS = 4
BATCH_SIZE = 4
MINI_BATCH_SIZE = 1
LEARNING_RATE = 1.41e-5

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
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH, trust_remote_code=True)
    if rm_tokenizer.pad_token is None: 
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        POLICY_MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        peft_config=lora_config
    )
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        num_labels=1
    )
    reward_model.eval()

    print("Loading SentenceTransformer model for Retriever...")
    similarity_model = SentenceTransformer(E5_MODEL_NAME)

    config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        num_ppo_epochs=MAX_PPO_EPOCHS,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"ppo-test"
    )

    trainer = RAGPPOTrainer(
        args=config, 
        policy_model=policy_model, 
        reward_model=reward_model,
        train_dataset=train_dataset,
        tokenizer=tokenizer, 
        reward_tokenizer=rm_tokenizer, 
        retrieval_model=similarity_model
    )

    print("Start Training...")
    trainer.train()

    print("Training finished. Saving...")
    trainer.save_model(OUTPUT_DIR)
