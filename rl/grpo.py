import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
from trl import GRPOConfig
from trainer import RAGTrainer
import os

DATA_PATH = "/home/v-zhaowan/ds/zhaowang/rag/data/train_rl.jsonl"
MODEL_PATH = "/home/v-zhaowan/ds/zhaowang/rag/save/sft/checkpoint-770"
PRM_PATH = "/home/v-zhaowan/ds/zhaowang/rag/save/rm"
E5_MODEL_NAME = 'intfloat/e5-large-v2'
OUTPUT_DIR = "/home/v-zhaowan/ds/zhaowang/rag/save/rl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["WANDB_PROJECT"] = "ProRAG"

EPOCHS = 3.0
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1.4e-5

def load_train_dataset():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

if __name__ == "__main__":
    print("Loading data...")
    train_dataset = load_train_dataset()

    print("Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        dtype=torch.bfloat16, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2"
    )

    # prm_tokenizer = AutoTokenizer.from_pretrained(PRM_PATH, trust_remote_code=True, padding_side='left')
    # if rm_tokenizer.pad_token is None:
    #     rm_tokenizer.pad_token = rm_tokenizer.eos_token

    # prm = AutoModelForSequenceClassification.from_pretrained(
    #     REWARD_MODEL_PATH, 
    #     num_labels=1,
    #     dtype=torch.bfloat16
    # )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Loading retriever model (E5)...")
    similarity_model = SentenceTransformer(E5_MODEL_NAME)

    def accuracy_reward()

    def format_reward()

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name="test",
        num_generations=2,
        max_prompt_length=3072,
        max_completion_length=1024,
        log_completions=True,
    )

    print("Initializing trainer...")
    trainer = RAGTrainer(
        model=policy_model,
        args=config,
        reward_funcs=[format_reward, accuracy_reward],
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        retrieval_model=similarity_model,
        reward_model=reward_model,
        rm_tokenizer=rm_tokenizer,
    )

    print("Start Training...")
    trainer.train()

    print("Training finished. Saving...")
    trainer.save_model(OUTPUT_DIR)
