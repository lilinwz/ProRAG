import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
from trainer import RAGTrainer
from trl import GRPOConfig
import os

RAW_DATA_PATH = "/home/aiscuser/ds/zhaowang/data/rag-dpo/raw/musique_ans_v1.0_train.jsonl"
DATA_PATH = "/home/aiscuser/ds/zhaowang/data/rag-dpo/raw/train_rl.json"
POLICY_MODEL_NAME = "/home/aiscuser/ds/zhaowang/data/rag-dpo/model/sft"
REWARD_MODEL_PATH = "/home/aiscuser/ds/zhaowang/data/rag-dpo/model/rm"
E5_MODEL_NAME = 'intfloat/e5-large-v2'

OUTPUT_DIR = "/home/aiscuser/ds/zhaowang/rag/grpo/model/v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["WANDB_PROJECT"] = "RAG-MCTS"

EPOCHS = 3.0
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE = 1.41e-5

def load_train_dataset():
    raw_data_map = {}
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            raw_data_map[idx] = json.loads(line)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data_indices = json.load(f)

    data = [raw_data_map[item['id']] for item in data_indices if item['id'] in raw_data_map]
    dataset = Dataset.from_list(data)
    return dataset

if __name__ == "__main__":
    print("Loading data...")
    train_dataset = load_train_dataset()

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

    def step_reward(completions, **kwargs):
        inputs = rm_tokenizer(
            completions,
            return_tensors='pt', 
            padding=True, 
            max_length=4096,
            truncation=True
        ).to(reward_model.device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits
            rewards = logits.squeeze(-1)
        return rewards.cpu().tolist()

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        report_to="wandb",
        run_name="grpo-test",
        num_generations=2,
        max_prompt_length=3072,
        max_completion_length=1024,
        log_completions=True,
    )

    trainer = RAGTrainer(
        model=policy_model,
        args=config,
        reward_funcs=[step_reward],
        train_dataset=train_dataset,
        eval_dataset=None,
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
