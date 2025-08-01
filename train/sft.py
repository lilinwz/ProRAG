import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trainer import CustomTrainer
from data_collator import CustomDataCollator
import json
import os

# --- config ---
MODEL_NAME = "Qwen/Qwen3-8B"
TRAIN_DATA_PATH = "/home/v-zhaowan/zhaowang/rag/data/train_sft.json"
OUTPUT_DIR = "/home/v-zhaowan/zhaowang/rag/save/81"

LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 8 
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5         
MAX_SEQ_LENGTH = 2048        
SAVE_STEPS = 100             
LOGGING_STEPS = 10          
BF16 = True
FP16 = False

# --- load model and tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32),
    device_map="auto"
)

custom_special_tokens = [
    "<think>", "</think>",
    "<subquery>", "</subquery>",
    "<retrieval>", "</retrieval>",
    "<subanswer>", "</subanswer>",
    "<answer>", "</answer>"
]

num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": custom_special_tokens})
model.resize_token_embeddings(len(tokenizer))
print(f"Added {num_added_tokens} new special tokens to the tokenizer and resized model embeddings.")

# --- LoRA ---
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- data ---
print(f"Loading and processing data from: {TRAIN_DATA_PATH}...")

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

            if content[:-1] in custom_special_tokens:
                current_labels_spe.extend(encoded_turn)
            else:
                current_labels_spe.extend([-100] * len(encoded_turn))

            if (i>0 and item["assistant"][i-1] == "<retrieval>\n") or content == "<retrieval>\n":
                current_labels_full.extend([-100] * len(encoded_turn))
            else:
                current_labels_full.extend(encoded_turn)

        encoded_turn = tokenizer.encode(assistant_suffix, add_special_tokens=False)
        full_text += assistant_suffix
        current_labels_full.extend([-100] * len(encoded_turn))
        current_labels_spe.extend([-100] * len(encoded_turn))

        full_text += tokenizer.eos_token
        encoded_eos = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        current_labels_spe.extend([-100] * len(encoded_eos))
        current_labels_full.extend([-100] * len(encoded_eos))
        
        formatted_texts.append(full_text)
        labels_full.append(current_labels_full)
        labels_spe.append(current_labels_spe)


    tokenized_inputs = tokenizer(
        formatted_texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length"
    )

    padded_labels_full = []
    padded_labels_spe = []
    for label_full_seq, label_spe_seq in zip(labels_full, labels_spe):
        if len(label_full_seq) > MAX_SEQ_LENGTH:
            padded_labels_full.append(label_full_seq[:MAX_SEQ_LENGTH])
        else:
            padded_labels_full.append(label_full_seq + [-100] * (MAX_SEQ_LENGTH - len(label_full_seq)))

        if len(label_spe_seq) > MAX_SEQ_LENGTH:
            padded_labels_spe.append(label_spe_seq[:MAX_SEQ_LENGTH])
        else:
            padded_labels_spe.append(label_spe_seq + [-100] * (MAX_SEQ_LENGTH - len(label_spe_seq)))

    tokenized_inputs["labels_full"] = padded_labels_full
    tokenized_inputs["labels_special"] = padded_labels_spe

    return tokenized_inputs


raw_data = load_data(TRAIN_DATA_PATH)
dataset = Dataset.from_dict({"conversation": raw_data})
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=os.cpu_count() if os.cpu_count() else 1,
    remove_columns=["conversation"],
)

print(processed_dataset[0].keys())  # Debugging line to check the first processed sample

# print(f"Processed dataset size: {len(processed_dataset)} samples.")
# print("Sample input:", tokenizer.decode(processed_dataset[0]["input_ids"], skip_special_tokens=False))
# print(f"Sample input IDs: {processed_dataset[0]['input_ids']}")
# print(f"Sample input IDs: {processed_dataset[0]['labels']}")

# --- Trainer Config ---
print("Setting up Trainer...")
custom_collator = CustomDataCollator()
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    bf16=BF16,
    gradient_checkpointing=True, 
    gradient_checkpointing_kwargs={'use_reentrant': False}, 
    load_best_model_at_end=False, 
    # metric_for_best_model="loss",
    # greater_is_better=False,
    optim="adamw_torch", 
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    remove_unused_columns=False  # <--- 添加这一行
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    data_collator=custom_collator,
    special_token_weight=2.0
)

# --- train ---
print("Starting training...")
trainer.train()

# --- save ---
print(f"Saving LoRA adapter to {OUTPUT_DIR}/final_adapter...")
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

print("Training complete! Model saved.")


# --- load lora ---
# 如果你想将LoRA权重合并到原始模型中以便于推理部署，可以这样做：
# from peft import PeftModel
# print("Loading base model for merging...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.bfloat16, # 使用与训练时相同的精度
#     device_map="auto"
# )
# print("Merging LoRA adapter...")
# merged_model = PeftModel.from_pretrained(base_model, os.path.join(OUTPUT_DIR, "final_adapter"))
# merged_model = merged_model.merge_and_unload()
# print(f"Saving merged model to {OUTPUT_DIR}/merged_model...")
# # 保存完整的模型和分词器
# merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "merged_model"))
# tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "merged_model"))
# print("Merged model saved!")