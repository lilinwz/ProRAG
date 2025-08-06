#!/bin/bash

MODEL_NAME="Qwen/Qwen3-8B"
TRAIN_DATA_PATH="/home/v-zhaowan/zhaowang/rag/data/train_sft_course8.json"
OUTPUT_DIR="/home/v-zhaowan/zhaowang/rag/save/course8"
ADAPTER_PATH="/home/v-zhaowan/zhaowang/rag/save/course7/final_adapter"

LORA_R=128
LORA_ALPHA=128
LORA_DROPOUT=0.05

NUM_TRAIN_EPOCHS=2
LEARNING_RATE=5e-5
SPECIAL_TOKEN_WEIGHT=10

PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
MAX_SEQ_LENGTH=2048
SAVE_STEPS=50
EVAL_STEPS=10
LOGGING_STEPS=10

python train.py \
    --model_name $MODEL_NAME \
    --train_data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --adapter_path $ADAPTER_PATH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --special_token_weight $SPECIAL_TOKEN_WEIGHT \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS