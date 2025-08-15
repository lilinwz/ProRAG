#!/bin/bash

MODEL_NAME="Qwen/Qwen3-8B"

LORA_R=128
LORA_ALPHA=128
LORA_DROPOUT=0.05

NUM_TRAIN_EPOCHS=2
LEARNING_RATE=5e-5
SPECIAL_TOKEN_WEIGHT=10

PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
MAX_SEQ_LENGTH=2048
SAVE_STEPS=30
EVAL_STEPS=10
LOGGING_STEPS=10

for i in {2..8}
do
    PREVIOUS_COURSE=$((i-1))
    TRAIN_DATA_PATH="/home/v-zhaowan/zhaowang/rag/data/train_sft_course${i}.json"
    OUTPUT_DIR="/home/v-zhaowan/zhaowang/rag/save/course${i}"
    ADAPTER_PATH="/home/v-zhaowan/zhaowang/rag/save/course${PREVIOUS_COURSE}/final_adapter"

    echo "================================================="
    echo "          STARTING TRAINING RUN FOR COURSE $i"
    echo "================================================="
    echo "Train Data: $TRAIN_DATA_PATH"
    echo "Output Dir: $OUTPUT_DIR"
    echo "Adapter Path: $ADAPTER_PATH"
    echo "-------------------------------------------------"

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
done

TRAIN_DATA_PATH="/home/v-zhaowan/zhaowang/rag/data/train_sft.json"
OUTPUT_DIR="/home/v-zhaowan/zhaowang/rag/save/final"
ADAPTER_PATH="/home/v-zhaowan/zhaowang/rag/save/course8/final_adapter"
NUM_TRAIN_EPOCHS=3


echo "================================================="
echo "          STARTING TRAINING RUN FOR FINAL TRAIN"
echo "================================================="
echo "Train Data: $TRAIN_DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Adapter Path: $ADAPTER_PATH"
echo "-------------------------------------------------"

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