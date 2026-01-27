#!/bin/bash
set -e

accelerate launch prorag/sft/train.py \
    --model_name Qwen/Qwen3-8B \
    --train_data_path data/train_sft.jsonl \
    --output_dir saves/sft \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 4096 \
    --save_steps 100 \
    --eval_steps 50 \
    --logging_steps 10 \
    --bf16 \
    --deepspeed configs/ds.json \
    2>&1 | tee train.log