#!/bin/bash
set -e

python -m prorag.rft.filter_em \
    --model_path saves/sft \
    --data_path data/raw/hotpotqa.jsonl,data/raw/mulsique.jsonl \
    --output_path data/raw/rft.jsonl

python -m prorag.rft.filter_prm \
    --model_path saves/prm \
    --data_path data/raw/rft.jsonl \
    --output_path data/train_rft.jsonl

accelerate launch prorag/rft/train.py \
    --model_name saves/sft \
    --train_data_path data/train_rft.jsonl \
    --output_dir saves/rft \
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