#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file configs/ds.yaml \
    prorag/rl/train.py \
    --model_path saves/rft \
    --prm_path saves/prm \
    --train_data_path data/train_rl.jsonl \
    --output_dir saves/rl \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --max_completion_length 4096 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --bf16 \
    --num_generations 8 \
    --prm_beta 0.3 \
    2>&1 | tee train.log

python -m prorag.rl.merge \
    --model_path saves/rft \
    --lora_path saves/rl \
    --output_path saves/model