#!/bin/bash
set -e

pids=()
cleanup() {
    echo "Stopping vLLM servers..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
    done
    wait
    echo "All servers stopped."
}
trap cleanup EXIT INT TERM

echo "Starting vLLM servers..."

# GPU 0 -> Port 8000
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model saves/sft \
    --served-model-name "rag-model" \
    --port 8001 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    > server_0.log 2>&1 &
pids+=($!)

# # GPU 1 -> Port 8001
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model saves/sft \
    --served-model-name "rag-model" \
    --port 8002 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    > server_1.log 2>&1 &
pids+=($!)

# # GPU 2 -> Port 8002
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
    --model saves/sft \
    --served-model-name "rag-model" \
    --port 8003 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    > server_2.log 2>&1 &
pids+=($!)

# # GPU 3 -> Port 8003
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
    --model saves/sft \
    --served-model-name "rag-model" \
    --port 8004 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    > server_3.log 2>&1 &
pids+=($!)

echo "Waiting for servers to initialize ..."
sleep 90

echo "Running MCTS..."
python -m prorag.prm.mcts \
    --data_path data/raw/hotpotqa.jsonl,data/raw/mulsique.jsonl \
    --output_path data/raw/mcts_trees.jsonl

cleanup

echo "Running Filter..."
python -m prorag.prm.filter \
    --input_file data/raw/mcts_trees.jsonl \
    --output_file data/train_prm.jsonl \
    --concurrency 10

echo "Training PRM..."
accelerate launch prorag/prm/train.py \
    --model_name saves/sft \
    --train_data_path data/train_prm.jsonl \
    --output_dir saves/prm \
    --num_train_epochs 3 \
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