#!/bin/bash
set -e

export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API_KEY_HERE}"

pids=()
cleanup() {
    echo "🛑 Stopping vLLM servers..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Killed process $pid"
        fi
    done
    pids=()
    wait
    echo "✅ All servers stopped."
}

trap cleanup EXIT INT TERM

echo "🚀 Starting vLLM servers..."

for i in {0..3}; do
    port=$((8001 + i))
    echo "Starting server on GPU $i -> Port $port"
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
        --model saves/sft \
        --served-model-name "rag-model" \
        --port $port \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        > "server_${i}.log" 2>&1 &
    pids+=($!)
done

echo "⏳ Waiting 90s for servers to initialize..."
sleep 90

echo "🌲 Running MCTS Data Generation..."
python -m prorag.prm.mcts \
    --data_path data/raw/hotpotqa.jsonl,data/raw/musique.jsonl \
    --output_path data/raw/mcts_trees.jsonl

cleanup
trap - EXIT INT TERM

echo "🔍 Running PRM Filter (using OpenAI)..."

python -m prorag.prm.filter \
    --input_file data/raw/mcts_trees.jsonl \
    --output_file data/train_prm.jsonl \
    --model gpt-4o \
    --concurrency 10

echo "🏋️‍♂️ Training PRM Model..."
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

echo "🎉 All pipelines finished successfully!"