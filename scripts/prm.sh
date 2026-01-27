MODEL_PATH="saves/sft"

GPU_UTIL=0.85
echo "Starting vLLM servers..."

# GPU 0 -> Port 8000
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "rag-model" \
    --port 8000 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    > server_0.log 2>&1 &

# # GPU 1 -> Port 8001
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "rag-model" \
    --port 8001 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    > server_1.log 2>&1 &

# # GPU 2 -> Port 8002
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "rag-model" \
    --port 8002 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    > server_2.log 2>&1 &

# # GPU 3 -> Port 8003
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "rag-model" \
    --port 8003 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    > server_2.log 2>&1 &

python -m prorag.prm.mcts \
    --data_path data/raw/hotpotqa.jsonl,data/raw/mulsique.jsonl \
    --output_path save/raw/mcts_trees.jsonl

python -m prorag.prm.filter \
    --input_file save/raw/mcts_trees.jsonl \
    --output_file save/train_prm.jsonl \
    --concurrency 10



MODEL_NAME="saves/sft"

NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-5

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
MAX_SEQ_LENGTH=4096
SAVE_STEPS=100
EVAL_STEPS=50
LOGGING_STEPS=10

TRAIN_DATA_PATH="data/train_prm.jsonl"
DEEPSPEED_CONFIG="configs/ds.json"
OUTPUT_DIR="saves/prm"

accelerate launch prorag/prm/train.py \
    --model_name $MODEL_NAME \
    --train_data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    2>&1 | tee train.log