MODEL_PATH="/home/aiscuser/ds/zhaowang/rag/save/sft"

GPU_UTIL=0.85
echo "Starting vLLM servers..."

# GPU 0 -> Port 8000
# CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
#     --model $MODEL_PATH \
#     --served-model-name "rag-model" \
#     --port 8000 \
#     --gpu-memory-utilization $GPU_UTIL \
#     --max-model-len 8192 \
#     > server_0.log 2>&1 &

# # GPU 1 -> Port 8001
# CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
#     --model $MODEL_PATH \
#     --served-model-name "rag-model" \
#     --port 8001 \
#     --gpu-memory-utilization $GPU_UTIL \
#     --max-model-len 8192 \
#     > server_1.log 2>&1 &

# # GPU 2 -> Port 8002
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "rag-model" \
    --port 8002 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    > server_2.log 2>&1 &

# # GPU 3 -> Port 8003
# CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
#     --model $MODEL_PATH \
#     --served-model-name "rag-model" \
#     --port 8003 \
#     --gpu-memory-utilization $GPU_UTIL \
#     --max-