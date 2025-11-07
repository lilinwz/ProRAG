MODEL_NAME="Qwen/Qwen3-8B"

NUM_TRAIN_EPOCHS=5
LEARNING_RATE=2e-5

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
MAX_SEQ_LENGTH=4096
SAVE_STEPS=100
EVAL_STEPS=50
LOGGING_STEPS=10

TRAIN_DATA_PATH="/home/aiscuser/ds/zhaowang/rag/data/train_sft.jsonl"
OUTPUT_DIR="/home/aiscuser/ds/zhaowang/rag/save/sft"

accelerate launch train.py \
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
    --bf16 True \
    2>&1 | tee train.log