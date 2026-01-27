python -m prorag.rft.filter_em \
    --model_path save/sft \
    --data_path data/raw/hotpotqa.jsonl,data/raw/mulsique.jsonl \
    --output_path data/raw/rft.jsonl

python -m prorag.rft.filter_em \
    --model_path save/prm \
    --data_path data/raw/rft.jsonl \
    --output_path data/train_rft.jsonl


MODEL_NAME="save/sft"

NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-5

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
MAX_SEQ_LENGTH=4096
SAVE_STEPS=100
EVAL_STEPS=50
LOGGING_STEPS=10

TRAIN_DATA_PATH="data/train_rft.jsonl"
DEEPSPEED_CONFIG="configs/ds.json"
OUTPUT_DIR="saves/rft"

accelerate launch prorag/rft/train.py \
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