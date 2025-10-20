python -m accelerate.commands.launch --use_deepspeed \
    --num_processes=4 \
    --deepspeed_config_file "/home/aiscuser/ds/zhaowang/rag/ppo/deepspeed_config.json" \
    train.py 2>&1 | tee train.log