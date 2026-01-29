#!/bin/bash
set -e

export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API_KEY_HERE}"

echo "[1/4] Downloading raw data..."
python -m prorag.data.download \
  --input bdsaglam/musique \
  --repo_filename musique_ans_v1.0_train.jsonl \
  --output_file data/raw/sft.jsonl \
  --dump_file data/raw/musique.jsonl

python -m prorag.data.download \
  --input hotpotqa/hotpot_qa \
  --repo_filename fullwiki/train-00000-of-00002.parquet,fullwiki/train-00001-of-00002.parquet \
  --dump_file data/raw/hotpotqa.jsonl

echo "[2/4] Cleaning data (Refinement)..."
python -m prorag.data.generate \
    --task clean \
    --input_file data/raw/sft.jsonl \
    --output_file data/raw/sft_cleaned.jsonl \
    --model gpt-4o \
    --concurrency 10

echo "[3/4] Generating CoT (Reasoning)..."
python -m prorag.data.generate \
    --task cot \
    --input_file data/raw/sft_cleaned.jsonl \
    --output_file data/raw/sft_gen.jsonl \
    --model gpt-4o \
    --concurrency 10

echo "[4/4] Constructing SFT dataset..."
python -m prorag.data.construct \
    --input_file data/raw/sft_gen.jsonl \
    --output_file data/train_sft.jsonl

echo "✅ Data preparation complete! Final dataset: data/train_sft.jsonl"