#!/bin/bash
set -e

# ================ Config ================
RAW_FILE="data/raw/sft.jsonl"
CLEAN_FILE="data/raw/sft_cleaned.jsonl"
GEN_FILE="data/raw/sft_gen.jsonl"
FINAL_FILE="data/train_sft.jsonl"

CONCURRENCY=10
# ========================================

echo "[1/4] Downloading raw data..."
python -m prorag.data.download \
  --input bdsaglam/musique \
  --repo_filename musique_ans_v1.0_train.jsonl \
  --output_file "$RAW_FILE" \
  --dump_file data/raw/mulsique.jsonl

python -m prorag.data.download \
  --input hotpotqa/hotpot_qa \
  --repo_filename fullwiki/train-00000-of-00002.parquet,fullwiki/train-00001-of-00002.parquet \
  --dump_file data/raw/hotpotqa.jsonl

echo "[2/4] Cleaning data (Refinement)..."
python -m prorag.data.generate \
    --task clean \
    --input_file "$RAW_FILE" \
    --output_file "$CLEAN_FILE" \
    --concurrency "$CONCURRENCY"

echo "[3/4] Generating CoT (Reasoning)..."
python -m prorag.data.generate \
    --task cot \
    --input_file "$CLEAN_FILE" \
    --output_file "$GEN_FILE" \
    --concurrency "$CONCURRENCY"

echo "[4/4] Constructing SFT dataset..."
python -m prorag.data.construct \
    --input_file "$GEN_FILE" \
    --output_file "$FINAL_FILE"

echo "✅ Data preparation complete! Final dataset: $FINAL_FILE"