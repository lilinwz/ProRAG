#!/bin/bash
set -e

SAVE_PATH="${RETRIEVAL_PATH:-data/wikipedia}"
mkdir -p "$SAVE_PATH"

python search/download.py --save_path $SAVE_PATH
cat $SAVE_PATH/part_* > $SAVE_PATH/e5_Flat.index
gzip -d $SAVE_PATH/wiki-18.jsonl.gz