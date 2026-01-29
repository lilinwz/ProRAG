#!/bin/bash
set -e

FILE_PATH="${RETRIEVAL_PATH:-data/wikipedia}"
index_file=$FILE_PATH/e5_Flat.index
corpus_file=$FILE_PATH/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search/server.py --index_path $index_file \
                --corpus_path $corpus_file \
                --topk 3 \
                --retriever_name $retriever_name \
                --retriever_model $retriever_path \
                --faiss_gpu