import os
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, TEXT, ID 
from datasets import load_dataset

INDEX_DIR = "/home/v-zhaowan/zhaowang/data/rag-dpo/retrieval"
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"

wiki_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
ix = open_dir(INDEX_DIR)

def search_whoosh_index(query_string, index_obj, default_field="content", limit=5):
    with index_obj.searcher() as searcher:
        parser = QueryParser(default_field, schema=searcher.schema)

        query = parser.parse(query_string)
        results = searcher.search(query, limit=limit)

        for i, hit in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"  **Title:** {hit.get('title')}")
            print(f"  **Path:** {hit.get('path')}")
            print(f"  **Score:** {hit.score:.4f}")
            retrieved_path = hit['path']
            print(f"文档路径 (Path/ID): {retrieved_path}")

            original_index = int(retrieved_path)
            original_record = wiki_dataset['train'][original_index]
            full_content = original_record['text']
            
            print("\n--- 获取到的原文 (显示前 500 个字符) ---")
            print(full_content[:500] + "...")
            print(len(full_content))
            print("---------------------------------\n")

print("begin retrieval")
search_whoosh_index("Artificial Intelligence", ix, default_field="content", limit=3)