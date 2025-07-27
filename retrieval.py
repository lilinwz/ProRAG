import os
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, TEXT, ID 

INDEX_DIR = "/home/v-zhaowan/zhaowang/rag/wiki"

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

search_whoosh_index("Artificial Intelligence", ix, default_field="content", limit=3)