import os
import faiss
from sentence_transformers import SentenceTransformer
import pickle

FAISS_INDEX_PATH = "/home/v-zhaowan/zhaowang/rag/wiki_full_faiss.index"
CHUNKS_PATH = "/home/v-zhaowan/zhaowang/rag/wiki_full_chunks.pkl"

def search_wiki_with_e5(query, k=5):
    print(f"Searching: '{query}' ...")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks_data = pickle.load(f)
        
    query_vector = model.encode([f"query: {query}"])
    distances, indices = index.search(query_vector, k)

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        retrieved_chunk = chunks_data[idx]
        print(f"结果 {i+1} (相似度得分/距离: {distances[0][i]:.4f})")
        print(f"  来源文章:《{retrieved_chunk['title']}》")
        print(f"  相关段落: \"...{retrieved_chunk['original_text']}...\"")
        print(f"  URL: {retrieved_chunk['url']}")
        print("-" * 30)