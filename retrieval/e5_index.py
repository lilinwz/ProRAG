import os
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# ==============================================================================
# 1. 配置参数
# ==============================================================================
FAISS_INDEX_PATH = "/home/v-zhaowan/zhaowang/rag/wiki_full_faiss_sorted.index"
CHUNKS_PATH = "/home/v-zhaowan/zhaowang/rag/wiki_full_chunks_sorted.pkl"

MODEL_NAME = 'intfloat/e5-large-v2'
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"

# 文本块的设置
CHUNK_SIZE = 384    
CHUNK_OVERLAP = 64   

# 编码时使用的批次大小，A100可以轻松应对更大的批次
ENCODE_BATCH_SIZE = 2048

# ==============================================================================
# 2. 初始化模型和文本分割器
# ==============================================================================
print("Initializing model and text splitter...")
model = SentenceTransformer(MODEL_NAME)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# ==============================================================================
# 3. 定义文档处理函数
# ==============================================================================
def process_document(doc):
    """从单个文档中提取、清洗并生成所有文本块及其元数据。"""
    title = doc.get('title', '').strip()
    content = doc.get('text', '').strip()
    url = doc.get('url', '')
    
    # 如果标题或内容为空，则跳过该文档
    if not title or not content:
        return []
    
    split_chunks = text_splitter.split_text(content)
    
    processed_chunks = []
    for chunk_text in split_chunks:
        # 为E5模型准备的、带有指令前缀的文本
        text_for_embedding = f"passage: {chunk_text}"
        # 用于存储和后续检索的元数据
        metadata = {
            'title': title,
            'original_text': chunk_text,
            'url': url
        }
        processed_chunks.append((text_for_embedding, metadata))
    return processed_chunks

# ==============================================================================
# 4. 加载数据集并进行分块
# ==============================================================================
print(f"Loading dataset '{DATASET_NAME}' ({DATASET_CONFIG})...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)

# ！！！如果需要测试，请取消下面的注释，只处理少量数据！！！
# print("--- RUNNING IN TEST MODE: PROCESSING ONLY 10000 DOCUMENTS ---")
# ds = ds.take(10000)

print("Starting to chunk all documents...")
start_time = time.time()

all_data = []
# 使用tqdm显示处理文档的进度
for doc in tqdm(ds, desc="Chunking documents"):
    chunks = process_document(doc)
    all_data.extend(chunks)

# 分离文本和元数据
texts_for_embedding, all_chunks_metadata = zip(*all_data)

end_time = time.time()
print(f"Chunking finished in {end_time - start_time:.2f} seconds.")
print(f"Generated {len(texts_for_embedding)} chunks in total.")

# ==============================================================================
# 5. 【核心优化】按文本长度排序以最大化GPU效率
# ==============================================================================
print("Sorting chunks by length to optimize batching for GPU...")
start_time = time.time()

# 1. 将文本和元数据打包回一个列表
#    (我们之前已经有all_data, 但为了代码清晰，我们重新打包)
combined_data = list(zip(texts_for_embedding, all_chunks_metadata))

# 2. 使用文本的长度作为排序的key，从长到短排序
combined_data.sort(key=lambda x: len(x[0]), reverse=True)

# 3. 解包，得到两个已经同步排序好的新列表
sorted_texts, sorted_metadata = zip(*combined_data)

end_time = time.time()
print(f"Sorting finished in {end_time - start_time:.2f} seconds.")

# ==============================================================================
# 6. 使用模型进行编码（Embedding）
# ==============================================================================
print("Starting embedding process with the sorted data...")
embeddings = model.encode(
    list(sorted_texts),  # model.encode 最好接收一个列表
    batch_size=ENCODE_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Embedding process finished. Generated embeddings of shape: {embeddings.shape}")

# ==============================================================================
# 7. 构建并保存Faiss索引
# ==============================================================================
print("Building and saving Faiss index...")
vector_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
index.add(embeddings)

print(f"Faiss index built successfully with {index.ntotal} vectors.")
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"Faiss index has been saved to: {FAISS_INDEX_PATH}")

# ==============================================================================
# 8. 保存排序后的文本块元数据
# ==============================================================================
print("Saving chunk metadata...")
# 非常重要：我们必须保存排序后的元数据，这样索引才能正确对应
with open(CHUNKS_PATH, 'wb') as f:
    pickle.dump(list(sorted_metadata), f) # 解包后是tuple元组，转回list列表保存更通用
print(f"Chunk metadata has been saved to: {CHUNKS_PATH}")
print("\n--- All steps completed successfully! ---\n")

# ==============================================================================
# 9. 检索系统使用示例
# ==============================================================================
print("\n--- Retrieval System Usage Example ---")

def search_wiki_with_e5(query, k=5):
    print(f"\nSearching for: '{query}' ...")
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print("Error: Index or chunk files not found. Please run the full script first.")
        return

    # 加载索引和数据
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks_data = pickle.load(f)
        
    # 为查询添加指令前缀并编码
    query_vector = model.encode([f"query: {query}"])
    
    # 在Faiss索引中搜索
    distances, indices = index.search(query_vector, k)

    print(f"\n--- Top {k} most relevant results ---")
    for i, idx in enumerate(indices[0]):
        # Faiss在找不到足够结果时会返回-1
        if idx == -1:
            continue
        
        retrieved_chunk = chunks_data[idx]
        print(f"Result {i+1} (Distance Score: {distances[0][i]:.4f})")
        print(f"  Source Article: '{retrieved_chunk['title']}'")
        print(f"  Relevant Passage: \"...{retrieved_chunk['original_text']}...\"")
        print(f"  URL: {retrieved_chunk['url']}")
        print("-" * 40)

# 在脚本构建完索引后，自动执行一次示例搜索
if __name__ == "__main__":
    if os.path.exists(FAISS_INDEX_PATH):
        search_wiki_with_e5("What is the theory of general relativity?")