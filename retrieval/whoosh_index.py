import os
import shutil
from datasets import load_dataset
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from tqdm import tqdm

INDEX_DIR = "/home/v-zhaowan/zhaowang/rag/retrieval/bm25"
COMMIT_BATCH_SIZE = 100000

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
print(ds)

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True))
if os.path.exists(INDEX_DIR):
    shutil.rmtree(INDEX_DIR)
os.makedirs(INDEX_DIR, exist_ok=True) 

ix = create_in(INDEX_DIR, schema)
writer_args = {'limitmb': 2048, 'procs': 8, 'multisegment': True}
writer = ix.writer(**writer_args)
doc_count = 0
total_docs_in_train = len(ds["train"])
print(f"indexing wiki corpus ({total_docs_in_train} in total)")

for idx, item in tqdm(enumerate(ds["train"]), total=total_docs_in_train, desc="Indexing"):
    title = item.get('title', '').strip()
    content = item.get('text', '').strip()

    if not title or not content:
        continue

    try:
        writer.add_document(title=title, path=str(idx), content=content)
        doc_count += 1

        if doc_count % COMMIT_BATCH_SIZE == 0:
            writer.commit() 
            writer = ix.writer()

    except Exception as e:
        print(f"Error: {e}")

writer.commit()