# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
import json
import chromadb
from sentence_transformers import SentenceTransformer
from config_v2 import CHUNK_FILE, CHROMA_DIR, EMBED_MODEL_PATH

def main():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print("片段数:", len(chunks))

    model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("zhaogu_v2")
    except Exception:
        pass
    collection = client.create_collection(
        "zhaogu_v2",
        metadata={"hnsw:sync_threshold": 100000}
    )

    batch = 64
    texts = [c["text"] for c in chunks]
    pages = [c["page"] for c in chunks]
    for i in range(0, len(texts), batch):
        b_texts = texts[i:i+batch]
        b_pages = pages[i:i+batch]
        emb = model.encode(b_texts, normalize_embeddings=True).tolist()
        collection.add(
            documents=b_texts,
            embeddings=emb,
            metadatas=[{"page": p} for p in b_pages],
            ids=[f"v2_{j}" for j in range(i, i+len(b_texts))]
        )
        print(f"{i+len(b_texts)}/{len(texts)}")
    print("done")

if __name__ == "__main__":
    main()