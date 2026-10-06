# 工单编号：人工智能 NLP-RAG-Query 理解优化任务
import json
import chromadb
from sentence_transformers import SentenceTransformer
from config_v5 import CHUNK_FILE, CHROMA_DIR, EMBED_MODEL_PATH


def main():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    print("片段数:", len(chunks))

    model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("zhaogu_v5")
    except Exception:
        pass
    collection = client.create_collection("zhaogu_v5", metadata={"hnsw:sync_threshold": 100000})

    batch = 64
    for i in range(0, len(chunks), batch):
        b = chunks[i:i+batch]
        emb = model.encode([x["text"] for x in b], normalize_embeddings=True).tolist()
        collection.add(
            documents=[x["text"] for x in b],
            embeddings=emb,
            metadatas=[{
                "page": x["page"],
                "type": x["type"],
                "source": x["source"]
            } for x in b],
            ids=[f"v5_{j}" for j in range(i, i+len(b))]
        )
        print(f"{i+len(b)}/{len(chunks)}")
    print("done")


if __name__ == "__main__":
    main()