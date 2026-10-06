# 临时：文本检索各环节耗时（人工智能NLP-RAG-图像内容解析及检索优化）
import os
import time

from dotenv import load_dotenv

load_dotenv()
from pymilvus import MilvusClient

from src.embedding import get_embedder
from src.reranker import Reranker

emb = get_embedder()
rr = Reranker()
client = MilvusClient(uri=f"http://{os.getenv('MILVUS_HOST','localhost')}:"
                          f"{os.getenv('MILVUS_PORT','19530')}")

Q = "前五大客户占营业收入的比例是多少"

t0 = time.perf_counter()
qv = emb.encode([Q], show_progress_bar=False)[0]
print(f"bge encode: {(time.perf_counter()-t0)*1000:.0f}ms")

t0 = time.perf_counter()
res = client.search(
    collection_name="rag_chunks",
    data=[qv.tolist()],
    limit=24,
    output_fields=["doc_id", "chunk_id", "content", "page"],
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    filter='doc_id == "招股说明书2"',
)
print(f"milvus vec search: {(time.perf_counter()-t0)*1000:.0f}ms hits={len(res[0])}")

t0 = time.perf_counter()
kres = client.query(
    collection_name="rag_chunks",
    filter='doc_id == "招股说明书2" and content like "%客户%"',
    output_fields=["chunk_id"],
    limit=24,
)
print(f"milvus LIKE query(单词): {(time.perf_counter()-t0)*1000:.0f}ms hits={len(kres)}")

cands = [{"content": h["entity"]["content"][:1500]} for h in res[0]][:8]
t0 = time.perf_counter()
rr.rerank(Q, cands, top_k=8)
print(f"rerank 8: {(time.perf_counter()-t0)*1000:.0f}ms")

t0 = time.perf_counter()
rr.rerank(Q, cands[:5], top_k=5)
print(f"rerank 5: {(time.perf_counter()-t0)*1000:.0f}ms")
