# 临时：热路径 encode/rerank 耗时（人工智能NLP-RAG-图像内容解析及检索优化）
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
QS = ["前五大客户占营业收入的比例是多少",
      "本次募集资金总额是多少万元",
      "公司的主营业务是什么产品"]

# 预热（含模型加载）
emb.encode(["warmup"], show_progress_bar=False)
rr.available
res0 = client.search(collection_name="rag_chunks",
                     data=[emb.encode(["warmup"], show_progress_bar=False)[0].tolist()],
                     limit=24, output_fields=["content"],
                     filter='doc_id == "招股说明书2"')
rr.rerank("warmup", [{"content": "热身"}] * 4, top_k=4)

for Q in QS:
    t0 = time.perf_counter()
    qv = emb.encode([Q], show_progress_bar=False)[0]
    enc_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res = client.search(collection_name="rag_chunks", data=[qv.tolist()], limit=24,
                        output_fields=["doc_id", "chunk_id", "content", "page"],
                        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                        filter='doc_id == "招股说明书2"')
    vec_ms = (time.perf_counter() - t0) * 1000

    for n in (12, 24, 40):
        cands = [{"content": h["entity"]["content"][:1500]} for h in res[0]][:n]
        if len(cands) < n:
            continue
        t0 = time.perf_counter()
        rr.rerank(Q, cands, top_k=8)
        r_ms = (time.perf_counter() - t0) * 1000
        print(f"Q={Q[:12]} encode={enc_ms:.0f} vec={vec_ms:.0f} rerank@{n}={r_ms:.0f}")
