# 临时：对比新旧 hybrid 融合对"注册资本"top5 的影响（工单四）
from dotenv import load_dotenv

load_dotenv()
from src.rag_engine_v3 import RAGEngineV3

q = "武汉兴图新科电子股份有限公司的注册资本是多少？"
eng = RAGEngineV3()
tr = eng.table_retriever
# 预热
eng.ask_rag("本次发行的发行股数是多少？", doc_id="招股说明书1")

res = tr.retrieve(q, top_k=5, doc_id="招股说明书1")
print("route:", res["route"].route)
print("-- 新逻辑 fused top5:")
for i, h in enumerate(res["fused"]):
    c = (h.get("content") or "")[:90].replace("\n", " ")
    has = "注册资本" in (h.get("content") or "")
    print(f"  N{i} {h.get('source')} p{h.get('page')} rr={h.get('rerank_score',0):.3f} "
          f"{'★含注册资本' if has else ''} :: {c}")

# 复现旧逻辑
tables = tr.search_tables(q, top_k=tr.table_top_k, doc_id="招股说明书1")
texts = tr._search_text(q, top_k=tr.text_top_k, doc_id="招股说明书1")
print(f"-- text_hits({len(texts)}):")
for i, h in enumerate(texts):
    has = "注册资本" in (h.get("content") or "")
    print(f"  X{i} p{h.get('page')} rr={h.get('rerank_score',0):.3f} "
          f"{'★' if has else ''} :: {(h.get('content') or '')[:70].strip()[:70]}")
fused_old = tr._rrf_fuse(tables, texts, tr.rrf_k)
old = tr._rerank(q, fused_old, 5)
print("-- 旧逻辑 fused top5:")
for i, h in enumerate(old):
    c = (h.get("content") or "")[:90].replace("\n", " ")
    has = "注册资本" in (h.get("content") or "")
    print(f"  O{i} {h.get('source')} p{h.get('page')} rr={h.get('rerank_score',0):.3f} "
          f"{'★含注册资本' if has else ''} :: {c}")
