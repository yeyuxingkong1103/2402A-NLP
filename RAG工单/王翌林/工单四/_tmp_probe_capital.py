# 临时：诊断"注册资本"题通道归属（工单四）
from dotenv import load_dotenv

load_dotenv()
from src.embedding import get_embedder
from src.vector_store import VectorStore

q = "武汉兴图新科电子股份有限公司的注册资本是多少？"
doc = "招股说明书1"
emb = get_embedder()
vs = VectorStore()
qv = emb.encode([q], show_progress_bar=False)[0]
vec = vs.search(qv, top_k=24, doc_id=doc)
for h in vec:
    h["content"] = h.get("content") or h.get("text") or ""
import jieba
toks = [t for t in jieba.lcut(q) if len(t.strip()) >= 2]
toks = list(dict.fromkeys(toks))
kres = vs.client.query(
    collection_name=vs.collection,
    filter=f'(doc_id == "{doc}") and ('
           + " or ".join(f'content like "%{t}%"' for t in toks[:5]) + ")",
    output_fields=["doc_id", "chunk_id", "content", "page"], limit=24)
vec_ids = {f"{h.get('doc_id')}|{h.get('chunk_id')}" for h in vec}
print("vec 中含'注册资本'或'万元注册'的：")
for i, h in enumerate(vec):
    if "注册资本" in h["content"] or "股本" in h["content"][:200]:
        print(f"  V{i} p{h.get('page')} :: {h['content'][:120]}")
kw_new = [r for r in (kres or [])
          if f"{r.get('doc_id')}|{r.get('chunk_id')}" not in vec_ids]
kw_new.sort(key=lambda r: sum(1 for t in toks if t in r.get("content", "")),
            reverse=True)
print(f"kw_new 共 {len(kw_new)}，含注册资本的：")
for i, r in enumerate(kw_new):
    if "注册资本" in r.get("content", ""):
        print(f"  K{i} p{r.get('page')} kw_hits="
              f"{sum(1 for t in toks if t in r['content'])} "
              f"{'<<< 在 top4 内' if i < 4 else '<<< 被截掉！！'}")
        print("   ", r["content"][:150].replace("\n", " "))
