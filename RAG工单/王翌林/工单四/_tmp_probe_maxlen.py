# 临时：rerank max_length 对延迟/排序影响（工单四）
import time

from dotenv import load_dotenv

load_dotenv()
from src.embedding import get_embedder
from src.text_retriever_v3 import TextRetrieverV3

CASES = [
    (3, "招股说明书1", "公司的主要关联方有哪些？", ["关联方"]),
    (5, "招股说明书1", "报告期内，公司来自军用领域的收入分别是多少？",
     ["6464", "18,780"]),
    (10, "招股说明书2", "力源信息报告期内主营业务收入构成是什么？",
     ["主营业务收入"]),
    ("s2", "招股说明书1", "本次发行的募集资金总额是多少？", ["募集资金"]),
]
tx = TextRetrieverV3()
tx.retrieve("预热问题发行股数", doc_id="招股说明书1")
emb, vs, rr = tx.embedder, tx.vector_store, tx._reranker._model

import jieba
STOP = set("的 了 和 是 在 有 这 那 就 不 也 都 一 个 上 下 中 到 多少 是".split())

for cid, doc, q, gold in CASES:
    print("=" * 90)
    print(f"id{cid} {q}")
    qv = emb.encode([q], show_progress_bar=False)[0]
    vec = vs.search(qv, top_k=24, doc_id=doc)
    for h in vec:
        h["content"] = h.get("content") or h.get("text") or ""
    toks = [t for t in jieba.lcut(q) if len(t.strip()) >= 2 and t not in STOP]
    kres = vs.client.query(collection_name=vs.collection,
                           filter=" or ".join(f'content like "%{t}%"' for t in toks[:5]),
                           output_fields=["doc_id", "chunk_id", "content", "page"],
                           limit=24)
    vec_ids = {f"{h.get('doc_id')}|{h.get('chunk_id')}" for h in vec}
    kw_new = [r for r in (kres or [])
              if f"{r.get('doc_id')}|{r.get('chunk_id')}" not in vec_ids]
    kw_new.sort(key=lambda r: sum(1 for t in toks if t in r.get("content", "")),
                reverse=True)
    pool = [{"content": h["content"], "page": h.get("page")} for h in vec]
    pool += [{"content": r["content"], "page": r.get("page")} for r in kw_new[:4]]
    for cut in (1500, 900, 600):
        pairs = [[q, c["content"][:cut]] for c in pool]
        # 预热一轮
        rr.compute_score(pairs[:4], normalize=True)
        t0 = time.perf_counter()
        sc = rr.compute_score(pairs, normalize=True)
        ms = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:8]
        tags = [f"p{pool[i]['page']}{'*' if any(g in pool[i]['content'] for g in gold) else ''}"
                for i in order]
        print(f"  cut={cut}: {ms:.0f}ms top8={tags}")
