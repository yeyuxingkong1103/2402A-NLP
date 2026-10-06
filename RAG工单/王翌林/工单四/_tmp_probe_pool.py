# 临时：量化 text 检索 rerank 池构成、通道归属、cutoff_layers 加速（工单四）
import time

from dotenv import load_dotenv

load_dotenv()
from FlagEmbedding import FlagReranker

from src.embedding import get_embedder
from src.text_retriever_v3 import TextRetrieverV3

MODEL = "/home/dabaie/models/bge-reranker-v2-m3"
CASES = [
    (3, "招股说明书1", "公司的主要关联方有哪些？", ["关联方", "兴图", "武汉兴图"]),
    (5, "招股说明书1", "报告期内，公司来自军用领域的收入分别是多少？",
     ["6464", "18,780", "18780"]),
    (10, "招股说明书2", "力源信息报告期内主营业务收入构成是什么？",
     ["主营业务收入", "构成"]),
]
tx = TextRetrieverV3()
tx.embedder.encode(["warmup"], show_progress_bar=False)
tx._reranker.available
tx.embedder  # 预热
tx.retrieve("预热问题发行股数", doc_id="招股说明书1")

emb = tx.embedder
vs = tx.vector_store

full_rr = tx._reranker._model
cut8 = FlagReranker(MODEL, use_fp16=True, cutoff_layers=[8])
cut6 = FlagReranker(MODEL, use_fp16=True, cutoff_layers=[6])

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
    print(f"vec={len(vec)} kw_total={len(kres or [])} kw_new={len(kw_new)}")

    # 智能池：全向量 + kw_new top4
    pool = [{"content": h["content"][:1500], "page": h.get("page"),
             "path": "vec"} for h in vec]
    pool += [{"content": r["content"][:1500], "page": r.get("page"),
              "path": "kw"} for r in kw_new[:4]]
    print(f"smart_pool={len(pool)}")

    pairs = [[q, c["content"]] for c in pool]
    for name, model in (("full", full_rr), ("cut8", cut8), ("cut6", cut6)):
        t0 = time.perf_counter()
        sc = model.compute_score(pairs, normalize=True)
        if isinstance(sc, float):
            sc = [sc]
        ms = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:8]
        tags = []
        for i in order:
            g = [g for g in gold if g in pool[i]["content"]]
            tags.append(f"p{pool[i]['page']}{'*' if g else ''}{pool[i]['path'][0]}")
        print(f"  {name}: {ms:.0f}ms top8={tags}")
