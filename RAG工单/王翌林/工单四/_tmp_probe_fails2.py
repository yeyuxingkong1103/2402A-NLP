# 临时：对比新旧融合/截断逻辑对 id5/id10 召回的影响（工单四）
from dotenv import load_dotenv

load_dotenv()
from src.rag_engine_v4 import RAGEngineV4
from src.table_parser.query_router import route_query as route3

CASES = [
    (5, "招股说明书1", "报告期内，公司来自军用领域的收入分别是多少？",
     ["6464", "14414", "18780", "军用", "国防"]),
    (10, "招股说明书2", "力源信息报告期内主营业务收入构成是什么？",
     ["主营业务收入", "构成", "IC卡", "电子元器件"]),
]
eng = RAGEngineV4()
tr = eng._v3.table_retriever
eng.ask("本次发行的发行股数是多少？", doc_id="招股说明书1")  # 预热

for cid, doc, q, gold in CASES:
    print("=" * 90)
    print(f"id{cid} {q}  GOLD={gold}")
    route = route3(q)
    print("主路由:", route.route, route.confidence)
    tables = tr.search_tables(q, top_k=tr.table_top_k, doc_id=doc)
    print(f"-- table_hits({len(tables)}):")
    for i, h in enumerate(tables):
        c = (h.get("content") or "")[:80].replace("\n", " ")
        hit = [g for g in gold if g in (h.get("content") or "")]
        print(f"   T{i} p{h.get('page')} score={h.get('score',0):.3f} "
              f"path={h.get('search_path')} gold={hit} :: {c}")
    texts = tr._search_text(q, top_k=tr.text_top_k, doc_id=doc)
    print(f"-- text_hits({len(texts)}):")
    for i, h in enumerate(texts):
        c = (h.get("content") or "")[:80].replace("\n", " ")
        hit = [g for g in gold if g in (h.get("content") or "")]
        print(f"   X{i} p{h.get('page')} rr={h.get('rerank_score',0):.3f} "
             f"gold={hit} :: {c}")

    # 新逻辑结果
    res_new = tr.retrieve(q, top_k=5, doc_id=doc, route=route)
    print("-- 新逻辑 fused top5:")
    for i, h in enumerate(res_new["fused"]):
        c = (h.get("content") or "")[:80].replace("\n", " ")
        hit = [g for g in gold if g in (h.get("content") or "")]
        print(f"   N{i} {h.get('source')} p{h.get('page')} "
              f"rr={h.get('rerank_score',0):.3f} gold={hit} :: {c}")

    # 旧逻辑：表格先精排 + RRF 全量 + 统一 rerank
    rt = tr._rerank(q, tables, len(tables))
    fused_old = tr._rrf_fuse(rt, texts, tr.rrf_k)
    old = tr._rerank(q, fused_old, 5)
    print("-- 旧逻辑(RRF+二次rerank) top5:")
    for i, h in enumerate(old):
        c = (h.get("content") or "")[:80].replace("\n", " ")
        hit = [g for g in gold if g in (h.get("content") or "")]
        print(f"   O{i} {h.get('source')} p{h.get('page')} "
              f"rr={h.get('rerank_score',0):.3f} gold={hit} :: {c}")
