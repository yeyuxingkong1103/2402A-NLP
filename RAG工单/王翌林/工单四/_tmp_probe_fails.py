# 临时：诊断 id3/id5/id10 的检索召回（工单四）
import os

from dotenv import load_dotenv

load_dotenv()
from src.rag_engine_v4 import RAGEngineV4

CASES = [
    (3, "招股说明书1", "公司的主要关联方有哪些？"),
    (5, "招股说明书1", "报告期内，公司来自军用领域的收入分别是多少？"),
    (10, "招股说明书2", "力源信息报告期内主营业务收入构成是什么？"),
]
eng = RAGEngineV4()
# 预热
eng.ask("本次发行的发行股数是多少？", doc_id="招股说明书1")

for cid, doc, q in CASES:
    print("=" * 80)
    print(f"id{cid} {q}")
    res = eng.ask(q, doc_id=doc)
    ctx = (res.get("retrieved_text_chunks") or []) + (
        res.get("retrieved_tables") or [])
    print("img_route:", res.get("route"), "| n_ctx:", len(ctx),
          "| n_img:", len(res.get("retrieved_images") or []),
          "| retrieve_ms:", res.get("breakdown", {}).get("retrieve_ms"))
    for i, c in enumerate(ctx[:8]):
        src = c.get("source")
        page = c.get("page")
        content = (c.get("content") or "")[:110].replace("\n", " ")
        print(f"  [{i}] {src} p{page} :: {content}")
    print("ANS:", (res.get("answer") or "")[:200].replace("\n", " "))
