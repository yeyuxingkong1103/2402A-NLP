#!/usr/bin/env python
# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
融合权重扫描：`w_keyword` × 融合方式，按**确定性指标**选最优点。

【为什么复用 sweep.py 的两阶段思路】**阶段一不调 LLM**。
查询向量与 `w_keyword` 无关，所以每题只嵌入一次、跨所有格子复用；
整轮网格几十格、十几秒跑完。只有真要报 TTFT 时才进阶段二（跑真实生成）。

跑法：
    PY=~/rag-data/venv/bin/python
    $PY scripts/sweep_fusion.py
    $PY scripts/sweep_fusion.py --weights 0,0.2,0.4,0.6,0.8,1.0 --fusion rrf,weighted
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.doc_profiles import get_doc_profile          # noqa: E402
from app.core.evaluator import keyword_hits, retrieval_metrics  # noqa: E402
from app.core.profiles import get_profile                   # noqa: E402
from app.core.retriever import Retriever, build_context     # noqa: E402


async def eval_cell(profile, questions) -> dict:
    r = Retriever(profile=profile)
    pr = ckc = rec = 0.0
    n = len(questions)
    for q in questions:
        res = await r.retrieve(q["question"])
        doc = get_doc_profile(res.routed_doc)
        rm = retrieval_metrics(res.hits, q.get("rule_keywords", []), q.get("evidence_page", ""),
                               page_re=(doc.evidence_re if doc else None),
                               recall_keywords=q.get("recall_keywords"))
        pr += rm["page_recall"] or 0
        ckc += rm["context_keyword_coverage"] or 0
        rec += rm["recall_coverage"] or 0
    return {"page_recall": pr / n, "ckc": ckc / n, "recall": rec / n,
            "context_chars": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description="工单06 · 融合权重扫描")
    ap.add_argument("--base", default="optimized", help="以哪个剖面为基准派生")
    ap.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--fusion", default="rrf,weighted")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qs = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))["questions"]
    if args.limit:
        qs = qs[: args.limit]
    base = get_profile(args.base)
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    fusions = [x.strip() for x in args.fusion.split(",") if x.strip()]

    async def run():
        print(f"基准剖面 {args.base}（{len(qs)} 题）\n")
        b = await eval_cell(base, qs)
        print(f"{'(基线) 向量检索':36s} 页召回 {b['page_recall']:6.1%}  "
              f"CKC {b['ckc']:6.1%}  要点召回 {b['recall']:6.1%}")
        print()
        rows = []
        for fusion in fusions:
            for w in weights:
                prof = base.derived(name=f"w{w}", retrieval_mode="hybrid",
                                    fusion=fusion, w_keyword=w,
                                    reranker="lexical", lexical_rerank=False)
                m = await eval_cell(prof, qs)
                rows.append((fusion, w, m))
                flag = ""
                if m["recall"] > b["recall"]:
                    flag = "  ← 要点召回优于基线"
                print(f"{'hybrid/'+fusion+f' w_keyword={w}':36s} 页召回 {m['page_recall']:6.1%}  "
                      f"CKC {m['ckc']:6.1%}  要点召回 {m['recall']:6.1%}{flag}")
            print()
        best = max(rows, key=lambda r: (r[2]["recall"], r[2]["ckc"]))
        print(f"最优：fusion={best[0]} w_keyword={best[1]}"
              f" → 要点召回 {best[2]['recall']:.1%} / CKC {best[2]['ckc']:.1%}"
              f" / 页召回 {best[2]['page_recall']:.1%}")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
