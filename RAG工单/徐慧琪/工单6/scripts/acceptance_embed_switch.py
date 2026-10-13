# 工单编号：人工智能NLP-RAG-混合检索任务
"""验收项 A2 加压复核：嵌入模型**真切换**跑通（不是只看注册表）。

前置：`python build_index.py --embedding-model bge-m3-ft-v1` 已建好第二个库。
本脚本：两个库分别点数 → 默认模型问一题 → 切到 bge-m3-ft-v1 再问同一题 →
切回默认，确认主库未被破坏。

用法：python scripts/acceptance_embed_switch.py
产出：logs/acceptance_embed_switch.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import bootstrap  # noqa: E402

OUT = os.path.join("logs", "acceptance_embed_switch.json")
Q = "武汉兴图新科电子股份有限公司注册资本是多少？"


def main() -> int:
    from src import config, rag, runtime, vector_store

    report: dict = {}
    rag.warmup()

    # 1) 两个库各自的点数
    counts = {}
    for m in ("bge-m3", "bge-m3-ft-v1"):
        name = config.collection_name(m)
        counts[m] = {"collection": name, "points": vector_store.count(name)}
    report["collections"] = counts

    # 2) 默认模型（bge-m3）
    runtime.update(strategy="vector", embedding_model="bge-m3", rerank_algorithm="model")
    r1 = rag.ask(Q, use_cache=False)
    report["default_bge_m3"] = {
        "collection": config.collection_name("bge-m3"),
        "answer": (r1.answer or "")[:120],
        "n_candidates": (r1.analysis or {}).get("recall_candidates"),
        "retrieval_info": {k: v for k, v in
                           ((r1.analysis or {}).get("retrieval_info") or {}).items()
                           if k in ("strategy", "n_candidates", "recall_counts")},
        "total_ms": (r1.timings or {}).get("total_ms"),
    }

    # 3) 切到 bge-m3-ft-v1（微调版），同一题
    runtime.update(embedding_model="bge-m3-ft-v1")
    r2 = rag.ask(Q, use_cache=False)
    report["switched_ft_v1"] = {
        "collection": config.collection_name("bge-m3-ft-v1"),
        "answer": (r2.answer or "")[:120],
        "n_candidates": (r2.analysis or {}).get("recall_candidates"),
        "retrieval_info": {k: v for k, v in
                           ((r2.analysis or {}).get("retrieval_info") or {}).items()
                           if k in ("strategy", "n_candidates", "recall_counts")},
        "total_ms": (r2.timings or {}).get("total_ms"),
    }

    # 4) 切回默认，确认主库仍可用
    runtime.update(embedding_model="bge-m3")
    r3 = rag.ask(Q, use_cache=False)
    report["back_to_bge_m3"] = {
        "answer": (r3.answer or "")[:120],
        "n_candidates": (r3.analysis or {}).get("recall_candidates"),
    }

    report["ok"] = bool(
        counts.get("bge-m3", {}).get("points", 0) > 0
        and counts.get("bge-m3-ft-v1", {}).get("points", 0) > 0
        and report["default_bge_m3"]["answer"]
        and report["switched_ft_v1"]["answer"]
        and report["back_to_bge_m3"]["answer"]
    )
    runtime.reset()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("ok =", report["ok"])
    for k in ("collections", "default_bge_m3", "switched_ft_v1", "back_to_bge_m3"):
        print(k, "->", json.dumps(report[k], ensure_ascii=False)[:220])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(bootstrap.run_with_large_stack(main))
