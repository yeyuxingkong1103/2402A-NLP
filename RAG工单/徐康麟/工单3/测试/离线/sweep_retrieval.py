# -*- coding: utf-8 -*-
"""T5 参数扫描：融合/召回参数对「14 题证据落进 top-5」的影响（实测，不猜）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

做法：固定索引不变，只改检索参数（不改代码），逐组合跑 14 题，
按 `text_utils.evidence_contains` 判定命中，输出命中率矩阵与逐题差异。
产物：优化/评估结果/retrieval_sweep_t5.json

用法：
    pwsh -NoProfile -File run_py.ps1 测试/离线/sweep_retrieval.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import text_utils  # noqa: E402
from app.core.config import load_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.retriever import HybridRetriever  # noqa: E402

EVAL_SET = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"
OUT = REPO_ROOT / "优化" / "评估结果" / "retrieval_sweep_t5.json"


def run_one(vector_weight: float, bm25_weight: float, rescue: float, questions: list[dict], *,
            top_k: int = 5) -> dict:
    """用给定参数跑一轮 14 题，返回逐题命中与命中率。"""
    cfg = load_config(env={
        "RAG_RETRIEVAL__VECTOR_WEIGHT": str(vector_weight),
        "RAG_RETRIEVAL__BM25_WEIGHT": str(bm25_weight),
        "RAG_RETRIEVAL__RANK_RESCUE_WEIGHT": str(rescue),
    })
    retriever = HybridRetriever(cfg=cfg).load(warmup=False)
    # 分词器已在外层预热（warmup=False 只跳过预热，不影响正确性）
    hits: dict[str, bool] = {}
    for item in questions:
        result = retriever.retrieve(item["question"], top_k=top_k)
        evidence = item.get("evidence_verbatim") or item["evidence"]
        hit = any(c.content and text_utils.evidence_contains(c.content, evidence) for c in result.chunks)
        hits[str(item["id"])] = hit
    return {"vector_weight": vector_weight, "bm25_weight": bm25_weight, "rescue": rescue,
            "hit": sum(hits.values()), "rate": round(sum(hits.values()) / len(hits), 4),
            "per_question": hits}


def main() -> int:
    """跑参数网格并打印结果矩阵。"""
    cfg = load_config()
    setup_logging(cfg, force=True)
    questions = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    text_utils.warmup_tokenizer()          # 只预热一次（jieba 冷启动约 1 s，不重复付）
    print(f"题集 {len(questions)} 题；参数网格 = (w_vector, w_bm25) × rescue")
    rows: list[dict] = []
    grid = [(1.0, 1.0), (0.8, 1.0), (1.0, 1.5), (1.0, 2.0), (0.8, 1.5), (1.0, 3.0)]
    for (wv, wb), rescue in itertools.product(grid, (0.0, 1.0)):
        row = run_one(wv, wb, rescue, questions)
        rows.append(row)
        flags = "".join("✅" if row["per_question"][str(q["id"])] else "❌" for q in questions)
        print(f"  w_vector={wv:<4} w_bm25={wb:<4} rescue={rescue:<4} → {row['hit']:>2}/14 = "
              f"{row['rate'] * 100:5.1f}%  {flags}")
    best = max(rows, key=lambda r: (r["hit"], -r["bm25_weight"], -r["rescue"]))
    print(f"\n最佳：w_vector={best['vector_weight']} w_bm25={best['bm25_weight']} rescue={best['rescue']} "
          f"命中 {best['hit']}/14")
    print("未命中：", [k for k, v in best["per_question"].items() if not v])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"grid": rows, "best": best,
                               "question_ids": [q["id"] for q in questions]},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写 {OUT}")
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
