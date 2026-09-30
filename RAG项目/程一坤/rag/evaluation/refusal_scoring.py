# -*- coding: utf-8 -*-
"""分数采集与阈值扫描（批次 24 自 `calibrate_refusal.py` 拆出）。

为什么单独成文件：`calibrate_refusal.py` 拆分前 314 行，超出项目"单文件 ≤300 行"
硬规则（见 `docs/目录与命名约定.md` §3.4）；本模块只装"把分数拿到手"和
"在分数上找分界点"这两件事，**零逻辑改动**（函数体逐字搬移，对外函数名不变）。

与 `refusal_report.py` 的分工：
- 本模块：产出**数据**（每题一行明细 + 阈值扫描曲线），不关心怎么排版；
- `refusal_report.py`：把数据渲染成 Markdown，不重算任何指标。
这样"算"与"写"分开，改报告版式不会碰到判定口径。

阈值口径（与 `guard.should_refuse` 配合）：`top1 分数 < 阈值 → 拒答`。
"""

from __future__ import annotations

import time
from typing import Any

# with_retry 仍从 run_eval 取（拆分后 run_eval 保留了这些名字的再导出），
# 避免为了一个重试包装再拆一层模块。
from run_eval import with_retry


def collect_scores(items: list[dict[str, Any]], retrieval_service, top_n: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['id']} …", flush=True)
        row: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "expect_refusal": bool(item.get("expect_refusal")),
            "question": item["question"],
        }
        started = time.time()
        try:
            result = with_retry(
                lambda: retrieval_service.retrieve(
                    item["question"],
                    rerank_top_n=top_n,
                    as_of_date=item.get("as_of_date"),
                    jurisdiction="中国大陆",
                    user_id="calib_runner",
                    session_id=None,
                ),
                attempts=5,
            )
        except Exception as error:  # noqa: BLE001
            row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            continue
        row["elapsed_seconds"] = round(time.time() - started, 2)
        articles = result.articles
        row["candidate_count"] = len(articles)
        rerank_scores = [a.rerank_score for a in articles if a.rerank_score is not None]
        vector_scores = [a.vector_score for a in articles if a.vector_score is not None]
        row["top1_rerank"] = round(max(rerank_scores), 4) if rerank_scores else None
        row["top1_vector"] = round(max(vector_scores), 4) if vector_scores else None
        row["top1_law"] = articles[0].document_title if articles else None
        row["top1_article"] = articles[0].article_number if articles else None
        row["rerank_scores_top10"] = [round(s, 4) for s in rerank_scores[:10]]
        rows.append(row)
    return rows


def scan_thresholds(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """在候选阈值上扫描，取平衡准确率最高的分界点。

    判定规则：top1 分数 < 阈值 → 拒答；>= 阈值 → 作答。
    拒答题判对 = 拒答；非拒答题判对 = 作答。
    """
    usable = [r for r in rows if r.get(key) is not None]
    refusals = [r for r in usable if r["expect_refusal"]]
    normals = [r for r in usable if not r["expect_refusal"]]
    if not refusals or not normals:
        return {"error": "分数样本不足，无法扫描阈值"}

    candidates = sorted({round(r[key], 4) for r in usable})
    best: dict[str, Any] | None = None
    curve: list[dict[str, Any]] = []
    for threshold in candidates:
        tp = sum(1 for r in refusals if r[key] < threshold)   # 拒答题被拒
        fn = len(refusals) - tp
        fp = sum(1 for r in normals if r[key] < threshold)    # 正常题被误拒
        tn = len(normals) - fp
        sensitivity = tp / len(refusals)
        specificity = tn / len(normals)
        balanced = (sensitivity + specificity) / 2
        entry = {
            "threshold": threshold,
            "refusal_recall": round(sensitivity, 4),
            "normal_precision": round(specificity, 4),
            "balanced_accuracy": round(balanced, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        }
        curve.append(entry)
        if best is None or balanced > best["balanced_accuracy"] or (
            balanced == best["balanced_accuracy"] and threshold < best["threshold"]
        ):
            best = entry

    assert best is not None
    boundary = sorted(usable, key=lambda r: r[key])
    return {
        "metric": key,
        "best": best,
        "refusal_top1": sorted(round(r[key], 4) for r in refusals),
        "normal_top1": sorted(round(r[key], 4) for r in normals),
        "boundary_samples": [
            {"id": r["id"], "expect_refusal": r["expect_refusal"], "score": round(r[key], 4),
             "top1": f"{r.get('top1_law')}/{r.get('top1_article')}"}
            for r in boundary[:8]
        ],
        "curve": curve,
    }
