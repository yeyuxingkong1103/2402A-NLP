# -*- coding: utf-8 -*-
"""拒答阈值校准（阶段 8.4）：装配 + CLI。

现状：拒答判定是"零候选才拒答"（`guard.should_refuse`），等于没有分数阈值；
实测库内 11 部法规总能召回若干条，因此超出知识库范围的问题照样被作答，
拒答准确率恒为 0。本脚本用评测集跑出"该拒的都拒、不该拒的不拒"的分界点。

一条命令：
    python evaluation/calibrate_refusal.py

采集口径：对每道题跑一次生产检索（top10），记录首条候选的重排分/向量分，
再按"拒答题 vs 非拒答题"的分数分布搜索最优阈值（最大化平衡准确率）。

产物（默认写项目根 reports/）：
    refusal_scores_<时间戳>.json     每题分数明细 + 候选阈值扫描结果
    refusal_calibration_<时间戳>.md  人可读：阈值取值、依据、校准前后对比

本文件只做"装配 + CLI"（批次 24 拆分后 314 → 约 120 行，守住"单文件 ≤300 行"，
见 docs/目录与命名约定.md §3.4）。其余职责各自成模块，**都是逐字搬移、零逻辑改动**：
    refusal_scoring.py   分数采集（collect_scores）+ 阈值扫描（scan_thresholds）
    refusal_report.py    Markdown 渲染（render_markdown）
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_eval import (
    DEFAULT_EVAL_SET,
    DEFAULT_OUTPUT_DIR,
    prepare_env,
)

# ↓ 以下三个名字在本模块是"再导出"（re-export），不要删、也不要当作未使用而清理：
#   拆分前它们是本模块的定义，对外契约是"函数名不变"；保留以便 `from calibrate_refusal import ...`
#   这类写法继续可用（同 run_eval.py 拆分后的处理方式）。
from refusal_report import render_markdown  # noqa: F401
from refusal_scoring import collect_scores, scan_thresholds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用评测集校准拒答阈值。")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse-scores", type=Path, help="复用已采集的分数 JSON，不重跑检索")
    parser.add_argument("--before", type=Path, help="校准前评测报告 JSON（写入对比表）")
    parser.add_argument("--after", type=Path, help="校准后评测报告 JSON（写入对比表）")
    args = parser.parse_args(argv)

    prepare_env()

    if args.reuse_scores:
        payload = json.loads(args.reuse_scores.read_text(encoding="utf-8"))
        rows = payload["rows"]
    else:
        from app.retrieval.assembly import build_default_retrieval_service

        items = [
            json.loads(line)
            for line in args.eval_set.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.limit:
            items = items[: args.limit]
        retrieval_service = build_default_retrieval_service()
        rows = collect_scores(items, retrieval_service)
        payload = {
            "run_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
            "eval_set": str(args.eval_set),
            "rows": rows,
            "scans": {
                "top1_vector": scan_thresholds(rows, "top1_vector"),
                "top1_rerank": scan_thresholds(rows, "top1_rerank"),
            },
        }

    # 写入配置的取值：两簇边界的中点（比"扫描最优阈值"更稳，两侧都有余量）
    vector_scan = payload["scans"].get("top1_vector") or {}
    if vector_scan.get("refusal_top1") and vector_scan.get("normal_top1"):
        chosen = round(
            (max(vector_scan["refusal_top1"]) + min(vector_scan["normal_top1"])) / 2, 4
        )
    else:
        chosen = 0.0
    payload["chosen_threshold"] = chosen

    if args.before and args.after:
        before_payload = json.loads(args.before.read_text(encoding="utf-8"))
        after_payload = json.loads(args.after.read_text(encoding="utf-8"))
        payload["comparison"] = {
            "before": before_payload["summary"],
            "after": after_payload["summary"],
            "before_file": str(args.before),
            "after_file": str(args.after),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"refusal_scores_{stamp}.json"
    md_path = args.output_dir / f"refusal_calibration_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    (args.output_dir / "latest_refusal_calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "latest_refusal_calibration.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )

    for key, result in payload["scans"].items():
        if result.get("error"):
            print(f"{key}: {result['error']}")
            continue
        best = result["best"]
        print(f"{key}: 扫描最优阈值={best['threshold']} 平衡准确率={best['balanced_accuracy']} "
              f"拒答召回={best['refusal_recall']} 非拒答保留={best['normal_precision']} 误拒={best['fp']}")
    print(f"写入配置的取值：REFUSAL_MIN_VECTOR_SCORE={chosen}")
    print(f"报告：{json_path}")
    print(f"报告：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
