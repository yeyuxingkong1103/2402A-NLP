# -*- coding: utf-8 -*-
"""工单13 优化前后对比生成器：把真实产物汇总成 对比.csv / 对比.md。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

输入（只读真实产物）：
  优化/评估结果/优化前_汇总.json、优化后_汇总.json
输出：
  优化/评估结果/对比.csv
  优化/评估结果/对比.md

设计要点：
  * 兼容两跑的字段命名差异（阶段均值键名带 `stage::` 前缀）；
  * **不隐藏负向结果**：优化后若更慢，变化列会显示正数（变慢），并在 md 里显式标注；
  * 同时写出两跑的「口径」段，便于判断可比性（轮数/缓存/题间休眠）。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"E:\gao6gongdan\工单13")
EVAL = ROOT / "优化" / "评估结果"

STAGES = ["1_查询处理与增强", "2_检索", "3_上下文组装与提示工程", "4_LLM生成", "5_后处理与响应格式化"]
METRICS = ["总耗时", "检索", "生成", "首字", "墙钟"]


def load(name: str) -> dict:
    p = EVAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def stat(s: dict, key: str, field: str = "mean") -> float | None:
    node = ((s.get("统计") or {}).get(key)) or {}
    v = node.get(field)
    return float(v) if isinstance(v, (int, float)) else None


def stage_mean(s: dict, stage: str) -> float | None:
    means = ((s.get("统计") or {}).get("阶段均值")) or {}
    for k, v in means.items():
        if stage in str(k) and isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict) and stage in str(k) and isinstance(v.get("mean"), (int, float)):
            return float(v["mean"])
    return None


def pct(b: float | None, a: float | None) -> str:
    if b in (None, 0) or a is None:
        return ""
    return f"{(a - b) / b * 100:+.1f}%"


def f1(v: float | None) -> str:
    return "" if v is None else f"{v:.1f}"


def main() -> int:
    before, after = load("优化前_汇总.json"), load("优化后_汇总.json")
    if not before or not after:
        print(f"[!!] 缺汇总：前={bool(before)} 后={bool(after)}")
        return 3

    rows: list[list[str]] = []
    for m in METRICS:
        for field, label in (("mean", "均值"), ("p50", "P50"), ("p90", "P90"), ("p95", "P95"), ("max", "最大")):
            b, a = stat(before, m, field), stat(after, m, field)
            rows.append([f"{m}({label})", f1(b), f1(a), pct(b, a)])
    for st in STAGES:
        b, a = stage_mean(before, st), stage_mean(after, st)
        rows.append([f"阶段-{st}", f1(b), f1(a), pct(b, a)])
    for label, s in (("优化前", before), ("优化后", after)):
        sla = (s.get("统计") or {}).get("SLA") or {}
        rows.append([f"SLA达标率-{label}(%)", "", f1((sla.get("达标率") or 0) * 100), ""])

    csv_path = EVAL / "对比.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["指标", "优化前", "优化后", "变化"])
        w.writerows(rows)

    b_sla = ((before.get("统计") or {}).get("SLA") or {}).get("达标率")
    a_sla = ((after.get("统计") or {}).get("SLA") or {}).get("达标率")
    b_p95, a_p95 = stat(before, "总耗时", "p95"), stat(after, "总耗时", "p95")

    lines = [
        "# 工单13 优化前后对比",
        "",
        f"**工单编号**：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化",
        "",
        f"**优化前**：{before.get('生成时间')}（档位 `{before.get('档位')}`，口径：轮数 {before.get('口径', {}).get('轮数')}／丢弃 {before.get('口径', {}).get('丢弃轮数')}／缓存 {before.get('口径', {}).get('缓存')}）",
        f"**优化后**：{after.get('生成时间')}（档位 `{after.get('档位')}`，口径：轮数 {after.get('口径', {}).get('轮数')}／丢弃 {after.get('口径', {}).get('丢弃轮数')}／缓存 {after.get('口径', {}).get('缓存')}）",
        "",
        "## 1 对比表",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
    lines += [
        "",
        "## 2 验收判定（SLA ≤ 3000 ms，按端到端 P95）",
        "",
        f"- 优化前 P95 = {f1(b_p95)} ms，达标率 = {f1((b_sla or 0) * 100)}%",
        f"- 优化后 P95 = {f1(a_p95)} ms，达标率 = {f1((a_sla or 0) * 100)}%",
        "",
    ]
    if a_p95 is not None and a_p95 <= 3000:
        lines.append("**判定：达标**（优化后端到端 P95 ≤ 3000 ms）。")
    else:
        lines += [
            "**判定：未达标，如实记录。**",
            "",
            "根因与说明见 `优化/瓶颈排查与修复报告.md`；若两次测量的机器负载不同（例如与其他重活并发），",
            "该对比不具可比性，需在机器空闲时重测 —— 本条由生成脚本按数据自动提示，不做人为修饰。",
        ]
    (EVAL / "对比.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] {csv_path}")
    print(f"[OK] {EVAL / '对比.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
