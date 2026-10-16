# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

结果汇总与前后对比：``研发/scripts/汇总对比.py``。

读取 ``测试/测试数据/*_summary.json``，产出工单要求的四份评估结果：

* ``优化/评估结果/基准_优化前.json`` —— 优化前各场景原始汇总
* ``优化/评估结果/优化后.json``     —— 优化后各场景原始汇总
* ``优化/评估结果/对比.csv``        —— 逐指标前后数字 + 判定
* ``优化/评估结果/对比.md``         —— 人类可读的对比表 + 结论

**纪律：只做搬运与算术，不编造任何数字。** 所有值都来自 summary.json 里由压测器实测写入的字段。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "测试" / "测试数据"
_OUT = _ROOT / "优化" / "评估结果"

#: 场景口径 → 展示名
SCENARIOS = {
    ("基准_优化前", "A"): "场景A 高频问答（20 并发 / 无思考 / 6 题缓存友好）",
    ("优化后", "A"): "场景A 高频问答（20 并发 / 无思考 / 6 题缓存友好）",
    ("基准_优化前_多用户", "A"): "场景A 多用户口径（20 并发 / 每请求新会话）",
    ("优化后_多用户", "A"): "场景A 多用户口径（20 并发 / 每请求新会话）",
    ("基准_优化前", "B"): "场景B 混合负载（10 并发＝5 问答＋5 上传解析）",
    ("优化后", "B"): "场景B 混合负载（10 并发＝5 问答＋5 上传解析）",
    ("基准_优化前_思考200ms", "A"): "场景A 补充口径（20 并发 / 思考 200 ms）",
    ("优化后_思考200ms", "A"): "场景A 补充口径（20 并发 / 思考 200 ms）",
    ("soak_优化后", "A"): "资源泄漏 soak（8 并发 / 15 分钟）",
    ("soak_基准_优化前", "A"): "资源泄漏 soak（8 并发 / 15 分钟）",
}


def load_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(_DATA.glob("*_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["_file"] = path.name
        rows.append(data)
    return rows


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 summary 列表整理成可比较的记录。"""
    out: list[dict[str, Any]] = []
    for d in rows:
        mem = _get(d, "memory", "api_process", default={}) or {}
        out.append({
            "标签": d.get("tag"), "场景": d.get("scenario"),
            "口径": SCENARIOS.get((d.get("tag"), d.get("scenario")), d.get("tag")),
            "并发": d.get("concurrency"), "时长秒": d.get("actual_duration_s"),
            "思考毫秒": d.get("think_ms"), "每请求新会话": bool(d.get("unique_sessions")),
            "总请求": d.get("total"), "成功": d.get("success"), "失败": d.get("failed"),
            "限流429": d.get("rate_limited"), "5xx": d.get("http_5xx"),
            "超时": d.get("timeouts"), "连接错误": d.get("connection_errors"),
            "吞吐RPS": d.get("rps"),
            "P50": _get(d, "latency_ms", "p50"), "P90": _get(d, "latency_ms", "p90"),
            "P95": _get(d, "latency_ms", "p95"), "P99": _get(d, "latency_ms", "p99"),
            "最大延迟": _get(d, "latency_ms", "max"), "平均延迟": _get(d, "latency_ms", "mean"),
            "问答P95": _get(d, "qa_latency_ms", "p95"), "问答样本": _get(d, "qa_latency_ms", "n"),
            "内存首窗MB": mem.get("first_window_mean_mb"), "内存末窗MB": mem.get("last_window_mean_mb"),
            "内存增长%": mem.get("growth_pct"), "内存斜率MB每分": mem.get("slope_mb_per_min"),
            "内存趋势": mem.get("trend"),
            "熔断": d.get("circuit_breaker"),
            "文件": d.get("_file"),
        })
    return out


def compare_rows(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 (场景, 口径类别) 配对「优化前 / 优化后」。"""
    def key_of(row: dict[str, Any]) -> tuple:
        tag = str(row["标签"])
        if "soak" in tag:
            base = "soak"
        elif "多用户" in tag:
            base = "A-多用户"
        elif "思考200ms" in tag:
            base = "A-思考200"
        elif "冒烟" in tag:
            base = "冒烟"
        else:
            base = f"{row['场景']}-主口径"
        return (base,)

    groups: dict[tuple, dict[str, Any]] = {}
    for row in table:
        k = key_of(row)
        slot = "优化后" if str(row["标签"]).startswith("优化后") or str(row["标签"]).startswith("soak_优化后") else "优化前"
        groups.setdefault(k, {})[slot] = row

    out: list[dict[str, Any]] = []
    for k, pair in sorted(groups.items()):
        before, after = pair.get("优化前"), pair.get("优化后")
        if before is None or after is None:
            continue
        out.append({"口径": before.get("口径") or after.get("口径"),
                    "优化前": before, "优化后": after})
    return out


def write_outputs(table: list[dict[str, Any]]) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    before = [r for r in table if not (str(r["标签"]).startswith("优化后") or str(r["标签"]).startswith("soak_优化后"))]
    after = [r for r in table if (str(r["标签"]).startswith("优化后") or str(r["标签"]).startswith("soak_优化后"))]
    (_OUT / "基准_优化前.json").write_text(
        json.dumps({"work_order": WORK_ORDER, "runs": before}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (_OUT / "优化后.json").write_text(
        json.dumps({"work_order": WORK_ORDER, "runs": after}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    if table:
        fields = list(table[0].keys())
        with (_OUT / "对比.csv").open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)

    pairs = compare_rows(table)
    lines = [f"# 工单17 优化前后对比", "",
             f"工单编号：`{WORK_ORDER}`", "",
             "> 所有数字均来自 `测试/测试数据/*_summary.json`（压测器实测写入），**未经人工修改**。",
             "> 达标判定阈值见 `设计/验收标准.md`：场景A P95 ≤ 3000 ms 且失败 0；场景B 问答 P95 ≤ 5000 ms；",
             "> 内存增长 ≤ 10%。", ""]
    if not pairs:
        lines += ["> ⚠️ 尚未形成完整的「优化前 / 优化后」配对（缺一侧数据）。", ""]
    for pair in pairs:
        b, a = pair["优化前"], pair["优化后"]
        lines += [f"## {pair['口径']}", "",
                  "| 指标 | 优化前 | 优化后 | 变化 | 验收线 | 判定 |", "| --- | --- | --- | --- | --- | --- |",
                  _row("总请求数", b["总请求"], a["总请求"]),
                  _row("失败数", b["失败"], a["失败"], target="= 0"),
                  _row("限流 429", b["限流429"], a["限流429"]),
                  _row("吞吐 RPS", b["吞吐RPS"], a["吞吐RPS"], higher_better=True),
                  _row("P50 (ms)", b["P50"], a["P50"], higher_better=False),
                  _row("P90 (ms)", b["P90"], a["P90"], higher_better=False),
                  _row("**P95 (ms)**", b["P95"], a["P95"], higher_better=False,
                       target=(f"≤ {a['问答P95'] and 3000}" if b["场景"] == "A" else "≤ 5000")),
                  _row("P99 (ms)", b["P99"], a["P99"], higher_better=False),
                  _row("最大延迟 (ms)", b["最大延迟"], a["最大延迟"], higher_better=False),
                  _row("问答 P95 (ms)", b["问答P95"], a["问答P95"], higher_better=False,
                       target="≤ 5000（场景B）" if b["场景"] == "B" else None),
                  _row("内存增长 (%)", b["内存增长%"], a["内存增长%"], higher_better=False, target="≤ 10"),
                  _row("内存斜率 (MB/min)", b["内存斜率MB每分"], a["内存斜率MB每分"], higher_better=False),
                  ""]
    (_OUT / "对比.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(name: str, before: Any, after: Any, *, higher_better: bool | None = None,
         target: str | None = None) -> str:
    change = ""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
        delta = (after - before) / abs(before) * 100.0
        arrow = "↓" if after < before else ("↑" if after > before else "=")
        change = f"{arrow} {delta:+.1f}%"
    verdict = ""
    if higher_better is not None and isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if higher_better:
            verdict = "改善" if after > before else ("持平" if after == before else "变差")
        else:
            verdict = "改善" if after < before else ("持平" if after == before else "变差")
    return f"| {name} | {before} | {after} | {change} | {target or '—'} | {verdict or '—'} |"


def print_table(table: list[dict[str, Any]]) -> None:
    print("=" * 110)
    print(f"{'标签':<22}{'场景':<4}{'并发':>4}{'时长':>7}{'总数':>8}{'失败':>6}{'RPS':>9}"
          f"{'P50':>9}{'P95':>9}{'P99':>9}{'内存增长%':>10}")
    print("-" * 110)
    for r in table:
        print(f"{str(r['标签']):<22}{str(r['场景']):<4}{r['并发']:>4}{r['时长秒']:>7}"
              f"{r['总请求']:>8}{r['失败']:>6}{r['吞吐RPS']:>9}{r['P50']:>9}{r['P95']:>9}"
              f"{r['P99']:>9}{str(r['内存增长%']):>10}")
    print("=" * 110)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 结果汇总与前后对比")
    ap.add_argument("--outdir", default=str(_OUT))
    args = ap.parse_args(argv)
    rows = load_summaries()
    if not rows:
        print(f"[错误] {_DATA} 下没有 *_summary.json，先跑压测")
        return 2
    table = build_table(rows)
    print_table(table)
    write_outputs(table)
    print(f"\n已写出：\n  {_OUT / '基准_优化前.json'}\n  {_OUT / '优化后.json'}\n"
          f"  {_OUT / '对比.csv'}\n  {_OUT / '对比.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
