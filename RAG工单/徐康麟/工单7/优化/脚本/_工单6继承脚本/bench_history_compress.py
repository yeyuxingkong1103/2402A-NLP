# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

对话历史压缩基准（``bench_history_compress``）—— 对合成的一批长历史（10 / 30 / 100 轮假
``TurnRecord``）跑 ``compress_history``，测**耗时与压缩比**，并给出不同摘预算下的结果。

## 口径
* ``token 近似数`` = ``len(text)``（中文 ≈ 1 字 1 token，见 ``history_compress.approx_tokens``）；
* ``压缩比`` = 压缩后字符数 ÷ 原始字符数（越小越省）；``节省比例`` = 1 − 压缩比；
  **只统计送给 LLM 的上下文**（摘要 + 保留的 K 轮原文），不含未发送的轮次之外的任何内容；
* 原始长度 = **全部 N 轮**的 question + rewritten_query + answer 字符数之和（即「不压缩会话」
  需要携带的上下文量）；
* 每档跑 ``--repeat`` 次取 P50/最大，避免单次抖动被当成结论；
* 合成历史全部由本脚本构造（``focus_entity`` 与 ``slot`` 按轮轮换），**不含真实用户数据**。

运行：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/bench_history_compress.py

产物：
* ``优化/评估结果/历史压缩基准.json``
* ``优化/评估结果/历史压缩基准.md``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发"))

from app.core.dialogue_state import TurnRecord              # noqa: E402
from app.core.history_compress import compress_history       # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"

COMPANIES: list[str] = [
    "武汉兴图新科电子股份有限公司",
    "武汉力源信息技术股份有限公司",
]
SLOTS: list[str] = ["法定代表人", "注册资本", "主营业务", "营业收入", "军用领域", "销售部门"]


def percentile(values: list[float], q: float) -> float:
    """最近序法百分位（与工单4 ``bench_cache.py`` 同口径）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return round(ordered[idx], 3)


def build_history(n: int) -> list[TurnRecord]:
    """构造 n 轮合成历史（公司/槽位轮换，贴近真实多轮形态）。"""
    turns: list[TurnRecord] = []
    for i in range(n):
        company = COMPANIES[i % len(COMPANIES)]
        slot = SLOTS[i % len(SLOTS)]
        question = f"{company}的{slot}是多少？"
        turns.append(TurnRecord(
            question=question,
            rewritten_query=question,
            answer=(f"根据招股说明书，{company}的{slot}为第 {i + 1} 项数据；"
                    f"报告期内该项指标较上期变动 {i % 7 + 1}.{i % 9}%。"),
            focus_entity=company,
            slot=slot,
            strategy=("none" if i % 3 == 0 else "pronoun"),
            cited_pages=[10 + i],
        ))
    return turns


def raw_chars(turns: list[TurnRecord]) -> int:
    """不压缩时需要携带的上下文字符数（全部轮次的问句 + 改写问句 + 答案）。"""
    return sum(len(t.question) + len(t.rewritten_query) + len(t.answer) for t in turns)


def measure(n: int, *, keep_recent: int, max_chars: int, repeat: int) -> dict[str, Any]:
    turns = build_history(n)
    latencies: list[float] = []
    result = None
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        result = compress_history(turns, keep_recent=keep_recent, max_chars=max_chars)
        latencies.append((time.perf_counter() - started) * 1000)
    assert result is not None
    last_kept_question = result.kept_turns[-1]["question"] if result.kept_turns else ""
    kept_chars = sum(len(t["question"]) + len(t["rewritten_query"]) + len(t["answer"])
                     for t in result.kept_turns)
    compressed_chars = len(result.digest) + kept_chars
    original_chars = raw_chars(turns)
    return {
        "轮数": n,
        "keep_recent": keep_recent,
        "max_chars": max_chars,
        "dropped_turns": result.dropped_turns,
        "kept_turns": len(result.kept_turns),
        "digest_chars": len(result.digest),
        "digest_tokens_approx": len(result.digest),
        "kept_chars": kept_chars,
        "compressed_chars": compressed_chars,
        "original_chars": original_chars,
        "compressed_tokens_approx": compressed_chars,
        "original_tokens_approx": original_chars,
        "字符压缩比": round(compressed_chars / original_chars, 4) if original_chars else 0.0,
        "token压缩比": round(compressed_chars / original_chars, 4) if original_chars else 0.0,
        "节省字符": original_chars - compressed_chars,
        "耗时_ms_p50": percentile(latencies, 0.5),
        "耗时_ms_mean": round(statistics.fmean(latencies), 3),
        "耗时_ms_max": round(max(latencies), 3),
        "repeat": max(1, repeat),
        "digest": result.digest,
        "focus_entity": result.focus_entity,
        "保留轮数上限被遵守": len(result.kept_turns) == min(keep_recent, n),
        "硬上限被遵守": len(result.digest) <= max_chars,
        "保留的是最近一轮": last_kept_question == (turns[-1].question if turns else ""),
        "摘要含公司名": all(c in result.digest for c in COMPANIES) if result.digest else None,
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 对话历史压缩基准（工单5 · 人工智能NLP-RAG-Query 理解优化）")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("* `token 近似数 = len(text)`（中文 ≈ 1 字 1 token，见 `history_compress.approx_tokens`）；")
    lines.append("* 原始长度 = **全部 N 轮**（question + rewritten_query + answer）字符数之和"
                 "，即「不压缩会话」要携带的上下文量；")
    lines.append("* 压缩后长度 = 摘要 digest + 保留的 K 轮原文（**送进 LLM 的全部内容**）；")
    lines.append(f"* 历史为**合成数据**（{len(COMPANIES)} 家公司 × {len(SLOTS)} 个字段槽位轮换），"
                 "不含真实用户数据；")
    lines.append(f"* 每档跑 `--repeat {payload['repeat']}` 次取 P50 / 最大，避免单次抖动当结论。")
    lines.append("")
    lines.append("## 主表（keep_recent=3，max_chars=240）")
    lines.append("")
    lines.append("| 轮数 | 丢弃轮 | 保留轮 | 原始字符 | 压缩后字符 | 字符压缩比 | 节省 | 摘要耗时 P50 (ms) | 最大 (ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in payload["主表"]:
        lines.append(
            f"| {item['轮数']} | {item['dropped_turns']} | {item['kept_turns']} | "
            f"{item['original_chars']} | {item['compressed_chars']} | "
            f"{item['字符压缩比']:.4f}（省 {1 - item['字符压缩比']:.2%}） | "
            f"{item['节省字符']} 字符 | {item['耗时_ms_p50']} | {item['耗时_ms_max']} |"
        )
    lines.append("")
    lines.append("## 摘要预算扫描（100 轮，max_chars 从 60 到 600）")
    lines.append("")
    lines.append("| max_chars | 摘要字符 | 硬上限 | 压缩后字符 | 字符压缩比 | 摘要耗时 P50 (ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in payload["预算扫描"]:
        lines.append(f"| {item['max_chars']} | {item['digest_chars']} | "
                     f"{'✓' if item['硬上限被遵守'] else '✗'} | {item['compressed_chars']} | "
                     f"{item['字符压缩比']:.4f} | {item['耗时_ms_p50']} |")
    lines.append("")
    lines.append("## 规模稳定性（耗时随轮数增长）")
    lines.append("")
    lines.append("| 轮数 | 摘要耗时 P50 (ms) | 平均 (ms) | 每轮均摊 (ms) |")
    lines.append("| --- | --- | --- | --- |")
    for item in payload["主表"]:
        per_turn = round(item["耗时_ms_mean"] / item["轮数"], 4) if item["轮数"] else 0.0
        lines.append(f"| {item['轮数']} | {item['耗时_ms_p50']} | {item['耗时_ms_mean']} | {per_turn} |")
    lines.append("")
    lines.append("## 摘要样例（原文子串拼接，不含任何编造内容）")
    lines.append("")
    for item in payload["主表"]:
        lines.append(f"* **{item['轮数']} 轮 → digest（{item['digest_chars']} 字符）**：")
        lines.append(f"  > {item['digest']}")
    lines.append("")
    lines.append("## 未完成项 / 边界")
    lines.append("")
    lines.append("* 本基准只测 **`compress_history` 本身**：`MultiTurnEngine` 尚未调用它"
                 "（接线由 captain 负责），故「长对话端到端延迟下降」**未运行**、不做估算。")
    lines.append("* 合成历史的答案长度偏短（约 50 字/轮），真实招股书答案更长；"
                 "压缩比由「保留 3 轮原文」主导，结论对答案长度不敏感，但绝对字符数会同步放大。")
    lines.append("* token 数按字符数近似，未接真实分词器；如需精确 token 请另跑 jieba/模型分词。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对话历史压缩基准（工单5）")
    parser.add_argument("--out-dir", default=str(ROOT / "优化" / "评估结果"))
    parser.add_argument("--sizes", default="10,30,100", help="轮数档位（逗号分隔）")
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args(argv)

    sizes = [int(x) for x in str(args.sizes).split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_table = [measure(n, keep_recent=args.keep_recent, max_chars=args.max_chars,
                          repeat=args.repeat) for n in sizes]
    budgets = [60, 120, 240, 360, 600]
    budget_scan = [measure(sizes[-1], keep_recent=args.keep_recent, max_chars=b,
                           repeat=args.repeat) for b in budgets]

    payload = {
        "work_order": WORK_ORDER,
        "protocol": ("合成历史（TurnRecord）+ compress_history；每档 repeat 次取 P50/最大；"
                     "token 近似 = len(text)"),
        "repeat": args.repeat,
        "keep_recent": args.keep_recent,
        "max_chars": args.max_chars,
        "sizes": sizes,
        "主表": main_table,
        "预算扫描": budget_scan,
        "notes": [
            "合成数据，不含真实用户数据",
            "压缩比 = (摘要 + 保留 K 轮原文) ÷ 全部 N 轮上下文",
            "MultiTurnEngine 尚未调用 compress_history，端到端收益未运行",
            "任何数字均为实跑产物",
        ],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = out_dir / "历史压缩基准.json"
    md_path = out_dir / "历史压缩基准.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")

    print(f"[历史压缩基准] keep_recent={args.keep_recent} max_chars={args.max_chars} "
          f"repeat={args.repeat}")
    for item in main_table:
        print(f"  {item['轮数']:>3} 轮：原始 {item['original_chars']} 字符 → 压缩后 "
              f"{item['compressed_chars']} 字符（比 {item['字符压缩比']:.4f}，省 "
              f"{1 - item['字符压缩比']:.2%}）；摘要 P50 {item['耗时_ms_p50']} ms / "
              f"max {item['耗时_ms_max']} ms；丢弃 {item['dropped_turns']} 轮，保留 "
              f"{item['kept_turns']} 轮")
    print("  预算扫描（100 轮）：" + "；".join(
        f"max_chars={b['max_chars']}→摘要 {b['digest_chars']} 字符/比 {b['字符压缩比']:.4f}"
        for b in budget_scan))
    print(f"  硬上限全部被遵守：{all(b['硬上限被遵守'] for b in main_table + budget_scan)}")
    print(f"  产物 → {json_path} / {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
