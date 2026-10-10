# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""从评估结果生成 16 题答案定位报告（`docs/答案定位报告.md`）。

用法：
    python scripts/gen_location_report.py
    python scripts/gen_location_report.py --src docs/reports/eval_full_04.json \\
                                         --out docs/答案定位报告.md

报告口径：逐题给出「问题 → 标准答案要点 → 证据出处（文档 + 页码 + block_type
+ 来源ID）→ 实际召回页码 → 答案与要点覆盖率」，并在开头给出 16 题总览表。

数据纪律：本脚本只搬运评估结果里已有的数字，缺失即写「不适用」或显式标注
待回填，绝不估算、不补零、不美化。评估结果 JSON 不存在时仍产出题册骨架
（题面 + 金标出处），供人工对照。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag04.config import get_settings                       # noqa: E402
from rag04.eval.questions import QUESTIONS, EvalQuestion    # noqa: E402

DEFAULT_SRC_NAME = "eval_full_04.json"
DEFAULT_OUT_NAME = "答案定位报告.md"

# 无评估结果时的显式标记（可 grep：`待回填`）。绝不拿空值冒充成绩。
PLACEHOLDER_NO_RESULT = (
    "> 待回填（评估结果 JSON 缺失：请先运行 "
    "`python scripts/run_eval.py full_04` 再重新生成）"
)

_NA = "不适用"


def _fmt(value: Any) -> str:
    """空值一律渲染为「不适用」，不用 0 或空串冒充。"""
    return _NA if value is None else str(value)


def _rel(path: Path) -> str:
    """报告里记录来源路径：项目内用相对路径，项目外用绝对路径（POSIX 风格）。"""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _rows_by_qid(result: dict | None) -> dict[int, dict]:
    """把评估结果的逐题行按 qid 建索引；结果缺失时返回空表。"""
    if not result:
        return {}
    out: dict[int, dict] = {}
    for row in result.get("rows") or []:
        try:
            out[int(row.get("qid"))] = row
        except (TypeError, ValueError):
            continue
    return out


def _pages_text(pages: Sequence[int] | None) -> str:
    """页码列表 → 「39」/「21、33、157」。"""
    if not pages:
        return "—"
    return "、".join(str(p) for p in pages)


def _verdict(row: dict) -> str:
    """判定文案：拒答优先（RC6 口径：拒答不得判对）。"""
    if row.get("refused"):
        return "🚫 拒答（不计正确）"
    correct = row.get("correct")
    if correct is None:
        return "不适用"
    return "✅ 正确" if correct else "❌ 不正确"


def _cite_table(citations: Sequence[dict] | None) -> list[str]:
    """引用来源表（最多 5 条，与精确度报告同口径）。"""
    lines = [
        "**引用来源**：",
        "",
        "| 文档 | 页码 | 类型 | 来源ID | 相似度 |",
        "| --- | --- | --- | --- | --- |",
    ]
    cites = list(citations or [])[:5]
    if not cites:
        lines.append("| — | — | — | — | — |")
    for c in cites:
        lines.append(
            f"| {c.get('doc_id', '—')} | {c.get('page', '—')} "
            f"| {c.get('block_type', '—')} | {c.get('source_id', '—')} "
            f"| {c.get('score', '—')} |"
        )
    lines.append("")
    return lines


def _evidence_lines(q: EvalQuestion, row: dict) -> list[str]:
    """证据出处：金标（题册标注）+ 真实命中该页的引用块（含来源ID 与图像路径）。"""
    out = [
        f"**证据出处（金标）**：`{q.doc_id}` 第 {_pages_text(q.gold_pages)} 页"
        f"（block_type: `{q.block_type}`）",
        "",
    ]
    golds = set(q.gold_pages or [])
    hits = [c for c in (row.get("citations") or []) if c.get("page") in golds]
    if hits:
        c = hits[0]
        extra = f"，相似度 {c.get('score')}"
        if c.get("image_path"):
            extra += f"，图像 `{Path(str(c['image_path'])).name}`"
        out += [
            f"**命中金标页的引用块**：`{c.get('source_id')}`"
            f"（block_type: `{c.get('block_type')}`，第 {c.get('page')} 页{extra}）",
            "",
        ]
    elif row:
        out += ["**命中金标页的引用块**：无（top-5 引用中不含金标页）", ""]
    return out


def _question_section(q: EvalQuestion, row: dict) -> list[str]:
    """单题小节：题面 → 要点 → 证据 → （有评估结果时）答案与判定。"""
    lines = [
        f"## id {q.qid}",
        "",
        f"**问题**：{q.question}",
        "",
        f"**英文**：{q.question_en}",
        "",
        f"**标准答案要点**：{'、'.join(q.answer_key)}",
        "",
        *(_evidence_lines(q, row)),
    ]
    if q.note:
        lines += [f"**定位说明**：{q.note}", ""]
    if not row:
        return lines

    lines += [
        f"**实际召回页码**：{_pages_text(row.get('retrieved_pages'))}"
        f"（计分页：{_pages_text(row.get('scored_pages'))}）",
        "",
        f"**答案**：{row.get('answer') or '（无）'}",
        "",
        f"**要点覆盖率**：{_fmt(row.get('coverage'))}　"
        f"**判定**：{_verdict(row)}　**用时**：{_fmt(row.get('latency_ms'))} ms　"
        f"**后端**：{_fmt(row.get('llm_backend'))}",
        "",
        *(_cite_table(row.get("citations"))),
    ]
    if row.get("error"):
        lines += [f"> ⚠️ 错误：{row['error']}", ""]
    return lines


def _overview(rows: dict[int, dict]) -> list[str]:
    """16 题总览表（无评估结果时不产出，避免整表都是占位符）。"""
    if not rows:
        return []
    lines = [
        "## 一、16 题总览",
        "",
        "| id | 题型 | 出处文档 | 出处页 | 命中金标页 | 召回页码 | 覆盖率 | 判定 | 用时(ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for q in QUESTIONS:
        row = rows.get(q.qid, {})
        gold = set(q.gold_pages or [])
        hit = next((c.get("page") for c in (row.get("citations") or [])
                    if c.get("page") in gold), None)
        lines.append(
            f"| {q.qid} | {q.block_type} | {q.doc_id} | {_pages_text(q.gold_pages)} "
            f"| {_fmt(hit)} | {_pages_text(row.get('retrieved_pages'))} "
            f"| {_fmt(row.get('coverage')) if row else '—'} "
            f"| {_verdict(row) if row else '—'} "
            f"| {_fmt(row.get('latency_ms')) if row else '—'} |"
        )
    lines.append("")
    return lines


def build_report(result: dict | None, questions: Sequence[EvalQuestion] = QUESTIONS,
                 source: str = "") -> str:
    """把评估结果渲染成答案定位报告 Markdown。

    ``result`` 为 `rag04.eval.runner.run_eval` 的返回值（或同形状字典）；
    传入 None 时只产出题面与金标出处的题册骨架，并标注待回填。
    """
    rows = _rows_by_qid(result)
    metrics = (result or {}).get("metrics") or {}

    lines = [
        "# 答案定位报告",
        "",
        "> 工单编号：人工智能NLP-RAG-图像内容解析及检索优化",
        "",
        "本报告逐题给出：问题 → 标准答案要点 → 证据出处"
        "（文档 + 页码 + block_type + 来源ID）→ 实际召回页码 → 答案与要点覆盖率。",
        "",
    ]
    if source:
        lines.append(f"- 数据来源：`{source}`")
    if result:
        lines.append(
            f"- 运行模式：`{result.get('mode', '—')}`　"
            f"语言：`{result.get('lang', '—')}`　"
            f"题目数：{result.get('n_questions', '—')}"
        )
        lines.append(
            f"- 总体指标：answer_accuracy {_fmt(metrics.get('answer_accuracy'))}　"
            f"hit_rate {_fmt(metrics.get('hit_rate'))}　"
            f"mrr {_fmt(metrics.get('mrr'))}　"
            f"recall@5 {_fmt(metrics.get('recall@5'))}　"
            f"latency_p50 {_fmt(metrics.get('latency_p50_ms'))} ms"
        )
    lines += [
        "- 复现：`python scripts/gen_location_report.py [--src 评估结果.json]"
        " [--out 输出.md]`（重跑评估后执行本命令即可刷新本报告）",
        "",
    ]
    if not result:
        lines += [
            PLACEHOLDER_NO_RESULT,
            "",
            "本次逐题只列出题面与金标出处；评估完成后重新生成本报告即可补齐"
            "召回页码、答案与判定。",
            "",
        ]
    elif not rows:
        lines += [
            PLACEHOLDER_NO_RESULT,
            "",
            "评估结果里没有任何逐题行（rows 为空），原因需查评估日志，"
            "此处不虚构数据。",
            "",
        ]

    lines += _overview(rows)
    for q in questions:
        lines += _question_section(q, rows.get(q.qid, {}))
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """命令行参数：--src 指定评估 JSON，--out 指定输出路径。"""
    s = get_settings()
    p = argparse.ArgumentParser(description="生成 16 题答案定位报告")
    p.add_argument("--src", default=str(Path(s.reports_dir) / DEFAULT_SRC_NAME),
                   help="评估结果 JSON（默认 docs/reports/eval_full_04.json）")
    p.add_argument("--out", default=str(Path(s.project_root) / "docs" / DEFAULT_OUT_NAME),
                   help="输出 Markdown 路径")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """读取评估 JSON → 生成 `docs/答案定位报告.md`。"""
    args = parse_args(argv)
    src = Path(args.src)
    result: dict | None = None
    if src.exists():
        try:
            result = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"警告：评估结果读取失败（{type(e).__name__}: {e}），"
                  f"本次只产出题册骨架", file=sys.stderr)
    else:
        print(f"警告：未找到评估结果 {src}，本次只产出题册骨架", file=sys.stderr)

    text = build_report(result, source=_rel(src))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"已生成：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
