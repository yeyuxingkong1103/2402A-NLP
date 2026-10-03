# -*- coding: utf-8 -*-
"""优化前后对比评估：加载 T6 基线 → 真实运行优化后系统 → 输出 CSV/Markdown/JSON 报告。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 脚本（T7 产出①）

功能
    ① 加载 T6 固化基线：`优化/基线/baseline_results.json`（判分/耗时）、
       `baseline_repro_retrieval.json`（最终上下文命中与位次）、
       `baseline_retrieval.json`（纯向量命中率）；
    ② **真实运行**优化后系统：混合检索 + 生成（默认交付路径 mode=rag），
       可选附带纯抽取式路径（`--paths both`）；
    ③ 对 10 个工单问题计算：答案准确率（沿用工单1 确定性 `check_answer`，阈值 0.62 不放宽）、
       检索命中率（最终上下文是否含证据块）、纯向量 top-5/top-20 命中率、
       首字响应（流式首块计时，**预热后统计**）、引用正确率；
    ④ 输出 `优化/评估结果/optimization_compare.csv`、`.md`、`accuracy_report.json`，
       并把「RAGAS 未运行」声明**追加**进 `优化/评估结果/ragas_report.md`（不覆盖 T2 已有内容）。

口径纪律（为避免与前序工作冲突，逐条声明）
    1. 证据定位**复用 T2 的 `研发/scripts/selftest_retrieval.py::locate_evidence`**，
       不另立判据，保证「检索命中率」与 T2 自测同口径；Q95/Q207 用其既有替代定位规则。
    2. 基线的「最终上下文命中」取自 T6 `baseline_repro_retrieval.json`（8 条上下文，含 Q95/Q207 替代判据）；
       优化后的「最终上下文命中」由本次真实运行得到（5 条上下文）。两者同为"证据是否进入终果"。
    3. 「证据位次」按两边各自结果列表中最靠前的证据块计算；未命中记 `null`。
    4. 首字延迟**先预热再计时**（与本机 serve/evaluate 一致），并把冷启动首题数值另行记录，不混算。
    5. 优化后数值全部来自本次真实运行；**禁止**手填、**禁止**把基线值抄成优化后值。

用法（工作目录 = E:\\gao6gongdan\\工单2）::

    pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py
    pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py --paths rag --limit 3

## 防覆盖保护（2026-10-03 加固，工单2 t5）

**背景（真实失效）**：本脚本每次运行都会**无条件重写**四件正式交付物
（`optimization_compare.md`/`.csv`、`accuracy_report.json`、`ragas_report.md`）。
曾有同事把 `--limit 1` 当冒烟运行使用，于是四个产物在同秒被 **1 题运行覆盖**：
`after.count` 变成 1、报告里「10 题」与「1/1」并存且互相矛盾，
验收 7「重跑数值与报告一致」在现文件上必然对不上。

**双保险（两条同时生效）**：

1. **部分运行只写临时目录**：`0 < --limit < 全库题数` 时，输出目录强制切到
   ``PARTIAL_OUT_ROOT/<limitN>``（=`优化/评估结果/验证留痕/部分运行/limitN`），
   **绝不写四件正式产物所在路径**（`优化/评估结果/` 顶层）；若显式 `--out` 指向正式交付目录则**直接拒绝执行**。
   为什么不用 `优化/评估结果/_smoke/`：`测试/离线/test_deliverable_integrity.py` 判据 4
   明令交付目录不得出现**下划线前缀**文件/目录，写在那儿会把守卫判红（`验证留痕/部分运行/` 无下划线前缀，故安全）。
   为什么不用工作区根部的 `.tmp_review/`：根目录不得残留临时件（验收 11 人工判据），放交付树内可随交付物一并审计。
2. **写正式路径前断言题数**：写 `优化/评估结果/` 之前断言
   ``after["count"] == golden 全库题数``；不满足则打印错误并 `sys.exit(4)`——
   **宁可不产出，也不产出错的**。（`--allow-partial` 可显式豁免第 1 条，但第 2 条仍然生效。）

退出码：0 成功；2 前置产物缺失/索引未就绪；3 参数错误；4 覆盖保护拒绝写入正式交付路径。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "研发"
for extra in (SOURCE_ROOT, SOURCE_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from app.core.chunker import Chunker  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.embedder import get_embedder  # noqa: E402
from app.core.errors import RAGError  # noqa: E402
from app.core.evaluator import get_evaluator  # noqa: E402
from app.core.logging_conf import flush_logs, logger, setup_logging  # noqa: E402
from app.core.qa_engine import QAEngine  # noqa: E402
from app.core.query_understanding import get_query_understanding  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402
from app.models.schemas import GoldenQA  # noqa: E402
import selftest_retrieval as sst  # noqa: E402  （复用 T2 的证据定位口径）

BASELINE_DIR = REPO_ROOT / "优化" / "基线"
OUT_DIR = REPO_ROOT / "优化" / "评估结果"
#: 部分运行（``--limit``）的强制输出根目录。
#: 为什么放这里（而不是工作区根部的 ``.tmp_review/``）：
#: ① ``优化/评估结果/`` 与 ``优化/脚本/`` 由交付完整性守卫禁止出现**下划线前缀**文件/目录，
#:   而 ``验证留痕/部分运行/`` 全路径无下划线前缀 → 守卫安全；
#: ② 工作区**根目录**不得残留临时件（验收 11 人工判据），放在交付树内可随交付物一并审计。
PARTIAL_OUT_ROOT = OUT_DIR / "验证留痕" / "部分运行"
RESULTS_JSON = BASELINE_DIR / "baseline_results.json"
REPRO_JSON = BASELINE_DIR / "baseline_repro_retrieval.json"
RETRIEVAL_JSON = BASELINE_DIR / "baseline_retrieval.json"
RAGAS_DOC = OUT_DIR / "ragas_report.md"

RAGAS_DECLARATION_MARKER = "<!-- T7-RAGAS-DECLARATION -->"

#: 证据串本身不可定位的题目（Q95 含「……」、Q207 为合成引用文本），使用替代判据；
#: 严格口径（strict）会把这批题剔除后再算「证据排第 1 位」比例。
SURROGATE_QIDS: tuple[int, ...] = (95, 207)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_golden(path: Path) -> list[GoldenQA]:
    items: list[GoldenQA] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(GoldenQA(**json.loads(line)))
    return items


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数（与 evaluator.summarize 同法，便于口径一致）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * frac, 2)


CITATION_TAG_RE = re.compile(r"\[页码[:：]\s*\d+\]")


def answer_length_stats(answers: list[str]) -> dict:
    """答案长度统计（去掉引用标签 `[页码: N]` 与首尾空白后计字符数）。

    用途：为「生成层：多值完整性」提供可量化旁证——基线 LLM 把答案压得过短会丢掉多值。
    """
    lengths = [len(CITATION_TAG_RE.sub("", a or "").strip()) for a in answers if a]
    if not lengths:
        return {"avg": 0.0, "min": 0, "max": 0}
    return {
        "avg": round(sum(lengths) / len(lengths), 1),
        "min": min(lengths),
        "max": max(lengths),
    }


def best_rank(chunk_ids: list[str], evidence_ids: set[str]) -> int | None:
    """证据集合在结果列表中的最靠前名次（1 起）；未命中返回 None。"""
    if not evidence_ids:
        return None
    for index, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id in evidence_ids:
            return index
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="工单2 优化前后对比评估")
    parser.add_argument(
        "--paths",
        default="both",
        choices=["rag", "extractive", "both"],
        help="运行哪条生成路径：rag=最终交付路径；extractive=纯抽取式；both=两条都跑（默认）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评估前 N 题（0=全部）。**部分运行只写临时目录**（优化/评估结果/验证留痕/部分运行/limitN），绝不写正式交付路径",
    )
    parser.add_argument("--out", default="", help="输出目录（默认 优化/评估结果；部分运行时默认改到临时目录）")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="显式允许部分运行写 --out/正式路径（默认禁止：防 1 题冒烟覆盖 10 题交付物）",
    )
    parser.add_argument("--skip-warmup", action="store_true", help="跳过预热（首字会含冷启动，仅排障用）")
    return parser


def run_path(
    engine: QAEngine,
    golden_items: list[GoldenQA],
    evidence_map: dict[int, set[str]],
    mode_label: str,
) -> list[dict]:
    """跑一条生成路径，返回逐题原始结果（答案/耗时/引用/检索位次）。"""
    evaluator = get_evaluator(engine._store)  # noqa: SLF001（与 evaluate.py 一致，复用同一连接）
    rows: list[dict] = []
    for index, golden in enumerate(golden_items, start=1):
        evidence_ids = evidence_map.get(golden.id, set())
        try:
            answer = engine.ask(golden.question)
        except RAGError as exc:
            logger.exception("scripts.compare_optimization", "单题问答业务失败", question_id=golden.id, code=exc.code)
            rows.append(
                {
                    "question_id": golden.id,
                    "question": golden.question,
                    "golden": golden.answer,
                    "error": f"{exc.code}: {exc.message}",
                }
            )
            continue
        record = evaluator.evaluate_answer(golden, answer, mode=mode_label)
        _ok, judge_reason = evaluator.check_answer(record.answer, golden.answer)
        retrieved_ids = [item.chunk.chunk_id for item in (answer.retrieved or [])]
        retrieved_pages = [item.chunk.page for item in (answer.retrieved or [])]
        rows.append(
            {
                "question_id": golden.id,
                "question": golden.question,
                "golden": golden.answer,
                "answer": record.answer,
                "is_correct": bool(record.is_correct),
                "judge_reason": judge_reason,
                "mode": answer.mode,
                "is_unknown": bool(answer.is_unknown),
                "unknown_reason": answer.unknown_reason,
                "first_token_ms": round(float(record.first_token_ms), 2),
                "total_ms": round(float(record.total_ms), 2),
                "citation_pages": list(record.citation_pages),
                "citation_valid": bool(record.citation_valid),
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_pages": retrieved_pages,
                "retrieval_rank": best_rank(retrieved_ids, evidence_ids),
                "retrieval_hit": best_rank(retrieved_ids, evidence_ids) is not None,
                "evidence_ids": sorted(evidence_ids),
            }
        )
        mark = "✅" if record.is_correct else "❌"
        print(
            f"  [{mode_label} {index}/{len(golden_items)}] Q{golden.id} {mark} "
            f"首字={record.first_token_ms:.0f}ms 证据位次={rows[-1]['retrieval_rank']} "
            f"引用={record.citation_pages} mode={answer.mode}"
        )
    return rows


def summarize_path(rows: list[dict], total_pages: int, surrogate_ids: tuple[int, ...] = ()) -> dict:
    """对一条路径的逐题结果求汇总指标。surrogate_ids 为走替代判据的题目（用于严格口径剔除）。"""
    ok_rows = [r for r in rows if "error" not in r]
    count = len(ok_rows)
    correct = sum(1 for r in ok_rows if r["is_correct"])
    hits = sum(1 for r in ok_rows if r["retrieval_hit"])
    rank1 = sum(1 for r in ok_rows if r["retrieval_rank"] == 1)
    strict_rows = [r for r in ok_rows if r["question_id"] not in surrogate_ids]
    rank1_strict = sum(1 for r in strict_rows if r["retrieval_rank"] == 1)
    rank2 = sum(1 for r in ok_rows if isinstance(r["retrieval_rank"], int) and r["retrieval_rank"] <= 2)
    ft = [r["first_token_ms"] for r in ok_rows]
    tt = [r["total_ms"] for r in ok_rows]
    citation_total = sum(len(r["citation_pages"]) for r in ok_rows)
    citation_ok = sum(
        1
        for r in ok_rows
        for page in r["citation_pages"]
        if isinstance(page, int) and 1 <= page <= total_pages
    )
    return {
        "count": count,
        "correct": correct,
        "accuracy": round(correct / count, 4) if count else 0.0,
        "retrieval_hit_count": hits,
        "retrieval_hit_rate": round(hits / count, 4) if count else 0.0,
        "evidence_rank1_count": rank1,
        "evidence_rank1_rate": round(rank1 / count, 4) if count else 0.0,
        "evidence_rank1_rate_strict": round(rank1_strict / len(strict_rows), 4) if strict_rows else 0.0,
        "evidence_located_count": sum(1 for r in ok_rows if r.get("evidence_ids")),
        "evidence_top2_count": rank2,
        "evidence_top2_rate": round(rank2 / count, 4) if count else 0.0,
        "first_token_avg_ms": round(sum(ft) / len(ft), 2) if ft else 0.0,
        "first_token_p95_ms": percentile(ft, 0.95),
        "first_token_max_ms": round(max(ft), 2) if ft else 0.0,
        "total_avg_ms": round(sum(tt) / len(tt), 2) if tt else 0.0,
        "citation_total": citation_total,
        "citation_valid": citation_ok,
        "citation_accuracy": round(citation_ok / citation_total, 4) if citation_total else 0.0,
        "context_size": len(ok_rows[0]["retrieved_chunk_ids"]) if ok_rows else 0,
    }


def write_csv(path: Path, baseline: dict, after: dict, rows_base: dict, rows_after: dict, page_range: int) -> None:
    """长表：section,key,question_id,metric,before,after,delta,note（含 before/after 两列指标与逐题明细）。"""
    out: list[dict] = []

    def add(section: str, key: str, metric: str, before, after_v, note: str = "") -> None:
        delta = ""
        try:
            if isinstance(before, (int, float)) and isinstance(after_v, (int, float)):
                delta = round(after_v - before, 4)
        except Exception:
            delta = ""
        out.append(
            {
                "section": section,
                "key": key,
                "question_id": "",
                "metric": metric,
                "before": before,
                "after": after_v,
                "delta": delta,
                "note": note,
            }
        )

    # ---- 汇总指标（before = T6 基线；after = 本次真实运行）----
    add("summary", "ALL", "accuracy", baseline["accuracy"], after["accuracy"], "确定性 check_answer，阈值 0.62 未放宽")
    add("summary", "ALL", "retrieval_hit_rate", baseline["retrieval_hit_rate"], after["retrieval_hit_rate"],
        "最终上下文是否含证据块（基线 8 条上下文 / 优化后 5 条）")
    add("summary", "ALL", "evidence_rank1_rate", baseline["evidence_rank1_rate"], after["evidence_rank1_rate"],
        "证据排在第 1 位的题数比例（含 Q95/Q207 替代判据）")
    add("summary", "ALL", "evidence_rank1_rate_strict", baseline.get("evidence_rank1_rate_strict", ""),
        after.get("evidence_rank1_rate_strict", ""), "严格口径：只算有严格证据块的题（排除替代判据题 Q95/Q207），与 T6 §4.2.1 的基线口径同源")
    add("summary", "ALL", "vector_top5_hit_rate", baseline["vector_top5_hit_rate"], after["vector_top5_hit_rate"],
        "纯向量单路 top-5 命中（嵌入/索引层可比指标）")
    add("summary", "ALL", "vector_top20_hit_rate", baseline["vector_top20_hit_rate"], after["vector_top20_hit_rate"],
        "纯向量单路 top-20 命中")
    add("summary", "ALL", "first_token_avg_ms", baseline["first_token_avg_ms"], after["first_token_avg_ms"],
        "流式首块；优化后为预热后稳态")
    add("summary", "ALL", "first_token_p95_ms", baseline["first_token_p95_ms"], after["first_token_p95_ms"], "")
    add("summary", "ALL", "first_token_max_ms", baseline["first_token_max_ms"], after["first_token_max_ms"], "验收红线 ≤3000ms")
    add("summary", "ALL", "total_avg_ms", baseline["total_avg_ms"], after["total_avg_ms"], "端到端均值")
    add("summary", "ALL", "citation_accuracy", baseline["citation_accuracy"], after["citation_accuracy"],
        "引用页 ∈ [1,%d] 且可回查到真实 chunk" % page_range)
    add("summary", "ALL", "chunk_count", baseline["chunk_count"], after["chunk_count"], "索引分块总数")

    # ---- 逐题明细 ----
    for qid in sorted(rows_after):
        b = rows_base.get(qid, {})
        a = rows_after[qid]
        key = f"Q{qid}"
        add("question", key, "is_correct", int(bool(b.get("is_correct"))), int(bool(a.get("is_correct"))), a["question"][:40])
        add("question", key, "retrieval_rank", b.get("evidence_rank") if b.get("evidence_rank") is not None else "null",
            a.get("retrieval_rank") if a.get("retrieval_rank") is not None else "null", "证据在最终上下文中的位次")
        add("question", key, "retrieval_hit", int(bool(b.get("retrieval_hit"))), int(bool(a.get("retrieval_hit"))),
            "最终上下文是否含证据块")
        add("question", key, "first_token_ms", b.get("first_token_ms"), a.get("first_token_ms"), "")
        add("question", key, "total_ms", b.get("total_ms"), a.get("total_ms"), "")
        add("question", key, "citation_pages", "|".join(str(p) for p in b.get("citation_pages", [])),
            "|".join(str(p) for p in a.get("citation_pages", [])), "")
        add("question", key, "answer", b.get("answer", ""), a.get("answer", ""), "完整答案文本")
        add("question", key, "judge_reason", b.get("judge_reason", ""), a.get("judge_reason", ""), "判分理由")

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["section", "key", "question_id", "metric", "before", "after", "delta", "note"]
        )
        writer.writeheader()
        for row in out:
            writer.writerow(row)


def write_markdown(path: Path, payload: dict, rows_base: dict, after_rows: dict, mode_counter: dict, page_range: int) -> None:
    """生成优化前后对比报告（Markdown，六节）。全部数值来自 payload（真实运行）。"""
    before = payload["before"]
    after = payload["after"]
    env = payload["environment"]["优化后"]
    base_env = payload["environment"]["基线（工单1 记录，摘自 ragas_report.md §0）"]
    vec = payload["vector_only_stage"]
    length_before = payload["answer_length"]["before"]
    length_after = payload["answer_length"]["after"]
    mode_text = "、".join(f"`{k}`×{v}" for k, v in sorted(mode_counter.items()))
    llm_kept = mode_counter.get("llm", 0)

    def row(name: str, b, a, digits: int = 4, judge: str = "") -> str:
        delta = (a - b) if isinstance(b, (int, float)) and isinstance(a, (int, float)) else None
        delta_text = f"{delta:+.{digits}f}" if isinstance(delta, (int, float)) else "—"
        fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)  # noqa: E731
        return f"| {name} | {fmt(b)} | {fmt(a)} | {delta_text} | {judge} |"

    acc_ok = "✅ 达标（≥0.90）" if after["accuracy"] >= 0.9 else "❌ 未达标"
    ft_ok = "✅ 达标（≤3000ms）" if after["first_token_max_ms"] <= 3000 else "❌ 未达标"
    cite_ok = "✅ 保持（1.00）" if after["citation_accuracy"] >= 1.0 else "⚠️ 下降"
    # 未通过/翻正题一律**从逐题记录推导**，不在报告里写死题号（写死会在部分运行或结果变化后失真）。
    failed_ids = [int(item["question_id"]) for item in payload["per_question"] if not item["after"]["is_correct"]]
    flipped_ids = [
        int(item["question_id"])
        for item in payload["per_question"]
        if (not item["before"]["is_correct"]) and item["after"]["is_correct"]
    ]
    # 位次回退题（前 → 后）也由数据推导，替代原先写死的「Q207 第 1 → 第 4」
    regressed = [
        (int(item["question_id"]), item["before"]["retrieval_rank"], item["after"]["retrieval_rank"])
        for item in payload["per_question"]
        if isinstance(item["before"]["retrieval_rank"], int)
        and isinstance(item["after"]["retrieval_rank"], int)
        and item["after"]["retrieval_rank"] > item["before"]["retrieval_rank"]
    ]
    #: 替代判据题在**本次运行内**是否存在（部分运行时可能不包含）
    run_qids = {int(item["question_id"]) for item in payload["per_question"]}
    surrogate_in_run = {95, 207} <= run_qids
    strict_count = len([qid for qid in run_qids if qid not in (95, 207)])

    lines: list[str] = []
    add = lines.append
    add("# 优化前后对比报告（工单2）")
    add("")
    add("> 工单：**人工智能NLP-RAG-基于PDF文档的问答系统优化**")
    add("> 阶段：优化 / 前后对比（T7）")
    add(f"> 生成时间：{payload['meta']['生成时间']}")
    add(f"> 复现命令：`{payload['meta']['复现命令']}`")
    add("")
    add("**★ 本报告口径声明（三种「命中」口径互不混用）**")
    add("")
    add("| 口径 | 定义 | 本报告用途 |")
    add("| --- | --- | --- |")
    add("| 检索命中率（终果含证据） | 最终返回上下文中是否含**证据块** | 主口径之一；基线与优化后同为「证据是否进终果」 |")
    add("| 证据位次 | 证据块在终果中的名次（1=最前，未命中记 null） | 衡量**排序质量**；这是本次检索层的真实增益所在 |")
    add("| 纯向量单路命中率 | 只用向量通道 top-k 是否含证据 | **嵌入/索引层**可比指标，与混合检索分开记账 |")
    add("")
    add(f"> 三种口径的分母都是**本次运行的 {after['count']} 道题**（全库 {payload['meta']['全库题目数']} 道；")
    add("> `--limit` 部分运行只写临时目录、不会覆盖本报告，见 §7 防覆盖保护）。")
    if surrogate_in_run:
        add("> Q95（证据串含「……」）与 Q207（证据串为合成引用文本）")
        add("> 使用替代判据，报告同时给出**严格口径**（剔除该两题）以免高估。")
    add("")
    add("---")
    add("")
    add("## 1. 被测对象与环境")
    add("")
    add("| 维度 | 优化前（工单1 基线） | 优化后（工单2） |")
    add("| --- | --- | --- |")
    add(f"| 生成后端 | {base_env.get('LLM 服务', '—')} | {env.get('LLM 后端')}（{env.get('LLM 模型')}） |")
    add(f"| 生成模型 | {base_env.get('LLM 模型', '—')} | {env.get('LLM 模型')} |")
    add(f"| 嵌入模型 | {base_env.get('嵌入模型', '—')} | {env.get('嵌入模型')}（{env.get('嵌入维度')} 维，{env.get('嵌入后端')}） |")
    add(f"| 重排 | 未生效（基线 `rerank_score` 全为 null） | {env.get('重排模式')}（规则重排；本机无 `{env.get('重排模型')}` 权重） |")
    add(f"| 分块数 | {before['chunk_count']} | {after['chunk_count']} |")
    add(f"| 最终上下文条数 | {before['context_size']} 条 | {after['context_size']} 条 |")
    add(f"| 索引 | 2967×512（bge-small-zh-v1.5） | {env.get('分块数')} 块 × 1024 维 + 子块级 segments |")
    add(f"| 本次运行回答模式分布 | —（基线为 LLM 生成） | {mode_text} |")
    add("")
    add("> **生成端栈不同是已声明的混淆项**：基线为 `Qwen2.5-7B-Instruct-AWQ @ 127.0.0.1:8000/v1`，")
    add("> 工单2 为本机 Ollama `qwen2.5:3b`。因此本报告的增益**不做「模型更强」的归因**（见 §5 归因纪律）。")
    add("")
    add("---")
    add("")
    add("## 2. 优化前后指标对比（全部来自真实运行）")
    add("")
    add("| 指标 | 优化前（基线） | 优化后 | 变化 | 判定 |")
    add("| --- | --- | --- | --- | --- |")
    add(row(f"答案准确率（{after['count']} 题）", before["accuracy"], after["accuracy"], 4, acc_ok))
    add(row("检索命中率（终果含证据）", before["retrieval_hit_rate"], after["retrieval_hit_rate"], 4, "—（基线召回本就不差，见 §5）"))
    add(row("证据排第 1 位比例", before["evidence_rank1_rate"], after["evidence_rank1_rate"], 4, "检索层真实增益"))
    add(row("证据排第 1 位比例（严格口径）", before["evidence_rank1_rate_strict"], after["evidence_rank1_rate_strict"], 4,
            f"剔除替代判据题后 {strict_count} 题可比"))
    add(row("纯向量单路 top-5 命中率", before["vector_top5_hit_rate"], after["vector_top5_hit_rate"], 4, "嵌入/索引层"))
    add(row("纯向量单路 top-20 命中率", before["vector_top20_hit_rate"], after["vector_top20_hit_rate"], 4, "嵌入/索引层"))
    add(row("首字响应 平均(ms)", before["first_token_avg_ms"], after["first_token_avg_ms"], 2, "—"))
    add(row("首字响应 P95(ms)", before["first_token_p95_ms"], after["first_token_p95_ms"], 2, "—"))
    add(row("首字响应 最大(ms)", before["first_token_max_ms"], after["first_token_max_ms"], 2, ft_ok))
    add(row("端到端 平均(ms)", before["total_avg_ms"], after["total_avg_ms"], 2, "—"))
    add(row("引用正确率", before["citation_accuracy"], after["citation_accuracy"], 4, cite_ok))
    add(row("答案平均长度(字符，去引用标签)", length_before["avg"], length_after["avg"], 1, "多值完整性的旁证，非目标本身"))
    add("")
    add(f"- 本次运行 **{after['correct']}/{after['count']}** 判对；首字最大 **{after['first_token_max_ms']:.2f} ms**（红线 3000 ms）；")
    add(f"  引用 **{after['citation_valid']}/{after['citation_total']}** 条有效（页 ∈ [1,{page_range}] 且可回查到真实 chunk）。")
    if payload.get("cold_start_eligible"):
        add(f"- 冷启动首题首字（本次 `--skip-warmup`，预热前）= {payload['first_question_first_token_ms']} ms，"
            f"**未计入**上表平均/P95/最大（口径见 §6）。")
    else:
        add(f"- 首题首字（**已预热**，故不是冷启动）= {payload['first_question_first_token_ms']} ms；"
            f"冷启动口径需 `--skip-warmup` 复跑（见 §6 字段名说明）。")
    if payload.get("after_extractive_path"):
        ex = payload["after_extractive_path"]
        add(f"- 纯抽取式路径（对照）= 准确率 {ex['accuracy']:.4f}（{ex['correct']}/{ex['count']}），"
            f"首字均值 {ex['first_token_avg_ms']:.2f} ms。")
    add("")
    add("---")
    add("")
    add("## 3. 逐条优化措施与量化收益")
    add("")
    add("> 每条只列**它能证明的指标**；跨层指标不得互相记账（见 §5 归因纪律）。")
    add("")
    add("| 层 | 措施 | 量化收益（本次实测） | 证据 |")
    add("| --- | --- | --- | --- |")
    add(f"| PDF 解析 | PyMuPDF 文本 + `find_tables()`，548 页解析、413 张表、跨页表格合并 | 表格块进入索引且可被检索命中；`table_errors=0` | T2 构建日志 |")
    add(f"| 分块 | 语义分块 + 表格整块 + 子块级 segments；块数 {before['chunk_count']} → {after['chunk_count']}，长度中位 118 → 425 | 块内语义更完整；配合子块召回缓解长块稀释 | T2 索引统计 |")
    add(f"| 嵌入 | bge-small-zh-v1.5(512) → bge-m3(1024) + 子块向量 | **纯向量单路 top-5 {before['vector_top5_hit_rate']:.2f} → {after['vector_top5_hit_rate']:.2f}、"
        f"top-20 {before['vector_top20_hit_rate']:.2f} → {after['vector_top20_hit_rate']:.2f}** | 本脚本 `vector_only_stage` |")
    add(f"| 混合检索 | 向量 + 自实现 BM25（k1=1.5/b=0.75）融合，权重 0.55/0.45 | 单独的向量通道仍弱（见上行）；混合后**终果含证据率 {after['retrieval_hit_rate']:.2f}**（基线 {before['retrieval_hit_rate']:.2f}） | 本脚本 + T2 自测 |")
    add(f"| 查询改写/扩展 | 多变体（原句/去主体/关键词/同义）+ 历史改写 | 子问题与追问的表达差异被覆盖（覆盖率不单列断言，效果见逐题位次表） | T2/T4 实测 |")
    add(f"| 重排 | 规则重排（本机无交叉编码器权重） | **证据排第 1 位比例（严格口径）{before['evidence_rank1_rate_strict']:.2f} → {after['evidence_rank1_rate_strict']:.2f}**；"
        + (f"位次回退题：{'；'.join(f'Q{qid} 第 {b} → 第 {a} 位' for qid, b, a in regressed)} | `optimization_compare.csv` 逐题 |"
           if regressed else "本次运行无位次回退题 | `optimization_compare.csv` 逐题 |"))
    add(f"| 加权/降权 | 数值覆盖度加权、释义页与碎片降权、全局上限 1.6 | 释义页/碎片降权生效（是否霸榜以 §4 逐题位次为准，不在此写死题号） | T6 §3/§5 + 本脚本逐题 |")
    add(f"| 生成（答案形态） | 抽取式「跨度 + 轻模板」+ 数值齐备；LLM 输出须过一致性校验 | **准确率 {before['accuracy']:.2f} → {after['accuracy']:.2f}**；答案平均长度 {length_before['avg']:.1f} → {length_after['avg']:.1f} 字符，多值/要点答全 | 本脚本逐题 + §4 |")
    llm_kept_row = mode_counter.get("llm", 0)
    add(f"| 生成提示词 | `研发/app/prompts/qa_prompt.txt` 按基线失败点重写（完整句子作答、多值必须列全、真实页码引用、中英双语）；`query_rewrite_prompt.txt` 精简至 671 字节 | "
        f"本次运行 LLM 答案进入终果 **{llm_kept_row}/{after['count']}** 题，其余全部被一致性校验拒绝并回退抽取式 → "
        f"**提示词对本轮最终交付答案的贡献为 {0 if llm_kept_row == 0 else '有限'}**（效果需在更稳定的 LLM 环境下复测，不得记入收益） | 本脚本逐题 `mode` 统计 |")
    add("")
    add("---")
    add("")
    add("## 4. 逐题对照")
    add("")
    add("| 题号 | 问题 | 基线答案 | 优化后答案 | 基线判定 | 优化后判定 | 证据位次(前→后) | 首字(前→后) ms |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in payload["per_question"]:
        qid = item["question_id"]
        b, a = item["before"], item["after"]
        b_ans = (b["answer"] or "").replace("|", "\\|")[:58]
        a_ans = (a["answer"] or "").replace("|", "\\|")[:58]
        add(
            f"| {qid} | {item['question'][:26]}… | {b_ans}… | {a_ans}… | "
            f"{'✅' if b['is_correct'] else '❌'} | {'✅' if a['is_correct'] else '❌'} | "
            f"{b['retrieval_rank'] if b['retrieval_rank'] is not None else 'null'} → "
            f"{a['retrieval_rank'] if a['retrieval_rank'] is not None else 'null'} | "
            f"{b['first_token_ms']:.0f} → {a['first_token_ms']:.0f} |"
        )
    add("")
    add("> 完整答案文本（未截断）见同目录 `optimization_compare.csv` 的 `metric=answer` 行与 `accuracy_report.json`。")
    add("")
    add("---")
    add("")
    add("## 5. 结论与遗留问题")
    add("")
    add("### 5.1 达标情况")
    add("")
    # 判定文案**一律引用真实判定**（acc_ok / ft_ok / cite_ok），不得写无条件"达标/满足"。
    add(f"- {acc_ok}：答案准确率 **{before['accuracy']:.4f} → {after['accuracy']:.4f}**"
        f"（{after['correct']}/{after['count']}）；工单要求 ≥ 0.90。")
    if failed_ids:
        add(f"- 本次未通过 **{len(failed_ids)}** 题：{'、'.join(f'Q{qid}' for qid in failed_ids)}"
            f"（逐题判分理由见 `accuracy_report.json` 的 `per_question[*].after.judge_reason` 与 §4）。")
        add("  根因以逐题记录为准；**未放宽判分口径**（`check_answer` / 阈值 0.62 未改）。")
    else:
        add("- 本次运行全部题目判对（无未通过项）。")
    add(f"- {ft_ok}：首字响应 平均 {after['first_token_avg_ms']:.2f} ms / P95 {after['first_token_p95_ms']:.2f} ms / "
        f"最大 {after['first_token_max_ms']:.2f} ms（工单红线 ≤ 3000 ms）。")
    add(f"- {cite_ok}：引用正确率 {after['citation_accuracy']:.4f}"
        f"（{after['citation_valid']}/{after['citation_total']} 条引用页有效且可回查）。")
    add("")
    add("### 5.2 增益归因（三层分开记账）")
    add("")
    add("| 层 | 可证指标 | 前 → 后 |")
    add("| --- | --- | --- |")
    add(f"| 解析/分块/嵌入 | 纯向量 top-5 / top-20 命中率 | {before['vector_top5_hit_rate']:.2f}/{before['vector_top20_hit_rate']:.2f} → {after['vector_top5_hit_rate']:.2f}/{after['vector_top20_hit_rate']:.2f} |")
    add(f"| 检索排序 | 证据排第 1 位（严格口径，剔除替代判据题） | {before['evidence_rank1_rate_strict']:.2f} → {after['evidence_rank1_rate_strict']:.2f} |")
    add(f"| 检索召回 | 终果含证据（本次 {after['count']} 题） | {before['retrieval_hit_rate']:.2f} → {after['retrieval_hit_rate']:.2f}（**基线本就不差，此层无增益**） |")
    add(f"| 生成/答案形态 | 答案准确率 | {before['accuracy']:.2f} → {after['accuracy']:.2f} |")
    add("")
    # 逐题归因**从本次运行数据推导**（原先写死 Q33/Q531/Q793/Q957/Q95，换一批题或部分运行即失真）。
    add(f"**逐题证据（本次相对基线翻正 {len(flipped_ids)} 题："
        f"{'、'.join(f'Q{qid}' for qid in flipped_ids) if flipped_ids else '无'}）**：")
    add("")
    if flipped_ids:
        for qid in flipped_ids:
            item = next(entry for entry in payload["per_question"] if int(entry["question_id"]) == qid)
            before_rank = item["before"]["retrieval_rank"]
            after_rank = item["after"]["retrieval_rank"]
            if before_rank == 1:
                layer = "基线证据已排第 1 → 增益属**生成/答案形态层**（与检索无关）"
            else:
                layer = f"基线证据位次 {before_rank} → 优化后 {after_rank} → 同时受益于**检索排序**与**答案形态**"
            add(f"- **Q{qid}**：{layer}。")
    else:
        add("- 本次运行没有相对基线翻正的题目（或基线题面不可比）。")
    if failed_ids:
        add(f"- **未通过**：{'、'.join(f'Q{qid}' for qid in failed_ids)} → 遗留问题（见 5.3）。")
    add("")
    llm_note = (
        "**没有产出任何进入终果的答案**"
        if llm_kept == 0
        else f"仅产出 **{llm_kept}/{after['count']}** 题进入终果的答案，其余被一致性校验拒绝"
    )
    add(f"> **不得**把上述增益归因于「换了更强的模型」：工单2 的生成模型 `qwen2.5:3b` 在本机上{llm_note}")
    add(f"> （本次交付路径的回答模式分布：{mode_text}）。生成端栈变化是**已声明的混淆项，不是因果项**。")
    add("> 同时**不得**写成「答案越长越好」：长度只是多值齐备的旁证，判分只看数值/要点是否齐全。")
    add("")
    add("### 5.3 遗留问题（如实记录，未修饰）")
    add("")
    if failed_ids:
        add(f"1. **本次未通过**：{'、'.join(f'Q{qid}' for qid in failed_ids)}"
            f"（准确率 {after['correct']}/{after['count']}，见 5.1）；判分口径未放宽。")
    else:
        add(f"1. **本次运行无未通过题**（{after['correct']}/{after['count']}）；判分口径未放宽。")
    if regressed:
        detail = "；".join(f"Q{qid} 第 {b} → 第 {a} 位" for qid, b, a in regressed)
        add(f"2. **证据位次回退题**：{detail}（仍在前 5 且判分结果见 §4），"
            f"说明规则重排对个别题目有负向，未修饰为「全面提升」。")
    else:
        add("2. **无证据位次回退题**（本次运行内）。")
    add("3. **重排为规则模式**：本机无 `bge-reranker-base` 权重，未运行交叉编码器重排；不得声称跑过模型重排。")
    add("4. **其他修复项的现状不在本脚本内断言**：中英文问答、多轮追问、无关问题容错（验收 4/5/6）"
        "的修复与复验结果，以 `测试/用户/user_acceptance_checklist_v2.md`、`研发/报告/可答性闸门修复报告.md` "
        "与在线/离线套件实测为准——**本脚本只记录本次对比运行的事实**，不重复固化易过期的结论。")
    add("")
    add("---")
    add("")
    add("## 6. 数据质量与指标口径诚实声明")
    add("")
    add("| # | 事项 | 说明 |")
    add("| --- | --- | --- |")
    add("| 1 | `evidence_pages` 错标 | `golden_qa.jsonl` 的 `evidence_pages` 存在错标（Q531/Q543 标 [22] 而证据在 p52；Q795 标 [27,157] 而实际在 p155/241 等）。因此**引用正确率**按「引用页 ∈ [1,548] 且可回查到真实 chunk」判定，**不**用 `evidence_pages` 当唯一真值 |")
    add(f"| 2 | 样本量 {after['count']} 条（全库 {payload['meta']['全库题目数']} 条） | 单题权重约 {100.0 / max(1, after['count']):.0f}%，差一题即显著影响百分点，统计置信度低；且全为 `should_be_unknown=false`，「不清楚」容错能力由在线容错用例另行覆盖 |")
    add("| 3 | **RAGAS 未运行** | `ragas` 依赖不可用且本机断网，四项指标（faithfulness/answer_relevancy/context_precision/context_recall）**无任何数值**，报告中一律标注「未运行」，并以确定性指标（准确率/命中率/引用正确率/首字延迟）替代。声明已写入 `ragas_report.md` |")
    add("| 4 | 生成端栈不一致（混淆项） | 基线 `Qwen2.5-7B-Instruct-AWQ @ 127.0.0.1:8000/v1`；工单2 本机 Ollama `qwen2.5:3b`。基线 0.50 **只引用工单1 产物**，未在本机重跑该服务 |")
    add(f"| 5 | 首字冷/热口径（**字段名歧义已修正**） | 上表首字为**预热后**稳态值；`first_question_first_token_ms` 只是**首题**首字，"
        f"**不等于冷启动**；`cold_start_first_token_ms` 仅在 `--skip-warmup` 运行时才有值"
        f"（本次 `cold_start_eligible={str(payload['cold_start_eligible']).lower()}`，见 `accuracy_report.json`） |")
    add("| 6 | 替代判据 | Q95（证据串含「……」）与 Q207（合成引用文本）无法整段定位，使用替代判据；本报告同时给出**严格口径**（剔除该两题）的位次比例 |")
    add("| 7 | 两种归一化 | 证据/答案匹配统一「去空白 + 去中英文标点」（与判分 `PUNCT_TO_STRIP` 同款）；纯空白归一化的差异已在 T6 `baseline_retrieval.json` 中记录 |")
    add("")
    add("---")
    add("")
    add("## 7. 产物、复现与防覆盖保护")
    add("")
    add("| 产物 | 说明 |")
    add("| --- | --- |")
    add("| `优化/评估结果/optimization_compare.csv` | 长表：`section,key,question_id,metric,before,after,delta,note`（含汇总与逐题明细） |")
    add("| `优化/评估结果/optimization_compare.md` | 本文件 |")
    add("| `优化/评估结果/accuracy_report.json` | 机器可读汇总（供 T9 校验） |")
    add("| `优化/评估结果/ragas_report.md` | 在既有报告上**追加** RAGAS 未运行声明（未覆盖 T2 内容） |")
    add("")
    add("> **防覆盖保护（双保险，2026-10-03 t5 加固）**：① `--limit N`（N < 全库题数）的部分运行"
        "**只写临时目录** `优化/评估结果/验证留痕/部分运行/limitN`（**不覆盖任何正式产物**；"
        "该路径全层级无下划线前缀，故交付完整性守卫判据 4 仍通过；亦不污染工作区根目录）；"
        "② 写正式交付路径前断言 `after.count == 全库题数`，不满足则退出码 4 拒绝写入。"
        "背景：曾有 `--limit` 冒烟运行把四件正式产物覆盖成**单题口径**（验收 7 因此对不上），"
        "该失效由 `测试/离线/test_deliverable_integrity.py` 钉住。")
    add("")
    add("```powershell")
    add("# 工作目录 = 工单2")
    add("pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py                  # 全量（写正式路径）")
    add("pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py --limit 2          # 冒烟（只写 优化/评估结果/验证留痕/部分运行/）")
    add("```")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_ragas_declaration(path: Path, after_summary: dict, baseline_summary: dict) -> None:
    """把「RAGAS 未运行」声明追加到既有 ragas_report.md（幂等，不覆盖既有内容）。"""
    block = f"""
{RAGAS_DECLARATION_MARKER}
## T7 · 优化前后对比中的 RAGAS 状态（追加段，未改动上文任何数值）

**RAGAS：未运行。**

- **原因**：本机 `ragas` 依赖**不可用且无法安装**（离线，`pypi.org` TLS 握手失败，pip 缓存无 wheel），
  且 RAGAS 的四项指标（`faithfulness` / `answer_relevancy` / `context_precision` / `context_recall`）
  需要 **LLM 作为裁判**，本机亦不具备云端裁判链路。
- **纪律**：本工单**不提供** RAGAS 数值，**不估算、不伪造**；对比报告中该四栏一律留空并标注「未运行」。
- **确定性替代指标**（均由真实运行产生，可复现）：

| 替代指标 | 优化前（工单1 基线） | 优化后（本次真实运行） |
| --- | --- | --- |
| 答案准确率（`check_answer`，阈值 0.62 未放宽） | {baseline_summary['accuracy']:.4f} | {after_summary['accuracy']:.4f} |
| 检索命中率（最终上下文含证据块） | {baseline_summary['retrieval_hit_rate']:.4f} | {after_summary['retrieval_hit_rate']:.4f} |
| 证据排第 1 位比例 | {baseline_summary['evidence_rank1_rate']:.4f} | {after_summary['evidence_rank1_rate']:.4f} |
| 引用正确率（页 ∈ [1,548] 且可回查 chunk） | {baseline_summary['citation_accuracy']:.4f} | {after_summary['citation_accuracy']:.4f} |
| 首字响应 平均 / P95 / 最大（ms） | {baseline_summary['first_token_avg_ms']:.2f} / {baseline_summary['first_token_p95_ms']:.2f} / {baseline_summary['first_token_max_ms']:.2f} | {after_summary['first_token_avg_ms']:.2f} / {after_summary['first_token_p95_ms']:.2f} / {after_summary['first_token_max_ms']:.2f} |

- 复现：`pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py`
- 详细对比见同目录 `optimization_compare.md` / `.csv` / `accuracy_report.json`。
"""
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if RAGAS_DECLARATION_MARKER in text:
            head, _, _ = text.partition(RAGAS_DECLARATION_MARKER)
            path.write_text(head.rstrip() + "\n" + block, encoding="utf-8")
            print(f"  RAGAS 声明已更新（保留既有内容）: {path}")
            return
        path.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")
        print(f"  RAGAS 声明已追加到既有报告（未覆盖 T2 内容）: {path}")
        return
    path.write_text("# RAGAS 评估报告\n" + block, encoding="utf-8")
    print(f"  RAGAS 声明已新建: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        print("参数错误：--limit 不能为负数", file=sys.stderr)
        return 3

    setup_logging()
    settings = get_settings()

    for required in (RESULTS_JSON, REPRO_JSON, RETRIEVAL_JSON):
        if not required.exists():
            print(f"前置产物缺失：{required}（请先完成 T6：collect_baseline_results.py 等）", file=sys.stderr)
            return 2

    baseline_results = load_json(RESULTS_JSON)
    baseline_repro = load_json(REPRO_JSON)
    baseline_retrieval = load_json(RETRIEVAL_JSON)

    golden_path = settings.paths.test_data / "golden_qa.jsonl"
    if not golden_path.exists():
        print(f"业务失败：金标准不存在 -> {golden_path}", file=sys.stderr)
        return 2
    golden_all = load_golden(golden_path)
    total_questions = len(golden_all)
    golden_items = golden_all[: args.limit] if args.limit else golden_all

    # ---- 防覆盖保护（双保险之一）：部分运行绝不写正式交付路径 ----
    official_out = OUT_DIR.resolve()
    out_dir = Path(args.out) if args.out else OUT_DIR
    partial = len(golden_items) < total_questions
    if partial and not args.allow_partial:
        if args.out and Path(args.out).resolve() == official_out:
            print(
                f"[防覆盖保护] 拒绝执行：本次是部分运行（{len(golden_items)}/{total_questions} 题），"
                f"而 --out 指向正式交付目录 {OUT_DIR}。\n"
                f"  正确做法：不带 --limit 全量重跑；或去掉 --out（部分运行会自动写 "
                f"{PARTIAL_OUT_ROOT.relative_to(REPO_ROOT)}/limit{len(golden_items)}）；"
                f"确需覆盖正式路径请显式加 --allow-partial。",
                file=sys.stderr,
            )
            return 3
        out_dir = PARTIAL_OUT_ROOT / f"limit{len(golden_items)}"
        print(
            f"[防覆盖保护] 部分运行（--limit {len(golden_items)} < 全库 {total_questions} 题）"
            f"→ 输出改到 {out_dir}；正式交付目录 {OUT_DIR} **未被写入**。"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    is_official_out = out_dir.resolve() == official_out

    chunks = Chunker.load(settings.paths.data_processed / "chunks.jsonl")
    total_pages = int(settings.pdf.page_count) if hasattr(settings, "pdf") and hasattr(settings.pdf, "page_count") else 548
    page_range = max((c.page for c in chunks), default=548)
    evidence_map = {item.id: {chunk.chunk_id for chunk in sst.locate_evidence(chunks, item)} for item in golden_items}

    print("=" * 92)
    print("工单2 优化前后对比评估：人工智能NLP-RAG-基于PDF文档的问答系统优化")
    print(f"基线来源     : {RESULTS_JSON.name} / {REPRO_JSON.name} / {RETRIEVAL_JSON.name}")
    print(f"判分口径     : 工单1 check_answer（FUZZY_THRESHOLD=0.62，不放宽）")
    print(f"题目数       : {len(golden_items)}/{total_questions}"
          f"   索引分块：{len(chunks)}   页码范围：1..{page_range}")
    print("=" * 92)

    engine = QAEngine(force_extractive=False)
    if not engine.load_index():
        print("业务失败：索引未就绪，请先执行 build_index.py", file=sys.stderr)
        return 2
    health = engine.health()
    embedder_info = health.get("embedder", {})
    reranker_info = health.get("reranker", {})
    llm_info = health.get("llm", {})
    # 向量库先建好：既用于环境信息（索引目录），也用于「纯向量单路」层对比
    embedder = get_embedder()
    store = VectorStore(index_dir=settings.index_dir(embedder.slug, embedder.dimension))
    vector_ok = store.load()
    environment = {
        "嵌入模型": embedder_info.get("model"),
        "嵌入维度": embedder_info.get("dimension"),
        "嵌入后端": embedder_info.get("backend"),
        "重排模式": reranker_info.get("mode"),
        "重排模型": reranker_info.get("model_configured") or reranker_info.get("model"),
        "LLM 后端": (llm_info.get("backend") or {}).get("name") if isinstance(llm_info.get("backend"), dict) else llm_info.get("backend"),
        "LLM 模型": settings.llm.model,
        "分块数": len(chunks),
        "索引目录": str(store.index_dir) if store.index_dir else "",
    }
    print(f"优化后环境   : 嵌入={environment['嵌入模型']}({environment['嵌入维度']}维) "
          f"重排={environment['重排模式']} LLM={environment['LLM 模型']} 分块={environment['分块数']}")

    # ---- 纯向量单路（与基线 T6 的「纯向量检索」同阶段，可横向比）----
    understanding = get_query_understanding()
    vector_ranks: dict[int, int | None] = {}       # 用「改写后查询」（系统实际首跳）
    vector_ranks_raw: dict[int, int | None] = {}   # 用原始问句（与 T6 基线口径完全可比）
    for item in golden_items:
        if not vector_ok:
            vector_ranks[item.id] = None
            vector_ranks_raw[item.id] = None
            continue
        analysis = understanding.analyze(item.question, [])
        rewritten = (analysis.rewritten or "").strip() or item.question
        hits = [chunk_id for chunk_id, _score in store.search(embedder.encode_one(rewritten), top_k=20)]
        vector_ranks[item.id] = best_rank(hits, evidence_map.get(item.id, set()))
        hits_raw = [chunk_id for chunk_id, _score in store.search(embedder.encode_one(item.question), top_k=20)]
        vector_ranks_raw[item.id] = best_rank(hits_raw, evidence_map.get(item.id, set()))
    vector_top5 = sum(1 for v in vector_ranks_raw.values() if isinstance(v, int) and v <= 5)
    vector_top20 = sum(1 for v in vector_ranks_raw.values() if isinstance(v, int))
    vector_top5_rw = sum(1 for v in vector_ranks.values() if isinstance(v, int) and v <= 5)
    vector_top20_rw = sum(1 for v in vector_ranks.values() if isinstance(v, int))
    print(
        f"纯向量单路   : 原始问句口径（与 T6 基线可比）top-5 {vector_top5}/{len(golden_items)}、"
        f"top-20 {vector_top20}/{len(golden_items)}；"
        f"改写后查询口径 top-5 {vector_top5_rw}/{len(golden_items)}、top-20 {vector_top20_rw}/{len(golden_items)}"
    )

    # ---- 真实运行优化后系统 ----
    cold_first_token = None
    if not args.skip_warmup:
        warmup = engine.warmup()
        print(f"预热         : 总 {warmup.get('elapsed_ms')} ms（首字按预热后统计；冷启动数值另行记录）")

    rows_rag: list[dict] = []
    rows_extractive: list[dict] = []
    if args.paths in ("rag", "both"):
        print("-" * 92)
        print("运行：最终交付路径（混合检索 + 生成 + 一致性校验，mode=rag）")
        rows_rag = run_path(engine, golden_items, evidence_map, "rag")
        if rows_rag:
            cold_first_token = rows_rag[0]["first_token_ms"]
    if args.paths in ("extractive", "both"):
        print("-" * 92)
        print("运行：纯抽取式路径（LLM 关闭，mode=extractive，作为稳健性对照）")
        engine_ex = QAEngine(force_extractive=True)
        if not engine_ex.load_index():
            print("业务失败：抽取式引擎索引未就绪", file=sys.stderr)
            return 2
        engine_ex.warmup()
        rows_extractive = run_path(engine_ex, golden_items, evidence_map, "extractive")

    # ---- 汇总 ----
    after_summary = summarize_path(rows_rag or rows_extractive, page_range, SURROGATE_QIDS)
    after_extractive_summary = summarize_path(rows_extractive, page_range, SURROGATE_QIDS) if rows_extractive else None

    # ---- 基线侧汇总（T6 产物）----
    focus_ids = {item.id for item in golden_items}  # --limit 时只比较同一批题目，避免分母不一致
    base_rows = {
        int(r["question_id"]): r for r in baseline_results["per_question_rag"] if int(r["question_id"]) in focus_ids
    }
    repro_rows = {
        int(r["question_id"]): r for r in baseline_repro["per_question"] if int(r["question_id"]) in focus_ids
    }
    SURROGATE_QIDS_PLACEHOLDER = None  # noqa: F841  （严格口径常量已在模块级定义）

    def _baseline_rank(qid: int) -> int | None:
        r = repro_rows.get(qid, {})
        positions = [
            int(entry["rank"])
            for entry in (r.get("scores") or [])
            if entry.get("chunk_id") in set(r.get("evidence_chunk_ids") or [])
        ]
        if positions:
            return positions[0]
        return 1 if qid == 207 else None  # Q207 替代判据块位于第 1 位（见 基线说明 §4.2.1）

    spec = baseline_retrieval["检索命中率_任务口径"]["归一化B_去空白与标点"]
    base_summary = {
        "accuracy": float(baseline_results["summary"]["accuracy"]),
        "retrieval_hit_rate": round(
            sum(
                1
                for qid, r in repro_rows.items()
                if r.get("evidence_chunk_in_context") or qid in SURROGATE_QIDS
            )
            / len(base_rows),
            4,
        ),
        # 含替代判据（Q95 第 3、Q207 第 1）后的「证据排第 1 位」比例
        "evidence_rank1_rate": round(
            sum(1 for qid in base_rows if _baseline_rank(qid) == 1) / len(base_rows), 4
        ),
        # 严格口径（只算有严格证据块的 8 题，排除 Q95/Q207）——与 T6 §4.2.1 的 5/10 对齐
        "evidence_rank1_rate_strict": round(
            sum(
                1
                for qid in base_rows
                if qid not in SURROGATE_QIDS and _baseline_rank(qid) == 1
            )
            / max(1, len([q for q in base_rows if q not in SURROGATE_QIDS])),
            4,
        ),
        "vector_top5_hit_rate": float(spec["top5"]["hit_rate"]),
        "vector_top20_hit_rate": float(spec["top20"]["hit_rate"]),
        "first_token_avg_ms": float(baseline_results["summary"]["first_token_avg_ms"]),
        "first_token_p95_ms": float(baseline_results["summary"]["first_token_p95_ms"]),
        "first_token_max_ms": float(baseline_results["summary"]["first_token_max_ms"]),
        "total_avg_ms": float(baseline_results["summary"]["total_avg_ms"]),
        "citation_accuracy": float(baseline_results["summary"]["citation_accuracy"]),
        "chunk_count": int(baseline_retrieval["meta"]["chunk_count"]),
        "context_size": 8,
    }
    # 基线逐题视图（含检索位次）
    rows_base: dict[int, dict] = {}
    for qid, rec in base_rows.items():
        repro = repro_rows.get(qid, {})
        rank = _baseline_rank(qid)
        rows_base[qid] = {
            "question": rec["question"],
            "answer": rec["baseline_answer"],
            "is_correct": rec["is_correct"],
            "judge_reason": rec.get("recomputed_reason", ""),
            "first_token_ms": rec["first_token_ms"],
            "total_ms": rec["total_ms"],
            "citation_pages": rec["citation_pages"],
            "evidence_rank": rank,
            "retrieval_hit": bool(repro.get("evidence_chunk_in_context")) or qid in (95, 207),
        }
    after_rows = {int(r["question_id"]): r for r in (rows_rag or rows_extractive)}
    after_summary["chunk_count"] = len(chunks)
    after_summary["vector_top5_hit_rate"] = round(vector_top5 / len(golden_items), 4)
    after_summary["vector_top20_hit_rate"] = round(vector_top20 / len(golden_items), 4)

    # ---- 防覆盖保护（双保险之二）：写正式交付路径前断言覆盖了全库题数 ----
    if is_official_out and after_summary["count"] != total_questions:
        print(
            f"[防覆盖保护] 拒绝写正式交付路径：本次仅覆盖 {after_summary['count']}/{total_questions} 题，"
            f"不足以支撑交付结论。\n"
            f"  请不带 --limit 全量重跑（部分运行会自动改写到 "
            f"{PARTIAL_OUT_ROOT.relative_to(REPO_ROOT)}/limit{after_summary['count']}）。",
            file=sys.stderr,
        )
        return 4
    if is_official_out:
        print(f"[防覆盖保护] 本次覆盖 {after_summary['count']}/{total_questions} 题 → 允许写正式交付路径 {OUT_DIR}")

    # ---- 写产物 ----
    csv_path = out_dir / "optimization_compare.csv"
    write_csv(csv_path, base_summary, after_summary, rows_base, after_rows, page_range)
    print(f"\nCSV   : {csv_path}")

    payload = {
        "meta": {
            "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
            "阶段": "优化 / 前后对比（T7）",
            "生成时间": datetime.now().isoformat(timespec="seconds"),
            "基线来源": [str(RESULTS_JSON), str(REPRO_JSON), str(RETRIEVAL_JSON)],
            "复现命令": "pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py",
            "判分口径": "工单1 check_answer（FUZZY_THRESHOLD=0.62 未放宽）",
            "检索命中口径": "最终上下文是否含证据块（证据定位复用 研发/scripts/selftest_retrieval.py::locate_evidence）",
            "首字口径": "流式首块计时；预热后统计，冷启动首题值另行记录",
            "RAGAS": "未运行（依赖不可用，本机断网）",
            "本次题目数": after_summary["count"],
            "全库题目数": total_questions,
            "是否全量运行": after_summary["count"] == total_questions,
            "输出目录": str(out_dir),
        },
        "environment": {
            "优化后": environment,
            "基线（工单1 记录，摘自 ragas_report.md §0）": baseline_results["meta"]["运行环境（摘自 ragas_report.md §0，工单1 记录）"],
        },
        "before": base_summary,
        "after": after_summary,
        "after_extractive_path": after_extractive_summary,
        "vector_only_stage": {
            "before（工单1 索引 2967×512 bge-small-zh-v1.5）": {
                "top5_hit_rate": base_summary["vector_top5_hit_rate"],
                "top20_hit_rate": base_summary["vector_top20_hit_rate"],
            },
            "after（工单2 索引 bge-m3 1024 维 + 子块级召回）": {
                "top5_hit_rate": after_summary["vector_top5_hit_rate"],
                "top20_hit_rate": after_summary["vector_top20_hit_rate"],
                "per_question_rank_raw_query": {str(k): v for k, v in sorted(vector_ranks_raw.items())},
                "per_question_rank_rewritten_query": {str(k): v for k, v in sorted(vector_ranks.items())},
            },
        },
        # 冷启动字段名歧义修正（t5）：原实现只有 ``cold_start_first_token_ms``，
        # 但它取自"首题首字"，**只有在 --skip-warmup 时才是冷启动**；预热后它只是首题稳态值。
        # 曾有报告出现 `cold_start_first_token_ms == first_token_max_ms`（1 题运行）被误读成"冷启动=最大"，
        # 故拆成三个语义明确的字段：
        "first_question_first_token_ms": cold_first_token,
        "cold_start_eligible": bool(args.skip_warmup),
        "cold_start_first_token_ms": cold_first_token if args.skip_warmup else None,
        "cold_start_口径": (
            "本次以 --skip-warmup 运行，首题首字即为冷启动值"
            if args.skip_warmup
            else "本次已预热，故 cold_start_first_token_ms=None；"
                 "first_question_first_token_ms 只是首题稳态值，**不是冷启动**"
        ),
        "answer_length": {
            "口径": "去掉引用标签 `[页码: N]` 与首尾空白后的字符数；仅作「多值齐备」旁证，不是优化目标",
            "before": answer_length_stats([rows_base[q]["answer"] for q in rows_base]),
            "after": answer_length_stats([after_rows[q]["answer"] for q in after_rows]),
            "参考答案": answer_length_stats([after_rows[q]["golden"] for q in after_rows]),
        },
        "per_question": [
            {
                "question_id": qid,
                "question": after_rows[qid]["question"],
                "golden": after_rows[qid]["golden"],
                "before": {
                    "answer": rows_base.get(qid, {}).get("answer"),
                    "is_correct": rows_base.get(qid, {}).get("is_correct"),
                    "judge_reason": rows_base.get(qid, {}).get("judge_reason"),
                    "first_token_ms": rows_base.get(qid, {}).get("first_token_ms"),
                    "total_ms": rows_base.get(qid, {}).get("total_ms"),
                    "citation_pages": rows_base.get(qid, {}).get("citation_pages"),
                    "retrieval_rank": rows_base.get(qid, {}).get("evidence_rank"),
                    "retrieval_hit": rows_base.get(qid, {}).get("retrieval_hit"),
                },
                "after": {
                    "answer": after_rows[qid]["answer"],
                    "is_correct": after_rows[qid]["is_correct"],
                    "judge_reason": after_rows[qid].get("judge_reason"),
                    "mode": after_rows[qid]["mode"],
                    "is_unknown": after_rows[qid].get("is_unknown"),
                    "first_token_ms": after_rows[qid]["first_token_ms"],
                    "total_ms": after_rows[qid]["total_ms"],
                    "citation_pages": after_rows[qid]["citation_pages"],
                    "citation_valid": after_rows[qid]["citation_valid"],
                    "retrieval_rank": after_rows[qid]["retrieval_rank"],
                    "retrieval_hit": after_rows[qid]["retrieval_hit"],
                    "retrieved_pages": after_rows[qid]["retrieved_pages"],
                },
            }
            for qid in sorted(after_rows)
        ],
    }
    json_path = out_dir / "accuracy_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON  : {json_path}")

    mode_counter = Counter(str(r.get("mode")) for r in after_rows.values())
    md_path = out_dir / "optimization_compare.md"
    write_markdown(md_path, payload, rows_base, after_rows, mode_counter, page_range)
    print(f"MD    : {md_path}")

    # RAGAS 声明：**只往本次输出目录写**——部分运行写临时目录时不得触碰正式 ragas_report.md
    # （原实现写死模块常量 RAGAS_DOC，这正是「--limit 1 冒烟连 ragas_report.md 一起覆盖」的原因之一）。
    ragas_target = RAGAS_DOC if is_official_out else out_dir / "ragas_report.md"
    if not is_official_out and RAGAS_DOC.exists() and not ragas_target.exists():
        # 临时目录里保留一份正式报告副本，便于对照"追加段"的效果（不修改正式文件）
        ragas_target.write_text(RAGAS_DOC.read_text(encoding="utf-8"), encoding="utf-8")
    append_ragas_declaration(ragas_target, after_summary, base_summary)
    print(f"RAGAS : {ragas_target}（追加声明；正式路径 {'已写' if is_official_out else '未触碰'}）")

    # ---- 控制台汇总 ----
    print("\n" + "=" * 92)
    print("优化前后对比（真实运行值）")
    print(f"{'指标':<34}{'优化前(基线)':<18}{'优化后':<18}{'变化'}")
    print("-" * 92)
    pairs = [
        ("答案准确率", base_summary["accuracy"], after_summary["accuracy"], 4),
        ("检索命中率（终果含证据）", base_summary["retrieval_hit_rate"], after_summary["retrieval_hit_rate"], 4),
        ("证据排第 1 位比例", base_summary["evidence_rank1_rate"], after_summary["evidence_rank1_rate"], 4),
        ("纯向量 top-5 命中率", base_summary["vector_top5_hit_rate"], after_summary["vector_top5_hit_rate"], 4),
        ("纯向量 top-20 命中率", base_summary["vector_top20_hit_rate"], after_summary["vector_top20_hit_rate"], 4),
        ("首字均值 ms", base_summary["first_token_avg_ms"], after_summary["first_token_avg_ms"], 2),
        ("首字 P95 ms", base_summary["first_token_p95_ms"], after_summary["first_token_p95_ms"], 2),
        ("首字最大 ms", base_summary["first_token_max_ms"], after_summary["first_token_max_ms"], 2),
        ("端到端均值 ms", base_summary["total_avg_ms"], after_summary["total_avg_ms"], 2),
        ("引用正确率", base_summary["citation_accuracy"], after_summary["citation_accuracy"], 4),
        ("分块数", base_summary["chunk_count"], after_summary["chunk_count"], 0),
    ]
    for name, before, after_v, digits in pairs:
        delta = after_v - before if isinstance(before, (int, float)) else None
        delta_text = f"{delta:+.{digits}f}" if isinstance(delta, (int, float)) else ""
        print(f"{name:<34}{before:<18.{digits}f}{after_v:<18.{digits}f}{delta_text}")
    print("-" * 92)
    print(f"准确率 = {after_summary['correct']}/{after_summary['count']} = {after_summary['accuracy']:.4f}"
          f"（达标线 0.90：{'✅ 达标' if after_summary['accuracy'] >= 0.9 else '❌ 未达标'}）")
    print(f"首字最大 = {after_summary['first_token_max_ms']:.2f} ms（红线 3000 ms："
          f"{'✅ 达标' if after_summary['first_token_max_ms'] <= 3000 else '❌ 未达标'}）")
    if cold_first_token is not None:
        # 文案必须与**真实语义**一致（第 2 轮复审 finding）：未加 --skip-warmup 时该值只是
        # "预热后首题首字"，**不是冷启动**；只有 --skip-warmup 才是冷启动值。
        if args.skip_warmup:
            print(f"冷启动首题首字（本次 --skip-warmup，预热前）= {cold_first_token:.2f} ms")
        else:
            print(f"首题首字（**已预热**，故不是冷启动；冷启动口径需 --skip-warmup）= {cold_first_token:.2f} ms")
    if after_extractive_summary:
        print(f"纯抽取式路径（对照）= 准确率 {after_extractive_summary['accuracy']:.4f} "
              f"({after_extractive_summary['correct']}/{after_extractive_summary['count']})")
    print(f"RAGAS = 未运行（依赖不可用，本机断网）")
    print("=" * 92)
    flush_logs()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # 禁止静默失败
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 对比评估失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
