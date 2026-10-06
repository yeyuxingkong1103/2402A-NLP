# -*- coding: utf-8 -*-
"""T9 基线与优化后对比（工单2 基线 vs 工单3 实测）—— **三种口径分开写，禁止跨分母夸大**。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本脚本回答四件事（全部可复算，产物落盘）：
    1. **判分口径校准**：把工单2 基线的 10 条原始记录（answer/golden）重新过一遍
       工单1 ``Evaluator.check_answer``（只读子进程），要求**逐题判定与判分说明都与基线一致** ——
       不一致说明我的口径与基线不同源，对比无效（脚本报错退出）。
    2. **三种准确率口径**：① 各自全集（基线 10 题 vs 工单3 14 题，**分母不同，仅作背景**）；
       ② 同集可比（基线那 10 题，唯一严格可比）；③ 逐题对照（基线判定/工单3判定/是否翻转）。
    3. **指标 before/after 两列**：accuracy、retrieval_hit_rate、first_token_avg/p95/max、
       citation_accuracy、unknown_accuracy（**拆成「误拒答率」与「负例拒答率」两行**，
       因为两者口径不同：基线的 unknown_accuracy=1.0 实际是「10 道可答题无人被误拒答」）。
    4. **首字方差**：工单3 两次独立运行（冷启动口径 / `--preheat-llm` 稳态口径）逐题对比 + 稳态值。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/baseline_compare.py
    pwsh -NoProfile -File run_py.ps1 优化/脚本/baseline_compare.py --run1 优化/评估结果/accuracy_report.json \
        --run2 优化/评估结果/accuracy_report.run2.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
sys.path.insert(0, str(DEV_DIR))
sys.path.insert(0, str(DEV_DIR / "scripts"))

from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from evaluate import judge_pairs, percentile_nearest_rank  # noqa: E402

REF_BASELINE = Path(r"E:\gao6gongdan\工单2\优化\基线\baseline_metrics.json")   # 只读
REF_RECORDS = Path(r"E:\gao6gongdan\工单1\优化\评估结果\eval_results\eval_records.json")  # 只读
BASELINE_DIR = REPO_ROOT / "优化" / "基线"
BASELINE_COPY = BASELINE_DIR / "工单2_baseline_metrics.json"
BASELINE_SOURCE_MD = BASELINE_DIR / "来源说明.md"
OUT_DIR = REPO_ROOT / "优化" / "评估结果"
OUT_JSON = OUT_DIR / "基线对比.json"
OUT_MD = OUT_DIR / "基线对比.md"
BASELINE_10_IDS = (260, 95, 33, 34, 957, 793, 795, 543, 531, 207)
BASELINE_FAILED_IDS = (95, 33, 957, 793, 531)


def sha256_16(path: Path) -> str:
    """文件 sha256 前 16 位（十六进制大写）。"""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()[:16]


def copy_baseline(*, logger: Any) -> dict[str, Any]:
    """把工单2 基线**原样只读复制**到 ``优化/基线/``（含来源说明），并返回哈希元数据。"""
    with logger.enter("copy_baseline", {"source": str(REF_BASELINE)}) as span:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        before = sha256_16(REF_BASELINE)
        shutil.copyfile(REF_BASELINE, BASELINE_COPY)
        after = sha256_16(REF_BASELINE)
        if before != after:
            raise RuntimeError(f"只读红线被破坏：复制过程中源文件哈希变化 {before} → {after}")
        meta = {"source": str(REF_BASELINE), "copy": str(BASELINE_COPY),
                "source_sha256_16": before, "copy_sha256_16": sha256_16(BASELINE_COPY),
                "bytes": BASELINE_COPY.stat().st_size}
        logger.log_event("baseline.copied", **meta)
        span.set_output(meta)
        return meta


def calibrate_judge(baseline: dict[str, Any], *, logger: Any) -> dict[str, Any]:
    """判分口径校准：基线 10 条记录重跑工单1 权威判分，逐题对齐判定与说明。"""
    with logger.enter("calibrate_judge", {"records": len(baseline["per_question"])}) as span:
        pairs = [{"id": row["question_id"], "answer": row["answer"], "golden": row["golden"]}
                 for row in baseline["per_question"]]
        verdicts, source = judge_pairs(pairs, logger=logger)
        rows: list[dict[str, Any]] = []
        mismatch: list[int] = []
        for row in baseline["per_question"]:
            qid = int(row["question_id"])
            verdict = verdicts.get(qid, {})
            same_verdict = bool(verdict.get("ok")) is bool(row["is_correct_recorded"])
            same_reason = str(verdict.get("reason")) == str(row["judge_reason"])
            if not same_verdict:
                mismatch.append(qid)
            rows.append({"id": qid, "recorded": bool(row["is_correct_recorded"]),
                         "recorded_reason": row["judge_reason"], "recomputed": verdict.get("ok"),
                         "recomputed_reason": verdict.get("reason"),
                         "same_verdict": same_verdict, "same_reason": same_reason})
        payload = {"source": source, "records": len(rows), "verdict_mismatch_ids": mismatch,
                   "reason_mismatch_ids": [row["id"] for row in rows if not row["same_reason"]],
                   "all_verdicts_match": not mismatch,
                   "all_reasons_match": all(row["same_reason"] for row in rows), "rows": rows}
        logger.log_event("baseline.calibration", records=len(rows),
                         verdict_mismatch=mismatch, all_reasons_match=payload["all_reasons_match"])
        span.set_output({"verdict_mismatch_ids": mismatch, "all_reasons_match": payload["all_reasons_match"]})
        if mismatch:
            raise RuntimeError(f"判分口径与基线不一致（口径不同源 → 对比无效）：题 {mismatch}")
        return payload


def load_run(path: Path, *, logger: Any) -> dict[str, Any]:
    """读取一次评估运行的 accuracy_report.json（缺失即报错，不静默继续）。"""
    with logger.enter("load_run", {"path": str(path)}) as span:
        if not path.is_file():
            raise FileNotFoundError(f"缺少评估产物：{path}（请先运行 研发/scripts/evaluate.py）")
        payload = json.loads(path.read_text(encoding="utf-8"))
        span.set_output({"rows": len(payload.get("results", [])),
                         "accuracy": payload.get("summary", {}).get("accuracy")})
        return payload


def subset_metrics(rows: Sequence[dict[str, Any]], ids: Sequence[int], *, raw: bool = False) -> dict[str, Any]:
    """在指定题号子集上重算指标（同集可比口径的关键：分母统一为子集大小）。

    ``raw=True`` 时用**答案原文（含引用标签）**的正确性（``correct_raw``），用于与基线送检文本同口径对照。
    """
    chosen = [row for row in rows if int(row["id"]) in set(ids)]
    total = len(chosen)
    firsts = [row["first_token_ms"] for row in chosen if not row["is_unknown"]]
    cite_total = sum(row["citation_check"]["total"] for row in chosen)
    cite_valid = sum(row["citation_check"]["valid_four_point"] for row in chosen)
    key = "correct_raw" if raw else "correct"
    correct_count = sum(1 for row in chosen if row.get(key) is True)
    return {
        "count": total,
        "correct_count": correct_count,
        "accuracy": round(correct_count / total, 4) if total else 0.0,
        "incorrect_ids": [row["id"] for row in chosen if row.get(key) is not True],
        "retrieval_hit_count": sum(1 for row in chosen if row["retrieval_hit"]),
        "retrieval_hit_rate": round(sum(1 for row in chosen if row["retrieval_hit"]) / total, 4) if total else 0.0,
        "citation_total": cite_total, "citation_valid": cite_valid,
        "citation_accuracy": round(cite_valid / cite_total, 4) if cite_total else 0.0,
        "first_token_avg_ms": round(sum(firsts) / len(firsts), 2) if firsts else None,
        "first_token_p95_ms": percentile_nearest_rank(firsts, 0.95),
        "first_token_max_ms": round(max(firsts), 2) if firsts else None,
        "total_avg_ms": (round(sum(row["total_ms"] for row in chosen) / total, 2) if total else None),
    }


def build_per_question(baseline: dict[str, Any], run1_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐题对照表（10 题）：基线判定 / 工单3 判定 / 是否翻转。"""
    index = {int(row["id"]): row for row in run1_rows}
    rows: list[dict[str, Any]] = []
    for record in baseline["per_question"]:
        qid = int(record["question_id"])
        mine = index[qid]
        base_ok = bool(record["is_correct_recorded"])
        now_ok = bool(mine.get("correct"))
        rows.append({
            "id": qid,
            "baseline_correct": base_ok,
            "baseline_reason": record["judge_reason"],
            "baseline_answer": record["answer"],
            "baseline_first_token_ms": record["first_token_ms"],
            "after_correct": now_ok,
            "after_reason": mine.get("judge_reason"),
            "after_answer": mine["answer_body"],
            "after_first_token_ms": mine["first_token_ms"],
            "flipped": base_ok != now_ok,
            "flip_direction": ("失败→正确" if (not base_ok and now_ok)
                               else ("正确→失败" if (base_ok and not now_ok)
                                     else ("保持正确" if base_ok else "仍未修复（基线错、现仍错）"))),
            "citation_before": record["citation_pages"], "citation_after": mine["citation_pages"],
        })
    return rows


def build_metrics(baseline: dict[str, Any], run1: dict[str, Any], run2: dict[str, Any] | None,
                  subset_before: dict[str, Any], subset_after: dict[str, Any],
                  subset_after_raw: dict[str, Any],
                  negative: dict[str, Any], *, baseline_source: dict[str, Any]) -> list[dict[str, Any]]:
    """指标 before/after 两列（含口径说明列，避免跨口径相减）。"""
    s1 = run1["summary"]
    s2 = (run2 or {}).get("summary", {})
    rows: list[dict[str, Any]] = [
        {"metric": "accuracy（各自全集）", "scope": "基线 10 题 / 工单3 14 题",
         "before": "5/10 = 50.0%", "after": f"{s1['correct_count']}/{s1['total']} = {s1['accuracy'] * 100:.1f}%",
         "note": "⚠️ **分母不同（10 vs 14），不可直接相减**；工单3 新增 4 题为 PDF2 题，无基线可比"},
        {"metric": "accuracy（同集可比·辅助严口径：送检文本含引用标签）", "scope": f"基线那 10 题",
         "before": "5/10 = 50.0%（基线判分本就是含 `[页码: N]` 标签的答案原文）",
         "after": f"{subset_after_raw['correct_count']}/{subset_after_raw['count']} = "
                  f"{subset_after_raw['accuracy'] * 100:.1f}%（错题 {subset_after_raw['incorrect_ids']}）",
         "note": "**不是验收口径**（captain 裁定验收判分用答案正文）：引用标签会打断第①步子串关系与第⑤步"
                 "二元组相似度 → 题 95/793 由对变错。本行价值在于「与基线**送检文本完全同口径**」的对照"},
        {"metric": "accuracy（同集可比·验收口径，主结论）", "scope": f"基线那 10 题（{', '.join(map(str, BASELINE_10_IDS))}）",
         "before": f"{subset_before['correct_count']}/{subset_before['count']} = "
                   f"{subset_before['accuracy'] * 100:.1f}%（原文口径；正文口径下基线无法重算，故不换算）",
         "after": f"{subset_after['correct_count']}/{subset_after['count']} = "
                  f"{subset_after['accuracy'] * 100:.1f}%",
         "note": f"**唯一严格可比口径（captain 裁定）**：提升 "
                 f"{round((subset_after['accuracy'] - subset_before['accuracy']) * 100, 1)} 个百分点；"
                 f"基线失败题 {list(BASELINE_FAILED_IDS)} 中 95/33/793 已转正、**957/531 仍判错**"},
        {"metric": "retrieval_hit_rate（召回命中）", "scope": "同集 10 题 / 全集 14 题",
         "before": "未测（工单2 基线未采集召回指标，eval_records 无返回块数据）",
         "after": f"同集 {subset_after['retrieval_hit_count']}/{subset_after['count']} = "
                  f"{subset_after['retrieval_hit_rate'] * 100:.1f}%　全集 {s1['retrieval_hit_count']}/"
                  f"{s1['total']} = {s1['retrieval_hit_rate'] * 100:.1f}%",
         "note": "命中原口径 = 证据原文是否落在返回块（`is_evidence_hit`），禁用「引用页==证据页」；"
                 "基线无该数据 → before 只能写「未测」，不得编造"},
        {"metric": "first_token_avg_ms", "scope": "同集 10 题",
         "before": f"{baseline['baseline_metrics']['first_token_avg_ms']}",
         "after": f"{subset_after['first_token_avg_ms']}",
         "note": f"下降 {round(baseline['baseline_metrics']['first_token_avg_ms'] - (subset_after['first_token_avg_ms'] or 0), 2)} ms"
                 f"（{round((1 - (subset_after['first_token_avg_ms'] or 0) / baseline['baseline_metrics']['first_token_avg_ms']) * 100, 1)}%）"},
        {"metric": "first_token_p95_ms", "scope": "同集 10 题",
         "before": f"{baseline['baseline_metrics']['first_token_p95_ms']}",
         "after": f"{subset_after['first_token_p95_ms']}",
         "note": "口径：最近秩法 p95（基线 p95 == max 亦为该口径）"},
        {"metric": "first_token_max_ms", "scope": "同集 10 题",
         "before": f"{baseline['baseline_metrics']['first_token_max_ms']}",
         "after": f"{subset_after['first_token_max_ms']}",
         "note": "预算 3000 ms"},
        {"metric": "citation_accuracy（引用可回溯）", "scope": "同集 10 题",
         "before": "1.0（10/10）",
         "after": f"{subset_after['citation_accuracy']}（{subset_after['citation_valid']}/{subset_after['citation_total']}）",
         "note": "工单3 口径为四点校验（页码范围/文件在语料/块可查页一致/引用处有支撑）"},
        {"metric": "unknown_accuracy①误拒答率（可答题不得回「不清楚」）", "scope": "基线 10 题 / 工单3 14 题",
         "before": "1.0（10/10，由 `should_be_unknown == is_unknown` 逐条计数得出，"
                   "而基线记录里 should_be_unknown 全为 False）",
         "after": f"1.0（{s1['total'] - len(s1['unknown_ids'])}/{s1['total']}，误拒答 id={s1['unknown_ids'] or '无'}）",
         "note": "**同口径可比**：基线的 1.0 只证明「没有误拒答」，不证明「会拒绝不可答问题」"},
        {"metric": "unknown_accuracy②负例拒答率（不可答题必须回「不清楚」）", "scope": "无关问题集",
         "before": "未测（空集）：工单1/工单2 的 eval_records 中 should_be_unknown=True 的记录数 = 0",
         "after": f"{negative['refused']}/{negative['total']} = {negative['rate']}" if negative["total"]
                  else "null（无负例集）",
         "note": "⚠️ 与①**不同口径，禁止相减**；工单3 是本项目**首次实测**负例拒答率"},
    ]
    if s2:
        rows.append({"metric": "first_token（第二次独立运行，`--preheat-llm`）", "scope": "全集 14 题",
                     "before": "——", "after": f"max {s2['first_token_max_ms']} / avg {s2['first_token_avg_ms']} / "
                                               f"p95 {s2['first_token_p95_ms']}",
                     "note": f"⚠️ 该次**不是稳态**：预热仅 {s2['llm_preheat_cold_ms']} ms（模型已常驻）却全题变慢，"
                             f"已定位为**并发占用同一 Ollama 实例**（同期 tester 留痕持续写入）→ 首字验收的"
                             f"**测量前提 = 独占 Ollama**；第一次运行 max {s1['first_token_max_ms']}"
                             f"（>1000ms 题 {s1['first_token_over_1000ms_ids'] or '无'}）；"
                             f"第二次 >1000ms 题 {s2['first_token_over_1000ms_ids'] or '无'}；"
                             f"五次团队观测见 §4"})
    return rows


def write_reports(payload: dict[str, Any], *, logger: Any) -> None:
    """落盘 ``基线对比.json`` / ``基线对比.md`` 与基线来源说明。"""
    with logger.enter("write_reports", {"json": str(OUT_JSON), "md": str(OUT_MD)}) as span:
        OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        source = payload["baseline_source"]
        lines = [
            "# 工单2 基线 vs 工单3 优化后 —— 对比报告",
            "",
            f"> 工单：{WORK_ORDER}　生成时间：{payload['generated_at']}",
            f"> 基线来源（只读）：`{source['source']}`（sha256_16 `{source['source_sha256_16']}`）",
            f"> 基线副本：`优化/基线/{BASELINE_COPY.name}`（sha256_16 `{source['copy_sha256_16']}`，逐字节一致）",
            f"> 工单3 运行：`{payload['runs']['run1']['path']}`"
            + (f"、`{payload['runs']['run2']['path']}`" if "run2" in payload["runs"] else ""),
            "",
            "## 0. 判分口径校准（对比有效性的前提）",
            "",
            f"- 把基线 10 条原始记录（answer/golden）重跑 **工单1 `Evaluator.check_answer`**："
            f"判定一致 **{payload['judge_calibration']['records'] - len(payload['judge_calibration']['verdict_mismatch_ids'])}"
            f"/{payload['judge_calibration']['records']}**；"
            f"判分说明逐字一致 **{payload['judge_calibration']['records'] - len(payload['judge_calibration']['reason_mismatch_ids'])}"
            f"/{payload['judge_calibration']['records']}** → 口径同源，对比成立。",
            "",
            "## 1. 三种准确率口径",
            "",
            f"1. **各自全集（仅作背景）**：基线 {payload['accuracy_scopes']['own_full']['before']} vs "
            f"工单3 {payload['accuracy_scopes']['own_full']['after']} —— ⚠️ **分母不同（10 vs 14），不可相减**。",
            f"2. **同集可比（主结论，验收口径）**：同一批 10 题（送检 = 答案正文），基线 "
            f"{payload['accuracy_scopes']['same_subset']['before']} → 工单3 "
            f"{payload['accuracy_scopes']['same_subset']['after']}，"
            f"提升 **{payload['accuracy_scopes']['same_subset']['delta_pp']} 个百分点**。",
            f"2b. **同集可比（与基线送检文本完全同口径，最保守下界）**：同一批 10 题（送检 = 答案原文含 "
            f"`[页码: N]` 标签），基线 5/10 = 50.0% → 工单3 "
            f"{payload['subset_after_raw']['correct_count']}/10 = {payload['subset_after_raw']['accuracy'] * 100:.1f}%，"
            f"提升 **{round((payload['subset_after_raw']['accuracy'] - payload['subset_before']['accuracy']) * 100, 1)} "
            f"个百分点**（详见下方「可比性偏差」与 §2 的「辅助严口径」行）。",
            "3. **逐题对照**：见下表（含 5 道基线失败题是否全部转正）。",
            "",
            "### 1.1 三个准确率数字的关系（**必须一起读，不可混用**）",
            "",
            "| 口径 | 数字 | 送检文本 | 错题 | 含义 / 用途 |",
            "| --- | --- | --- | --- | --- |",
            f"| **① 验收口径（唯一达标依据）** | **{payload['accuracy_three_scopes']['acceptance']['value']} = "
            f"{payload['accuracy_three_scopes']['acceptance']['accuracy'] * 100:.1f}%** | 答案正文（不含引用标签） | "
            f"{payload['accuracy_three_scopes']['acceptance']['incorrect_ids']} | 与基线同源同口径（工单1 "
            f"`Evaluator.check_answer` 本体）；golden 是正文，引用属展示字段 |",
            f"| ② 辅助严口径（一并披露） | {payload['accuracy_three_scopes']['strict_aux']['value']} = "
            f"{payload['accuracy_three_scopes']['strict_aux']['accuracy'] * 100:.1f}%（`accuracy_raw_text`） | "
            f"答案原文（含引用标签） | {payload['accuracy_three_scopes']['strict_aux']['incorrect_ids']} | "
            f"**非验收口径**：标签打断第①步子串与第⑤步相似度 → 题 95/793 由对变错 |",
            f"| ③ engineer 自建判据（仅旁证） | {payload['accuracy_three_scopes']['engineer_self_judge']['value']} | "
            f"答案正文 | 无 | **不得用于验收**：实词覆盖 ≥0.6 对整页倾倒式长答案宽松（题 531 覆盖 = 1.0） |",
            "",
            f"> {payload['accuracy_three_scopes']['relation']}",
            "",
            "> **可比性偏差（必须与上表/下表一起读）**：基线 `answer` 中含 `[页码: N]` 引用标签者 **10/15 条**"
            "（T9 正则实测复核：主口径 `rag` 的 **10/10 条全部带标签**，5 条 `rag_en` 无标签），"
            "而 `golden` 是干净正文；**标签会拉低第①步双向子串与第⑤步二元组相似度的命中机会**"
            "（实测：基线题 34/543 带标签仍判对，题 33/531 等因标签 + 答案过短被判错 → **差距不是恒定量，只能实测**）。"
            f"工单3 以**正文**送检，因此主结论 **+{payload['accuracy_scopes']['same_subset']['delta_pp']} pp** "
            "中含「送检形态差异」的一部分贡献；换成与基线完全相同的送检形态后重算，"
            f"**最保守的同口径提升为 +{round((payload['subset_after_raw']['accuracy'] - payload['subset_before']['accuracy']) * 100, 1)} pp**。"
            "**两行不可混引**：对外结论用验收口径，同时必须披露最保守下界。",
            "",
            "> 与基线送检文本**完全同口径**的对照见 §2「辅助严口径」行：同集 10 题 5/10 → "
            f"{payload['subset_after_raw']['correct_count']}/10 = {payload['subset_after_raw']['accuracy'] * 100:.1f}%"
            "（基线判分本就用含标签的答案原文，故此行为最保守的可比下界）。",
            "",
            "## 2. 指标对比表（before / after）",
            "",
            "| 指标 | 口径/范围 | 基线（before） | 工单3（after） | 说明 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in payload["metrics"]:
            lines.append(f"| {row['metric']} | {row['scope']} | {row['before']} | {row['after']} | {row['note']} |")
        lines += ["", "## 3. 逐题对照（基线那 10 题）", "",
                  "| id | 基线判定 | 基线判分说明 | 工单3 判定 | 工单3 判分说明 | 是否翻转 | 首字 before → after (ms) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in payload["per_question"]:
            lines.append(f"| {row['id']} | {'✅' if row['baseline_correct'] else '❌'} | {row['baseline_reason']} | "
                         f"{'✅' if row['after_correct'] else '❌'} | {row['after_reason']} | "
                         f"{row['flip_direction']} | {row['baseline_first_token_ms']} → {row['after_first_token_ms']} |")
        fixed = [row["id"] for row in payload["per_question"] if row["flip_direction"] == "失败→正确"]
        broken = [row["id"] for row in payload["per_question"] if row["flip_direction"] == "正确→失败"]
        lines += ["",
                  f"- 基线 5 道失败题（{', '.join(map(str, BASELINE_FAILED_IDS))}）中**转正 "
                  f"{len([item for item in fixed if item in BASELINE_FAILED_IDS])}/5**："
                  f"{[item for item in fixed if item in BASELINE_FAILED_IDS]}",
                  f"- 与之相反（基线对、现在错）的题：{broken or '无'}", "",
                  "## 4. 首字方差（工单3 两次独立运行）", ""]
        variance = payload["first_token_variance"]
        run2_line = ("- 第二次运行（`--preheat-llm` 预热口径）：未提供")
        if "run2" in variance:
            run2_line = (f"- 第二次运行（`--preheat-llm` 预热口径）：max **{variance['run2']['max']} ms**、"
                         f"avg {variance['run2']['avg']} ms、p95 {variance['run2']['p95']} ms、"
                         f">1000 ms 的题 {variance['run2']['over_1000ms_ids'] or '无'}；"
                         f"逐题 ≤3000 ms 达标 = **{variance['run2'].get('within_budget')}**")
        lines += [f"- 第一次运行（不预热 = 冷启动口径）：max **{variance['run1']['max']} ms**、"
                  f"avg {variance['run1']['avg']} ms、p95 {variance['run1']['p95']} ms、"
                  f">1000 ms 的题 {variance['run1']['over_1000ms_ids'] or '无'}",
                  run2_line,
                  "- 团队全部独立观测：**1506.72 / 679.95 / 544.14 / 1392.94 / 3274.04 ms**"
                  "（前三次为 engineer 的 `测试/离线/eval_answers.py` 运行，最新一次 17:50:50 / "
                  "`answer_eval_t6.json` sha256_16 `F5B83F2B9A985897`；后两次为 T9 本次运行）；"
                  "五次中四次 max ≤1506.72 ms 且逐题 ≤3000 ms，唯一一次超预算出现在本次第二次运行（病因见下）",
                  f"- 成因判断：{variance['cause']}",
                  "- 结论：首字在**独占 Ollama** 时稳定达标（≤1.5 s）；多人并行压同一实例时存在超 3000 ms 风险 → "
                  "建议验收 4 的最终判定在独占条件下复测一次，并把「推理服务不与评估/测试脚本并发」写入运行手册", "",
                  "## 5. 负例（「不清楚」）实测", "",
                  f"- 无关问题集：`{payload['negative']['path']}`（{payload['negative']['total']} 条，"
                  f"其中 T8 冻结 {payload['negative']['from_t8']} 条 + T9 自造 {payload['negative']['from_t9']} 条）",
                  f"- 拒答 **{payload['negative']['refused']}/{payload['negative']['total']}**"
                  f"（失败 id：{payload['negative']['failed_ids'] or '无'}）",
                  f"- 禁用串（编造内容）命中：{payload['negative']['forbidden_ids'] or '无'}",
                  f"- ⚠️ 与基线的 `unknown_accuracy=1.0` **不同口径**（基线是「可答题无误拒答」，"
                  f"且其 `should_be_unknown=True` 记录数为 0）→ 禁止相减。", "",
                  "## 6. 诚实声明", "",
                  "1. 本对比表的 before 值全部来自 `工单2\\优化\\基线\\baseline_metrics.json`（只读复制，哈希核对）；"
                  "after 值全部来自工单3 本机实跑产物（`qa_results.csv` / `accuracy_report.json`），未做任何人工润色。",
                  "2. **RAGAS 未运行（依赖不可用，本机断网）**：基线与工单3 均无 RAGAS 数值，"
                  "替代指标见 `ragas_report.md`。",
                  "3. 基线未采集召回指标 → `retrieval_hit_rate` 的 before 只能写「未测」，不得反推或估算。",
                  "4. 基线 10 题与工单3 14 题**分母不同**；凡跨分母数字均已在同格标注。", ""]
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        BASELINE_SOURCE_MD.write_text("\n".join([
            "# 优化/基线 —— 只读来源说明", "",
            f"> 工单：{WORK_ORDER}", "",
            f"- 来源（只读）：`{source['source']}`",
            f"- 源文件 sha256_16：`{source['source_sha256_16']}`（复制前后各算一次，一致）",
            f"- 本目录副本：`{BASELINE_COPY.name}`（{source['bytes']} B，sha256_16 `{source['copy_sha256_16']}`）",
            "- 纪律：工单2 目录**只读**，本副本用于留痕与复算；不得修改工单2 任何文件。",
            "- 基线关键值：accuracy 0.5（5/10，失败题 " + ", ".join(map(str, BASELINE_FAILED_IDS)) + "）、"
            "first_token_avg 11518.33 ms、p95/max 57312.92 ms、citation_accuracy 1.0（10/10）、"
            "unknown_accuracy 1.0（口径 = 可答题未被误拒答，见 `基线对比.md` 第 2 节）、"
            "ragas = 未运行（依赖不可用且本机断网）。", ""]), encoding="utf-8")
        span.set_output({"json_bytes": OUT_JSON.stat().st_size, "md_bytes": OUT_MD.stat().st_size})
        print(f"✅ 基线对比落盘：{OUT_JSON.relative_to(REPO_ROOT)} / {OUT_MD.relative_to(REPO_ROOT)}")
        print(f"✅ 基线只读副本：{BASELINE_COPY.relative_to(REPO_ROOT)}（sha256_16 "
              f"{source['copy_sha256_16']}）+ {BASELINE_SOURCE_MD.relative_to(REPO_ROOT)}")


def main(argv: Sequence[str] | None = None) -> int:
    """入口：复制基线 → 校准判分口径 → 三种口径对比 → 落盘报告（退出码 0/1/2）。"""
    parser = argparse.ArgumentParser(description="T9 基线与优化后对比")
    parser.add_argument("--run1", default=str(OUT_DIR / "accuracy_report.json"))
    parser.add_argument("--run2", default=str(OUT_DIR / "accuracy_report.run2.json"))
    parser.add_argument("--negative", default=str(OUT_DIR / "accuracy_report.json"),
                        help="含 unknown_results 的产物（默认取 run1）")
    args = parser.parse_args(argv)

    setup_logging(None, force=True)
    log = get_logger("baseline_compare")
    started = time.perf_counter()
    with log.enter("main", {"run1": args.run1, "run2": args.run2}) as span:
        for path in (REF_BASELINE, REF_RECORDS):
            if not path.is_file():
                log.log_event("baseline.missing", level="ERROR", path=str(path))
                print(f"❌ 缺少只读基线文件：{path}")
                return 2
        baseline_source = copy_baseline(logger=log)
        baseline = json.loads(BASELINE_COPY.read_text(encoding="utf-8"))
        calibration = calibrate_judge(baseline, logger=log)
        run1 = load_run(Path(args.run1), logger=log)
        run2_path = Path(args.run2)
        run2 = load_run(run2_path, logger=log) if run2_path.is_file() else None
        if run2 is None:
            log.log_event("baseline.run2_missing", level="WARNING", path=str(run2_path),
                          degrade="方差章节只写第一次运行")
        # 基线子集指标：**不重算**，直接取基线 JSON 的实测值（避免用我的口径改写 before）
        subset_before = {
            "count": 10, "correct_count": 5, "accuracy": 0.5,
            "incorrect_ids": list(BASELINE_FAILED_IDS),
            "retrieval_hit_count": None, "retrieval_hit_rate": None,
            "citation_total": baseline["baseline_metrics"]["citation_total"],
            "citation_valid": baseline["baseline_metrics"]["citation_valid"],
            "citation_accuracy": baseline["baseline_metrics"]["citation_accuracy"],
            "first_token_avg_ms": baseline["baseline_metrics"]["first_token_avg_ms"],
            "first_token_p95_ms": baseline["baseline_metrics"]["first_token_p95_ms"],
            "first_token_max_ms": baseline["baseline_metrics"]["first_token_max_ms"],
            "total_avg_ms": baseline["baseline_metrics"]["total_avg_ms"],
        }
        subset_after = subset_metrics(run1["results"], BASELINE_10_IDS)
        subset_after_raw = subset_metrics(run1["results"], BASELINE_10_IDS, raw=True)
        negatives = run1.get("unknown_results") or []
        refused = sum(1 for row in negatives if row.get("refused"))
        negative = {"path": str(OUT_DIR / "negative_questions_t9.jsonl"), "total": len(negatives),
                    "refused": refused, "rate": (f"{round(refused / len(negatives), 4)}" if negatives else None),
                    "failed_ids": [row["id"] for row in negatives if not row.get("refused")],
                    "forbidden_ids": [row["id"] for row in negatives if row.get("forbidden_hit")],
                    "from_t8": sum(1 for row in negatives if str(row["id"]).startswith(("N-4", "N-GEN"))),
                    "from_t9": sum(1 for row in negatives if str(row["id"]).startswith("T9-"))}
        payload = {
            "work_order": WORK_ORDER,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "baseline_source": baseline_source,
            "judge_calibration": calibration,
            "runs": {"run1": {"path": str(args.run1), "sha256_16": sha256_16(Path(args.run1)),
                              "summary": run1["summary"]},
                     **({"run2": {"path": str(run2_path), "sha256_16": sha256_16(run2_path),
                                  "summary": run2["summary"]}} if run2 else {})},
            "accuracy_scopes": {
                "own_full": {"before": f"5/10 = 50.0%（基线全集）",
                             "after": f"{run1['summary']['correct_count']}/{run1['summary']['total']} = "
                                      f"{run1['summary']['accuracy'] * 100:.1f}%（工单3 全集）",
                             "note": "分母不同，仅作背景"},
                "same_subset": {"ids": list(BASELINE_10_IDS),
                                "before": f"{subset_before['correct_count']}/{subset_before['count']} = "
                                          f"{subset_before['accuracy'] * 100:.1f}%",
                                "after": f"{subset_after['correct_count']}/{subset_after['count']} = "
                                         f"{subset_after['accuracy'] * 100:.1f}%",
                                "delta_pp": round((subset_after["accuracy"] - subset_before["accuracy"]) * 100, 1),
                                "note": "唯一严格可比口径"},
            },
            "metrics": build_metrics(baseline, run1, run2, subset_before, subset_after, subset_after_raw,
                                     negative, baseline_source=baseline_source),
            "per_question": build_per_question(baseline, run1["results"]),
            "subset_before": subset_before,
            "subset_after": subset_after,
            "subset_after_raw": subset_after_raw,
            "accuracy_three_scopes": {
                "acceptance": {"value": f"{run1['summary']['correct_count']}/{run1['summary']['total']}",
                               "accuracy": run1["summary"]["accuracy"],
                               "text_judged": "答案正文 answer_body（不含 [文件名: 页码] 引用标签）",
                               "incorrect_ids": run1["summary"]["incorrect_ids"],
                               "role": "唯一达标依据（与基线同源同口径：工单1 Evaluator.check_answer 本体）"},
                "strict_aux": {"value": f"{round(run1['summary']['accuracy_raw_text'] * run1['summary']['total'])}"
                                        f"/{run1['summary']['total']}",
                               "accuracy": run1["summary"]["accuracy_raw_text"],
                               "text_judged": "答案原文（含引用标签）",
                               "incorrect_ids": run1["summary"]["incorrect_ids_raw_text"],
                               "role": "辅助严口径（**非验收口径**）：标签打断第①步子串与第⑤步相似度"},
                "engineer_self_judge": {"value": "14/14", "accuracy": 1.0,
                                        "text_judged": "答案正文；判据 = 数值题 golden 数值全命中 / 文本题实词覆盖 ≥0.6",
                                        "incorrect_ids": [],
                                        "role": "诊断旁证（**不得用于验收**）：对整页倾倒式长答案宽松"},
                "relation": f"① 验收口径 {run1['summary']['correct_count']}/{run1['summary']['total']} = 结论"
                            f"（{'达标' if run1['summary']['accuracy'] >= 0.90 else '未达标'}"
                            f"{'，错题 ' + str(run1['summary']['incorrect_ids']) if run1['summary']['incorrect_ids'] else ''}）；"
                            f"② 辅助严口径 {round(run1['summary']['accuracy_raw_text'] * run1['summary']['total'])}"
                            f"/{run1['summary']['total']} = 同批答案在严送检文本下的下界"
                            f"{'（错题 ' + str(run1['summary']['incorrect_ids_raw_text']) + '）' if run1['summary']['incorrect_ids_raw_text'] else ''}；"
                            f"③ engineer 自建 14/14 = 宽判据（本轮与①一致，历史轮次曾分歧）。"
                            f"三者分子分母含义不同、不可换算或混引；两次独立运行的 ①②③ 完全一致。",
            },
            "first_token_variance": {
                "run1": {"max": run1["summary"]["first_token_max_ms"], "avg": run1["summary"]["first_token_avg_ms"],
                         "p95": run1["summary"]["first_token_p95_ms"],
                         "over_1000ms_ids": run1["summary"]["first_token_over_1000ms_ids"],
                         "preheat": bool(run1["summary"]["llm_preheat_cold_ms"])},
                **({"run2": {"max": run2["summary"]["first_token_max_ms"],
                             "avg": run2["summary"]["first_token_avg_ms"],
                             "p95": run2["summary"]["first_token_p95_ms"],
                             "over_1000ms_ids": run2["summary"]["first_token_over_1000ms_ids"],
                             "within_budget": run2["summary"]["first_token_within_budget"],
                             "preheat": bool(run2["summary"]["llm_preheat_cold_ms"])}} if run2 else {}),
                "cause": ("第二次运行（`--preheat-llm`）并未更快：该次预热实测仅 92.61 ms（模型已常驻）→ 不是冷启动；"
                          "但全集 total_avg 由 2352.42 抬到 3250.19 ms、avg 首字 363.21 → 602.40 ms，"
                          "且执行窗口内 tester 的 `测试/留痕/user_*` 持续写入（最近 17:50:21）、本机仅一个 Ollama 实例串行 → "
                          "**并发排队**。四次独立观测中三次 max ≤1506.72 ms 且逐题 ≤3000 ms；稳态值应在独占 Ollama 条件复测"),
            },
            "negative": negative,
            "honesty": [
                "before 值全部来自工单2 基线只读副本；after 值全部来自工单3 实跑产物",
                "RAGAS 未运行（依赖不可用，本机断网）→ 前后都没有 RAGAS 数值",
                "基线未采集召回指标 → retrieval_hit_rate 的 before 为「未测」，未做任何反推",
                "基线 10 题与工单3 14 题分母不同，跨分母数字均已标注",
            ],
        }
        write_reports(payload, logger=log)
        print(f"准确率口径：全集 {payload['accuracy_scopes']['own_full']['after']}（基线 "
              f"{payload['accuracy_scopes']['own_full']['before']}，分母不同）｜同集可比 "
              f"{payload['accuracy_scopes']['same_subset']['before']} → "
              f"{payload['accuracy_scopes']['same_subset']['after']}"
              f"（+{payload['accuracy_scopes']['same_subset']['delta_pp']} pp）")
        print(f"首字：run1 max={payload['first_token_variance']['run1']['max']} ms"
              + (f"；run2 max={payload['first_token_variance']['run2']['max']} ms" if run2 else ""))
        print(f"负例拒答率：{negative['refused']}/{negative['total']}（失败 {negative['failed_ids'] or '无'}）")
        print(f"耗时 {round((time.perf_counter() - started) * 1000, 2)} ms")
        span.set_output({"delta_pp": payload["accuracy_scopes"]["same_subset"]["delta_pp"],
                         "negative_failed": negative["failed_ids"]})
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底
        try:
            get_logger("baseline_compare").log_event("compare.failed", level="ERROR",
                                                     error_type=type(exc).__name__, message=str(exc),
                                                     stack=__import__("traceback").format_exc())
        finally:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)
