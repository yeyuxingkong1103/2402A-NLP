# -*- coding: utf-8 -*-
"""T9 评估 CLI：14 题端到端问答评估 + 指标产物（设计/接口设计.md §6.2 **冻结契约**）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

契约（设计 §6.2）：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/evaluate.py \\
      [--golden 研发/data/eval/golden_qa.jsonl] [--out 优化/评估结果] \\
      [--mode offline|online] [--base-url http://127.0.0.1:8600] [--top-k 5] \\
      [--backend auto|ollama|openai|extractive] [--limit N] [--log-level INFO]

产物（缺一即验收 8 不通过）：``qa_results.csv``、``accuracy_report.json``、``accuracy_report.md``、
``ragas_report.md``（首屏必须标注「RAGAS 未运行（依赖不可用，本机断网）」）。

判分口径（**不得放宽**，设计/验收标准.md §2.1，``FUZZY_THRESHOLD = 0.62``）：
    ① 参考答案是答案的子串（忽略标点/空白，双向）；
    ② 参考答案中**全部带单位数值**命中（万元/亿元/元 折算后比较）；
    ③ 关键实体（引号内专有名称）有交集；
    ④ 比例（百分比）全部命中；
    ⑤ 在不缺任何数值的前提下，字符二元组 Jaccard ≥ 0.62；否则判错。

    主判分器 = **工单1 权威 ``Evaluator.check_answer``**（只读子进程，见 ``优化/脚本/ref_judge_runner.py``，
    与工单2 基线的 ``judge_reason`` 逐条可复现）；子进程不可用时**显式降级**到本文件的五步等价实现，
    并把口径来源写进每行结果与报告（禁止静默换口径）。

召回命中口径：``evidence_verbatim`` 是否落在返回块（``is_evidence_hit``，唯一实现），
    **不是**「引用页 == evidence_pages」；题 207 的 golden ``evidence`` 是合成串+编辑注记 → 用它必为 13/14，
    本 CLI 同时给出 ``retrieval_hit``（evidence_verbatim，主口径）与
    ``retrieval_hit_strict_golden_evidence``（严格原值口径，如实披露）。

退出码：0 = 评估完成（不代表达标）；1 = 评估中断（异常已落 error.log）；2 = 入参/金标准错误。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))

from app.core import citation as citation_mod  # noqa: E402
from app.core.answerability import classify_subject_expectation, subject_gate  # noqa: E402
from app.core.config import AppConfig, discover_pdfs, get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import QAEngine, build_engine  # noqa: E402
from app.core.query_understanding import understand  # noqa: E402
from app.core.retrieval_utils import is_evidence_hit  # noqa: E402

DEFAULT_GOLDEN = DEV_DIR / "data" / "eval" / "golden_qa.jsonl"
DEFAULT_OUT = REPO_ROOT / "优化" / "评估结果"
MIRROR_DIR = DEV_DIR / "data" / "eval" / "eval_results"
PROCESS_LOG_DIR = DEFAULT_OUT / "过程日志"
DEFAULT_UNKNOWN = REPO_ROOT / "测试" / "测试数据" / "unknown_questions.jsonl"
REFERENCE_DEV = Path(r"E:\gao6gongdan\工单1\研发")
REF_JUDGE_RUNNER = REPO_ROOT / "优化" / "脚本" / "ref_judge_runner.py"

RAGAS_NOT_RUN = "RAGAS 未运行（依赖不可用，本机断网）"
REQUIRED_GOLDEN_FIELDS = ("id", "question", "answer", "evidence", "evidence_verbatim", "evidence_pages")

# ---------------------------------------------------------------------------
# 判分口径（工单1 ``Evaluator.check_answer`` 的等价实现；仅在权威子进程不可用时使用）
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = 0.62
NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
ENTITY_PATTERN = re.compile(r"[“\"]([^”\"]{4,60})[”\"]")
ALL_NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"
UNIT_SCALES: tuple[tuple[str, float], ...] = (("亿元", 1e8), ("亿", 1e8), ("万元", 1e4), ("万", 1e4), ("元", 1.0))
_AMOUNT_SCAN = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(亿元|亿|万元|万|元)?")


def _normalize(text: str) -> str:
    """归一化：统一逗号/括号/百分号并去空白（与工单1 Evaluator 同口径）。"""
    body = str(text or "")
    for old, new in (("，", ","), ("（", "("), ("）", ")"), ("％", "%"), (" ", ""), ("\u3000", "")):
        body = body.replace(old, new)
    return body.strip()


def _bigrams(text: str) -> set[str]:
    """字符二元组集合（中文无需分词即可度量字面重叠）。"""
    cleaned = "".join(char for char in str(text or "") if char not in PUNCT_TO_STRIP)
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard 相似度（任一为空 → 0.0）。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _bigram_number(raw: str) -> str:
    """数字字面量规范化（去千分位、去多余小数零）。"""
    try:
        value = float(str(raw).replace(",", ""))
    except ValueError:
        return str(raw)
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _trim_float(value: float) -> str:
    """浮点转稳定字符串（去多余零）。"""
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def normalize_number_strings(text: str) -> set[str]:
    """把文本中**带单位**的数字折算成元后规范化（与工单1 ``number_utils`` 同口径）。"""
    normalized: set[str] = set()
    for match in _AMOUNT_SCAN.finditer(str(text or "")):
        number_text, unit = match.group(1), match.group(2)
        if not unit:
            continue
        scale = next((factor for name, factor in UNIT_SCALES if name == unit), 1.0)
        normalized.add(_trim_float(float(number_text.replace(",", "")) * scale))
    return normalized


def check_answer_local(answer: str, golden: str) -> tuple[bool, str]:
    """五步判分的本地等价实现（返回 ``(是否正确, 判分说明)``）。

    逐条对齐工单1 ``Evaluator.check_answer``：包含关系 → 带单位数值 → 关键实体 → 比例 → 模糊相似度。
    """
    if not answer or not golden:
        return False, "答案或参考答案为空"
    norm_answer, norm_golden = _normalize(answer), _normalize(golden)

    plain_answer = "".join(char for char in norm_answer if char not in PUNCT_TO_STRIP)
    plain_golden = "".join(char for char in norm_golden if char not in PUNCT_TO_STRIP)
    if plain_golden and plain_golden in plain_answer:
        return True, "参考答案为答案子串（忽略标点）"
    if plain_answer and plain_answer in plain_golden:
        return True, "答案为参考答案子串（忽略标点）"

    golden_amounts = normalize_number_strings(norm_golden)
    answer_amounts = normalize_number_strings(norm_answer)
    if golden_amounts and golden_amounts.issubset(answer_amounts):
        return True, f"金额全部命中({len(golden_amounts)}个)"
    if golden_amounts and all(
        any(abs(float(got) - float(want)) <= max(1.0, abs(float(want))) * 1e-6 for got in answer_amounts)
        for want in golden_amounts
    ):
        return True, f"金额等价命中({len(golden_amounts)}个)"

    golden_entities = set(ENTITY_PATTERN.findall(golden))
    answer_entities = set(ENTITY_PATTERN.findall(answer))
    if golden_entities and golden_entities & answer_entities:
        return True, "关键实体命中"

    golden_percents = {pct.replace(" ", "") for pct in PERCENT_PATTERN.findall(norm_golden)}
    answer_percents = {pct.replace(" ", "") for pct in PERCENT_PATTERN.findall(norm_answer)}
    if golden_percents and golden_percents.issubset(answer_percents):
        return True, "比例全部命中"

    missing_amounts = golden_amounts - answer_amounts
    golden_all = {_bigram_number(match) for match in ALL_NUMBER_PATTERN.findall(norm_golden)}
    answer_all = {_bigram_number(match) for match in ALL_NUMBER_PATTERN.findall(norm_answer)}
    missing_all = golden_all - answer_all
    if not missing_amounts and not missing_all:
        similarity = _jaccard(_bigrams(norm_answer), _bigrams(norm_golden))
        if similarity >= FUZZY_THRESHOLD:
            return True, f"字符二元组相似度 {similarity:.2f} ≥ {FUZZY_THRESHOLD}"
    missing = sorted(missing_all) or sorted(missing_amounts)
    return False, (f"未命中：缺少数值 {missing[:6]}" if missing else "未命中：文本与数值均不匹配")


# ---------------------------------------------------------------------------
# 权威判分（只读子进程）
# ---------------------------------------------------------------------------
def judge_authoritative(pairs: Sequence[dict[str, Any]], *, logger: Any) -> dict[str, Any] | None:
    """用只读子进程调工单1 ``Evaluator.check_answer``；不可用返回 ``None``（由调用方降级）。

    用「文件重定向」而不是管道捕获输出：沙箱下命名管道可能被拒绝，文件重定向稳定且留痕完整。
    """
    log = logger
    with log.enter("judge_authoritative", {"pairs": len(pairs), "runner": str(REF_JUDGE_RUNNER)}) as span:
        if not REF_JUDGE_RUNNER.is_file() or not (REFERENCE_DEV / "app" / "core" / "evaluator.py").is_file():
            log.log_event("judge.authoritative_unavailable", level="WARNING",
                          runner_exists=REF_JUDGE_RUNNER.is_file(),
                          evaluator_exists=(REFERENCE_DEV / "app" / "core" / "evaluator.py").is_file())
            return None
        PROCESS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        pairs_path = PROCESS_LOG_DIR / f"_judge_pairs_{stamp}.json"
        out_path = PROCESS_LOG_DIR / f"_judge_verdicts_{stamp}.json"
        stdout_path = PROCESS_LOG_DIR / f"_judge_stdout_{stamp}.log"
        stderr_path = PROCESS_LOG_DIR / f"_judge_stderr_{stamp}.log"
        pairs_path.write_text(json.dumps(list(pairs), ensure_ascii=False, indent=1), encoding="utf-8")
        cmd = [sys.executable, "-B", "-X", "utf8", str(REF_JUDGE_RUNNER),
               str(pairs_path), str(out_path), str(REFERENCE_DEV)]
        try:
            with stdout_path.open("w", encoding="utf-8") as out_handle, \
                    stderr_path.open("w", encoding="utf-8") as err_handle:
                completed = subprocess.run(cmd, stdout=out_handle, stderr=err_handle,
                                           timeout=300, check=False)
        except Exception as exc:  # noqa: BLE001 —— 显式降级：调用方改用本地五步口径
            log.log_event("judge.authoritative_failed", level="ERROR", error_type=type(exc).__name__,
                          message=str(exc), degrade="改用本地五步等价实现")
            return None
        if completed.returncode != 0 or not out_path.is_file():
            log.log_event("judge.authoritative_failed", level="ERROR", returncode=completed.returncode,
                          stdout=str(stdout_path), stderr=str(stderr_path), degrade="改用本地五步等价实现")
            return None
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        log.log_event("judge.authoritative_ok", count=payload.get("count"),
                      evaluator_file=payload.get("evaluator_file"), fuzzy=payload.get("fuzzy_threshold"))
        span.set_output({"count": payload.get("count"), "ok_true": sum(
            1 for row in payload.get("rows", []) if row.get("ok") is True)})
        return payload


def judge_pairs(pairs: Sequence[dict[str, Any]], *, logger: Any) -> tuple[dict[Any, dict[str, Any]], str]:
    """批量判分：优先权威子进程，失败则本地五步；返回 ``(id→判定, 口径来源)``。"""
    payload = judge_authoritative(pairs, logger=logger)
    verdicts: dict[Any, dict[str, Any]] = {}
    if payload is not None:
        for row in payload.get("rows", []):
            verdicts[row.get("id")] = {"ok": row.get("ok"), "reason": row.get("reason"),
                                       "source": "工单1 Evaluator.check_answer（只读子进程）"}
        return verdicts, "工单1 Evaluator.check_answer（只读子进程；见 优化/脚本/ref_judge_runner.py）"
    for pair in pairs:
        ok, reason = check_answer_local(str(pair.get("answer") or ""), str(pair.get("golden") or ""))
        verdicts[pair.get("id")] = {"ok": bool(ok), "reason": reason,
                                    "source": "本地五步等价实现（权威子进程不可用，已显式降级）"}
    return verdicts, "本地五步等价实现（权威子进程不可用，已显式降级）"


# ---------------------------------------------------------------------------
# 金标准 / 输入
# ---------------------------------------------------------------------------
def load_golden(path: Path, *, logger: Any) -> list[dict[str, Any]]:
    """读取并校验金标准（缺字段/重复 id → ValueError，由 main 转退出码 2）。"""
    with logger.enter("load_golden", {"path": str(path)}) as span:
        if not path.is_file():
            raise ValueError(f"金标准不存在：{path}")
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [int(item.get("id") or 0) for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"金标准 id 重复：{sorted(ids)}")
        for item in items:
            missing = [field for field in REQUIRED_GOLDEN_FIELDS if not item.get(field)]
            if missing:
                raise ValueError(f"题 {item.get('id')} 缺字段 {missing}（设计/验收标准.md §3.1）")
        span.set_output({"count": len(items), "ids": sorted(ids),
                         "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest().upper()[:16]})
        return items


def load_unknowns(path: Path, *, logger: Any) -> list[dict[str, Any]]:
    """读取无关问题集（不得答出内容）；文件不存在返回空列表并 WARNING 留痕。"""
    with logger.enter("load_unknowns", {"path": str(path)}) as span:
        if not path.is_file():
            logger.log_event("unknown.fixture_missing", level="WARNING", path=str(path),
                             degrade="unknown_accuracy 记为 null")
            span.set_output({"count": 0, "missing": True})
            return []
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        span.set_output({"count": len(items)})
        return items


def page_counts(*, logger: Any) -> dict[str, int]:
    """自动发现语料并取页数（引用越界判定用；**不硬编码文件名**）。"""
    with logger.enter("page_counts", {}) as span:
        counts = {source.file_name: int(source.page_count or 0) for source in discover_pdfs(logger=logger)}
        span.set_output({"files": counts})
        return counts


def corpus_file_map(*, logger: Any) -> dict[str, str]:
    """自动发现 PDF 后按「第 1 页发行人全称」映射 ``corpus -> file_name``（**不硬编码文件名**）。

    与 ``优化/脚本/build_golden_qa.py::resolve_corpus`` 同一口径：题 1~4 属 id 以
    「武汉力源信息技术股份有限公司」开头的 PDF（corpus=pdf2），其余属「武汉兴图新科电子股份有限公司」。
    """
    with logger.enter("corpus_file_map", {}) as span:
        import pymupdf

        issuers = {"pdf1": "武汉兴图新科电子股份有限公司", "pdf2": "武汉力源信息技术股份有限公司"}
        mapping: dict[str, str] = {}
        for source in discover_pdfs(logger=logger):
            doc = pymupdf.open(str(source.path))
            try:
                head = "".join(doc[index].get_text() for index in range(min(2, doc.page_count)))
            finally:
                doc.close()
            for corpus, issuer in issuers.items():
                if issuer in head and corpus not in mapping:
                    mapping[corpus] = source.file_name
        span.set_output({"mapping": mapping})
        return mapping


# ---------------------------------------------------------------------------
# 单题执行
# ---------------------------------------------------------------------------
def citation_page_checks(answer: Any, item: dict[str, Any], *, cfg: AppConfig, page_counts_map: dict[str, int],
                         chunk_lookup: Any, page_text_lookup: Any, logger: Any) -> dict[str, Any]:
    """引用四点校验（设计/验收标准.md §3.5）：页码范围 / 文件在语料 / 块可查且页一致 / 引用处有支撑。"""
    with logger.enter("citation_page_checks", {"qid": item["id"], "citations": len(answer.citations)}) as span:
        golden_union = {int(page) for page in (item.get("evidence_pages_union")
                                               or item.get("evidence_pages") or [])}
        rows: list[dict[str, Any]] = []
        for cite in answer.citations:
            page = int(cite.page)
            total = int(page_counts_map.get(cite.file_name, 0))
            chunk = chunk_lookup(cite.chunk_id) if cite.chunk_id else None
            evidence = citation_mod.evidence_text_for(cite, chunk_lookup=chunk_lookup,
                                                      page_text_lookup=page_text_lookup)
            supported, why = citation_mod.answer_support_check(answer.text, evidence)
            rows.append({
                "citation": cite.render(language=answer.language),
                "file_name": cite.file_name, "page": page, "chunk_id": cite.chunk_id,
                "page_in_range": bool(1 <= page <= total),
                "file_in_corpus": cite.file_name in page_counts_map,
                "chunk_page_match": bool(chunk is not None and int(getattr(chunk, "page", 0)) == page),
                "supported": bool(supported), "support_why": why,
                "page_in_golden": page in golden_union,
            })
        total_cites = len(rows)
        valid = sum(1 for row in rows if row["page_in_range"] and row["file_in_corpus"]
                    and row["chunk_page_match"] and row["supported"])
        report = citation_mod.validate_citations(answer.citations, page_counts=page_counts_map,
                                                chunk_lookup=chunk_lookup, page_text_lookup=page_text_lookup,
                                                answer_text=answer.text, logger=logger)
        payload = {"checks": rows, "total": total_cites, "valid_four_point": valid,
                   "accuracy_four_point": round(valid / total_cites, 4) if total_cites else 0.0,
                   "product_report_valid": report.valid, "product_report_accuracy": report.accuracy,
                   "product_report_reasons": list(report.reasons),
                   "page_in_golden_ratio": (round(sum(1 for row in rows if row["page_in_golden"])
                                                  / total_cites, 4) if total_cites else 0.0)}
        span.set_output({"total": total_cites, "valid_four_point": valid,
                         "product_valid": report.valid})
        return payload


def evaluate_question(engine: QAEngine, item: dict[str, Any], *, cfg: AppConfig, top_k: int,
                      page_counts_map: dict[str, int], corpus_map: dict[str, str], logger: Any,
                      warm: bool) -> dict[str, Any]:
    """单题端到端评估：问答 → 召回命中 → 引用校验 → 闸门/数值信号断言 → 结构化日志。"""
    qid = int(item["id"])
    started = time.perf_counter()
    with logger.enter("evaluate_question", {"qid": qid, "question": str(item["question"])[:60],
                                            "top_k": top_k, "warm": warm}) as span:
        answer = engine.ask(str(item["question"]), top_k=top_k)
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        retrieval = getattr(answer, "retrieval", None)
        chunks = list(getattr(retrieval, "chunks", []) or [])
        support_chunks = list(getattr(retrieval, "support_chunks", []) or [])
        context = chunks + support_chunks
        evidence_verbatim = str(item["evidence_verbatim"])
        hit_primary = bool(is_evidence_hit(context, evidence_verbatim))
        hit_topk = bool(is_evidence_hit(chunks, evidence_verbatim))
        hit_strict = bool(is_evidence_hit(context, str(item.get("evidence") or "")))
        alt = str(item.get("evidence_verbatim_alt") or "")
        hit_alt = bool(alt) and bool(is_evidence_hit(context, alt))

        citations = citation_page_checks(answer, item, cfg=cfg, page_counts_map=page_counts_map,
                                        chunk_lookup=engine.generator.chunk_lookup,
                                        page_text_lookup=engine.generator.page_text_lookup, logger=logger)

        body = citation_mod.answer_body(answer.text)
        required = list(item.get("required_substrings") or [])
        missing_required = [token for token in required if token not in answer.text]
        forbidden_hit = [token for token in (item.get("forbidden_substrings") or []) if token in answer.text]

        info = understand(str(item["question"]), [], cfg=cfg, llm=None, logger=logger)
        expects_numeric = bool(info.get("expects_numeric"))
        gate = subject_gate(str(item["question"]), answer.text, context, cfg=cfg, logger=logger)

        row = {
            "id": qid,
            "question": str(item["question"]),
            "corpus": str(item.get("corpus") or ""),
            "expected_file": corpus_map.get(str(item.get("corpus") or ""), ""),
            "category": str(item.get("category") or ""),
            "answer": answer.text,
            "answer_body": body,
            "citations": [cite.render(language=answer.language) for cite in answer.citations],
            "citation_pages": [int(cite.page) for cite in answer.citations],
            "is_unknown": bool(answer.is_unknown),
            "unknown_reason": answer.unknown_reason,
            "language": answer.language,
            "backend": answer.backend,
            "model": getattr(answer, "model", ""),
            "first_token_ms": round(float(answer.first_token_ms or 0.0), 2),
            "total_ms": round(float(answer.total_ms or 0.0), 2),
            "wall_ms": wall_ms,
            "warm": bool(warm),
            "retrieval_hit": hit_primary,
            "retrieval_hit_topk_only": hit_topk,
            "retrieval_hit_alt_evidence": hit_alt,
            "retrieval_hit_strict_golden_evidence": hit_strict,
            "top_chunks": [str(getattr(chunk, "chunk_id", "")) for chunk in chunks],
            "top_pages": sorted({int(getattr(chunk, "page", 0)) for chunk in chunks}),
            "support_chunks": [str(getattr(chunk, "chunk_id", "")) for chunk in support_chunks],
            "citation_check": citations,
            "required_substrings": required,
            "missing_required": missing_required,
            "forbidden_hit": forbidden_hit,
            "expects_numeric": expects_numeric,
            "numeric_signal_expected": bool(item.get("numeric_signal_expected")),
            "field_type": info.get("field_type"),
            "subject_expected": gate.expected,
            "subject_expected_golden": str(item.get("subject_expectation") or "any"),
            "subject_gate_ok": bool(gate.ok),
            "subject_gate_counted": bool(gate.counted),
            "subject_gate_counted_expected": item.get("subject_gate_counted"),
            "subject_gate_reason": gate.reason,
            "subject_allowed": list(gate.allowed),
            "subject_found": list(gate.found),
            "subject_leaked": list(gate.leaked),
            "golden_answer": str(item["answer"]),
            "evidence_verbatim": evidence_verbatim,
            "evidence_pages": list(item.get("evidence_pages") or []),
            "expected_citation_quote": str(item.get("citation_quote") or ""),
            "trace_id": getattr(answer, "trace_id", ""),
            "error": None,
        }
        logger.log_event("eval.question", trace_id=row["trace_id"], qid=qid, category=row["category"],
                         is_unknown=row["is_unknown"], answer_chars=len(answer.text),
                         citations=row["citations"], first_token_ms=row["first_token_ms"],
                         total_ms=row["total_ms"], wall_ms=wall_ms,
                         retrieval_hit=hit_primary, retrieval_hit_topk=hit_topk,
                         citation_valid=citations["valid_four_point"], citation_total=citations["total"],
                         expects_numeric=expects_numeric, subject_expected=gate.expected,
                         subject_gate_counted=bool(gate.counted), subject_gate_ok=bool(gate.ok),
                         missing_required=missing_required, forbidden_hit=forbidden_hit)
        span.set_output({"qid": qid, "chars": len(answer.text), "first_token_ms": row["first_token_ms"],
                         "hit": hit_primary, "citation_valid": citations["valid_four_point"]})
        return row


def evaluate_unknown(engine: QAEngine, item: dict[str, Any], *, top_k: int, logger: Any) -> dict[str, Any]:
    """单条无关问题：是否统一回「不清楚」，以及是否违规给出被禁内容。"""
    qid = str(item.get("id"))
    with logger.enter("evaluate_unknown", {"qid": qid, "question": str(item["question"])[:60]}) as span:
        file_names = list(item.get("file_names") or []) or None
        answer = engine.ask(str(item["question"]), file_names=file_names, top_k=top_k)
        forbidden_hit = [token for token in (item.get("must_not_contain") or []) if token in answer.text]
        refused = bool(answer.is_unknown) or answer.text.strip() == citation_mod.UNKNOWN_TEXT
        row = {"id": qid, "question": str(item["question"]), "answer": answer.text,
               "is_unknown": bool(answer.is_unknown), "unknown_reason": answer.unknown_reason,
               "refused": refused, "forbidden_hit": forbidden_hit,
               "first_token_ms": round(float(answer.first_token_ms or 0.0), 2),
               "total_ms": round(float(answer.total_ms or 0.0), 2),
               "reason_in_expected": (answer.unknown_reason in
                                      list(item.get("reason_expected") or [])) if answer.unknown_reason else None}
        logger.log_event("eval.unknown", qid=qid, refused=refused, is_unknown=row["is_unknown"],
                         unknown_reason=row["unknown_reason"], forbidden_hit=forbidden_hit,
                         answer_digest=answer.text[:80])
        span.set_output({"qid": qid, "refused": refused, "forbidden_hit": forbidden_hit})
        return row


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def percentile_nearest_rank(values: Sequence[float], q: float) -> float | None:
    """最近秩法百分位（与工单2 基线 p95 == max 的口径一致；n=10 时线性插值会给出中值，故不用）。"""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, -(-int(round(q * 1000)) * len(ordered) // 1000) - 1))
    return round(ordered[index], 2)


def summarize(rows: Sequence[dict[str, Any]], unknown_rows: Sequence[dict[str, Any]], *,
              backend: dict[str, Any], cfg: AppConfig, run_tag: str, judge_source: str,
              mode: str, top_k: int, golden_path: Path, warmup: dict[str, Any],
              cold_first_question_ms: float | None, llm_preheat_ms: float | None) -> dict[str, Any]:
    """汇总指标（准确率 / 首字 / 引用 / 「不清楚」/ 召回命中 / 断言）。"""
    total = len(rows)
    correct = [row for row in rows if row.get("correct") is True]
    answered = [row for row in rows if not row.get("is_unknown")]
    firsts = [row["first_token_ms"] for row in answered]
    firsts_warm = [row["first_token_ms"] for row in answered if row.get("warm")]
    cite_total = sum(row["citation_check"]["total"] for row in rows)
    cite_valid = sum(row["citation_check"]["valid_four_point"] for row in rows)
    hit = sum(1 for row in rows if row["retrieval_hit"])
    hit_topk = sum(1 for row in rows if row["retrieval_hit_topk_only"])
    hit_strict = sum(1 for row in rows if row["retrieval_hit_strict_golden_evidence"])
    refused = sum(1 for row in unknown_rows if row.get("refused"))
    return {
        "total": total,
        "answered": len(answered),
        "unknown_ids": [row["id"] for row in rows if row.get("is_unknown")],
        "correct_count": len(correct),
        "incorrect_ids": [row["id"] for row in rows if row.get("correct") is not True],
        "accuracy": round(len(correct) / total, 4) if total else 0.0,
        "wrong_detail": {str(row["id"]): row.get("judge_reason") for row in rows
                         if row.get("correct") is not True},
        "accuracy_raw_text": (round(sum(1 for row in rows if row.get("correct_raw") is True) / total, 4)
                              if total else 0.0),
        "incorrect_ids_raw_text": [row["id"] for row in rows if row.get("correct_raw") is not True],
        "first_token_avg_ms": (round(sum(firsts) / len(firsts), 2) if firsts else None),
        "first_token_p95_ms": percentile_nearest_rank(firsts, 0.95),
        "first_token_max_ms": (round(max(firsts), 2) if firsts else None),
        "first_token_budget_ms": 3000.0,
        "first_token_within_budget": bool(firsts) and max(firsts) <= 3000.0,
        "first_token_avg_ms_warm_only": (round(sum(firsts_warm) / len(firsts_warm), 2) if firsts_warm else None),
        "first_token_max_ms_warm_only": (round(max(firsts_warm), 2) if firsts_warm else None),
        "first_token_over_1000ms_ids": [row["id"] for row in answered if row["first_token_ms"] > 1000.0],
        "first_question_cold_ms": (round(float(cold_first_question_ms), 2)
                                   if cold_first_question_ms is not None else None),
        "llm_preheat_cold_ms": (round(float(llm_preheat_ms), 2) if llm_preheat_ms is not None else None),
        "total_avg_ms": (round(sum(row["total_ms"] for row in rows) / total, 2) if total else None),
        "citation_total": cite_total,
        "citation_valid": cite_valid,
        "citation_accuracy": round(cite_valid / cite_total, 4) if cite_total else 0.0,
        "citation_product_accuracy": (round(sum(row["citation_check"]["product_report_valid"] for row in rows)
                                            / cite_total, 4) if cite_total else 0.0),
        "retrieval_hit_count": hit,
        "retrieval_hit_rate": round(hit / total, 4) if total else 0.0,
        "retrieval_hit_topk_only": hit_topk,
        "retrieval_hit_strict_golden_evidence": hit_strict,
        "retrieval_hit_strict_ratio": round(hit_strict / total, 4) if total else 0.0,
        "retrieval_miss_ids": [row["id"] for row in rows if not row["retrieval_hit"]],
        "retrieval_miss_strict_ids": [row["id"] for row in rows if not row["retrieval_hit_strict_golden_evidence"]],
        "unknown_total": len(unknown_rows),
        "unknown_refused": refused,
        "unknown_accuracy": (round(refused / len(unknown_rows), 4) if unknown_rows else None),
        "unknown_failed_ids": [row["id"] for row in unknown_rows if not row.get("refused")],
        "unknown_forbidden_ids": [row["id"] for row in unknown_rows if row.get("forbidden_hit")],
        "required_missing_ids": [row["id"] for row in rows if row["missing_required"]],
        "forbidden_hit_ids": [row["id"] for row in rows if row["forbidden_hit"]],
        "numeric_signal_ids": [row["id"] for row in rows if row["expects_numeric"]],
        "numeric_signal_expected_ids": [row["id"] for row in rows if row["numeric_signal_expected"]],
        "numeric_signal_mismatch_ids": [row["id"] for row in rows
                                        if row["expects_numeric"] != row["numeric_signal_expected"]],
        "subject_distribution": {name: sum(1 for row in rows if row["subject_expected"] == name)
                                 for name in ("any", "person", "organization")},
        "subject_gate_rows": {str(row["id"]): {"expected": row["subject_expected"],
                                               "counted": row["subject_gate_counted"],
                                               "ok": row["subject_gate_ok"], "reason": row["subject_gate_reason"],
                                               "found": len(row["subject_found"]),
                                               "leaked": list(row["subject_leaked"])}
                              for row in rows if row["subject_expected"] != "any" or int(row["id"]) in (34, 793)},
        "subject_gate_counted_mismatch_ids": [row["id"] for row in rows
                                              if row["subject_gate_counted"] != bool(row["subject_gate_counted_expected"])
                                              and row["subject_gate_counted_expected"] is not None],
        "backend": backend,
        "judge_source": judge_source,
        "judge_fuzzy_threshold": FUZZY_THRESHOLD,
        "mode": mode,
        "top_k": top_k,
        "run_tag": run_tag,
        "golden_path": str(golden_path),
        "golden_sha256_16": hashlib.sha256(golden_path.read_bytes()).hexdigest().upper()[:16],
        "config": {"retrieval": {"top_k": cfg.retrieval.top_k, "vector_k": cfg.retrieval.vector_k,
                                 "bm25_k": cfg.retrieval.bm25_k, "rrf_k": cfg.retrieval.rrf_k,
                                 "rank_rescue_weight": cfg.retrieval.rank_rescue_weight},
                   "answer": {"min_score": cfg.answer.min_score,
                              "min_keyword_coverage": cfg.answer.min_keyword_coverage,
                              "enable_subject_gate": cfg.answer.enable_subject_gate},
                   "chunk": {"size": cfg.chunk.size, "overlap": cfg.chunk.overlap}},
        "warmup": warmup,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ragas": {"status": RAGAS_NOT_RUN, "metrics": None,
                  "note": "ragas 依赖不可用（本机断网）→ 四项指标一律 null，禁止伪造数值"},
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], *, run_tag: str, generated_at: str,
              logger: Any) -> int:
    """标准库 ``csv`` 落盘逐题结果（一行一题，列表字段序列化为 JSON 字符串）。"""
    with logger.enter("write_csv", {"path": str(path), "rows": len(rows)}) as span:
        columns = ["id", "category", "corpus", "question", "correct", "judge_reason", "judge_source",
                   "is_unknown", "unknown_reason", "answer_body", "citations", "citation_pages",
                   "retrieval_hit", "retrieval_hit_topk_only", "retrieval_hit_strict_golden_evidence",
                   "first_token_ms", "total_ms", "wall_ms", "warm",
                   "citation_total", "citation_valid", "citation_accuracy",
                   "missing_required", "forbidden_hit", "expects_numeric", "numeric_signal_expected",
                   "subject_expected", "subject_gate_counted", "subject_gate_ok", "subject_gate_reason",
                   "top_pages", "top_chunks", "support_chunks",
                   "golden_answer", "evidence_verbatim", "evidence_pages", "backend", "model", "trace_id",
                   "run_tag", "generated_at"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                flat["citation_total"] = row["citation_check"]["total"]
                flat["citation_valid"] = row["citation_check"]["valid_four_point"]
                flat["citation_accuracy"] = row["citation_check"]["accuracy_four_point"]
                flat["top_pages"] = json.dumps(row["top_pages"], ensure_ascii=False)
                flat["top_chunks"] = json.dumps(row["top_chunks"], ensure_ascii=False)
                flat["support_chunks"] = json.dumps(row["support_chunks"], ensure_ascii=False)
                flat["citations"] = json.dumps(row["citations"], ensure_ascii=False)
                flat["citation_pages"] = json.dumps(row["citation_pages"], ensure_ascii=False)
                flat["missing_required"] = json.dumps(row["missing_required"], ensure_ascii=False)
                flat["forbidden_hit"] = json.dumps(row["forbidden_hit"], ensure_ascii=False)
                flat["evidence_pages"] = json.dumps(row["evidence_pages"], ensure_ascii=False)
                flat["run_tag"] = run_tag
                flat["generated_at"] = generated_at
                flat["judge_source"] = row.get("judge_source")
                writer.writerow(flat)
        span.set_output({"bytes": path.stat().st_size, "rows": len(rows)})
        return len(rows)


def write_ragas_report(path: Path, summary: dict[str, Any], *, logger: Any) -> None:
    """RAGAS 报告：**首屏必须**可见「RAGAS 未运行（依赖不可用，本机断网）」+ 确定性替代指标。"""
    with logger.enter("write_ragas_report", {"path": str(path)}) as span:
        lines = [
            "# RAGAS 评估报告",
            "",
            f"## ⚠️ {RAGAS_NOT_RUN}",
            "",
            f"> 工单：{WORK_ORDER}　生成时间：{summary['generated_at']}　运行标记：`{summary['run_tag']}`",
            "> **本报告不含任何 RAGAS 数值**：`ragas`（及其 `langchain*` 依赖）在本机未安装且**断网无法安装**，",
            "> 四项指标（faithfulness / answer_relevancy / context_precision / context_recall）一律为 `null`。",
            "> 与工单2 基线对齐：`工单2/优化/基线/baseline_metrics.json` 的 `ragas` 字段同样是",
            "> 「未运行（依赖不可用且本机断网）」、`eval_records` 中四项均为 `null` → **前后两次都未运行 RAGAS**，",
            "> 该口径一致，不存在「基线有、优化后没有」的缺失。",
            "",
            "## 1. 依赖探测（真实执行，失败即如实记录）",
            "",
            "| 包 | import 结果 | 说明 |",
            "| --- | --- | --- |",
        ]
        probes = probe_ragas_dependencies(logger=logger)
        for row in probes:
            lines.append(f"| `{row['package']}` | {row['import']} | {row['note']} |")
        unknown_value = "null（未提供无关问题集）"
        if summary["unknown_accuracy"] is not None:
            unknown_value = (f"{summary['unknown_refused']}/{summary['unknown_total']} = "
                             f"{summary['unknown_accuracy'] * 100:.1f}%")
        lines += [
            "",
            "## 2. 确定性替代指标（本工单实际使用，全部可复算）",
            "",
            "| 替代指标 | 值 | 对应的 RAGAS 关注点 | 计算口径 |",
            "| --- | --- | --- | --- |",
            f"| 准确率（14 题，工单1 Evaluator 口径） | {summary['correct_count']}/{summary['total']} = "
            f"{summary['accuracy'] * 100:.1f}% | answer_relevancy / faithfulness | 五步确定性判分（子串/带单位数值/"
            f"实体/比例/二元组相似度 ≥0.62） |",
            f"| 召回命中率（证据原文落在返回块） | {summary['retrieval_hit_count']}/{summary['total']} = "
            f"{summary['retrieval_hit_rate'] * 100:.1f}% | context_recall | `is_evidence_hit(chunks, "
            f"evidence_verbatim)` |",
            f"| 引用可回溯（四点校验） | {summary['citation_valid']}/{summary['citation_total']} = "
            f"{summary['citation_accuracy'] * 100:.1f}% | faithfulness（引用可核） | 页码范围 + 文件在语料 + "
            f"块可查且页一致 + 引用处有支撑原文 |",
            f"| 「不清楚」正确率（无关问题集） | {unknown_value} | context_precision（负例侧） | "
            f"无关问题必须回「不清楚」且不得含禁用串 |",
            f"| 首字延迟 max / p95 | {summary['first_token_max_ms']} ms / {summary['first_token_p95_ms']} ms "
            f"| ——（时延，RAGAS 不覆盖） | 逐题 LLM 首 token（预热后口径） |",
            "",
            "## 3. 未运行 RAGAS 的影响与替代理由",
            "",
            "1. **为什么不用 RAGAS**：`ragas` 需要 `langchain*` / `datasets` / 联网下载指标提示词与 judge 模型，"
            "本机无网络、无对应 wheel，装不上（探测证据见第 1 节）。",
            "2. **为什么替代指标足够**：RAGAS 的四项本质是「答案是否被上下文支持 / 上下文是否召回了答案依据」，"
            "本工单用**可复算的确定性判据**覆盖同一关注点：`is_evidence_hit` 判定证据原文是否落在返回块（= recall），"
            "`answer_support_check` 判定答案能否在引用处核实（= faithfulness 的可核版本），"
            "答案与金标准的五步判分（= answer_relevancy 的确定性代理）。",
            "3. **诚实声明**：以上替代指标与 RAGAS 的定义**不等价**，不能相互换算；"
            "任何把替代数值标注为 RAGAS 数值的做法都是伪造，本报告不做。",
            "",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        span.set_output({"bytes": path.stat().st_size, "probes": len(probes)})


def probe_ragas_dependencies(*, logger: Any) -> list[dict[str, str]]:
    """真实探测 RAGAS 依赖链的 import 结果（失败如实记录，绝不伪造可用性）。"""
    with logger.enter("probe_ragas_dependencies", {}) as span:
        packages = ["ragas", "langchain_openai", "langchain_community", "datasets", "pandas"]
        rows: list[dict[str, str]] = []
        for name in packages:
            try:
                module = __import__(name)
                rows.append({"package": name, "import": "✅ 可导入",
                             "note": f"版本 {getattr(module, '__version__', 'unknown')}"})
            except Exception as exc:  # noqa: BLE001 —— 探测失败是预期结果，如实记录
                rows.append({"package": name, "import": "❌ 不可导入（ModuleNotFoundError/其它）",
                             "note": f"{type(exc).__name__}: {exc}".replace("|", "/")[:120]})
            logger.log_event("ragas.probe", package=name, status=rows[-1]["import"])
        span.set_output({"probes": len(rows), "importable": sum(1 for row in rows if "✅" in row["import"])})
        return rows


def write_accuracy_reports(json_path: Path, md_path: Path, summary: dict[str, Any],
                           rows: Sequence[dict[str, Any]], unknown_rows: Sequence[dict[str, Any]],
                           *, logger: Any) -> None:
    """落盘 ``accuracy_report.json`` 与 ``accuracy_report.md``（指标 + 断言 + 逐题明细）。"""
    with logger.enter("write_accuracy_reports", {"json": str(json_path), "md": str(md_path)}) as span:
        payload = {"summary": summary, "results": list(rows), "unknown_results": list(unknown_rows),
                   "work_order": WORK_ORDER}
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ok = "✅" if summary["correct_count"] == summary["total"] else "⚠️"
        unknown_value = "null"
        if summary["unknown_accuracy"] is not None:
            unknown_value = (f"{summary['unknown_refused']}/{summary['unknown_total']} = "
                             f"{summary['unknown_accuracy'] * 100:.1f}%")
        lines = [f"# 14 题评估报告（{summary['mode']} / run={summary['run_tag']}）", "",
                 f"> 工单：{WORK_ORDER}　生成时间：{summary['generated_at']}",
                 f"> 金标准：`{summary['golden_path']}`（sha256_16 `{summary['golden_sha256_16']}`）",
                 f"> 判分口径：{summary['judge_source']}（FUZZY_THRESHOLD={summary['judge_fuzzy_threshold']}）",
                 f"> 生成后端：{summary['backend'].get('name')} / {summary['backend'].get('model')}",
                 "", f"> ⚠️ **{RAGAS_NOT_RUN}** —— 本报告与 `ragas_report.md` 均不含任何 RAGAS 数值。", "",
                 "## 1. 指标", "", "| 指标 | 值 |", "| --- | --- |",
                 f"| 准确率（{summary['total']} 题） | {ok} **{summary['correct_count']}/{summary['total']} = "
                 f"{summary['accuracy'] * 100:.1f}%**（错题：{summary['incorrect_ids'] or '无'}） |",
                 f"| 作答题数（非「不清楚」） | {summary['answered']}/{summary['total']}"
                 f"（误拒答：{summary['unknown_ids'] or '无'}） |",
                 f"| 召回命中率（证据原文落在返回块） | {summary['retrieval_hit_count']}/{summary['total']} = "
                 f"{summary['retrieval_hit_rate'] * 100:.1f}%（未命中：{summary['retrieval_miss_ids'] or '无'}） |",
                 f"| 召回命中（仅 top-k ∪ 支持块，主口径） | {summary['retrieval_hit_count']}/{summary['total']} |",
                 f"| 召回命中（仅 top-k） | {summary['retrieval_hit_topk_only']}/{summary['total']} |",
                 f"| 召回命中（严格单证据 = golden `evidence` 原值） | "
                 f"{summary['retrieval_hit_strict_golden_evidence']}/{summary['total']}"
                 f"（未命中：{summary['retrieval_miss_strict_ids'] or '无'}） |",
                 f"| 引用可回溯（四点） | {summary['citation_valid']}/{summary['citation_total']} = "
                 f"{summary['citation_accuracy'] * 100:.1f}% |",
                 f"| 首字 max / 平均 / p95（预热后） | {summary['first_token_max_ms']} ms / "
                 f"{summary['first_token_avg_ms']} ms / {summary['first_token_p95_ms']} ms"
                 f"（预算 ≤{int(summary['first_token_budget_ms'])} ms，达标={summary['first_token_within_budget']}） |",
                 f"| 首字 >1000 ms 的题 | {summary['first_token_over_1000ms_ids'] or '无'} |",
                 f"| 「不清楚」正确率（无关问题集 {summary['unknown_total']} 条） | {unknown_value} |",
                 "", "## 2. 断言（设计/验收标准.md §3.3 / §3.6 的 N-6/N-7/N-8）", "",
                 f"- 数值信号题（expects_numeric=True）：{summary['numeric_signal_ids']}"
                 f"（期望 {summary['numeric_signal_expected_ids']}，不一致："
                 f"{summary['numeric_signal_mismatch_ids'] or '无'}）",
                 f"- 主体类型分布：{summary['subject_distribution']}",
                 f"- 主体闸门：`{json.dumps(summary['subject_gate_rows'], ensure_ascii=False)}`",
                 f"- required_substrings 缺失题：{summary['required_missing_ids'] or '无'}；"
                 f"禁用串命中题：{summary['forbidden_hit_ids'] or '无'}",
                 "", "## 3. 逐题明细", "",
                 "| id | 判定 | 首字 ms | 引用 | 召回命中 | 答案摘要（前 60 字） |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for row in sorted(rows, key=lambda item: int(item["id"])):
            lines.append(f"| {row['id']} | {'✅' if row.get('correct') is True else '❌'} | "
                         f"{row['first_token_ms']} | {', '.join(row['citations']) or '—'} | "
                         f"{'✅' if row['retrieval_hit'] else '❌'} | "
                         f"{row['answer_body'][:60].replace('|', '/')} |")
        lines += ["", "## 4. 逐题答案与判分（完整）", ""]
        for row in sorted(rows, key=lambda item: int(item["id"])):
            lines += [f"### 题 {row['id']}：{row['question']}", "",
                      f"- 答案：{row['answer_body']}",
                      f"- 引用：{row['citations']}",
                      f"- 判定：{'✅ 正确' if row.get('correct') is True else '❌ 错误'}（{row.get('judge_reason')}）",
                      f"- 首字 {row['first_token_ms']} ms / 总 {row['total_ms']} ms / 召回命中 "
                      f"{row['retrieval_hit']}（top-k 仅 {row['retrieval_hit_topk_only']}）",
                      f"- 证据原文（逐字）：{row['evidence_verbatim']}",
                      f"- golden：{row['golden_answer']}", ""]
        lines += ["## 5. 无关问题（「不清楚」正确率）", "",
                  "| id | 是否拒答 | unknown_reason | 禁用串命中 | 回答摘要 |", "| --- | --- | --- | --- | --- |"]
        for row in unknown_rows:
            lines.append(f"| {row['id']} | {'✅' if row['refused'] else '❌'} | {row['unknown_reason']} | "
                         f"{row['forbidden_hit'] or '无'} | {row['answer'][:60].replace('|', '/')} |")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        span.set_output({"json_bytes": json_path.stat().st_size, "md_bytes": md_path.stat().st_size})


def mirror_products(paths: Sequence[Path], *, logger: Any) -> dict[str, str]:
    """把产物镜像到 ``研发/data/eval/eval_results/``（captain 指定的落地路径，内容逐字节一致）。"""
    with logger.enter("mirror_products", {"count": len(paths), "target": str(MIRROR_DIR)}) as span:
        MIRROR_DIR.mkdir(parents=True, exist_ok=True)
        digests: dict[str, str] = {}
        for source in paths:
            target = MIRROR_DIR / source.name
            target.write_bytes(source.read_bytes())
            digest = hashlib.sha256(target.read_bytes()).hexdigest().upper()[:16]
            digests[target.name] = digest
            logger.log_event("mirror.copy", source=str(source), target=str(target),
                             bytes=target.stat().st_size, sha256_16=digest)
        span.set_output({"mirrored": list(digests), "dir": str(MIRROR_DIR)})
        return digests


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_offline(golden: Sequence[dict[str, Any]], unknowns: Sequence[dict[str, Any]], *, cfg: AppConfig,
                top_k: int, limit: int, preheat_llm: bool, logger: Any) -> dict[str, Any]:
    """离线模式：进程内直接调 ``QAEngine``（含启动预热，首题冷启动单独记录）。"""
    with logger.enter("run_offline", {"questions": len(golden), "top_k": top_k,
                                      "preheat_llm": preheat_llm}) as span:
        engine = build_engine(cfg=cfg, warmup=True, logger=logger)
        backend = getattr(getattr(engine.generator, "llm", None), "backend", None)
        warmup = dict(engine.warmup_info or {})
        preheat_ms = None
        if preheat_llm and backend is not None:
            started = time.perf_counter()
            try:
                engine.generator.llm.generate_full("预热：请只回答「好」。", max_tokens=4, temperature=0.0,
                                                   logger=logger)
                preheat_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.log_event("eval.preheat_done", cold_ms=preheat_ms)
            except Exception as exc:  # noqa: BLE001 —— 预热失败必须留痕，不阻断评估
                logger.log_event("eval.preheat_failed", level="ERROR", error_type=type(exc).__name__,
                                 message=str(exc))
        items = list(golden)[:limit] if limit else list(golden)
        counts = page_counts(logger=logger)
        corpus_map = corpus_file_map(logger=logger)
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            rows.append(evaluate_question(engine, item, cfg=cfg, top_k=top_k, page_counts_map=counts,
                                          corpus_map=corpus_map, logger=logger,
                                          warm=bool(preheat_llm or index > 0)))
        unknown_rows = [evaluate_unknown(engine, item, top_k=top_k, logger=logger) for item in unknowns]
        cold_first = rows[0]["first_token_ms"] if rows else None
        span.set_output({"rows": len(rows), "unknown_rows": len(unknown_rows), "preheat_ms": preheat_ms})
        return {"rows": rows, "unknown_rows": unknown_rows, "warmup": warmup, "preheat_ms": preheat_ms,
                "cold_first_question_ms": cold_first,
                "backend": ({"name": backend.name, "model": backend.model, "base_url": backend.base_url}
                            if backend else {"name": "extractive", "model": "rule-based", "base_url": "local"})}


def http_ask(base_url: str, item: dict[str, Any], *, top_k: int, logger: Any) -> dict[str, Any]:
    """在线模式：POST ``/api/ask``（端到端首字由服务端返回，另记本地 wall_ms）。"""
    import urllib.request

    with logger.enter("http_ask", {"base_url": base_url, "qid": item.get("id")}) as span:
        body = json.dumps({"question": item["question"], "top_k": top_k, "session_id": None},
                          ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(base_url.rstrip("/") + "/api/ask", data=body,
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        payload["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        span.set_output({"chars": len(str(payload.get("text") or "")),
                         "first_token_ms": payload.get("first_token_ms")})
        return payload


def run_online(golden: Sequence[dict[str, Any]], unknowns: Sequence[dict[str, Any]], *, cfg: AppConfig,
               base_url: str, top_k: int, limit: int, logger: Any) -> dict[str, Any]:
    """在线模式：经 HTTP 调服务（首字为端到端口径）。"""
    with logger.enter("run_online", {"base_url": base_url, "questions": len(golden)}) as span:
        import sqlite3

        conn = sqlite3.connect(str(cfg.paths.index_dir / "rag.sqlite3"))
        conn.row_factory = sqlite3.Row

        def chunk_lookup(chunk_id: str) -> Any:
            row = conn.execute("SELECT chunk_id,file_name,page,type,content FROM chunks WHERE chunk_id=?",
                               (str(chunk_id),)).fetchone()
            if row is None:
                return None
            return type("ChunkView", (), {"chunk_id": row["chunk_id"], "file_name": row["file_name"],
                                          "page": int(row["page"]), "type": row["type"],
                                          "content": row["content"]})()

        def page_text(chunk_id_or_file: str, page: int | None = None) -> str:
            if page is None:
                return ""
            row = conn.execute("SELECT text FROM pages WHERE file_name=? AND page=?",
                               (str(chunk_id_or_file), int(page))).fetchone()
            return str(row["text"]) if row else ""

        rows: list[dict[str, Any]] = []
        items = list(golden)[:limit] if limit else list(golden)
        counts = page_counts(logger=logger)
        for item in items:
            payload = http_ask(base_url, item, top_k=top_k, logger=logger)
            citations = [citation_mod.Citation.from_dict(entry) for entry in payload.get("citations") or []]
            text = str(payload.get("text") or "")
            answer = type("AnswerView", (), {
                "text": text, "citations": citations, "is_unknown": bool(payload.get("is_unknown")),
                "unknown_reason": payload.get("unknown_reason"), "language": payload.get("language", "zh"),
                "first_token_ms": float(payload.get("first_token_ms") or 0.0),
                "total_ms": float(payload.get("total_ms") or 0.0), "backend": payload.get("backend", ""),
                "model": payload.get("model", ""), "trace_id": payload.get("trace_id", ""),
                "retrieval": None})()
            chunk_views = [chunk_lookup(entry.get("chunk_id")) for entry in payload.get("chunks") or []]
            chunk_views = [view for view in chunk_views if view is not None]
            answer.retrieval = type("RetrievalView", (), {"chunks": chunk_views, "support_chunks": []})()
            row = evaluate_question_stub(answer, item, cfg=cfg, top_k=top_k, page_counts_map=counts,
                                        chunk_lookup=chunk_lookup, page_text=page_text, logger=logger,
                                        wall_ms=payload.get("wall_ms"))
            rows.append(row)
        unknown_rows: list[dict[str, Any]] = []
        for item in unknowns:
            payload = http_ask(base_url, item, top_k=top_k, logger=logger)
            text = str(payload.get("text") or "")
            forbidden = [token for token in (item.get("must_not_contain") or []) if token in text]
            refused = bool(payload.get("is_unknown")) or text.strip() == citation_mod.UNKNOWN_TEXT
            unknown_rows.append({"id": item.get("id"), "question": item["question"], "answer": text,
                                 "is_unknown": bool(payload.get("is_unknown")),
                                 "unknown_reason": payload.get("unknown_reason"), "refused": refused,
                                 "forbidden_hit": forbidden,
                                 "first_token_ms": payload.get("first_token_ms"),
                                 "total_ms": payload.get("total_ms"), "reason_in_expected": None})
        conn.close()
        span.set_output({"rows": len(rows), "unknown_rows": len(unknown_rows)})
        cold_first = rows[0]["first_token_ms"] if rows else None
        return {"rows": rows, "unknown_rows": unknown_rows, "warmup": {}, "preheat_ms": None,
                "cold_first_question_ms": cold_first,
                "backend": {"name": "online-http", "model": "", "base_url": base_url}}


def evaluate_question_stub(answer: Any, item: dict[str, Any], *, cfg: AppConfig, top_k: int,
                           page_counts_map: dict[str, int], chunk_lookup: Any, page_text: Any,
                           logger: Any, wall_ms: float | None) -> dict[str, Any]:
    """在线模式的单题统计（与离线同口径，仅数据来源不同）。"""
    qid = int(item["id"])
    with logger.enter("evaluate_question_online", {"qid": qid}) as span:
        chunks = list(getattr(getattr(answer, "retrieval", None), "chunks", []) or [])
        evidence_verbatim = str(item["evidence_verbatim"])
        hit = bool(is_evidence_hit(chunks, evidence_verbatim))
        citations = citation_page_checks(answer, item, cfg=cfg, page_counts_map=page_counts_map,
                                        chunk_lookup=chunk_lookup, page_text_lookup=page_text, logger=logger)
        body = citation_mod.answer_body(answer.text)
        required = list(item.get("required_substrings") or [])
        info = understand(str(item["question"]), [], cfg=cfg, llm=None, logger=logger)
        gate = subject_gate(str(item["question"]), answer.text, chunks, cfg=cfg, logger=logger)
        row = {
            "id": qid, "question": str(item["question"]), "corpus": str(item.get("corpus") or ""),
            "category": str(item.get("category") or ""), "answer": answer.text, "answer_body": body,
            "citations": [cite.render(language=getattr(answer, "language", "zh")) for cite in answer.citations],
            "citation_pages": [int(cite.page) for cite in answer.citations],
            "is_unknown": bool(answer.is_unknown), "unknown_reason": answer.unknown_reason,
            "language": getattr(answer, "language", "zh"), "backend": getattr(answer, "backend", ""),
            "model": getattr(answer, "model", ""),
            "first_token_ms": round(float(getattr(answer, "first_token_ms", 0.0)), 2),
            "total_ms": round(float(getattr(answer, "total_ms", 0.0)), 2),
            "wall_ms": wall_ms, "warm": True,
            "retrieval_hit": hit, "retrieval_hit_topk_only": hit,
            "retrieval_hit_alt_evidence": bool(item.get("evidence_verbatim_alt"))
            and bool(is_evidence_hit(chunks, str(item["evidence_verbatim_alt"]))),
            "retrieval_hit_strict_golden_evidence": bool(is_evidence_hit(chunks, str(item.get("evidence") or ""))),
            "top_chunks": [view.chunk_id for view in chunks],
            "top_pages": sorted({int(view.page) for view in chunks}),
            "support_chunks": [],
            "citation_check": citations,
            "required_substrings": required,
            "missing_required": [token for token in required if token not in answer.text],
            "forbidden_hit": [token for token in (item.get("forbidden_substrings") or []) if token in answer.text],
            "expects_numeric": bool(info.get("expects_numeric")),
            "numeric_signal_expected": bool(item.get("numeric_signal_expected")), "field_type": info.get("field_type"),
            "subject_expected": gate.expected, "subject_expected_golden": str(item.get("subject_expectation") or "any"),
            "subject_gate_ok": bool(gate.ok), "subject_gate_counted": bool(gate.counted),
            "subject_gate_reason": gate.reason, "subject_allowed": list(gate.allowed),
            "subject_found": list(gate.found), "subject_leaked": list(gate.leaked),
            "golden_answer": str(item["answer"]), "evidence_verbatim": evidence_verbatim,
            "evidence_pages": list(item.get("evidence_pages") or []),
            "expected_citation_quote": str(item.get("citation_quote") or ""),
            "trace_id": getattr(answer, "trace_id", ""), "error": None,
        }
        logger.log_event("eval.question", qid=qid, mode="online", is_unknown=row["is_unknown"],
                         citations=row["citations"], first_token_ms=row["first_token_ms"],
                         total_ms=row["total_ms"], wall_ms=wall_ms, retrieval_hit=hit,
                         citation_valid=citations["valid_four_point"], citation_total=citations["total"])
        span.set_output({"qid": qid, "first_token_ms": row["first_token_ms"], "hit": hit})
        return row


def attach_verdicts(rows: list[dict[str, Any]], *, logger: Any) -> str:
    """批量判分并写回 ``correct`` / ``judge_reason`` / ``judge_source``（正文口径 + 原文本口径）。"""
    with logger.enter("attach_verdicts", {"rows": len(rows)}) as span:
        pairs_body = [{"id": row["id"], "answer": row["answer_body"], "golden": row["golden_answer"]}
                      for row in rows]
        pairs_raw = [{"id": row["id"], "answer": row["answer"], "golden": row["golden_answer"]} for row in rows]
        body_verdicts, source = judge_pairs(pairs_body, logger=logger)
        raw_verdicts, _ = judge_pairs(pairs_raw, logger=logger)
        for row in rows:
            verdict = body_verdicts.get(row["id"], {"ok": None, "reason": "未判分"})
            row["correct"] = bool(verdict.get("ok")) if verdict.get("ok") is not None else None
            row["judge_reason"] = verdict.get("reason")
            row["judge_source"] = verdict.get("source") or source
            raw = raw_verdicts.get(row["id"], {})
            row["correct_raw"] = bool(raw.get("ok")) if raw.get("ok") is not None else None
            row["judge_reason_raw"] = raw.get("reason")
        span.set_output({"source": source, "correct": sum(1 for row in rows if row["correct"] is True)})
        return source


def main(argv: Sequence[str] | None = None) -> int:
    """入口：评估 14 题 + 无关问题，落盘 `qa_results.csv` / `accuracy_report.*` / `ragas_report.md`。"""
    parser = argparse.ArgumentParser(description="T9 14 题评估（设计/接口设计.md §6.2 CLI 契约）")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="金标准 JSONL（默认 研发/data/eval/golden_qa.jsonl）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="产物目录（默认 优化/评估结果）")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--base-url", default="http://127.0.0.1:8600")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", choices=("auto", "ollama", "openai", "extractive"), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--unknown", default=str(DEFAULT_UNKNOWN), help="无关问题集（算「不清楚」正确率）")
    parser.add_argument("--run-tag", default="", help="运行标记（非空时产物名追加 .<tag>）")
    parser.add_argument("--preheat-llm", action="store_true", help="评估前显式预热 LLM（稳态口径）")
    parser.add_argument("--mirror", action="store_true", help="把产物镜像到 研发/data/eval/eval_results/")
    args = parser.parse_args(argv)

    if args.backend != "auto":
        os.environ["RAG_LLM__BACKEND"] = args.backend
    if args.log_level:
        os.environ["RAG_LOG__LEVEL"] = str(args.log_level).upper()

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("evaluate")
    started = time.perf_counter()
    suffix = f".{args.run_tag}" if args.run_tag else ""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"qa_results{suffix}.csv"
    json_path = out_dir / f"accuracy_report{suffix}.json"
    md_path = out_dir / f"accuracy_report{suffix}.md"
    ragas_path = out_dir / f"ragas_report{suffix}.md"

    with log.enter("main", {"mode": args.mode, "top_k": args.top_k, "backend": args.backend,
                            "run_tag": args.run_tag, "preheat_llm": args.preheat_llm}) as span:
        try:
            golden = load_golden(Path(args.golden), logger=log)
        except ValueError as exc:                                    # 入参/金标准错误 → 退出码 2
            log.log_event("eval.bad_golden", level="ERROR", error_type=type(exc).__name__, message=str(exc))
            print(f"❌ 金标准错误：{exc}")
            return 2
        unknowns = load_unknowns(Path(args.unknown), logger=log)
        if args.mode == "offline":
            outcome = run_offline(golden, unknowns, cfg=cfg, top_k=args.top_k, limit=args.limit,
                                  preheat_llm=args.preheat_llm, logger=log)
        else:
            outcome = run_online(golden, unknowns, cfg=cfg, base_url=args.base_url, top_k=args.top_k,
                                 limit=args.limit, logger=log)
        rows, unknown_rows = outcome["rows"], outcome["unknown_rows"]
        judge_source = attach_verdicts(rows, logger=log)
        summary = summarize(rows, unknown_rows, backend=outcome["backend"], cfg=cfg, run_tag=args.run_tag or "run1",
                            judge_source=judge_source, mode=args.mode, top_k=args.top_k,
                            golden_path=Path(args.golden), warmup=outcome["warmup"],
                            cold_first_question_ms=outcome["cold_first_question_ms"],
                            llm_preheat_ms=outcome["preheat_ms"])
        summary["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        write_csv(csv_path, rows, run_tag=summary["run_tag"], generated_at=summary["generated_at"], logger=log)
        write_accuracy_reports(json_path, md_path, summary, rows, unknown_rows, logger=log)
        write_ragas_report(ragas_path, summary, logger=log)
        products = [csv_path, json_path, md_path, ragas_path]
        mirrored: dict[str, str] = mirror_products(products, logger=log) if args.mirror else {}
        manifest = {
            "work_order": WORK_ORDER,
            "generated_at": summary["generated_at"],
            "run_tag": summary["run_tag"],
            "mode": summary["mode"],
            "products": {path.name: {"bytes": path.stat().st_size,
                                     "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest().upper()[:16]}
                         for path in products},
            "mirror_dir": (str(MIRROR_DIR) if args.mirror else ""),
            "mirror_sha256_16": mirrored,
        }
        manifest_path = out_dir / f"eval_manifest{suffix}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("=" * 78)
        print(f"模式 {args.mode} / run={summary['run_tag']} / 后端 {outcome['backend'].get('name')}"
              f"（{outcome['backend'].get('model')}）")
        print(f"准确率 {summary['correct_count']}/{summary['total']} = {summary['accuracy'] * 100:.1f}%"
              f"（错题 {summary['incorrect_ids'] or '无'}）")
        print(f"作答 {summary['answered']}/{summary['total']}；误拒答 {summary['unknown_ids'] or '无'}")
        print(f"召回命中 {summary['retrieval_hit_count']}/{summary['total']}"
              f"（严格 golden evidence 口径 {summary['retrieval_hit_strict_golden_evidence']}/"
              f"{summary['total']}，差集 {summary['retrieval_miss_strict_ids'] or '无'}）")
        print(f"引用可回溯 {summary['citation_valid']}/{summary['citation_total']} = "
              f"{summary['citation_accuracy'] * 100:.1f}%")
        print(f"首字 max={summary['first_token_max_ms']} ms（预算 3000，达标="
              f"{summary['first_token_within_budget']}）/ avg={summary['first_token_avg_ms']} / "
              f"p95={summary['first_token_p95_ms']}；>1000ms 题={summary['first_token_over_1000ms_ids'] or '无'}")
        print(f"不清楚正确率 {summary['unknown_accuracy']}（{summary['unknown_refused']}/"
              f"{summary['unknown_total']}）；judge={summary['judge_source']}")
        for path in products:
            print(f"产物 {path}（{path.stat().st_size} B）")
        print(f"清单 {manifest_path}")
        print(f"⚠️ {RAGAS_NOT_RUN}")
        span.set_output({"accuracy": summary["accuracy"], "first_token_max_ms": summary["first_token_max_ms"],
                         "csv_rows": len(rows)})
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 —— 评估中断：结构化日志 + 退出码 1
        try:
            get_logger("evaluate").log_event("eval.failed", level="ERROR", error_type=type(exc).__name__,
                                             message=str(exc), stack=__import__("traceback").format_exc())
        finally:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)
