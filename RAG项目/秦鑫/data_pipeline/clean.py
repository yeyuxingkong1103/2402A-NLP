from __future__ import annotations

import re
from typing import Any


PLACEHOLDER_CASE_PHRASES = ("案情内容暂缺", "本案案情内容暂缺", "详见原文链接", "详见原文")
PAGE_MARKER_RE = re.compile(r"^[－—–-]\s*\d+\s*[－—–-]$")
SEPARATOR_RE = re.compile(r"^[=_*#-]{8,}$")
HORIZONTAL_SPACE_RE = re.compile(r"[\t \u3000]+")
RECORD_FIELDS = {
    "civil_code_articles": ("id", "article_content"),
    "civil_interpretations": ("id", "content"),
    "civil_cases": ("case_id", "summary"),
    "civil_elements": ("serial_number", "case_summary"),
    "civil_evidence": ("evidence_id", "content"),
    "civil_processes": ("process_id", "content"),
    "civil_questions": ("question_id", "content"),
    "civil_citations": ("citation_id", "content"),
}


def clean_line(value: Any) -> str:
    return HORIZONTAL_SPACE_RE.sub(" ", str(value or "").replace("\ufeff", "").replace("\x00", "")).strip()


def clean_text(value: Any) -> str:
    lines: list[str] = []
    for raw_line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = clean_line(raw_line)
        compact = line.replace(" ", "")
        if not line or PAGE_MARKER_RE.fullmatch(compact) or SEPARATOR_RE.fullmatch(compact):
            continue
        lines.append(line)
    return "\n".join(lines)


def is_placeholder_case(title: str, summary: str) -> bool:
    content = f"{title}\n{summary}"
    return any(phrase in content for phrase in PLACEHOLDER_CASE_PHRASES)


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value) if "\n" in value or "\r" in value else clean_line(value)
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_value(item) for key, item in value.items()}
    return value


def _case_indexes(cases: list[dict]) -> tuple[set[str], dict[str, str]]:
    identifiers: set[str] = set()
    numbers: dict[str, str] = {}
    for row in cases:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            continue
        identifiers.add(case_id)
        case_number = _normalize_identifier(row.get("case_number"))
        if case_number:
            numbers.setdefault(case_number, case_id)
    return identifiers, numbers


def _resolve_case_id(row: dict, valid_ids: set[str], ids_by_number: dict[str, str]) -> str:
    current = str(row.get("case_id") or row.get("source_case_id") or "").strip()
    if current in valid_ids:
        return current
    for field in ("case_number", "source_case_number"):
        match = ids_by_number.get(_normalize_identifier(row.get(field)))
        if match:
            return match
    return ""


def clean_public_collections(collections: dict[str, list[dict]]) -> dict[str, list[dict]]:
    cleaned = {
        name: [{key: _clean_value(value) for key, value in row.items()} for row in rows if isinstance(row, dict)]
        for name, rows in collections.items()
    }
    cleaned["civil_cases"] = [
        row for row in cleaned.get("civil_cases", [])
        if not is_placeholder_case(str(row.get("title", "")), str(row.get("summary", "")))
    ]
    article_ids = {
        int(row["article_number"]): str(row["id"])
        for row in cleaned.get("civil_code_articles", [])
        if str(row.get("article_number", "")).isdigit()
    }
    valid_cases, cases_by_number = _case_indexes(cleaned.get("civil_cases", []))
    for row in cleaned.get("civil_elements", []):
        article_number = str(row.get("serial_number", "")).strip()
        if article_number.isdigit() and int(article_number) in article_ids:
            row["rag_doc_id"] = article_ids[int(article_number)]
        previous = str(row.get("case_id") or "").strip()
        row["case_id"] = _resolve_case_id(row, valid_cases, cases_by_number)
        if previous and previous != row["case_id"]:
            row["original_case_id"] = previous
        if "发条" in str(row.get("document_type", "")):
            row["document_type"] = "民法典法律要件"
    for row in cleaned.get("civil_evidence", []):
        previous = str(row.get("source_case_id") or "").strip()
        row["source_case_id"] = _resolve_case_id(row, valid_cases, cases_by_number)
        if previous and previous != row["source_case_id"]:
            row["original_source_case_id"] = previous
    return cleaned


def _article_ids(rows: list[dict]) -> set[str]:
    return {str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()}


def validate_public_records(collections: dict[str, list[dict]]) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for collection, rows in collections.items():
        if collection not in RECORD_FIELDS:
            raise ValueError(f"未知知识库：{collection}")
        primary_field, content_field = RECORD_FIELDS[collection]
        keys = [str(row.get(primary_field) or "").strip() for row in rows]
        if any(not key for key in keys):
            raise ValueError(f"{collection} 存在空主键")
        if len(keys) != len(set(keys)):
            raise ValueError(f"{collection} 存在主键重复")
        empty = sum(not str(row.get(content_field) or "").strip() for row in rows)
        corrupted = sum(str(row).count("�") for row in rows)
        separators = sum(bool(SEPARATOR_RE.search(line.replace(" ", ""))) for row in rows for line in str(row.get(content_field) or "").splitlines())
        if empty:
            raise ValueError(f"{collection} 存在 {empty} 条空正文")
        if corrupted:
            raise ValueError(f"{collection} 存在乱码替换字符")
        if separators:
            raise ValueError(f"{collection} 存在 {separators} 条装饰分隔线")
        report[collection] = {"record_count": len(rows), "empty_content": 0, "duplicate_primary_keys": 0, "replacement_characters": 0, "separator_lines": 0}

    articles = _article_ids(collections.get("civil_code_articles", []))
    cases = {str(row.get("case_id") or "").strip() for row in collections.get("civil_cases", [])} - {""}
    if "civil_cases" in report:
        placeholders = sum(is_placeholder_case(str(row.get("title", "")), str(row.get("summary", ""))) for row in collections["civil_cases"])
        report["civil_cases"]["placeholder_cases"] = placeholders
        if placeholders:
            raise ValueError(f"civil_cases 存在 {placeholders} 条占位案例")
    if "civil_elements" in report:
        bad_articles = sum(bool(row.get("rag_doc_id")) and str(row["rag_doc_id"]).strip() not in articles for row in collections["civil_elements"])
        bad_cases = sum(bool(row.get("case_id")) and str(row["case_id"]).strip() not in cases for row in collections["civil_elements"])
        report["civil_elements"].update(dangling_rag_doc_id=bad_articles, dangling_case_id=bad_cases)
        if bad_articles or bad_cases:
            raise ValueError(f"civil_elements 存在悬空关联：rag_doc_id={bad_articles}, case_id={bad_cases}")
    if "civil_evidence" in report:
        bad_cases = sum(bool(row.get("source_case_id")) and str(row["source_case_id"]).strip() not in cases for row in collections["civil_evidence"])
        report["civil_evidence"]["dangling_source_case_id"] = bad_cases
        if bad_cases:
            raise ValueError(f"civil_evidence 存在 {bad_cases} 条悬空 source_case_id")
    if "civil_citations" in report:
        all_ids = {
            "civil_code_articles": articles,
            "civil_cases": cases,
            "civil_interpretations": {str(row.get("id") or "").strip() for row in collections.get("civil_interpretations", [])},
            "civil_elements": {str(row.get("serial_number") or "").strip() for row in collections.get("civil_elements", [])},
            "civil_evidence": {str(row.get("evidence_id") or "").strip() for row in collections.get("civil_evidence", [])},
            "civil_processes": {str(row.get("process_id") or "").strip() for row in collections.get("civil_processes", [])},
            "civil_questions": {str(row.get("question_id") or "").strip() for row in collections.get("civil_questions", [])},
        }
        bad_source = sum(str(row.get("source_id") or "").strip() not in all_ids.get(str(row.get("source_collection") or ""), set()) for row in collections["civil_citations"])
        bad_target = sum(str(row.get("target_id") or "").strip() not in all_ids.get(str(row.get("target_collection") or ""), set()) for row in collections["civil_citations"])
        report["civil_citations"].update(bad_source_ids=bad_source, bad_target_ids=bad_target)
        if bad_source or bad_target:
            raise ValueError(f"civil_citations 存在无效引用：source_id={bad_source}, target_id={bad_target}")
    return report


__all__ = ["RECORD_FIELDS", "clean_line", "clean_public_collections", "clean_text", "is_placeholder_case", "validate_public_records"]
