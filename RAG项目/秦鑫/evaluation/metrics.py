from __future__ import annotations

from collections.abc import Iterable
from math import log2
from statistics import mean, median
from typing import Any


DEFAULT_KS = (1, 3, 5, 10)

ID_FIELDS = (
    "source_id",
    "id",
    "document_id",
    "chunk_id",
    "citation_id",
    "question_id",
    "case_id",
    "article_id",
)

RETURNED_FIELDS = ("returned", "returned_sources", "retrieval", "results", "evidence", "sources")
ANSWER_FIELDS = ("answer", "response", "output", "answer_text")


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("items", "rows", "results", "sources", "evidence", "returned", "citations"):
            if key in value:
                return _stable_sequence(value[key])
        return [value]
    if isinstance(value, (str, bytes)):
        return [value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def normalize_source_id(item: Any) -> str:
    if isinstance(item, dict):
        for field in ID_FIELDS:
            value = item.get(field)
            if value:
                return str(value).strip()
        collection = item.get("collection")
        for field in ("raw_id", "pk", "primary_key"):
            value = item.get(field)
            if collection and value:
                return f"{collection}:{value}".strip()
        return ""
    return str(item or "").strip()


def ranked_ids(items: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in _stable_sequence(items):
        source_id = normalize_source_id(item)
        if source_id and source_id not in seen:
            ids.append(source_id)
            seen.add(source_id)
    return ids


def normalize_ks(ks: Iterable[int] | int | None = None) -> list[int]:
    if ks is None:
        values = list(DEFAULT_KS)
    elif isinstance(ks, int):
        values = [ks]
    else:
        values = [int(k) for k in ks]
    result = sorted({k for k in values if k > 0})
    return result or list(DEFAULT_KS)


def recall_at_k(expected: set[str], returned: list[str], k: int) -> float:
    if not expected:
        return 1.0
    return len(expected & set(returned[:k])) / len(expected)


def precision_at_k(expected: set[str], returned: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    selected = returned[:k]
    if not selected:
        return 0.0
    return len(expected & set(selected)) / len(selected)


def f1_at_k(expected: set[str], returned: list[str], k: int) -> float:
    precision = precision_at_k(expected, returned, k)
    recall = recall_at_k(expected, returned, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(expected: set[str], returned: list[str], k: int) -> float:
    if not expected:
        return 1.0
    return 1.0 if expected & set(returned[:k]) else 0.0


def reciprocal_rank_at_k(expected: set[str], returned: list[str], k: int) -> float:
    if not expected:
        return 1.0
    for index, source_id in enumerate(returned[:k], start=1):
        if source_id in expected:
            return 1.0 / index
    return 0.0


def average_precision_at_k(expected: set[str], returned: list[str], k: int) -> float:
    if not expected:
        return 1.0
    hits = 0
    precision_sum = 0.0
    for index, source_id in enumerate(returned[:k], start=1):
        if source_id in expected:
            hits += 1
            precision_sum += hits / index
    return precision_sum / min(len(expected), k)


def ndcg_at_k(expected: set[str], returned: list[str], k: int) -> float:
    if not expected:
        return 1.0
    dcg = 0.0
    for index, source_id in enumerate(returned[:k], start=1):
        if source_id in expected:
            dcg += 1 / log2(index + 1)
    ideal_hits = min(len(expected), k)
    ideal_dcg = sum(1 / log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def retrieval_metrics(expected_ids: Iterable[str], returned_ids: Iterable[str], ks: Iterable[int] | int | None = None) -> dict[str, float]:
    expected = {str(item).strip() for item in expected_ids if str(item).strip()}
    returned = ranked_ids(list(returned_ids))
    metrics: dict[str, float] = {}
    for k in normalize_ks(ks):
        metrics[f"precision@{k}"] = precision_at_k(expected, returned, k)
        metrics[f"recall@{k}"] = recall_at_k(expected, returned, k)
        metrics[f"f1@{k}"] = f1_at_k(expected, returned, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(expected, returned, k)
        metrics[f"mrr@{k}"] = reciprocal_rank_at_k(expected, returned, k)
        metrics[f"map@{k}"] = average_precision_at_k(expected, returned, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(expected, returned, k)
    metrics["expected_count"] = float(len(expected))
    metrics["returned_count"] = float(len(returned))
    return metrics


def keyword_coverage(answer: str, expected_keywords: Iterable[str]) -> dict[str, Any]:
    keywords = [str(item).strip() for item in expected_keywords if str(item).strip()]
    if not keywords:
        return {"coverage": 1.0, "matched": [], "missing": []}
    folded_answer = answer.casefold()
    matched = [term for term in keywords if term.casefold() in folded_answer]
    missing = [term for term in keywords if term.casefold() not in folded_answer]
    return {
        "coverage": len(matched) / len(keywords),
        "matched": matched,
        "missing": missing,
    }


def forbidden_terms(answer: str, terms: Iterable[str]) -> list[str]:
    folded_answer = answer.casefold()
    return [term for term in (str(item).strip() for item in terms) if term and term.casefold() in folded_answer]


def citation_metrics(expected_ids: Iterable[str], citation_ids: Iterable[str], grounding_ids: Iterable[str] | None = None) -> dict[str, float]:
    expected = {str(item).strip() for item in expected_ids if str(item).strip()}
    citations = ranked_ids(list(citation_ids))
    grounding = {str(item).strip() for item in (grounding_ids or []) if str(item).strip()}
    metrics: dict[str, float] = {
        "count": float(len(citations)),
        "groundedness": 1.0,
    }
    if citations and grounding:
        metrics["groundedness"] = len(set(citations) & grounding) / len(citations)
    if expected:
        citation_set = set(citations)
        precision = len(expected & citation_set) / len(citation_set) if citation_set else 0.0
        recall = len(expected & citation_set) / len(expected)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return metrics


def percentile(values: Iterable[float], percent: float) -> float | None:
    items = sorted(float(value) for value in values)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    position = (len(items) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(items) - 1)
    weight = position - lower
    return items[lower] * (1 - weight) + items[upper] * weight


def latency_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values if _safe_float(value) is not None]
    if not items:
        return {"count": 0, "avg_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(items),
        "avg_ms": round(mean(items), 3),
        "p50_ms": round(median(items), 3),
        "p95_ms": round(percentile(items, 95) or 0.0, 3),
        "max_ms": round(max(items), 3),
    }


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _has_non_null_key(row: dict[str, Any], keys: Iterable[str]) -> bool:
    return any(key in row and row[key] is not None for key in keys)


def extract_expected_ids(row: dict[str, Any]) -> list[str]:
    return ranked_ids(_first_present(row, ("expected", "expected_sources", "expected_source_ids", "relevant", "relevant_source_ids", "ground_truth")))


def extract_returned_ids(row: dict[str, Any]) -> list[str]:
    value = _first_present(row, RETURNED_FIELDS)
    return ranked_ids(value)


def extract_answer_text(row: dict[str, Any]) -> str:
    value = _first_present(row, ANSWER_FIELDS)
    if isinstance(value, dict):
        value = _first_present(value, ("answer", "text", "content", "final_answer"))
    return str(value or "").strip()


def extract_citation_ids(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    if "citations" in row:
        return ranked_ids(row["citations"])
    if isinstance(answer, dict) and "citations" in answer:
        return ranked_ids(answer["citations"])
    return []


def extract_expected_citation_ids(row: dict[str, Any]) -> list[str]:
    value = _first_present(row, ("expected_citations", "expected_citation_ids", "required_citations", "required_citation_ids"))
    return ranked_ids(value)


def extract_keywords(row: dict[str, Any]) -> list[str]:
    value = _first_present(row, ("expected_answer_keywords", "required_keywords", "answer_keywords", "keywords"))
    return [str(item).strip() for item in _stable_sequence(value) if str(item).strip()]


def extract_forbidden_terms(row: dict[str, Any]) -> list[str]:
    value = _first_present(row, ("forbidden_answer_terms", "forbidden_terms", "must_not_contain"))
    return [str(item).strip() for item in _stable_sequence(value) if str(item).strip()]


def normalize_latency(row: dict[str, Any]) -> dict[str, float]:
    latency = _first_present(row, ("latency_ms", "timings_ms", "stage_timings"))
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if latency is None:
        latency = meta.get("stage_timings")
    result: dict[str, float] = {}
    value = _safe_float(latency)
    if value is not None:
        result["total_ms"] = value
    elif isinstance(latency, dict):
        aliases = {
            "total": "total_ms",
            "elapsed": "total_ms",
            "elapsed_ms": "total_ms",
            "retrieval": "retrieval_ms",
            "rerank": "rerank_ms",
            "generation": "generation_ms",
            "llm": "generation_ms",
            "planning": "planning_ms",
            "processing": "processing_ms",
            "context": "context_ms",
        }
        for key, raw_value in latency.items():
            numeric = _safe_float(raw_value)
            if numeric is None:
                continue
            target = aliases.get(str(key), str(key))
            if not target.endswith("_ms"):
                target = f"{target}_ms"
            result[target] = numeric
    elapsed = _safe_float(meta.get("elapsed_ms"))
    if elapsed is not None:
        result.setdefault("total_ms", elapsed)
    return result


def evaluate_record(row: dict[str, Any], ks: Iterable[int] | int | None = None) -> dict[str, Any]:
    expected = extract_expected_ids(row)
    returned = extract_returned_ids(row)
    answer_text = extract_answer_text(row)
    citations = extract_citation_ids(row)
    expected_citations = extract_expected_citation_ids(row)
    keywords = extract_keywords(row)
    blocked_terms = extract_forbidden_terms(row)
    keyword_result = keyword_coverage(answer_text, keywords)
    blocked_found = forbidden_terms(answer_text, blocked_terms)
    latency = normalize_latency(row)
    has_retrieval_result = _has_non_null_key(row, RETURNED_FIELDS)
    has_answer_result = _has_non_null_key(row, ANSWER_FIELDS)
    golden_only = bool(expected or keywords or expected_citations) and not has_retrieval_result and not has_answer_result and not latency

    answer_metrics: dict[str, Any] = {
        "present": 1.0 if bool(answer_text) else 0.0,
        "length_chars": float(len(answer_text)),
        "keyword_coverage": keyword_result["coverage"],
        "forbidden_term_free": 1.0 if not blocked_found else 0.0,
    }
    if not returned:
        caution_terms = ("证据", "依据不足", "未检索", "不能直接", "仅供参考", "建议咨询")
        answer_metrics["no_evidence_caution"] = 1.0 if any(term in answer_text for term in caution_terms) else 0.0

    return {
        "id": str(row.get("id") or row.get("case_id") or row.get("question_id") or "").strip(),
        "question": str(row.get("question") or row.get("query") or "").strip(),
        "expected": expected,
        "returned": returned,
        "retrieval": retrieval_metrics(expected, returned, ks),
        "answer": {
            **answer_metrics,
            "matched_keywords": keyword_result["matched"],
            "missing_keywords": keyword_result["missing"],
            "forbidden_terms": blocked_found,
        },
        "citation": citation_metrics(expected_citations, citations, grounding_ids=returned),
        "latency": latency,
        "data_status": {
            "has_retrieval_result": has_retrieval_result,
            "has_answer_result": has_answer_result,
            "has_latency": bool(latency),
            "golden_only": golden_only,
        },
    }


def _average_numeric(records: list[dict[str, Any]], section: str) -> dict[str, float]:
    bucket: dict[str, list[float]] = {}
    for record in records:
        values = record.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            numeric = _safe_float(value)
            if numeric is not None:
                bucket.setdefault(key, []).append(numeric)
    return {key: round(mean(values), 6) for key, values in sorted(bucket.items()) if values}


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latency_values: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.get("latency", {}).items():
            numeric = _safe_float(value)
            if numeric is not None:
                latency_values.setdefault(key, []).append(numeric)
    data_statuses = [record.get("data_status", {}) for record in records]
    return {
        "count": len(records),
        "retrieval": _average_numeric(records, "retrieval"),
        "answer": _average_numeric(records, "answer"),
        "citation": _average_numeric(records, "citation"),
        "latency": {key: latency_summary(values) for key, values in sorted(latency_values.items())},
        "data_status": {
            "golden_only_count": sum(1 for status in data_statuses if status.get("golden_only")),
            "retrieval_result_count": sum(1 for status in data_statuses if status.get("has_retrieval_result")),
            "answer_result_count": sum(1 for status in data_statuses if status.get("has_answer_result")),
            "latency_result_count": sum(1 for status in data_statuses if status.get("has_latency")),
        },
    }


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_numeric(item, next_prefix))
    else:
        numeric = _safe_float(value)
        if numeric is not None and prefix:
            flat[prefix] = numeric
    return flat


def _metric_lookup(flat: dict[str, float], key: str) -> float | None:
    if key in flat:
        return flat[key]
    matches = [value for name, value in flat.items() if name.endswith(f".{key}") or name == key]
    if len(matches) == 1:
        return matches[0]
    return None


def check_thresholds(summary: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    flat = flatten_numeric(summary)
    checks = []
    passed = True
    for raw_key, expected_value in thresholds.items():
        mode = "max" if raw_key.startswith("max:") else "min"
        key = raw_key[4:] if raw_key.startswith("max:") else raw_key
        actual = _metric_lookup(flat, key)
        ok = False
        if actual is not None:
            ok = actual <= expected_value if mode == "max" else actual >= expected_value
        passed = passed and ok
        checks.append({
            "metric": key,
            "mode": mode,
            "expected": expected_value,
            "actual": actual,
            "passed": ok,
        })
    return {"passed": passed, "checks": checks}


def evaluate_dataset(rows: Iterable[dict[str, Any]], ks: Iterable[int] | int | None = None, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    k_values = normalize_ks(ks)
    records = [evaluate_record(row, k_values) for row in rows]
    summary = aggregate_records(records)
    data_status = summary.get("data_status", {})
    golden_only_count = int(data_status.get("golden_only_count", 0) or 0)
    mode = "golden_only" if records and golden_only_count == len(records) else "evaluated"
    report: dict[str, Any] = {
        "schema_version": "law-rag-eval-v1",
        "status": "ok",
        "mode": mode,
        "ks": k_values,
        "count": len(records),
        "summary": summary,
        "records": records,
    }
    if thresholds:
        threshold_result = check_thresholds(summary, thresholds)
        report["thresholds"] = threshold_result
        report["status"] = "passed" if threshold_result["passed"] else "failed"
    return report
