"""Reusable dataset cleaning pipeline and artifact writer."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from app.ingestion.dataset_cleaner import clean_jsonl, write_json, write_jsonl
from app.ingestion.dataset_dedup import (
    duplicate_groups,
    near_duplicate_pairs,
    select_canonical_questions,
)

OUTPUT_FILES = {
    "normalized": "normalized.jsonl",
    "curated": "curated.jsonl",
    "product": "product.jsonl",
    "after_sales": "after_sales.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _report_payload(
    source: Path,
    report: Any,
    curated: list[Any],
    near_duplicates: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    report.curated_records = len(curated)
    report.question_variants_removed = report.output_records - len(curated)
    report.curated_category_counts = dict(
        sorted(Counter(record.category for record in curated).items())
    )
    payload = report.as_dict()
    payload.update({
        "source": str(source),
        "source_sha256": _sha256(source),
        "near_duplicate_threshold": threshold,
        "near_duplicate_pairs_for_review": len(near_duplicates),
        "policy": {
            "original_file_modified": False,
            "exact_record_duplicates": "removed",
            "same_normalized_question": (
                "best actionable answer selected for curated output"
            ),
            "near_duplicates": "reported only; not automatically removed",
            "invalid_records": "quarantined",
        },
    })
    return payload


def clean_dataset(
    source: str | Path,
    output_dir: str | Path,
    threshold: float = 0.92,
) -> dict[str, Any]:
    """Clean one JSONL file and atomically write all audit artifacts."""
    source_path = Path(source)
    target = Path(output_dir)
    result = clean_jsonl(source_path)
    curated = select_canonical_questions(result.records)
    duplicate_review = duplicate_groups(result.records)
    near_duplicates = near_duplicate_pairs(curated, threshold=threshold)
    product = [record for record in curated if record.category == "product"]
    after_sales = [record for record in curated if record.category == "after_sales"]

    write_jsonl(result.records, target / OUTPUT_FILES["normalized"])
    write_jsonl(curated, target / OUTPUT_FILES["curated"])
    write_jsonl(product, target / OUTPUT_FILES["product"])
    write_jsonl(after_sales, target / OUTPUT_FILES["after_sales"])
    write_jsonl(result.rejected, target / "rejected.jsonl")
    write_json({"groups": duplicate_review}, target / "duplicate_questions.json")
    write_json({"pairs": near_duplicates}, target / "near_duplicates.json")
    report = _report_payload(
        source_path, result.report, curated, near_duplicates, threshold
    )
    write_json(report, target / "report.json")
    return report


def cleaned_dataset_path(output_dir: str | Path, mode: str) -> Path:
    """Resolve an indexable output by its public mode name."""
    try:
        file_name = OUTPUT_FILES[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported cleaned dataset mode: {mode}") from exc
    return Path(output_dir) / file_name
