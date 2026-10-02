from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .chunk import chunk_records
from .clean import clean_public_collections
from .load import file_sha256, load_docx, load_json, load_text
from .parse import (
    ARTICLE_RE,
    SOURCE_FILES,
    build_citations,
    extract_articles,
    extract_cases,
    extract_elements,
    extract_evidence,
    extract_interpretations,
    extract_processes,
    extract_questions,
)


PARSER_VERSION = "1.0.0"
CHUNK_VERSION = "1.0.0"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def _source_path(data_dir: Path, collection: str) -> Path:
    path = data_dir / SOURCE_FILES[collection]
    if not path.is_file():
        raise FileNotFoundError(f"{collection} is missing source file: {path}")
    return path


def _case_filter_reasons(text: str) -> dict[str, int]:
    reasons: dict[str, int] = {}
    seen: set[str] = set()
    blocks = [
        block for block in re.split(r"(?m)(?=^[ \t]*【案例\s*\d+】[ \t]*$)", text)
        if re.search(r"(?m)^[ \t]*【案例\s*\d+】[ \t]*$", block)
    ]
    for block in blocks:
        rows = extract_cases(block)
        if not rows:
            if "司法解释" in block or "意见稿" in block:
                reasons["judicial_interpretation_material"] = reasons.get("judicial_interpretation_material", 0) + 1
            continue
        case_id = str(rows[0].get("case_id", ""))
        if case_id in seen:
            reasons["duplicate_record"] = reasons.get("duplicate_record", 0) + 1
        else:
            seen.add(case_id)
    return reasons


def _read_and_parse(data_dir: Path):
    articles_text = load_text(_source_path(data_dir, "civil_code_articles"))
    interpretations_text = load_docx(_source_path(data_dir, "civil_interpretations"))
    cases_text = load_text(_source_path(data_dir, "civil_cases"))
    elements_payload = load_json(_source_path(data_dir, "civil_elements"))
    evidence_text = load_text(_source_path(data_dir, "civil_evidence"))
    processes_text = load_text(_source_path(data_dir, "civil_processes"))
    questions_text = load_text(_source_path(data_dir, "civil_questions"))

    element_rows = elements_payload.get("civil_elements", []) if isinstance(elements_payload, dict) else elements_payload
    input_counts = {
        "civil_code_articles": sum(bool(ARTICLE_RE.match(line.strip())) for line in articles_text.splitlines()),
        "civil_interpretations": sum(bool(ARTICLE_RE.match(line.strip())) for line in interpretations_text.splitlines()),
        "civil_cases": len(re.findall(r"(?m)^[ \t]*【案例\s*\d+】[ \t]*$", cases_text)),
        "civil_elements": len(element_rows or []),
        "civil_evidence": len(re.findall(r"(?m)^案例编号：", evidence_text)),
        "civil_processes": len(re.findall(r"(?m)^# CPC-\d+", processes_text)),
        "civil_questions": len(re.findall(r"(?m)^#\s+\d+\s*$", questions_text)),
    }
    parsed = {
        "civil_code_articles": extract_articles(articles_text),
        "civil_interpretations": extract_interpretations(interpretations_text),
        "civil_cases": extract_cases(cases_text),
        "civil_elements": extract_elements(elements_payload),
        "civil_evidence": extract_evidence(evidence_text),
        "civil_processes": extract_processes(processes_text),
        "civil_questions": extract_questions(questions_text),
    }
    return parsed, input_counts, _case_filter_reasons(cases_text)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_public_data(
    data_dir: str | Path = "data/public",
    output_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    source_dir = Path(data_dir)
    target_dir = Path(output_dir)
    source_paths = {name: _source_path(source_dir, name) for name in SOURCE_FILES}
    parsed, input_counts, case_filter_reasons = _read_and_parse(source_dir)
    parsed_counts = {name: len(rows) for name, rows in parsed.items()}
    cleaned, quality_report = clean_public_collections(parsed)

    for collection, item in quality_report.items():
        item["input_records"] = input_counts[collection]
        item["parsed_records"] = parsed_counts[collection]
        item["parser_dropped_records"] = max(0, input_counts[collection] - parsed_counts[collection])
    quality_report["civil_cases"]["filter_reasons"] = dict(sorted(case_filter_reasons.items()))

    citations = build_citations(cleaned)
    cleaned_citations, citation_report = clean_public_collections({"civil_citations": citations})
    cleaned["civil_citations"] = cleaned_citations["civil_citations"]
    citation_report["civil_citations"].update(
        input_records=len(citations), parsed_records=len(citations), parser_dropped_records=0
    )
    quality_report.update(citation_report)

    for rows in cleaned.values():
        for row in rows:
            row.pop("embedding", None)
            row.pop("embedding_text", None)

    replacement_characters = sum(item["replacement_characters"] for item in quality_report.values())
    if replacement_characters:
        raise ValueError(f"processed data contains {replacement_characters} replacement characters")

    target_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = target_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_counts: dict[str, int] = {}
    for collection, rows in cleaned.items():
        _write_json(target_dir / f"{collection}.json", rows)
        chunks = chunk_records(collection, rows, CHUNK_SIZE, CHUNK_OVERLAP)
        _write_json(chunks_dir / f"{collection}_chunks.json", chunks)
        chunk_counts[collection] = len(chunks)

    _write_json(target_dir / "quality_report.json", quality_report)
    record_counts = {name: len(rows) for name, rows in cleaned.items()}
    manifest = {
        "dataset_version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "parser_version": PARSER_VERSION,
        "chunk_version": CHUNK_VERSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": None,
        "embedding_dimension": None,
        "source_hashes": {name: file_sha256(path) for name, path in source_paths.items()},
        "record_counts": record_counts,
        "chunk_counts": chunk_counts,
        "status": "processed",
    }
    _write_json(target_dir / "build_manifest.json", manifest)
    return {
        "output_dir": str(target_dir.resolve()),
        "record_counts": record_counts,
        "chunk_counts": chunk_counts,
        "quality_report": quality_report,
        "manifest": manifest,
    }


def main() -> int:
    # 旧命令兼容：统一走main，不再维护第二套命令参数。
    # process_public_data仍保留，供旧代码生成历史分块和统计报告。
    from .main import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
