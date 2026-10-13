"""Offline document quality assessment with actionable review queues."""

from __future__ import annotations

import hashlib
import html
import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles
from typing import Any, Iterable

from .config import AssessmentConfig


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".md",
    ".markdown",
    ".mdx",
    ".txt",
    ".html",
    ".htm",
    ".json",
    ".csv",
    ".xlsx",
    ".xls",
    ".pptx",
}


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeError):
        return ""


def _pdf_text(path: Path) -> tuple[list[str], str | None]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path), strict=False)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return pages, None
    except Exception as exc:  # corrupted PDFs remain in the report for review
        return [], f"{type(exc).__name__}: {exc}"


def _docx_text(path: Path) -> str:
    try:
        from xml.etree import ElementTree

        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        return " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
    except Exception:
        return ""


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(_pdf_text(path)[0])
    if path.suffix.lower() == ".docx":
        return _docx_text(path)
    if path.suffix.lower() in {".txt", ".md", ".markdown", ".mdx", ".html", ".htm", ".json", ".csv"}:
        return _safe_read_text(path)
    return ""


def _simhash(text: str) -> int:
    tokens = re.findall(r"\w+|[\u4e00-\u9fff]", text.lower())
    if not tokens:
        return 0
    shingles = [" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))]
    weights = [0] * 64
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class DocumentQualityAssessment:
    """Run all five quality checks required before ingestion."""

    def __init__(self, config: AssessmentConfig | None = None):
        self.config = config or AssessmentConfig()

    def _files(self, paths: Iterable[str | Path]) -> list[Path]:
        result: list[Path] = []
        for value in paths:
            path = Path(value).expanduser()
            if path.is_file():
                result.append(path)
            elif path.is_dir():
                result.extend(p for p in path.rglob("*") if p.is_file())
        return sorted(set(result), key=lambda item: str(item).lower())

    def assess(self, paths: Iterable[str | Path]) -> dict[str, Any]:
        files = self._files(paths)
        documents: list[dict[str, Any]] = []
        texts: dict[str, str] = {}
        exact: dict[str, list[str]] = {}
        simhashes: list[tuple[str, int]] = []
        for path in files:
            relative = str(path)
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            exact.setdefault(digest, []).append(relative)
            text = _extract_text(path)
            texts[relative] = text
            length = len(text)
            item: dict[str, Any] = {
                "path": relative,
                "extension": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "characters": length,
                "md5": digest,
            }
            if path.suffix.lower() == ".pdf":
                pages, error = _pdf_text(path)
                scanned_pages = sum(len(page) < self.config.scan_text_threshold for page in pages)
                ratio = scanned_pages / len(pages) if pages else 1.0
                if error:
                    kind = "unknown"
                elif ratio > self.config.scan_ratio_threshold:
                    kind = "scanned"
                elif scanned_pages:
                    kind = "mixed"
                else:
                    kind = "text"
                item.update(
                    {
                        "pages": len(pages),
                        "scanned_pages": scanned_pages,
                        "scanned_ratio": round(ratio, 4),
                        "pdf_type": kind,
                        "parse_error": error,
                        "needs_confirmation": bool(error or (0.55 <= ratio <= 0.85)),
                    }
                )
            if text:
                simhashes.append((relative, _simhash(text)))
            documents.append(item)

        lengths = sorted(item["characters"] for item in documents)
        percentiles: dict[str, float] = {}
        if lengths:
            if len(lengths) == 1:
                percentiles = {name: float(lengths[0]) for name in ("p25", "p50", "p75", "p90", "p99")}
            else:
                values = quantiles(lengths, n=100, method="inclusive")
                percentiles = {name: round(values[int(percentile) - 1], 2) for name, percentile in (("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("p99", 99))}
        duplicate_groups = [paths for paths in exact.values() if len(paths) > 1]
        conflicts: list[dict[str, Any]] = []
        for index, (left_path, left_hash) in enumerate(simhashes):
            for right_path, right_hash in simhashes[index + 1 :]:
                distance = _hamming(left_hash, right_hash)
                if distance <= self.config.simhash_max_distance:
                    conflicts.append({"left": left_path, "right": right_path, "distance": distance, "status": "needs_review"})
                    if len(conflicts) >= self.config.simhash_max_pairs:
                        break
            if len(conflicts) >= self.config.simhash_max_pairs:
                break

        patterns = dict(self.config.sensitive_patterns)
        if self.config.enable_bank_card:
            patterns["bank_card"] = r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
        sensitive: list[dict[str, Any]] = []
        for path, content in texts.items():
            for kind, pattern in patterns.items():
                try:
                    matches = list(re.finditer(pattern, content))
                except re.error as exc:
                    sensitive.append({"path": path, "type": kind, "error": str(exc), "status": "needs_review"})
                    continue
                for match in matches:
                    start = max(0, match.start() - self.config.context_radius)
                    end = min(len(content), match.end() + self.config.context_radius)
                    sensitive.append(
                        {
                            "path": path,
                            "type": kind,
                            "match": match.group(0),
                            "context": content[start:end].replace("\n", " "),
                            "offset": [match.start(), match.end()],
                            "status": "needs_review",
                        }
                    )
        by_extension = Counter(item["extension"] or "[none]" for item in documents)
        result = {
            "skill": "DocumentQualityAssessmentSkill",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "summary": {
                "files": len(documents),
                "formats": dict(sorted(by_extension.items())),
                "exact_duplicate_groups": len(duplicate_groups),
                "similar_version_conflicts": len(conflicts),
                "sensitive_matches": len(sensitive),
            },
            "documents": documents,
            "length_distribution": {"count": len(lengths), "percentiles": percentiles, "values": lengths},
            "exact_duplicates": [{"md5": digest, "paths": paths, "status": "confirmed_duplicate"} for digest, paths in exact.items() if len(paths) > 1],
            "version_conflicts": conflicts,
            "sensitive_review": sensitive,
        }
        return result

    @staticmethod
    def write_json(report: dict[str, Any], output: str | Path) -> None:
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def write_html(report: dict[str, Any], output: str | Path) -> None:
        summary = report["summary"]
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(item.get(key, '')))}</td>" for key in ("path", "extension", "characters", "pdf_type", "needs_confirmation")) + "</tr>"
            for item in report["documents"]
        )
        body = f"""<!doctype html><meta charset='utf-8'><title>Document quality report</title>
<style>body{{font:14px system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.4rem;text-align:left}}.review{{background:#fff3cd}}</style>
<h1>Document quality report</h1><p>Files: {summary['files']} | Exact duplicate groups: {summary['exact_duplicate_groups']} | Version conflicts: {summary['similar_version_conflicts']} | Sensitive review items: {summary['sensitive_matches']}</p>
<table><thead><tr><th>Path</th><th>Extension</th><th>Characters</th><th>PDF type</th><th>Needs confirmation</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Pending version review</h2><pre>{html.escape(json.dumps(report['version_conflicts'], ensure_ascii=False, indent=2))}</pre>
<h2>Pending sensitive review</h2><pre>{html.escape(json.dumps(report['sensitive_review'], ensure_ascii=False, indent=2))}</pre>"""
        Path(output).write_text(body, encoding="utf-8")
