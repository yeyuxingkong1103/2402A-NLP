"""Work order 18: document-quality evaluation skill and agent-workflow integration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class QualityIssue:
    code: str
    severity: str
    message: str
    location: str = ""
    evidence: str = ""


@dataclass
class QualityReport:
    score: float
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class DocumentParser(Protocol):
    def parse(self, path: str) -> list[dict[str, Any]]: ...


class DocumentQualitySkill:
    REQUIRED_FIELDS = ("title", "source", "text")

    def evaluate(self, pages: list[dict[str, Any]]) -> QualityReport:
        issues: list[QualityIssue] = []
        total_chars = sum(len(str(page.get("text", ""))) for page in pages)
        missing_pages = sum(not str(page.get("text", "")).strip() for page in pages)
        replacement_chars = sum(str(page.get("text", "")).count("\ufffd") for page in pages)
        duplicate_lines = self._duplicate_line_ratio(pages)
        if missing_pages:
            issues.append(QualityIssue("EMPTY_PAGE", "high", f"{missing_pages} pages contain no extracted text"))
        if replacement_chars:
            issues.append(QualityIssue("ENCODING_DAMAGE", "high", f"{replacement_chars} replacement characters detected"))
        if duplicate_lines > 0.25:
            issues.append(QualityIssue("REPEATED_CONTENT", "medium", "Repeated lines may indicate noisy headers or OCR"))
        score = max(0.0, 100.0 - 25 * missing_pages - min(40, replacement_chars * 2) - duplicate_lines * 30)
        return QualityReport(
            score=round(score, 2),
            issues=issues,
            metrics={
                "page_count": float(len(pages)),
                "character_count": float(total_chars),
                "empty_page_ratio": missing_pages / max(1, len(pages)),
                "replacement_characters": float(replacement_chars),
                "duplicate_line_ratio": duplicate_lines,
            },
        )

    @staticmethod
    def _duplicate_line_ratio(pages: list[dict[str, Any]]) -> float:
        lines = [line.strip() for page in pages for line in str(page.get("text", "")).splitlines() if len(line.strip()) > 8]
        if not lines:
            return 0.0
        return 1 - len(set(lines)) / len(lines)

    def repair_plan(self, report: QualityReport) -> list[str]:
        actions = []
        codes = {issue.code for issue in report.issues}
        if "EMPTY_PAGE" in codes:
            actions.append("Render empty pages and run OCR; compare OCR confidence before replacing native text.")
        if "ENCODING_DAMAGE" in codes:
            actions.append("Inspect font maps and OCR the affected pages; keep original text for audit.")
        if "REPEATED_CONTENT" in codes:
            actions.append("Detect repeated headers/footers and remove them only after page-level validation.")
        return actions


class AgentWorkflow:
    """Minimal tool contract for calling the skill before indexing a document."""

    def __init__(self, parser: DocumentParser, quality_skill: DocumentQualitySkill | None = None):
        self.parser = parser
        self.quality_skill = quality_skill or DocumentQualitySkill()

    def run(self, path: str, minimum_score: float = 70.0) -> dict[str, Any]:
        pages = self.parser.parse(path)
        report = self.quality_skill.evaluate(pages)
        return {
            "document": path,
            "quality": report,
            "indexing_allowed": report.score >= minimum_score,
            "repair_plan": self.quality_skill.repair_plan(report),
            "pages": pages,
        }
