"""Quality-gated ingestion workflow used by agents and the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .assessment import DocumentQualityAssessment


class DocumentIngestionWorkflow:
    name = "document_ingestion_workflow"

    def __init__(self, assessor: DocumentQualityAssessment | None = None):
        self.assessor = assessor or DocumentQualityAssessment()

    def run(self, paths: Iterable[str | Path]) -> dict[str, Any]:
        report = self.assessor.assess(paths)
        routes: list[dict[str, str]] = []
        for document in report["documents"]:
            pdf_type = document.get("pdf_type")
            parser = "ocr" if pdf_type == "scanned" else "deepdoc"
            if pdf_type == "mixed":
                parser = "deepdoc+ocr"
            routes.append({"path": document["path"], "parser": parser, "reason": pdf_type or "format"})
        report["workflow"] = {"name": self.name, "routes": routes}
        return report
