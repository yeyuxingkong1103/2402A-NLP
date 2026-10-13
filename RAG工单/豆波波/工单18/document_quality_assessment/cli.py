"""Command line entry point for DocumentQualityAssessmentSkill."""

from __future__ import annotations

import argparse
from pathlib import Path

from .assessment import DocumentQualityAssessment
from .config import AssessmentConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess files before RAG ingestion")
    parser.add_argument("paths", nargs="+", help="Files or directories to inspect")
    parser.add_argument("--config", help="YAML or JSON assessment config")
    parser.add_argument("--json", dest="json_output", default="quality_report.json")
    parser.add_argument("--html", dest="html_output", default="quality_report.html")
    args = parser.parse_args()
    config = AssessmentConfig.from_file(args.config) if args.config else AssessmentConfig()
    report = DocumentQualityAssessment(config).assess(args.paths)
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.html_output).parent.mkdir(parents=True, exist_ok=True)
    DocumentQualityAssessment.write_json(report, args.json_output)
    DocumentQualityAssessment.write_html(report, args.html_output)
    print(f"assessed={report['summary']['files']} json={args.json_output} html={args.html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
