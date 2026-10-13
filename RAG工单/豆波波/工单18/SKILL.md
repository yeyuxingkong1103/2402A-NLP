---
name: DocumentQualityAssessmentSkill
description: Assess files before ingestion, classify PDFs, find duplicates, and queue sensitive matches for human review.
---

# Document quality assessment

Run `python -m document_quality_assessment.cli <path> --json quality_report.json --html quality_report.html`.

The report contains format counts, per-page PDF classification, length percentiles, exact MD5 duplicates, SimHash version conflicts, and contextual sensitive-information review items. Thresholds live in `assessment_config.yaml` and should be tuned against a representative sample.

`document_ingestion_workflow.py` routes scanned PDFs to OCR, text PDFs to DeepDoc, and mixed PDFs to both parsers. Review queues are intentionally non-destructive: no file is deleted or automatically merged.
