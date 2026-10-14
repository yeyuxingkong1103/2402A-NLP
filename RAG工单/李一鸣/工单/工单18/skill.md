---
name: document-quality-assessment
description: Evaluate document format distribution, PDF page type, length, duplicates and sensitive information.
---

# Document Quality Assessment Skill

## Input

Accept a directory containing PDF, DOCX, MD and text documents.

## Steps

1. Count files by extension.
2. Classify PDF pages as text, scanned or mixed.
3. Calculate document length percentiles.
4. Detect exact duplicates with MD5 and possible near-duplicates for manual review.
5. Detect phone numbers, email addresses, identity numbers and bank cards.

## Output

Return structured JSON with summary metrics, pending-version-conflicts and pending-sensitive-information-review.
