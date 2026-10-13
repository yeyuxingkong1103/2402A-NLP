"""Evaluate terminology coverage and diagram-reasoning answers."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def evaluate(reference: Path, predictions: Path, output: Path):
    refs = {str(item.get("id", i)): item for i, item in enumerate(_read(reference))}
    rows = []
    for i, pred in enumerate(_read(predictions)):
        key = str(pred.get("id", i))
        expected = str(refs.get(key, {}).get("answer", ""))
        actual = str(pred.get("prediction") or pred.get("answer") or "")
        expected_terms = set(re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]", expected.lower()))
        actual_terms = set(re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]", actual.lower()))
        terminology = len(expected_terms & actual_terms) / len(expected_terms) if expected_terms else 1.0
        exact = int(expected.strip().lower() == actual.strip().lower())
        diagram = terminology if any(word in expected for word in ("部件", "图", "位置", "经过", "直径")) else exact
        rows.append({"id": key, "exact": exact, "terminology": round(terminology, 4), "diagram_reasoning": round(diagram, 4), "expected": expected, "actual": actual})
    report = {"count": len(rows), "exact_accuracy": round(sum(r["exact"] for r in rows) / len(rows), 4) if rows else 0, "terminology_accuracy": round(sum(r["terminology"] for r in rows) / len(rows), 4) if rows else 0, "diagram_reasoning_accuracy": round(sum(r["diagram_reasoning"] for r in rows) / len(rows), 4) if rows else 0, "failures": [r for r in rows if not r["exact"]], "rows": rows}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.reference, args.predictions, args.output)
    print(json.dumps({key: report[key] for key in ("count", "exact_accuracy", "terminology_accuracy", "diagram_reasoning_accuracy")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
