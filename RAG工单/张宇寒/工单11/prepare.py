from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import build_triplets, parse_questions, save_jsonl


ROOT = Path(__file__).resolve().parent


def prepare(chunks_path: Path, questions_path: Path, output_dir: Path) -> dict:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    questions = parse_questions(questions_path.read_text(encoding="utf-8"))
    split = build_triplets(chunks, questions, seed=42, train_ratio=0.7)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(split["train"], output_dir / "train.jsonl")
    save_jsonl(split["evaluation"], output_dir / "evaluation.jsonl")
    save_jsonl(split["train"] + split["evaluation"], output_dir / "qa_pairs.jsonl")
    summary = {
        "source_chunks": len(chunks),
        "questions": len(questions),
        "train_triplets": len(split["train"]),
        "evaluation_triplets": len(split["evaluation"]),
        "seed": 42,
        "format": "query-positive-negative triplet",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="生成金融Embedding微调数据")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval_questions.md")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(json.dumps(prepare(args.chunks, args.questions, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
