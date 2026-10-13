"""Convert IMDR QA records and RAGFlow image blocks to VLM JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _load(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def convert(questions: Path, output: Path, image_root: Path | None = None) -> int:
    image_root = image_root or questions.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for record in _load(questions):
            question = str(record.get("question") or record.get("query") or "").strip()
            answer = str(record.get("answer") or record.get("response") or "").strip()
            if not question or not answer:
                continue
            raw_image = record.get("image") or record.get("image_path") or record.get("figure")
            image = str((image_root / raw_image).resolve()) if raw_image and not Path(str(raw_image)).is_absolute() else raw_image
            content = ([{"type": "image", "image": image}] if image else []) + [{"type": "text", "text": question}]
            stream.write(json.dumps({"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": answer}]}], "image": image, "question": question, "answer": answer, "document": record.get("document", "")}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()
    print(f"converted={convert(args.questions, args.output, args.image_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
