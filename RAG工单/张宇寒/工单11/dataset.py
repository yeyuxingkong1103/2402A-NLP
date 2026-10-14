from __future__ import annotations

import json
import random
import re
from pathlib import Path


def _alternatives(value: str) -> list[list[str]]:
    return [[item.strip() for item in group.split("|") if item.strip()] for group in value.split(";") if group.strip()]


def _pages(value: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for group in value.split(";"):
        if ":" not in group:
            continue
        stock, raw_pages = group.split(":", 1)
        result[stock.strip()] = [int(item) for item in raw_pages.split(",") if item.strip().isdigit()]
    return result


def parse_questions(content: str) -> list[dict]:
    records: list[dict] = []
    current: dict[str, str] = {}
    identifier: int | None = None
    for raw in content.splitlines():
        heading = re.match(r"^##\s+(\d+)\s*$", raw.strip())
        if heading:
            if current:
                records.append(_finish_question(identifier, current))
            identifier, current = int(heading.group(1)), {}
            continue
        field = re.match(r"^-\s+([A-Za-z]+):\s*(.*)$", raw.strip())
        if field:
            current[field.group(1).lower()] = field.group(2).strip()
    if current:
        records.append(_finish_question(identifier, current))
    if not records:
        raise ValueError("问题文件中没有有效问题")
    return records


def _finish_question(identifier: int | None, fields: dict[str, str]) -> dict:
    if identifier is None or not fields.get("question"):
        raise ValueError("问题缺少编号或Question字段")
    return {
        "id": identifier,
        "question": fields["question"],
        "reference": fields.get("reference", ""),
        "stocks": [item.strip() for item in fields.get("stocks", "").split(",") if item.strip()],
        "pages": _pages(fields.get("pages", "")),
        "facts": _alternatives(fields.get("facts", "")),
    }


def _score(chunk: dict, question: dict) -> tuple[int, int]:
    text = str(chunk.get("text", ""))
    stock = str(chunk.get("stock_code", ""))
    page = int(chunk.get("page", -1))
    page_hit = int(page in question["pages"].get(stock, []))
    fact_hit = sum(any(term.replace(",", "") in text.replace(",", "") for term in group) for group in question["facts"])
    return page_hit, fact_hit


def build_triplets(
    chunks: list[dict], questions: list[dict], seed: int = 42, train_ratio: float = 0.7
) -> dict[str, list[dict]]:
    if not chunks:
        raise ValueError("语料为空，无法生成微调数据")
    if not questions:
        raise ValueError("问题为空，无法生成微调数据")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio必须在0和1之间")

    rows: list[dict] = []
    for question in questions:
        eligible = [chunk for chunk in chunks if str(chunk.get("stock_code", "")) in question["stocks"]]
        candidates = eligible or chunks
        positive = max(candidates, key=lambda chunk: _score(chunk, question))
        same_company = [chunk for chunk in eligible if chunk.get("chunk_id") != positive.get("chunk_id")]
        other = [chunk for chunk in chunks if chunk.get("chunk_id") != positive.get("chunk_id")]
        negative_pool = same_company or other
        if not negative_pool:
            raise ValueError("至少需要两个不同文本块才能生成负例")
        negative = max(negative_pool, key=lambda chunk: len(set(str(chunk.get("text", ""))) & set(question["question"])))
        rows.append(
            {
                "question_id": question["id"],
                "query": question["question"],
                "answer": question["reference"],
                "positive": str(positive.get("text", "")),
                "negative": str(negative.get("text", "")),
                "positive_id": str(positive.get("chunk_id", "")),
                "negative_id": str(negative.get("chunk_id", "")),
                "company": str(positive.get("company", "")),
                "document": str(positive.get("document", "")),
                "page": int(positive.get("page", 0)),
            }
        )

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split = min(max(1, round(len(shuffled) * train_ratio)), len(shuffled) - 1) if len(shuffled) > 1 else 1
    return {"train": shuffled[:split], "evaluation": shuffled[split:]}


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
