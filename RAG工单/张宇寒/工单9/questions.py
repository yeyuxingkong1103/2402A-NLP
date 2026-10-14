from __future__ import annotations

import re
from pathlib import Path


REQUIRED_FIELDS = ("Question", "Reference", "Stocks", "Pages", "Facts")


def _targets(value: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for group in value.split(";"):
        code, separator, pages = group.strip().partition(":")
        if not separator or not re.fullmatch(r"\d{6}", code):
            raise ValueError(f"Pages格式错误：{group}")
        result[code] = [int(page.strip()) for page in pages.split(",") if page.strip()]
    return result


def _finish(identifier: int, fields: dict[str, str]) -> dict:
    for name in REQUIRED_FIELDS:
        if not fields.get(name, "").strip():
            raise ValueError(f"问题{identifier}缺少{name}")
    stocks = [item.strip() for item in fields["Stocks"].split(",") if item.strip()]
    if not stocks or any(not re.fullmatch(r"\d{6}", item) for item in stocks):
        raise ValueError(f"问题{identifier}的Stocks格式错误")
    facts = [
        [alternative.strip() for alternative in group.split("|") if alternative.strip()]
        for group in fields["Facts"].split(";")
        if group.strip()
    ]
    return {
        "id": identifier,
        "question": fields["Question"].strip(),
        "reference_answer": fields["Reference"].strip(),
        "stocks": stocks,
        "targets": _targets(fields["Pages"]),
        "expected_facts": facts,
    }


def load_questions(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"题库不存在：{path}")
    questions: list[dict] = []
    current_id: int | None = None
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        heading = re.fullmatch(r"##\s+(\d+)\s*", raw.strip())
        if heading:
            if current_id is not None:
                questions.append(_finish(current_id, fields))
            current_id = int(heading.group(1))
            fields = {}
            continue
        field = re.match(r"-\s*(Question|Reference|Stocks|Pages|Facts):\s*(.*)", raw.strip())
        if field and current_id is not None:
            fields[field.group(1)] = field.group(2)
    if current_id is not None:
        questions.append(_finish(current_id, fields))
    if not questions:
        raise ValueError("题库中没有问题")
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("题库问题ID重复")
    return questions

