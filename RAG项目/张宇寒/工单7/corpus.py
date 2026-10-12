from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Report:
    stock_code: str
    company: str
    year: int
    document: str


@dataclass(frozen=True)
class Chunk:
    document: str
    stock_code: str
    page: int
    text: str
    title: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


REPORTS = {
    "000001": Report("000001", "平安银行", 2019, "平安银行2019年年报"),
    "601318": Report("601318", "中国平安", 2019, "中国平安2019年年报"),
    "600036": Report("600036", "招商银行", 2019, "招商银行2019年年报"),
    "601658": Report("601658", "邮储银行", 2019, "邮储银行2019年年报"),
    "600030": Report("600030", "中信证券", 2020, "中信证券2020年年报"),
    "601628": Report("601628", "中国人寿", 2020, "中国人寿2020年年报"),
    "601601": Report("601601", "中国太保", 2021, "中国太保2021年年报"),
    "600999": Report("600999", "招商证券", 2021, "招商证券2021年年报"),
    "601211": Report("601211", "国泰君安", 2021, "国泰君安2021年年报"),
}


def report_for_path(path: Path) -> Report:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", path.name)
    if not match or match.group(1) not in REPORTS:
        raise ValueError(f"无法从文件名识别报告股票代码：{path.name}")
    return REPORTS[match.group(1)]


def iter_jsonl_rows(path: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for raw in stream:
            try:
                item = json.loads(raw)
                text = str(item.get("inside", "")).strip()
                page = int(item.get("page", 0))
                if not text or page <= 0:
                    skipped += 1
                    continue
                rows.append({"page": page, "type": str(item.get("type", "text")), "inside": text})
            except (json.JSONDecodeError, TypeError, ValueError):
                skipped += 1
    return rows, skipped


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    return [text[start : start + max_chars] for start in range(0, len(text), step) if text[start : start + max_chars].strip()]


def load_corpus(root: Path, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    text_dir = root / "txt"
    if not text_dir.is_dir():
        raise FileNotFoundError(f"未找到结构化文本目录：{text_dir}")

    chunks: list[Chunk] = []
    for path in sorted(text_dir.glob("*.txt")):
        report = report_for_path(path)
        rows, _ = iter_jsonl_rows(path)
        by_page: dict[int, list[str]] = {}
        for row in rows:
            by_page.setdefault(row["page"], []).append(row["inside"])
        for page, lines in by_page.items():
            title = next((line for line in lines if 2 <= len(line) <= 40), "")
            page_text = "\n".join(lines)
            for part in _split_text(page_text, max_chars=max_chars, overlap=overlap):
                chunks.append(Chunk(report.document, report.stock_code, page, part, title))
    if not chunks:
        raise ValueError(f"目录中没有可用文本：{text_dir}")
    return chunks
