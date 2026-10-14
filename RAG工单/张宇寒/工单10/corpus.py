from __future__ import annotations

import json
import re
import zipfile
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
    chunk_id: str
    document: str
    stock_code: str
    company: str
    year: int
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


def report_for_name(name: str) -> Report:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", name)
    if not match or match.group(1) not in REPORTS:
        raise ValueError(f"无法从文件名识别报告股票代码：{name}")
    return REPORTS[match.group(1)]


def _decode(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    return [text[start : start + max_chars] for start in range(0, len(text), step) if text[start : start + max_chars].strip()]


def load_zip_corpus(zip_path: Path, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"年报压缩包不存在：{zip_path}")
    chunks: list[Chunk] = []
    with zipfile.ZipFile(zip_path) as archive:
        entries = [item for item in archive.infolist() if "/txt/" in item.filename.replace("\\", "/") and item.filename.lower().endswith(".txt")]
        for entry in sorted(entries, key=lambda item: item.filename):
            report = report_for_name(entry.filename)
            by_page: dict[int, list[str]] = {}
            for raw in _decode(archive.read(entry)).splitlines():
                try:
                    row = json.loads(raw)
                    page = int(row.get("page", 0))
                    text = str(row.get("inside", "")).strip()
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if page > 0 and text:
                    by_page.setdefault(page, []).append(text)
            for page, lines in sorted(by_page.items()):
                title = next((line for line in lines if 2 <= len(line) <= 40), "")
                for part_number, part in enumerate(_split("\n".join(lines), max_chars, overlap), 1):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{report.stock_code}-p{page}-{part_number}",
                            document=report.document,
                            stock_code=report.stock_code,
                            company=report.company,
                            year=report.year,
                            page=page,
                            text=part,
                            title=title,
                        )
                    )
    if not chunks:
        raise ValueError("压缩包中没有可用的年报文本")
    return chunks

