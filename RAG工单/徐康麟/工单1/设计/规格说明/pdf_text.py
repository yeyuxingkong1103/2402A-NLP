"""PDF 文本抽取小工具（仅用于本次开发期核对内容）。

优先使用 PyMuPDF；若当前解释器没有 PyMuPDF，则回退到 pypdf。
用法:
    python pdf_text.py <pdf> [起始页] [结束页]
输出每页文本，页之间用 ===PAGE n=== 分隔。
"""
from __future__ import annotations

import sys


def extract_with_pymupdf(path: str, start: int, end: int) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF

    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        total = doc.page_count
        end = min(end, total) if end > 0 else total
        for i in range(start - 1, end):
            pages.append((i + 1, doc.load_page(i).get_text("text")))
    return pages


def extract_with_pypdf(path: str, start: int, end: int) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    total = len(reader.pages)
    end = min(end, total) if end > 0 else total
    return [(i + 1, reader.pages[i].extract_text() or "") for i in range(start - 1, end)]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    for extractor in (extract_with_pymupdf, extract_with_pypdf):
        try:
            pages = extractor(path, start, end)
        except ImportError:
            continue
        for number, text in pages:
            print(f"===PAGE {number}===")
            print(text)
        return 0
    print("没有可用的 PDF 解析库（需要 PyMuPDF 或 pypdf）", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
