"""把同一份 PDF 的前几页分别交给四种解析器，保存答辩用 TXT。"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.offline_pipeline import load_env, parse_document  # noqa: E402


PARSERS = ("pymupdf", "pdfplumber", "paddleocr", "mineru")


def first_pages(source: Path, target: Path, count: int) -> int:
    """制作临时节选 PDF；原始指南和知识库不会被改动。"""
    import pymupdf

    with pymupdf.open(source) as original:
        actual = min(count, original.page_count)
        if actual < 1:
            raise ValueError("PDF没有可解析的页面")
        with pymupdf.open() as excerpt:
            excerpt.insert_pdf(original, from_page=0, to_page=actual - 1)
            excerpt.save(target)
    return actual


def pdfplumber_text(pdf_path: Path) -> str:
    """PDFPlumber读取正文；发现表格时保留页码、行和列。"""
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as document:
        for page_number, page in enumerate(document.pages, 1):
            parts = [page.extract_text() or ""]
            for table_number, table in enumerate(page.extract_tables(), 1):
                rows = [" | ".join((cell or "").strip() for cell in row)
                        for row in table]
                parts.append(f"[第{page_number}页表格{table_number}]\n" + "\n".join(rows))
            pages.append("\n\n".join(part for part in parts if part.strip()))
    return "\n\f\n".join(pages)


def export_samples(source: Path, output_dir: Path, page_count: int,
                   parser_names: tuple[str, ...] = PARSERS) -> dict[str, Path]:
    """每种工具独立解析，成功且有正文才写入自己的 TXT。"""
    source = source.resolve()
    if not source.is_file() or source.suffix.lower() != ".pdf":
        raise ValueError(f"找不到有效PDF：{source}")
    if page_count < 1:
        raise ValueError("--pages必须大于0")
    unknown = set(parser_names) - set(PARSERS)
    if unknown:
        raise ValueError(f"未知解析器：{', '.join(sorted(unknown))}")

    load_env(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    with tempfile.TemporaryDirectory(prefix="rag-parser-sample-") as directory:
        excerpt = Path(directory) / "first-pages.pdf"
        actual_pages = first_pages(source, excerpt, page_count)
        for parser in parser_names:
            try:
                if parser == "pdfplumber":
                    body = pdfplumber_text(excerpt)
                else:
                    body, used = parse_document(excerpt, engine=parser)
                    if used != parser:
                        raise RuntimeError(f"实际解析器为{used}，与指定的{parser}不符")
                if not body.strip():
                    raise RuntimeError("没有解析出文字")
                target = output_dir / f"{parser}.txt"
                header = (f"解析器：{parser}\n来源：{source.name}\n"
                          f"范围：原PDF第1-{actual_pages}页，共{actual_pages}页（节选）\n"
                          "下面是该工具的真实解析结果。\n\n")
                target.write_text(header + body.strip() + "\n", encoding="utf-8")
                written[parser] = target
                print(f"EXPORTED: {parser} -> {target} ({len(body)}字)")
            except Exception as error:
                print(f"FAILED: {parser} -> {error}", file=sys.stderr)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="生成四种PDF解析器的答辩TXT")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/raw/国家基层高血压防治管理指南2020版.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "data/parser_samples")
    parser.add_argument("--pages", type=int, default=3, help="原PDF前几页，默认3页")
    parser.add_argument("--parser", choices=("all", *PARSERS), default="all")
    args = parser.parse_args()
    names = PARSERS if args.parser == "all" else (args.parser,)
    written = export_samples(args.input, args.output, args.pages, names)
    print(f"RESULT: {len(written)}/{len(names)} 个真实TXT已生成")
    return 0 if len(written) == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
