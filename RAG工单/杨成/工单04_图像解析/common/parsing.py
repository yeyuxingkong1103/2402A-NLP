"""文档读取与文本分块。"""

from pathlib import Path
from typing import Iterable

from common.models import DocumentChunk


def read_document(path: str | Path) -> list[tuple[int, str]]:
    """读取 PDF 或 TXT，返回 1-based 页码与文本。"""
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(document_path))
        return [(page_number, page.extract_text() or "") for page_number, page in enumerate(reader.pages, 1)]
    if suffix != ".txt":
        raise ValueError("仅支持 .pdf 和 .txt 文件")

    text = document_path.read_text(encoding="utf-8")
    return [(1, text)]


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    chunk_size: int = 800,
    overlap: int = 100,
    source: str = "",
) -> list[DocumentChunk]:
    """将带页码文本切成字符级重叠块。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for page, text in pages:
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(DocumentChunk(source=source, page=page, content=text[start:end]))
            if end == len(text):
                break
            start = end - overlap
    return chunks
