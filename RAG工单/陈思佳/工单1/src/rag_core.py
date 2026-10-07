from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int
    page: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def load_pdf(path: str | Path) -> list[Document]:
    """Extract page-level text from a PDF without requiring an LLM."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("请先安装 pypdf：pip install pypdf") from exc

    path = Path(path)
    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(Document(text=text, source=str(path), page=page_number))
    return documents


def split_documents(
    documents: Iterable[Document], chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Split pages while preserving source and page metadata."""
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size 必须大于 0，overlap 不能为负数")
    overlap = min(overlap, chunk_size - 1)

    chunks: list[Chunk] = []
    for document in documents:
        text = re.sub(r"\s+", " ", document.text).strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=document.source,
                        page=document.page,
                        index=len(chunks),
                        metadata=document.metadata,
                    )
                )
            if end == len(text):
                break
            start = end - overlap
    return chunks
