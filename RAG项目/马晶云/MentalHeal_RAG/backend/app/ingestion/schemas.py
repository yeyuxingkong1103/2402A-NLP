from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExtractedPage:
    page_number: int
    native_text: str = ""
    ocr_text: str = ""
    vision_ocr_text: str = ""
    tables: list[str] = field(default_factory=list)
    mineru_text: str = ""


@dataclass
class TextChunk:
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    document_id: str
    source_path: str
    title: str
    page_count: int
    pages: list[ExtractedPage]
    chunks: list[TextChunk]
    engines: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
