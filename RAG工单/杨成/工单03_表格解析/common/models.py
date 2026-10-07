"""RAG 共享数据模型。"""

from dataclasses import dataclass, field
import hashlib
from typing import Any


def _stable_id(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass
class DocumentChunk:
    source: str
    page: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.metadata = {**self.metadata, "page": self.page}
        self.chunk_id = _stable_id(self.source, self.page, self.content)


@dataclass
class TableRecord:
    source: str
    page: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    table_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.metadata = {**self.metadata, "page": self.page}
        self.table_id = _stable_id(self.source, self.page, self.content)


@dataclass
class ImageRecord:
    source: str
    page: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str = ""
    image_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.path and "image_path" not in self.metadata:
            self.metadata["image_path"] = self.path
        elif not self.path and self.metadata.get("image_path"):
            self.path = str(self.metadata["image_path"])
        self.metadata = {**self.metadata, "page": self.page}
        self.image_id = _stable_id(self.source, self.page, self.content)

    @property
    def text(self) -> str:
        return self.content


@dataclass
class SearchResult:
    content: str
    source: str
    score: float
    page: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.answer

    @property
    def citations(self) -> list[SearchResult]:
        return self.sources
