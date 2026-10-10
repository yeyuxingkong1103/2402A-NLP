# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""统一数据契约。所有模块间传递的对象在此定义，保证接口稳定。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

BlockType = Literal["text", "table", "image"]


@dataclass
class TextBlock:
    doc_id: str
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    has_rotated_text: bool = False
    section_path: str = ""

    @property
    def block_type(self) -> BlockType:
        return "text"


@dataclass
class TableBlock:
    doc_id: str
    page: int
    bbox: tuple[float, float, float, float]
    markdown: str
    n_rows: int = 0
    n_cols: int = 0
    section_path: str = ""

    @property
    def block_type(self) -> BlockType:
        return "table"


@dataclass
class FigureBlock:
    """图区。bbox 为已按 crop_margin 扩边后的最终裁切框。"""
    doc_id: str
    page: int
    bbox: tuple[float, float, float, float]
    image_path: str
    caption: str = ""
    detect_method: Literal["vector_cluster", "caption_anchor", "full_page"] = "vector_cluster"
    confidence: float = 0.0
    section_path: str = ""
    # 下列两项由 vlparser 回填
    description: str = ""
    clip_vector: list[float] = field(default_factory=list)
    parse_warning: str = ""

    @property
    def block_type(self) -> BlockType:
        return "image"

    @property
    def figure_id(self) -> str:
        return f"{self.doc_id}#p{self.page}#fig"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    block_type: BlockType
    source_id: str
    text: str
    section_path: str = ""
    lang: str = "zh"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("chunk_id", None)
        return d


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    page: int
    block_type: BlockType
    source_id: str
    text: str
    score: float
    channel: str = ""          # dense / sparse / clip / fused / rerank
    image_path: str = ""
    # RC2：融合去重键。图描述文本块指向其 image 块 chunk_id（同图只占一席）；
    # 其余块留空，融合时按 chunk_id 去重（行为与修复前一致）。
    dedup_key: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    lang: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    latency_ms: float = 0.0
    llm_backend: str = ""
    refused: bool = False
    # RC7：LLM 结构化输出的「证据是否充分 / 能否作答」字段（主判据）。
    # 非 JSON 后端（Ollama / retrieval_only）或模型未产出标记时为 None，
    # 此时 refused 由 detect_answer_refusal 正则兜底。
    answerable: bool | None = None
    refusal_source: str = ""   # structured / structured+regex / regex / retrieval_only / offtopic
