# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/schemas_v4.py —— 工单四 API v4 Pydantic 模型（新增文件，不影响 api_v3）
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------- 工单四：/api/v4/ask ----------------
class AskV4Request(BaseModel):
    """工单四：问答请求（question 必填；use_image 控制图像检索）"""
    question: str
    doc_id: Optional[str] = None            # 工单四：单册过滤（如"招股说明书2"）
    company: Optional[str] = None
    use_image: bool = True                  # 工单四：图像感知检索开关
    use_rag: bool = True                    # 工单四：False=纯 LLM 模式
    lang_override: Optional[str] = None     # 工单四：zh/en 中英文问答
    top_k: Optional[int] = None


class ImageRef(BaseModel):
    """工单四：图像引用（路径/页码/caption/OCR/VQA）"""
    type: str = "image"
    ref_id: str
    doc_id: Optional[str] = None
    image_id: Optional[str] = None
    page: Optional[int] = None
    path: Optional[str] = None
    caption: str = ""
    ocr_text: str = ""
    vqa_text: str = ""
    score: float = 0.0


class AskV4Response(BaseModel):
    """工单四：问答响应（含三路引用与明细）"""
    mode: str
    query: str
    route: Dict[str, Any] = {}
    answer: str
    references: List[Dict[str, Any]] = []
    retrieved_text_chunks: List[Dict[str, Any]] = []
    retrieved_tables: List[Dict[str, Any]] = []
    retrieved_images: List[Dict[str, Any]] = []
    latency_ms: float = 0.0
    breakdown: Dict[str, float] = {}
    token_usage: Dict[str, Any] = {}


# ---------------- 工单四：/api/v4/health ----------------
class HealthV4Response(BaseModel):
    status: str = "ok"
    engine: str = "rag_v4"
    milvus_images_rows: int = 0
    work_order: str = Field(default="人工智能NLP-RAG-图像内容解析及检索优化")


# ---------------- 工单四：/api/v4/images/{doc_id} ----------------
class ImageItem(BaseModel):
    image_id: str
    page: int = 0
    path: str
    caption: str = ""
    ocr_text: str = ""
    vqa_text: str = ""


class ImagesV4Response(BaseModel):
    doc_id: str
    total: int
    images: List[ImageItem]


# ---------------- 工单四：/api/v4/upload ----------------
class UploadV4Response(BaseModel):
    doc_name: str
    images_extracted: int = 0
    images_parsed: int = 0
    images_ingested: int = 0
    message: str = ""


# ---------------- 工单四：/api/v4/evaluate ----------------
class EvalQuestion(BaseModel):
    id: int
    question: str
    doc_id: Optional[str] = None
    expected_keywords: List[str] = []
    qtype: str = "text"                     # 工单四：text/table/image


class EvaluateV4Request(BaseModel):
    questions: List[EvalQuestion]
    use_rag: bool = True


class EvalItem(BaseModel):
    id: int
    question: str
    qtype: str
    answer: str
    hit_keywords: List[str] = []
    hit_ratio: float = 0.0
    correct: bool = False
    latency_ms: float = 0.0
    route: Dict[str, Any] = {}


class EvaluateV4Response(BaseModel):
    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    items: List[EvalItem]
