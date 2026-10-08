# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/schemas_v3.py —— 工单三 Pydantic 请求/响应模型
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ================= 请求模型 =================
class AskV3Request(BaseModel):
    """工单三：POST /api/v3/ask 请求"""
    question: str = Field(..., description="用户问题（中/英）")
    doc_id: Optional[str] = Field(None, description="文档过滤：招股说明书1/招股说明书2")
    top_k: int = Field(5, ge=1, le=20)
    use_table: bool = Field(True, description="是否启用表格检索")
    use_rag: bool = Field(True, description="False=纯LLM模式")
    lang: Optional[str] = Field(None, description="强制回答语言 zh/en")


class UploadRequest(BaseModel):
    """工单三：POST /api/v3/upload 请求（JSON 元数据）"""
    pdf_path: str = Field(..., description="PDF 文件路径")
    doc_name: Optional[str] = Field(None, description="文档名称")
    company: Optional[str] = Field(None, description="公司名")


class EvaluateRequest(BaseModel):
    """工单三：POST /api/v3/evaluate 请求"""
    questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="自定义问题列表（默认14题）"
    )
    modes: List[str] = Field(
        ["rag_v3", "pure_llm"], description="评估模式"
    )


# ================= 响应模型 =================
class ReferenceV3(BaseModel):
    """工单三：引用来源"""
    type: str = Field(..., description="text/table")
    ref_id: str = Field(...)
    page: Optional[Any] = None
    doc_id: Optional[str] = None
    table_id: Optional[str] = None
    score: float = 0.0
    preview: str = ""


class AskV3Response(BaseModel):
    """工单三：POST /api/v3/ask 响应"""
    mode: str = Field(..., description="rag_v3/pure_llm")
    question: str
    answer: str
    references: List[ReferenceV3] = []
    latency_ms: float = 0.0
    breakdown: Dict[str, float] = {}
    route: Dict[str, Any] = {}
    retrieved_text_chunks: List[Dict[str, Any]] = []
    retrieved_tables: List[Dict[str, Any]] = []
    lang: str = "zh"
    translated: bool = False


class TableV3Response(BaseModel):
    """工单三：GET /api/v3/tables/{doc_id} 响应"""
    doc_id: str
    total: int
    tables: List[Dict[str, Any]] = []


class UploadResponse(BaseModel):
    """工单三：POST /api/v3/upload 响应"""
    success: bool
    doc_name: str
    doc_id: str
    text_chunks: int = 0
    table_chunks: int = 0
    message: str = ""


class EvaluateResponse(BaseModel):
    """工单三：POST /api/v3/evaluate 响应"""
    total_questions: int
    results: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """工单三：健康检查"""
    status: str = "ok"
    version: str = "v3"
    table_collection: str = "rag_tables"
    text_collection: str = "rag_chunks"
