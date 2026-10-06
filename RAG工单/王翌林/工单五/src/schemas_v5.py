# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query 理解优化任务
src/schemas_v5.py —— 工单五 API v5 Pydantic 模型（新增文件）

多轮对话相关模型：Chat 请求/响应、历史记录、反馈。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------- 工单五：/api/v5/chat ----------------
class ChatV5Request(BaseModel):
    """工单五：多轮对话请求"""
    question: str
    session_id: Optional[str] = None       # 不传则自动创建新会话
    doc_id: Optional[str] = None           # 可选：强制指定文档
    use_image: bool = True
    use_rag: bool = True
    lang_override: Optional[str] = None
    top_k: Optional[int] = None


class ChatV5Response(BaseModel):
    """工单五：多轮对话响应（在 v4 ask 基础上增加会话/指代消解字段）"""
    session_id: str
    resolved_query: str                    # 消解改写后的问句
    entity: Optional[str] = None           # 当前实体（公司）
    doc_id: Optional[str] = None
    is_followup: bool = False
    coref_strategy: str = "direct"
    # 以下复用 v4 ask 输出
    mode: str = "rag_v5"
    query: str = ""
    route: Dict[str, Any] = {}
    answer: str
    references: List[Dict[str, Any]] = []
    retrieved_text_chunks: List[Dict[str, Any]] = []
    retrieved_tables: List[Dict[str, Any]] = []
    retrieved_images: List[Dict[str, Any]] = []
    latency_ms: float = 0.0
    conversation_latency_ms: float = 0.0
    breakdown: Dict[str, float] = {}
    token_usage: Dict[str, Any] = {}


# ---------------- 工单五：/api/v5/history/{session_id} ----------------
class HistoryTurn(BaseModel):
    role: str
    content: str
    ts: float = 0.0


class HistoryV5Response(BaseModel):
    session_id: str
    history: List[Dict[str, Any]] = []
    current_entity: Optional[str] = None
    current_doc: Optional[str] = None


# ---------------- 工单五：/api/v5/feedback ----------------
class FeedbackV5Request(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: str                           # up / down
    comment: str = ""
    latency_ms: float = 0.0


class FeedbackV5Response(BaseModel):
    ok: bool = True
    message: str = ""


# ---------------- 工单五：/api/v5/health ----------------
class HealthV5Response(BaseModel):
    status: str = "ok"
    engine: str = "conv_v5"
    sessions: int = 0
    milvus_images_rows: int = 0
    work_order: str = Field(
        default="人工智能NLP-RAG-Query 理解优化任务")
