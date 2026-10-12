# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：接口数据契约。所有请求/响应用 Pydantic 模型显式声明，让 FastAPI 自动校验并生成文档。
from typing import List, Optional                                                # 类型标注：列表与可空字段

from pydantic import BaseModel, Field                                            # Pydantic 基类与字段约束


class Reference(BaseModel):                                                      # 单条引用来源（工单要求"检索结果必须带引用来源"）
    """回答所依据的原文片段及其出处，前端据此展示文件名、页码与原文。"""
    doc: str = Field(default="", description="来源文件名，如 招股说明书2")          # 来源文档名
    page: int = Field(default=0, description="页码，1 起，与 PDF 页码一致")         # 来源页码
    chunk_id: str = Field(default="", description="全局唯一块号")                  # 块标识，便于排查与去重
    score: float = Field(default=0.0, description="重排得分，越大越相关")           # 重排后的相关性得分
    snippet: str = Field(default="", description="引用原文片段")                   # 原文片段（已截断，供前端展示）
    locator: str = Field(default="", description="可读定位串，如 招股说明书2 P66")   # 供前端直接展示的定位信息


class ChatRequest(BaseModel):                                                    # POST /chat 的请求体
    """一轮对话请求：会话号决定多轮上下文，question 为中英文皆可的自然语言问题。"""
    session_id: str = Field(default="default", description="会话标识，用于隔离多轮上下文")  # 会话号
    question: str = Field(..., min_length=1, description="用户问题，支持中文与英文")          # 用户问题，必填


class ChatResponse(BaseModel):                                                   # POST /chat 的响应体
    """一轮对话响应：除答案与引用外，回带改写后的问题与缓存标记，便于前端透明展示。"""
    answer: str = Field(..., description="基于检索片段的回答")                      # 最终回答
    references: List[Reference] = Field(default_factory=list, description="引用来源列表")  # 引用列表
    rewritten_query: str = Field(default="", description="Query 改写后的独立问题")   # 改写结果，前端可用于展示
    cached: bool = Field(default=False, description="是否命中缓存")                # 缓存命中标记
    degraded: List[str] = Field(default_factory=list, description="降级项，如 rerank/milvus/bm25")  # 降级提示，缺了它会静默失效
    latency_ms: int = Field(default=0, description="服务端处理耗时（毫秒）")         # 端到端耗时
    session_id: str = Field(default="default", description="会话标识，原样回带")     # 会话号回带


class FeedbackRequest(BaseModel):                                                # POST /feedback 的请求体
    """用户反馈：点赞/点踩 + 文字意见，落盘留存供后续优化参考。"""
    session_id: str = Field(default="default", description="会话标识")             # 会话号
    question: str = Field(default="", description="被评价的问题")                  # 对应问题
    answer: str = Field(default="", description="被评价的回答")                    # 对应回答
    rating: int = Field(default=0, ge=-1, le=1, description="1=点赞, -1=点踩, 0=中立")  # 评分
    comment: Optional[str] = Field(default="", description="文字反馈")             # 文字意见
    rewritten_query: str = Field(default="", description="该轮改写后的问题，便于定位问题")  # 便于复盘


class FeedbackResponse(BaseModel):                                               # POST /feedback 的响应体
    """反馈受理结果。"""
    ok: bool = Field(default=True, description="是否记录成功")                     # 成功标记
    message: str = Field(default="", description="提示信息")                       # 提示文案


class HealthResponse(BaseModel):                                                 # GET /health 的响应体
    """健康检查：逐个组件报告可用性，前端首屏据此提示用户。"""
    status: str = Field(default="ok", description="overall: ok / degraded")       # 总体状态
    milvus: bool = Field(default=False, description="Milvus 是否可用")             # 向量库
    redis: bool = Field(default=False, description="Redis 是否可用")               # 缓存
    llm: bool = Field(default=False, description="DeepSeek 密钥是否已配置")         # 大模型
    embed_model: bool = Field(default=False, description="向量模型是否已加载")      # 向量模型
    rerank_model: bool = Field(default=False, description="重排模型是否已加载")     # 重排模型
    chunks: int = Field(default=0, description="知识库分块总数")                   # 知识库规模
    docs: List[str] = Field(default_factory=list, description="已入库文档名")       # 已入库文档
    detail: str = Field(default="", description="补充说明")                        # 补充说明
