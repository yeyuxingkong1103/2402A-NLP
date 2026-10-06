"""知识库 Schema（文档管理、上传结果、索引重建、检索命中与 RAG 评测）。

知识库是 RAG 系统的“资料源头”，本模块对应它的完整生命周期：
上传文档 → 切块入库（chunk）→ 重建/刷新向量索引 → 在线检索 → 离线质量评测（Ragas）。

关键概念：
- doc（文档）= 一份原始资料（PDF/Word/TXT 等）；
- chunk（文本块）= 文档切分后的段落，是向量库最小检索单元；
- 一条检索结果 hits 里通常包含多个 chunk，每个 chunk 带相似度 score。
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocItem(BaseModel):
    """知识文档列表中的单条记录。"""
    model_config = ConfigDict(from_attributes=True)  # 允许从 ORM 对象直接构造

    id: int
    # 归属角色：知识库按 persona 隔离，保证“这位心理医生只引用自己的资料”
    persona_id: int
    # 文档标题：可选，未填时后端通常用文件名兜底
    title: Optional[str] = None
    # 来源：原始文件名或 URL，便于溯源
    source: Optional[str] = None
    # 文件类型（pdf/docx/txt/md 等），前端可据此显示对应图标
    file_type: Optional[str] = None
    # 处理状态：pending=待处理、processing=切块中、ready=可用、failed=失败。
    # 默认 "pending" 表示刚上传尚未开始处理
    status: str = "pending"
    # 切块数量：让用户看到“这份资料被拆成了多少段”，也为 0 时提示可能切分失败
    chunk_count: int = 0
    # 失败原因：仅当 status="failed" 时有值，便于用户自助排查（如解析乱码）
    error_msg: Optional[str] = None
    created_at: Optional[str] = None


class KnowledgeUploadResult(BaseModel):
    """文档上传接口的返回结果。"""
    # 新建的文档主键，前端据此轮询处理进度
    doc_id: int
    title: Optional[str] = None
    persona_id: int
    # 上传后的最终状态（同步处理完则可能是 ready，异步则为 pending）
    status: str
    # 切出的文本块总数
    chunk_count: int
    # 实际写入向量库的条数。它与 chunk_count 可能不等：
    # 若部分块因空内容/重复被过滤，则 stored < chunk_count，
    # 单独暴露这个差值有助于判断入库是否完整
    stored: int


class KnowledgeRebuildRequest(BaseModel):
    """知识库索引重建请求体。"""
    # 目标角色：可选。不传表示重建全部角色的索引；
    # 传了则只重建该角色的知识库，避免影响其他角色（可选是为了支持全量运维操作）
    persona_id: Optional[int] = None
    # 是否先清空已有向量再重建。默认 False（增量/幂等覆盖，更安全）；
    # 置 True 才做“全量重建”，属于较重且不可逆的操作
    drop_existing: bool = False


class KnowledgeSearchRequest(BaseModel):
    """知识库检索请求体（调试/演示用，也服务于在线问答的检索环节）。"""
    persona_id: int
    # 检索问题：至少 1 个字符，空串无法做向量化，故设下限 1
    query: str = Field(min_length=1)
    # 返回条数 top_k，默认 5。
    # 选 5 是“召回质量”与“上下文长度”的折中：太少可能漏掉关键信息，
    # 太多会稀释提示词、提高成本并增加“答非所问”的风险
    top_k: int = 5


class KnowledgeSearchHit(BaseModel):
    """单条检索命中结果（一个文本块）。"""
    # 所属文档与文本块主键，可选：来自不同存储后端时可能拿不全
    doc_id: Optional[int] = None
    chunk_id: Optional[int] = None
    # 来源标识，用于展示引用出处
    source: Optional[str] = None
    # 命中的原始文本内容：必填，这是给用户看/给模型用的核心信息
    text: str
    # 相似度分数：必填，便于排序与前端展示置信度
    score: float


class KnowledgeSearchResult(BaseModel):
    """检索结果集合。"""
    # 用户原始问题，原样回显便于前端确认请求对应关系
    query: str
    # 改写后的查询词：可选。RAG 常做“查询改写/扩展”以提升召回，
    # 这里把改写结果透出，便于调试检索质量
    rewritten_query: Optional[str] = None
    # 命中列表，默认空列表表示未召回到任何内容
    hits: List[KnowledgeSearchHit] = []


class RagasEvalRequest(BaseModel):
    """RAG 质量评测（Ragas）请求体。"""
    persona_id: int
    # 评测数据集路径：可选，不传则使用项目内置的默认测试集
    dataset_path: Optional[str] = None
    # 参与评测的样本数，默认 10。
    # 评测需调用大模型逐条打分，成本高、耗时长，故默认只跑小样本快速把关
    limit: int = 10


class RagasEvalResult(BaseModel):
    """RAG 评测结果。"""
    persona_id: int
    # 实际评测的样本数（可能小于请求的 limit，如数据集不足）
    samples: int
    # 各项指标（如 faithfulness、answer_relevancy、context_precision）的键值对。
    # 用裸 dict 是因为指标项随 Ragas 版本可变，写死字段会导致升级即报错
    metrics: dict
    # 详细评测报告落盘路径，可选：便于人工复核每一条样本的打分明细
    report_path: Optional[str] = None