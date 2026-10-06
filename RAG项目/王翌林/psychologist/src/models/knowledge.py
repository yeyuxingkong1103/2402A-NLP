"""知识库文档与分块表。

RAG（检索增强生成）的数据落库层：
- KnowledgeDoc：一份上传的知识文档（PDF/Word 等）及其解析状态；
- KnowledgeChunk：文档切分后的文本块，是向量检索的最小单位，并与向量库（Milvus）关联。
"""
import datetime as dt

# ForeignKey：文档归属人设/分块归属文档都用外键维护引用完整性；Index：加速按外键过滤
from sqlalchemy import (BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, func)
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class KnowledgeDoc(Base):
    """知识文档表：记录上传的原始文件及其解析进度。

    一份文档解析后会切成多个 chunk 存入 KnowledgeChunk，chunk_count 缓存切块数量。
    """

    __tablename__ = "knowledge_docs"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 外键 -> counselor_personas.id：文档归属于某个人设，实现知识隔离（不同角色用不同知识）
    persona_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("counselor_personas.id"), nullable=False
    )
    # 文档标题：可空（有些文件无标题）
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    # 来源说明：如书名/网址，可空
    source: Mapped[str] = mapped_column(String(512), nullable=True)
    # 文件存储路径：指向磁盘或对象存储中的原始文件，可空
    file_path: Mapped[str] = mapped_column(String(512), nullable=True)
    # 文件类型：如 pdf、docx，用于选择解析器，可空
    file_type: Mapped[str] = mapped_column(String(32), nullable=True)
    # 解析状态：pending/processing/success/failed，默认 pending，驱动异步解析流程
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # 切块数量：默认 0，解析完成后更新；冗余存储可避免每次都 count 关联表
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # 错误信息：解析失败时记录原因，Text 类型以容纳堆栈/详情，可空
    error_msg: Mapped[str] = mapped_column(Text, nullable=True)
    # 上传时间
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    # 更新时间：解析状态变化时刷新
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # 按人设查文档是常用筛选，建索引加速
    __table_args__ = (Index("idx_doc_persona", "persona_id"),)


class KnowledgeChunk(Base):
    """知识分块表：文档切分后的文本片段，是检索命中的最小单元。

    每个 chunk 通过 milvus_id 与本条记录对应，检索时由向量库返回 id 再回表取原文。
    """

    __tablename__ = "knowledge_chunks"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 外键 -> knowledge_docs.id：分块所属文档，保证删除文档时可级联定位分块
    doc_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("knowledge_docs.id"), nullable=False
    )
    # 所属人设 id：冗余字段，检索时按人设过滤而无需 join 文档表，提升查询效率
    # 注：此处刻意不加外键，避免跨表约束影响高频写入与批量导入
    persona_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 块序号：标识该分块在文档中的顺序，便于按原文顺序拼装上下文
    chunk_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 分块正文：Text 类型（单块可能较长），检索后作为上下文喂给 LLM，不可空
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 分块摘要：可空，长度 2048，用于展示或粗筛，减少大文本直接比较
    summary: Mapped[str] = mapped_column(String(2048), nullable=True)
    # 向量库中的主键（Milvus id）：可空，未入库前为空，用于与向量检索结果互相映射
    milvus_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    # 创建时间
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    # 两个索引分别加速“按文档查分块”“按人设查分块”这两类高频查询
    __table_args__ = (
        Index("idx_chunk_doc", "doc_id"),
        Index("idx_chunk_persona", "persona_id"),
    )