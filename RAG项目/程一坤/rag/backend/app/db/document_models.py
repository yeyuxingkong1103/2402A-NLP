"""文档域实体定义：文档身份、版本、分块与采集/导入审计。

实体定义按域分置；对外统一入口是 app/db/sql_models.py
（调用方一律从 sql_models 导入，不要直接 import 本文件，
否则实体定义分散后容易出现两套 import 并存的维护盲区）。
"""

from datetime import datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utc_now


class Document(Base):
    """法律文档的逻辑身份；同一来源 URL 的不同内容版本归到同一文档。

    身份与内容刻意分离：身份稳定（URL 不变则 Document 行不变），
    内容演进（哈希变化才产生新版本行）；检索与引用只面向版本，
    身份行只负责把同一来源的多个版本聚在一起。
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 对外稳定标识（32 位十六进制），业务表存 key 不存自增 id，换主键不牵动下游
    document_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # MySQL utf8mb4 唯一索引最大 3072 字节，768 字符是安全上限。
    source_url: Mapped[str] = mapped_column(String(768), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    # 指向当前版本的 id 但不加外键：版本行先于文档行落库时会因前向引用失败，
    # 且 versions 表按 content_hash 幂等去重后 id 可能变化，弱关联更稳
    current_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)


class DocumentVersion(Base):
    """文档内容版本；同一文档的内容哈希变化会生成新版本。"""

    __tablename__ = "document_versions"
    __table_args__ = (
        # 唯一键包含 chunk_fingerprint：同一正文允许因切块方式不同而并存多个版本
        # （chunking_changed 场景旧版本保留为历史）；老行指纹为 NULL，
        # MySQL/SQLite 对含 NULL 的唯一键视为互不冲突，不影响存量数据
        UniqueConstraint(
            "document_id", "content_hash", "chunk_fingerprint",
            name="uq_document_versions_document_hash_fp",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 对外稳定标识（版本粒度），引用与日志使用它而非自增 id
    version_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False)
    # 正文 SHA-256（增量判定第一比较位），与切块指纹构成双比较增量规则
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # 切块指纹（切块器版本 + 分块结构哈希）；NULL = 升级前产出的存量版本，未知
    chunk_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 原始 HTML 落盘路径：正文大文件不入库，重切分/重清洗时可从原件重放
    raw_file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    # 内容形态（text/html 等），决定清洗器与切块参数的选择
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    # 清洗后正文：引用展示与切分输入的唯一权威来源（Text 大字段）
    cleaned_content: Mapped[str] = mapped_column(Text, nullable=False)
    # 审核状态（阶段6）：pending_review-待审核 / approved-已发布 / rejected-已驳回
    # 新导入版本一律 pending_review；检索侧（向量+关键词）只读 approved。
    # 存量 11 版本（审核机制上线前已在用）已显式迁移为 approved（migrate_review_status.py）
    version_status: Mapped[str] = mapped_column(String(32), nullable=False)
    # 嵌入处理进度（awaiting_embedding/embedded...），与审核状态正交：两条流水线各管各的
    processing_status: Mapped[str] = mapped_column(String(32), nullable=False)
    # 审核留痕（谁、何时、什么决定/说明；未审核时三者均 NULL）
    reviewed_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    review_note: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


class DocumentChunk(Base):
    """可检索的父子分块；MySQL 以正文和检索文本承担关键词检索。

    子块与父块同表、用 parent_chunk_key 表达层级而非拆两张表：
    归并、溯源、引用展示都只需要"按 key 查行"，单表即可完成。
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        # SQLite 单元测试使用普通索引；MySQL FULLTEXT 在迁移或集成建表阶段补充。
        Index("ix_document_chunks_content", "content"),
        Index("ix_document_chunks_retrieval_text", "retrieval_text"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 对外稳定标识（块粒度）；向量库与检索结果都以它为对账键
    chunk_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    document_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_versions.id"), nullable=False
    )
    # 冗余存储父块 key：融合归并只需本表即可回溯父子，省一次自联
    parent_chunk_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # parent / article 两级；检索链路按类型决定归并粒度
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # 条号（阿拉伯数字，如 "47"），引用展示的锚点字段；款/项号同理
    article_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    paragraph_number: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 款号（阿拉伯数字）
    item_number: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 项号（阿拉伯数字）
    # 版本内顺序号：还原条款顺序、定位上下文相邻块都依赖它
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    # 块正文原文；retrieval_text 是给 BM25 用的清洗版——两者分开存，
    # 引用展示用原文、关键词检索用清洗版，互不污染
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


class ChunkSummary(Base):
    """条文一句话摘要（批次 37）：chunk_key → summary 的单一事实来源。

    设计要点：
    - 独立成表而不加在 document_chunks 上：摘要是离线 LLM 产物（summarize_chunks.py 生成），
      有自己的生命周期（重生成、按 model 口径追溯），与分块本身的导入/替换解耦；
    - chunk_key 与 Milvus 主键同名同值：索引服务写向量时按 key 批量 join 本表，
      引用展示（chat sources）也按 key 取，两边同源不打架；
    - model 列记录生成时用的 LLM 名，换模型重跑时可解释"摘要口径变化"。
    """

    __tablename__ = "chunk_summaries"

    # 与 document_chunks.chunk_key 同值（Milvus 主键也是它）；一个 chunk 只有一句摘要
    chunk_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # 生成摘要时使用的 LLM 模型名（追溯口径用，如 deepseek-flash）
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=utc_now, onupdate=utc_now, nullable=False
    )


class CrawlRecord(Base):
    """采集审计记录；失败摘要只保存脱敏后的短文本。

    审计的是"抓取行为"而非"入库结果"（后者看 import_records）：
    即使包最终没入库，抓取本身发生过就要留痕，失败排障全靠这张表。
    """

    __tablename__ = "crawl_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    package_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    # 可空：失败采集没有产物可指（只有抓取行为本身），审计仍需留痕
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    document_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_versions.id"), nullable=True
    )
    # success/failed 两态；失败详情看 failed_stage + error_summary
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # 失败发生在管道哪一步（下载/校验/切分...），排障只看这一列即可定位环节
    failed_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 只存异常类型等短摘要（512 截断），不落正文/连接串，防止敏感信息入审计表
    error_summary: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


class ImportRecord(Base):
    """数据包级导入审计；`package_id` 保证重复导入可幂等判断。"""

    __tablename__ = "import_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    package_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    # 双版本号：schema 是数据包格式版本，pipeline 是入库代码版本——
    # 排障时能区分"包是旧格式"还是"入库逻辑是旧版本"
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # imported/already_imported/failed；幂等判定与重复导入去重都依据它
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    document_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_versions.id"), nullable=True
    )
    error_summary: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)
