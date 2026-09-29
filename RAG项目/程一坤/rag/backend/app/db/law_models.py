"""法规域实体定义：法规身份、版本时效与条/款/项三级条文结构。

实体定义按域分置；对外统一入口是 app/db/sql_models.py
（调用方一律从 sql_models 导入，不要直接 import 本文件，
否则实体定义分散后容易出现两套 import 并存的维护盲区）。
"""

from datetime import datetime

from sqlalchemy import Date, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utc_now


class Law(Base):
    """法规身份表：存储法规的基本身份信息，与具体版本无关。

    一部法规（如《劳动合同法》）可能有多个版本（原始版、修订版），
    但只有一个法规身份记录。
    """

    __tablename__ = "laws"

    # 法规主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 法规唯一标识（由系统生成，格式如 law_labor_contract）
    law_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    # 法规全称（如"中华人民共和国劳动合同法"）
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)

    # 法规简称（如"劳动合同法"）
    short_name: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # 文书类型（法律、行政法规、司法解释、部门规章等）
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # 效力等级（1-宪法，2-法律，3-行政法规，4-地方性法规，5-规章，数字越小效力越高）
    authority_level: Mapped[int] = mapped_column(Integer, nullable=False)

    # 发布机关（如"全国人民代表大会常务委员会"）
    issuing_authority: Mapped[str] = mapped_column(String(256), nullable=False)

    # 适用法域（national-全国，province-省级，city-市级等）
    jurisdiction: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # 法律领域（劳动法、合同法、民法等）
    legal_domain: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # 官方来源 URL（首次采集的来源）
    source_url: Mapped[str] = mapped_column(String(768), nullable=False)

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)

    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)


class LawVersion(Base):
    """法规版本表：存储法规的时效信息和版本关系。

    同一部法规的不同修订会产生不同的版本记录，每个版本有独立的
    生效日期、失效日期和状态。
    """

    __tablename__ = "law_versions"
    __table_args__ = (
        UniqueConstraint("law_id", "version_number", name="uq_law_versions_law_version"),
        Index("ix_law_versions_effective_dates", "effective_date", "expiration_date"),
    )

    # 版本主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 版本唯一标识（格式如 law_labor_contract_v2）
    version_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    # 所属法规 ID
    law_id: Mapped[int] = mapped_column(ForeignKey("laws.id"), nullable=False, index=True)

    # 版本号（如"2008版"、"2012修正"、"2013修订"）
    version_number: Mapped[str] = mapped_column(String(64), nullable=False)

    # 公布日期（法规正式公布的日期）
    promulgation_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)

    # 生效日期（法规开始生效的日期，用于时效查询）
    effective_date: Mapped[datetime | None] = mapped_column(Date, nullable=True, index=True)

    # 失效日期（法规停止生效的日期，NULL 表示现行有效）
    expiration_date: Mapped[datetime | None] = mapped_column(Date, nullable=True, index=True)

    # 版本状态，取值见 app/db/law_status.py（唯一定义处）：
    #   现行有效 / 已废止 / 已失效 / 不适用
    # 什么时候填：现行有效=页面标注现行有效或人工核对确认有效（历次修正属修订沿革，
    #   写进 revision_note，不影响"现行有效"这个判定）；已废止=有明确废止声明；
    #   已失效=页面标注已失效；不适用=案例材料等非规范性文件，本身没有生效/失效概念。
    # 谁参与时效判断：只有已废止 / 已失效 参与（判为失效）；不适用 不参与时效判断，
    #   空值同理不判失效（缺数据不等于已失效，否则待补录的法规会被静默隐藏）。
    # 历史教训：本列注释原写英文词表（effective/amended/repealed/superseded），
    #   而抽取器与消费点用的是中文，英文取值写进来不会被任何判断命中（静默失效）。
    status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    # 关联的文档版本 ID（指向 document_versions 表）
    document_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_versions.id"), nullable=True
    )

    # 修订说明（如"第一次修正"、"第二次修订"）
    revision_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)

    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)


class Article(Base):
    """条文表：存储法规的条文内容，支持条/款/项三级结构。

    一个"条"可以包含多个"款"，一个"款"可以包含多个"项"。
    例如："第四十七条第二款第一项"对应 article_path="47-2-1"。

    编号存储约定：
    - article_number / paragraph_number / item_number 一律存阿拉伯数字纯编号
    - 例如："第四十七条" → article_number="47"
    - "第九十九条之一" → article_number="99之1"（之一用「之」表示，避免与款号的 "-" 冲突）
    - "第九十九条之一第一款" → article_path="99之1-1"
    - 展示时转中文只需一处；入库时规范化由切块器完成（任务 1.2）
    - 同一版本内 article_path 唯一（唯一约束保证）
    - article_path 由 build_article_path() 统一生成（app/ingest/article_number_rules.py）
    - 长度约束：单个编号 ≤ 16 字符，article_path ≤ 128 字符
      （与 VARCHAR(64) / VARCHAR(128) 字段定义一致，修改字段长度必须同步修改校验逻辑）
    """

    __tablename__ = "articles"
    __table_args__ = (
        UniqueConstraint(
            "law_version_id",
            "article_path",
            name="uq_articles_version_path",
        ),
        Index("ix_articles_article_number", "article_number"),
    )

    # 条文主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 条文唯一标识（由 version_key + article_path 派生，如 law_labor_contract_v2_art47-2-1）
    article_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    # 所属法规版本 ID
    law_version_id: Mapped[int] = mapped_column(
        ForeignKey("law_versions.id"), nullable=False, index=True
    )

    # 条号（阿拉伯数字，如 "47" 表示第四十七条，"99之1" 表示第九十九条之一）
    article_number: Mapped[str] = mapped_column(String(64), nullable=False)

    # 款号（阿拉伯数字，如 "2" 表示第二款，NULL 表示条本身没有分款）
    paragraph_number: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 项号（阿拉伯数字，如 "1" 表示第一项，NULL 表示款本身没有分项）
    item_number: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 条文路径（"47" / "47-2" / "47-2-1" / "99之1"，由 build_article_path 生成，用于唯一约束）
    article_path: Mapped[str] = mapped_column(String(128), nullable=False)

    # 章节路径（如"第三章 劳动合同的履行和变更"）
    chapter_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 条文原文（法条的完整文本）
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 条文序号（用于排序，确保条文按正确顺序显示）
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
