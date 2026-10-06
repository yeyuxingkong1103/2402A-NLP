"""检索结果的数据结构，以及注入给大模型的「法源清单」组装。

单独成文件的原因：数据结构和文本组装是"纯"逻辑（不碰网络、不碰数据库），
和检索流程分开后，两边都能单独读懂、单独验证。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# 法规效力状态词表（唯一定义处）：展示文案与"失效判定"都取它，
# 不再在本文件里另抄一份中文（历史教训见 app/db/law_status.py 模块说明）
from app.db.law_status import EFFECTIVE, EXPIRED_STATUSES, LAPSED


@dataclass(frozen=True)
class RetrievedArticle:
    """一条可用于回答的法条片段。

    带齐"回答"和"引用"需要的全部信息（11 个引用字段）：
    正文、条号、款号、项号、法规名称、文书类型、法域、生效/失效日期、
    发布机关、来源地址、是否现行有效、检索分数。
    """

    # === 必需字段 ===
    chunk_key: str  # 最终引用的父块唯一标识
    content: str  # 条文正文（通常是父块，即整条法条）
    document_title: str  # 法规名称，如"中华人民共和国劳动合同法"
    source_url: str  # 官方来源地址
    recall_score: float  # 兼容字段；仅表示候选自身来源的召回分

    # === 条文定位字段（可选）===
    article_number: str | None = None  # 条号，如 "40"；案例类文档可能没有
    paragraph_number: str | None = None  # 款号，如 "1"（阿拉伯数字）
    item_number: str | None = None  # 项号，如 "2"（阿拉伯数字）

    # === 引用元数据字段（任务书 3-B 要求的 11 个字段）===
    document_type: str | None = None  # 文书类型，如"法律"、"行政法规"、"司法解释"
    jurisdiction: str | None = None  # 法域，如"中国大陆"
    effective_date: str | None = None  # 生效日期，YYYY-MM-DD 格式
    expiration_date: str | None = None  # 失效日期，YYYY-MM-DD 格式；现行有效时为 None
    issuing_authority: str | None = None  # 发布机关，如"全国人民代表大会常务委员会"
    is_current: bool | None = None  # 是否现行有效

    # === 检索分数字段 ===
    score_sources: tuple[str, ...] = ()  # 命中来源，如 ("vector", "keyword")
    vector_score: float | None = None  # 向量余弦相似度
    keyword_score: float | None = None  # BM25 分数，与向量分不可直接比较
    fusion_score: float | None = None  # RRF 融合分数
    rerank_score: float | None = None  # 重排阶段的真实相关度分数；未重排时为 None
    parent_chunk_key: str | None = None  # 子块命中时对应的父块 key
    document_id: str | None = None  # 文档公开标识（Document.document_key）
    # 批次 37：条文一句话摘要（chunk_summaries 表，离线 LLM 生成）；
    # 随正文一起取出，供 citation 事件展示在引用卡片上。未生成时为 None。
    summary: str | None = None


def build_context_block(
    articles: Sequence[RetrievedArticle],
    max_items: int | None = None,
    max_total_length: int | None = None,
) -> str:
    """把检索结果拼成注入给大模型的法源清单。

    Args:
        articles: 检索到的法条列表
        max_items: 最多包含的条文数量，默认不限制
        max_total_length: 上下文总长度上限（字符数），超出时截断，默认不限制

    Returns:
        格式化后的法源清单文本，包含 11 个引用字段

    两条硬规则：
    1. 编号必须与最终回答里的引用编号一致（[1]、[2]…），否则模型会引错
    2. 条文原文原样注入，不做摘要、不改写 —— 一旦改写，引用就不可信
    """
    # 限制条文数量
    if max_items is not None and len(articles) > max_items:
        articles = articles[:max_items]

    # 每条法源单独成块，块之间用空行分开
    blocks: list[str] = []
    total_length = 0

    for position, article in enumerate(articles, start=1):
        # === 标题行：编号 + 法规名 + 条号/款号/项号 ===
        header_parts = [f"[{position}]", f"《{article.document_title}》"]

        # 条号
        if article.article_number:
            header_parts.append(f"第 {article.article_number} 条")

        # 款号
        if article.paragraph_number:
            header_parts.append(f"第 {article.paragraph_number} 款")

        # 项号
        if article.item_number:
            header_parts.append(f"第 {article.item_number} 项")

        # === 元数据行（引用所需字段）===
        metadata_lines = []

        # 文书类型
        if article.document_type:
            metadata_lines.append(f"文书类型：{article.document_type}")

        # 发布机关
        if article.issuing_authority:
            metadata_lines.append(f"发布机关：{article.issuing_authority}")

        # 法域
        if article.jurisdiction:
            metadata_lines.append(f"法域：{article.jurisdiction}")

        # 生效日期与失效日期
        date_info = []
        if article.effective_date:
            date_info.append(f"生效：{article.effective_date}")
        if article.expiration_date:
            date_info.append(f"失效：{article.expiration_date}")
        if date_info:
            metadata_lines.append("  ".join(date_info))

        # 是否现行有效（展示文案取自词表，避免与判定口径各写一套）
        if article.is_current is not None:
            status = EFFECTIVE if article.is_current else LAPSED
            metadata_lines.append(f"状态：{status}")

        # 来源 URL：唯一无条件追加的字段——引用必须可溯源，
        # 缺来源的条文不允许进入法源清单
        metadata_lines.append(f"来源：{article.source_url}")

        # === 组装完整块 ===
        block_lines = [
            " ".join(header_parts),
            *metadata_lines,
            f"条文原文：{article.content}",
        ]
        block_text = "\n".join(block_lines)

        # 检查总长度限制
        if max_total_length is not None:
            if total_length + len(block_text) > max_total_length:
                # 超出长度限制，停止添加更多条文
                # 直接 break 而不是截断本条：法条截半句比少一条更误导模型
                break
            # +2 预支块间分隔符 "\n\n" 的长度，累计口径与最终拼接一致
            total_length += len(block_text) + 2  # +2 for "\n\n"

        blocks.append(block_text)

    # 空行分隔各条法源
    return "\n\n".join(blocks)


# === 法律时效元数据转换（原 legal_metadata.py 并入）===


def date_to_timestamp(value: Any, unknown: int | None = None) -> int | None:
    """把数据库日期转为 UTC 时间戳；缺失生效日期可使用 0 哨兵。

    Args:
        value: 数据库取出的日期（datetime 或 date），可为 None
        unknown: value 为 None 时的返回值；传 0 可当作排序哨兵（最旧优先）

    Returns:
        UTC 秒级时间戳；value 为 None 时返回 unknown
    """
    if value is None:
        return unknown
    # naive datetime 视为 UTC：库里的日期无时区概念，统一按 UTC 处理
    # 才能保证与 Milvus 里存的数值型标量可比
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    else:
        # date（无时分秒）补齐为零点再转
        moment = datetime(value.year, value.month, value.day, tzinfo=UTC)
    return int(moment.timestamp())


def resolve_current_status(status: str | None, expiration_date: Any) -> bool | None:
    """按索引构建规则计算现行状态；缺少法规元数据时返回未知。

    Args:
        status: 法规效力状态（取值见 app/db/law_status.py），可为 None
        expiration_date: 失效日期（date/datetime），可为 None

    Returns:
        True=现行有效，False=已失效；两个入参都缺失时返回 None
        （未知——既不判死也不判活，避免把没数据的法规标成已失效）

    规则与索引构建时写入 Milvus 的口径一致，检索侧直接复用，
    防止"索引一个口径、展示另一个口径"。
    """
    if status is None and expiration_date is None:
        return None
    # 状态字段优先级最高：显式标注已废止/已失效的，日期再新也判失效。
    # 失效取值取自词表：历史上这里手抄中文元组，写进库的英文取值静默漏过该判断。
    # 注：不适用（案例材料）不在 EXPIRED_STATUSES 里，故判为"非失效"。
    expired_by_status = status in EXPIRED_STATUSES
    # 没写失效日期视为长期有效；有失效日期则与今天比较
    return (not expired_by_status) and (
        expiration_date is None or expiration_date > datetime.now(UTC).date()
    )
