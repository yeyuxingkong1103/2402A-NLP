"""检索链路的装配与依赖协议。

本模块合并了原先分散的四个装配小文件，集中放置：
  · 向量检索与重排的最小依赖协议（EmbeddingClientProtocol / RerankerProtocol）
  · 默认向量检索器装配（build_default_retriever）
  · 检索服务默认依赖装配（build_default_retrieval_service）
  · 检索前查询改写的安全调用边界（rewrite_query_safely）

单独成文件是为了守住"单文件 ≤ 300 行"的项目硬规则；
函数体逻辑与合并前完全一致，只做搬家。
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote_plus

from pymilvus import MilvusClient

from app.core.config import PROJECT_ROOT_DIRECTORY, settings
from app.db.engine import create_database_engine, create_session_factory
from app.db.redis_client import create_redis_client
from app.memory.short_term import ShortTermMemoryStore
from app.models.embedding import SiliconFlowEmbeddingClient
from app.models.reranker import build_reranker_from_settings
from app.retrieval.keyword_search import KeywordSearcher
from app.retrieval.query_rewrite import QueryRewriteResult, QueryRewriter

logger = logging.getLogger(__name__)


class EmbeddingClientProtocol(Protocol):
    """向量化客户端最小接口。

    Args/Returns（embed）:
        texts: 待向量化的文本列表
        Returns: 与输入等长的向量列表（每条为浮点数组）

    用 Protocol 而不是直接依赖 SiliconFlowEmbeddingClient：
    测试可注入 fake 实现而不必 mock 网络层，装配层也不关心具体厂商。
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class RerankerProtocol(Protocol):
    """重排客户端最小接口。

    Args/Returns（rerank）:
        query: 用户查询
        documents: 候选文档文本列表
        top_n: 只保留前 n 条；None 表示全部返回
        Returns: 重排后的结果列表（元素结构由实现方决定）

    同样是为了测试可替换——重排是纯外部依赖，接口收窄到一个方法。
    """

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[Any]: ...


def build_default_retriever():
    """连接真实向量模型、Milvus、MySQL 和重排服务。

    Returns:
        LegalRetriever：向量检索（embedding + Milvus）+ 关键词兜底（MySQL）
        + 重排的完整实例，生产链路用它

    延迟 import LegalRetriever：把重组件的导入推迟到真正装配时，
    避免本模块被轻量引用（如只取协议定义）时连带加载全部检索依赖。
    """
    from app.retrieval.vector_search import LegalRetriever

    # 用户名/密码做 URL 转义：口令里出现 @ 或 / 会直接把连接串撕碎
    database_url = (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:{quote_plus(settings.mysql_password)}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )
    embedding_client = SiliconFlowEmbeddingClient(
        api_url=settings.embedding_api_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        timeout=settings.embedding_timeout_seconds,
    )
    return LegalRetriever(
        embedding_client=embedding_client,
        milvus_client=MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
        ),
        collection_name=settings.milvus_collection_name,
        session_factory=create_session_factory(create_database_engine(database_url)),
        reranker=build_reranker_from_settings(),
    )


def build_query_expander_from_settings():
    """按配置装配同义术语扩写器；开关关掉时返回 None（关键词路行为不变）。

    Returns:
        SynonymExpander 实例；开关关闭或术语表不可用时返回 None
        （None 会被 KeywordSearcher 解释为"不做扩展"，而不是报错）

    失败策略：术语表只是召回增益，任何读取失败都不允许影响检索主链路。
    """
    if not settings.synonym_expansion_enabled:
        return None

    from app.retrieval.synonym_expansion import (
        SynonymExpander,
        SynonymTableError,
    )

    table_path = Path(settings.synonym_table_path)
    if not table_path.is_absolute():
        # 相对路径按项目根解析（术语表放 data/，与 backend/ 解耦）
        table_path = PROJECT_ROOT_DIRECTORY / table_path
    try:
        expander = SynonymExpander(table_path)
    except SynonymTableError as error:
        # 开关开着但表读不到：明确告警并退回不扩展，不让检索整体挂掉
        logger.warning("同义术语表不可用，已退回不扩展：%s", error)
        return None
    logger.info(
        "同义术语扩写已启用：表 %s（版本 %s，%d 组）",
        table_path,
        expander.version,
        len(expander.table.groups),
    )
    return expander


def build_default_retrieval_service():
    """连接真实向量、关键词检索和短期记忆。

    Returns:
        RetrievalService：检索主服务实例——向量路 + 关键词路（RRF 融合在
        service 内完成）+ 查询改写 + 短期记忆上下文，问答链路的检索层入口

    装配顺序有讲究：先建 session_factory（向量路/关键词路共用同一个
    连接池），再建依赖它的组件，避免两套池导致连接数翻倍。
    """
    from app.retrieval.service import RetrievalService

    # 与 build_default_retriever 相同的转义规则；不复用是因为两个装配
    # 点的连接生命周期独立（检索器与会话工厂各自持有 engine）
    database_url = (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:{quote_plus(settings.mysql_password)}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )
    session_factory = create_session_factory(create_database_engine(database_url))
    keyword_searcher = KeywordSearcher(
        session_factory=session_factory,
        query_expander=build_query_expander_from_settings(),
        expansion_weight=settings.synonym_expansion_weight,
    )
    # BM25 索引预构建：refresh 后关键词检索才能命中，冷启动做一次
    keyword_searcher.refresh()
    # 窗口 20 条：既给改写器足够上文，又不让 prompt 无限膨胀
    short_term_memory = ShortTermMemoryStore(
        create_redis_client(settings.redis_url),
        ttl_seconds=settings.session_ttl_seconds,
        max_messages=20,
    )
    return RetrievalService(
        vector_retriever=build_default_retriever(),
        keyword_searcher=keyword_searcher,
        session_factory=session_factory,
        query_rewriter=QueryRewriter(),
        short_term_memory=short_term_memory,
        reranker=build_reranker_from_settings(),
    )


def rewrite_query_safely(
    question: str,
    *,
    query_rewriter: QueryRewriter | None,
    short_term_memory: Any | None,
    user_id: str | None,
    session_id: str | None,
    request_id: str | None = None,
) -> QueryRewriteResult:
    """读取短期上下文并改写查询，任何失败均记录类型后回退原问题。

    Args:
        question: 用户原始问题
        query_rewriter: 改写器实例；None 表示改写功能未启用
        short_term_memory: 短期记忆存储；None 表示不上下文改写
        user_id / session_id: 定位历史消息；任一缺失即放弃改写
        request_id: 日志链路追踪 id，与主链路的日志串起来

    Returns:
        QueryRewriteResult：成功返回改写结果；任何环节不可用/出错
        都返回 changed=False 的原问题结果

    边界设计：改写是"锦上添花"环节，这里把 try 收敛到整个改写动作，
    任何异常（Redis 抖动、改写规则 bug）都只降级为"用原问题检索"。
    """
    # 统一的回退对象：四个前置条件任一缺失都走它
    unchanged = QueryRewriteResult(question, question, False, [])
    if not query_rewriter or not short_term_memory or not user_id or not session_id:
        return unchanged
    try:
        messages = short_term_memory.read_messages(user_id, session_id)
        result = query_rewriter.rewrite(question, messages)
        # 任务 5.4：改写生效时留痕（原问题 → 改写后 + 依据），
        # 供端到端验收观测"短期上下文已接通"
        if result.changed:
            logger.info(
                "查询改写生效：%s -> %s（%s）",
                result.original_query,
                result.rewritten_query,
                "、".join(result.reasons),
                extra={"request_id": request_id},
            )
        return result
    except Exception as error:
        # 只记异常类型不记详情：改写失败不需要告警升级，类型足够定位；
        # 也不把用户问题打进错误日志，避免噪音
        logger.warning(
            "查询改写失败，已回退原问题：%s",
            type(error).__name__,
            extra={"request_id": request_id},
        )
        return unchanged
