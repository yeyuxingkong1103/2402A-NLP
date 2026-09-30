# -*- coding: utf-8 -*-
"""检索引擎注册与路由。

不再做关键词实体抽取或意图规则路由。统一 AI 计划决定是否启用 vector 和
graph，引擎分别消费计划中的语义查询与图谱查询。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.core.retrieval.base.retriever_base import BaseRetriever

logger = logging.getLogger(__name__)

ENGINE_WEIGHTS: dict[str, float] = {
    "vector": 1.0,
    "graph": 0.9,
}


def _probe_graph_connectivity(graph_retriever, timeout: float = 5.0) -> bool:
    """探测 Neo4j 连通性（带超时），返回是否可达。

    ``HybridGraphRetriever.is_available()`` 只判断 driver 是否构造成功，不验证
    连通；而 Neo4j 不可达时检索会阻塞一个连接超时（可能 20s+）。这里在启动时
    用 ``verify_connectivity`` 探测一次，失败则跳过图检索。
    """
    import concurrent.futures

    service = getattr(graph_retriever, "_service", None)
    verify = getattr(service, "verify_connectivity", None)
    if verify is None:
        return False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(verify)
    try:
        future.result(timeout=max(0.5, float(timeout)))
        return True
    except Exception:
        return False
    finally:
        # 不等待后台探测线程（Neo4j 不可达时 verify 可能仍在等待），避免阻塞启动。
        executor.shutdown(wait=False)


@dataclass
class RouteDecision:
    """路由决策：可用引擎实例、额外参数和固定融合权重。"""

    engines: list[BaseRetriever]
    params: dict[str, dict] = field(default_factory=dict)
    weights: Optional[dict[str, float]] = None
    reason: str = ""


class RetrievalRouter:
    """检索路由：返回当前可用的检索引擎。

    Parameters
    ----------
    retrievers : dict[str, BaseRetriever] | None
        name -> 引擎实例。None 时用空表（通过 ``register`` 注册）。
    """

    def __init__(self, retrievers: Optional[dict[str, BaseRetriever]] = None):
        self._retrievers: dict[str, BaseRetriever] = dict(retrievers or {})

    @classmethod
    def with_defaults(cls, include_vector: Optional[bool] = None):
        """创建核心默认路由。

        图检索直接使用 ``src.knowledge_graph.graph_retrieval``；向量引擎按配置决定是否注册。
        """
        from src import config
        router = cls()

        if config.ENABLE_GRAPH_RETRIEVAL:
            try:
                from src.knowledge_graph.hybrid_graph import HybridGraphRetriever

                graph = HybridGraphRetriever()
                # 启动时探测 Neo4j 连通性（带超时）。HybridGraphRetriever.is_available()
                # 只判断 driver 是否构造成功，不验证连通；若 Neo4j 不可达，直接跳过
                # 图检索，避免每次查询白等一个连接超时。
                if _probe_graph_connectivity(graph, timeout=config.NEO4J_CONNECTION_TIMEOUT):
                    router.register(graph)
                else:
                    graph.close()
                    logger.warning("Neo4j 不可达，跳过图检索引擎（避免查询阻塞）")
            except Exception:
                # 图模块依赖或 Neo4j 配置不可用时跳过，不阻断其他检索引擎。
                logger.warning("图检索引擎注册失败，跳过", exc_info=True)

        if include_vector is None:
            include_vector = config.ENABLE_VECTOR_RETRIEVAL
        if include_vector:
            from src.core.retrieval.engines.vector_store import VectorRetriever

            router.register(VectorRetriever())

        return router

    def close(self) -> None:
        """关闭所有支持关闭的引擎资源。"""
        for retriever in self._retrievers.values():
            close = getattr(retriever, "close", None)
            if callable(close):
                close()

    def register(self, retriever: BaseRetriever) -> None:
        """注册一个检索引擎。"""
        self._retrievers[retriever.name] = retriever

    def unregister(self, name: str) -> None:
        self._retrievers.pop(name, None)

    def get(self, name: str) -> Optional[BaseRetriever]:
        return self._retrievers.get(name)

    @property
    def engines(self) -> dict[str, BaseRetriever]:
        return dict(self._retrievers)

    def _available_engines(self, names: list[str]) -> list[BaseRetriever]:
        """按名字取引擎并按可用性过滤（保持优先级；异常引擎跳过）。"""
        engines: list[BaseRetriever] = []
        for name in names:
            retriever = self._retrievers.get(name)
            if retriever is None:
                continue
            try:
                if retriever.is_available():
                    engines.append(retriever)
            except Exception:
                continue
        return engines

    def route(self, *, use_vector: bool = True,
              use_graph: bool = True) -> RouteDecision:
        """根据 AI 计划启用检索引擎；不执行本地关键词或规则判断。"""
        requested = []
        if use_vector:
            requested.append("vector")
        if use_graph:
            requested.append("graph")
        engines = self._available_engines(requested)
        names = {engine.name for engine in engines}
        weights = {
            name: weight for name, weight in ENGINE_WEIGHTS.items() if name in names
        }
        return RouteDecision(
            engines=engines,
            weights=weights,
            reason="ai_query_plan",
        )

    def route_names(self, *, use_vector: bool = True,
                    use_graph: bool = True) -> list[str]:
        """返回当前可用引擎名（供调试/界面展示）。"""
        return [
            engine.name for engine in self.route(
                use_vector=use_vector, use_graph=use_graph
            ).engines
        ]
