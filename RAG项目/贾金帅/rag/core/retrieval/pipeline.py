# -*- coding: utf-8 -*-
"""检索流水线。

完整链路：AI 语义规划 -> vector/graph 并行召回 -> RRF 融合排序 -> 输出 Top-K。
返回最终结果 + 各引擎得分明细（供界面 debug 与生成器使用）。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 各引擎单次检索超时（秒）：避免远程 embedding / Neo4j 不可达时整条查询被拖死。
# 用 os.getenv 兜底，不新增 config.py 配置项。
_ENGINE_TIMEOUTS: dict[str, float] = {
    "graph": float(os.getenv("RETRIEVAL_GRAPH_TIMEOUT", "6.0")),
    "vector": float(os.getenv("RETRIEVAL_VECTOR_TIMEOUT", "10.0")),
}
_DEFAULT_RETRIEVE_TIMEOUT = 10.0

from src.core.fusion.result_fusion import fusion
from src.core.retrieval.base.retriever_base import RetrievalResult
from src.core.retrieval.query_planner import AIQueryPlan, AIQueryPlanner
from src.core.retrieval.retrieval_log import new_retrieval_id
from src.core.retrieval.router import RetrievalRouter


@dataclass
class RetrievalOutput:
    """检索流水线输出。"""

    query: str
    query_plan: AIQueryPlan | None = None
    fused_results: list[RetrievalResult] = field(default_factory=list)
    engine_results: dict[str, list[RetrievalResult]] = field(default_factory=dict)
    used_engines: list[str] = field(default_factory=list)
    retrieval_id: str = ""
    web_search_requested: bool = False
    web_search_error: str = ""

    def to_context_text(self, max_chars: int = 2000) -> str:
        """把融合结果拼成 LLM 上下文文本（供生成器使用）。

        输出完整关键字段（含剂量/禁忌/相互作用），支撑生成器的
        MR002（剂量）/ MR003（相互作用）业务规则。
        ``max_chars`` 默认 2000，与生成器 ``truncate_context(2000 token)`` 对齐。
        """
        from src.core.generator.context import format_retrieval_context

        return format_retrieval_context(self.fused_results, max_chars=max_chars)


class RetrievalPipeline:
    """检索流水线：路由 -> 多路召回 -> 融合。

    Parameters
    ----------
    router : RetrievalRouter
        已注册引擎的路由器。
    strategy : str
        融合策略：rrf（默认）或 weighted。
    reranker : Reranker | None
        交叉编码重排器；None 时惰性创建（按 config.RERANKER_ENABLED），
        不可用时自动跳过重排（兜底）。
    query_planner : AIQueryPlanner | None
        统一 AI 查询规划器；None 时使用共享模型惰性创建。
    """

    def __init__(self, router: Optional[RetrievalRouter] = None, strategy: Optional[str] = None,
                 reranker=None, query_planner=None):
        self.router = router or RetrievalRouter.with_defaults()
        from src import config

        self.strategy = strategy or config.FUSION_STRATEGY
        self.reranker = reranker
        self.query_planner = query_planner

    def close(self) -> None:
        """释放检索引擎连接。"""
        close = getattr(self.router, "close", None)
        if callable(close):
            close()

    def generate_answer(self, query: str, top_k: Optional[int] = None,
                        llm_client=None, images=None, history: str = "", **kwargs) -> dict:
        """执行 AI 查询规划、图/向量检索、融合并生成结构化答案。

        ``history`` 为多轮对话历史（可选），透传给生成器用于代指消解。
        """
        output = self.search(query, top_k=top_k, images=images, **kwargs)
        if llm_client is None:
            from src.core.generator.service import RAGGenerator

            llm_client = RAGGenerator()
        answer = llm_client.generate_answer(
            query=output.query,
            context=output.to_context_text(),
            intent=output.query_plan.intent if output.query_plan else "general_medical",
            images=images,
            history=history,
        )
        return {
            "query": output.query,
            "retrieval": output,
            "answer": answer,
        }

    async def generate_answer_async(self, query: str, top_k: Optional[int] = None,
                                    llm_client=None, images=None, history: str = "",
                                    **kwargs) -> dict:
        """异步执行 AI 查询规划、检索、融合和答案生成。

        ``history`` 为多轮对话历史（可选），透传给生成器用于代指消解。
        """
        output = await self.search_async(
            query, top_k=top_k, images=images, **kwargs)
        if llm_client is None:
            from src.core.generator.service import RAGGenerator

            llm_client = RAGGenerator()
        answer = await llm_client.generate_answer_async(
            query=output.query,
            context=output.to_context_text(),
            intent=output.query_plan.intent if output.query_plan else "general_medical",
            images=images,
            history=history,
        )
        return {
            "query": output.query,
            "retrieval": output,
            "answer": answer,
        }

    def _get_reranker(self):
        """惰性创建重排器；未启用/不可用时返回 None（流水线跳过重排）。

        仅在 config.RERANKER_ENABLED 开启时才实例化，避免禁用时仍触发
        模型导入/加载。
        """
        from src import config as _config

        if not _config.RERANKER_ENABLED:
            return None
        if self.reranker is None:
            try:
                from src.core.fusion.reranker import Reranker

                self.reranker = Reranker()
            except Exception:
                self.reranker = None
        if self.reranker is not None and not self.reranker.is_available():
            return None
        return self.reranker

    def _get_query_planner(self) -> AIQueryPlanner:
        """惰性创建统一 AI 查询规划器。"""
        if self.query_planner is None:
            self.query_planner = AIQueryPlanner()
        return self.query_planner

    def search(self, query: str, top_k: Optional[int] = None,
               **kwargs) -> RetrievalOutput:
        """执行一次完整检索（同步入口，内部用 asyncio.run 调度）。

        Parameters
        ----------
        query : str
            用户查询（原始文本）。
        top_k : int | None
            融合结果上限。
        """
        import asyncio

        return asyncio.run(self._run_search_async(query, top_k, **kwargs))

    def _fuse(self, engine_results: dict, top_k: Optional[int],
              weights: Optional[dict] = None) -> list[RetrievalResult]:
        """融合各引擎结果，不做关键词实体覆盖或意图增强。"""
        return fusion(
            engine_results,
            strategy=self.strategy,
            top_k=top_k,
            weights=weights,
            source_weights=weights,
        )

    async def _post_process_async(self, fused: list[RetrievalResult],
                                  search_query: str,
                                  top_k: Optional[int]) -> list[RetrievalResult]:
        """重排兜底（异步）：rerank 为同步 CPU 推理，放入线程池避免
        阻塞事件循环（高并发下不卡其他请求）。"""
        import asyncio

        try:
            reranker = self._get_reranker()
            if reranker is not None and fused:
                return await asyncio.to_thread(
                    reranker.rerank, search_query, fused)
        except Exception:
            pass  # 重排失败不阻断链路
        return fused

    async def _run_engines_async(self, decision, plan: AIQueryPlan,
                                 top_k: Optional[int],
                                 kwargs: dict) -> dict[str, list[RetrievalResult]]:
        """异步并发执行多路检索的公共实现（供 search/search_async 复用）。

        向量库使用语义扩展后的 vector_query；图数据库使用 graph_query 和同一次
        AI 生成的图计划，避免 Neo4j 再次调用模型分析问题。
        """
        import asyncio

        async def _retrieve_one(engine) -> tuple[str, list[RetrievalResult]]:
            timeout = _ENGINE_TIMEOUTS.get(engine.name, _DEFAULT_RETRIEVE_TIMEOUT)
            try:
                engine_kwargs = dict(kwargs)
                engine_kwargs.update(decision.params.get(engine.name, {}))
                engine_query = (
                    plan.graph_query if engine.name == "graph" else plan.vector_query
                )
                if engine.name == "graph":
                    engine_kwargs["graph_plan"] = plan.graph_plan(top_k or 10)
                # 引擎为同步实现，放入线程池避免阻塞事件循环
                coro = asyncio.to_thread(
                    engine.retrieve, engine_query, top_k, **engine_kwargs)
                results = await asyncio.wait_for(coro, timeout=timeout)
                return engine.name, list(results or [])
            except asyncio.TimeoutError:
                logger.warning("引擎 %s 检索超时（%.1fs），跳过", engine.name, timeout)
                return engine.name, []
            except Exception as exc:
                logger.warning("引擎 %s 检索失败: %s", engine.name, exc)
                return engine.name, []

        tasks = [_retrieve_one(e) for e in decision.engines]
        pairs = await asyncio.gather(*tasks)
        return {name: results for name, results in pairs}

    async def _web_search_async(self, query: str,
                                top_k: Optional[int]) -> tuple[dict[str, list[RetrievalResult]], str]:
        """按请求执行一次 Tavily 联网搜索，返回结果和错误信息。"""
        import asyncio
        from src import config
        from src.web_search import WebSearchService

        try:
            service = WebSearchService()
            if not service.is_available():
                return {}, "Tavily 联网搜索不可用"
            results = await asyncio.to_thread(
                service.search, query, top_k or config.WEB_SEARCH_TOP_K)
            return ({"web": results} if results else {}), service.last_error
        except Exception as exc:
            logger.warning("Tavily 联网搜索执行失败：%s", exc)
            return {}, f"{type(exc).__name__}: {exc}"

    async def _run_search_async(self, query: str,
                                top_k: Optional[int] = None,
                                **kwargs) -> RetrievalOutput:
        """检索核心逻辑：sync/async 共用（sync 用 asyncio.run 包一层） 。"""
        q = (query or "").strip()
        web_search_enabled = bool(kwargs.pop("web_search", False))
        kwargs.pop("images", None)
        retrieval_id = str(kwargs.setdefault("retrieval_id", new_retrieval_id()))
        import asyncio

        plan = await asyncio.to_thread(self._get_query_planner().plan, q)
        decision = self.router.route(
            use_vector=plan.use_vector,
            use_graph=plan.use_graph,
        )

        web_search_error = ""
        if web_search_enabled:
            local_results, web_payload = await asyncio.gather(
                self._run_engines_async(decision, plan, top_k, kwargs),
                self._web_search_async(plan.vector_query or q, top_k),
            )
            engine_results = local_results
            web_results, web_search_error = web_payload
            if web_results:
                engine_results = {**engine_results, **web_results}
        else:
            engine_results = await self._run_engines_async(
                decision, plan, top_k, kwargs)

        fused = self._fuse(engine_results, top_k, weights=decision.weights)
        fused = await self._post_process_async(fused, plan.vector_query, top_k)
        return RetrievalOutput(
            query=q,
            query_plan=plan,
            fused_results=fused,
            engine_results=engine_results,
            used_engines=list(engine_results.keys()),
            retrieval_id=retrieval_id,
            web_search_requested=web_search_enabled,
            web_search_error=web_search_error,
        )

    async def search_async(self, query: str,
                           top_k: Optional[int] = None,
                           **kwargs) -> RetrievalOutput:
        """异步版本：供 asyncio 应用（如 FastAPI）直接 await，避免嵌套事件循环。"""
        return await self._run_search_async(query, top_k, **kwargs)
