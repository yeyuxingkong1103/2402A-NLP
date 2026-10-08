# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/rag_engine_v6.py —— 工单六 可配置混合检索 RAG 引擎（新增文件）

增量设计（继承工单四 RAGEngineV4，不重写既有链路）：
  - 文本检索替换为 src.retrieval.hybrid_retriever_v6.HybridRetrieverV6，
    支持 vector / fulltext / hybrid 三模式、RRF/加权平均两融合算法、
    LLM/TF-IDF/用户反馈自适应三种重排器，全部可配置；
  - 表格、图像通道与多模态 prompt、LLM 生成完全复用工单四；
  - ask() 新增 retrieval_config 参数（dict 或 RetrievalConfig）。
输出在 v4 基础上增加 retrieval 字段（实际生效策略与两路命中数/耗时）。
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.llm_client_v4 import RAG_V4_SYSTEM, build_multimodal_prompt, chat
from src.perf_v13 import new_request, record, stage, with_request_id
from src.retrieval.retrieval_config import RetrievalConfig
from src.table_parser.query_router import route_query as route_query_v3

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"


class RAGEngineV6:
    """工单六：可配置混合检索 RAG 引擎（文本策略可配 + 表格/图像沿用 v4）"""

    def __init__(self, v4_engine: Optional[Any] = None,
                 default_config: Optional[RetrievalConfig] = None,
                 top_k: int = 5):
        from src.rag_engine_v4 import RAGEngineV4
        self._v4 = v4_engine or RAGEngineV4(top_k=top_k)
        self._v3 = self._v4._v3
        self.default_config = default_config or RetrievalConfig(top_k=top_k)
        self._hybrid_v6 = None

    # ------------------------------------------------------------------
    def _get_hybrid(self) -> Any:
        """工单六：懒加载统一混合检索器"""
        if self._hybrid_v6 is None:
            from src.retrieval.hybrid_retriever_v6 import HybridRetrieverV6
            self._hybrid_v6 = HybridRetrieverV6(self.default_config)
        return self._hybrid_v6

    def warmup(self) -> None:
        """工单六：预热（bge/reranker/CLIP + 全文倒排索引 + 一条真实查询）"""
        self._v4.warmup()
        try:
            self._get_hybrid().fulltext_recall(
                "公司主营业务", None, 8, self.default_config)
            logger.info("[rag_v6] 全文倒排索引预热完成")
        except Exception as e:
            logger.warning(f"[rag_v6] 全文索引预热跳过: {e}")

    # ================= 工单六：RAG 主入口 =================
    def ask(self, query: str, doc_id: Optional[str] = None,
            company: Optional[str] = None, use_image: Optional[bool] = None,
            lang_override: Optional[str] = None,
            top_k: Optional[int] = None,
            retrieval_config: Union[RetrievalConfig, Dict[str, Any], None] = None
            ) -> Dict[str, Any]:
        """工单六：可配置混合检索问答

        retrieval_config: 覆盖默认策略（mode/fusion/reranker/weights/match/...）
        """
        t0 = time.perf_counter()
        # 工单十三：本次请求 ID（用于分阶段结构化日志追踪）
        rid = new_request(query)
        stages_ms: Dict[str, float] = {}
        cfg = self._merge_config(retrieval_config, top_k)
        k = cfg.top_k
        use_img = cfg.use_image if use_image is None else use_image

        # 1) 工单六：图像感知路由（沿用 v4）｜工单十三：阶段① 查询处理与增强
        from src.image_parser.image_query_router import route_query as route_image
        with stage("query_routing", mode=cfg.mode) as _s:
            img_route = route_image(query)
            # 2) 工单六：表格路由（沿用 v3）
            route3 = route_query_v3(query)
            need_table = cfg.use_table and route3.route in ("table_only", "hybrid")
            need_images = use_img and (
                img_route["mode"] == "image_first" or img_route["hits"])
        stages_ms["query_routing"] = round(_s.ms, 1)

        # 3) 工单六：文本(v6 可配置) / 表格(v3) / 图像(v4) 三路并行
        # 工单十三：阶段② 检索——分别记录各通道墙钟耗时（并行，取最大为瓶颈）
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_text = pool.submit(with_request_id(rid, self._get_hybrid().retrieve),
                                   query, k, doc_id, cfg)
            fut_table = (pool.submit(
                with_request_id(rid, self._v3.table_retriever.search_tables),
                query, top_k=k, doc_id=doc_id, company=company)
                if need_table else None)
            fut_img = (pool.submit(with_request_id(rid, self._v4.retrieve_images),
                                   query, doc_id or company, top_k=self._v4.image_top_k)
                       if need_images else None)
            with stage("retrieval_text") as _s:
                text_result = fut_text.result()
            stages_ms["retrieval_text"] = round(_s.ms, 1)
            table_hits: List[Dict[str, Any]] = []
            if fut_table is not None:
                with stage("retrieval_table") as _s:
                    try:
                        table_hits = fut_table.result()
                    except Exception as e:
                        logger.warning(f"[rag_v6] 表格检索失败: {e}")
                stages_ms["retrieval_table"] = round(_s.ms, 1)
            image_hits: List[Dict[str, Any]] = []
            if fut_img is not None:
                with stage("retrieval_image") as _s:
                    try:
                        image_hits = fut_img.result()
                    except Exception as e:
                        logger.warning(f"[rag_v6] 图像检索失败: {e}")
                stages_ms["retrieval_image"] = round(_s.ms, 1)

        text_chunks = text_result["results"]
        # 工单六：文本+表格跨源合并（有 rerank_score 按归一化分合并，否则 RRF 兜底）
        with stage("merge_sources") as _s:
            fused = self._merge_text_table(text_chunks, table_hits, k)
        stages_ms["merge_sources"] = round(_s.ms, 1)
        text_chunks = [h for h in fused if h.get("source") == "text"]
        table_chunks = [h for h in fused if h.get("source") == "table"]
        retrieve_ms = (time.perf_counter() - t0) * 1000

        if not (text_chunks or table_chunks or image_hits):
            return {"mode": "rag_v6", "query": query,
                    "answer": "抱歉，未能在知识库（文本/表格/图像）中找到相关信息。",
                    "references": [], "retrieved_text_chunks": [],
                    "retrieved_tables": [], "retrieved_images": [],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "request_id": rid, "stages_ms": stages_ms,
                    "breakdown": {"retrieve_ms": round(retrieve_ms, 1),
                                  "llm_ms": 0.0},
                    "retrieval": self._retrieval_meta(text_result),
                    "token_usage": {}}

        # 4) 工单六：三源上下文 + LLM（复用 v4）
        # 工单十三：阶段③ 上下文组装与提示工程
        with stage("context_assembly", ctx_chars=sum(len(c.get("content") or "")
                                                     for c in text_chunks)) as _s:
            prompt = build_multimodal_prompt(query, text_chunks, table_chunks,
                                             image_hits, self._v3.max_context_chars)
        stages_ms["context_assembly"] = round(_s.ms, 1)
        # 工单十三：阶段④ LLM 生成
        t2 = time.perf_counter()
        try:
            llm_result = chat(
                messages=[{"role": "system", "content": RAG_V4_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=700)
        except Exception as e:
            logger.warning(f"[rag_v6] LLM 调用异常，重试: {e}")
            llm_result = chat(
                messages=[{"role": "system", "content": RAG_V4_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=700)
        if not (llm_result.get("content") or "").strip():
            llm_result = chat(
                messages=[{"role": "system", "content": RAG_V4_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=700)
        llm_ms = (time.perf_counter() - t2) * 1000
        stages_ms["llm_generation"] = round(llm_ms, 1)
        record("llm_generation", llm_ms,
               prompt_chars=len(prompt),
               output_tokens=(llm_result.get("token_usage") or {}).get("completion_tokens", 0))

        # 5) 工单六：引用组装（复用 v4 格式）｜工单十三：阶段⑤ 后处理与响应格式化
        with stage("post_processing", refs=len(text_chunks) + len(table_chunks)
                   + len(image_hits)) as _s:
            references = []
            for i, c in enumerate(text_chunks, 1):
                references.append({"type": "text", "ref_id": f"资料{i}",
                                   "page": c.get("page"), "doc_id": c.get("doc_id"),
                                   "score": c.get("rerank_score", c.get("score", 0)),
                                   "preview": (c.get("content") or "")[:100]})
            for i, tb in enumerate(table_chunks, 1):
                references.append({"type": "table", "ref_id": f"表{i}",
                                   "page": tb.get("page"), "doc_id": tb.get("doc_id"),
                                   "table_id": tb.get("table_id"),
                                   "score": tb.get("rerank_score", tb.get("score", 0)),
                                   "preview": (tb.get("content") or "")[:100]})
            for i, im in enumerate(image_hits, 1):
                references.append({
                    "type": "image", "ref_id": f"图{i}", "doc_id": im.get("doc_id"),
                    "image_id": im.get("image_id"), "page": im.get("page"),
                    "path": im.get("path"), "caption": (im.get("caption") or "")[:100],
                    "ocr_text": (im.get("ocr_text") or "")[:100],
                    "vqa_text": (im.get("vqa_text") or "")[:100],
                    "score": round(im.get("rrf", 0), 4)})
        stages_ms["post_processing"] = round(_s.ms, 1)

        return {
            "mode": "rag_v6", "query": query,
            "route": img_route, "answer": llm_result["content"],
            "references": references,
            "retrieved_text_chunks": text_chunks,
            "retrieved_tables": table_chunks,
            "retrieved_images": image_hits,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "request_id": rid, "stages_ms": stages_ms,
            "breakdown": {"retrieve_ms": round(retrieve_ms, 1),
                          "llm_ms": round(llm_ms, 1)},
            "retrieval": self._retrieval_meta(text_result),
            "token_usage": llm_result.get("token_usage", {}),
        }

    # ------------------------------------------------------------------
    def _merge_config(self,
                      override: Union[RetrievalConfig, Dict[str, Any], None],
                      top_k: Optional[int]) -> RetrievalConfig:
        """工单六：合并默认配置与请求级覆盖"""
        if isinstance(override, RetrievalConfig):
            cfg = override
        else:
            base = self.default_config.to_dict()
            if override:
                base.update({k: v for k, v in override.items()
                             if v is not None and k in base})
            cfg = RetrievalConfig(**base)
        if top_k:
            cfg.top_k = top_k
        return cfg

    @staticmethod
    def _retrieval_meta(text_result: Dict[str, Any]) -> Dict[str, Any]:
        return {k: text_result.get(k) for k in
                ("mode", "fusion", "reranker", "match",
                 "vector_weight", "fulltext_weight",
                 "vector_hits", "fulltext_hits", "elapsed_ms")}

    @staticmethod
    def _merge_text_table(text_hits: List[Dict[str, Any]],
                          table_hits: List[Dict[str, Any]],
                          top_k: int) -> List[Dict[str, Any]]:
        """工单六：文本/表格跨源合并（复用 v4 思路：归一化 rerank_score 同尺度可比）"""
        pool = list(text_hits) + list(table_hits)
        if not pool:
            return []
        for h in pool:
            h.setdefault("rerank_score", h.get("rrf_score", h.get("score", 0.0)))
        return sorted(pool, key=lambda x: x["rerank_score"],
                      reverse=True)[: max(top_k, 8)]

    # ------------------------------------------------------------------
    def ask_llm(self, query: str) -> Dict[str, Any]:
        """工单六：纯 LLM 对照基线（沿用 v4）"""
        return self._v4.ask_llm(query)

    def health(self) -> Dict[str, Any]:
        """工单六：健康检查（全文索引规模 + 图像库）"""
        stats = self._v4.health()
        try:
            stats["fulltext_docs"] = self._get_hybrid()._get_fulltext().size
        except Exception:
            stats["fulltext_docs"] = 0
        return stats
