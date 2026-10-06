# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/rag_engine_v4.py —— 工单四 图文表融合 RAG 引擎（新增文件）

增量设计（不重写工单三）：
  - 文本/表格检索与 RRF/rerank 完全复用 RAGEngineV3 内部组件（TextRetrieverV3
    经由 TableRetriever 聚合，见 v3 __init__）；
  - 新增：ImageQueryRouter 路由 + ImageRetriever 图像检索；
  - 新增：build_multimodal_prompt 三源上下文 + LLM 生成。
输出：answer/references(text+table+image)/latency_ms/三路明细。
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from loguru import logger

from src.llm_client_v4 import RAG_V4_SYSTEM, build_multimodal_prompt, chat
from src.table_parser.query_router import route_query as route_query_v3
from src.table_parser.query_router import RouteResult

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"


class RAGEngineV4:
    """工单四：文本+表格+图像三路融合 RAG 引擎"""

    def __init__(self, v3_engine: Optional[Any] = None,
                 image_retriever: Optional[Any] = None,
                 top_k: int = 5, image_top_k: int = 4,
                 use_rerank: bool = True):
        # 工单四：复用工单三引擎（其内部已聚合 TextRetrieverV3 + TableRetriever）
        from src.rag_engine_v3 import RAGEngineV3
        self._v3 = v3_engine or RAGEngineV3(top_k=top_k, use_rerank=use_rerank)
        self.image_top_k = image_top_k
        self._image_retriever = image_retriever
        self._store = None                      # 工单四：图像库行数缓存（health 用）
        self._clipper = None                    # 工单四：CLIP 嵌入器单例（避免每问重载模型）

    # ------------------------------------------------------------------
    def _get_image_retriever(self):
        """工单四：懒加载 ImageRetriever（rag_images 空库/异常时优雅降级）"""
        if self._image_retriever is None:
            from src.image_parser.image_retriever import ImageRetriever
            self._image_retriever = ImageRetriever()
        return self._image_retriever

    # ------------------------------------------------------------------
    def retrieve_images(self, query: str, doc_ids: Optional[List[str]],
                        top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """工单四：图像三路召回（文本向量 + CLIP + 关键词重排，RRF 融合）"""
        ir = self._get_image_retriever()
        k = top_k or self.image_top_k
        doc_list = [doc_ids] if isinstance(doc_ids, str) else doc_ids
        try:
            hits_text = ir.retrieve(query, top_k=k, doc_ids=doc_list)     # bge-m3 通道
        except Exception as e:                  # 工单四：图像检索失败不阻塞主流程
            logger.warning(f"[rag_v4] 图像文本通道失败: {e}")
            hits_text = []
        # 工单四：CLIP 通道（模型缺失时 search_clip 自然返回空；单例避免重复加载）
        try:
            if self._clipper is None:
                from src.image_parser.image_embedding import ImageClipEmbedder
                self._clipper = ImageClipEmbedder()
            qvec = self._clipper.embed_query_text(query)
            hits_clip = (self._get_image_retriever()._get_store()
                         .search_clip(qvec, top_k=k, doc_ids=doc_list)
                         if qvec else [])
        except Exception as e:
            logger.debug(f"[rag_v4] CLIP 通道跳过: {e}")
            hits_clip = []

        # 工单四：双通道 RRF 融合（k=60，与工单三 RRF 常数一致）
        rrf: Dict[str, Dict[str, Any]] = {}
        for rank, h in enumerate(hits_text):
            e = rrf.setdefault(h["image_id"], {**h, "rrf": 0.0})
            e["rrf"] += 1.0 / (60 + rank + 1)
        for rank, h in enumerate(hits_clip):
            e = rrf.setdefault(h["image_id"], {**h, "rrf": 0.0})
            e["rrf"] += 1.0 / (60 + rank + 1)
            e["clip_hit"] = True
        fused = sorted(rrf.values(), key=lambda x: x["rrf"], reverse=True)[:k]
        return fused

    # ================= 工单四：模型预热 =================
    def warmup(self) -> None:
        """工单四（人工智能NLP-RAG-图像内容解析及检索优化）：串行预加载
        bge-m3 / bge-reranker / Chinese-CLIP 到 GPU。

        ask() 的图像通道与文本/表格通道在线程池内并行，若模型尚未加载，
        多线程会在锁外各自触发懒加载，曾导致同一进程加载 3 份 bge-m3，
        8GB 显存超卖、单次检索卡数分钟。服务启动/引擎初始化时串行预热一次，
        此后并行只做 forward，稳态检索 ≤1.3s。
        """
        from src.embedding import get_embedder
        from src.reranker import get_reranker
        t0 = time.perf_counter()
        get_embedder().encode(["预热：组织结构图 增长率"], show_progress_bar=False)
        get_reranker().available
        if self._clipper is None:
            from src.image_parser.image_embedding import ImageClipEmbedder
            self._clipper = ImageClipEmbedder()
        self._clipper.embed_query_text("预热：组织结构图")
        logger.info(f"[rag_v4] 模型预热完成，耗时 {time.perf_counter()-t0:.1f}s")

    # ================= 工单四：RAG 主入口 =================
    def ask(self, query: str, doc_id: Optional[str] = None,
            company: Optional[str] = None, use_image: bool = True,
            lang_override: Optional[str] = None,
            top_k: Optional[int] = None) -> Dict[str, Any]:
        """工单四：路由 → 三路检索 → 融合上下文 → LLM

        Returns: answer/references/retrieved_*/latency_ms/breakdown/route
        """
        # 工单四：perf_counter 单调时钟，规避 WSL2 墙钟跳变（人工智能NLP-RAG-图像内容解析及检索优化）
        t0 = time.perf_counter()
        k = top_k or self._v3.top_k

        # 1) 工单四：图像感知路由（image_first 时图像优先；hybrid 也带图像召回）
        from src.image_parser.image_query_router import route_query as route_image
        img_route = route_image(query)
        logger.info(f"[rag_v4] image_route={img_route['mode']} "
                    f"conf={img_route['confidence']:.2f} hits={img_route['hits']}")

        # 2)+3) 工单四：文本/表格主检索 与 图像检索并行（两路相互独立，
        # bge-m3/reranker 为 eval 单例、Milvus 调用为独立 gRPC，线程安全；
        # 串行改并行可省下一整个图像通道耗时，保障 ≤3s 响应指标）
        route3 = route_query_v3(query)
        need_images = use_image and (
            img_route["mode"] == "image_first" or img_route["hits"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_main = pool.submit(
                self._v3.table_retriever.retrieve,
                query, top_k=k, doc_id=doc_id, company=company, route=route3)
            fut_img = (pool.submit(self.retrieve_images, query, doc_id or company,
                                   top_k=self.image_top_k)
                       if need_images else None)
            retrieve_result = fut_main.result()
            image_hits: List[Dict[str, Any]] = []
            if fut_img is not None:
                try:
                    image_hits = fut_img.result()
                except Exception as e:          # 工单四：图像通道异常不阻塞主链路
                    logger.warning(f"[rag_v4] 图像检索并行任务失败: {e}")
                    image_hits = []
        fused = retrieve_result["fused"]
        text_chunks = [h for h in fused if h.get("source") == "text"]
        table_chunks = [h for h in fused if h.get("source") == "table"]
        retrieve_ms = retrieve_result["elapsed_ms"]

        if not (text_chunks or table_chunks or image_hits):
            return {"mode": "rag_v4", "query": query, "route": img_route,
                    "answer": "抱歉，未能在知识库（文本/表格/图像）中找到相关信息。",
                    "references": [], "retrieved_text_chunks": [],
                    "retrieved_tables": [], "retrieved_images": [],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "breakdown": {"retrieve_ms": retrieve_ms, "llm_ms": 0.0},
                    "token_usage": {}}

        # 4) 工单四：三源上下文组装 + LLM 生成
        prompt = build_multimodal_prompt(query, text_chunks, table_chunks,
                                         image_hits, self._v3.max_context_chars)
        t2 = time.perf_counter()
        try:
            llm_result = chat(messages=[{"role": "system", "content": RAG_V4_SYSTEM},
                                        {"role": "user", "content": prompt}],
                              temperature=0.2, max_tokens=700)
        except Exception as e:                  # 工单四：LLM 异常（超时等）重试一次
            logger.warning(f"[rag_v4] LLM 调用异常，重试: {e}")
            llm_result = chat(messages=[{"role": "system", "content": RAG_V4_SYSTEM},
                                        {"role": "user", "content": prompt}],
                              temperature=0.2, max_tokens=700)
        if not (llm_result.get("content") or "").strip():   # 工单四：空响应重试
            llm_result = chat(messages=[{"role": "system", "content": RAG_V4_SYSTEM},
                                        {"role": "user", "content": prompt}],
                              temperature=0.2, max_tokens=700)
        llm_ms = (time.perf_counter() - t2) * 1000

        # 5) 工单四：引用组装（text/table 沿用工单三格式 + image 新增）
        references = []
        for i, c in enumerate(text_chunks, 1):
            references.append({"type": "text", "ref_id": f"资料{i}",
                               "page": c.get("page"), "doc_id": c.get("doc_id"),
                               "score": c.get("rerank_score", c.get("rrf_score", 0)),
                               "preview": (c.get("content") or "")[:100]})
        for i, t in enumerate(table_chunks, 1):
            references.append({"type": "table", "ref_id": f"表{i}",
                               "page": t.get("page"), "doc_id": t.get("doc_id"),
                               "table_id": t.get("table_id"),
                               "score": t.get("rerank_score", t.get("rrf_score", 0)),
                               "preview": (t.get("content") or "")[:100]})
        for i, im in enumerate(image_hits, 1):
            references.append({
                "type": "image", "ref_id": f"图{i}", "doc_id": im.get("doc_id"),
                "image_id": im.get("image_id"), "page": im.get("page"),
                "path": im.get("path"), "caption": (im.get("caption") or "")[:100],
                "ocr_text": (im.get("ocr_text") or "")[:100],
                "vqa_text": (im.get("vqa_text") or "")[:100],
                "score": round(im.get("rrf", 0), 4)})

        return {
            "mode": "rag_v4", "query": query, "route": img_route,
            "answer": llm_result["content"],
            "references": references,
            "retrieved_text_chunks": text_chunks,
            "retrieved_tables": table_chunks,
            "retrieved_images": image_hits,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "breakdown": {"retrieve_ms": round(retrieve_ms, 1),
                          "llm_ms": round(llm_ms, 1)},
            "token_usage": llm_result.get("token_usage", {}),
        }

    # ------------------------------------------------------------------
    def ask_llm(self, query: str) -> Dict[str, Any]:
        """工单四：纯 LLM 模式（对照基线，复用工单三）"""
        return self._v3.ask_llm(query)

    # ------------------------------------------------------------------
    def ask_full(self, query: str, doc_id: Optional[str] = None,
                 use_image: bool = True, **kw) -> Dict[str, Any]:
        """工单四：RAG + 纯 LLM 双模式（对比实验用）"""
        return {"rag": self.ask(query, doc_id=doc_id, use_image=use_image, **kw),
                "pure_llm": self.ask_llm(query)}

    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """工单四：rag_images 行数（健康检查）"""
        try:
            if self._store is None:
                from src.image_parser.image_store import ImageStore
                self._store = ImageStore()
            return {"milvus_images_rows": self._store.count()}
        except Exception:
            return {"milvus_images_rows": 0}
