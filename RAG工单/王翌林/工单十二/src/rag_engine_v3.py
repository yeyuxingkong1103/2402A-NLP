# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/rag_engine_v3.py —— 工单三 RAG 引擎（表格+文本融合）

在工单二 RAGEngine 基础上扩展：
  1. 输入：问题 + doc_id（可选，多文档隔离）
  2. 路由：query_router（text_only / table_only / hybrid）
  3. 检索：table_retriever（rag_tables）+ 工单二 text_retriever（rag_chunks）
  4. 重排序：bge-reranker（复用工单二 Reranker）
  5. 上下文组装：文本 chunk + 表格 table_text + 表格结构化 JSON 摘要
  6. LLM：DeepSeek-v4-flash（复用工单二，禁用思维链降延迟）
  7. 输出：answer / references / latency_ms / retrieved_text_chunks / retrieved_tables

两种模式：
  - RAG 模式：路由 → 检索 → 重排 → 上下文 → LLM
  - 纯 LLM 模式：跳过检索，直接问 LLM（对照基线）
"""
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from src.llm_client_v3 import (
    chat, build_table_aware_prompt, RAG_V3_SYSTEM,
)
from src.table_parser.query_router import route_query, RouteResult
from src.table_parser.table_retriever import TableRetriever

# 工单三：纯 LLM 模式 system prompt
PURE_LLM_SYSTEM = "你是一名专业的投资分析师。"


class RAGEngineV3:
    """工单三：表格感知 RAG 引擎

    依赖：
      - TableRetriever（rag_tables + 工单二 Retriever + Reranker）
      - LLM（DeepSeek-v4-flash）
    """

    def __init__(
        self,
        table_retriever: Optional[TableRetriever] = None,
        top_k: int = 5,
        text_top_k: int = 8,
        table_top_k: int = 8,
        max_context_chars: int = 8000,
        use_rerank: bool = True,
    ):
        self.top_k = top_k
        self.text_top_k = text_top_k
        self.table_top_k = table_top_k
        self.max_context_chars = max_context_chars
        # 工单三：懒加载 TextRetrieverV3（多文档文本检索）
        text_retriever = self._load_text_retriever(use_rerank=use_rerank)
        # 工单三：懒加载 TableRetriever（含 text_retriever）
        self.table_retriever = table_retriever or TableRetriever(
            use_rerank=use_rerank,
            table_top_k=table_top_k,
            text_top_k=text_top_k,
            final_top_k=top_k,
            text_retriever=text_retriever,
        )

    @staticmethod
    def _load_text_retriever(use_rerank: bool = True):
        """工单三：懒加载 TextRetrieverV3（多文档文本检索）

        从 Milvus rag_chunks collection 检索，支持 doc_id 过滤。
        需先运行 scripts/ingest_text_v3.py 入库文本 chunk。
        """
        try:
            from src.text_retriever_v3 import TextRetrieverV3
            retriever = TextRetrieverV3(use_rerank=use_rerank)
            logger.info("[rag_engine_v3] TextRetrieverV3 已加载（多文档）")
            return retriever
        except Exception as e:
            logger.warning(f"[rag_engine_v3] TextRetrieverV3 加载失败: {e}")
            return None

    # ================= RAG 模式 =================
    def ask_rag(
        self,
        query: str,
        doc_id: Optional[str] = None,
        company: Optional[str] = None,
        top_k: Optional[int] = None,
        route: Optional[RouteResult] = None,
    ) -> Dict[str, Any]:
        """工单三：RAG 模式主入口

        流程：路由 → 检索 → RRF 融合 → reranker → 上下文组装 → LLM

        Returns:
            {
              "mode": "rag_v3",
              "query": str,
              "route": {...},
              "answer": str,
              "references": [...],     # 含 text + table 引用
              "retrieved_text_chunks": [...],
              "retrieved_tables": [...],
              "latency_ms": float,
              "breakdown": {retrieve_ms, llm_ms},
              "token_usage": {...},
            }
        """
        t0 = time.perf_counter()  # 工单四：单调时钟，规避 WSL2 墙钟跳变
        k = top_k or self.top_k

        # 1) 路由
        if route is None:
            route = route_query(query)
        logger.info(f"[rag_engine_v3] route={route.route} conf={route.confidence:.2f}")

        # 2) 检索（路由驱动）
        retrieve_result = self.table_retriever.retrieve(
            query, top_k=k, doc_id=doc_id, company=company, route=route,
        )
        retrieve_ms = retrieve_result["elapsed_ms"]
        fused = retrieve_result["fused"]

        if not fused:
            return {
                "mode": "rag_v3", "query": query,
                "route": _route_to_dict(route),
                "answer": "抱歉，未能在知识库中找到相关信息。",
                "references": [],
                "retrieved_text_chunks": [],
                "retrieved_tables": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "breakdown": {"retrieve_ms": retrieve_ms, "llm_ms": 0.0},
                "token_usage": {},
            }

        # 3) 拆分 text / table 候选（fused 已含 source 字段）
        text_chunks = [h for h in fused if h.get("source") == "text"]
        table_chunks = [h for h in fused if h.get("source") == "table"]

        # 4) 上下文组装
        prompt = build_table_aware_prompt(
            query, text_chunks, table_chunks,
            max_chars=self.max_context_chars,
        )

        # 5) LLM 生成
        t2 = time.perf_counter()  # 工单四：单调时钟，规避 WSL2 墙钟跳变
        llm_result = chat(
            messages=[{"role": "system", "content": RAG_V3_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=600,
        )
        # 工单三容错：LLM 偶发空响应重试
        if not (llm_result.get("content") or "").strip():
            logger.warning("[rag_engine_v3] LLM 空响应，重试一次")
            llm_result = chat(
                messages=[{"role": "system", "content": RAG_V3_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=600,
            )
        llm_ms = (time.perf_counter() - t2) * 1000

        # 6) 引用组装（text + table 分开标号）
        references = []
        for i, c in enumerate(text_chunks, 1):
            references.append({
                "type": "text", "ref_id": f"资料{i}",
                "page": c.get("page"), "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "score": c.get("rerank_score", c.get("rrf_score", 0)),
                "preview": (c.get("content") or "")[:100],
            })
        for i, t in enumerate(table_chunks, 1):
            references.append({
                "type": "table", "ref_id": f"表{i}",
                "page": t.get("page"), "doc_id": t.get("doc_id"),
                "table_id": t.get("table_id"),
                "score": t.get("rerank_score", t.get("rrf_score", 0)),
                "preview": (t.get("content") or t.get("table_text") or "")[:100],
            })

        return {
            "mode": "rag_v3", "query": query,
            "route": _route_to_dict(route),
            "answer": llm_result["content"],
            "references": references,
            "retrieved_text_chunks": text_chunks,
            "retrieved_tables": table_chunks,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "breakdown": {
                "retrieve_ms": round(retrieve_ms, 1),
                "llm_ms": round(llm_ms, 1),
            },
            "token_usage": llm_result.get("token_usage", {}),
        }

    # ================= 纯 LLM 模式 =================
    def ask_llm(self, query: str) -> Dict[str, Any]:
        """工单三：纯 LLM 模式（不检索，对照基线）"""
        t0 = time.perf_counter()  # 工单四：单调时钟，规避 WSL2 墙钟跳变
        llm_result = chat(
            messages=[{"role": "system", "content": PURE_LLM_SYSTEM},
                      {"role": "user", "content": query}],
            temperature=0.3, max_tokens=600,
        )
        return {
            "mode": "pure_llm", "query": query,
            "answer": llm_result["content"],
            "references": [],
            "retrieved_text_chunks": [],
            "retrieved_tables": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "token_usage": llm_result.get("token_usage", {}),
        }

    # ================= 便捷：两种模式对比 =================
    def ask_both(
        self, query: str, doc_id: Optional[str] = None, **kwargs,
    ) -> Dict[str, Any]:
        """工单三：同时跑 RAG + 纯 LLM，便于对比"""
        rag = self.ask_rag(query, doc_id=doc_id, **kwargs)
        llm = self.ask_llm(query)
        return {"rag": rag, "pure_llm": llm}


def _route_to_dict(route: RouteResult) -> Dict[str, Any]:
    """工单三：RouteResult → dict（便于 JSON 序列化）"""
    return {
        "route": route.route,
        "confidence": route.confidence,
        "matched_keywords": route.matched_keywords,
        "reason": route.reason,
    }


# ================= CLI =================
if __name__ == "__main__":  # pragma: no cover
    import argparse, json
    from dotenv import load_dotenv
    load_dotenv()

    p = argparse.ArgumentParser(
        description="工单三 RAG 引擎 CLI（人工智能NLP-RAG-PDF文档的表格解析及检索优化）"
    )
    p.add_argument("--query", "-q", required=True)
    p.add_argument("--doc-id", default=None, help="按文档过滤（招股说明书1/招股说明书2）")
    p.add_argument("--company", default=None)
    p.add_argument("--mode", choices=["rag", "llm", "both"], default="rag")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    engine = RAGEngineV3(top_k=args.top_k)
    if args.mode in ("rag", "both"):
        print("\n" + "=" * 60 + "\n RAG 模式（表格+文本融合）\n" + "=" * 60)
        r = engine.ask_rag(args.query, doc_id=args.doc_id, company=args.company)
        print(f"路由: {r['route']['route']} (conf={r['route']['confidence']:.2f})")
        print(f"延迟: 总 {r['latency_ms']:.0f}ms  "
              f"检索 {r['breakdown']['retrieve_ms']:.0f}ms  "
              f"LLM {r['breakdown']['llm_ms']:.0f}ms")
        print(f"\n答案:\n{r['answer']}")
        print(f"\n引用 ({len(r['references'])} 条):")
        for ref in r["references"]:
            print(f"  [{ref['ref_id']}] type={ref['type']} "
                  f"page={ref.get('page')} doc={ref.get('doc_id')} "
                  f"score={ref.get('score', 0):.4f}  "
                  f"{ref.get('preview', '')[:60]}")
        print(f"\n文本候选: {len(r['retrieved_text_chunks'])} 条, "
              f"表格候选: {len(r['retrieved_tables'])} 条")
    if args.mode in ("llm", "both"):
        print("\n" + "=" * 60 + "\n 纯 LLM 模式\n" + "=" * 60)
        r2 = engine.ask_llm(args.query)
        print(f"延迟: {r2['latency_ms']:.0f}ms\n\n答案:\n{r2['answer']}")
