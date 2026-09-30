# -*- coding: utf-8 -*-
"""Neo4j retrieval for the disease-treatment knowledge graph.

公开类名 ``DrugGraphRetriever`` 保持不变（app.py / 混合检索路由依赖它）。
本文件只保留：常量 + ``DrugGraphRetriever`` + CLI 入口。

配套部分已拆分：
  - ``cypher_queries.py``：预置参数化 Cypher 模板
  - ``graph_plan.py``：类型契约 + plan 归一化
  - ``hybrid_graph.py``：HybridGraphRetriever（上层适配器）
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from neo4j import GraphDatabase

try:
    from .data_to_neo4j import DEFAULT_PASSWORD, DEFAULT_URI, DEFAULT_USERNAME
    from .neo4j_prompt import ANALYSIS_PROMPT, ANSWER_PROMPT
except ImportError:
    from data_to_neo4j import DEFAULT_PASSWORD, DEFAULT_URI, DEFAULT_USERNAME
    from neo4j_prompt import ANALYSIS_PROMPT, ANSWER_PROMPT

from src.core.retrieval.base.retriever_base import BaseRetriever, RetrievalResult
from src.core.retrieval.retrieval_log import log_retrieval_results
from src.model.llm import ModelClient, get_default_model

from .cypher_queries import (DISEASE_TREATMENT_QUERY,
                             EVIDENCE_DISEASE_TREATMENT_QUERY, RETRIEVAL_QUERY)
from .graph_plan import (_extract_json, _fallback_entity, _normalize_plan,
                         MAX_RESULT_LIMIT, QueryPlan, RetrievalState)

logger = logging.getLogger(__name__)


DATABASE = "neo4j"


DEFAULT_RESULT_LIMIT = 10


DEFAULT_QUESTION = "高脂血症有哪些治疗药品？"


class DrugGraphRetriever:
    """Canonical disease-treatment graph service.

    The legacy class name is preserved as a compatibility API.
    """

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        database: str = DATABASE,
        model_client: ModelClient | None = None,
    ) -> None:
        self.database = database
        self.model_client = model_client or get_default_model()
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.retrieval_workflow = self._build_retrieval_workflow()
        self.preplanned_retrieval_workflow = self._build_preplanned_retrieval_workflow()
        self.workflow = self._build_workflow()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "DrugGraphRetriever":
        self.verify_connectivity()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _analyze_question(self, state: RetrievalState) -> dict[str, Any]:
        requested_limit = max(
            1, min(state.get("requested_limit", DEFAULT_RESULT_LIMIT), MAX_RESULT_LIMIT)
        )
        try:
            response = self.model_client.generate(
                ANALYSIS_PROMPT.format(question=state["question"]),
                use_thinking=False,
            )
            plan = _normalize_plan(
                _extract_json(response), state["question"], requested_limit
            )
            return {"plan": plan, "error": ""}
        except Exception as error:
            plan: QueryPlan = {
                "intent": "general_search",
                "entities": [{
                    "name": _fallback_entity(state["question"]), "type": "Any"
                }],
                "focus": ["overview"],
                "limit": requested_limit,
            }
            return {"plan": plan, "error": f"Question analysis fallback: {error}"}

    def _retrieve_graph(self, state: RetrievalState) -> dict[str, Any]:
        plan = state["plan"]
        keywords = list(dict.fromkeys(
            entity["name"].strip()
            for entity in plan["entities"]
            if entity["name"].strip()
        ))
        if not keywords:
            return {"records": [], "context": "[]", "error": "No searchable entity"}
        try:
            with self.driver.session(database=self.database) as session:
                records = [
                    record.data()
                    for record in session.run(
                        RETRIEVAL_QUERY,
                        keywords=keywords,
                        limit=plan["limit"],
                    )
                ]
            return {
                "records": records,
                "context": json.dumps(records, ensure_ascii=False, indent=2),
            }
        except Exception as error:
            previous_error = state.get("error", "")
            separator = "; " if previous_error else ""
            return {
                "records": [],
                "context": "[]",
                "error": f"{previous_error}{separator}Neo4j retrieval failed: {error}",
            }

    def _generate_answer(self, state: RetrievalState) -> dict[str, str]:
        if not state.get("records"):
            detail = f" ({state.get('error')})" if state.get("error") else ""
            return {"answer": f"当前知识图谱没有检索到相关信息。{detail}"}
        try:
            answer = self.model_client.generate(
                self._answer_prompt(state), use_thinking=False
            )
            return {"answer": answer.strip()}
        except Exception as error:
            return {
                "answer": (
                    f"已检索到图谱数据，但大模型生成答案失败：{error}\n\n"
                    f"原始图谱结果：\n{state['context']}"
                )
            }

    @staticmethod
    def _answer_prompt(state: RetrievalState) -> str:
        return ANSWER_PROMPT.format(
            question=state["question"],
            intent=state["plan"]["intent"],
            focus="、".join(state["plan"].get("focus", ["overview"])),
            context=state["context"],
        )

    def _build_retrieval_workflow(self):
        builder = StateGraph(RetrievalState)
        builder.add_node("analyze_question", self._analyze_question)
        builder.add_node("retrieve_graph", self._retrieve_graph)
        builder.add_edge(START, "analyze_question")
        builder.add_edge("analyze_question", "retrieve_graph")
        builder.add_edge("retrieve_graph", END)
        return builder.compile()

    def _build_preplanned_retrieval_workflow(self):
        """构建跳过模型问题分析、直接执行既有图计划的检索流。"""
        builder = StateGraph(RetrievalState)
        builder.add_node("retrieve_graph", self._retrieve_graph)
        builder.add_edge(START, "retrieve_graph")
        builder.add_edge("retrieve_graph", END)
        return builder.compile()

    def _build_workflow(self):
        builder = StateGraph(RetrievalState)
        builder.add_node("analyze_question", self._analyze_question)
        builder.add_node("retrieve_graph", self._retrieve_graph)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_edge(START, "analyze_question")
        builder.add_edge("analyze_question", "retrieve_graph")
        builder.add_edge("retrieve_graph", "generate_answer")
        builder.add_edge("generate_answer", END)
        return builder.compile()

    def ask(
        self,
        question: str,
        limit: int = DEFAULT_RESULT_LIMIT,
        retrieval_id: str = "",
    ) -> RetrievalState:
        return self._invoke(
            self.workflow, question, limit, retrieval_id, generate_answer=True
        )

    def retrieve(
        self,
        question: str,
        limit: int = DEFAULT_RESULT_LIMIT,
        retrieval_id: str = "",
        plan: dict[str, Any] | None = None,
    ) -> RetrievalState:
        normalized_plan = (
            _normalize_plan(plan, question, limit) if plan is not None else None
        )
        return self._invoke(
            (
                self.preplanned_retrieval_workflow
                if normalized_plan is not None else self.retrieval_workflow
            ),
            question, limit, retrieval_id,
            generate_answer=False, plan=normalized_plan,
        )

    def retrieve_disease_treatments(
        self,
        disease_name: str,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[dict[str, Any]]:
        """按规范疾病名直接检索治疗关系，不调用 LLM 做问题解析。

        调用方已经得到规范疾病名时，使用确定性的参数化图查询，避免再次让
        模型从自然语言中猜测疾病实体。
        """
        disease = (disease_name or "").strip()
        if not disease:
            return []
        bounded_limit = max(1, min(limit, MAX_RESULT_LIMIT))
        with self.driver.session(database=self.database) as session:
            return [
                record.data()
                for record in session.run(
                    DISEASE_TREATMENT_QUERY,
                    disease=disease,
                    limit=bounded_limit,
                )
            ]

    def retrieve_evidence_disease_treatments(
        self,
        evidence_texts: list[str],
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[dict[str, Any]]:
        """从 Milvus 召回文本中识别图谱已有疾病，并返回其治疗关系。"""
        evidence = [str(text).strip() for text in evidence_texts if str(text).strip()]
        if not evidence:
            return []
        bounded_limit = max(1, min(limit, MAX_RESULT_LIMIT))
        with self.driver.session(database=self.database) as session:
            return [
                record.data()
                for record in session.run(
                    EVIDENCE_DISEASE_TREATMENT_QUERY,
                    evidence_texts=evidence,
                    limit=bounded_limit,
                )
            ]

    def _invoke(
        self,
        workflow: Any,
        question: str,
        limit: int,
        retrieval_id: str,
        *,
        generate_answer: bool,
        plan: QueryPlan | None = None,
    ) -> RetrievalState:
        query = question.strip()
        if not query:
            raise ValueError("问题不能为空")
        bounded_limit = max(1, min(limit, MAX_RESULT_LIMIT))
        started_at = time.perf_counter()
        try:
            input_state: RetrievalState = {
                "question": query,
                "requested_limit": bounded_limit,
            }
            if plan is not None:
                input_state["plan"] = plan
            state = workflow.invoke(input_state)
        except Exception as exc:
            log_retrieval_results(
                engine="graph", query=query, results=[],
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                retrieval_id=retrieval_id, status="error",
                error=f"{type(exc).__name__}: {exc}",
                details={
                    "database": self.database,
                    "requested_top_k": bounded_limit,
                    "generate_answer": generate_answer,
                },
            )
            raise

        records = state.get("records", [])
        state_error = state.get("error", "")
        log_retrieval_results(
            engine="graph", query=query, results=records,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            retrieval_id=retrieval_id,
            status=(
                "partial" if records and state_error
                else "success" if records
                else "error" if state_error
                else "empty"
            ),
            error=state_error,
            details={"database": self.database, "plan": state.get("plan", {})},
        )
        return state

    def stream(
        self,
        question: str,
        limit: int = DEFAULT_RESULT_LIMIT,
        include_context: bool = False,
    ) -> Iterator[dict[str, Any]]:
        yield {"type": "status", "stage": "analyzing", "message": "正在理解问题"}
        state = self.retrieve(question, limit)
        plan = state["plan"]
        records = state.get("records", [])
        sources = list(dict.fromkeys(
            str(record.get("entity_name", "")).strip()
            for record in records if record.get("entity_name")
        ))
        yield {
            "type": "meta", "intent": plan["intent"],
            "focus": plan.get("focus", ["overview"]),
            "entities": plan["entities"], "result_count": len(records),
            "sources": sources[:8], "error": state.get("error", ""),
        }
        if include_context:
            yield {
                "type": "context", "plan": plan,
                "context": state.get("context", "[]"),
            }
        if not records:
            error = state.get("error", "")
            answer = (
                "知识图谱服务暂时无法连接，请检查 Neo4j 服务和 Bolt 地址后再试。"
                if "Neo4j retrieval failed" in error
                else "当前知识图谱没有检索到相关信息。请尝试使用更完整的疾病、药品、食药物质、治疗措施或证型名称。"
            )
            yield {"type": "token", "text": answer}
            yield {"type": "done"}
            return

        yield {
            "type": "status", "stage": "answering",
            "message": "正在整理图谱证据并生成回答",
        }
        has_answer = False
        try:
            for chunk in self.model_client.stream(
                self._answer_prompt(state), use_thinking=False
            ):
                if chunk["type"] == "reasoning":
                    continue
                has_answer = True
                yield {"type": "token", "text": chunk["text"]}
            if not has_answer:
                raise RuntimeError("大模型没有返回最终答案")
        except Exception as error:
            yield {"type": "error", "message": f"大模型生成失败：{error}"}
        yield {"type": "done"}


ClinicalGraphRetriever = DrugGraphRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disease-treatment Neo4j graph retrieval"
    )
    parser.add_argument(
        "question", nargs="?", default=DEFAULT_QUESTION,
        help=f"Query question (default: {DEFAULT_QUESTION})",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_RESULT_LIMIT)
    parser.add_argument("--database", default=DATABASE)
    parser.add_argument(
        "--show-context", action="store_true",
        help="Print the query plan and raw graph records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_event: dict[str, Any] = {}
    with DrugGraphRetriever(database=args.database) as retriever:
        for event in retriever.stream(
            args.question, args.limit, include_context=args.show_context
        ):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "error":
                print(f"\n{event['message']}", end="", flush=True)
            elif event["type"] == "context":
                context_event = event
    print()
    if args.show_context:
        print("\n--- Query plan ---")
        print(json.dumps(
            context_event.get("plan", {}), ensure_ascii=False, indent=2
        ))
        print("\n--- Graph records ---")
        print(context_event.get("context", "[]"))

if __name__ == "__main__":
    main()

