"""Work order 05: query understanding, rewriting and intent-aware retrieval."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, dense_search, lexical_search, rrf


class QueryLLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


@dataclass
class QueryPlan:
    original: str
    standalone: str
    sub_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    intent: str = "lookup"


def detect_intent(query: str) -> str:
    if re.search(r"比较|区别|对比|哪个更", query):
        return "comparison"
    if re.search(r"为什么|原因|原理", query):
        return "explanation"
    if re.search(r"步骤|如何|怎么|流程", query):
        return "procedure"
    return "lookup"


def heuristic_plan(query: str) -> QueryPlan:
    terms = re.split(r"(?:和|与|以及|并且|、)", query)
    sub_queries = [term.strip() for term in terms if len(term.strip()) >= 2]
    return QueryPlan(query, query, sub_queries or [query], intent=detect_intent(query))


def llm_plan(query: str, llm: QueryLLM) -> QueryPlan:
    prompt = (
        "Convert the user question into JSON with keys standalone, sub_queries, filters, intent. "
        "Do not answer the question. User question: " + query
    )
    import json

    data = json.loads(llm.invoke(prompt))
    return QueryPlan(query, data.get("standalone", query), data.get("sub_queries", [query]), data.get("filters", {}), data.get("intent", "lookup"))


class QueryAwareRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def retrieve(self, plan: QueryPlan, top_k: int = 8) -> list[tuple[Chunk, float]]:
        lists = []
        for query in [plan.standalone, *plan.sub_queries]:
            dense = dense_search(query, self.chunks, top_k=top_k * 2)
            lexical = lexical_search(query, self.chunks, top_k=top_k * 2)
            lists.append(rrf(dense, lexical, top_k=top_k * 2))
        merged: dict[str, tuple[Chunk, float]] = {}
        for ranked in lists:
            for rank, (chunk, score) in enumerate(ranked, 1):
                if plan.filters and any(chunk.metadata.get(k) != v for k, v in plan.filters.items()):
                    continue
                merged[chunk.id] = (chunk, merged.get(chunk.id, (chunk, 0.0))[1] + score / rank)
        return sorted(merged.values(), key=lambda item: item[1], reverse=True)[:top_k]
