"""Work order 09: optimize Graph RAG with entity linking, graph reranking and caching."""

from __future__ import annotations

import functools
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, dense_search


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


class OptimizedKnowledgeGraph:
    def __init__(self, edges: Iterable[Edge] = ()):
        self.adjacency: defaultdict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            self.adjacency[edge.source].append(edge)
            self.adjacency[edge.target].append(Edge(edge.target, edge.source, edge.relation, edge.weight))

    def expand(self, seeds: set[str], depth: int = 2) -> tuple[set[str], list[Edge]]:
        nodes = set(seeds)
        edges: list[Edge] = []
        frontier = set(seeds)
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for edge in self.adjacency.get(node, []):
                    edges.append(edge)
                    if edge.target not in nodes:
                        nodes.add(edge.target)
                        next_frontier.add(edge.target)
            frontier = next_frontier
        return nodes, edges


def normalize_entity(name: str) -> str:
    return re.sub(r"(股份有限公司|有限公司|集团)$", "", name).strip().lower()


class GraphRAGOptimizer:
    def __init__(self, chunks: list[Chunk], graph: OptimizedKnowledgeGraph):
        self.chunks = chunks
        self.graph = graph

    @functools.lru_cache(maxsize=512)
    def _cached_query(self, query: str, top_k: int) -> tuple[str, ...]:
        return tuple(chunk.id for chunk, _ in dense_search(query, self.chunks, top_k))

    def retrieve(self, query: str, top_k: int = 8) -> list[Chunk]:
        ids = set(self._cached_query(query, top_k * 2))
        seeds = {node for node in self.graph.adjacency if normalize_entity(node) in normalize_entity(query)}
        graph_nodes, edges = self.graph.expand(seeds)
        scored = []
        for chunk in self.chunks:
            lexical_graph_bonus = sum(node.lower() in chunk.text.lower() for node in graph_nodes)
            semantic_bonus = 1.0 if chunk.id in ids else 0.0
            scored.append((chunk, semantic_bonus + 0.2 * lexical_graph_bonus))
        return [chunk for chunk, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]]
