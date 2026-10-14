"""Work order 08: financial Graph RAG with entity/relation extraction."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, dense_search


@dataclass
class Entity:
    name: str
    kind: str = "unknown"


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    source: str = ""
    confidence: float = 1.0


class GraphStore:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []

    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None:
        for entity in entities:
            self.entities[entity.name] = entity
        self.relations.extend(relations)

    def neighborhood(self, names: set[str], hops: int = 2) -> list[Relation]:
        selected = set(names)
        output = []
        for _ in range(hops):
            changed = False
            for relation in self.relations:
                if relation.subject in selected or relation.object in selected:
                    if relation not in output:
                        output.append(relation)
                    for node in (relation.subject, relation.object):
                        if node not in selected:
                            selected.add(node)
                            changed = True
            if not changed:
                break
        return output


class FinancialExtractor(Protocol):
    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]: ...


class RegexFinancialExtractor:
    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]:
        names = re.findall(r"[\u4e00-\u9fff]{2,12}(?:公司|银行|集团|证券|基金)", text)
        entities = [Entity(name, "company") for name in sorted(set(names))]
        relations = []
        for left, right in zip(entities, entities[1:]):
            relations.append(Relation(left.name, "mentioned_with", right.name))
        return entities, relations


class FinancialGraphRAG:
    def __init__(self, chunks: list[Chunk], graph: GraphStore, extractor: FinancialExtractor | None = None):
        self.chunks = chunks
        self.graph = graph
        self.extractor = extractor or RegexFinancialExtractor()

    def ingest(self) -> None:
        for chunk in self.chunks:
            entities, relations = self.extractor.extract(chunk.text)
            for relation in relations:
                relation.source = chunk.source
            self.graph.upsert(entities, relations)

    def retrieve(self, query: str, top_k: int = 6) -> dict[str, Any]:
        semantic = dense_search(query, self.chunks, top_k)
        mentioned = {name for name in self.graph.entities if name in query}
        graph_context = self.graph.neighborhood(mentioned)
        return {
            "documents": [{"chunk": chunk.to_dict(), "score": score} for chunk, score in semantic],
            "graph": [relation.__dict__ for relation in graph_context],
        }

    def prompt(self, query: str) -> str:
        result = self.retrieve(query)
        return "Answer the financial question using both document evidence and graph facts.\n" + str(result)
