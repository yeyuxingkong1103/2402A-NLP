# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
知识图谱构建：实体/关系抽取、去重、增量更新。
"""
from extractor import extract_entities, extract_relations, extract_with_llm


class KnowledgeGraph:
    def __init__(self):
        self.entities = {}     # name -> {"type": ..., "doc_id": ...}
        self.relations = []    # (head, rel, tail)
        self.adj = {}          # head -> [(rel, tail)]

    def add_entity(self, name, etype, doc_id):
        if name not in self.entities:
            self.entities[name] = {"type": etype, "doc_id": doc_id}

    def add_relation(self, head, rel, tail):
        self.relations.append((head, rel, tail))
        self.adj.setdefault(head, []).append((rel, tail))

    def build(self, docs, use_llm=False):
        for d in docs:
            if use_llm:
                res = extract_with_llm(d["text"])
                for name, etype in res.get("entities", []):
                    self.add_entity(name, etype, d["id"])
                for h, r, t in res.get("relations", []):
                    self.add_relation(h, r, t)
            else:
                ents = extract_entities(d["text"])
                for name, etype in ents:
                    self.add_entity(name, etype, d["id"])
                for h, r, t in extract_relations(ents, d["text"]):
                    self.add_relation(h, r, t)

    def update(self, new_docs, use_llm=False):
        """增量更新：仅对新文档抽取，无需重建整个知识库。"""
        self.build(new_docs, use_llm=use_llm)

    def neighbors(self, entity, depth=1):
        """沿关系扩展邻居实体（全局检索用）。"""
        seen, frontier = {entity}, [entity]
        for _ in range(depth):
            nxt = []
            for n in frontier:
                for rel, t in self.adj.get(n, []):
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
            frontier = nxt
        return seen

    def stats(self):
        return {"entities": len(self.entities), "relations": len(self.relations)}

    def export(self):
        return {
            "entities": [{"name": k, "type": v["type"]} for k, v in self.entities.items()],
            "relations": [{"head": h, "relation": r, "tail": t} for h, r, t in self.relations],
        }
