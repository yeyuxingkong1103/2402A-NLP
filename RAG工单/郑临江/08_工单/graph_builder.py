# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
知识图谱构建模块：
  1. 实体抽取：公司、人物、金融指标；
  2. 关系抽取：共现关系 + 规则关系（如“董事长”“指标”）；
  3. 图谱结构：nodes（实体）+ edges（关系）。
"""
import re
import config


def extract_entities(text):
    """从文本抽取实体，返回实体集合（去重）。"""
    entities = set()
    # 公司
    for name in config.COMPANIES:
        if name in text:
            entities.add(("COMPANY", name))
    # 指标
    for m in config.METRICS:
        if m in text:
            entities.add(("METRIC", m))
    # 人物：头衔 + 2~4 个汉字姓名（简化）
    for title in config.PERSON_TITLES:
        for m in re.finditer(title + r"([一-龥]{2,4})", text):
            entities.add(("PERSON", m.group(1)))
    return entities


def extract_relations(text):
    """从文本抽取关系，返回 [(头实体, 关系, 尾实体)]。"""
    entities = extract_entities(text)
    ents = [e for e in entities]
    relations = []
    # 规则关系：人物头衔
    for title in config.PERSON_TITLES:
        for m in re.finditer(r"([一-龥]{2,8}?公司)?[一-龥]*?" + title + r"([一-龥]{2,4})", text):
            company = m.group(1) or "某公司"
            relations.append((company, "高管", m.group(2)))
    # 指标关系：公司 与 指标共现
    companies = [name for t, name in entities if t == "COMPANY"]
    metrics = [name for t, name in entities if t == "METRIC"]
    for c in companies:
        for m in metrics:
            relations.append((c, "财务指标", m))
    # 共现关系：同段内公司两两关联
    for i in range(len(companies)):
        for j in range(i + 1, len(companies)):
            relations.append((companies[i], "相关", companies[j]))
    return relations


class KnowledgeGraph:
    """知识图谱：nodes + edges，支持去重与查询。"""

    def __init__(self):
        self.nodes = {}  # id -> {"label":.., "type":..}
        self.edges = {}  # (src, rel, dst) -> count

    def add_entity(self, etype, name):
        self.nodes.setdefault(name, {"label": name, "type": etype})

    def add_relation(self, src, rel, dst):
        if src and dst and src != dst:
            key = (src, rel, dst)
            self.edges[key] = self.edges.get(key, 0) + 1

    def build_from_documents(self, documents):
        for doc in documents:
            # 按窗口切段
            for start in range(0, len(doc), config.COOCCUR_WINDOW):
                seg = doc[start:start + config.COOCCUR_WINDOW]
                for etype, name in extract_entities(seg):
                    self.add_entity(etype, name)
                for src, rel, dst in extract_relations(seg):
                    self.add_relation(src, rel, dst)

    def to_dict(self):
        return {
            "nodes": [{"id": k, **v} for k, v in self.nodes.items()],
            "edges": [{"source": s, "target": t, "relation": r, "weight": w}
                      for (s, r, t), w in self.edges.items()],
        }

    def neighbors(self, entity, hops=1):
        """返回实体的邻居（一跳或两跳）。"""
        seen = {entity}
        frontier = {entity}
        for _ in range(hops):
            nxt = set()
            for (s, r, t) in self.edges:
                if s in frontier and t not in seen:
                    nxt.add(t)
                if t in frontier and s not in seen:
                    nxt.add(s)
            seen |= nxt
            frontier = nxt
        return seen

    def summary(self):
        return f"节点 {len(self.nodes)} 个，关系 {len(self.edges)} 条"
