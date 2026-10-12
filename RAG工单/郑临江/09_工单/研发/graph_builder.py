# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务
知识图谱构建模块：
  - build_graph_baseline：基线（共现关系，粗粒度）
  - build_graph_optimized：优化（prompt 引导 LLM 抽取精确实体/关系，离线降级为规则）
"""
import re
import json
import config


def _rule_entities(text):
    ents = set()
    for name in config.COMPANIES:
        if name in text:
            ents.add(("公司", name))
    for m in config.METRICS:
        if m in text:
            ents.add(("金融指标", m))
    for title in ["董事长", "行长", "总经理", "总裁"]:
        for m in re.finditer(title + r"([一-龥]{2,4})", text):
            ents.add(("人物", m.group(1)))
    return ents


def _rule_relations(text, entities):
    rels = []
    comps = [n for t, n in entities if t == "公司"]
    metrics = [n for t, n in entities if t == "金融指标"]
    for c in comps:
        for m in metrics:
            rels.append((c, "指标", m))
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            rels.append((comps[i], "相关", comps[j]))
    return rels


def build_graph_baseline(documents):
    """基线：规则 + 共现，关系较粗。"""
    nodes, edges = {}, {}
    for doc in documents:
        for s in range(0, len(doc), config.COOCCUR_WINDOW):
            seg = doc[s:s + config.COOCCUR_WINDOW]
            ents = _rule_entities(seg)
            for t, n in ents:
                nodes.setdefault(n, {"label": n, "type": t})
            for src, rel, dst in _rule_relations(seg, ents):
                edges[(src, rel, dst)] = edges.get((src, rel, dst), 0) + 1
    return {"nodes": nodes, "edges": edges}


# —— 优化：prompt 引导 LLM 抽取 ——
EXTRACT_PROMPT = (
    "你是金融知识图谱构建助手。请从下面文本中抽取实体与关系，"
    "实体类型限定为：{entity_types}；关系类型限定为：{relation_types}。\n"
    "只输出 JSON，格式：{{\"entities\":[{{\"name\":\"..\",\"type\":\"..\"}}],"
    "\"relations\":[{{\"head\":\"..\",\"relation\":\"..\",\"tail\":\"..\"}}]}}\n"
    "文本：{text}"
)


def extract_with_llm(text, llm):
    """使用 LLM prompt 抽取精确实体与关系。"""
    if not (llm and llm.api_key):
        return None
    prompt = EXTRACT_PROMPT.format(
        entity_types="/".join(config.ENTITY_TYPES),
        relation_types="/".join(config.RELATION_TYPES),
        text=text[:800],
    )
    try:
        out = llm.generate(prompt)
        # 提取 JSON 片段
        start = out.find("{")
        end = out.rfind("}") + 1
        data = json.loads(out[start:end])
        return data
    except Exception:
        return None


def build_graph_optimized(documents, llm=None):
    """优化：LLM prompt 抽取；不可用时降级为规则抽取。"""
    nodes, edges = {}, {}
    for doc in documents:
        for s in range(0, len(doc), config.COOCCUR_WINDOW):
            seg = doc[s:s + config.COOCCUR_WINDOW]
            data = extract_with_llm(seg, llm)
            if data:
                for e in data.get("entities", []):
                    nodes.setdefault(e["name"], {"label": e["name"], "type": e.get("type", "实体")})
                for r in data.get("relations", []):
                    key = (r["head"], r["relation"], r["tail"])
                    edges[key] = edges.get(key, 0) + 1
            else:
                ents = _rule_entities(seg)
                for t, n in ents:
                    nodes.setdefault(n, {"label": n, "type": t})
                for src, rel, dst in _rule_relations(seg, ents):
                    edges[(src, rel, dst)] = edges.get((src, rel, dst), 0) + 1
    return {"nodes": nodes, "edges": edges}


def neighbors(graph, entity, hops=1):
    seen = {entity}
    frontier = {entity}
    for _ in range(hops):
        nxt = set()
        for (s, r, t) in graph["edges"]:
            if s in frontier and t not in seen:
                nxt.add(t)
            if t in frontier and s not in seen:
                nxt.add(s)
        seen |= nxt
        frontier = nxt
    return seen
