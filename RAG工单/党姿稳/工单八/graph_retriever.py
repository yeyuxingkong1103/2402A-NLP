# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
基于知识图谱的检索（Graph RAG 的检索侧）：
  1. 实体链接：从用户问题中识别图谱实体（公司名/指标/领域等），支持别名与子串匹配；
  2. 子图扩展：沿 GRAPH_HOP 跳扩展得到相关实体集合（1 跳即可覆盖"公司-指标-数值"链）；
  3. 文本块定位：把命中实体关联的原文块作为候选（即"答案所在的文本块"）；
  4. 与 01-06 的混合检索（向量+全文）结果做加权融合，返回统一排序的 Top-K 块，
     同时输出用于展示的图谱结构（实体、关系）。
"""
import math
import re

from config import GRAPH_HOP, GRAPH_WEIGHT, RECALL_TOP_N, TOP_K


class GraphRetriever:
    """知识图谱检索器"""

    def __init__(self, chunks, graph_data, hybrid_retriever=None, hop=GRAPH_HOP,
                 graph_weight=GRAPH_WEIGHT):
        self.chunks = chunks
        self._pos = {id(c): i for i, c in enumerate(chunks)}
        self.data = graph_data
        self.hybrid = hybrid_retriever
        self.hop = hop
        self.graph_weight = graph_weight
        self.node_names = [n["id"] for n in graph_data["nodes"]]
        self.node_type = {n["id"]: n["type"] for n in graph_data["nodes"]}
        self.entity_chunks = {k: v for k, v in graph_data.get("entity_chunks", {}).items()}
        self.adj = {}
        for e in graph_data["edges"]:
            self.adj.setdefault(e["source"], set()).add(e["target"])
            self.adj.setdefault(e["target"], set()).add(e["source"])

    # ---------- 1. 实体链接 ----------
    # 主要公司别名（避免"中国"这类公共前缀造成误链接）
    COMPANY_ALIASES = {
        "平安银行": ["平安银行", "平安银行股份"],
        "中国平安": ["中国平安", "平安集团"],
        "招商银行": ["招商银行", "招行"],
        "邮储银行": ["邮储银行", "邮储", "邮政储蓄"],
        "中信证券": ["中信证券", "中信证"],
        "中国人寿": ["中国人寿", "人寿保险"],
        "招商证券": ["招商证券"],
        "中国太保": ["中国太保", "太保", "太平洋保险"],
        "国泰君安": ["国泰君安"],
    }
    # 主题类实体（指标/策略/领域/机构）的查询扩展词表
    TOPIC_EXPAND = {
        "风险管理": ["风险管理", "信用风险", "风险管控", "风控"],
        "资本管理": ["资本结构", "资本管理", "资本充足"],
        "核心一级资本充足率": ["资本充足"],
        "不良贷款率": ["资产质量", "不良贷款", "逾期贷款"],
        "拨备覆盖率": ["拨备"],
        "金融科技": ["金融科技", "数字化转型", "科技金融"],
        "绿色金融": ["绿色金融", "绿色信贷", "环保"],
        "中长期发展战略纲要（2019-2025）": ["战略", "发展战略"],
    }

    def link_entities(self, query):
        """
        从问题中链接图谱实体：
          1) 实体名直接出现在问题中（精确子串）；
          2) 公司别名命中（带别名词表，避免公共前缀误匹配）；
          3) 主题词扩展命中（处理"资本结构优化"这类表述差异）。
        """
        q = query.replace(" ", "")
        hits = set()
        for name in self.node_names:
            if len(name) < 2 or name not in q:
                continue
            if re.fullmatch(r"\d+", name):
                continue                     # 纯数字（如"2019"）不做内容锚点
            if self.node_type.get(name) == "数值":
                continue
            if self.node_type.get(name) == "年份" and len(self.entity_chunks.get(name, [])) < 5:
                continue                     # 语料中极少出现的年份属噪声
            hits.add(name)
        for canon, aliases in self.COMPANY_ALIASES.items():
            if canon in self.node_type and any(a in q for a in aliases):
                hits.add(canon)
        for canon, kws in self.TOPIC_EXPAND.items():
            if canon in self.node_type and any(k in q for k in kws):
                hits.add(canon)
        return hits

    # ---------- 2. 子图扩展 ----------
    def expand(self, seeds, hop=None):
        hop = self.hop if hop is None else hop
        cur, seen = set(seeds), set(seeds)
        for _ in range(hop):
            nxt = set()
            for n in cur:
                nxt |= self.adj.get(n, set())
            nxt -= seen
            seen |= nxt
            cur = nxt
        return seen

    # ---------- 3. 文本块定位 ----------
    def graph_search(self, query, top_k=TOP_K):
        """
        图检索：返回 [(块索引, 分数), ...] 与命中的实体/关系。
        分数 = 关联实体权重之和，实体越具体（名字越长）权重越高，避免大量平票。
        """
        seeds = self.link_entities(query)
        entities = self.expand(seeds) if seeds else set()
        # 计分实体 = 种子实体 + 过滤后的扩展实体：
        # 扩展实体里 df 极小的"数值/未知/人物"多为抽取噪声，一旦计入会压过真正的种子实体。
        scored = set(seeds)
        for e in entities:
            if e in seeds:
                continue
            if len(self.entity_chunks.get(e, [])) >= 3 and \
                    self.node_type.get(e) not in ("数值", "未知", "人物"):
                scored.add(e)
        # 两级计分：种子实体权重 = 1/df（越"稀有"越具体，如"内含价值"压过"中国人寿"）；
        # 扩展实体只作次要证据，权重取种子的 15%，避免稀释主证据。
        seed_sc, exp_sc = {}, {}
        for e in scored:
            n = len(self.entity_chunks.get(e, []))
            if n == 0:
                continue
            t = self.node_type.get(e)
            if t == "公司":
                w = 1.0                       # 公司=报告级硬限定，权重固定最高
            else:
                w = 1.0 / (1.0 + math.log1p(n))
                if t == "年份":
                    w *= 0.2                  # 年份只做"报告年份"过滤，不承担定位答案的职责
            tgt = seed_sc if e in seeds else exp_sc
            for ci in self.entity_chunks[e]:
                tgt[ci] = tgt.get(ci, 0.0) + w
        if seed_sc:
            # 扩展证据必须受限于种子证据（≤50%），否则"公司→全部块"再扩一跳
            # 会把无关报告的块累加到种子之前，Graph RAG 反而不如纯混合检索。
            idx_score = {ci: s + min(0.15 * exp_sc.get(ci, 0.0), 0.5 * s)
                         for ci, s in seed_sc.items()}
        else:
            idx_score = dict(exp_sc)
        ranked = sorted(idx_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked, seeds, entities, self.subgraph(entities)


    # ---------- 4. 图谱结构（供可视化/展示） ----------
    def subgraph(self, entities):
        """返回与给定实体相关的子图（节点+边）"""
        ents = set(entities)
        nodes = [{"id": n, "type": self.node_type.get(n, "未知")} for n in ents]
        edges = [{"source": e["source"], "target": e["target"], "relation": e["relation"]}
                 for e in self.data["edges"] if e["source"] in ents and e["target"] in ents]
        return {"nodes": nodes, "edges": edges}

    def full_graph(self):
        return self.data

    # ---------- 统一检索入口 ----------
    def search(self, query, top_k=TOP_K):
        """
        Graph RAG 混合检索：图谱定位的块 ⊕ 向量/全文混合检索的块
        返回 (results, graph_struct)，results=[(chunk, score, source), ...]
        """
        g_ranked, seeds, entities, gstruct = self.graph_search(query, top_k=max(top_k, RECALL_TOP_N))
        g_map = {i: s for i, s in g_ranked}
        g_norm = {i: s / (max(g_map.values()) or 1.0) for i, s in g_map.items()} if g_map else {}

        text_ranked = []
        if self.hybrid is not None:
            text_ranked = self.hybrid.hybrid_search(query, top_k=max(top_k, RECALL_TOP_N))
        t_map, t_norm = {}, {}
        if text_ranked:
            scores = [s for _, s in text_ranked]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0
            for c, s in text_ranked:
                i = self._pos.get(id(c))
                if i is None:
                    continue
                t_map[i] = s
                t_norm[i] = (s - lo) / span

        # 兜底：问题中没有直接命中图谱实体（如跨文档综合问题）时，
        # 从混合检索Top块的原文中回捞实体，仍可展示与之相关的图谱结构
        if not gstruct["nodes"] and text_ranked:
            in_text = set()
            for c, _s in text_ranked[:5]:
                t = c["text"]
                for name in self.node_names:
                    if len(name) >= 3 and name in t:
                        in_text.add(name)
            if in_text:
                ents = self.expand(in_text)
                gstruct = self.subgraph(ents)
                gstruct["entities"] = sorted(ents)
                entities = ents

        fused = {}
        for i in set(g_norm) | set(t_norm):
            fused[i] = self.graph_weight * g_norm.get(i, 0.0) + (1 - self.graph_weight) * t_norm.get(i, 0.0)
        out = []
        for i, s in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            src = "图谱" if i in g_norm and i not in t_norm else (
                "混合" if i in g_norm and i in t_norm else "向量/全文")
            out.append((self.chunks[i], s, src, g_map.get(i)))
        return out, {"seeds": sorted(seeds), "entities": sorted(entities), **gstruct}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from kb import load_chunks
    from graph_builder import load_graph
    from hybrid_retriever import HybridRetriever
    cs = load_chunks()
    gr = GraphRetriever(cs, load_graph(), HybridRetriever(cs, reranker="none"))
    for q in ["平安银行的拨备覆盖率是多少？", "中国太保的新业务价值是多少？"]:
        res, gs = gr.search(q, 3)
        print(f"\nQ: {q}\n  命中实体: {gs['seeds']}\n  扩展实体: {gs['entities'][:10]}")
        for c, s, src, g in res:
            print(f"  [{src}] {c['doc']} 第{c['page']}页 {s:.3f}: {c['text'][:80]}")
