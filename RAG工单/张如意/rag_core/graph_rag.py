# -*- coding: utf-8 -*-
"""
Graph RAG 模块（知识图谱构建 + 图谱检索问答）
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答
          人工智能NLP-RAG-Graph RAG 优化任务

核心流程：
  ① 文本分块 → ② LLM 抽取实体与关系 → ③ 实体归一化/去重 → ④ 建图谱
  → ⑤ 社区检测（Leiden/Louvain）生成社区摘要 → ⑥ 图谱检索（局部+全局）→ ⑦ 生成

工单09 的优化点全部固化在 P`romptProfile` 里，通过 profile 参数切换：
  · baseline  —— 朴素抽取 prompt（只见实体，无类型体系）
  · optimized —— 金融领域实体/关系类型体系 + few-shot + 二次抽取(gleaning)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import config, llm

GRAPH_DIR = config.GRAPH_DIR
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 金融领域实体 / 关系类型体系（工单09 的核心优化之一）
# ---------------------------------------------------------------------------
FIN_ENTITY_TYPES = {
    "公司": "企业法人主体，如平安银行股份有限公司、中国平安保险（集团）股份有限公司",
    "机构": "监管机构、交易所、行业协会等，如中国人民银行、深圳证券交易所",
    "人物": "董事、监事、高级管理人员、股东等自然人",
    "金融产品": "贷款、存款、理财、保险、债券、基金等金融产品或服务",
    "业务板块": "零售金融、对公业务、资金同业、资产管理等业务条线",
    "财务指标": "营业收入、净利润、不良贷款率、拨备覆盖率、资本充足率等",
    "风险类型": "信用风险、市场风险、操作风险、流动性风险等",
    "行业": "制造业、批发零售业、房地产业等国民经济行业分类",
    "地区": "省、市、自治区等地理区域",
    "监管政策": "法律法规、监管规定、标准规范",
    "时间": "年份、报告期、季度等时间概念",
}

FIN_RELATION_TYPES = {
    "子公司": "A 是 B 的子公司 / 控股子公司",
    "控股股东": "A 是 B 的控股股东",
    "参股": "A 参股 B",
    "投资": "A 投资于 B（含金融资产投资）",
    "任职": "A 在 B 担任某职务",
    "提供产品服务": "A 向 B 提供某产品或服务",
    "发行": "A 发行了某金融产品",
    "披露指标": "某报告中披露了某财务指标及其数值",
    "指标数值": "财务指标在特定报告期的取值",
    "面临风险": "A 面临某类风险",
    "受监管于": "A 受 B 监管",
    "属于行业": "A 属于某行业",
    "位于地区": "A 位于某地区",
    "同比变化": "某指标相对上期上升/下降",
    "提及": "文档 A 中提及了实体 B",
}


@dataclass
class PromptProfile:
    """抽取 Prompt 配置（工单09 优化前后对比用）。"""
    name: str = "baseline"
    use_type_system: bool = False        # 是否给出实体/关系类型体系
    use_few_shot: bool = False           # 是否给 few-shot 示例
    gleaning_rounds: int = 0             # 二次抽取轮数（补漏）
    max_entities: int = 20               # 单块最多抽多少实体
    max_relations: int = 30
    description: str = ""


PROFILES = {
    "baseline": PromptProfile(
        name="baseline", use_type_system=False, use_few_shot=False,
        gleaning_rounds=0,
        description="朴素抽取：只要求「抽出实体和关系」，不给类型体系，不做补漏"),
    "optimized": PromptProfile(
        name="optimized", use_type_system=True, use_few_shot=True,
        gleaning_rounds=1, max_entities=30, max_relations=45,
        description="优化抽取：金融领域类型体系 + few-shot + 二次补漏抽取"),
}


# ---------------------------------------------------------------------------
# 抽取 Prompt
# ---------------------------------------------------------------------------
_SYS_BASELINE = """从给定的文本中抽取实体和关系，输出 JSON。
格式：{"entities": [{"name": "...", "type": "...", "description": "..."}],
      "relations": [{"source": "...", "target": "...", "type": "...", "description": "..."}]}
只输出 JSON。"""

_SYS_OPTIMIZED = """你是金融领域知识图谱构建专家。请从给定的招股说明书/年报文本中
抽取实体与关系，用于构建金融知识图谱。

【实体类型体系】（type 必须从这个列表中选择）
{entity_types}

【关系类型体系】（type 必须从这个列表中选择）
{relation_types}

【抽取要求】
1. 实体名称必须是完整规范的全称，不要用「公司」「本公司」「该行」等指代，
   遇到指代要还原成文本中出现的具体名称（如「平安银行股份有限公司」）。
2. 财务指标实体要带期间，例如「营业收入（2019年）」，并把它在文本中的
   具体数值写进 description。
3. 关系必须是文本中**明确陈述**的，不要推断。每条关系的 description 要
   引用原文的关键短语作为依据。
4. 注意抽取「指标数值」「同比变化」这两类关系——它们是回答数值型问题的关键。
5. 遗漏比冗余更严重：宁可多抽有原文依据的关系，也不要漏掉关键数值关系。

【示例】
输入：「2019年，本行实现营业收入1,379.58亿元，同比增长18.2%。」
输出：
{{"entities": [
  {{"name": "营业收入（2019年）", "type": "财务指标", "description": "1,379.58亿元"}},
  {{"name": "报告期（2019年）", "type": "时间", "description": "2019年"}}],
 "relations": [
  {{"source": "营业收入（2019年）", "target": "报告期（2019年）", "type": "指标数值",
    "description": "2019年营业收入为1,379.58亿元"}},
  {{"source": "营业收入（2019年）", "target": "报告期（2019年）", "type": "同比变化",
    "description": "同比增长18.2%"}}]}}

只输出 JSON，不要任何解释。"""

_SYS_GLEAN = """在上一轮抽取的基础上，检查是否遗漏了以下内容，只补充遗漏项：
1. 只出现过一次的数字（金额、比例、年份）
2. 中文括号、顿号分隔的并列实体（如「A、B、C 三家公司」中的 B 和 C）
3. 表格中的行列对应关系
4. 否定表述中的实体（如「不存在控制关系的关联方」）

只输出新增的实体和关系（不要重复已有的），格式同上。若没有遗漏，
输出 {{"entities": [], "relations": []}}。"""


# ---------------------------------------------------------------------------
# 知识图谱
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    name: str
    type: str = "其他"
    description: str = ""
    aliases: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)     # 出现过的 chunk_id
    degree: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type,
                "description": self.description,
                "aliases": sorted(self.aliases), "degree": self.degree,
                "n_sources": len(self.sources)}


@dataclass
class Relation:
    source: str
    target: str
    type: str = "相关"
    description: str = ""
    weight: float = 1.0
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "type": self.type,
                "description": self.description, "weight": self.weight}


class KnowledgeGraph:
    """
    金融知识图谱。

    底层用 networkx 存储（无需外部数据库即可运行）；
    若环境中配置了 Neo4j，可通过 `to_neo4j()` 导入做可视化与 Cypher 查询。
    """

    def __init__(self):
        try:
            import networkx as nx
            self.g = nx.MultiDiGraph()
        except ImportError:
            self.g = None
            self._edges: list[Relation] = []
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self._alias_index: dict[str, str] = {}          # 别名 -> 规范名
        self.communities: dict[int, list[str]] = {}

    # -- 构建 ---------------------------------------------------------------
    def add_extraction(self, data: dict, chunk_id: str = "",
                       min_desc_len: int = 2) -> None:
        """把一次 LLM 抽取结果并入图谱（自动做实体归一化）。"""
        ents = data.get("entities", []) or []
        rels = data.get("relations", []) or []

        for e in ents:
            name = _norm_name(e.get("name", ""))
            if not name or len(name) < min_desc_len:
                continue
            canon = self._resolve_alias(name)
            if canon not in self.entities:
                self.entities[canon] = Entity(name=canon, type=e.get("type", "其他"))
            ent = self.entities[canon]
            # 描述信息累加（同一实体的多次描述合并去重）
            desc = (e.get("description") or "").strip()
            if desc and desc not in ent.description:
                ent.description = (ent.description + "；" + desc).strip("；")[:400]
            if chunk_id and chunk_id not in ent.sources:
                ent.sources.append(chunk_id)
            # 登记别名，便于检索时归一化
            if name != canon:
                ent.aliases.add(name)
                self._alias_index[name] = canon

        for r in rels:
            s = self._resolve_alias(_norm_name(r.get("source", "")))
            t = self._resolve_alias(_norm_name(r.get("target", "")))
            if not s or not t or s == t:
                continue
            # 端点必须已作为实体存在，否则补建（避免关系悬空）
            for n in (s, t):
                if n not in self.entities:
                    self.entities[n] = Entity(name=n, type="其他")
            rel = Relation(source=s, target=t, type=r.get("type", "相关"),
                           description=(r.get("description") or "")[:300],
                           sources=[chunk_id] if chunk_id else [])
            self.relations.append(rel)

        self._rebuild_graph()

    def _resolve_alias(self, name: str) -> str:
        """别名归一化：把「本公司」「平安银行」等指代映射到规范全称。"""
        if name in self._alias_index:
            return self._alias_index[name]
        # 后缀匹配：短名是长名的前缀且长度接近时，视为同一实体
        for canon in self.entities:
            if canon == name:
                return canon
            if len(name) >= 6 and len(canon) >= 6 and (name in canon or canon in name):
                self._alias_index[name] = canon
                return canon
        return name

    def _rebuild_graph(self, max_edges_per_node: int = 200) -> None:
        """重建 networkx 图，并把描述相似的边做加权合并（降噪）。"""
        if self.g is None:
            return
        import networkx as nx
        self.g = nx.MultiDiGraph()

        # 按 (source, target) 聚合，合并同向多条关系
        merged: dict[tuple[str, str], Relation] = {}
        for r in self.relations:
            key = (r.source, r.target)
            if key in merged:
                m = merged[key]
                m.weight += 1
                if r.type not in m.type:
                    m.type = f"{m.type}|{r.type}"
                if r.description and r.description not in m.description:
                    m.description = (m.description + "；" + r.description)[:400]
                m.sources.extend(s for s in r.sources if s not in m.sources)
            else:
                merged[key] = Relation(**{**r.__dict__, "sources": list(r.sources)})

        self.relations_merged = list(merged.values())
        for name, e in self.entities.items():
            self.g.add_node(name, type=e.type, description=e.description)
        for r in self.relations_merged:
            self.g.add_edge(r.source, r.target, type=r.type,
                            description=r.description, weight=r.weight)

        for name in self.entities:
            if name in self.g:
                self.entities[name].degree = self.g.degree(name)

    # -- 社区检测（Graph RAG 的「全局检索」基础）----------------------------
    def detect_communities(self, resolution: float = 1.0) -> dict[int, list[str]]:
        """
        社区检测：把图谱划分为若干主题簇，并为每个簇生成摘要。
        这是 Graph RAG 相比普通 RAG 的关键能力——
        能回答「这些公司应对经济周期的共同策略是什么」这类全局性问题。
        """
        if self.g is None or self.g.number_of_nodes() == 0:
            return {}
        try:
            import networkx as nx
            undirected = nx.Graph(self.g)
            comms = nx.community.louvain_communities(undirected, resolution=resolution,
                                                     seed=42)
        except Exception:
            comms = []
        self.communities = {i: sorted(c) for i, c in enumerate(comms)}
        return self.communities

    def summarize_communities(self, top_n: int = 10) -> dict[int, str]:
        """用 LLM 为每个社区生成主题摘要（全局检索的语料）。"""
        out = {}
        for cid, members in list(self.communities.items())[:top_n]:
            sample = members[:40]
            rels = [r for r in self.relations_merged
                    if r.source in sample and r.target in sample][:60]
            rel_text = "\n".join(
                f"- {r.source} --{r.type}--> {r.target}：{r.description[:60]}"
                for r in rels
            )
            prompt = (f"以下是金融知识图谱中一个社区包含的实体与关系：\n\n"
                      f"实体：{'、'.join(sample)}\n\n关系：\n{rel_text}\n\n"
                      f"请用 150 字以内概括这个社区的主题、涉及的主要机构与关键结论。"
                      f"直接输出摘要文本，不要任何前缀。")
            try:
                out[cid] = llm.chat([{"role": "user", "content": prompt}],
                                    temperature=0.0, max_tokens=400,
                                    tag="community").strip()
            except Exception as e:
                out[cid] = f"（社区摘要生成失败：{e}）"
        self.community_summaries = out
        return out

    # -- 检索 ---------------------------------------------------------------
    def local_search(self, query: str, top_k: int = 10,
                     hops: int = 2) -> dict:
        """
        局部检索：定位问题中的实体 → 沿关系扩展 k 跳 → 收集子图作为上下文。
        适合「某公司的某指标是多少」这类局部事实问题。
        """
        seeds = self.link_entities(query)
        if not seeds:
            return {"seeds": [], "nodes": [], "edges": [], "context": ""}

        nodes, edges = set(seeds), []
        frontier = set(seeds)
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                if self.g is None or n not in self.g:
                    continue
                for _, tgt, d in self.g.out_edges(n, data=True):
                    nxt.add(tgt)
                    edges.append((n, tgt, d))
                for src, _, d in self.g.in_edges(n, data=True):
                    nxt.add(src)
                    edges.append((src, n, d))
            frontier = nxt - nodes
            nodes |= nxt

        return {
            "seeds": seeds,
            "nodes": sorted(nodes),
            "edges": edges,
            "context": self._render_subgraph(query, seeds, nodes, edges),
        }

    def global_search(self, query: str, top_k: int = 5) -> dict:
        """
        全局检索：在社区摘要上做相关性匹配，回答全局性/归纳性问题。
        """
        if not getattr(self, "community_summaries", None):
            self.detect_communities()
            self.summarize_communities()

        summaries = getattr(self, "community_summaries", {})
        if not summaries:
            return {"communities": [], "context": ""}

        picked = _rank_by_overlap(query, summaries, top_k)
        ctx = "\n\n".join(
            f"[社区{cid}｜涉及实体：{'、'.join(self.communities.get(cid, [])[:8])}]\n{s}"
            for cid, s in picked
        )
        return {"communities": [(c, s) for c, s in picked], "context": ctx}

    def link_entities(self, query: str) -> list[str]:
        """实体链接：把问题中的字符串匹配到图谱节点（含别名与模糊匹配）。"""
        hits = []
        for name, e in self.entities.items():
            if name and name in query:
                hits.append(name)
            elif any(a and a in query for a in e.aliases):
                hits.append(name)
        # 补充：抽取问题里的公司名，做后缀匹配
        if not hits:
            for cand in re.findall(r"[一-鿿]{4,}(?:股份有限公司|有限公司|集团|银行)", query):
                for name in self.entities:
                    if cand in name or name in cand:
                        hits.append(name)
                        break
        # 按度数排序，优先返回重要实体
        return sorted(set(hits), key=lambda n: -self.entities[n].degree)[:5]

    def _render_subgraph(self, query, seeds, nodes, edges) -> str:
        """把子图渲染成 LLM 可读的文本（含实体属性与关系描述）。"""
        lines = [f"【知识图谱子图】（问题涉及实体：{'、'.join(seeds)}）", ""]
        lines.append("实体：")
        for n in sorted(nodes, key=lambda x: -self.entities[x].degree)[:30]:
            e = self.entities[n]
            desc = f"：{e.description}" if e.description else ""
            lines.append(f"  · {n}（{e.type}）{desc}")
        lines.append("")
        lines.append("关系：")
        seen = set()
        for s, t, d in edges[:80]:
            key = (s, t, d.get("type", ""))
            if key in seen:
                continue
            seen.add(key)
            desc = f"（{d.get('description', '')[:80]}）" if d.get("description") else ""
            lines.append(f"  · {s} --[{d.get('type', '相关')}]--> {t}{desc}")
        return "\n".join(lines)

    # -- 增量更新（LightRAG 的核心特性，工单12 对比用）---------------------
    def incremental_add(self, chunk_id: str, extraction: dict) -> dict:
        """
        增量更新：只并入新抽取的实体/关系，不重建全图。
        返回本次新增的实体数与关系数。
        """
        n_e, n_r = len(self.entities), len(self.relations)
        self.add_extraction(extraction, chunk_id=chunk_id)
        return {"new_entities": len(self.entities) - n_e,
                "new_relations": len(self.relations) - n_r}

    # -- 可视化 -------------------------------------------------------------
    def visualize(self, path: Path | None = None, max_nodes: int = 120,
                  layout: str = "spring") -> Path:
        """导出知识图谱可视化图（matplotlib，中文节点标签）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        path = Path(path or (GRAPH_DIR / "knowledge_graph.png"))
        path.parent.mkdir(parents=True, exist_ok=True)

        # 剪枝：只保留度数最高的节点，保证出图可读
        deg = sorted(self.entities.items(), key=lambda x: -x[1].degree)
        keep = {n for n, _ in deg[:max_nodes]}
        sub = self.g.subgraph(keep).copy() if self.g else nx.MultiDiGraph()

        if sub.number_of_nodes() == 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "图谱为空", ha="center", va="center", fontsize=16)
            plt.axis("off")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            return path

        pos = (nx.spring_layout(sub, k=1.2, iterations=80, seed=42) if layout == "spring"
               else nx.circular_layout(sub))
        type_colors = {t: c for t, c in zip(FIN_ENTITY_TYPES, plt.cm.tab20.colors)}
        colors = [type_colors.get(self.entities[n].type, "#999999") for n in sub.nodes]
        sizes = [200 + 80 * self.entities[n].degree for n in sub.nodes]

        fig, ax = plt.subplots(figsize=(18, 13))
        nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.3, arrows=True,
                               arrowsize=8, width=0.6)
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors,
                               node_size=sizes, alpha=0.9, linewidths=0.5,
                               edgecolors="white")
        nx.draw_networkx_labels(sub, pos, ax=ax, font_size=7,
                                font_family=_cn_font())
        ax.set_title(f"金融知识图谱（{sub.number_of_nodes()} 实体 / "
                     f"{sub.number_of_edges()} 关系）", fontsize=15)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    def to_neo4j(self, uri: str, user: str, password: str) -> dict:
        """把图谱导入 Neo4j（用于交互式可视化与 Cypher 查询）。"""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return {"ok": False, "msg": "未安装 neo4j 驱动：pip install neo4j"}

        driver = GraphDatabase.driver(uri, auth=(user, password))
        n_e = n_r = 0
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")           # 清空（演示环境）
            for name, e in self.entities.items():
                s.run("MERGE (n:Entity {name:$name}) "
                      "SET n.type=$type, n.description=$desc",
                      name=name, type=e.type, desc=e.description[:500])
                n_e += 1
            for r in self.relations_merged:
                s.run("MATCH (a:Entity {name:$s}), (b:Entity {name:$t}) "
                      "MERGE (a)-[rel:REL {type:$type}]->(b) "
                      "SET rel.description=$desc, rel.weight=$w",
                      s=r.source, t=r.target, type=r.type,
                      desc=r.description[:500], w=r.weight)
                n_r += 1
        driver.close()
        return {"ok": True, "entities": n_e, "relations": n_r}

    # -- 持久化 -------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or (GRAPH_DIR / "kg.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations_merged],
            "communities": {str(k): v for k, v in self.communities.items()},
            "community_summaries": {str(k): v for k, v
                                    in getattr(self, "community_summaries", {}).items()},
            "stats": self.stats(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def stats(self) -> dict:
        types = Counter(e.type for e in self.entities.values())
        rel_types = Counter(r.type for r in self.relations_merged)
        return {
            "n_entities": len(self.entities),
            "n_relations": len(self.relations_merged),
            "entity_types": dict(types.most_common()),
            "relation_types": dict(rel_types.most_common(20)),
            "n_communities": len(self.communities),
        }


# ---------------------------------------------------------------------------
# 抽取器
# ---------------------------------------------------------------------------
class GraphExtractor:
    """基于 LLM 的实体关系抽取器（支持 baseline / optimized 两套 prompt）。"""

    def __init__(self, profile: str | PromptProfile = "optimized"):
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile

    def _system(self) -> str:
        p = self.profile
        if not p.use_type_system:
            return _SYS_BASELINE
        ent = "\n".join(f"  - {k}：{v}" for k, v in FIN_ENTITY_TYPES.items())
        rel = "\n".join(f"  - {k}：{v}" for k, v in FIN_RELATION_TYPES.items())
        return _SYS_OPTIMIZED.format(entity_types=ent, relation_types=rel)

    def extract(self, text: str, chunk_id: str = "") -> dict:
        """从单个文本块抽取实体与关系（optimized 档位带二次补漏）。"""
        p = self.profile
        msgs = [{"role": "system", "content": self._system()},
                {"role": "user", "content": f"文本：\n{text[:3500]}"}]
        try:
            data = llm.chat_json(msgs, temperature=0.0, max_tokens=3000, tag="kg_extract")
        except Exception as e:
            print(f"  [warn] 抽取失败 {chunk_id}: {e}")
            return {"entities": [], "relations": []}

        data.setdefault("entities", [])
        data.setdefault("relations", [])

        # 二次抽取（gleaning）：补漏
        for _ in range(p.gleaning_rounds):
            more = self._glean(text, data)
            if not more.get("entities") and not more.get("relations"):
                break
            data["entities"].extend(more.get("entities", []))
            data["relations"].extend(more.get("relations", []))

        data["entities"] = data["entities"][: p.max_entities]
        data["relations"] = data["relations"][: p.max_relations]
        return data

    def _glean(self, text: str, existing: dict) -> dict:
        have = "、".join(e.get("name", "") for e in existing.get("entities", [])[:40])
        msgs = [
            {"role": "system", "content": _SYS_GLEAN},
            {"role": "user",
             "content": f"原文：\n{text[:3500]}\n\n已抽取的实体：{have}\n\n请补充遗漏项。"},
        ]
        try:
            return llm.chat_json(msgs, temperature=0.0, max_tokens=2000, tag="kg_glean")
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Graph RAG 问答
# ---------------------------------------------------------------------------
GRAPH_RAG_SYS = """你是金融知识图谱问答助手。你会收到【知识图谱上下文】和可选的
【原文片段】，请据此回答问题。

规则：
1. 优先使用知识图谱中的关系与实体属性作答，图谱信息不足时再参考原文片段。
2. 数值必须精确，注意区分不同报告期与不同口径。
3. 如果图谱和原文都不足以回答，明确说明「知识库中未找到相关信息」。
4. 回答末尾标注信息来源（社区编号或原文章节）。"""


class GraphRAG:
    """
    Graph RAG 问答器：局部检索 + 全局检索融合。

    Args:
        kg: 已构建的知识图谱
        retriever: 可选的向量检索器（图谱检索结果不足时兜底）
        mode: local | global | hybrid
    """

    def __init__(self, kg: KnowledgeGraph, retriever=None, mode: str = "hybrid"):
        self.kg = kg
        self.retriever = retriever
        self.mode = mode

    def answer(self, question: str, top_k: int = 5) -> dict:
        import time
        t0 = time.perf_counter()

        contexts, trace = [], {}

        # 局部检索
        if self.mode in ("local", "hybrid"):
            local = self.kg.local_search(question)
            trace["local"] = {"seeds": local["seeds"],
                              "n_nodes": len(local["nodes"]),
                              "n_edges": len(local["edges"])}
            if local["context"]:
                contexts.append(local["context"])

        # 全局检索
        if self.mode in ("global", "hybrid"):
            glob = self.kg.global_search(question, top_k=3)
            trace["global"] = {"n_communities": len(glob["communities"])}
            if glob["context"]:
                contexts.append(glob["context"])

        # 向量检索兜底（工单08 验收要求「比对 07 工单的测试结果」，
        # 因此保留原文片段通道，图谱不足时可回退）
        if self.retriever is not None:
            docs = self.retriever.search(
                question, strategy="hybrid", top_k=top_k, reranker="tfidf")
            trace["vector"] = {"n_docs": len(docs)}
            if docs:
                ctx = self.retriever.format_context(docs, max_chars=2500)
                contexts.append(f"【原文片段】\n{ctx}")

        ctx_text = "\n\n" + ("=" * 60) + "\n\n".join(contexts) if contexts else ""
        t_ret = time.perf_counter()

        if not ctx_text.strip():
            return {"question": question,
                    "answer": "知识库中未找到相关信息。",
                    "contexts": [], "trace": trace, "latency": t_ret - t0}

        try:
            ans = llm.chat(
                [{"role": "system", "content": GRAPH_RAG_SYS},
                 {"role": "user", "content": f"{ctx_text}\n\n【问题】\n{question}"}],
                temperature=0.0, max_tokens=1500, tag="graph_rag",
            ).strip()
        except Exception as e:
            ans = f"生成失败：{e}"

        return {
            "question": question, "answer": ans, "contexts": contexts,
            "trace": trace, "latency": round(time.perf_counter() - t0, 3),
        }


# ---------------------------------------------------------------------------
# 构建流水线
# ---------------------------------------------------------------------------
def build_knowledge_graph(
    chunks: list,
    profile: str = "optimized",
    cache_path: Path | None = None,
    max_chunks: int | None = None,
    verbose: bool = True,
    resume: bool = True,
) -> KnowledgeGraph:
    """
    从文本块构建知识图谱（带增量缓存，支持断点续跑）。

    Args:
        chunks: Chunk 列表（来自 rag_core.chunk）
        profile: baseline | optimized
        resume: 存在缓存时跳过已抽取的块（图谱抽取很贵，必须可续跑）
    """
    cache_path = Path(cache_path or (GRAPH_DIR / f"extractions_{profile}.jsonl"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if resume and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["chunk_id"])
            except Exception:
                continue

    extractor = GraphExtractor(profile)
    kg = KnowledgeGraph()
    todo = [c for c in chunks if c.chunk_id not in done_ids]
    if max_chunks:
        todo = todo[:max_chunks]
    if verbose:
        print(f"图谱抽取：{len(chunks)} 块，已完成 {len(done_ids)}，待处理 {len(todo)}")

    # 先把已缓存的抽取结果灌回图谱
    if done_ids:
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
                kg.add_extraction(o["data"], o["chunk_id"])
            except Exception:
                continue

    with cache_path.open("a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            data = extractor.extract(c.text, c.chunk_id)
            kg.add_extraction(data, c.chunk_id)
            f.write(json.dumps({"chunk_id": c.chunk_id, "data": data},
                               ensure_ascii=False) + "\n")
            f.flush()
            if verbose and i % 10 == 0:
                print(f"  进度 {i}/{len(todo)}｜实体 {len(kg.entities)}｜"
                      f"关系 {len(kg.relations_merged)}")

    if verbose:
        s = kg.stats()
        print(f"图谱构建完成：{s['n_entities']} 实体，{s['n_relations']} 关系")
    return kg


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _norm_name(name: str) -> str:
    """实体名规范化：去空白、去首尾标点、统一全半角括号。"""
    if not name:
        return ""
    s = re.sub(r"\s+", "", str(name))
    s = s.strip("　 \t\n、，,。.；;：:（）()【】[]\"'“”")
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"^(本公司|公司|该行|本行|发行人|该公司)$", "", s)
    return s[:60]


def _rank_by_overlap(query: str, texts: dict, top_k: int) -> list[tuple]:
    """用字符 n-gram 重叠度对文本排序（轻量，无需向量模型）。"""
    def grams(s, n=2):
        return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}

    qg = grams(query)
    scored = []
    for k, t in texts.items():
        tg = grams(t or "")
        jac = len(qg & tg) / max(len(qg | tg), 1)
        scored.append((k, t, jac))
    scored.sort(key=lambda x: -x[2])
    return [(k, t) for k, t, _ in scored[:top_k]]


def _cn_font() -> str:
    """挑选可用的中文字体（matplotlib 默认字体不含中文，会显示方块）。"""
    import matplotlib.font_manager as fm
    for name in ("SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "Arial Unicode MS"):
        try:
            if any(f.name == name for f in fm.fontManager.ttflist):
                return name
        except Exception:
            continue
    return "sans-serif"
