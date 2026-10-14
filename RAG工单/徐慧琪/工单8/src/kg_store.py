# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：KnowledgeGraph —— NetworkX 图谱存储与查询（硬性要求3「图数据库」）

选型说明（工单要求「推荐 Neo4j 或 NetworkX；若用 NetworkX，需支持持久化与查询」）：
本机无 Java 与 Neo4j 服务端，且已确认「零下载」策略，故采用 NetworkX。
持久化三件套：
  · kg.pkl      —— pickle，加载最快，供运行时使用；
  · kg.graphml  —— 可读文本，供外部工具（Gephi/yEd）查看与人工核验；
  · kg_*.jsonl  —— 实体/关系明细，供审计与 diff。

并发说明：图在**离线构建期**写入，构建完成后转为**只读**，
在线查询不加锁（工单「高并发稳定运行」要求）。
"""
from __future__ import annotations

import io
import json
import os
import pickle

import networkx as nx

from src import config
from src.kg_schema import (ENTITY_TYPES, Entity, Relation, node_color,
                           validate_entity, validate_relation)

# 确定性边（confidence >= 1.0）的来源累积上限：热门事实（如「某公司属于某行业」）
# 可在上千个块中被提及，超过上限即停止累积，避免 attrs 无界膨胀；
# 边顶层的 source_doc/page/evidence 始终保留首次出现，单来源读者不受影响。
SOURCES_CAP = 8


def _provenance(r: Relation) -> dict:
    """把边的顶层 provenance 三元组转成来源记录。"""
    return {"source_doc": r.source_doc, "page": r.page, "evidence": r.evidence}


def _same_prov(a: dict, b: dict) -> bool:
    """来源判等：沿用原去重键的口径（evidence 只比前 60 字符）。"""
    return (a.get("source_doc"), a.get("page"),
            (a.get("evidence") or "")[:60]) == \
           (b.get("source_doc"), b.get("page"),
            (b.get("evidence") or "")[:60])


def _accumulate_sources(dst: list, incoming: list) -> None:
    """把 incoming 中未出现过的来源追加进 dst，最多 SOURCES_CAP 条。"""
    for p in incoming:
        if len(dst) >= SOURCES_CAP:
            return
        if not any(_same_prov(p, s) for s in dst):
            dst.append(p)


def _graphml_safe(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """返回供 GraphML 导出的**副本**：list/dict 属性转 JSON 字符串。

    GraphML 只支持 str/int/float/bool 等基础类型，而本图的 `aliases`（list）、
    `attrs`（dict）都是复合类型 —— 直接 write_graphml 会抛
    `TypeError: GraphML does not support type <class 'list'> as data values.`。
    这里仅对副本做无损降级（JSON 字符串），原图（pickle 用）保持原生类型。
    """
    def conv(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return json.dumps(v, ensure_ascii=False)

    h = g.copy()
    for _n, d in h.nodes(data=True):
        for k in list(d):
            d[k] = conv(d[k])
    for _u, _v, d in h.edges(data=True):
        for k in list(d):
            d[k] = conv(d[k])
    return h


class KnowledgeGraph:
    """金融知识图谱（MultiDiGraph：允许同一对节点间存在多种关系）。"""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def add_entity(self, e: Entity) -> str:
        """加入实体。非法实体抛 ValueError；已存在则合并别名与属性。"""
        errors = validate_entity(e)
        if errors:
            raise ValueError(f"非法实体 {e.entity_id!r}: {errors}")

        if self.g.has_node(e.entity_id):
            data = self.g.nodes[e.entity_id]
            for a in e.aliases:
                if a not in data["aliases"]:
                    data["aliases"].append(a)
            for k, v in e.attrs.items():
                if k not in data["attrs"] or data["attrs"][k] in (None, "", []):
                    if v not in (None, "", []):
                        data["attrs"][k] = v
            return e.entity_id

        self.g.add_node(e.entity_id, name=e.name, type=e.type,
                        aliases=list(e.aliases), attrs=dict(e.attrs),
                        color=node_color(e.type))
        return e.entity_id

    def add_relation(self, r: Relation) -> bool:
        """加入关系。非法、缺端点或完全重复返回 False。

        去重口径按边的确定性分两类：
          · 确定性边（confidence >= 1.0，规则/词典层）：(head, rel_type, tail)
            即一条客观事实，只保留一条边；同一事实的后续提及不建新边，而是把
            provenance 累积进已有边的 attrs["sources"]（去重，上限 SOURCES_CAP）。
            否则「平安银行 属于 银行」会随语料提及数膨胀成上千条重复边。
          · LLM 边（confidence < 1.0）：证据逐次抽取而不同、彼此独立，
            保持原 (source_doc, page, evidence[:60]) 逐条去重。
        """
        if validate_relation(r):
            return False
        if not (self.g.has_node(r.head_id) and self.g.has_node(r.tail_id)):
            return False

        deterministic = float(r.confidence) >= 1.0
        # 本边携带的全部来源：自身 provenance + 调用方随 attrs 带入的 sources
        # （merge()/子图拷贝会把已累积的来源一并传回，不能丢）。
        carried = [s for s in (r.attrs.get("sources") or [])
                   if isinstance(s, dict)]
        incoming = [_provenance(r)] + carried

        # 注：头尾节点存在不代表二者间已有边（首次为某对节点建边时），
        # 因此必须用 get_edge_data(default={}) 取邻接，直接 g[u][v] 会 KeyError。
        for _k, data in self.g.get_edge_data(r.head_id, r.tail_id,
                                             default={}).items():
            if data.get("rel_type") != r.rel_type:
                continue
            if deterministic:
                if float(data.get("confidence", 1.0)) < 1.0:
                    continue          # 不把确定性事实并入 LLM 边
                # 命中同一事实：只累积来源，不建新边。
                sources = data.setdefault("attrs", {}).setdefault("sources", [])
                if not sources:       # 兼容修复前建的老图/老 pickle：补入本边来源
                    sources.append({"source_doc": data.get("source_doc"),
                                    "page": data.get("page"),
                                    "evidence": data.get("evidence")})
                _accumulate_sources(sources, incoming)
                return False
            if (data.get("source_doc"), data.get("page"),
                    (data.get("evidence") or "")[:60]) == \
                    (r.source_doc, r.page, (r.evidence or "")[:60]):
                return False

        key = (r.head_id, r.tail_id, r.rel_type, r.source_doc, r.page,
               (r.evidence or "")[:60])
        attrs = dict(r.attrs)
        if deterministic:
            sources: list[dict] = []
            _accumulate_sources(sources, incoming)
            attrs["sources"] = sources
        self.g.add_edge(r.head_id, r.tail_id, key=str(abs(hash(key))),
                        rel_type=r.rel_type, evidence=r.evidence,
                        source_doc=r.source_doc, page=r.page,
                        confidence=r.confidence, attrs=attrs)
        return True

    def merge(self, other: "KnowledgeGraph") -> dict:
        """合并另一张图（增量构建 / 多文档融合）。返回统计。"""
        n_e = n_r = 0
        for nid, data in other.g.nodes(data=True):
            self.add_entity(Entity(entity_id=nid, name=data["name"],
                                   type=data["type"],
                                   aliases=list(data.get("aliases", [])),
                                   attrs=dict(data.get("attrs", {}))))
            n_e += 1
        for u, v, data in other.g.edges(data=True):
            ok = self.add_relation(Relation(
                head_id=u, rel_type=data["rel_type"], tail_id=v,
                evidence=data.get("evidence", ""),
                source_doc=data.get("source_doc", ""),
                page=int(data.get("page", -1)),
                confidence=float(data.get("confidence", 1.0)),
                attrs=dict(data.get("attrs", {}))))
            n_r += int(ok)
        return {"added_nodes": n_e, "added_edges": n_r}

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def has_entity(self, eid: str) -> bool:
        return self.g.has_node(eid)

    def get_entity(self, eid: str) -> Entity | None:
        if not self.g.has_node(eid):
            return None
        d = self.g.nodes[eid]
        return Entity(entity_id=eid, name=d["name"], type=d["type"],
                      aliases=list(d.get("aliases", [])),
                      attrs=dict(d.get("attrs", {})))

    def entity_ids(self) -> list[str]:
        return list(self.g.nodes)

    def nodes_of_type(self, etype: str) -> list[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == etype]

    def find_by_alias(self, text: str) -> list[str]:
        """别名字面命中：text 中出现某实体的 name 或 alias 即算命中。"""
        probe = (text or "").strip()
        if not probe:
            return []
        hits: list[str] = []
        for nid, d in self.g.nodes(data=True):
            names = [d.get("name", "")] + list(d.get("aliases", []))
            if any(nm and nm in probe for nm in names):
                hits.append(nid)
        return hits

    # ------------------------------------------------------------------
    # 子图检索
    # ------------------------------------------------------------------
    def neighbors(self, eid: str, hops: int = 2, rel_types: list[str] | None = None,
                  max_nodes: int = 60) -> "KnowledgeGraph":
        """k-hop 邻居子图（无向遍历，因为关系有方向但查询是双向的）。

        裁剪：节点超过 max_nodes 时，按「与中心的最短距离升序 + 度数降序」截断，
        保证中心附近的高价值节点优先保留。
        """
        if not self.g.has_node(eid):
            return KnowledgeGraph()

        ug = self.g.to_undirected(as_view=True)
        reached = nx.single_source_shortest_path_length(ug, eid, cutoff=hops)
        cands = list(reached.keys())

        if len(cands) > max_nodes:
            cands.sort(key=lambda n: (reached[n], -self.g.degree(n), n))
            cands = cands[:max_nodes]
        keep = set(cands)

        out = KnowledgeGraph()
        for n in keep:
            d = self.g.nodes[n]
            out.add_entity(Entity(n, d["name"], d["type"],
                                  list(d.get("aliases", [])),
                                  dict(d.get("attrs", {}))))
        for u, v, d in self.g.edges(data=True):
            if u in keep and v in keep:
                if rel_types and d.get("rel_type") not in rel_types:
                    continue
                out.add_relation(Relation(
                    u, d["rel_type"], v, d.get("evidence", ""),
                    d.get("source_doc", ""), int(d.get("page", -1)),
                    float(d.get("confidence", 1.0)), dict(d.get("attrs", {}))))
        return out

    def shortest_path(self, a: str, b: str, max_hops: int = 4) -> list[str] | None:
        """两实体间最短关系链（无向），超过 max_hops 返回 None。"""
        if not (self.g.has_node(a) and self.g.has_node(b)):
            return None
        ug = self.g.to_undirected(as_view=True)
        try:
            path = nx.shortest_path(ug, a, b)
        except nx.NetworkXNoPath:
            return None
        return path if len(path) - 1 <= max_hops else None

    def subtree(self, root: str, rel_type: str = "BELONGS_TO",
                max_nodes: int = 100) -> "KnowledgeGraph":
        """沿指定关系的子树（用于组织结构图：公司 → 板块 → 部门）。

        方向说明：本体中 BELONGS_TO 由子指向父（部门 → 公司），
        因此「从根向下展开」沿入边（前驱）遍历。
        """
        if not self.g.has_node(root):
            return KnowledgeGraph()
        keep = {root}
        frontier = [root]
        while frontier and len(keep) < max_nodes:
            nxt: list[str] = []
            for n in frontier:
                # 本体中 BELONGS_TO 的方向是「部门 → 所属公司」，子节点指向父节点，
                # 故从根（公司）向下展开要走**入边**（前驱），而不是出边。
                for u, _v, d in self.g.in_edges(n, data=True):
                    # 硬上限：每加一个子节点前都检查，单层大扇出（如根挂 300 个子节点）
                    # 也不会越过 max_nodes —— 这是子图检索层约束上下文大小的依据。
                    if len(keep) >= max_nodes:
                        break
                    if d.get("rel_type") != rel_type or u in keep:
                        continue
                    keep.add(u)
                    nxt.append(u)
                if len(keep) >= max_nodes:
                    break
            frontier = nxt
            if len(keep) >= max_nodes:
                break

        out = KnowledgeGraph()
        for n in keep:
            d = self.g.nodes[n]
            out.add_entity(Entity(n, d["name"], d["type"],
                                  list(d.get("aliases", [])),
                                  dict(d.get("attrs", {}))))
        for u, v, d in self.g.edges(data=True):
            if u in keep and v in keep and d.get("rel_type") == rel_type:
                out.add_relation(Relation(
                    u, d["rel_type"], v, d.get("evidence", ""),
                    d.get("source_doc", ""), int(d.get("page", -1)),
                    float(d.get("confidence", 1.0)), dict(d.get("attrs", {}))))
        return out

    def metric_compare(self, metric: str, periods: list[str] | None = None,
                       companies: list[str] | None = None,
                       period: str | list[str] | None = None) -> list[dict]:
        """结构化指标对比：同名指标，跨公司/跨期取数。

        这是 Graph RAG 相对普通 RAG 的核心增量能力 ——
        07 需要向量检索分别召回两份文档再让 LLM 比较；
        08 可直接遍历 `Company -HAS_METRIC-> Metric` 边拿到确定性结果。

        period / periods 等价入参（period 为单个字符串的便捷写法）。
        """
        if period is not None:
            periods = [period] if isinstance(period, str) else list(period)
        rows: list[dict] = []
        for nid, d in self.g.nodes(data=True):
            if d.get("type") != "Metric":
                continue
            attrs = d.get("attrs", {})
            if attrs.get("metric") != metric:
                continue
            if periods and attrs.get("period") not in periods:
                continue
            # in_edges 产出 (源, 目标, 数据)，且目标是当前 Metric 节点，
            # 故公司是**源**（即此处的 company），不能把第二个元素当公司。
            for company, _metric, ed in self.g.in_edges(nid, data=True):
                if ed.get("rel_type") != "HAS_METRIC":
                    continue
                cname = self.g.nodes[company].get("name", company)
                if companies and cname not in companies:
                    continue
                rows.append({
                    "company": cname, "metric": metric,
                    "value": attrs.get("value"), "unit": attrs.get("unit"),
                    "period": attrs.get("period"), "yoy": attrs.get("yoy"),
                    "source_doc": attrs.get("source_doc"),
                    "page": attrs.get("page"),
                    "evidence": ed.get("evidence", ""),
                })
        rows.sort(key=lambda r: (r["period"] or "", r["company"]))
        return rows

    # ------------------------------------------------------------------
    # 统计 / 持久化
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for _n, d in self.g.nodes(data=True):
            by_type[d.get("type", "?")] = by_type.get(d.get("type", "?"), 0) + 1
        by_rel: dict[str, int] = {}
        for _u, _v, d in self.g.edges(data=True):
            by_rel[d.get("rel_type", "?")] = by_rel.get(d.get("rel_type", "?"), 0) + 1
        with_page = sum(1 for _u, _v, d in self.g.edges(data=True)
                        if int(d.get("page", -1)) >= 0)
        return {
            "n_nodes": self.g.number_of_nodes(),
            "n_edges": self.g.number_of_edges(),
            "by_type": by_type, "by_relation": by_rel,
            "edges_with_provenance": with_page,
            "n_docs": len({d.get("source_doc") for _u, _v, d in self.g.edges(data=True)
                           if d.get("source_doc")}),
        }

    def save(self, path_pickle: str | None = None,
             path_graphml: str | None = None) -> dict:
        """原子写：先写临时文件再 rename，避免中断留下半截文件。"""
        path_pickle = path_pickle or config.KG_PICKLE
        path_graphml = path_graphml or config.KG_GRAPHML
        os.makedirs(os.path.dirname(path_pickle), exist_ok=True)

        tmp = path_pickle + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.g, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path_pickle)

        tmp_g = path_graphml + ".tmp"
        try:
            # 复杂属性（aliases/attrs）经 _graphml_safe 转 JSON 字符串后写入副本
            nx.write_graphml(_graphml_safe(self.g), tmp_g)
            os.replace(tmp_g, path_graphml)
        except Exception:                       # noqa: BLE001
            # 兜底：GraphML 仍写不出时清掉临时文件，不影响已成功的 pickle
            # （原子写保证不会留下半截 .graphml）
            if os.path.exists(tmp_g):
                os.remove(tmp_g)

        return {"pickle": path_pickle, "graphml": path_graphml,
                "stats": self.stats()}

    @classmethod
    def load(cls, path_pickle: str | None = None) -> "KnowledgeGraph":
        path_pickle = path_pickle or config.KG_PICKLE
        if not os.path.isfile(path_pickle):
            raise FileNotFoundError(
                f"图谱不存在：{path_pickle}\n请先运行 python build_kg.py")
        obj = cls()
        with open(path_pickle, "rb") as f:
            obj.g = pickle.load(f)
        return obj

    def dump_jsonl(self, path_entities: str | None = None,
                   path_relations: str | None = None) -> dict:
        """导出实体/关系明细 JSONL（审计与 diff 用）。"""
        path_entities = path_entities or config.KG_ENTITIES_JSONL
        path_relations = path_relations or config.KG_RELATIONS_JSONL
        with io.open(path_entities, "w", encoding="utf-8") as f:
            for nid, d in self.g.nodes(data=True):
                f.write(json.dumps({"entity_id": nid, **{
                    k: d[k] for k in ("name", "type", "aliases", "attrs")
                    if k in d}}, ensure_ascii=False) + "\n")
        with io.open(path_relations, "w", encoding="utf-8") as f:
            for u, v, d in self.g.edges(data=True):
                f.write(json.dumps({"head_id": u, "tail_id": v, **{
                    k: d[k] for k in ("rel_type", "evidence", "source_doc",
                                      "page", "confidence", "attrs")
                    if k in d}}, ensure_ascii=False) + "\n")
        return {"entities": path_entities, "relations": path_relations}
