# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：Graph RAG 检索（工单硬性要求2、3）

两部分：
  1. 实体识别与链接（本文件上半）—— 把用户问题里的实体落到图谱节点上；
  2. 子图检索（本文件下半，Task 14）—— 4 种模式取回结构化事实。

实体识别策略（分层，与图谱构建一致）：
  · 先用**词典**做最长匹配（零成本、零幻觉、可复现）；
  · 链接时优先精确匹配图谱节点；图谱中没有的实体按「别名命中」兜底。
LLM 兜底识别在 graph_qa 层按需启用（当词典一个实体都没命中时）。
"""
from __future__ import annotations

import threading
from collections import OrderedDict

from src.kg_dict import (COMPANY_TABLE, METRIC_TABLE, find_companies,
                         find_companies_en, find_metrics)
from src.kg_schema import make_entity_id
from src.kg_store import KnowledgeGraph

# 各匹配来源的置信度（用于排序）
SCORE_COMPANY_FULL = 1.00     # 公司全称
SCORE_COMPANY_SHORT = 0.95    # 公司简称
SCORE_COMPANY_ALIAS = 0.90    # 公司别名
SCORE_METRIC = 0.85           # 指标
SCORE_GRAPH_ALIAS = 0.80      # 图谱别名命中（兜底）
SCORE_GRAPH_FUZZY = 0.60      # 图谱名称子串命中（最弱）


class EntityLinkCache:
    """极简 LRU 缓存（线程安全）。

    用途：同一问题在多轮对话中被反复检索，实体链接结果可复用；
    也支撑工单「响应时间 ≤ 3 秒」的目标。
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._d: OrderedDict = OrderedDict()
        self._max = maxsize
        self._lock = threading.RLock()

    def get(self, key: str):
        with self._lock:
            if key not in self._d:
                return None
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key: str, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


_link_cache = EntityLinkCache()


def _graph_has_metric(kg: KnowledgeGraph, canonical: str) -> bool:
    """图谱里是否真实存在该指标的取值节点。

    图谱的指标节点带取值与期间（如 `Metric:不良贷款率(1.65%)@2019年末`），
    与词典规范名不同名，故按 attrs["metric"] 判定存在性。
    存在的才链接，避免返回图谱中并不存在的「幻觉实体」。
    """
    return any(d.get("type") == "Metric"
               and d.get("attrs", {}).get("metric") == canonical
               for _n, d in kg.g.nodes(data=True))


def _company_score(surface: str, canonical: str) -> float:
    meta = COMPANY_TABLE.get(canonical, {})
    if surface == meta.get("full"):
        return SCORE_COMPANY_FULL
    if surface == canonical:
        return SCORE_COMPANY_SHORT
    return SCORE_COMPANY_ALIAS


def link_entities(question: str, kg: KnowledgeGraph,
                  use_cache: bool = True) -> list[dict]:
    """把问题中的实体链接到图谱节点。

    返回 [{"entity_id", "name", "type", "score", "surface"}]，按 score 降序。
    """
    if not question or not question.strip():
        return []

    if use_cache:
        cached = _link_cache.get(question)
        if cached is not None:
            return cached

    found: dict[str, dict] = {}

    # 1) 公司（词典最长匹配）—— 中文名 + 英文名（多语言支持）
    for canonical, start, end in (list(find_companies(question))
                                  + list(find_companies_en(question))):
        surface = question[start:end]
        eid = make_entity_id("Company", canonical)
        score = _company_score(surface, canonical)
        if eid not in found or found[eid]["score"] < score:
            found[eid] = {"entity_id": eid, "name": canonical, "type": "Company",
                          "score": score, "surface": surface}

    # 2) 指标（词典）：只链接图谱中真实存在取值的指标
    for canonical, surface, _s, _e in find_metrics(question):
        eid = make_entity_id("Metric", canonical)
        if not (kg.has_entity(eid) or _graph_has_metric(kg, canonical)):
            continue
        found.setdefault(eid, {"entity_id": eid, "name": canonical,
                               "type": "Metric", "score": SCORE_METRIC,
                               "surface": surface})

    # 3) 图谱别名兜底（词典没覆盖的公司/部门/人物等）
    for eid in kg.find_by_alias(question):
        if eid in found:
            continue
        d = kg.g.nodes[eid]
        if d.get("type") == "Metric":
            continue                    # 指标节点名带数值，不做子串匹配
        found[eid] = {"entity_id": eid, "name": d.get("name", eid),
                      "type": d.get("type", "?"), "score": SCORE_GRAPH_ALIAS,
                      "surface": d.get("name", eid)}

    out = sorted(found.values(), key=lambda x: (-x["score"], x["entity_id"]))
    if use_cache:
        _link_cache.put(question, out)
    return out


def clear_link_cache() -> None:
    _link_cache.clear()


# ---------------------------------------------------------------------------
# 子图检索（4 种模式）
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field          # noqa: E402

from src import config                            # noqa: E402
from src.kg_schema import node_color              # noqa: E402

ORG_KEYWORDS = ("组织结构", "组织架构", "板块", "部门", "下设", "内设",
                "直属单位", "隶属")
COMPARE_KEYWORDS = ("对比", "比较", "谁更", "哪个更", "哪家更", "相差", "vs", "VS")
RELATION_KEYWORDS = ("什么关系", "关系是", "持股", "控股", "股东", "参股", "投资")

# 可作为子图检索种子的实体类型（按信息量排序）：
# Industry/Time 等类型多由「图谱别名兜底」引入（如「银行」命中平安银行里的“银行”），
# 是低分噪声，不能作为种子驱动检索 —— 否则子图会被无关的行业/年份边淹没。
SEED_TYPES = ("Company", "Department", "Metric")


def _seed_ids(linked: list[dict], types: tuple = SEED_TYPES,
              limit: int = 3) -> list[str]:
    """按 score 降序选种子实体，且只保留承载结构化事实的类型。

    注意：linked 可能是缓存对象（link_entities 按引用返回），
    这里用 sorted 生成新列表，绝不原地排序/修改。
    """
    cands = [x for x in linked if x.get("type") in types]
    cands = sorted(cands, key=lambda x: -float(x.get("score", 0.0)))
    return [x["entity_id"] for x in cands[:limit]]


def detect_mode(question: str, linked: list[dict]) -> str:
    """判定用哪种子图检索模式。

    优先级：metric_compare > org_tree > path > khop
    （对比/组织类问法一旦命中，其结构约束比通用 k-hop 更精确）
    """
    if not linked:
        return "none"

    companies = {x["name"] for x in linked if x["type"] == "Company"}
    has_metric = any(x["type"] == "Metric" for x in linked)
    metric_words = any(k in question for k in METRIC_TABLE)

    if len(companies) >= 2 and (has_metric or metric_words):
        return "metric_compare"

    if any(k in question for k in ORG_KEYWORDS):
        return "org_tree"

    if (len(companies) >= 2 or len(linked) >= 2) and \
            any(k in question for k in RELATION_KEYWORDS):
        return "path"

    return "khop"


@dataclass
class SubgraphResult:
    """子图检索结果。

    facts        —— 结构化事实（自然语言，供 LLM 上下文使用）；
    nodes/edges  —— 子图数据（供可视化）；
    source_pages —— provenance 汇总：[{source_doc, page, evidence}]，
                    支撑工单「返回答案所在的文本块」。
    """
    mode: str = "none"
    facts: list[str] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    source_pages: list[dict] = field(default_factory=list)

    def to_html_data(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges, "mode": self.mode}


def _collect_provenance(kg: KnowledgeGraph, node_ids: list[str]) -> list[dict]:
    """汇总子图内所有边的 provenance（去重）。"""
    keep = set(node_ids)
    seen: set[tuple] = set()
    out: list[dict] = []
    for u, v, d in kg.g.edges(data=True):
        if u not in keep or v not in keep:
            continue
        key = (d.get("source_doc"), int(d.get("page", -1)),
               (d.get("evidence") or "")[:50])
        if key in seen or not d.get("source_doc"):
            continue
        seen.add(key)
        out.append({"source_doc": d.get("source_doc"),
                    "page": int(d.get("page", -1)),
                    "evidence": d.get("evidence", "")})
    return out


def _node_entry(kg: KnowledgeGraph, nid: str) -> dict:
    d = kg.g.nodes[nid]
    label = d.get("name", nid)
    if d.get("type") == "Metric":
        attrs = d.get("attrs", {})
        if attrs.get("value") is not None:
            label = f"{attrs.get('metric', label)} {attrs['value']}{attrs.get('unit', '')}"
    return {"id": nid, "label": label, "type": d.get("type", "?"),
            "color": node_color(d.get("type", "?")),
            "attrs": dict(d.get("attrs", {}))}


def _edges_of(kg: KnowledgeGraph, node_ids: set[str]) -> list[dict]:
    out: list[dict] = []
    for u, v, d in kg.g.edges(data=True):
        if u in node_ids and v in node_ids:
            out.append({"from": u, "to": v, "label": d.get("rel_type", ""),
                        "page": int(d.get("page", -1)),
                        "source_doc": d.get("source_doc", ""),
                        "evidence": d.get("evidence", "")})
    return out


def _metric_compare_facts(kg: KnowledgeGraph, linked: list[dict]) -> SubgraphResult:
    """结构化指标对比：同名指标跨公司取数（仅限问题链接到的公司）。"""
    sub = SubgraphResult(mode="metric_compare")
    metrics = [x["name"] for x in linked if x["type"] == "Metric"]
    if not metrics:
        metrics = list(METRIC_TABLE.keys())

    # 对比范围 = 问题链接到的公司。未提及的公司不得进入事实区，
    # 否则 LLM 会把范围外的取值当作对比对象（实测「平安 vs 招商」答成邮储最低）。
    companies = [x["name"] for x in linked if x["type"] == "Company"]

    node_ids: set[str] = set()
    for linked_item in linked:
        if linked_item["type"] == "Company":
            node_ids.add(linked_item["entity_id"])

    for metric in metrics[:3]:
        rows = kg.metric_compare(metric, companies=companies or None)
        if len(rows) < 2:
            continue
        for r in rows:
            val = r["value"]
            unit = r["unit"] or ""
            yoy = f"，同比 {r['yoy']}%" if r.get("yoy") is not None else ""
            sub.facts.append(
                f"{r['company']}{r['period'] or ''}的{metric}为 {val}{unit}{yoy}"
                f"（来源：{r['source_doc']} 第 {r['page']} 页）")
            if r.get("source_doc"):
                sub.source_pages.append({"source_doc": r["source_doc"],
                                         "page": r["page"],
                                         "evidence": r.get("evidence", "")})
            # 把涉及的公司与指标节点纳入可视化子图
            for nid in kg.g.nodes:
                d = kg.g.nodes[nid]
                if d.get("type") == "Company" and d.get("name") == r["company"]:
                    node_ids.add(nid)
                if d.get("type") == "Metric" and \
                        d.get("attrs", {}).get("metric") == metric and \
                        d.get("attrs", {}).get("value") == val and \
                        d.get("attrs", {}).get("period") == r.get("period"):
                    node_ids.add(nid)

    # 数值比较结论
    for metric in metrics[:3]:
        rows = [r for r in kg.metric_compare(metric, companies=companies or None)
                if r["value"] is not None]
        if len(rows) >= 2:
            lo = min(rows, key=lambda r: r["value"])
            hi = max(rows, key=lambda r: r["value"])
            sub.facts.append(
                f"比较结论：{metric} 中 {hi['company']} 最高（{hi['value']}{hi['unit']}），"
                f"{lo['company']} 最低（{lo['value']}{lo['unit']}）。")

    sub.nodes = [_node_entry(kg, n) for n in node_ids]
    sub.edges = _edges_of(kg, node_ids)
    return sub


def _org_tree_facts(kg: KnowledgeGraph, linked: list[dict]) -> SubgraphResult:
    """组织结构树：Company -BELONGS_TO-> Department 的子树。"""
    sub = SubgraphResult(mode="org_tree")
    # 只以公司/部门为根（按 score 降序），Industry/Time 等兜底实体不作为根
    roots = _seed_ids(linked, types=("Company", "Department"), limit=2)
    if not roots:
        return sub

    node_ids: set[str] = set()
    for root in roots:
        tree = kg.subtree(root, "BELONGS_TO", max_nodes=config.KG_SUBGRAPH_MAX_NODES * 2)
        by_parent: dict[str, list[str]] = {}
        for u, v, d in tree.g.edges(data=True):
            # 本体中 BELONGS_TO 的方向是「子 → 父」（部门指向所属组织），
            # 因此 u 是子、v 是父；写反会把子部门当父节点陈述（事实行颠倒）。
            by_parent.setdefault(v, []).append(u)
        for parent, children in by_parent.items():
            pname = kg.g.nodes[parent].get("name", parent)
            cnames = [kg.g.nodes[c].get("name", c) for c in children]
            sub.facts.append(f"{pname} 下设：{'、'.join(sorted(cnames))}。")
            node_ids.add(parent)
            node_ids.update(children)

    sub.nodes = [_node_entry(kg, n) for n in node_ids]
    sub.edges = _edges_of(kg, node_ids)
    sub.source_pages = _collect_provenance(kg, list(node_ids))
    return sub


def _path_facts(kg: KnowledgeGraph, linked: list[dict]) -> SubgraphResult:
    """实体间最短关系链。"""
    sub = SubgraphResult(mode="path")
    ids = _seed_ids(linked, limit=4)      # 按分数选种子，过滤 Industry/Time 噪声
    node_ids: set[str] = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            path = kg.shortest_path(a, b)
            if not path:
                continue
            names = [kg.g.nodes[n].get("name", n) for n in path]
            node_ids.update(path)
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                # shortest_path 在无向视图上求解，边可能按反方向存储：
                # 直接 kg.g[u][v] 会 KeyError，且事实行必须按边的真实方向陈述。
                fwd = kg.g.get_edge_data(u, v)
                if fwd:
                    rels = "/".join(str(d.get("rel_type")) for d in fwd.values())
                    sub.facts.append(f"{names[j]} --{rels}--> {names[j + 1]}")
                else:
                    rev = kg.g.get_edge_data(v, u) or {}
                    rels = "/".join(str(d.get("rel_type")) for d in rev.values())
                    sub.facts.append(f"{names[j + 1]} --{rels}--> {names[j]}")
            sub.facts.append(f"关系链：{' → '.join(names)}")
    if not node_ids:                       # 无路径则退回 k-hop
        return _khop_facts(kg, linked)
    sub.nodes = [_node_entry(kg, n) for n in node_ids]
    sub.edges = _edges_of(kg, node_ids)
    sub.source_pages = _collect_provenance(kg, list(node_ids))
    return sub


def _khop_facts(kg: KnowledgeGraph, linked: list[dict],
                max_nodes: int | None = None) -> SubgraphResult:
    """默认模式：k-hop 邻域事实。"""
    max_nodes = max_nodes or config.KG_SUBGRAPH_MAX_NODES
    sub = SubgraphResult(mode="khop")
    # 种子优先取公司/部门（Metric 是叶子，以它为心检索价值低）；
    # 全部按 score 降序，避免别名兜底的低分噪声（Industry:银行/Time:2019年）主导检索。
    seeds = _seed_ids(linked, types=("Company", "Department"))
    if not seeds:
        seeds = _seed_ids(linked)          # 兜底：允许 Metric 作为种子
    if not seeds:
        return sub

    node_ids: set[str] = set()
    for sid in seeds:
        if not kg.has_entity(sid):
            continue
        nb = kg.neighbors(sid, hops=config.KG_SUBGRAPH_HOPS,
                          max_nodes=max_nodes // max(len(seeds), 1))
        node_ids.update(nb.g.nodes)

    for u, v, d in kg.g.edges(data=True):
        if u not in node_ids or v not in node_ids:
            continue
        un = kg.g.nodes[u].get("name", u)
        vn = kg.g.nodes[v].get("name", v)
        rel = d.get("rel_type", "")
        ev = (d.get("evidence") or "")[:60]
        page = int(d.get("page", -1))
        doc = d.get("source_doc", "")
        sub.facts.append(f"{un} --{rel}--> {vn}"
                         + (f"（依据：{ev}…，{doc} 第 {page} 页）" if ev else ""))

    sub.nodes = [_node_entry(kg, n) for n in node_ids]
    sub.edges = _edges_of(kg, node_ids)
    sub.source_pages = _collect_provenance(kg, list(node_ids))
    return sub


def retrieve_subgraph(question: str, linked: list[dict], kg: KnowledgeGraph,
                      max_nodes: int | None = None) -> SubgraphResult:
    """按 detect_mode 选定的模式做子图检索。"""
    mode = detect_mode(question, linked)
    if mode == "none":
        return SubgraphResult(mode="none")

    if mode == "metric_compare":
        sub = _metric_compare_facts(kg, linked)
    elif mode == "org_tree":
        sub = _org_tree_facts(kg, linked)
    elif mode == "path":
        sub = _path_facts(kg, linked)
    else:
        sub = _khop_facts(kg, linked, max_nodes)

    # 事实条数上限（防上下文爆炸）：按检索顺序截断，模式内的顺序即优先级
    # （指标对比的事实/结论、路径链等按语义顺序生成，直接截尾即丢弃最低优先者）。
    limit = max_nodes or config.KG_SUBGRAPH_MAX_NODES
    if len(sub.facts) > limit:
        sub.facts = sub.facts[:limit]
    return sub
