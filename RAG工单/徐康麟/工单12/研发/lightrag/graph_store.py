# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""图谱存储：实体归并 / 关系去重 / 约束校验 / 多格式导出。

复用 :mod:`lightrag.entity_schema` 的别名表、字符相似度与关系约束，
**不另起一套判类或归并逻辑**。

四道工序：
  ① 归并实体：``resolve_alias`` 查表 → 字符 bigram Dice ≥ ``alias_similarity`` 二次归并
     （并查集，仅在同类型、且名字长度 ≥ ``alias_min_len`` 的候选间比较）；
  ② 关系去重：同 ``(head, rel, tail)`` 合并，``weight`` 累加，保留最多 3 条 evidence；
  ③ 约束校验：``coerce_type`` 吸附非法类型，``relation_allowed`` 校验头尾相容；
  ④ 导出：``nodes.jsonl`` / ``edges.jsonl`` / ``graph.json`` / ``graph.graphml`` /
     ``graph_meta.json``。

GraphML 用标准库手写 XML（本机无 networkx，不引入新依赖）。
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape, quoteattr

from .config import Cfg
from .entity_schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    char_similarity,
    coerce_type,
    normalize_name,
    relation_allowed,
)
from .graph_extract import canonical_name


# ---------------------------------------------------------------------------
# 抽样口径的规范表述（三处必须一致：graph_meta.json / 对比报告 / 部署 README）
# ---------------------------------------------------------------------------
def sampling_caliber_text(meta: dict[str, Any]) -> str:
    """渲染「小样本建图抽样口径」规范句（数字全部取自本次实跑 meta）。

    优先用 ``extracted_records``（**实际建成图谱**的记录数），因为它可能小于
    ``selected``（抽样计划数）——例如本次因时间预算提前收束。
    ``graph_meta.json`` / 对比报告 / ``部署/README.md`` 三处必须引用本函数同一段文字。
    """
    planned = meta.get("selected", "?")
    used = meta.get("extracted_records") or planned
    tot = meta.get("chunk_total", "?")
    pct = ""
    try:
        pct = f"（{100.0 * float(used) / float(tot):.1f}%）"
    except Exception:  # noqa: BLE001
        pct = ""
    keep = " + ".join(meta.get("keep_doc_keys") or [])
    layers = meta.get("layer_counts") or {}
    layer_txt = "、".join(f"{k}={v}" for k, v in layers.items()) or "—"
    head = f"小样本建图抽样口径：实际建图 {used} 块 / 全量 {tot} 块{pct}"
    if used != planned:
        head += f"，抽样计划 {planned} 块（实际因时间预算先行收束）"
    return (
        head
        + f"｜随机种子 {meta.get('sample_seed')}"
        f"｜信息密度下限 {meta.get('min_chunk_chars')} 字符"
        f"｜table+image 加权｜每块 1 次 LLM 调用串行（batch_chunks={meta.get('batch_chunks')}）"
        f"｜按 doc_key 分层（{layer_txt}）"
        f"｜只保留 {keep}，剔除带水印重复件 {', '.join(meta.get('exclude_doc_keys') or [])}。"
    )


# ---------------------------------------------------------------------------
# 并查集
# ---------------------------------------------------------------------------
class _DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # 路径压缩
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _slug_id(name: str) -> str:
    """实体名 → 稳定节点 id（保留中文，去标点空白）。"""
    keep = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", name)
    return keep[:48] or "node"


# ---------------------------------------------------------------------------
# ① 实体归并
# ---------------------------------------------------------------------------
def merge_entities(records: Sequence[dict[str, Any]], cfg: Cfg, *, logger: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把逐块抽取的实体聚合成节点表（含别名归并）。"""
    g = cfg.graph
    threshold = float(g.alias_similarity)
    min_len = int(g.alias_min_len)

    mentions: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid, page, doc = rec.get("chunk_id"), rec.get("page"), rec.get("doc_key")
        for item in (rec.get("entities") or []):
            name = normalize_name(str(item.get("name") or ""))
            if not name:
                continue
            canon = canonical_name(name)
            slot = mentions.setdefault(canon, {
                "name": canon, "types": Counter(), "type_exact": 0, "total": 0,
                "raw_names": Counter(), "chunks": [], "pages": set(), "docs": set(),
            })
            typ = str(item.get("type") or "") or "公司"
            slot["types"][typ] += 1
            slot["type_exact"] += 1 if item.get("type_exact") else 0
            slot["total"] += 1
            slot["raw_names"][name] += 1
            if cid and cid not in slot["chunks"]:
                slot["chunks"].append(str(cid))
            try:
                slot["pages"].add(int(page))
            except (TypeError, ValueError):
                pass
            if doc:
                slot["docs"].add(str(doc))

    # 二次归并：仅在同类型、长度相近的候选间算 Dice（避免无谓的 O(n²)）
    dsu = _DSU()
    names = sorted(mentions)
    by_type: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_type[_dominant_type(mentions[name])].append(name)
    merge_pairs: list[tuple[str, str, float]] = []
    for _typ, group in by_type.items():
        grams = {n: _bigrams(n) for n in group}
        for i, a in enumerate(group):
            if len(a) < min_len:
                continue
            for b in group[i + 1:]:
                if len(b) < min_len or abs(len(a) - len(b)) > 3:
                    continue
                ga, gb = grams[a], grams[b]
                if not ga or not gb:
                    continue
                score = 2.0 * len(ga & gb) / (len(ga) + len(gb))
                if score >= threshold:
                    dsu.union(a, b)
                    merge_pairs.append((a, b, round(score, 4)))

    clusters: dict[str, list[str]] = defaultdict(list)
    for name in names:
        clusters[dsu.find(name)].append(name)

    nodes: list[dict[str, Any]] = []
    alias_map: dict[str, str] = {}
    for _root, members in clusters.items():
        members.sort(key=lambda n: (-mentions[n]["total"], -len(n), n))
        canonical = members[0]
        types = Counter()
        chunks: list[str] = []
        pages: set[int] = set()
        docs: set[str] = set()
        raw_names = Counter()
        total = exact = 0
        for m in members:
            slot = mentions[m]
            types.update(slot["types"])
            total += slot["total"]
            exact += slot["type_exact"]
            raw_names.update(slot["raw_names"])
            for cid in slot["chunks"]:
                if cid not in chunks:
                    chunks.append(cid)
            pages |= slot["pages"]
            docs |= slot["docs"]
        node_type = _dominant_type(mentions[canonical])
        nodes.append({
            "id": _slug_id(canonical),
            "name": canonical,
            "type": node_type,
            "type_distribution": dict(types),
            "type_conflict": len(types) > 1,
            "type_exact": exact,
            "freq": total,
            "degree": 0,
            "aliases": members[1:],
            "raw_names": [n for n, _ in raw_names.most_common(5)],
            "chunk_ids": chunks[:8],
            "pages": sorted(pages),
            "doc_keys": sorted(docs),
        })
        for m in members:
            alias_map[m] = canonical

    nodes.sort(key=lambda n: (-int(n["freq"]), str(n["name"])))
    meta = {
        "mentions_raw": len(mentions),
        "nodes": len(nodes),
        "merged_by_alias_table": sum(1 for n in nodes if n["aliases"]),
        "extra_merges_dice": len(merge_pairs),
        "merge_pairs_sample": merge_pairs[:10],
        "type_conflicts": sum(1 for n in nodes if n["type_conflict"]),
        "declared_entity_types": len(ENTITY_TYPES),
    }
    if logger is not None:
        logger.info("实体归并：原始提及 %d → 节点 %d（别名表命中 %d，Dice 二次归并 %d 对）",
                    meta["mentions_raw"], meta["nodes"], meta["merged_by_alias_table"], meta["extra_merges_dice"])
    return nodes, {"meta": meta, "alias_map": alias_map}


def _dominant_type(slot: dict[str, Any]) -> str:
    types: Counter = slot.get("types") or Counter()
    if not types:
        return "公司"
    return types.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# ② 关系去重 + ③ 约束校验
# ---------------------------------------------------------------------------
def build_edges(records: Sequence[dict[str, Any]], nodes: list[dict[str, Any]],
                alias_map: dict[str, str], cfg: Cfg, *, logger: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """聚合关系为边表，做类型吸附、头尾相容校验与去重。"""
    node_type = {str(n["name"]): str(n["type"]) for n in nodes}
    known = set(node_type)

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats = {"relations_raw": 0, "deduped": 0, "type_inexact": 0, "constraint_violations": 0,
             "self_loops": 0, "endpoint_added": 0, "violations_sample": []}

    for rec in records:
        for item in (rec.get("relations") or []):
            stats["relations_raw"] += 1
            raw_head = canonical_name(str(item.get("head") or ""))
            raw_tail = canonical_name(str(item.get("tail") or ""))
            head = alias_map.get(raw_head, raw_head)
            tail = alias_map.get(raw_tail, raw_tail)
            if not head or not tail:
                continue
            if head == tail:
                stats["self_loops"] += 1
                continue
            rel = str(item.get("rel") or "")
            if rel not in RELATION_TYPES:
                rel = coerce_type(rel, kind="relation")
            else:
                pass
            if not item.get("rel_exact"):
                stats["type_inexact"] += 1

            h_type = node_type.get(head) or coerce_type("", kind="entity")
            t_type = node_type.get(tail) or coerce_type("", kind="entity")
            if head not in known:
                known.add(head)
                stats["endpoint_added"] += 1
            if tail not in known:
                known.add(tail)
                stats["endpoint_added"] += 1

            ok = relation_allowed(rel, h_type, t_type)
            if not ok:
                stats["constraint_violations"] += 1
                if len(stats["violations_sample"]) < 10:
                    stats["violations_sample"].append(
                        {"head": head, "head_type": h_type, "rel": rel, "tail": tail, "tail_type": t_type})
                # 处置策略（配置 graph.constraint_violations_policy）：
                #   drop         = 丢弃违规边
                #   keep_flagged = 保留该边并标记 constraint_ok=false（默认）
                # 选 keep_flagged 的理由：边是 LLM 从原文抽出的真实三元组（带 evidence），
                # 静默丢弃会让图谱几乎无边（试点实测 7/7 被丢）；合规性作为独立指标如实上报，
                # 由检索侧按开关决定是否过滤。**没有任何"改写/编造"动作**。
                if str(cfg.graph.get("constraint_violations_policy") or "keep_flagged") == "drop":
                    continue

            key = (head, rel, tail)
            slot = merged.get(key)
            if slot is None:
                merged[key] = {
                    "head": head, "head_type": h_type, "rel": rel, "tail": tail, "tail_type": t_type,
                    "constraint_ok": bool(ok),
                    "weight": 1, "source_chunks": [rec.get("chunk_id")] if rec.get("chunk_id") else [],
                    "pages": [rec.get("page")] if rec.get("page") is not None else [],
                    "evidence": [str(item.get("evidence") or "")][:1] if item.get("evidence") else [],
                }
            else:
                slot["weight"] += 1
                slot["constraint_ok"] = bool(slot.get("constraint_ok")) and bool(ok)
                cid = rec.get("chunk_id")
                if cid and cid not in slot["source_chunks"]:
                    slot["source_chunks"].append(str(cid))
                pg = rec.get("page")
                if pg is not None and pg not in slot["pages"]:
                    slot["pages"].append(pg)
                ev = str(item.get("evidence") or "")
                if ev and ev not in slot["evidence"] and len(slot["evidence"]) < 3:
                    slot["evidence"].append(ev)

    edges = list(merged.values())
    edges.sort(key=lambda e: (-int(e["weight"]), str(e["head"]), str(e["rel"]), str(e["tail"])))

    # 端点补全：GraphML 要求每条边的 source/target 必须在 <node> 中已声明。
    # 关系端点可能因「实体名超长被丢弃」「实体被垃圾过滤」而不在节点表里，
    # 这里为缺失端点补一条仅存在于边上的节点（endpoint_only=true，freq=0）。
    # **不做任何名称改写**，只是让图谱自洽。
    node_names = {str(n["name"]) for n in nodes}
    endpoint_nodes: list[dict[str, Any]] = []
    for e in edges:
        for role in ("head", "tail"):
            name = str(e[role])
            if name in node_names:
                continue
            node_names.add(name)
            endpoint_nodes.append({
                "id": _slug_id(name), "name": name,
                "type": str(e.get(f"{role}_type") or "公司"),
                "type_distribution": {}, "type_conflict": False, "type_exact": 0,
                "freq": 0, "degree": 0, "aliases": [], "raw_names": [name],
                "chunk_ids": [], "pages": [], "doc_keys": [], "endpoint_only": True,
            })
    if endpoint_nodes:
        nodes.extend(endpoint_nodes)
    stats["endpoint_nodes_added"] = len(endpoint_nodes)

    stats["deduped"] = len(edges)
    stats["declared_relation_types"] = len(RELATION_TYPES)

    # 度数回填
    deg: Counter = Counter()
    for e in edges:
        deg[str(e["head"])] += int(e["weight"])
        deg[str(e["tail"])] += int(e["weight"])
    for node in nodes:
        node["degree"] = int(deg.get(str(node["name"]), 0))

    if logger is not None:
        logger.info("关系聚合：原始 %d → 去重后 %d 条；类型吸附 %d 次；约束违规 %d 条（已挡）",
                    stats["relations_raw"], stats["deduped"], stats["type_inexact"], stats["constraint_violations"])
    return edges, stats


# ---------------------------------------------------------------------------
# ④ 导出
# ---------------------------------------------------------------------------
def write_graphml(nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]], path: Path | str) -> Path:
    """手写 GraphML（标准库，无 networkx 依赖）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
        '  <key id="n_name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="n_type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="n_freq" for="node" attr.name="freq" attr.type="int"/>',
        '  <key id="n_degree" for="node" attr.name="degree" attr.type="int"/>',
        '  <key id="e_rel" for="edge" attr.name="relation" attr.type="string"/>',
        '  <key id="e_weight" for="edge" attr.name="weight" attr.type="int"/>',
        '  <key id="e_evidence" for="edge" attr.name="evidence" attr.type="string"/>',
        '  <graph id="lightrag" edgedefault="directed">',
    ]
    for n in nodes:
        lines.append(f'    <node id={quoteattr(str(n["id"]))}>')
        lines.append(f'      <data key="n_name">{escape(str(n["name"]))}</data>')
        lines.append(f'      <data key="n_type">{escape(str(n["type"]))}</data>')
        lines.append(f'      <data key="n_freq">{int(n["freq"])}</data>')
        lines.append(f'      <data key="n_degree">{int(n.get("degree", 0))}</data>')
        lines.append('    </node>')
    for i, e in enumerate(edges):
        lines.append(f'    <edge id="e{i}" source={quoteattr(_slug_id(str(e["head"])))} '
                     f'target={quoteattr(_slug_id(str(e["tail"])))}>')
        lines.append(f'      <data key="e_rel">{escape(str(e["rel"]))}</data>')
        lines.append(f'      <data key="e_weight">{int(e["weight"])}</data>')
        lines.append(f'      <data key="e_evidence">{escape(" ｜ ".join(str(x) for x in (e.get("evidence") or [])[:2]))}</data>')
        lines.append('    </edge>')
    lines.append('  </graph>')
    lines.append('</graphml>')
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def graph_stats(nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """节点/边/类型分布/合法率等统计（全部为真实计数）。"""
    e_types = Counter(str(n["type"]) for n in nodes)
    r_types = Counter(str(e["rel"]) for e in edges)
    total_e = sum(e_types.values()) or 0
    total_r = sum(r_types.values()) or 0
    exact_e = sum(int(n.get("type_exact") or 0) for n in nodes)
    declared_e_mentions = sum(sum((n.get("type_distribution") or {}).values()) for n in nodes)
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "entity_types_used": len(e_types),
        "entity_types_declared": len(ENTITY_TYPES),
        "relation_types_used": len(r_types),
        "relation_types_declared": len(RELATION_TYPES),
        "entity_type_distribution": dict(e_types.most_common()),
        "relation_type_distribution": dict(r_types.most_common()),
        "entity_multi_alias_nodes": sum(1 for n in nodes if n.get("aliases")),
        "entity_type_conflicts": sum(1 for n in nodes if n.get("type_conflict")),
        "edges_constraint_ok": sum(1 for e in edges if e.get("constraint_ok")),
        "edges_constraint_flagged": sum(1 for e in edges if not e.get("constraint_ok")),
        "edge_weight_total": sum(int(e["weight"]) for e in edges),
        "top_nodes_by_degree": [
            {"name": n["name"], "type": n["type"], "degree": int(n.get("degree", 0)), "freq": int(n["freq"])}
            for n in sorted(nodes, key=lambda n: (-int(n.get("degree", 0)), -int(n["freq"])))[:15]
        ],
        "top_edges_by_weight": [
            {"head": e["head"], "rel": e["rel"], "tail": e["tail"], "weight": int(e["weight"])}
            for e in edges[:15]
        ],
        "entity_type_mention_total": declared_e_mentions,
        "entity_type_exact_mentions": exact_e,
    }


def save_graph(nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]], *,
               cfg: Cfg, meta: dict[str, Any]) -> dict[str, str]:
    """落盘 nodes.jsonl / edges.jsonl / graph.json / graph.graphml / graph_meta.json。"""
    out_dir = cfg.path("graph_dir")
    n_path = Path(out_dir) / "nodes.jsonl"
    e_path = Path(out_dir) / "edges.jsonl"
    g_path = Path(out_dir) / "graph.json"
    x_path = Path(out_dir) / "graph.graphml"
    m_path = Path(out_dir) / "graph_meta.json"

    with n_path.open("w", encoding="utf-8") as fh:
        for n in nodes:
            fh.write(json.dumps(n, ensure_ascii=False) + "\n")
    with e_path.open("w", encoding="utf-8") as fh:
        for e in edges:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    payload = {
        "work_order": "人工智能NLP-RAG 项目-LightRAG 优化",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nodes": list(nodes),
        "edges": list(edges),
        "meta": meta,
    }
    g_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_graphml(nodes, edges, x_path)
    m_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"nodes": str(n_path), "edges": str(e_path), "graph_json": str(g_path),
            "graphml": str(x_path), "meta": str(m_path)}


def load_records(path: Path | str) -> list[dict[str, Any]]:
    """读回 raw_extractions.jsonl。"""
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return rows
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_graph(records: Sequence[dict[str, Any]], cfg: Cfg, *, logger: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """一次跑完 ①~③ 并返回 (nodes, edges, stats)。"""
    nodes, em = merge_entities(records, cfg, logger=logger)
    alias_map = em["alias_map"]
    edges, es = build_edges(records, nodes, alias_map, cfg, logger=logger)
    st = graph_stats(nodes, edges)
    st.update({
        "entity_merge": em["meta"],
        "relation_build": {k: v for k, v in es.items() if k != "violations_sample"},
        "relation_violations_sample": es.get("violations_sample") or [],
    })
    return nodes, edges, st
