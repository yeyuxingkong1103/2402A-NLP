# -*- coding: utf-8 -*-
"""工单8 图谱存储层：落盘 / 加载 / 邻接索引 / GraphML 导出。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

产物契约（``研发/data/graph/``）：
    * ``entities.jsonl``  —— 每行一个实体（含提及块，供图谱扩展召回）;
    * ``relations.jsonl`` —— 每行一条关系（含证据片段与抽取器标记，可审计）;
    * ``graph.json``      —— 单文件图谱（nodes/edges/schema/统计），可视化与 UI 直接读它;
    * ``graph.graphml``   —— XML 交换格式（Gephi / yEd 可直接打开）;
    * ``图谱构建统计.json`` —— 构建过程的实测数字（供测试报告与截图引用，禁止手写）。

设计纪律：
    * 加载后建**邻接表**（双向），一跳/两跳扩展是 O(邻居数) 而不是 O(边数)；
    * 所有读取都做 try/except 兜底：图谱缺失时 ``load_graph`` 返回空图谱并给 ``ready=False``，
      检索层据此**自动退化为纯向量+BM25**（不崩、不静默：调用方必须把 graph_used=False 如实上报）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.sax.saxutils import escape as _xml_escape

from .config import GRAPH_FILES, GraphRAGConfig
from .schema import SCHEMA_VERSION, WORK_ORDER, entity_type_zh, group_of_relation, schema_dict

__all__ = ["GraphStore", "load_graph", "save_graph", "write_graphml"]


@dataclass(slots=True)
class GraphStore:
    """内存态知识图谱（节点 + 边 + 邻接表 + 实体→证据块反向索引）。"""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    #: 邻接表：node_id → [(邻居 node_id, 关系名, 边的下标)]
    adjacency: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)
    #: 实体名 → node_id（含别名），供问句实体识别做 O(1) 命中
    name_index: dict[str, str] = field(default_factory=dict)
    #: chunk_id → {page, file_name}（图谱扩展回来的块必须能给出页码/文件名）
    chunk_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    ready: bool = False
    source_dir: str = ""
    load_ms: float = 0.0

    # -- 基本属性 ---------------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def node(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes.get(str(node_id))

    def name_of(self, node_id: str) -> str:
        node = self.nodes.get(str(node_id))
        return str(node.get("name")) if node else str(node_id)

    def type_of(self, node_id: str) -> str:
        node = self.nodes.get(str(node_id))
        return str(node.get("type")) if node else "unknown"

    def resolve(self, name: str) -> str | None:
        """实体名/别名 → node_id（先精确、再大小写无关、最后子串兜底）。"""
        key = str(name or "").strip()
        if not key:
            return None
        hit = self.name_index.get(key)
        if hit:
            return hit
        low = key.lower()
        for candidate, node_id in self.name_index.items():
            if candidate.lower() == low:
                return node_id
        return None

    def match_names(self, text: str, *, min_len: int = 2, limit: int = 40) -> list[str]:
        """从问句里找出图谱中出现的实体（**长名优先**，避免「银行」这类碎片先命中）。"""
        flat = "".join(str(text or "").split())
        found: list[tuple[int, str]] = []
        for name, node_id in self.name_index.items():
            if len(name) < min_len:
                continue
            if name in flat:
                found.append((len(name), node_id))
        found.sort(key=lambda item: (-item[0], item[1]))
        out: list[str] = []
        for _length, node_id in found:
            if node_id not in out:
                out.append(node_id)
            if len(out) >= limit:
                break
        return out

    # -- 扩展 -------------------------------------------------------------
    def neighbors(self, node_id: str) -> list[tuple[str, str, int]]:
        return self.adjacency.get(str(node_id), [])

    def expand(self, seeds: Iterable[str], *, hop: int = 2, max_nodes: int = 40,
               min_confidence: float = 0.0) -> dict[str, Any]:
        """BFS 一跳/两跳邻域扩展。

        返回 ``{"distances": {node_id: 距离}, "visited": [...], "truncated": bool,
        "edges_used": [{head, relation, tail, confidence, chunk_id}...]}``。
        距离 0 = 种子实体本身（问句直接点名），1 = 一跳，2 = 两跳。
        """
        distances: dict[str, int] = {}
        queue: list[str] = []
        for seed in seeds:
            sid = str(seed)
            if sid in self.nodes and sid not in distances:
                distances[sid] = 0
                queue.append(sid)
        edges_used: list[dict[str, Any]] = []
        truncated = False
        depth = 0
        while queue and depth < max(int(hop), 0):
            depth += 1
            next_queue: list[str] = []
            for node_id in queue:
                for neighbor, relation, edge_idx in self.neighbors(node_id):
                    edge = self.edges[edge_idx] if 0 <= edge_idx < len(self.edges) else {}
                    if float(edge.get("confidence") or 0.0) < float(min_confidence):
                        continue
                    edges_used.append({"head": edge.get("head"), "relation": relation,
                                       "tail": edge.get("tail"),
                                       "confidence": edge.get("confidence"),
                                       "chunk_id": edge.get("chunk_id")})
                    if neighbor in distances:
                        continue
                    if len(distances) >= int(max_nodes):
                        truncated = True
                        continue
                    distances[neighbor] = depth
                    next_queue.append(neighbor)
            queue = next_queue
        return {"distances": distances, "edges_used": edges_used, "truncated": truncated,
                "hop": int(hop)}

    def chunks_of(self, node_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
        """实体提及的块（含 ``page`` / ``file_name``，检索层据此返回「答案所在文本块」）。"""
        node = self.nodes.get(str(node_id))
        if not node:
            return []
        return list(node.get("chunks") or [])[: max(int(limit), 1)]

    # -- 序列化 -----------------------------------------------------------
    def to_public_dict(self, *, max_nodes: int = 120, max_edges: int = 240,
                       max_local_nodes: int = 40) -> dict[str, Any]:
        """给 UI / 截图用的「图谱结构缩略」：按度数取最重要的节点。"""
        return graph_digest(self, max_nodes=max_nodes, max_edges=max_edges,
                            max_local_nodes=max_local_nodes)

    def to_graph_json(self) -> dict[str, Any]:
        """完整 graph.json 结构。"""
        nodes = []
        for node in self.nodes.values():
            data = dict(node)
            data["type_zh"] = entity_type_zh(str(node.get("type")))
            data["degree"] = len(self.adjacency.get(str(node.get("id")), []))
            nodes.append(data)
        edges = [dict(edge, group=group_of_relation(str(edge.get("relation")))) for edge in self.edges]
        return {
            "work_order": WORK_ORDER,
            "schema_version": SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": dict(self.stats),
            "schema": self.schema or schema_dict(),
            "nodes": nodes,
            "edges": edges,
            "legend": {"entity_types": {k: v["zh"] for k, v in (self.schema or schema_dict())["entity_types"].items()},
                       "relation_groups": (self.schema or schema_dict())["relation_groups"]},
        }


# ---------------------------------------------------------------------------
# 构建 / 加载
# ---------------------------------------------------------------------------
def build_store(entities: Iterable[Mapping[str, Any]], relations: Iterable[Mapping[str, Any]], *,
                stats: Mapping[str, Any] | None = None) -> GraphStore:
    """由实体/关系行构建内存图谱（建邻接表 + 名字索引 + 块元信息）。

    兼容性：节点 id 可能是「裸名」（当前口径）也可能带历史类型前缀（旧产物），
    这里统一用**名字**做邻接键，保证两种产物都能正确建图、且同名不会分裂成两个节点。
    """
    store = GraphStore(schema=schema_dict(), stats=dict(stats or {}))
    name_to_id: dict[str, str] = {}
    for row in entities:
        node_id = str(row.get("id") or "")
        if not node_id:
            continue
        name = str(row.get("name") or node_id.split(":", 1)[-1])
        if name in name_to_id:                      # 同名合并（首次判定者优先）
            existing = store.nodes.get(name_to_id[name])
            if existing is not None:
                existing["count"] = int(existing.get("count") or 0) + int(row.get("count") or 0)
            continue
        node = {
            "id": node_id, "name": name, "type": str(row.get("type") or "unknown"),
            "type_zh": entity_type_zh(str(row.get("type"))),
            "confidence": float(row.get("confidence") or 0.0), "count": int(row.get("count") or 0),
            "aliases": list(row.get("aliases") or []), "chunks": list(row.get("chunks") or []),
        }
        store.nodes[node_id] = node
        name_to_id[name] = node_id
        for alias in [name, *node["aliases"]]:
            key = str(alias or "").strip()
            if key and key not in store.name_index:
                store.name_index[key] = node_id
        for chunk in node["chunks"]:
            cid = str(chunk.get("chunk_id") or "")
            if cid and cid not in store.chunk_meta:
                store.chunk_meta[cid] = {"page": int(chunk.get("page") or 0),
                                         "file_name": str(chunk.get("file_name") or ""),
                                         "type": str(chunk.get("type") or "text")}
    for row in relations:
        head = str(row.get("head") or "")
        tail = str(row.get("tail") or "")
        if not head or not tail:
            continue
        edge_idx = len(store.edges)
        edge = {
            "head": head, "relation": str(row.get("relation") or ""), "tail": tail,
            "head_type": str(row.get("head_type") or ""), "tail_type": str(row.get("tail_type") or ""),
            "confidence": float(row.get("confidence") or 0.0),
            "extractor": str(row.get("extractor") or "rule"), "evidence": str(row.get("evidence") or ""),
            "chunk_id": str(row.get("chunk_id") or ""), "page": int(row.get("page") or 0),
            "file_name": str(row.get("file_name") or ""),
            "group": group_of_relation(str(row.get("relation") or "")),
        }
        store.edges.append(edge)
        head_id = name_to_id.get(head) or store.resolve(head) or f"unknown:{head}"
        tail_id = name_to_id.get(tail) or store.resolve(tail) or f"unknown:{tail}"
        store.adjacency.setdefault(head_id, []).append((tail_id, edge["relation"], edge_idx))
        store.adjacency.setdefault(tail_id, []).append((head_id, edge["relation"], edge_idx))
    store.stats.setdefault("nodes", store.node_count)
    store.stats.setdefault("edges", store.edge_count)
    store.ready = store.node_count > 0 and store.edge_count > 0
    return store


def load_graph(cfg: GraphRAGConfig, *, logger: Any = None) -> GraphStore:
    """从 ``cfg.graph_dir`` 加载图谱；任何缺失/损坏都返回 ``ready=False`` 的空图谱（显式降级）。"""
    started = time.perf_counter()
    graph_dir = Path(cfg.graph_dir)
    store = GraphStore(source_dir=str(graph_dir))
    graph_json = graph_dir / GRAPH_FILES["graph"]
    entities_path = graph_dir / GRAPH_FILES["entities"]
    relations_path = graph_dir / GRAPH_FILES["relations"]
    try:
        if graph_json.is_file():
            payload = json.loads(graph_json.read_text(encoding="utf-8"))
            store = build_store(payload.get("nodes") or [], payload.get("edges") or [],
                                stats=payload.get("stats") or {})
            store.source_dir = str(graph_dir)
        elif entities_path.is_file() and relations_path.is_file():
            store = build_store(_read_jsonl(entities_path), _read_jsonl(relations_path))
            store.source_dir = str(graph_dir)
        else:
            if logger is not None:
                logger.log_event("graph.missing", level="WARNING", graph_dir=str(graph_dir),
                                 action="图谱不可用 ⇒ 检索退化为纯向量+BM25（graph_used=False）")
    except Exception as exc:  # noqa: BLE001 —— 图谱损坏不得让问答崩掉
        if logger is not None:
            logger.log_event("graph.load_failed", level="ERROR", graph_dir=str(graph_dir),
                             error=f"{type(exc).__name__}: {exc}",
                             action="图谱不可用 ⇒ 检索退化为纯向量+BM25（graph_used=False）")
        store = GraphStore(source_dir=str(graph_dir))
    store.load_ms = round((time.perf_counter() - started) * 1000, 2)
    if logger is not None:
        logger.log_event("graph.loaded", nodes=store.node_count, edges=store.edge_count,
                         ready=store.ready, elapsed_ms=store.load_ms, dir=store.source_dir)
    return store


def save_graph(cfg: GraphRAGConfig, store: GraphStore, *, entities: Iterable[Mapping[str, Any]] | None = None,
               relations: Iterable[Mapping[str, Any]] | None = None, logger: Any = None) -> dict[str, str]:
    """写 entities.jsonl / relations.jsonl / graph.json / graph.graphml（原子写：先临时文件再替换）。"""
    graph_dir = Path(cfg.graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    payload = store.to_graph_json()
    if entities is not None:
        _write_jsonl(graph_dir / GRAPH_FILES["entities"], entities)
        written["entities"] = str(graph_dir / GRAPH_FILES["entities"])
    if relations is not None:
        _write_jsonl(graph_dir / GRAPH_FILES["relations"], relations)
        written["relations"] = str(graph_dir / GRAPH_FILES["relations"])
    _atomic_write(graph_dir / GRAPH_FILES["graph"], json.dumps(payload, ensure_ascii=False, indent=1))
    written["graph"] = str(graph_dir / GRAPH_FILES["graph"])
    _atomic_write(graph_dir / GRAPH_FILES["graphml"], render_graphml(payload))
    written["graphml"] = str(graph_dir / GRAPH_FILES["graphml"])
    if logger is not None:
        logger.log_event("graph.saved", **written)
    return written


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def render_graphml(payload: Mapping[str, Any]) -> str:
    """GraphML（XML）导出：Gephi / yEd 可直接打开；节点带 type/degree，边带 relation/confidence。"""
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="d_name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="d_type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="d_type_zh" for="node" attr.name="type_zh" attr.type="string"/>',
        '  <key id="d_degree" for="node" attr.name="degree" attr.type="int"/>',
        '  <key id="d_count" for="node" attr.name="count" attr.type="int"/>',
        '  <key id="d_rel" for="edge" attr.name="relation" attr.type="string"/>',
        '  <key id="d_conf" for="edge" attr.name="confidence" attr.type="double"/>',
        '  <key id="d_extractor" for="edge" attr.name="extractor" attr.type="string"/>',
        '  <graph id="GraphRAG" edgedefault="undirected">',
    ]
    ids = {str(node.get("id")): f"n{idx}" for idx, node in enumerate(payload.get("nodes") or [])}
    for node in payload.get("nodes") or []:
        nid = ids[str(node.get("id"))]
        lines.append(f'    <node id="{_xml_escape(nid)}">')
        lines.append(f'      <data key="d_type">{_xml_escape(str(node.get("type")))}</data>')
        lines.append(f'      <data key="d_type_zh">{_xml_escape(str(node.get("type_zh")))}</data>')
        lines.append(f'      <data key="d_degree">{int(node.get("degree") or 0)}</data>')
        lines.append(f'      <data key="d_count">{int(node.get("count") or 0)}</data>')
        lines.append(f'      <data key="d_name">{_xml_escape(str(node.get("name")))}</data>')
        lines.append('    </node>')
    for idx, edge in enumerate(payload.get("edges") or []):
        head = ids.get(str(edge.get("head")))
        tail = ids.get(str(edge.get("tail")))
        if not head or not tail:
            continue
        lines.append(f'    <edge id="e{idx}" source="{head}" target="{tail}">')
        lines.append(f'      <data key="d_rel">{_xml_escape(str(edge.get("relation")))}</data>')
        lines.append(f'      <data key="d_conf">{float(edge.get("confidence") or 0.0):.4f}</data>')
        lines.append(f'      <data key="d_extractor">{_xml_escape(str(edge.get("extractor")))}</data>')
        lines.append('    </edge>')
    lines.append('  </graph>')
    lines.append('</graphml>')
    return "\n".join(lines) + "\n"


def graph_digest(store: GraphStore, *, max_nodes: int = 120, max_edges: int = 240,
                 max_local_nodes: int = 40) -> dict[str, Any]:
    """图谱结构缩略（UI 展示 / 截图用）：按度数取核心子图，附类型与关系统计。

    ``max_local_nodes`` 限制 ``local`` 字段规模 —— 该字段是「问句命中实体的局部邻域」，
    由调用方按需填充（见 ``engine.answer``），此处只给空壳结构，保持函数纯粹。
    """
    degree = {nid: len(store.adjacency.get(nid, [])) for nid in store.nodes}
    top = sorted(store.nodes.values(), key=lambda n: (-degree.get(str(n["id"]), 0), str(n["name"])))
    top = top[: max(int(max_nodes), 1)]
    keep = {str(n["id"]) for n in top}
    nodes = [{"id": str(n["id"]), "name": str(n["name"]), "type": str(n["type"]),
              "type_zh": str(n.get("type_zh") or n["type"]), "degree": degree.get(str(n["id"]), 0),
              "count": int(n.get("count") or 0)} for n in top]
    edges = []
    for edge in store.edges:
        head, tail = str(edge.get("head")), str(edge.get("tail"))
        if head in keep and tail in keep:
            edges.append({"head": head, "relation": str(edge.get("relation")), "tail": tail,
                          "confidence": float(edge.get("confidence") or 0.0),
                          "group": str(edge.get("group") or group_of_relation(str(edge.get("relation"))))})
        if len(edges) >= int(max_edges):
            break
    by_type: dict[str, int] = {}
    for node in store.nodes.values():
        key = str(node.get("type_zh") or node.get("type"))
        by_type[key] = by_type.get(key, 0) + 1
    by_group: dict[str, int] = {}
    for edge in store.edges:
        key = str(edge.get("group") or group_of_relation(str(edge.get("relation"))))
        by_group[key] = by_group.get(key, 0) + 1
    return {
        "ready": bool(store.ready),
        "node_total": store.node_count,
        "edge_total": store.edge_count,
        "chunk_total": len(store.chunk_meta),
        "nodes_shown": len(nodes),
        "edges_shown": len(edges),
        "nodes": nodes,
        "edges": edges,
        "by_entity_type": by_type,
        "by_relation_group": by_group,
        "schema_version": SCHEMA_VERSION,
        "local": {"seed_names": [], "distances": {}, "nodes": [], "edges": []},
    }
