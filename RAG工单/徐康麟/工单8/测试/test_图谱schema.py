# -*- coding: utf-8 -*-
"""工单8 测试：知识图谱 Schema 校验（实体/关系类型、头尾约束、悬空边、统计口径）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

校验目标（对应 设计/知识图谱Schema设计.md §6 的 7 条不变量）：
    1. 所有实体 ``type`` ∈ ``ENTITY_TYPES``；
    2. 所有关系 ``relation`` ∈ ``RELATION_TYPES``；
    3. 每条关系满足 ``head_type`` / ``tail_type`` 约束；
    4. 无自环边；
    5. 关系引用的实体名能在图谱节点里找到（无悬空边）；
    6. ``confidence`` ∈ (0, 1]、``count`` ≥ 1；
    7. ``graph.json`` 的 schema 版本与代码一致。

运行：``pwsh -NoProfile -File run_py.ps1 -m pytest 测试/test_图谱schema.py -q``
"""

from __future__ import annotations

from conftest import GRAPH_DIR, WORK_ORDER, load_json, load_jsonl  # noqa: E402
from app.graphrag.schema import (ENTITY_TYPES, RELATION_TYPES, SCHEMA_VERSION,  # noqa: E402
                                 is_valid_relation, relation_canonical)

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答


def test_图谱产物齐全() -> None:
    """四类图谱产物必须存在且非空（JSONL / JSON / GraphML）。"""
    for name in ("entities.jsonl", "relations.jsonl", "graph.json", "graph.graphml",
                 "图谱构建统计.json"):
        path = GRAPH_DIR / name
        assert path.is_file(), f"缺少图谱产物：{path}"
        assert path.stat().st_size > 0, f"图谱产物为空：{path}"


def test_实体类型都在schema内() -> None:
    """所有实体 type ∈ ENTITY_TYPES（9 类）。"""
    rows = load_jsonl(GRAPH_DIR / "entities.jsonl")
    assert rows, "entities.jsonl 为空"
    bad = sorted({str(r.get("type")) for r in rows} - set(ENTITY_TYPES))
    assert not bad, f"发现 schema 未登记的实体类型：{bad}"
    assert len(ENTITY_TYPES) == 9, f"实体类型数量变化（{len(ENTITY_TYPES)}）需同步更新设计文档"


def test_关系类型都在schema内() -> None:
    """所有关系 relation ∈ RELATION_TYPES（且必须是规范名）。"""
    rows = load_jsonl(GRAPH_DIR / "relations.jsonl")
    assert rows, "relations.jsonl 为空"
    unknown = sorted({str(r.get("relation")) for r in rows if not is_valid_relation(str(r.get("relation")))})
    assert not unknown, f"发现 schema 未登记的关系类型：{unknown}"
    alias_left = [str(r.get("relation")) for r in rows
                  if relation_canonical(str(r.get("relation"))) != str(r.get("relation"))]
    assert not alias_left, f"关系名未归一到规范名：{alias_left[:5]}"


def test_关系头尾类型满足约束() -> None:
    """每条关系的 head_type / tail_type 必须落在 RELATION_TYPES 声明的集合内。"""
    rows = load_jsonl(GRAPH_DIR / "relations.jsonl")
    violations: list[str] = []
    for row in rows:
        spec = RELATION_TYPES.get(str(row.get("relation")))
        if not spec:
            continue
        head_ok = (not spec.get("head")) or str(row.get("head_type")) in spec["head"]
        tail_ok = (not spec.get("tail")) or str(row.get("tail_type")) in spec["tail"]
        if not (head_ok and tail_ok):
            violations.append(f"{row.get('head')}-[{row.get('relation')}]->{row.get('tail')}")
    assert not violations, f"头尾类型越界 {len(violations)} 条，例如：{violations[:5]}"


def test_无自环边() -> None:
    """不得存在 head == tail 的自环边。"""
    rows = load_jsonl(GRAPH_DIR / "relations.jsonl")
    loops = [r for r in rows if str(r.get("head")) == str(r.get("tail"))]
    assert not loops, f"发现自环边：{[r.get('head') for r in loops[:5]]}"


def test_无悬空边(load_nodes: set[str] | None = None) -> None:
    """关系引用的实体名必须能在实体表里找到（无悬空边）。"""
    nodes = load_nodes or {str(r.get("name")) for r in load_jsonl(GRAPH_DIR / "entities.jsonl")}
    rows = load_jsonl(GRAPH_DIR / "relations.jsonl")
    dangling = [f"{r.get('head')}/{r.get('tail')}" for r in rows
                if str(r.get("head")) not in nodes or str(r.get("tail")) not in nodes]
    assert not dangling, f"发现悬空边 {len(dangling)} 条，例如：{dangling[:5]}"


def test_置信度与计数区间合法() -> None:
    """实体 confidence ∈ (0, 1]、count ≥ 1；关系 confidence ∈ (0, 1]。"""
    entities = load_jsonl(GRAPH_DIR / "entities.jsonl")
    bad_e = [e.get("name") for e in entities
             if not (0.0 < float(e.get("confidence") or 0.0) <= 1.0) or int(e.get("count") or 0) < 1]
    assert not bad_e, f"实体置信度/计数越界：{bad_e[:5]}"
    relations = load_jsonl(GRAPH_DIR / "relations.jsonl")
    bad_r = [f"{r.get('head')}-{r.get('relation')}" for r in relations
             if not (0.0 < float(r.get("confidence") or 0.0) <= 1.0)]
    assert not bad_r, f"关系置信度越界：{bad_r[:5]}"


def test_graph_json结构自洽() -> None:
    """graph.json 的节点/边结构与 JSONL 一致，且 schema 版本与代码相同。"""
    payload = load_json(GRAPH_DIR / "graph.json", {}) or {}
    assert payload.get("nodes"), "graph.json 缺 nodes"
    assert payload.get("edges"), "graph.json 缺 edges"
    assert payload.get("schema_version") == SCHEMA_VERSION, (
        f"schema 版本不一致：文件 {payload.get('schema_version')} vs 代码 {SCHEMA_VERSION}")
    assert payload.get("work_order") == WORK_ORDER, "graph.json 的工单编号缺失或不一致"
    entity_count = len(load_jsonl(GRAPH_DIR / "entities.jsonl"))
    assert len(payload["nodes"]) == entity_count, (
        f"graph.json 节点数 {len(payload['nodes'])} 与 entities.jsonl {entity_count} 不一致")


def test_graphml是合法XML() -> None:
    """GraphML 必须是可解析的 XML，且节点/边数与 graph.json 一致。"""
    import xml.etree.ElementTree as ET

    tree = ET.parse(GRAPH_DIR / "graph.graphml")
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    nodes = root.findall(".//g:node", ns)
    edges = root.findall(".//g:edge", ns)
    payload = load_json(GRAPH_DIR / "graph.json", {}) or {}
    assert len(nodes) == len(payload.get("nodes") or []), "GraphML 节点数与 graph.json 不一致"
    assert len(edges) <= len(payload.get("edges") or []), "GraphML 边数不应多于 graph.json"


def test_提及块带页码与文件名() -> None:
    """实体的提及块必须带 chunk_id / 页码 / 文件名（「返回答案所在文本块」的硬要求）。"""
    entities = load_jsonl(GRAPH_DIR / "entities.jsonl")
    missing = []
    for entity in entities[:2000]:
        for chunk in entity.get("chunks") or []:
            if not chunk.get("chunk_id") or not chunk.get("page") or not chunk.get("file_name"):
                missing.append(entity.get("name"))
                break
    assert not missing, f"提及块字段不全（前 2000 个实体中 {len(missing)} 个）：{missing[:5]}"


def test_构建统计与产物一致() -> None:
    """图谱构建统计.json 里的实体/关系数必须与 JSONL 行数一致（防止手写数字）。"""
    stats = load_json(GRAPH_DIR / "图谱构建统计.json", {}) or {}
    assert int(stats.get("entities") or 0) == len(load_jsonl(GRAPH_DIR / "entities.jsonl")), \
        "统计里的实体数与 entities.jsonl 不一致"
    assert int(stats.get("relations") or 0) == len(load_jsonl(GRAPH_DIR / "relations.jsonl")), \
        "统计里的关系数与 relations.jsonl 不一致"
    assert int(stats.get("chunks_scanned") or 0) > 0, "统计里 chunks_scanned 异常"
    assert int(stats.get("elapsed_ms") or 0) > 0, "统计里 elapsed_ms 异常"
