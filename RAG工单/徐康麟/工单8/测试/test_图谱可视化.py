# -*- coding: utf-8 -*-
"""工单8 测试：图谱可视化产物（PNG / SVG / HTML 自包含）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

验收项对应：工单原文验收标准 1「知识图谱可视化：可视化展示 PDF 中的文档解析出来的知识图谱数据中的实体、关系」。

校验点：
    * PNG 存在、尺寸合规、节点数 ≥120（含关系连线与图例由渲染脚本保证）；
    * SVG / HTML **自包含**（0 外链，断网可用）；
    * 可视化统计.json 的节点/边数与 PNG 描述一致（数字可追溯）。
"""

from __future__ import annotations

import re
from pathlib import Path

from conftest import GRAPH_DIR, SHOT_DIR, load_json

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答

PNG = SHOT_DIR / "知识图谱可视化.png"
SVG = GRAPH_DIR / "graph.svg"
HTML = GRAPH_DIR / "graph.html"
STATS = GRAPH_DIR / "可视化统计.json"


def test_可视化产物齐全() -> None:
    """PNG / SVG / HTML / 可视化统计 四件套必须齐全且非空。"""
    for path in (PNG, SVG, HTML, STATS):
        assert path.is_file(), f"缺少可视化产物：{path}"
        assert path.stat().st_size > 0, f"可视化产物为空：{path}"


def test_PNG尺寸与节点数达标() -> None:
    """PNG 必须能被 Pillow 打开、画布 ≥1200×800、节点数 ≥120（工单硬性要求）。"""
    from PIL import Image

    with Image.open(PNG) as image:
        width, height = image.size
    assert width >= 1200 and height >= 800, f"画布过小：{width}×{height}"
    stats = load_json(STATS, {}) or {}
    nodes = int((stats.get("layout") or {}).get("nodes") or 0)
    assert nodes >= 120, f"可视化节点数 {nodes} < 120（工单要求 ≥120 节点）"
    assert int((stats.get("layout") or {}).get("edges") or 0) > 0, "可视化未绘制任何关系连线"


def test_SVG自包含且含节点与连线() -> None:
    """SVG 必须自包含（0 外链）且包含 circle/line 元素（实体与关系）。"""
    text = SVG.read_text(encoding="utf-8")
    external = text.count("http://") + text.count("https://") - text.count("http://www.w3.org/2000/svg")
    assert external == 0, f"SVG 存在 {external} 处外链（要求自包含）"
    assert text.count("<circle") >= 120, f"SVG 节点（circle）不足：{text.count('<circle')}"
    assert text.count("<line") > 0, "SVG 未绘制关系连线（line）"
    assert "图例" in text or "legend" in text.lower() or "Schema" in text, "SVG 缺少图例或版本标注"


def test_HTML自包含且内嵌数据() -> None:
    """HTML 必须自包含（0 外链）且内嵌节点/边 JSON（断网可渲染）。"""
    text = HTML.read_text(encoding="utf-8")
    external = text.count("http://") + text.count("https://") - text.count("http://www.w3.org/2000/svg")
    assert external == 0, f"HTML 存在 {external} 处外链（要求自包含）"
    assert "const DATA =" in text, "HTML 未内嵌数据（const DATA）"
    assert "<canvas" in text, "HTML 未使用 Canvas 渲染"
    stats = load_json(STATS, {}) or {}
    nodes = int((stats.get("layout") or {}).get("nodes") or 0)
    match = re.search(r'"nodes":\s*\[', text)
    assert match, "HTML 内嵌数据里没有 nodes"


def test_可视化统计与图谱规模自洽() -> None:
    """可视化统计里的图谱总规模必须与图谱构建统计一致（数字可追溯）。"""
    viz = load_json(STATS, {}) or {}
    build = load_json(GRAPH_DIR / "图谱构建统计.json", {}) or {}
    assert int(viz.get("graph_nodes_total") or 0) == int(build.get("entities") or 0), \
        "可视化统计的节点总数与图谱构建统计不一致"
    assert int(viz.get("graph_edges_total") or 0) == int(build.get("relations") or 0), \
        "可视化统计的边总数与图谱构建统计不一致"
    assert int((viz.get("png") or {}).get("nodes") or 0) >= 120, "PNG 记录的节点数不足 120"
