# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单8 图谱可视化：**自实现力导向布局** + PIL 自绘 PNG + 自包含 SVG/HTML。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

为什么自实现布局（设计取舍，见 ``设计/技术选型.md`` §2.2）：
    * 本机**未安装** networkx / pyvis / matplotlib，且依赖红线禁止裸 ``pip install``；
    * 需求只要「实体-关系」的可视化，力导向（斥力 + 弹簧 + 阻尼）几十行即可实现，
      且固定随机种子 ⇒ **每次渲染的图完全一致**（可复现，截图不会漂移）；
    * 自实现能精确控制性能：2400×1600 画布 / 120+ 节点 / 320 轮迭代在纯 Python 下 < 2 s。

产物：
    * ``知识图谱可视化.png`` —— PIL 绘制（节点按实体类型着色、按度数定大小、关系连线、图例、统计角标）
    * ``graph.svg`` —— 手写 SVG（矢量、可缩放、**无外网引用**）
    * ``graph.html`` —— 单文件 HTML（内嵌节点/边 JSON + 原生 Canvas 脚本，**无 CDN 依赖**）
"""

from __future__ import annotations

import html
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import ENTITY_TYPES, RELATION_GROUPS, SCHEMA_VERSION
from .store import GraphStore

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 实体类型 → 填充色（RGB）；图例与节点共用，保证「一眼看出类型」
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "company": (31, 105, 200),       # 公司：蓝
    "institution": (46, 158, 108),   # 机构：绿
    "metric": (222, 138, 24),        # 指标：橙
    "amount": (196, 66, 122),        # 金额：洋红
    "ratio": (146, 84, 196),         # 比例：紫
    "date": (120, 128, 140),         # 日期：灰
    "product": (208, 78, 62),        # 产品/业务：红
    "person": (24, 150, 168),        # 人物：青
    "city": (132, 150, 60),          # 地区：橄榄
}
DEFAULT_COLOR = (110, 110, 110)

#: 关系分组 → 连线色
GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    "所有权与结构": (31, 105, 200),
    "信息披露": (222, 138, 24),
    "人事": (24, 150, 168),
    "经营与业务": (208, 78, 62),
    "数值与变动": (146, 84, 196),
    "证据挂载": (150, 150, 150),
    "其它": (120, 120, 120),
}


@dataclass(slots=True)
class LayoutResult:
    """布局结果：节点坐标 + 绘制用的边 + 统计。"""

    positions: dict[str, tuple[float, float]]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    width: int
    height: int
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# 图数据选取
# ---------------------------------------------------------------------------
def _short_label(name: str, *, limit: int = 14) -> str:
    """节点短标签：公司全称去后缀（「平安银行股份有限公司」→「平安银行」），其余按长度截断。

    依据（实测）：年报语料里公司名多为全称，直接画标签会互相重叠、且一行挤不下；
    去后缀后长度降到 2~6 字，可读性显著提升（图例与 tooltip 仍保留全称）。
    """
    text = str(name or "").strip()
    for suffix in ("股份有限公司", "有限责任公司", "有限公司", "保险集团", "集团"):
        if len(text) > len(suffix) + 1 and text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text[:limit]


def select_subgraph(store: GraphStore, *, top_nodes: int = 120, max_edges: int = 400,
                    min_edges: int = 120, min_degree: int = 2,
                    seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """选取用于可视化的**核心子图**：按度数取 Top-N 节点，再取「两端都在节点集内」的边。

    选取依据（可解释）：度数高 = 图谱里的枢纽实体，它们才是「一眼看懂这张图」的骨架。

    ⚠️ 三个实测坑（都已修）：
        1. 节点 id 形如 ``company:平安银行``，而边的 ``head`` / ``tail`` 是**实体名** ——
           直接比较会「选中 120 节点却连 0 条边」，必须显式做名字→id 归一；
        2. 只取 Top-N（按度数）时，度数最高的实体彼此之间**未必有边**（它们各自连着一堆叶子），
           可能只连出十几条边 ⇒ 加 ``min_edges`` 约束：边不够时**逐步放宽节点名额**（每次 +25%），
           直到边数达标或达到上限（``top_nodes × 4``）；
        3. 度数 = 1 的叶子实体对「看懂图谱结构」没有贡献，只会让图更乱 ⇒
           ``min_degree`` 过滤（默认 ≥2），同时保证候选池足够大（不足时自动放宽到 1）。
    """
    degree = {nid: len(store.adjacency.get(nid, [])) for nid in store.nodes}
    pool = [n for n in store.nodes.values() if degree.get(str(n["id"]), 0) >= int(min_degree)]
    if len(pool) < max(int(top_nodes), 1):
        pool = list(store.nodes.values())         # 池子太小 ⇒ 放宽度数门槛（不硬失败）
    ranked = sorted(
        pool,
        key=lambda n: (-degree.get(str(n["id"]), 0), -int(n.get("count") or 0), str(n.get("name"))),
    )
    limit = max(int(top_nodes), 1)
    hard_limit = min(len(ranked), max(limit * 4, limit + 100))
    chosen: list[dict[str, Any]] = ranked[:limit]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    while True:
        nodes, edges = _subgraph_of(store, chosen, degree, max_edges=max_edges)
        if len(edges) >= int(min_edges) or len(chosen) >= hard_limit:
            break
        limit = min(int(limit * 1.25) + 1, hard_limit)
        chosen = ranked[:limit]
    return nodes, edges


def _subgraph_of(store: GraphStore, chosen: Sequence[Mapping[str, Any]],
                 degree: Mapping[str, int], *, max_edges: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keep = {str(n["id"]) for n in chosen}
    name_to_id = {str(n["name"]): str(n["id"]) for n in chosen}
    nodes: list[dict[str, Any]] = []
    for node in chosen:
        nid = str(node["id"])
        nodes.append({"id": nid, "name": str(node["name"]), "type": str(node["type"]),
                      "type_zh": str(node.get("type_zh") or node["type"]),
                      "degree": degree.get(nid, 0), "count": int(node.get("count") or 0)})
    edges: list[dict[str, Any]] = []
    for edge in store.edges:
        head, tail = str(edge.get("head")), str(edge.get("tail"))
        head_id = name_to_id.get(head) or (head if head in keep else None)
        tail_id = name_to_id.get(tail) or (tail if tail in keep else None)
        if head_id and tail_id and head_id in keep and tail_id in keep:
            edges.append({"head": head, "tail": tail, "relation": str(edge.get("relation")),
                          "group": str(edge.get("group") or "其它"),
                          "confidence": float(edge.get("confidence") or 0.0)})
    edges.sort(key=lambda e: -float(e.get("confidence") or 0.0))
    return nodes, edges[: int(max_edges)]


# ---------------------------------------------------------------------------
# 力导向布局（自实现）
# ---------------------------------------------------------------------------
def force_directed_layout(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]], *,
                          width: int = 2400, height: int = 1600, iterations: int = 320,
                          repulsion: float = 12000.0, spring: float = 0.012, damping: float = 0.86,
                          seed: int = 42) -> LayoutResult:
    """Fruchterman-Reingold 风格力导向布局（斥力 + 弹簧 + 阻尼 + 冷却 + **碰撞消除**）。

    实现要点（都是实测调出来的，不是照本宣科）：
        * 初始位置按**圆形分布** + 固定随机扰动（固定种子 ⇒ 可复现；不用纯随机撒点，避免初始重叠成团）；
        * **度数感知斥力**：斥力乘以 ``1 + 0.35·√degree`` —— 否则枢纽节点会被大量叶子拽成一团，
          实测第一版所有高连接度节点都糊在画布中心，标签完全不可读；
        * 引力 ``spring × (d - 理想边长)``（仅对存在的边）；
        * 阻尼 + 线性冷却，防止后期震荡；
        * 收尾做**碰撞消除**：对每个节点检查若干轮，与过近的节点互相推开
          （保证标签有地方可画，这是「可读性」的关键一步）；
        * 布局结束后把坐标压回画布内边距范围。
    """
    rng = random.Random(int(seed))
    names = [str(n.get("name")) for n in nodes]
    index = {name: i for i, name in enumerate(names)}
    count = len(names)
    if count == 0:
        return LayoutResult({}, [], [], width, height, {"nodes": 0, "edges": 0})
    weight = [1.0 + 0.35 * math.sqrt(max(int(n.get("degree") or 0), 0)) for n in nodes]
    radius = min(width, height) * 0.36
    pos: list[list[float]] = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        pos.append([width / 2 + radius * math.cos(angle) + rng.uniform(-12, 12),
                    height / 2 + radius * math.sin(angle) + rng.uniform(-12, 12)])
    links: list[tuple[int, int]] = []
    for edge in edges:
        a, b = index.get(str(edge.get("head"))), index.get(str(edge.get("tail")))
        if a is not None and b is not None and a != b:
            links.append((a, b))
    ideal = math.sqrt((width * height) / max(count, 1)) * 0.62
    started = time.perf_counter()
    for step in range(max(int(iterations), 1)):
        disp = [[0.0, 0.0] for _ in range(count)]
        for i in range(count):
            xi, yi = pos[i]
            for j in range(i + 1, count):
                xj, yj = pos[j]
                dx, dy = xi - xj, yi - yj
                dist2 = dx * dx + dy * dy
                if dist2 < 1e-6:
                    dx, dy, dist2 = rng.uniform(-1, 1), rng.uniform(-1, 1), 1e-6
                dist = math.sqrt(dist2)
                force = repulsion * weight[i] * weight[j] / dist2
                ux, uy = dx / dist, dy / dist
                disp[i][0] += ux * force
                disp[i][1] += uy * force
                disp[j][0] -= ux * force
                disp[j][1] -= uy * force
        for a, b in links:
            dx = pos[a][0] - pos[b][0]
            dy = pos[a][1] - pos[b][1]
            dist = math.hypot(dx, dy) or 1e-6
            force = spring * (dist - ideal)
            ux, uy = dx / dist, dy / dist
            disp[a][0] -= ux * force
            disp[a][1] -= uy * force
            disp[b][0] += ux * force
            disp[b][1] += uy * force
        temperature = 1.0 - step / max(int(iterations), 1)      # 线性冷却
        for i in range(count):
            dx, dy = disp[i]
            length = math.hypot(dx, dy) or 1e-6
            limit = 28.0 * temperature + 1.0
            move = min(length, limit)
            pos[i][0] += dx / length * move
            pos[i][1] += dy / length * move
            pos[i][0] = min(max(pos[i][0], 40.0), width - 40.0)
            pos[i][1] = min(max(pos[i][1], 40.0), height - 40.0)
    # ---- 碰撞消除（保证标签不重叠；这是可读性的关键一步）----
    # 间距取 72 px：中文标签 19 px 字号、短名 4~6 字 ⇒ 宽约 80~120 px，
    # 实测 46 px 时标签仍会互相压字（第一版截图即如此），故按「半个标签宽」留白。
    min_gap = 72.0
    for _round in range(30):
        moved = False
        for i in range(count):
            for j in range(i + 1, count):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.hypot(dx, dy)
                gap = min_gap + 6.0 * (weight[i] + weight[j] - 2.0)
                if dist >= gap:
                    continue
                if dist < 1e-6:
                    dx, dy, dist = rng.uniform(-1, 1), rng.uniform(-1, 1), 1e-6
                push = (gap - dist) / 2.0
                ux, uy = dx / dist, dy / dist
                pos[i][0] = min(max(pos[i][0] + ux * push, 40.0), width - 40.0)
                pos[i][1] = min(max(pos[i][1] + uy * push, 40.0), height - 40.0)
                pos[j][0] = min(max(pos[j][0] - ux * push, 40.0), width - 40.0)
                pos[j][1] = min(max(pos[j][1] - uy * push, 40.0), height - 40.0)
                moved = True
        if not moved:
            break
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    positions = {name: (round(pos[i][0], 2), round(pos[i][1], 2)) for name, i in index.items()}
    return LayoutResult(positions=positions, nodes=[dict(n) for n in nodes],
                        edges=[dict(e) for e in edges], width=int(width), height=int(height),
                        stats={"nodes": count, "edges": len(links), "iterations": int(iterations),
                               "layout_ms": elapsed_ms, "ideal_edge_len": round(ideal, 2),
                               "collision_gap": min_gap, "canvas": [int(width), int(height)]})


# ---------------------------------------------------------------------------
# PNG 渲染（PIL）
# ---------------------------------------------------------------------------
def render_png(layout: LayoutResult, out_path: Path | str, *, title: str,
               subtitle: str = "", legend: bool = True, label_min_degree: int = 2,
               radius_min: int = 7, radius_max: int = 26,
               font_path: str | None = None) -> dict[str, Any]:
    """用 ``PIL.ImageDraw`` 绘制力导向图谱 PNG（节点 + 关系连线 + 图例 + 统计角标）。"""
    from PIL import Image, ImageDraw, ImageFont

    width, height = layout.width, layout.height
    image = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(image, "RGBA")
    fonts = _fonts(font_path)

    # ① 关系连线（先画线，节点覆盖其上）
    degrees = [int(n.get("degree") or 0) for n in layout.nodes] or [1]
    max_degree = max(degrees) or 1
    edge_drawn = 0
    for edge in layout.edges:
        a = layout.positions.get(str(edge.get("head")))
        b = layout.positions.get(str(edge.get("tail")))
        if not a or not b:
            continue
        color = GROUP_COLORS.get(str(edge.get("group")), GROUP_COLORS["其它"])
        draw.line([a, b], fill=(*color, 130), width=1)
        edge_drawn += 1

    # ② 节点
    for node in layout.nodes:
        point = layout.positions.get(str(node.get("name")))
        if not point:
            continue
        degree = int(node.get("degree") or 0)
        ratio = degree / max_degree if max_degree else 0.0
        radius = int(radius_min + (radius_max - radius_min) * math.sqrt(max(ratio, 0.0)))
        color = TYPE_COLORS.get(str(node.get("type")), DEFAULT_COLOR)
        draw.ellipse([point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                     fill=(*color, 235), outline=(255, 255, 255, 255), width=2)

    # ③ 标签（只画度数达标的节点；短标签 + 白底 + **避免互相压字**）
    placed: list[tuple[float, float, float, float]] = []
    for node in layout.nodes:
        point = layout.positions.get(str(node.get("name")))
        if not point or int(node.get("degree") or 0) < int(label_min_degree):
            continue
        text = _short_label(node.get("name"))
        font = fonts["label"]
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        chosen = None
        for dx, dy in ((13, -th / 2), (13, -th - 14), (13, 12), (-tw - 17, -th / 2), (13, -th / 2 - 40)):
            tx = min(max(point[0] + dx, 6), width - tw - 12)
            ty = min(max(point[1] + dy, 104), height - th - 50)
            rect = (tx - 4, ty - 3, tx + tw + 4, ty + th + 3)
            if not any(_overlaps(rect, other) for other in placed):
                chosen = (tx, ty, rect)
                break
        if chosen is None:
            continue                        # 五个候选位都拥挤 ⇒ 宁可不画，避免压字
        tx, ty, rect = chosen
        placed.append(rect)
        draw.rectangle(list(rect), fill=(255, 255, 255, 226))
        draw.text((tx, ty), text, fill=(30, 34, 42), font=font)

    # ④ 标题与统计角标
    draw.rectangle([0, 0, width, 92], fill=(24, 34, 52))
    draw.text((32, 18), title, fill=(255, 255, 255), font=fonts["title"])
    if subtitle:
        draw.text((34, 60), subtitle, fill=(198, 212, 232), font=fonts["small"])
    stats = layout.stats
    badge = (f"节点 {stats.get('nodes')} · 边 {stats.get('edges')} · "
             f"布局 {stats.get('layout_ms')} ms · 画布 {width}×{height} · Schema {SCHEMA_VERSION}")
    draw.text((34, height - 38), badge, fill=(70, 78, 92), font=fonts["small"])

    # ⑤ 图例（实体类型 + 关系分组）
    legend_rows: list[tuple[str, tuple[int, int, int], str]] = []
    if legend:
        present = {str(n.get("type")) for n in layout.nodes}
        for key, info in ENTITY_TYPES.items():
            if key in present:
                count = sum(1 for n in layout.nodes if str(n.get("type")) == key)
                legend_rows.append((f"{info['zh']}（{count}）", TYPE_COLORS.get(key, DEFAULT_COLOR), "circle"))
        groups = {str(e.get("group")) for e in layout.edges}
        for name in RELATION_GROUPS:
            if name in groups:
                count = sum(1 for e in layout.edges if str(e.get("group")) == name)
                legend_rows.append((f"{name}关系（{count}）", GROUP_COLORS.get(name, DEFAULT_COLOR), "line"))
        box_w, row_h = 268, 30
        box_h = 30 + row_h * len(legend_rows)
        x0, y0 = width - box_w - 28, 116
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(255, 255, 255, 236),
                       outline=(180, 186, 196), width=1)
        draw.text((x0 + 14, y0 + 8), "图例", fill=(24, 34, 52), font=fonts["label"])
        for i, (text, color, kind) in enumerate(legend_rows):
            cy = y0 + 40 + i * row_h
            if kind == "circle":
                draw.ellipse([x0 + 16, cy - 7, x0 + 30, cy + 7], fill=(*color, 240),
                             outline=(255, 255, 255), width=1)
            else:
                draw.line([x0 + 16, cy, x0 + 34, cy], fill=(*color, 220), width=3)
            draw.text((x0 + 44, cy - 9), text, fill=(48, 54, 66), font=fonts["small"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    return {"png": str(out), "bytes": out.stat().st_size, "nodes": stats.get("nodes"),
            "edges": stats.get("edges"), "canvas": [width, height],
            "legend_rows": len(legend_rows), "edges_drawn": edge_drawn}


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    """两个矩形是否相交（轴对齐，用于标签避让）。"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _fonts(font_path: str | None) -> dict[str, Any]:
    """加载中文字体（Windows 自带 msyh/ simhei）；找不到时退回默认位图字体（不崩）。"""
    from PIL import ImageFont

    candidates = [font_path] if font_path else []
    candidates += [
        r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc",
    ]
    def load(size: int) -> Any:
        for path in candidates:
            if not path:
                continue
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001 —— 字体缺失时继续尝试下一个
                continue
        return ImageFont.load_default()

    return {"title": load(38), "label": load(19), "small": load(17), "tiny": load(15)}


# ---------------------------------------------------------------------------
# SVG / HTML（自包含，无外网依赖）
# ---------------------------------------------------------------------------
def render_svg(layout: LayoutResult, out_path: Path | str, *, title: str, subtitle: str = "") -> dict[str, Any]:
    """手写 SVG：矢量、可缩放、**不含任何外链**（字体用通用 family，不引用 CDN）。"""
    width, height = layout.width, layout.height
    parts: list[str] = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Microsoft YaHei, SimHei, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#fcfcfa"/>',
        f'<rect width="{width}" height="92" fill="#182234"/>',
        f'<text x="32" y="52" font-size="38" fill="#ffffff">{html.escape(title)}</text>',
        f'<text x="34" y="80" font-size="17" fill="#c6d4e8">{html.escape(subtitle)}</text>',
    ]
    for edge in layout.edges:
        a = layout.positions.get(str(edge.get("head")))
        b = layout.positions.get(str(edge.get("tail")))
        if not a or not b:
            continue
        color = GROUP_COLORS.get(str(edge.get("group")), GROUP_COLORS["其它"])
        stroke = "#%02x%02x%02x" % color
        parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{stroke}" '
                     f'stroke-opacity="0.5" stroke-width="1"/>')
    degrees = [int(n.get("degree") or 0) for n in layout.nodes] or [1]
    max_degree = max(degrees) or 1
    for node in layout.nodes:
        point = layout.positions.get(str(node.get("name")))
        if not point:
            continue
        ratio = int(node.get("degree") or 0) / max_degree
        radius = int(7 + 19 * math.sqrt(max(ratio, 0.0)))
        color = TYPE_COLORS.get(str(node.get("type")), DEFAULT_COLOR)
        fill = "#%02x%02x%02x" % color
        parts.append(f'<circle cx="{point[0]}" cy="{point[1]}" r="{radius}" fill="{fill}" '
                     f'stroke="#ffffff" stroke-width="2"><title>{html.escape(str(node.get("name")))}'
                     f'（{html.escape(str(node.get("type_zh")))}，度 {node.get("degree")}）</title></circle>')
    for node in layout.nodes:
        point = layout.positions.get(str(node.get("name")))
        if not point or int(node.get("degree") or 0) < 2:
            continue
        label = html.escape(str(node.get("name"))[:14])
        parts.append(f'<text x="{point[0] + 9}" y="{point[1] + 5}" font-size="17" fill="#232328" '
                     f'stroke="#ffffff" stroke-width="3" paint-order="stroke">{label}</text>')
    stats = layout.stats
    parts.append(f'<text x="34" y="{height - 24}" font-size="17" fill="#464e5c">'
                 f'节点 {stats.get("nodes")} · 边 {stats.get("edges")} · 布局 {stats.get("layout_ms")} ms '
                 f'· Schema {SCHEMA_VERSION}</text>')
    parts.append("</svg>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return {"svg": str(out), "bytes": out.stat().st_size,
            "external_refs": _count_external_refs(out)}


def render_html(layout: LayoutResult, out_path: Path | str, *, title: str, subtitle: str = "",
                full_graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """单文件 HTML：内嵌节点/边 JSON + 原生 Canvas 脚本（**无 CDN / 无外链**，断网可用）。"""
    payload = {
        "title": title, "subtitle": subtitle, "width": layout.width, "height": layout.height,
        "nodes": [{**n, "x": layout.positions.get(str(n.get("name")), (0, 0))[0],
                   "y": layout.positions.get(str(n.get("name")), (0, 0))[1]} for n in layout.nodes],
        "edges": layout.edges, "stats": layout.stats,
        "type_colors": {k: "#%02x%02x%02x" % v for k, v in TYPE_COLORS.items()},
        "group_colors": {k: "#%02x%02x%02x" % v for k, v in GROUP_COLORS.items()},
        "schema_version": SCHEMA_VERSION,
        "overview": (full_graph or {}).get("stats") or {},
    }
    page = _HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {"html": str(out), "bytes": out.stat().st_size, "external_refs": _count_external_refs(out)}


def _count_external_refs(path: Path) -> int:
    """统计产物里的外链数量（交付要求「自包含、无外网依赖」⇒ 必须为 0）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    count = 0
    for token in ("http://", "https://", "//cdn.", "src=\"//", "href=\"//"):
        count += text.count(token)
    # 允许 SVG 命名空间声明（不是网络请求）
    count -= text.count("http://www.w3.org/2000/svg")
    return max(count, 0)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>工单8 知识图谱可视化</title>
<style>
  body { margin:0; font-family:"Microsoft YaHei",SimHei,sans-serif; background:#fcfcfa; color:#1c2330; }
  header { background:#182234; color:#fff; padding:14px 24px; }
  header h1 { margin:0; font-size:20px; } header p { margin:6px 0 0; color:#c6d4e8; font-size:13px; }
  #wrap { position:relative; }
  canvas { display:block; width:100%; height:auto; background:#fcfcfa; }
  #legend { position:absolute; right:16px; top:16px; background:rgba(255,255,255,.94);
            border:1px solid #b4bac4; border-radius:6px; padding:10px 14px; font-size:13px; }
  #legend div { display:flex; align-items:center; gap:8px; margin:4px 0; }
  .dot { width:13px; height:13px; border-radius:50%; border:1px solid #fff; }
  .bar { width:20px; height:3px; }
  #tip { position:absolute; pointer-events:none; background:#1c2330; color:#fff; font-size:12px;
         padding:4px 8px; border-radius:4px; display:none; }
</style>
</head>
<body>
<header><h1>__TITLE__</h1><p>单文件自包含（无外网依赖）· 力导向布局 · 节点按实体类型着色、按度数定大小</p></header>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="legend"></div>
  <div id="tip"></div>
</div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const dpr = window.devicePixelRatio || 1;
cv.width = DATA.width * dpr; cv.height = DATA.height * dpr;
cv.style.height = (DATA.height * (cv.clientWidth || DATA.width) / DATA.width) + 'px';
ctx.scale(dpr, dpr);
const idx = {}; DATA.nodes.forEach(n => idx[n.name] = n);
function draw() {
  ctx.clearRect(0, 0, DATA.width, DATA.height);
  DATA.edges.forEach(e => {
    const a = idx[e.head], b = idx[e.tail];
    if (!a || !b) return;
    ctx.strokeStyle = DATA.group_colors[e.group] || '#787878';
    ctx.globalAlpha = .45; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  });
  ctx.globalAlpha = 1;
  const maxDeg = Math.max(1, ...DATA.nodes.map(n => n.degree || 0));
  DATA.nodes.forEach(n => {
    const r = 7 + 19 * Math.sqrt((n.degree || 0) / maxDeg);
    ctx.fillStyle = DATA.type_colors[n.type] || '#6e6e6e';
    ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  });
  ctx.font = '17px "Microsoft YaHei",SimHei,sans-serif';
  DATA.nodes.forEach(n => {
    if ((n.degree || 0) < 2) return;
    const t = n.name.length > 14 ? n.name.slice(0, 14) : n.name;
    ctx.lineWidth = 3; ctx.strokeStyle = '#fff'; ctx.strokeText(t, n.x + 9, n.y + 5);
    ctx.fillStyle = '#232328'; ctx.fillText(t, n.x + 9, n.y + 5);
  });
}
const legend = document.getElementById('legend');
const typeRows = Object.keys(DATA.type_colors).filter(k => DATA.nodes.some(n => n.type === k));
typeRows.forEach(k => {
  const c = DATA.nodes.filter(n => n.type === k).length;
  legend.insertAdjacentHTML('beforeend',
    `<div><span class="dot" style="background:${DATA.type_colors[k]}"></span>${k}（${c}）</div>`);
});
Object.keys(DATA.group_colors).forEach(k => {
  const c = DATA.edges.filter(e => e.group === k).length;
  if (!c) return;
  legend.insertAdjacentHTML('beforeend',
    `<div><span class="bar" style="background:${DATA.group_colors[k]}"></span>${k}（${c}）</div>`);
});
legend.insertAdjacentHTML('beforeend',
  `<div style="border-top:1px solid #d6dae2;margin-top:6px;padding-top:6px;color:#5a6272">
   节点 ${DATA.stats.nodes} · 边 ${DATA.stats.edges}<br/>布局 ${DATA.stats.layout_ms} ms · Schema ${DATA.schema_version}</div>`);
draw();
const tip = document.getElementById('tip');
cv.addEventListener('mousemove', ev => {
  const rect = cv.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * DATA.width / rect.width;
  const y = (ev.clientY - rect.top) * DATA.height / rect.height;
  let hit = null, best = 1e9;
  DATA.nodes.forEach(n => { const d = Math.hypot(n.x - x, n.y - y); if (d < best) { best = d; hit = n; } });
  if (hit && best < 26) {
    tip.style.display = 'block';
    tip.style.left = (ev.clientX - rect.left + 14) + 'px';
    tip.style.top = (ev.clientY - rect.top + 10) + 'px';
    tip.textContent = `${hit.name}（${hit.type}，度 ${hit.degree}，出现 ${hit.count} 次）`;
  } else { tip.style.display = 'none'; }
});
</script>
</body>
</html>
"""
