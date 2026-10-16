# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""图谱可视化：只用 Pillow 手绘（本机无 matplotlib），中文标签 + 真实数字脚注。

两张图（文件名中文、按测试内容命名，落 ``测试/测试截图/``）：
  * ``知识图谱可视化.png``     —— 力导向布局的实体关系图 + 右侧 Top 关系面板 + 类型图例
  * ``知识图谱类型分布.png``   —— 实体类型 / 关系类型分布条形图（真实计数）

纪律：图上出现的每个数字都来自本次实跑的 nodes/edges，**不做任何装饰性虚构**；
无计数依据的元素（如节点位置）才使用布局算法生成。
"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from .config import Cfg

# 明色系配色（图例与节点共用）
_PALETTE: list[tuple[int, int, int]] = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (188, 189, 34),
    (23, 190, 207), (174, 199, 232), (255, 187, 120), (152, 223, 138),
    (255, 152, 150), (197, 176, 213), (196, 156, 148), (247, 182, 210),
    (199, 199, 199), (219, 219, 141),
]
_BG = (255, 255, 255)
_INK = (33, 37, 41)
_GREY = (170, 175, 180)


def _font(size: int, cfg: Cfg | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """取中文字体；缺失时降级为 PIL 默认字体（会有方块，但不崩）。"""
    candidates: list[str] = []
    if cfg is not None:
        try:
            candidates.append(str(cfg.viz.font_path))
        except Exception:  # noqa: BLE001
            pass
    candidates += [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001 - 字体缺失不应阻断出图
            continue
    return ImageFont.load_default()


def type_color(entity_type: str, types: Sequence[str]) -> tuple[int, int, int]:
    """类型 → 颜色（按出现顺序稳定映射）。"""
    order = list(types)
    idx = order.index(entity_type) if entity_type in order else len(order)
    return _PALETTE[idx % len(_PALETTE)]


def spring_layout(names: Sequence[str], edges: Sequence[dict[str, Any]], *,
                  iterations: int, seed: int, width: int, height: int,
                  margin: int = 110) -> dict[str, tuple[float, float]]:
    """Fruchterman-Reingold 简化版力导向布局（纯 Python，n≤60 时开销可忽略）。"""
    rng = random.Random(seed)
    n = max(len(names), 1)
    pos: dict[str, list[float]] = {
        name: [rng.uniform(margin, width - margin), rng.uniform(margin, height - margin)] for name in names
    }
    area = float(max(width - 2 * margin, 10) * max(height - 2 * margin, 10))
    k = math.sqrt(area / n)
    for it in range(max(int(iterations), 1)):
        disp: dict[str, list[float]] = {name: [0.0, 0.0] for name in names}
        for i, a in enumerate(names):                      # 斥力
            for b in names[i + 1:]:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                dist2 = dx * dx + dy * dy or 0.01
                dist = math.sqrt(dist2)
                force = k * k / dist
                ux, uy = dx / dist, dy / dist
                disp[a][0] += ux * force
                disp[a][1] += uy * force
                disp[b][0] -= ux * force
                disp[b][1] -= uy * force
        index = {name: i for i, name in enumerate(names)}
        for e in edges:                                    # 引力
            a, b = str(e["head"]), str(e["tail"])
            if a not in index or b not in index:
                continue
            dx = pos[a][0] - pos[b][0]
            dy = pos[a][1] - pos[b][1]
            dist = math.sqrt(dx * dx + dy * dy) or 0.01
            force = dist * dist / k
            ux, uy = dx / dist, dy / dist
            disp[a][0] -= ux * force
            disp[a][1] -= uy * force
            disp[b][0] += ux * force
            disp[b][1] += uy * force
        temp = (width / 8.0) * (1.0 - it / float(max(int(iterations), 1))) + 0.5
        for name in names:
            dx, dy = disp[name]
            dist = math.hypot(dx, dy) or 0.01
            step = min(dist, temp)
            pos[name][0] = min(max(pos[name][0] + dx / dist * step, margin * 0.5), width - margin * 0.5)
            pos[name][1] = min(max(pos[name][1] + dy / dist * step, margin * 0.9), height - margin * 0.5)
    return {name: (pos[name][0], pos[name][1]) for name in names}


def draw_graph(nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]], path: Path | str, *,
               cfg: Cfg, title: str, footnote: str, max_nodes: int | None = None,
               max_edges: int | None = None) -> Path:
    """画实体关系图 PNG，返回输出路径。"""
    v = cfg.viz
    max_nodes = int(max_nodes or v.max_nodes)
    max_edges = int(max_edges or v.max_edges)
    dpi = int(v.dpi)
    width = int(v.figsize[0] * dpi)
    height = int(v.figsize[1] * dpi)

    ranked = sorted(nodes, key=lambda n: (-int(n.get("degree", 0)), -int(n["freq"]), str(n["name"])))
    shown = ranked[:max_nodes]
    shown_names = {str(n["name"]) for n in shown}
    shown_edges = [e for e in edges if str(e["head"]) in shown_names and str(e["tail"]) in shown_names][:max_edges]

    img = Image.new("RGB", (width, height), _BG)
    dr = ImageDraw.Draw(img)

    f_title = _font(52, cfg)
    f_sub = _font(26, cfg)
    f_node = _font(22, cfg)
    f_small = _font(20, cfg)
    f_tiny = _font(18, cfg)

    dr.text((width // 2, 42), title, font=f_title, fill=_INK, anchor="mm")
    dr.text((width // 2, 100), footnote, font=f_sub, fill=(90, 95, 100), anchor="mm")

    panel_w = int(width * 0.30)
    graph_w = width - panel_w
    pos = spring_layout([str(n["name"]) for n in shown], shown_edges,
                        iterations=int(v.layout_iterations), seed=42,
                        width=graph_w, height=height - 200, margin=110)
    shift_y = 150
    pos = {k: (vx, vy + shift_y) for k, (vx, vy) in pos.items()}

    types_order = list(dict.fromkeys(str(n["type"]) for n in shown))
    max_w = max((int(e["weight"]) for e in shown_edges), default=1)
    for e in shown_edges:                                   # 边
        a, b = pos[str(e["head"])], pos[str(e["tail"])]
        w = 1 + int(4 * int(e["weight"]) / max_w)
        dr.line([a, b], fill=(205, 210, 215), width=w)
    for n in shown:                                         # 节点
        x, y = pos[str(n["name"])]
        r = 10 + int(22 * min(int(n["freq"]), 12) / 12)
        color = type_color(str(n["type"]), types_order)
        dr.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=3)
        label = str(n["name"])
        if len(label) > 14:
            label = label[:13] + "…"
        dr.text((x, y + r + 14), label, font=f_node, fill=_INK, anchor="ma")

    # 右侧面板：类型图例 + Top 关系（真实计数）
    px = graph_w + 24
    py = 160
    dr.rectangle([px - 12, py - 20, width - 24, height - 60], outline=(220, 224, 228), width=2)
    dr.text((px, py), "实体类型图例", font=f_small, fill=_INK)
    py += 36
    ent_counter = Counter(str(n["type"]) for n in shown)
    for t in types_order:
        dr.rectangle([px, py, px + 26, py + 18], fill=type_color(t, types_order))
        dr.text((px + 38, py - 2), f"{t}（{ent_counter.get(t, 0)}）", font=f_tiny, fill=_INK)
        py += 28
    py += 16
    dr.text((px, py), "权重最高的关系", font=f_small, fill=_INK)
    py += 34
    for e in shown_edges[:14]:
        head, tail = str(e["head"]), str(e["tail"])
        if len(head) > 11:
            head = head[:10] + "…"
        if len(tail) > 11:
            tail = tail[:10] + "…"
        dr.text((px, py), f"{head} —{e['rel']}→ {tail}  ×{e['weight']}", font=f_tiny, fill=(60, 65, 70))
        py += 26
    py += 16
    dr.text((px, py), f"入图节点 {len(shown)}/{len(nodes)} 个", font=f_tiny, fill=_INK)
    dr.text((px, py + 26), f"入图关系 {len(shown_edges)}/{len(edges)} 条", font=f_tiny, fill=_INK)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def draw_type_distribution(entity_dist: dict[str, int], relation_dist: dict[str, int],
                           path: Path | str, *, cfg: Cfg, title: str, footnote: str) -> Path:
    """画「实体类型 / 关系类型」分布条形图（真实计数）。"""
    v = cfg.viz
    dpi = int(v.dpi)
    width = int(v.figsize[0] * dpi)
    height = int(v.figsize[1] * dpi)
    img = Image.new("RGB", (width, height), _BG)
    dr = ImageDraw.Draw(img)
    f_title = _font(52, cfg)
    f_sub = _font(26, cfg)
    f_lab = _font(24, cfg)
    f_val = _font(22, cfg)

    dr.text((width // 2, 42), title, font=f_title, fill=_INK, anchor="mm")
    dr.text((width // 2, 100), footnote, font=f_sub, fill=(90, 95, 100), anchor="mm")

    for panel, (dist, panel_title, color) in enumerate([
        (entity_dist, "实体类型分布（节点数）", (31, 119, 180)),
        (relation_dist, "关系类型分布（边数）", (255, 127, 14)),
    ]):
        left = 120 + panel * (width // 2)
        top = 200
        bar_max = width // 2 - 320
        items = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
        max_val = max((val for _k, val in items), default=1)
        dr.text((left, top - 46), panel_title, font=f_lab, fill=_INK)
        dy = 46
        for i, (name, val) in enumerate(items):
            y = top + i * dy
            label = name if len(name) <= 10 else name[:9] + "…"
            dr.text((left, y + 6), label, font=f_lab, fill=_INK)
            w = int(bar_max * val / max_val) if max_val else 0
            dr.rectangle([left + 230, y + 2, left + 230 + w, y + 30], fill=color)
            dr.text((left + 240 + w, y + 4), str(val), font=f_val, fill=(60, 65, 70))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
