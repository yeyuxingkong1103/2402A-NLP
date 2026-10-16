# -*- coding: utf-8 -*-
"""工单4 矢量框图结构解析（diagram_parser）。

任务：**人工智能NLP-RAG 项目-金融问答系统部署**（RAG 内核沿用 工单7 实现）。

把 PDF 里的「组织结构图 / 流程图」这类**矢量框图**还原成**父子层级树**，
并输出「人类可读结构树文本 + 可检索归纳句 + 每个节点的 bbox」，供 RAG 索引引用。

设计依据：``设计/图像解析方案.md`` §3。

------------------------------------------------------------------------
本模块相对设计稿的关键改进：**用连线证据（connector graph）代替纯几何猜测**
------------------------------------------------------------------------
设计稿 §3.3 的父子判定是「上一层中水平区间覆盖子框中心 x 的框」。
实测该规则**在物理页 39 会失败**：
    大客户销售部 box x∈[341.82, 368.82]，而 6 个销售处 box 的**中心 x**
    分别落在 247.3/292.3/337.3/382.3/427.3/472.3 —— **没有一个**被父框区间覆盖，
    于是几何规则会把 6 个销售处拆给「电话及网络销售部 / 渠道销售部 / 国际贸易部」。

实测（对物理页 39 的 ``page.get_drawings()`` 逐 item 转储：该页共 **31 条绘制路径 / 36 个
``l``（线段）item**，没有别的连线载体；该一次性转储脚本按「交付物只落五个中文文件夹」的纪律
**不入交付物**，用下面 3 行等价命令即可复现）发现该页的连线**全部在
``page.get_drawings()`` 的 ``l``（线段）item 里**，于是本模块改为：

.. code-block:: python

    import pymupdf
    page = pymupdf.open("研发/data/raw/招股说明书2.pdf")[38]   # 0-based 38 = 物理页 39
    items = [it[0] for d in page.get_drawings() for it in d.get("items", [])]
    print(items.count("l"))                                   # 实测 36（连线全部为线段 item）

    ① 把所有线段端点（以及线段与矩形边界的交点）建成「折点图」；
    ② 从某框 **下边** 上的折点出发做 **「遇框即停」的 BFS**（halting BFS）：
       走到另一个框的边界就记为该框的子节点，**不再穿过该框继续走**；
    ③ 父节点同理：从全部 attach 折点出发，取「停下的、且位于本框上方最近」的框。

该算法在页 39 上**精确**还原出：
    销售部 → 4 个下级；大客户销售部 → 6 个销售处；
    且「上海分公司」不会被算作销售处（它挂在总经理侧）。
若某页**没有连线**，自动退回几何分层规则（``evidence="geometry"``），并如实标注。

------------------------------------------------------------------------
文本单元构造（竖排文字的坑）
------------------------------------------------------------------------
页 39 下半部分是**竖排文字、每个汉字一个 span**。实测：
  * ✅ 必须 **按 x 列聚类（±6pt）+ 列内按 y 升序拼接**；
  * ❌ 按 y 行聚类会把 6 个方框的首字串成 `财场息销政物` 乱码；
  * 补充规则 R7：同一矩形框内相邻列需合并（处理「薪酬与考核委员会」框内换行缩进 x 144.6→155.1）。
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import fitz  # PyMuPDF

__all__ = [
    "Box",
    "TextUnit",
    "DiagramResult",
    "find_diagram_pages",
    "parse_diagram_page",
    "looks_like_diagram",
    "structure_tree_text",
    "unit_word_of",
    "collect_cross_references",
]

# 框图判定阈值（设计 §3.1）
MIN_BOXES = 6
MIN_BOX_W = 20.0
MIN_BOX_H = 15.0
MIN_TEXT_UNITS = 3

# 几何容差
TOL = 1.6           # 点/边贴合容差（pt）
COL_TOL = 6.0       # 列聚类容差（pt，实测口径）
COL_GAP = 14.0      # 同列相邻字符的最大 y 间距（pt）

_UNIT_WORD_MAP = {"部": "部门"}   # 量词推断：共同后缀「部」→「部门」；其余原样使用


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    name: str = ""
    children: list[str] = field(default_factory=list)
    parent: str = ""
    unit_bbox: list[float] = field(default_factory=list)   # 文字实际 bbox（可能小于框）

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class TextUnit:
    x: float
    y0: float
    y1: float
    text: str
    bbox: list[float]


@dataclass
class DiagramResult:
    corpus: str
    page: int
    page_basis: str = "1-based-physical"
    n_boxes: int = 0
    n_segments: int = 0
    boxes: list[Box] = field(default_factory=list)
    evidence: str = "connector_lines"       # connector_lines | geometry
    node_bbox: dict[str, list[float]] = field(default_factory=dict)
    children_count: dict[str, int] = field(default_factory=dict)
    edges: list[list[str]] = field(default_factory=list)
    tree_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "page": self.page,
            "page_basis": self.page_basis,
            "n_boxes": self.n_boxes,
            "n_segments": self.n_segments,
            "evidence": self.evidence,
            "nodes": [b.name for b in self.boxes if b.name],
            "node_bbox": self.node_bbox,
            "children_count": self.children_count,
            "edges": self.edges,
            "tree_text": self.tree_text,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# 文本单元：span → 竖排/横排文字单元
# ---------------------------------------------------------------------------
def extract_spans(page: "fitz.Page") -> list[tuple[list[float], str]]:
    """取该页所有 span 的 (bbox, text)；bbox 为 (x0,y0,x1,y1)。"""

    out: list[tuple[list[float], str]] = []
    d = page.get_text("dict")
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text") or ""
                if txt.strip():
                    out.append(([float(v) for v in span["bbox"]], txt))
    # 若无 dict（异常 PDF），退回 rawdict→blocks
    if not out:
        for b in page.get_text("blocks"):
            txt = str(b[4] or "")
            if txt.strip():
                out.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], txt))
    return out


def _cluster_columns(spans: Sequence[tuple[list[float], str]]) -> list[TextUnit]:
    """按 x 列聚类（±COL_TOL）+ 列内按 y 升序拼接。

    实测口径（设计 §3.2）：这是竖排文字唯一正确的拼法。
    """

    cols: list[dict[str, Any]] = []
    # 先按 (x0, y0) 排序，保证同列字符按阅读顺序进入
    for bbox, txt in sorted(spans, key=lambda s: (round(s[0][0], 1), round(s[0][1], 1))):
        x0, y0, x1, y1 = bbox
        placed = False
        for c in cols:
            # 列内相邻：x 接近，且 y 间距 < COL_GAP（允许轻微越界）
            if abs(c["x"] - x0) <= COL_TOL and (0.0 - COL_TOL) <= (y0 - c["y_last"]) < COL_GAP:
                c["chars"].append((y0, txt))
                c["y_last"] = y1
                c["y1"] = max(c["y1"], y1)
                c["x1"] = max(c["x1"], x1)
                c["x"] = (c["x"] + x0) / 2.0
                placed = True
                break
        if not placed:
            cols.append({"x": x0, "y0": y0, "y1": y1, "x1": x1, "y_last": y1, "chars": [(y0, txt)]})

    units: list[TextUnit] = []
    for c in cols:
        chars = sorted(c["chars"], key=lambda t: t[0])
        text = "".join(t[1] for t in chars)
        text = re.sub(r"\s+", "", text)
        if not text:
            continue
        units.append(
            TextUnit(x=c["x"], y0=c["y0"], y1=c["y1"], text=text,
                     bbox=[min(c["x"], c["x"]), c["y0"], c["x1"], c["y1"]])
        )
    return units


# ---------------------------------------------------------------------------
# 方框与连线提取
# ---------------------------------------------------------------------------
def extract_boxes_and_segments(
    page: "fitz.Page", page_rect: "fitz.Rect"
) -> tuple[list[Box], list[tuple[tuple[float, float], tuple[float, float]]]]:
    """从 ``get_drawings()`` 取候选矩形（=方框）与线段（=连线）。"""

    rects: list[tuple[float, float, float, float]] = []
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    page_area = float(page_rect.width * page_rect.height)
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for d in drawings:
        for it in d.get("items", []):
            if it[0] == "re":
                r = it[1]
                x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                w, h = x1 - x0, y1 - y0
                if w < MIN_BOX_W or h < MIN_BOX_H:
                    continue
                # 整页边框（外框）不是节点
                if w * h > page_area * 0.85:
                    continue
                rects.append((x0, y0, x1, y1))
            elif it[0] == "l":
                p1, p2 = it[1], it[2]
                segs.append(((float(p1.x), float(p1.y)), (float(p2.x), float(p2.y))))
            elif it[0] == "c":
                # 贝塞尔：用折线近似（p1→p2→p3→p4），足够做连通性判定
                pts = [it[1], it[2], it[3], it[4]]
                for a, b in zip(pts, pts[1:]):
                    segs.append(((float(a.x), float(a.y)), (float(b.x), float(b.y))))

    # 去重（同一矩形可能被多段命中）
    seen: set[tuple[float, float, float, float]] = set()
    boxes: list[Box] = []
    for r in rects:
        key = tuple(round(v, 2) for v in r)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(Box(*r))
    return boxes, segs


def _quant(p: tuple[float, float]) -> tuple[float, float]:
    return (round(p[0] * 2.0) / 2.0, round(p[1] * 2.0) / 2.0)


def _cluster_points(points: Sequence[tuple[float, float]], tol: float = 1.2) -> tuple[list[tuple[float, float]], list[int]]:
    """把距离 ≤ tol 的点合并成同一折点（**不用网格量化**）。

    为什么必须这样做：实测页 39 的竖线端点是 ``(287.82, 223.44)`` 与 ``(287.88, 301.44)``，
    两条端点 y 相差 0.06 pt 的线如果按 0.5 pt 网格量化，会被切到相邻格子，
    结果「干管」与「母线」虽然几何相接却在图里不连通，整棵树就散架。

    返回 (代表点列表, 每个输入点所属的簇号)。
    """

    reps: list[list[float]] = []      # [x, y, count]，代表点取簇内均值
    ids: list[int] = []
    for (x, y) in points:
        best, best_d = -1, tol
        for i, r in enumerate(reps):
            d = ((r[0] - x) ** 2 + (r[1] - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = i, d
        if best < 0:
            reps.append([x, y, 1.0])
            ids.append(len(reps) - 1)
        else:
            r = reps[best]
            r[0] = (r[0] * r[2] + x) / (r[2] + 1.0)
            r[1] = (r[1] * r[2] + y) / (r[2] + 1.0)
            r[2] += 1.0
            ids.append(best)
    return [(r[0], r[1]) for r in reps], ids


def _point_on_seg(pt: tuple[float, float], p: tuple[float, float], q: tuple[float, float], tol: float = 1.2) -> bool:
    """点是否落在线段上（含端点），用于把「T 形接头」接起来。"""

    px, py = pt
    x0, y0 = p
    x1, y1 = q
    dx, dy = x1 - x0, y1 - y0
    seg_len = (dx * dx + dy * dy) ** 0.5
    if seg_len < 1e-6:
        return ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5 <= tol
    t = ((px - x0) * dx + (py - y0) * dy) / (seg_len * seg_len)
    if t < -0.02 or t > 1.02:
        return False
    proj = (x0 + t * dx, y0 + t * dy)
    return ((px - proj[0]) ** 2 + (py - proj[1]) ** 2) ** 0.5 <= tol


def _seg_boundary_hits(
    p: tuple[float, float],
    q: tuple[float, float],
    boxes: Sequence[Box],
    tol: float = TOL,
) -> list[tuple[float, float]]:
    """线段与若干矩形**边界**的交点（含端点贴合），用于把长线段在框边处分断。"""

    hits: list[tuple[float, float]] = []
    px, py = p
    qx, qy = q
    lo_x, hi_x = sorted((px, qx))
    lo_y, hi_y = sorted((py, qy))
    for b in boxes:
        if b.x1 < lo_x - tol or b.x0 > hi_x + tol or b.y1 < lo_y - tol or b.y0 > hi_y + tol:
            continue
        cand: list[tuple[float, float]] = []
        if abs(py - qy) <= 0.6:                      # 水平线
            if b.y0 - tol <= py <= b.y1 + tol:
                for xe in (b.x0, b.x1):
                    if lo_x - tol <= xe <= hi_x + tol:
                        cand.append((xe, py))
        elif abs(px - qx) <= 0.6:                    # 垂直线
            if b.x0 - tol <= px <= b.x1 + tol:
                for ye in (b.y0, b.y1):
                    if lo_y - tol <= ye <= hi_y + tol:
                        cand.append((px, ye))
        else:                                        # 斜线：四边求交
            for (ax, ay), (bx, by) in (
                ((b.x0, b.y0), (b.x1, b.y0)),
                ((b.x1, b.y0), (b.x1, b.y1)),
                ((b.x1, b.y1), (b.x0, b.y1)),
                ((b.x0, b.y1), (b.x0, b.y0)),
            ):
                pt = _seg_seg_intersect(p, q, (ax, ay), (bx, by))
                if pt is not None:
                    cand.append(pt)
        for c in cand:
            if min(lo_x, lo_y) - 50 <= c[0] <= max(hi_x, hi_y) + 50:
                hits.append(c)
    return hits


def _seg_seg_intersect(p1, p2, p3, p4):
    """两线段交点（平行/无交返回 None）。"""

    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / den
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / den
    if -0.02 <= t <= 1.02 and -0.02 <= u <= 1.02:
        return (p1[0] + t * d1x, p1[1] + t * d1y)
    return None


def build_point_graph(
    segs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    boxes: Sequence[Box],
) -> tuple[dict[int, set[int]], list[tuple[float, float]]]:
    """把线段转成**折点图**（节点 = 合并后的折点，边 = 线段）。

    两个必须做的接线动作（缺一个都会让层级树断掉）：

    a) **框边分断**：一条长线段穿过若干矩形边界时，在交点处切成多段，
       这样「母线」才能停在每个子框的上边。
    b) **T 形接头**：某条线段的端点落在另一条线段**内部**时，把后者在该处切断。
       实测页 39 的「干管」端点 ``(287.85, 301.44)`` 正落在母线内部，
       不做这一步则总经理与 6 个部门永远连不上。

    返回 ``(邻接表, 代表点列表)``；邻接表的键是折点编号。
    """

    # ---- 1) 收集所有原始点：线段端点 + 线段与框边的交点 ----
    raw_pts: list[tuple[float, float]] = []
    for p, q in segs:
        raw_pts.append(p)
        raw_pts.append(q)
        raw_pts.extend(_seg_boundary_hits(p, q, boxes))

    # ---- 2) T 形接头：把落在其它线段内部的点也算进来 ----
    if len(segs) <= 3000:
        base_pts = list(raw_pts)          # 必须先取快照：下面要往 raw_pts 追加，边遍历边追加会死循环
        for p, q in segs:
            for (x, y) in base_pts:
                if _point_on_seg((x, y), p, q):
                    raw_pts.append((x, y))

    reps, ids = _cluster_points(raw_pts)
    n_pts = len(raw_pts)

    graph: dict[int, set[int]] = {}

    def add_edge(a: int, b: int) -> None:
        if a == b:
            return
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    # 每条线段：把落在它上面的所有折点按参数排序后相邻相连
    for p, q in segs:
        on: list[tuple[float, int]] = []
        dx, dy = q[0] - p[0], q[1] - p[1]
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        for idx in range(n_pts):
            if _point_on_seg(raw_pts[idx], p, q):
                t = ((raw_pts[idx][0] - p[0]) * dx + (raw_pts[idx][1] - p[1]) * dy) / norm
                on.append((t, ids[idx]))
        on.sort()
        for (_, a), (_, b) in zip(on, on[1:]):
            add_edge(a, b)
    return graph, reps


def attach_points(
    b: Box, graph: dict[int, set[int]], reps: Sequence[tuple[float, float]], tol: float = TOL
) -> list[int]:
    """落在某个框**边界**上的折点编号（外扩 tol），即该框的连线上/下端点。"""

    out = []
    for i in graph:
        x, y = reps[i]
        on_v = (abs(x - b.x0) <= tol or abs(x - b.x1) <= tol) and (b.y0 - tol <= y <= b.y1 + tol)
        on_h = (abs(y - b.y0) <= tol or abs(y - b.y1) <= tol) and (b.x0 - tol <= x <= b.x1 + tol)
        if on_v or on_h:
            out.append(i)
    return out


def _owner_boxes(
    pt: tuple[float, float], boxes: Sequence[Box], exclude: int | None = None, tol: float = TOL
) -> list[int]:
    idx = []
    x, y = pt
    for i, b in enumerate(boxes):
        if i == exclude:
            continue
        on_v = (abs(x - b.x0) <= tol or abs(x - b.x1) <= tol) and (b.y0 - tol <= y <= b.y1 + tol)
        on_h = (abs(y - b.y0) <= tol or abs(y - b.y1) <= tol) and (b.x0 - tol <= x <= b.x1 + tol)
        if on_v or on_h:
            idx.append(i)
    return idx


def halting_reach(
    starts: Sequence[int],
    graph: dict[int, set[int]],
    reps: Sequence[tuple[float, float]],
    boxes: Sequence[Box],
    owner: int,
    tol: float = TOL,
) -> dict[int, tuple[float, float]]:
    """「遇框即停」BFS：从某框的 attach 折点出发，走到别的框边界就记录并**停止**。

    这是本模块的核心：如果允许穿过别的框继续走，
    「大客户销售部」会顺着母线把它的兄弟（电话及网络销售部等）也当成长辈/晚辈，
    层级树立刻变成一团。
    """

    visited: set[int] = set()
    hits: dict[int, tuple[float, float]] = {}
    dq: deque[int] = deque()
    for s in starts:
        if s not in visited:
            visited.add(s)
            dq.append(s)
    while dq:
        node = dq.popleft()
        for nb in graph.get(node, ()):  # type: ignore[arg-type]
            if nb in visited:
                continue
            owners = _owner_boxes(reps[nb], boxes, exclude=owner, tol=tol)
            if owners:
                for oi in owners:
                    hits.setdefault(oi, reps[nb])
                visited.add(nb)       # 命中即停，不再穿过该框
                continue
            visited.add(nb)
            dq.append(nb)
    return hits


# ---------------------------------------------------------------------------
# 几何回退规则（无连线时用）
# ---------------------------------------------------------------------------
def _geometric_edges(boxes: Sequence[Box]) -> list[list[int]]:
    """几何分层父子判定（设计 §3.3 的 1)~4) 步）。"""

    if not boxes:
        return []
    heights = sorted(b.y1 - b.y0 for b in boxes)
    med_h = heights[len(heights) // 2]
    tol_y = max(6.0, med_h * 0.6)

    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].y0, boxes[i].cx))
    layers: list[list[int]] = []
    for i in order:
        placed = False
        for layer in layers:
            ref = sum(boxes[j].y0 for j in layer) / len(layer)
            if abs(boxes[i].y0 - ref) <= tol_y:
                layer.append(i)
                placed = True
                break
        if not placed:
            layers.append([i])
    layers.sort(key=lambda L: min(boxes[j].y0 for j in L))

    edges: list[list[int]] = []
    for li in range(1, len(layers)):
        for ci in layers[li]:
            cb = boxes[ci]
            best, best_d = None, 1e9
            for lj in range(li - 1, -1, -1):
                for pi in layers[lj]:
                    pb = boxes[pi]
                    if pb.x0 - 1 <= cb.cx <= pb.x1 + 1:
                        d = abs(cb.cx - pb.cx)
                    else:
                        d = abs(cb.cx - pb.cx) + 500.0   # 覆盖不了就重罚
                    if d < best_d:
                        best, best_d = pi, d
                if best is not None and best_d < 500.0:
                    break
            if best is not None:
                edges.append([best, ci])
    return edges


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _looks_like_table_grid(boxes: Sequence[Box], max_boxes: int = 200) -> bool:
    """把**表格单元格网格**与**组织结构图方框**区分开。

    实测教训：两份 PDF 里表格极多，单元格边框也是 ``type=='re'`` 的矩形，
    仅用「矩形数 ≥ 6」会把整本招股书的表格页判成框图（实测 PDF2 有 36 页通过该阈值）。

    判据 = **「相邻贴边矩形对 / 总矩形对」**（``adj_ratio``）：表格单元格彼此紧贴
    （间距 ≈ 0），框图方框之间有明显留白。

    实测标定（T2 2026-10 对 26 个疑似页扫描 ``boxes`` / ``segs`` / ``adj_ratio`` / ``fill``
    得到下表；该一次性标定脚本按「交付物只落五个中文文件夹」的纪律**不入交付物**，
    ``adj_ratio`` 的复现口径就是本函数下方的邻接对统计，物理页 → adj_ratio）：

    | 真框图 | 值 | 表格页 | 值 |
    | --- | --- | --- | --- |
    | 38 股东结构图 | 0.016 | 248 / 291 / 294 | 0.042 / 0.048 / 0.058 |
    | 39 组织结构图 | **0.000** | 179 / 181 / 265 | 0.082 / 0.179 / 0.142 |
    | 106 业务流程图 | **0.000** | 295 / 297 / 292 | 0.057 / 0.026 / 0.044 |

    → 阈值取 **0.025**：真框图 ≤ 0.016，表格页 ≥ 0.026，两侧都留有余量。
    """

    bs = list(boxes)[:max_boxes]
    n = len(bs)
    if n < 6:
        return False
    adj = 0
    total = 0
    for i in range(n):
        a = bs[i]
        for j in range(i + 1, n):
            b = bs[j]
            total += 1
            ovl_x = min(a.x1, b.x1) - max(a.x0, b.x0)
            ovl_y = min(a.y1, b.y1) - max(a.y0, b.y0)
            gap_x = max(a.x0, b.x0) - min(a.x1, b.x1)
            gap_y = max(a.y0, b.y0) - min(a.y1, b.y1)
            if (ovl_y > 1.0 and -2.5 <= gap_x <= 2.5) or (ovl_x > 1.0 and -2.5 <= gap_y <= 2.5):
                adj += 1
    return total > 0 and (adj / total) >= 0.025


def looks_like_diagram(page: "fitz.Page", page_rect: "fitz.Rect") -> tuple[bool, int, int]:
    """判定该页是否含「矢量框图」。返回 (是否, 方框数, 线段数)。"""

    boxes, segs = extract_boxes_and_segments(page, page_rect)
    if len(boxes) < MIN_BOXES or len(segs) < 1:
        return False, len(boxes), len(segs)
    if _looks_like_table_grid(boxes):
        return False, len(boxes), len(segs)
    return True, len(boxes), len(segs)


def find_diagram_pages(doc: "fitz.Document", *, max_pages: int | None = None) -> list[int]:
    """全量扫描，返回疑似框图的 **1-based 物理页码** 列表。"""

    out: list[int] = []
    limit = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    for pno in range(limit):
        page = doc[pno]
        ok, nb, ns = looks_like_diagram(page, page.rect)
        if ok:
            out.append(pno + 1)
    return out


def parse_diagram_page(
    page: "fitz.Page",
    corpus: str,
    page_no: int,
    *,
    prefer_connectors: bool = True,
) -> DiagramResult | None:
    """解析单页框图，返回 ``DiagramResult``；不是框图则返回 ``None``。"""

    res = DiagramResult(corpus=corpus, page=page_no)
    boxes, segs = extract_boxes_and_segments(page, page.rect)
    res.n_boxes = len(boxes)
    res.n_segments = len(segs)
    if len(boxes) < MIN_BOXES:
        return None

    # ---- ① 文字单元：span → 列聚类 → 落到框内 ----
    spans = extract_spans(page)
    units = _cluster_columns(spans)

    assigned: dict[int, list[TextUnit]] = {}
    for u in units:
        # 取「包含该单元 bbox（外扩 2pt）」的最小矩形框
        cand: list[tuple[float, int]] = []
        for i, b in enumerate(boxes):
            if (
                b.x0 - 2.0 <= u.bbox[0]
                and b.y0 - 2.0 <= u.bbox[1]
                and b.x1 + 2.0 >= u.bbox[2]
                and b.y1 + 2.0 >= u.bbox[3]
            ):
                cand.append((b.x1 - b.x0, i))
        if not cand:
            continue
        cand.sort()
        assigned.setdefault(cand[0][1], []).append(u)

    # ---- ② R7：同一框内多个列单元按 y 合并成一个节点名 ----
    for i, us in assigned.items():
        us_sorted = sorted(us, key=lambda u: (round(u.y0, 1), u.x))
        boxes[i].name = "".join(u.text for u in us_sorted).strip()
        boxes[i].unit_bbox = [
            min(u.bbox[0] for u in us_sorted),
            min(u.bbox[1] for u in us_sorted),
            max(u.bbox[2] for u in us_sorted),
            max(u.bbox[3] for u in us_sorted),
        ]

    named = [i for i, b in enumerate(boxes) if b.name]
    if len(named) < MIN_TEXT_UNITS:
        return None
    # 无名框（纯装饰）不参与层级
    keep_idx = sorted(named, key=lambda i: (boxes[i].y0, boxes[i].cx))
    boxes = [boxes[i] for i in keep_idx]

    # ---- ③ 父子关系：连线优先，几何兜底 ----
    graph, reps = build_point_graph(segs, boxes) if segs else ({}, [])
    edge_pairs: list[tuple[int, int]] = []
    if prefer_connectors and graph:
        for i, b in enumerate(boxes):
            att = attach_points(b, graph, reps)
            if not att:
                continue
            bottom = [p for p in att if abs(reps[p][1] - b.y1) <= TOL]
            if bottom:
                hits = halting_reach(bottom, graph, reps, boxes, i)
                for j, pt in hits.items():
                    if boxes[j].cy > b.cy + 1.0:
                        edge_pairs.append((i, j))
        if edge_pairs:
            res.evidence = "connector_lines"
            # 去重
            edge_pairs = sorted(set(edge_pairs))
    if not edge_pairs:
        res.evidence = "geometry"
        edge_pairs = [(p, c) for p, c in _geometric_edges(boxes)]

    # 一个子节点只留一个父：取「上方最近」的那个
    by_child: dict[int, list[int]] = {}
    for p, c in edge_pairs:
        by_child.setdefault(c, []).append(p)
    final_pairs: list[tuple[int, int]] = []
    for c, parents in by_child.items():
        parents = [p for p in parents if boxes[p].cy < boxes[c].cy]
        if not parents:
            continue
        parents.sort(key=lambda p: (abs(boxes[c].cy - boxes[p].cy), abs(boxes[c].cx - boxes[p].cx)))
        final_pairs.append((parents[0], c))
    if len(by_child) > len(final_pairs):
        res.warnings.append("有子节点未找到父节点（已跳过，见 by_child=%d）" % len(by_child))

    for p, c in final_pairs:
        boxes[p].children.append(boxes[c].name)
        boxes[c].parent = boxes[p].name
        res.edges.append([boxes[p].name, boxes[c].name])

    # ---- ④ 输出 ----
    res.boxes = boxes
    res.node_bbox = {b.name: [round(v, 2) for v in b.rect] for b in boxes if b.name}
    res.children_count = {b.name: len(b.children) for b in boxes if b.name and b.children}
    res.tree_text = structure_tree_text(res)
    return res


def unit_word_of(child_names: Sequence[str]) -> str:
    """由子节点名的**最长公共后缀**推断量词（如「部」→「部门」、「销售处」）。

    这是「语言学归一化」，不是答案常量：后缀完全由当页解析出的节点名算出。
    实测（物理页 39）：{电话及网络销售部, 渠道销售部, 大客户销售部, 国际贸易部} 的
    最长公共后缀是「部」→ 输出「部门」；6 个销售处的最长公共后缀是「销售处」→ 直接可用。

    子节点少于 3 个时**不做后缀推断**（两三个名字的公共后缀没有统计意义，
    实测会把 {监事会, 董事会} 推成「事会」这种读不通的量词）。
    """

    if not child_names:
        return "下级"
    if len(set(child_names)) < 3:
        return "下级单元"
    rev = [n[::-1] for n in child_names]
    suf = ""
    for chars in zip(*rev):
        if len(set(chars)) == 1:
            suf += chars[0]
        else:
            break
    suf = suf[::-1]
    if not suf:
        return "下级"
    if len(suf) > 6:
        suf = suf[-6:]
    return _UNIT_WORD_MAP.get(suf, suf)


def structure_tree_text(res: DiagramResult) -> str:
    """把层级树渲染成人类可读文本（供 chunk.text 直接使用）。"""

    name2box = {b.name: b for b in res.boxes if b.name}
    children = {b.name: list(b.children) for b in res.boxes if b.name}
    child_set = {c for cs in children.values() for c in cs}
    roots = [b.name for b in res.boxes if b.name and b.name not in child_set]
    roots.sort(key=lambda n: (name2box[n].y0, name2box[n].cx))

    lines: list[str] = []

    def walk(name: str, depth: int, seen: set[str]) -> None:
        if name in seen:
            lines.append("  " * depth + name + "（环路，已截断）")
            return
        seen = seen | {name}
        kids = children.get(name, [])
        suffix = ""
        if kids:
            suffix = "  ← %d 个下级" % len(kids)
        lines.append("  " * depth + name + suffix)
        for k in sorted(kids, key=lambda n: name2box[n].cx if n in name2box else 0.0):
            walk(k, depth + 1, seen)

    for r in roots:
        walk(r, 0, set())
    # 未被任何边连接、且不是根也没父的孤立节点
    covered = set()
    for b in res.boxes:
        if b.name:
            covered.add(b.name)
    if not lines:
        lines = [b.name for b in res.boxes if b.name]
    return "\n".join(lines)


def summarize_hierarchy(res: DiagramResult) -> list[str]:
    """为每个「有下级的节点」生成可检索归纳句（数字全部由结构树算出）。"""

    out: list[str] = []
    for b in sorted(res.boxes, key=lambda x: (x.y0, x.cx)):
        if not b.name or not b.children:
            continue
        kids = sorted(set(b.children))
        word = unit_word_of(kids)
        out.append("%s 由 %d 个%s构成：%s。" % (b.name, len(kids), word, "、".join(kids)))
    return out


# ---------------------------------------------------------------------------
# 交叉印证：在全文文本层里找与该框图一致的句子
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[。！？；])")


def collect_cross_references(
    doc: "fitz.Document",
    res: DiagramResult,
    *,
    max_hits: int = 6,
    exclude_page: int | None = None,
) -> list[dict[str, Any]]:
    """在**全文文本层**里找与框图结论一致的句子（供检索融合与交叉校验）。

    三种命中信号（通用，不硬编码句子）：
      * ``n_child``  ：句子含某个父节点的**多个子节点全名**（如页40「销售部下设渠道销售部、…」）；
      * ``n_prefix`` ：把子节点名去掉共同后缀得到"前缀"（如 珠海销售处→珠海），
                       句子含多个前缀（如页113「在北京、广州、成都、深圳、武汉、珠海各设有1家销售处」）；
                       —— 这一条是必需的：页113 的原文**不写全名**，只写城市名，
                       只比对全名会漏掉这条关键证据。
      * ``n_unit``   ：句子含子节点的量词（如「销售处」）。

    打分与排序：``score = 3*n_child + 2*n_prefix + 1*n_unit``，同分按页码升序。
    """

    parents = {b.name: b.children for b in res.boxes if b.name and b.children}
    if not parents:
        return []
    parent_names = set(parents)
    all_children: set[str] = set()
    unit_words: set[str] = set()
    prefixes: set[str] = set()
    for kids in parents.values():
        all_children.update(kids)
        w = unit_word_of(kids)
        if 2 <= len(w) <= 6:
            unit_words.add(w)
        for k in kids:
            if w and len(w) < len(k) and k.endswith(w):
                p = k[: -len(w)]
            else:
                p = k
            if len(p) >= 2:
                prefixes.add(p)

    hits: list[dict[str, Any]] = []
    for pno in range(doc.page_count):
        if exclude_page is not None and pno + 1 == exclude_page:
            continue
        try:
            txt = doc[pno].get_text()
        except Exception:
            continue
        for sent in _SENT_SPLIT.split(txt):
            s = re.sub(r"\s+", "", sent)
            if len(s) < 8 or len(s) > 140:
                continue
            n_child = sum(1 for c in all_children if c in s)
            n_parent = sum(1 for p in parent_names if p in s)
            n_unit = sum(1 for w in unit_words if w in s)
            n_prefix = sum(1 for p in prefixes if p in s)
            if ((n_child >= 2 and n_parent >= 1) or n_prefix >= 2
                    or (n_unit >= 1 and (n_child >= 1 or n_prefix >= 1)) or n_unit >= 2):
                hits.append({
                    "page": pno + 1, "quote": s, "n_child": n_child, "n_parent": n_parent,
                    "n_prefix": n_prefix, "n_unit": n_unit,
                    "score": 3 * n_child + 2 * n_prefix + n_unit,
                })
                if len(hits) >= max_hits * 8:
                    break
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for h in sorted(hits, key=lambda z: (-z["score"], z["page"])):
        if h["quote"] in seen:
            continue
        seen.add(h["quote"])
        uniq.append(h)
        if len(uniq) >= max_hits:
            break
    return uniq
