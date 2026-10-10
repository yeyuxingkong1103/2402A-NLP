# -*- coding: utf-8 -*-
"""图形语义解析：把 PDF 中的矢量「组织结构图」还原为可检索的结构化文本。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

本模块沿用 04 工单《图像内容解析及检索优化》的「连接线图还原」能力，为多轮
问答补齐**图形知识**（如「武汉力源信息技术股份有限公司组织结构图」）：

招股说明书中的组织结构图是**矢量绘图**（大量矩形节点框 + 线段连线）。图内文字
虽在 PDF 文本层可抽取，但阅读顺序被打散（常为竖排单字，如「销/售/部」），
必须按「节点框 + 连线」还原层级，才能得到

    销售部由4个部门构成：渠道销售部、电话及网络销售部、大客户销售部、国际贸易部。
    大客户销售部由6个销售处构成：北京销售处、广州销售处、成都销售处、深圳销售处、武汉销售处、珠海销售处。

这类可被 RAG 检索与抽取的语义文本。

关键难点：组织结构图用**共享母线**连接同一父节点下的多个子节点。若做无向连通
遍历，兄弟节点会经由母线互相连通，产生「A 是 B 的子节点、B 也是 A 的子节点」的
错误。本实现用**母线归属法**消除该问题（见 `_build_tree`）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pymupdf
except ImportError:      # 兼容旧版本包名
    import fitz as pymupdf

BBox = Tuple[float, float, float, float]
_PT = Tuple[float, float]

_CJK_RE = re.compile(r"[\u4e00-\u9fa5]")

# 子节点名称的公共后缀 → 计量单位（生成「销售部由4个部门构成」这类可检索表述）
_CHILD_UNITS = (("销售处", "销售处"), ("委员会", "委员会"), ("分公司", "分公司"),
                ("中心", "中心"), ("部门", "部门"), ("部", "部门"))


@dataclass
class FigureSemantics:
    """一个图形区域的语义解析结果。"""
    source: str
    page: int
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    caption: str = ""
    semantic_text: str = ""
    relations: List[Tuple[str, str]] = field(default_factory=list)


# ---------------- 一、基础几何工具 ---------------------------------------------
def _pt_in_box(p: _PT, box: BBox, tol: float = 4.0) -> bool:
    return box[0] - tol <= p[0] <= box[2] + tol and box[1] - tol <= p[1] <= box[3] + tol


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _seg_touch(a: Tuple[_PT, _PT], b: Tuple[_PT, _PT], tol: float = 3.0) -> bool:
    """两条轴对齐线段是否相接（端点接触或交叉）。"""
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    a_h = abs(ay1 - ay0) <= tol
    b_h = abs(by1 - by0) <= tol
    axlo, axhi = min(ax0, ax1), max(ax0, ax1)
    aylo, ayhi = min(ay0, ay1), max(ay0, ay1)
    bxlo, bxhi = min(bx0, bx1), max(bx0, bx1)
    bylo, byhi = min(by0, by1), max(by0, by1)
    if a_h and b_h:
        return abs(aylo - bylo) <= tol and not (axhi < bxlo - tol or bxhi < axlo - tol)
    if not a_h and not b_h:
        return abs(axlo - bxlo) <= tol and not (ayhi < bylo - tol or byhi < aylo - tol)
    if a_h:
        return axlo - tol <= bxlo <= axhi + tol and bylo - tol <= aylo <= byhi + tol
    return bxlo - tol <= axlo <= bxhi + tol and aylo - tol <= bylo <= ayhi + tol


# ---------------- 二、节点框与连线抽取 -----------------------------------------
def _node_label(page, box: BBox) -> str:
    """节点框内的文字（竖排单字按 y 排序拼接，横排按阅读顺序）。"""
    try:
        spans = []
        for blk in page.get_text("dict").get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    t = (sp.get("text") or "").strip()
                    if not t:
                        continue
                    sx0, sy0, sx1, sy1 = sp["bbox"]
                    cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
                    if box[0] - 2 <= cx <= box[2] + 2 and box[1] - 2 <= cy <= box[3] + 2:
                        spans.append((round(sy0, 1), round(sx0, 1), t))
    except Exception:
        return ""
    if not spans:
        return ""
    spans.sort(key=lambda s: (s[0], s[1]))
    return "".join(s[2] for s in spans).strip()


def _node_boxes(page, region_box: BBox) -> List[Tuple[BBox, str]]:
    """区域内的节点框（纯矩形 drawing），并绑定框内文字作为节点名。"""
    out: List[Tuple[BBox, str]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return out
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        kinds = {it[0] for it in d.get("items", [])}
        if kinds != {"re"}:
            continue
        box = (r.x0, r.y0, r.x1, r.y1)
        if box[0] < region_box[0] - 6 or box[2] > region_box[2] + 6:
            continue
        if box[1] < region_box[1] - 6 or box[3] > region_box[3] + 6:
            continue
        if (box[2] - box[0]) < 8 or (box[3] - box[1]) < 8:
            continue
        label = _node_label(page, box)
        if label:
            out.append((box, label))
    return out


def _region_segments(page, region_box: BBox) -> List[Tuple[_PT, _PT]]:
    """区域内所有线段（横线/竖线），作为连接线图的边。"""
    segs: List[Tuple[_PT, _PT]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return segs
    for d in drawings:
        for it in d.get("items", []):
            if it[0] != "l":
                continue
            p1, p2 = it[1], it[2]
            a, b = (p1.x, p1.y), (p2.x, p2.y)
            if region_box[0] - 8 <= a[0] <= region_box[2] + 8 and \
               region_box[1] - 8 <= a[1] <= region_box[3] + 8:
                segs.append((a, b))
    return segs


# ---------------- 三、连接线图 → 有向层级树 -------------------------------------
def _child_unit(kids: List[str]) -> str:
    """按子节点名称的公共后缀推断计量单位；后缀不统一时用「机构」。"""
    if not kids:
        return "机构"
    for suf, unit in _CHILD_UNITS:
        if all(k.endswith(suf) for k in kids):
            return unit
    return "机构"


def _build_tree(boxes: List[Tuple[BBox, str]],
                segs: List[Tuple[_PT, _PT]], tol: float = 4.0
                ) -> Tuple[str, Dict[str, List[str]]]:
    """自上而下遍历连接线图，还原「父 → 子」层级。

    关键难点：组织结构图用**共享母线**连接同一父节点下的多个子节点。若做无向连通
    遍历，兄弟节点会经由母线互相连通，产生「A 是 B 的子节点、B 也是 A 的子节点」的
    错误。本实现用**母线归属法**消除该问题：

    1. 只把「不锚定在任何节点框上」的线段（即母线）标记为**已被占用**；
    2. 从父节点出发，沿其引线 → 母线 → 各子节点引线推进；
    3. 子节点再展开时，父节点引线与母线均已占用，无法回溯，兄弟亦不可达。
    """
    if not boxes:
        return "", {}

    def box_at(pt: _PT) -> Optional[str]:
        for b in boxes:
            if _pt_in_box(pt, b[0], tol):
                return b[1]
        return None

    def connectors(name: str, owned: set) -> List[int]:
        pbox = next(b[0] for b in boxes if b[1] == name)
        out = []
        for i, s in enumerate(segs):
            if i in owned:
                continue
            if _pt_in_box(s[0], pbox, tol) != _pt_in_box(s[1], pbox, tol):
                out.append(i)
        return out

    has_conn = [b for b in boxes if connectors(b[1], set())]
    if not has_conn:
        return "", {}
    has_conn.sort(key=lambda b: (b[0][1], b[0][0]))
    root = has_conn[0][1]

    owned: set = set()
    visited: set = set()
    children: Dict[str, List[str]] = {}
    stack = [root]
    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)

        local: set = set()
        queue: List[int] = []
        for i in connectors(name, owned):
            local.add(i)
            queue.append(i)

        kids: List[str] = []
        while queue:
            i = queue.pop()
            s = segs[i]
            for pt in (s[0], s[1]):
                c = box_at(pt)
                if c and c != name and c not in kids:
                    kids.append(c)
            for k, t in enumerate(segs):
                if k in local or k in owned:
                    continue
                if _seg_touch(s, t):
                    local.add(k)
                    queue.append(k)

        for k in local:
            if not any(_pt_in_box(segs[k][0], b[0], tol) or _pt_in_box(segs[k][1], b[0], tol)
                       for b in boxes):
                owned.add(k)
            else:
                for b in boxes:
                    if b[1] in visited and (
                        _pt_in_box(segs[k][0], b[0], tol) or _pt_in_box(segs[k][1], b[0], tol)):
                        owned.add(k)
                        break

        if kids:
            children[name] = kids
            stack.extend(kids)

    return root, children


def parse_org_chart(page, region_box: BBox,
                    caption: str = "") -> Tuple[str, List[Tuple[str, str]]]:
    """还原矢量组织结构图：返回 (语义文本, 父子关系列表)。"""
    # 超长标签（>20 字）是文本层把多行说明误聚合到同一框，非节点名，需剔除
    boxes = [b for b in _node_boxes(page, region_box)
             if _has_cjk(b[1]) and len(b[1]) <= 20]
    if len(boxes) < 3:
        return "", []
    segs = _region_segments(page, region_box)
    root, children_map = _build_tree(boxes, segs)
    if not children_map:
        return "", []

    relations: List[Tuple[str, str]] = []
    for parent, kids in children_map.items():
        for k in kids:
            relations.append((parent, k))

    lines: List[str] = []
    order: List[str] = []
    seen = set()
    frontier = [root] if root else []
    while frontier:
        nxt = []
        for n in frontier:
            if n in seen:
                continue
            seen.add(n)
            order.append(n)
            nxt.extend(children_map.get(n, []))
        frontier = nxt
    for n in order:
        kids = children_map.get(n, [])
        if kids:
            lines.append(f"{n}由{len(kids)}个{_child_unit(kids)}构成：{'、'.join(kids)}。")
    covered = {n for rel in relations for n in rel}
    orphans = [b[1] for b in boxes if b[1] not in covered and b[1] != root]
    if orphans:
        lines.append("其他部门/机构：" + "、".join(orphans) + "。")
    if not lines:
        return "", relations

    src_name = Path(page.parent.name).name
    text = (f"【组织结构图】{caption or '组织结构图'}（《{src_name}》第{page.number + 1}页）\n"
            + "\n".join(lines))
    return text, relations


# ---------------- 四、区域检测（矢量矩形簇聚类） --------------------------------
def _cluster(rects: List, gap: float = 40.0) -> List[Tuple[BBox, List[int]]]:
    """并查集聚类：把间隙小于 gap 的矩形合并为同一图形区域。"""
    n = len(rects)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    boxes = [(r.x0, r.y0, r.x1, r.y1) for r in rects]
    for i, b in enumerate(boxes):
        ex = (b[0] - gap, b[1] - gap, b[2] + gap, b[3] + gap)
        for j in range(i + 1, n):
            c = boxes[j]
            if not (ex[2] < c[0] or ex[0] > c[2] or ex[3] < c[1] or ex[1] > c[3]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: List[Tuple[BBox, List[int]]] = []
    for idxs in groups.values():
        xs0 = min(boxes[i][0] for i in idxs)
        ys0 = min(boxes[i][1] for i in idxs)
        xs1 = max(boxes[i][2] for i in idxs)
        ys1 = max(boxes[i][3] for i in idxs)
        out.append(((xs0, ys0, xs1, ys1), idxs))
    return out


def _is_rule(rect, page_rect) -> bool:
    """版式发丝线（贯穿全宽的页眉/页脚横线）。"""
    h, w = rect.y1 - rect.y0, rect.x1 - rect.x0
    if h <= 3 and w >= 0.7 * (page_rect.x1 - page_rect.x0):
        return True
    if w <= 3 and h >= 0.7 * (page_rect.y1 - page_rect.y0):
        return True
    return False


def _page_caption(page) -> str:
    """在页面文本中寻找「组织结构图」类图题。"""
    try:
        text = page.get_text() or ""
    except Exception:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if re.search(r"(组织结构图|组织架构图|结构图)", line) and len(line) <= 40:
            return line
    return ""


_ORG_SUFFIX = ("部", "处", "中心", "委员会", "分公司", "公司", "董事会",
              "监事会", "股东大会", "大会")
_ORG_CAP_RE = re.compile(r"(组织结构图|组织架构图|结构图)")


def _is_org_like(rels: List[Tuple[str, str]], caption: str) -> bool:
    """判定图形是否为「组织结构」类，排除被误检成图形的表格 / 流程图。

    组织结构图的两个特征：图题含「结构图」；或子节点多为部门/销售处/委员会等
    组织单元（以「部/处/中心/委员会/公司」等后缀结尾）。
    """
    if _ORG_CAP_RE.search(caption or ""):
        return True
    kids = [k for _p, k in rels]
    if not kids:
        return False
    hit = sum(1 for k in kids if k.endswith(_ORG_SUFFIX))
    return hit >= max(2, len(kids) // 2)


def extract_figure_semantics(pdf_path: Path | str,
                             min_boxes: int = 4, max_boxes: int = 60
                             ) -> List[FigureSemantics]:
    """抽取单个 PDF 中的矢量组织结构图并还原语义文本。"""
    path = Path(pdf_path)
    out: List[FigureSemantics] = []
    with pymupdf.open(str(path)) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            page_rect = page.rect
            try:
                drawings = page.get_drawings()
            except Exception:
                continue
            rect_items = []
            for d in drawings:
                r = d.get("rect")
                if r is None or _is_rule(r, page_rect):
                    continue
                if {it[0] for it in d.get("items", [])} == {"re"}:
                    rect_items.append(r)
            if len(rect_items) < min_boxes:
                continue
            for box, idxs in _cluster(rect_items, gap=40.0):
                if not (min_boxes <= len(idxs) <= max_boxes):
                    continue
                sem_text, rels = parse_org_chart(page, box, caption=_page_caption(page))
                if sem_text and len(rels) >= 2 and _is_org_like(rels, _page_caption(page)):
                    out.append(FigureSemantics(
                        source=path.name, page=i + 1, bbox=box,
                        caption=_page_caption(page),
                        semantic_text=sem_text, relations=rels))
    return out


def extract_all_figure_semantics(pdf_paths) -> List[FigureSemantics]:
    out: List[FigureSemantics] = []
    for p in pdf_paths:
        try:
            out.extend(extract_figure_semantics(p))
        except Exception:
            continue
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config

    sems = extract_all_figure_semantics(config.PDF_PATHS)
    print(f"共解析图形 {len(sems)} 个")
    for s in sems:
        print("-" * 70)
        print(f"{s.source} p.{s.page} 关系 {len(s.relations)} 条")
        print(s.semantic_text[:800])