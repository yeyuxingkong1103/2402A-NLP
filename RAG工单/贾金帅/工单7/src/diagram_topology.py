"""
方框-连接线图的几何拓扑抽取（工单4）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

为什么需要它 —— 多模态模型在「层级归属」上**不稳定**
----------------------------------------------------
id=5 问的是「销售部有几个部门构成，其中大客户销售部有几个销售处构成？」
力源 p39 组织图实测：整张图底部有一条横向母线，
**从「大客户销售部」正下方引出的竖线**接上这条母线，母线上再垂下 6 条竖线连到 6 个销售处。
正确答案：销售部 4 个子部门、大客户销售部 6 个销售处。

同一个提示词、temperature=0，qwen-vl-plus 三次跑出三种结果：
    #1  电话及网络销售部→珠海、深圳；大客户销售部→北京、武汉、广州、成都   （拆成 2+4，错）
    #2  大客户销售部→珠海、深圳、北京、武汉、广州、成都                    （6 个，对）
    #3  电话及网络销售部→珠海、深圳；渠道销售部→北京；大客户销售部→武汉；
        国际贸易部→广州、成都                                              （拆成 2+1+1+2，错）
**靠加采样投票兜不住**（两次错一次对，3 次采样有 1/4 概率凑不出多数）。

所以这里做**几何校验**：这类图的拓扑不是"看"出来的，而是**算**出来的 ——
节点是位图矩形、连接线是细长矢量矩形，父子关系由它们的相接关系唯一确定。
多模态模型继续负责它擅长的部分（图类型、图题、图内文字、图表数值），
**层级归属改由几何裁决**。

判据（已用 p39 逐条核对过）
--------------------------
* 母线：宽高比很大的细长矩形（``252,543,468,543``）。
* 上引竖线：``y1 ≈ bus.y`` 的竖线 → 它上端连着父框的底边。
* 下引竖线：``y0 ≈ bus.y`` 的竖线 → 它下端连着子框的顶边。
* 也有不走母线的**直连**：父框底边 → 一根竖线 → 子框顶边（p39 的「市场开发部 → 研发中心」）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import WORK_ORDER_NOS  # noqa: F401  (工单编号，模块标识)

# 节点框的最小尺寸（pt）：小于这个的是装饰块/图例小块，不是节点
NODE_MIN_W = 18.0
NODE_MIN_H = 12.0
# 线状图元的判定：最短边 <= 该值
LINE_MAX_THICK = 2.5
# 线端与框边的贴合容差（pt）
TOUCH_TOL = 2.0
# 竖线与框中心的横向归属容差：允许竖线略偏离框中心
CENTER_TOL = 3.0


@dataclass
class TopoNode:
    label: str
    bbox: tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0


@dataclass
class Topology:
    nodes: list[TopoNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)   # (父下标, 子下标)

    def children_of(self, i: int) -> list[int]:
        return [c for p, c in self.edges if p == i]

    def as_pairs(self) -> list[tuple[str, list[str]]]:
        """按父节点聚合，父按 (y, x) 排序 —— 和看图读的顺序一致。"""
        out = []
        order = sorted(range(len(self.nodes)), key=lambda i: (self.nodes[i].bbox[1], self.nodes[i].bbox[0]))
        for i in order:
            kids = self.children_of(i)
            if not kids:
                continue
            kids_sorted = sorted(kids, key=lambda k: (self.nodes[k].bbox[1], self.nodes[k].bbox[0]))
            out.append((self.nodes[i].label, [self.nodes[k].label for k in kids_sorted]))
        return out

    def render(self) -> str:
        """渲染成可入索引的层级文本。

        特意写成「父 → 子1、子2、…」整句形式，而不是分多行：
        BM25 只有在「销售部」和「大客户销售部」出现在**同一行**里时，
        才能把 id=5 那种一句话问两层的问法命中。
        """
        pairs = self.as_pairs()
        if not pairs:
            return ""
        lines = []
        for parent, kids in pairs:
            lines.append(f"{parent} → " + "、".join(kids))
        counts = "；".join(f"{p} {len(k)} 个" for p, k in pairs)
        lines.append(f"各层级子节点数量：{counts}")
        return "\n".join(lines)


def _is_line(r) -> bool:
    return min(r[2] - r[0], r[3] - r[1]) <= LINE_MAX_THICK


def _contains_x(r, x: float, tol: float = 0.0) -> bool:
    return r[0] - tol <= x <= r[2] + tol


def _box_text(page, rect, max_chars: int = 24) -> str:
    """取节点框内的文字。

    ⚠️ 竖排节点名在 PDF 文字层里是**逐字**存放的（「财」「务」「部」），
    所以不能直接用 words 的结果，要把框内的单字按阅读顺序拼起来。
    中文竖排是**自上而下、列自右向左**，因此排序键用 (-col, row)：
      * col 由 x 量化得到（同列的字 x 相近）；
      * 同一列内按 y 从小到大。
    实测这样能把 p39 的「电/话/及/网/络/销/售/部」正确还原成「电话及网络销售部」。
    """
    words = []
    for w in page.get_text("words"):
        wr = (w[0], w[1], w[2], w[3])
        cx, cy = (wr[0] + wr[2]) / 2.0, (wr[1] + wr[3]) / 2.0
        if rect[0] - 1 <= cx <= rect[2] + 1 and rect[1] - 1 <= cy <= rect[3] + 1:
            words.append((cx, cy, w[4]))
    if not words:
        return ""

    # 行高约 9~11pt，用 6pt 量化列
    col_q = 6.0
    words.sort(key=lambda t: (-round(t[0] / col_q), round(t[1] / col_q)))
    text = "".join(t[2] for t in words)
    return re.sub(r"\s+", "", text)[:max_chars]


def extract_topology(page, bbox, exclude_boxes=None) -> Topology:
    """从页面矢量图元里抽出「方框-连接线」的层级拓扑。

    bbox：图区域（来自 image_detector）。
    exclude_boxes：表格区域（表格线框不能当节点/连接线）。
    """
    from .image_detector import _inside_any, _overlap_area

    topo = Topology()
    x0, y0, x1, y1 = bbox
    excl = list(exclude_boxes or [])

    boxes: list[tuple] = []
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            rr = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
            if rr[2] - rr[0] < NODE_MIN_W or rr[3] - rr[1] < NODE_MIN_H:
                continue
            # 必须**大部分落在图区域内**：图外的页眉 logo 之类不算节点
            a = (rr[2] - rr[0]) * (rr[3] - rr[1])
            if _overlap_area(rr, bbox) / a < 0.9:
                continue
            if _inside_any(rr, excl):
                continue
            boxes.append(rr)

    verts: list[tuple] = []
    horgs: list[tuple] = []
    for d in page.get_drawings():
        r = d["rect"]
        rr = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
        if _overlap_area(rr, bbox) <= 0 and not _inside_x_y(rr, bbox):
            continue
        if _inside_any(rr, excl):
            continue
        if not _is_line(rr):
            continue
        w, h = rr[2] - rr[0], rr[3] - rr[1]
        if h > w and h >= 6:          # 竖线
            verts.append(rr)
        elif w > h and w >= 6:        # 横线（母线）
            horgs.append(rr)

    # 去重（同一个矩形会被画多次）
    boxes = _dedup(boxes)
    verts = _dedup(verts)
    horgs = _dedup(horgs)

    # 节点：归类成 label
    for b in boxes:
        label = _box_text(page, b)
        topo.nodes.append(TopoNode(label=label or f"未命名{b[0]:.0f},{b[1]:.0f}", bbox=b))

    if not topo.nodes:
        return topo

    def find_parent_of_stem(v):
        """竖线 v 的**上端**连到哪个框（框底边 ≈ v 顶端）。"""
        for i, n in enumerate(topo.nodes):
            if not _contains_x(n.bbox, v[0], CENTER_TOL):
                continue
            if abs(n.bbox[3] - v[1]) <= TOUCH_TOL:
                return i
        return None

    def find_child_of_stem(v):
        """竖线 v 的**下端**连到哪个框（框顶边 ≈ v 底端）。"""
        for i, n in enumerate(topo.nodes):
            if not _contains_x(n.bbox, v[0], CENTER_TOL):
                continue
            if abs(n.bbox[1] - v[3]) <= TOUCH_TOL:
                return i
        return None

    edges: set[tuple[int, int]] = set()

    # ① 母线型：父 → 上引竖线 → 母线 → 下引竖线 → 子
    for bus in horgs:
        by = bus[1]
        stems = [v for v in verts if bus[0] - TOUCH_TOL <= v[0] <= bus[2] + TOUCH_TOL
                 and abs(v[3] - by) <= TOUCH_TOL and v[1] < by]
        drops = [v for v in verts if bus[0] - TOUCH_TOL <= v[0] <= bus[2] + TOUCH_TOL
                 and abs(v[1] - by) <= TOUCH_TOL and v[3] > by]
        for st in stems:
            p = find_parent_of_stem(st)
            if p is None:
                continue
            for dp in drops:
                c = find_child_of_stem(dp)
                if c is not None and c != p:
                    edges.add((p, c))

    # ② 直连型：父框底边 → 一根竖线 → 子框顶边
    for v in verts:
        p = find_parent_of_stem(v)
        c = find_child_of_stem(v)
        if p is not None and c is not None and p != c:
            edges.add((p, c))

    topo.edges = sorted(edges)
    return topo


def _inside_x_y(r, bbox) -> bool:
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    return bbox[0] - 2 <= cx <= bbox[2] + 2 and bbox[1] - 2 <= cy <= bbox[3] + 2


def _dedup(rects: list[tuple], tol: float = 0.6) -> list[tuple]:
    out: list[tuple] = []
    for r in rects:
        if any(all(abs(r[k] - o[k]) <= tol for k in range(4)) for o in out):
            continue
        out.append(r)
    return out


# 判定"这张图是不是方框-连接线结构"，以及拓扑是否可用
MIN_NODES_FOR_TOPO = 5
MIN_EDGES_FOR_TOPO = 3


def topology_usable(topo: Topology) -> bool:
    """拓扑够不够格作为权威结果。

    门槛定得保守：节点太少（<5）或边太少（<3）说明这根本不是层级图，
    或者图元检测没抽全 —— 这时候宁可回到多模态模型的说法，不要用残缺拓扑去覆盖它。
    """
    return len(topo.nodes) >= MIN_NODES_FOR_TOPO and len(topo.edges) >= MIN_EDGES_FOR_TOPO
