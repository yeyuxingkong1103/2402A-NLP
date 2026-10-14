# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
# 工单编号：人工智能NLP-RAG-Query理解优化
# 工单编号：人工智能NLP-RAG-混合检索任务
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""
模块：组织结构图矢量解析（服务 5 轮验收序列的 Q5）

招股2 第 38 页（0 基索引；PDF 阅读器第 39 页）的组织结构图是**矢量绘制**（不是位图），
因此可以零模型、确定性、毫秒级地还原成层级树。本机无任何视觉大模型（见设计文档
§2.1），CLIP/VLM 路线既不可行也违反「禁止下载」硬约束。

建树算法（依据本机实测的连线几何，见设计文档 §2.3）：

  1. 节点框 = get_drawings() 里除页眉横线外的 29 个矩形
     （页眉横线实测 rect=(88.4,55.3,507.0,56.0)：高 0.7pt、宽 418.6pt）
  2. 连线   = 36 条 'l' 线段；端点重合（含 T 型接头，端点落在另一段内部）的线段聚成一簇
  3. 文字   = get_text("words")，先按「图区」过滤掉页眉/页脚/页码（见 _chart_region），
     再逐字/逐行合并成标签，最后绑定到「文字中心所在的框」
  4. 每个节点框按「框的哪条边被线段端点碰到」归类：top / bottom / left / right
  5. **方向判定（核心规则）**：在一簇里，
       · 有且仅有 1 个框从 bottom 接入、其余从 top 接入 → bottom 那个是父（水平总线）
       · 有且仅有 1 个框从 left 接入、且 ≥2 个框从 right 接入 → left 那个是父（竖直总线）
       · 只剩两个框直连时按几何定向：bottom/top 者 → 上框为父；right/left 者 → 左框为父
     〔实测修正〕原稿把「单侧 left 接入者即父」当成竖直总线的通用规则，会在
       seg11(董事会右边缘 323.8 → 董事会秘书左边缘 377.8) 这条**水平直连**上把父子
       接反（实测产出 27 条边、2 个根）。故对二框直连单独按几何定向。
  6. 校验：29 节点 / 28 边 / 单根，不满足即抛错（宁可失败也不产出错误的树）

实测该规则产出：
  股东大会 → {董事会, 监事会}
  董事会   → {薪酬与考核委员会, 提名委员会, 战略委员会, 审计委员会, 总经理, 董事会秘书}
  总经理   → {财务部, 市场开发部, 信息系统部, 销售部, 行政人事部, 物流部, 上海分公司}
  销售部   → {电话及网络销售部, 渠道销售部, 大客户销售部, 国际贸易部}
  大客户销售部 → {珠海, 深圳, 北京, 武汉, 广州, 成都}销售处
与图上实际绘制完全一致（29 节点 / 28 边 / 单根股东大会）。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

import fitz

from src import config

# 本页图结构的实测事实（工单5 Q5 的答案就长在这张图上，错了就会静默答错）
ORGCHART_EXPECTED_NODES = 29
ORGCHART_EXPECTED_EDGES = 28

# 图区外扩量：用于剔除页眉/页脚/页码。实测页眉在 y≈44-53、页码在 y≈780，
# 而图区 y∈[75.2,644.6]；外扩 5pt 既容得下贴边文字，又碰不到页眉页脚。
_REGION_PAD_PT = 5.0


@dataclass
class OrgNode:
    name: str
    bbox: tuple[float, float, float, float]
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class OrgChart:
    source: str
    page_idx: int
    root: str
    nodes: dict[str, OrgNode]
    verify: dict = field(default_factory=dict)
    ocr_check: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. 文字合并（竖排 + 多行横排）
# ---------------------------------------------------------------------------
def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(min(a1 - a0, b1 - b0), 1e-6)


def _close(a, b, gap: float) -> bool:
    """两个字块是否应合并：一个方向重叠、另一方向间距小于 gap。"""
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    v_gap = max(by0 - ay1, ay0 - by1)
    h_gap = max(bx0 - ax1, ax0 - bx1)
    if _overlap_ratio(ax0, ax1, bx0, bx1) > 0.5 and v_gap <= gap:
        return True
    if _overlap_ratio(ay0, ay1, by0, by1) > 0.5 and h_gap <= gap:
        return True
    return False


def merge_words_to_labels(words, gap: float = None) -> list[tuple[str, tuple]]:
    """把 word 级文字块合并成「标签」。

    为什么需要：招股2 p38 上「财务部」「销售部」「大客户销售部」等是**竖排逐字**
    存放的（每个字一个 word），而「薪酬与考核委员会」是**两行横排**。二者都必须
    合回一个标签才能绑定到节点框。

    读取顺序：先按 y 重叠分组为「行」，行按 y 升序，行内按 x 升序——这样竖排
    （每字一行）与横排（一行多字）都能得到正确语序。输出再按 (y0,x0) 排序，
    使绑定顺序与文字在页面上的位置一致（确定性，与 word 输入顺序无关）。
    """
    gap = config.ORGCHART_MERGE_GAP_PT if gap is None else gap
    items = [tuple(w) for w in words]
    if not items:
        return []

    # 并查集
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _close(items[i], items[j], gap):
                union(i, j)

    groups: dict[int, list] = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(items[i])

    out: list[tuple[str, tuple]] = []
    for members in groups.values():
        # 分行：y 区间重叠 > 50% 视为同一行
        lines: list[list] = []
        for w in sorted(members, key=lambda x: (x[1], x[0])):
            placed = False
            for ln in lines:
                if _overlap_ratio(ln[0][1], ln[0][3], w[1], w[3]) > 0.5:
                    ln.append(w)
                    placed = True
                    break
            if not placed:
                lines.append([w])
        lines.sort(key=lambda ln: min(w[1] for w in ln))
        text = "".join(w[4] for ln in lines for w in sorted(ln, key=lambda x: x[0]))
        x0 = min(w[0] for w in members)
        y0 = min(w[1] for w in members)
        x1 = max(w[2] for w in members)
        y1 = max(w[3] for w in members)
        out.append((text.strip(), (x0, y0, x1, y1)))
    out.sort(key=lambda lab: (lab[1][1], lab[1][0]))
    return out


# ---------------------------------------------------------------------------
# 2. 取节点框与连线
# ---------------------------------------------------------------------------
def extract_boxes_and_lines(page) -> tuple[list[tuple], list[tuple]]:
    """返回 (boxes, segments)。boxes=[(rect, order)]；segments=[(p0, p1)]。

    排除页眉横线：它是一个扁到几乎没有高度的矩形（实测 y 跨度 0.7pt、宽 418.6pt）。
    """
    boxes: list[tuple] = []
    segments: list[tuple] = []
    for dr in page.get_drawings():
        r = dr["rect"]
        items = dr["items"]
        kinds = {it[0] for it in items}
        if kinds == {"re"}:
            w, h = r.x1 - r.x0, r.y1 - r.y0
            if h < 3 or w < 3:            # 页眉横线 / 退化的框
                continue
            boxes.append((r, len(boxes)))
        elif "l" in kinds:
            for it in items:
                if it[0] == "l":
                    a, b = it[1], it[2]
                    segments.append(((a.x, a.y), (b.x, b.y)))
    return boxes, segments


def _chart_region(boxes, segments) -> tuple[float, float, float, float]:
    """图区 = 全部节点框与线段的并集（外扩 _REGION_PAD_PT）。

    实测必要性：页眉「武汉力源信息技术股份有限公司 / 招股意向书」（y≈44-53）与
    页脚页码「38」（y≈780）在被「最近框」兜底绑定后会污染节点标签——
    实测把 box7 变成「武汉力源信息技术股份有限公司薪酬与考核委员会」、
    box3 变成「招股意向书监事会」、box24 变成「38深圳销售处」。
    设计文档 §7.1 第 4 条也要求「过滤页眉、页脚、页码、对角水印」。
    """
    xs0, ys0, xs1, ys1 = [], [], [], []
    for rect, _ in boxes:
        xs0.append(rect.x0)
        ys0.append(rect.y0)
        xs1.append(rect.x1)
        ys1.append(rect.y1)
    for a, b in segments:
        xs0.append(min(a[0], b[0]))
        ys0.append(min(a[1], b[1]))
        xs1.append(max(a[0], b[0]))
        ys1.append(max(a[1], b[1]))
    if not xs0:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs0) - _REGION_PAD_PT, min(ys0) - _REGION_PAD_PT,
            max(xs1) + _REGION_PAD_PT, max(ys1) + _REGION_PAD_PT)


def filter_words_in_region(words, region) -> list[tuple]:
    """只保留「中心落在图区内」的文字块（剔除页眉/页脚/页码）。"""
    x0, y0, x1, y1 = region
    out = []
    for w in words:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append(tuple(w))
    return out


# ---------------------------------------------------------------------------
# 3. 建树
# ---------------------------------------------------------------------------
def _on_segment(p, a, b, tol: float) -> bool:
    ax, ay = a
    bx, by = b
    px, py = p
    if abs(ax - bx) <= tol:                      # 竖线
        return abs(px - ax) <= tol and min(ay, by) - tol <= py <= max(ay, by) + tol
    if abs(ay - by) <= tol:                      # 横线
        return abs(py - ay) <= tol and min(ax, bx) - tol <= px <= max(ax, bx) + tol
    return False


def _cluster_segments(segments, tol_point: float) -> list[list[int]]:
    """端点重合或 T 型接头（端点落在另一段内部）的线段聚成一簇。"""
    n = len(segments)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            a, b = segments[i]
            c, d = segments[j]
            hit = False
            for p in (a, b):
                for q in (c, d):
                    if abs(p[0] - q[0]) <= tol_point and abs(p[1] - q[1]) <= tol_point:
                        hit = True
            if not hit:                          # T 型接头
                for p in (a, b):
                    if _on_segment(p, c, d, tol_point):
                        hit = True
                for q in (c, d):
                    if _on_segment(q, a, b, tol_point):
                        hit = True
            if hit:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _attach_edges(p, rect, tol: float) -> list[str]:
    """点 p 贴在 rect 的哪条边上。"""
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    edges = []
    if abs(p[1] - y0) <= tol and x0 - tol <= p[0] <= x1 + tol:
        edges.append("top")
    if abs(p[1] - y1) <= tol and x0 - tol <= p[0] <= x1 + tol:
        edges.append("bottom")
    if abs(p[0] - x0) <= tol and y0 - tol <= p[1] <= y1 + tol:
        edges.append("left")
    if abs(p[0] - x1) <= tol and y0 - tol <= p[1] <= y1 + tol:
        edges.append("right")
    return edges


def _orient_cluster(touched: dict[int, set[str]], names) -> list[tuple[str, str]]:
    """把一簇线段的「父 → 子」定向成边列表。返回 [(parent, child)]。

    规则见模块 docstring 第 5 条。`touched` 是 {框序号: 被贴到的边集合}。
    """
    bottom_only = [bi for bi, e in touched.items() if e == {"bottom"}]
    top_like = [bi for bi, e in touched.items() if e and "top" in e]
    left_only = [bi for bi, e in touched.items() if e == {"left"}]
    right_like = [bi for bi, e in touched.items() if e and "right" in e]

    def rest_of(parent_bi):
        return [bi for bi in touched if bi != parent_bi]

    # ① 水平总线：唯一从 bottom 接入者 = 父（实测 y=301.4 与 y=410.6、y=543.2 三条）
    if len(bottom_only) == 1 and len(top_like) >= 1:
        p = bottom_only[0]
        return [(names[p], names[bi]) for bi in rest_of(p)]

    # ② 竖直总线：唯一从 left 接入者 = 父；要求 ≥2 个从 right 接入者，
    #    以免把「两个框水平直连」也套上这条规则（那样父子会接反）
    if len(left_only) == 1 and len(right_like) >= 2:
        p = left_only[0]
        return [(names[p], names[bi]) for bi in rest_of(p)]

    # ③ 两框直连：按几何定向——bottom/top → 上者为父；right/left → 左者为父
    if len(touched) == 2:
        (bi_a, e_a), (bi_b, e_b) = sorted(touched.items())
        if e_a == {"bottom"} and e_b == {"top"}:
            return [(names[bi_a], names[bi_b])]
        if e_a == {"top"} and e_b == {"bottom"}:
            return [(names[bi_b], names[bi_a])]
        if e_a == {"right"} and e_b == {"left"}:
            return [(names[bi_a], names[bi_b])]     # 左框（从右边缘出线）是父
        if e_a == {"left"} and e_b == {"right"}:
            return [(names[bi_b], names[bi_a])]
    return []


def build_tree(boxes, segments, names: dict[int, str],
               tol_line: float = None, tol_point: float = None
               ) -> tuple[dict[str, str], dict]:
    """返回 (child->parent, verify)。见模块 docstring 的规则说明。"""
    tol_line = config.ORGCHART_LINE_TOL_PT if tol_line is None else tol_line
    tol_point = config.ORGCHART_POINT_TOL_PT if tol_point is None else tol_point

    parent_of: dict[str, str] = {}
    clusters = _cluster_segments(segments, tol_point)

    for members in clusters:
        pts = []
        for idx in members:
            pts.extend(segments[idx])

        # 每个框 → 它被贴到的边集合
        touched: dict[int, set[str]] = {}
        for bi, (rect, _) in enumerate(boxes):
            for p in pts:
                es = _attach_edges(p, rect, tol_line)
                if es:
                    touched.setdefault(bi, set()).update(es)
        if len(touched) < 2:
            continue

        for p_name, c_name in _orient_cluster(touched, names):
            if p_name == c_name or c_name in parent_of:
                continue
            parent_of[c_name] = p_name

    # --- BFS 从根定向剩余的无向边 -------------------------------------------
    root = _find_root(boxes, names, parent_of)
    undirected: dict[str, set[str]] = defaultdict(set)
    for c, p in parent_of.items():
        undirected[c].add(p)
        undirected[p].add(c)
    # 补上尚未定向、但同簇的二框直连
    for members in clusters:
        pts = []
        for idx in members:
            pts.extend(segments[idx])
        tset = []
        for bi, (rect, _) in enumerate(boxes):
            if any(_attach_edges(p, rect, tol_line) for p in pts):
                tset.append(names[bi])
        if len(tset) == 2 and tset[1] not in undirected[tset[0]]:
            undirected[tset[0]].add(tset[1])
            undirected[tset[1]].add(tset[0])

    seen = {root}
    q = deque([root])
    while q:
        cur = q.popleft()
        for nb in undirected[cur]:
            if nb in seen:
                continue
            seen.add(nb)
            parent_of.setdefault(nb, cur)
            q.append(nb)

    nameset = set(names.values())
    dropped = [c for c in parent_of if c not in nameset]
    for c in dropped:
        parent_of.pop(c, None)

    n_nodes = len(nameset)
    n_edges = len(parent_of)
    roots = nameset - set(parent_of)
    verify = {"n_nodes": n_nodes, "n_edges": n_edges,
              "roots": sorted(roots), "single_root": len(roots) == 1,
              "n_boxes_detected": len(boxes), "n_segments": len(segments),
              "n_clusters": len(clusters)}
    return parent_of, verify


def _find_root(boxes, names: dict[int, str], parent_of: dict[str, str]) -> str:
    """根 = 唯一没有父的节点；有多个时取 y 最小的（图的最上方）。"""
    all_names = set(names.values())
    roots = all_names - set(parent_of)
    if len(roots) == 1:
        return next(iter(roots))
    by_name = {names[i]: boxes[i][0] for i in range(len(boxes))}
    return min(sorted(roots), key=lambda n: by_name[n].y0)


def assert_expected(verify: dict) -> None:
    """树不合预期即抛错——宁可失败，也不产出会静默答错的树。"""
    bad = []
    if verify.get("n_nodes") != ORGCHART_EXPECTED_NODES:
        bad.append(f"节点 {verify.get('n_nodes')} != {ORGCHART_EXPECTED_NODES}")
    if verify.get("n_edges") != ORGCHART_EXPECTED_EDGES:
        bad.append(f"边 {verify.get('n_edges')} != {ORGCHART_EXPECTED_EDGES}")
    if not verify.get("single_root"):
        bad.append(f"根不唯一：{verify.get('roots')}")
    if bad:
        raise RuntimeError("组织结构图解析失败（不产出错误的树）：" + "；".join(bad)
                           + f"；verify={verify}")


# ---------------------------------------------------------------------------
# 4. 组装
# ---------------------------------------------------------------------------
def parse_orgchart(pdf_path: str, page_idx: int = None,
                   source: str = None) -> OrgChart:
    page_idx = config.ORGCHART_PAGE_IDX if page_idx is None else page_idx
    source = source or os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    page = doc[page_idx]

    boxes, segments = extract_boxes_and_lines(page)
    region = _chart_region(boxes, segments)
    words = filter_words_in_region(page.get_text("words"), region)
    doc.close()
    labels = merge_words_to_labels(words)

    # 文字 → 框：优先「文字中心落在框内」，退回「最近框」
    names: dict[int, str] = {}
    unbound: list[str] = []
    for text, (x0, y0, x1, y1) in labels:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best, best_d = None, 1e18
        for bi, (rect, _) in enumerate(boxes):
            if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
                best, best_d = bi, 0.0
                break
            d = (rect.x0 + rect.x1 - 2 * cx) ** 2 + (rect.y0 + rect.y1 - 2 * cy) ** 2
            if d < best_d:
                best, best_d = bi, d
        if best is None:
            unbound.append(text)
            continue
        # 同一框内的多段文字拼接（如两行）
        names[best] = (names[best] + text) if best in names else text

    parent_of, verify = build_tree(boxes, segments, names)
    root = _find_root(boxes, names, parent_of)
    verify["unbound_labels"] = unbound
    verify["page_idx"] = page_idx
    assert_expected(verify)

    nodes: dict[str, OrgNode] = {}
    for bi, (rect, _) in enumerate(boxes):
        nm = names.get(bi)
        if not nm:
            continue
        nodes[nm] = OrgNode(name=nm, bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            parent=parent_of.get(nm))
    for c, p in parent_of.items():
        if c in nodes and p in nodes:
            nodes[p].children.append(c)
    # 深度
    q = deque([(root, 0)])
    while q:
        nm, d = q.popleft()
        if nm not in nodes:
            continue
        nodes[nm].depth = d
        for c in nodes[nm].children:
            q.append((c, d + 1))

    return OrgChart(source=source, page_idx=page_idx, root=root,
                    nodes=nodes, verify=verify)


# ---------------------------------------------------------------------------
# 5. 聚合与渲染（Q5 的答案来源）
# ---------------------------------------------------------------------------
def descendants(chart: OrgChart, name: str, depth: int | None = None) -> list[str]:
    """返回 name 的下级（不含自身），depth 限制层数。Q5「哪个销售部的销售处最多」靠它。"""
    if name not in chart.nodes:
        return []
    out: list[str] = []
    q = deque([(name, 0)])
    while q:
        cur, d = q.popleft()
        if depth is not None and d >= depth:
            continue
        for c in chart.nodes[cur].children:
            out.append(c)
            q.append((c, d + 1))
    return out


def path_of(chart: OrgChart, name: str) -> str:
    """自根到 name 的路径，如 `销售部/大客户销售部`。"""
    parts, cur, guard = [], name, 0
    while cur and guard < 40:
        parts.append(cur)
        cur = chart.nodes[cur].parent if cur in chart.nodes else None
        guard += 1
    return "/".join(reversed(parts))


def render_text(chart: OrgChart) -> str:
    """整树可读展开（作为父块正文，命中任一节点即可还原全树）。

    每行是「缩进 + 节点名」，非根节点在行尾附 `← 路径：股东大会/…/本节点`：
    这样「路径」型问法（多跳）在父块正文里也能直接命中，不必依赖别处检索。
    """
    lines: list[str] = []

    def walk(name: str, indent: int):
        node = chart.nodes.get(name)
        if not node:
            return
        line = "  " * indent + name
        if node.parent:
            line += f"  ← 路径：{path_of(chart, name)}"
        lines.append(line)
        for c in node.children:
            walk(c, indent + 1)

    walk(chart.root, 0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5.1 结构化块（工单5 Task7：入库 + 检索侧注入）
# ---------------------------------------------------------------------------
def _aggregate_line(chart: OrgChart, name: str) -> str:
    """把「下一级有几个、都是谁」算好写进块文本，让检索/生成直接可用。

    【踩坑修正（工单6 id5）】旧实现按名字后缀把下级分成「销售处/销售部/其他」，
    并**丢弃"其他"组**，于是「销售部」的下级被报成"3个销售部"
    （电话及网络销售部、渠道销售部、大客户销售部），漏掉了同为直接下级的
    **国际贸易部** —— 工单6 的问题「销售部有几个部门构成」正确答案是 4，
    旧口径回答 3，属于统计口径直接答错。
    修正：先给出**全部直接下级的准确计数与枚举**，再按需补充"销售处"专项计数
    （服务「哪个销售部的销售处最多」这类问法），两者一致、可交叉验证。
    """
    subs = descendants(chart, name, depth=1)
    if not subs:
        return ""
    parts = [f"下级共{len(subs)}个：{'、'.join(subs)}"]
    # 只补充"销售处"专项计数：它服务「大客户销售部有几个销售处」这类问法。
    # 不再输出"其中销售部N个"——同一块里出现"4 个下级"与"3 个销售部"两个数字，
    # 会诱导模型答错（实测：旧口径 id5 答 3）。宁少勿混。
    grp = [s for s in subs if s.endswith("销售处")]
    if grp:
        parts.append(f"其中销售处{len(grp)}个：{'、'.join(grp)}")
    return "；".join(parts)


def build_chunks(chart: OrgChart) -> list[dict]:
    """组织结构图 → 结构化块（沿用工单3 的「子块检索 + 父块上下文」机制）。

    · 节点块：一个节点一条，含路径、下级、聚合结论；parent_id 指向整树块
    · 整树块：一条，正文是整树展开（命中任一节点即可还原全树）

    heading_path 用**列表**（与工单3 正文/表格块同口径）：vector_store 入库时按
    list() 取用、context 展示时按 " / ".join 拼接，写成字符串会被逐字拆开。
    节点路径另存为字符串字段 `org_path`（引用与 UI 展示用）。
    """
    tree_id = f"{chart.source}#p{chart.page_idx}#orgchart"
    tree_text = render_text(chart)
    top_heading = ["发行人基本情况", "发行人内部组织结构图"]

    chunks: list[dict] = [{
        "chunk_id": tree_id,
        "parent_id": tree_id,
        "block_type": "orgchart",
        "source": chart.source,
        "page_idx": chart.page_idx,
        "heading_path": list(top_heading),
        "org_node": "",
        "org_path": "",
        "org_tree_id": tree_id,
        "image_path": f"data/tables/orgchart_p{chart.page_idx}.png",
        "text": f"表：武汉力源信息技术股份有限公司组织结构图（第{chart.page_idx + 1}页）\n{tree_text}",
        "parent_text": tree_text,
    }]

    for name, node in chart.nodes.items():
        path = path_of(chart, name)
        agg = _aggregate_line(chart, name)
        direct = "、".join(node.children) if node.children else "无下级"
        text = (f"组织：武汉力源信息技术股份有限公司 | 节点：{name} | 路径：{path} | "
                f"下级：{direct} | 页码：{chart.page_idx + 1}")
        if agg:
            text += f" | 统计：{agg}"
        chunks.append({
            "chunk_id": f"{tree_id}#{name}",
            "parent_id": tree_id,
            "block_type": "orgchart",
            "source": chart.source,
            "page_idx": chart.page_idx,
            "heading_path": [*top_heading, path],
            "org_node": name,
            "org_path": path,
            "org_tree_id": tree_id,
            "image_path": f"data/tables/orgchart_p{chart.page_idx}.png",
            "text": text,
            "parent_text": tree_text,
        })
    return chunks


# ---------------------------------------------------------------------------
# 6. PaddleOCR-VL 交叉验证（仅构建期跑一次）
# ---------------------------------------------------------------------------
def verify_with_ocr(chart: OrgChart, pdf_path: str) -> dict:
    """用本机唯一的多模态资产 PaddleOCR-VL 独立复核图中的文字。

    层级关系以矢量连线为准（OCR 给不出可靠的树结构），OCR 承担的是
    **文字集合的独立复核**——两种方法抽出的文字一致，才认为矢量解析没读错字。
    """
    expected = set(chart.nodes.keys())
    try:
        from src import ocr_fallback
        doc = fitz.open(pdf_path)
        pix = doc[chart.page_idx].get_pixmap(dpi=config.ORGCHART_RENDER_DPI)
        doc.close()
        png = os.path.join(config.TABLE_DIR, f"orgchart_p{chart.page_idx}.png")
        pix.save(png)
        got_text = ocr_fallback.ocr_image_text(png)        # 见 Step 4
        got = {t for t in got_text if t}
    except Exception as exc:                                # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                "expected": sorted(expected)}

    matched = {e for e in expected if any(e in g or g in e for g in got)}
    return {"ok": True,
            "n_expected": len(expected),
            "n_ocr_labels": len(got),
            "matched": sorted(matched),
            "missing": sorted(expected - matched),
            "coverage": round(len(matched) / max(len(expected), 1), 4)}
