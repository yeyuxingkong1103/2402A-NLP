# -*- coding: utf-8 -*-
"""图像语义解析模块（PDF 图像内容解析第二步）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

本模块把 `figure_extractor` 抽出的**图形区域**解析为**可检索的结构化语义文本**，
是本工单「图像内容解析」的核心：只有把图形变成语义，图形才能进入 RAG 检索链路。

招股说明书中的图形分两类，语义还原方式完全不同：

1. **矢量组织结构图 / 流程图**（如「公司组织结构图」）
   - 图内文字在 PDF 文本层可抽取，但阅读顺序被打散（常为竖排单字，如
     「销/售/部」），必须按「节点框 + 连线」还原**层级结构**；
   - 本模块用**连接线图（connector graph）**做有向遍历：以父节点边界上的线段为
     起点，沿「竖直下引线 → 横向母线 → 竖直下引线」推进，遇到节点框即停止，
     从而还原「父 → 子」关系，得到 `销售部 → 大客户销售部 → 珠海/深圳/…销售处`
     这样的可检索结构（对应工单 id 5 类问题）。

2. **位图统计图**（如「2008 年中国 IC 市场应用结构与增长图」）
   - 图内数值（坐标轴 / 数据标签）不可被文本层抽取，必须做**图像文字识别**；
   - 本模块用 OCR（rapidocr，若不可用则优雅降级）识别图内文字，再按「同排配对」
     把行业名与数值绑定，并派生**极值结论**（增长率最快 / 负增长），
     对应工单 id 6 类问题。

输出 `FigureSemantics`：含 `semantic_text`（进入文本检索库）、`relations`（层级关系）
与 `key_values`（字段—数值），供 `image_index`（CLIP 跨模态索引）与检索/答案合成复用。

多模态模型使用：图形类型判定由 Chinese-CLIP 零样本分类完成（见 image_encoder）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src import config
from src.figure_extractor import BBox, FigureRegion

logger = logging.getLogger(__name__)

try:
    import pymupdf
except ImportError:      # 兼容旧版本包名
    import fitz as pymupdf

_PT = Tuple[float, float]

# 竖排文字拼接：同一节点框内的单字按 y 排序后拼接
_OCR_NUM_RE = re.compile(r"[-+]?\d[\d,，.]*\s*[%％]?")
_PCT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*[%％]")
_CAT_CHAR_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z]")
_NUM_ONLY_RE = re.compile(r"^[\d\s%％\-~～.．]+$")


def _is_category(name: str) -> bool:
    """系列名是否为「类别名」（而非 OCR 误配出的纯数值 / 百分比 / 区间刻度）。

    招股说明书的「获利率」等图把区间刻度（11%-15%、7%-10%）也印在图内，OCR 会把它们
    当成系列名与百分比配对，派生出「增长率最快的是7%-10%」这类无意义结论，进而污染
    其他图形题（如组织结构图）的答案。故要求系列名含中英文字符且非纯数值。
    """
    if not name or not _CAT_CHAR_RE.search(name):
        return False
    return not _NUM_ONLY_RE.match(name.strip())


@dataclass
class FigureSemantics:
    """一个图形区域的语义解析结果。"""

    source: str
    page: int
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    kind: str = "vector"                      # raster / vector
    image_path: str = ""
    caption: str = ""
    figure_type: str = ""                     # 组织结构图 / 柱状图 / 饼图 …
    type_score: float = 0.0
    semantic_text: str = ""                   # 进入文本检索库的语义文本
    relations: List[Tuple[str, str]] = field(default_factory=list)   # (父节点, 子节点)
    key_values: List[Tuple[str, str]] = field(default_factory=list)  # (字段/系列, 取值)

    @property
    def key(self) -> str:
        return f"{self.source}#p{self.page}#{int(self.bbox[0])}_{int(self.bbox[1])}"

    @property
    def page_label(self) -> str:
        return f"《{self.source}》第{self.page}页"


# ================= 一、矢量组织结构图：连接线图还原 ==============================
def _pt_in_box(p: _PT, box: BBox, tol: float = 4.0) -> bool:
    return box[0] - tol <= p[0] <= box[2] + tol and box[1] - tol <= p[1] <= box[3] + tol


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
    if a_h and b_h:                       # 两条横线：同高且 x 区间相交
        return abs(aylo - bylo) <= tol and not (axhi < bxlo - tol or bxhi < axlo - tol)
    if not a_h and not b_h:               # 两条竖线：同列且 y 区间相交
        return abs(axlo - bxlo) <= tol and not (ayhi < bylo - tol or byhi < aylo - tol)
    if a_h:                               # a 横 b 竖
        return axlo - tol <= bxlo <= axhi + tol and bylo - tol <= aylo <= byhi + tol
    return bxlo - tol <= axlo <= bxhi + tol and aylo - tol <= bylo <= ayhi + tol


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
        if kinds != {"re"}:                       # 仅取“纯矩形”作为节点框（排除线段集合）
            continue
        box = (r.x0, r.y0, r.x1, r.y1)
        # 只保留完全落在图形区域内的矩形，且面积足够（过滤装饰细线）
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


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", text or ""))


# 子节点名称的公共后缀 → 计量单位（用于生成「销售部由4个部门构成」这类可检索表述）
_CHILD_UNITS = (("销售处", "销售处"), ("委员会", "委员会"), ("分公司", "分公司"),
                ("中心", "中心"), ("部门", "部门"))


def _child_unit(kids: List[str]) -> str:
    """按子节点名称的公共后缀推断计量单位；后缀不统一时用「机构」。"""
    if not kids:
        return "机构"
    for suf, unit in _CHILD_UNITS:
        if all(k.endswith(suf) for k in kids):
            return unit
    if all(k.endswith("部") for k in kids):
        return "部门"
    return "机构"


def _build_tree(boxes: List[Tuple[BBox, str]],
                segs: List[Tuple[_PT, _PT]], tol: float = 4.0
                ) -> Tuple[str, Dict[str, List[str]]]:
    """自上而下遍历连接线图，还原「父 → 子」层级。

    关键难点：组织结构图用**共享母线**连接同一父节点下的多个子节点。若做无向连通
    遍历，兄弟节点会经由母线互相连通，产生「A 是 B 的子节点、B 也是 A 的子节点」
    的错误。本实现用**母线归属法**消除该问题：

    1. 只把「不锚定在任何节点框上」的线段（即母线）标记为**已被占用**；
    2. 从父节点出发，沿其引线 → 母线 → 各子节点引线推进；
    3. 子节点再展开时，父节点引线与母线均已占用，无法回溯，兄弟亦不可达。

    这样得到的是一棵**有向树**，与人工阅读层级一致。
    """
    if not boxes:
        return "", {}

    def box_at(pt: _PT) -> Optional[str]:
        for b in boxes:
            if _pt_in_box(pt, b[0], tol):
                return b[1]
        return None

    def connectors(name: str, owned: set) -> List[int]:
        """锚定在该节点上、且未被占用的引线（线段索引）。"""
        pbox = next(b[0] for b in boxes if b[1] == name)
        out = []
        for i, s in enumerate(segs):
            if i in owned:
                continue
            if _pt_in_box(s[0], pbox, tol) != _pt_in_box(s[1], pbox, tol):
                out.append(i)
        return out

    # 根节点：位置最靠上、且拥有引线的节点（组织结构图的“股东大会”）
    has_conn = [b for b in boxes if connectors(b[1], set())]
    if not has_conn:
        return "", {}
    has_conn.sort(key=lambda b: (b[0][1], b[0][0]))
    root = has_conn[0][1]

    owned: set = set()          # 已被占用的线段（母线 / 祖先引线）
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

        # 母线（不锚定任何节点）与“已访问节点的引线”占用，阻断回溯与兄弟互连
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


def parse_org_chart(page, region: FigureRegion) -> Tuple[str, List[Tuple[str, str]]]:
    """还原矢量组织结构图：返回 (语义文本, 父子关系列表)。"""
    boxes = [b for b in _node_boxes(page, region.bbox) if _has_cjk(b[1])]
    if len(boxes) < 3:
        return "", []
    segs = _region_segments(page, region.bbox)
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
            # 每行以句号收尾：保证下游「句子级抽取」能把它切成独立候选句
            lines.append(f"{n}由{len(kids)}个{_child_unit(kids)}构成：{'、'.join(kids)}。")
    covered = {n for rel in relations for n in rel}
    orphans = [b[1] for b in boxes if b[1] not in covered and b[1] != root]
    if orphans:
        lines.append("其他部门/机构：" + "、".join(orphans) + "。")

    cap = region.caption or "组织结构图"
    text = (f"【组织结构图】{cap}（{region.page_label}）\n"
            + "\n".join(lines))
    return text, relations


# ================= 二、位图统计图：OCR + 数值配对 ===============================
_OCR_ENGINE = None
_OCR_TRIED = False


def _get_ocr():
    """懒加载 rapidocr（不可用则返回 None，主链路降级）。"""
    global _OCR_ENGINE, _OCR_TRIED
    if _OCR_TRIED:
        return _OCR_ENGINE
    _OCR_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
        logger.info("OCR 引擎就绪: rapidocr-onnxruntime")
    except Exception as exc:
        logger.warning("OCR 引擎不可用（位图图形将降级为仅图题/类型）: %s", exc)
        _OCR_ENGINE = None
    return _OCR_ENGINE


def ocr_available() -> bool:
    return _get_ocr() is not None


def _norm_ocr(text: str) -> str:
    """OCR 结果轻量纠错（大小写 / 常见形近）。"""
    t = re.sub(r"\s+", "", str(text or ""))
    t = re.sub(r"(?i)\bic\s*卡", "IC卡", t)      # “Ic卡” → “IC卡”
    return t.strip()


def _preprocess_for_ocr(image_path: str):
    """OCR 前处理：去除浅灰水印，返回三通道数组（失败则回退原图路径）。

    招股说明书图形区域叠加了浅灰水印（作者名/机构名），OCR 会把它误识为图内标签，
    污染「系列名—数值」配对。水印亮度明显高于图内文字，故按亮度阈值置白即可滤除。
    """
    try:
        import numpy as np
        from PIL import Image

        arr = np.array(Image.open(image_path).convert("L"))
        arr = np.where(arr > config.OCR_WATERMARK_LUMA, 255, arr).astype("uint8")
        return np.stack([arr] * 3, axis=-1)
    except Exception:
        return image_path


def _ocr_tokens(image_path: str) -> List[dict]:
    """OCR 识别图形区域内的文字，返回 [{text,x,y,w,h}]。"""
    ocr = _get_ocr()
    if ocr is None or not image_path:
        return []
    src = _preprocess_for_ocr(image_path)
    try:
        result, _ = ocr(src)
    except Exception as exc:
        logger.warning("OCR 识别失败 %s: %s", image_path, exc)
        return []
    out: List[dict] = []
    for item in (result or []):
        box, text = item[0], item[1]
        text = _norm_ocr(text)
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({
            "text": text, "x": min(xs), "y": min(ys),
            "w": max(xs) - min(xs), "h": max(ys) - min(ys),
        })
    return out


def _is_value(text: str) -> bool:
    """token 是否“以数值为主体”（如 10.5% / 2515 / -2.0%）。"""
    t = text.strip()
    return bool(re.fullmatch(r"[-+]?\d[\d,，.]*\s*[%％]?", t)) or bool(_PCT_RE.fullmatch(t))


def _pct_num(v: str) -> float:
    """从「12.5%」这类文本中取出数值；无法解析返回 0。"""
    m = _PCT_RE.search(v or "")
    return float(m.group(1)) if m else 0.0


def _pair_vertical(labels: List[dict], values: List[dict],
                   max_dy: float = 70.0, max_dx: float = 80.0) -> List[Tuple[str, str]]:
    """「标签在上、数值在下」的竖排块配对（饼图常见）。

    饼图各扇区的「系列名 / 数值 / 占比」常排成一小段竖排文字块，块内自上而下依次为
    标签与数值。故对每个标签，取「位于其下方、横向最近的数值」配对。
    """
    pairs: List[Tuple[str, str]] = []
    used = set()
    for lb in sorted(labels, key=lambda t: (t["y"], t["x"])):
        lx = lb["x"] + lb["w"] / 2
        ly = lb["y"] + lb["h"]
        best, bd = None, 1e9
        for v in values:
            if id(v) in used:
                continue
            vx = v["x"] + v["w"] / 2
            if v["y"] < ly - 6 or abs(vx - lx) > max_dx:
                continue
            d = ((vx - lx) ** 2 + (v["y"] - ly) ** 2) ** 0.5
            if d < bd:
                best, bd = v, d
        if best is not None and bd <= max_dy:
            used.add(id(best))
            pairs.append((lb["text"], best["text"]))
    return pairs


def _pair_labels_values(tokens: List[dict]) -> List[Tuple[str, str]]:
    """把「系列名」与「数值」配对，形成 (系列名, 取值)。

    两类统计图的配对策略不同：
    - **条形 / 柱状图**：系列名排成一列（左），数值排成一列（右），二者按 y 顺序
      一一对应。此时用「按 y 排序后顺序配对」最稳（可正确处理数值标签与柱体错位，
      如「汽车」的 14.0% 标签略高于其系列名）。
    - **饼图**：系列名与数值散落在扇区附近，用「最近邻」配对。
    """
    values = [t for t in tokens if _is_value(t["text"])]
    labels = [t for t in tokens if not _is_value(t["text"])]
    if not values or not labels:
        return []

    # 剔除位于所有数值之上 / 之下的“标题”文字（如“增长率”）
    v_lo = min(v["y"] for v in values)
    v_hi = max(v["y"] + v["h"] for v in values)
    body_labels = [t for t in labels if t["y"] + t["h"] / 2 >= v_lo - 6
                   and t["y"] + t["h"] / 2 <= v_hi + 6]

    x_spread = (max(t["x"] for t in body_labels) - min(t["x"] for t in body_labels)) \
        if body_labels else 1e9
    # ---- 条形/柱状图：左列系列名 + 右列数值，顺序配对 ----
    if len(body_labels) == len(values) and x_spread < 160 and len(values) >= 3:
        labels_sorted = sorted(body_labels, key=lambda t: t["y"] + t["h"] / 2)
        values_sorted = sorted(values, key=lambda t: t["y"] + t["h"] / 2)
        return [(lb["text"], v["text"]) for lb, v in zip(labels_sorted, values_sorted)]

    # ---- 饼图：标签在上、数值在下的「竖排块」配对 ----
    vpairs = _pair_vertical(body_labels, values)
    if len(vpairs) >= max(3, len(values) - 2):
        return vpairs

    # ---- 其他：最近邻配对 ----
    pairs: List[Tuple[str, str]] = []
    used = set()
    for v in sorted(values, key=lambda t: t["y"]):
        vx, vy = v["x"], v["y"] + v["h"] / 2
        best, best_d = None, 1e9
        for lb in labels:
            if id(lb) in used:
                continue
            lx, ly = lb["x"] + lb["w"], lb["y"] + lb["h"] / 2
            d = ((lx - vx) ** 2 + (ly - vy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = lb, d
        if best is not None and best_d <= 90:
            used.add(id(best))
            pairs.append((best["text"], v["text"]))
    return pairs


def parse_chart(page, region: FigureRegion) -> Tuple[str, List[Tuple[str, str]]]:
    """解析位图统计图：返回 (语义文本, 键值对)。"""
    tokens = _ocr_tokens(region.image_path)
    cap = region.caption or "统计图"
    if not tokens:
        text = (f"【{region.figure_type or '统计图'}】{cap}（{region.page_label}）\n"
                f"（图内数值需图像识别，当前 OCR 不可用）")
        return text, []

    lines = [t["text"] for t in sorted(tokens, key=lambda t: (round(t["y"] / 12), t["x"]))]
    pairs = _pair_labels_values(tokens)
    kv: List[Tuple[str, str]] = list(pairs)

    # 派生结论：若为“增长率”类图（存在负值或图内出现“增长率”字样），给出最快/负增长
    conclusions: List[str] = []
    # 系列名须是真实类别名，排除 OCR 把刻度区间（11%-15%）误当系列名的情形
    pct = [(name, val) for name, val in pairs
           if _PCT_RE.search(val) and _is_category(name)]
    # 仅当图内明确出现「增长率」字样、或存在负值时才判定为增长率图；
    # 旧逻辑用「率」字判定，会把「获利率」「毛利率」等图也误判为增长率图。
    has_growth = "增长率" in cap or any("增长率" in t["text"] for t in tokens)
    has_neg = any(v.strip().startswith("-") for _, v in pct)
    if pct and (has_growth or has_neg):
        ranked = sorted(pct, key=lambda x: _pct_num(x[1]), reverse=True)
        fastest = ranked[0]
        neg = [x for x in ranked if _pct_num(x[1]) < 0]
        conclusions.append(f"增长率最快的是{fastest[0]}（{fastest[1]}）")
        if neg:
            slow = neg[-1]
            conclusions.append(f"负增长的是{slow[0]}（{slow[1]}）")
    # 派生结论：结构占比类图，给出占比最高的行业
    share = [(name, val) for name, val in pairs
             if _PCT_RE.search(val) and _is_category(name) and "结构" in cap]
    if len(share) >= 3 and not has_growth and not has_neg:
        top = max(share, key=lambda x: _pct_num(x[1]))
        conclusions.append(f"占比最高的是{top[0]}（{top[1]}）")

    parts = [f"【{region.figure_type or '统计图'}】{cap}（{region.page_label}）",
             "图内文字识别结果：" + "；".join(lines)]
    if kv:
        parts.append("图内数值：" + "；".join(f"{k}={v}" for k, v in kv))
    if conclusions:
        parts.append("由图可知：" + "；".join(conclusions))
    return "\n".join(parts), kv


# ================= 三、类型判定（Chinese-CLIP 零样本分类） ======================
def classify_figure(region: FigureRegion) -> Tuple[str, float]:
    """用 Chinese-CLIP 对图形做零样本类型分类；不可用时用几何启发式兜底。"""
    if config.CLIP_ENABLE and region.image_path:
        try:
            from src.image_encoder import get_encoder

            enc = get_encoder()
            if enc.available:
                ranked = enc.classify(region.image_path)
                if ranked:
                    return ranked[0]
        except Exception as exc:
            logger.debug("CLIP 分类失败（回退启发式）: %s", exc)
    # 启发式兜底：矢量簇且矩形多 → 组织结构图；位图 → 统计图
    if region.kind.startswith("vector") and region.n_boxes >= 4:
        return "组织结构图", 0.0
    return "统计图", 0.0


# ================= 四、主入口 ==================================================
def analyze_figure(page, region: FigureRegion) -> FigureSemantics:
    """解析单个图形区域。"""
    ftype, score = classify_figure(region)
    sem = FigureSemantics(
        source=region.source, page=region.page, bbox=region.bbox,
        kind=region.kind, image_path=region.image_path, caption=region.caption,
        figure_type=ftype, type_score=round(float(score), 4),
    )
    # 矢量簇（组织结构图 / 流程图）→ 连接线图还原；位图 → OCR 数值解析
    if region.kind.startswith("vector") and region.n_boxes >= 4:
        text, rels = parse_org_chart(page, region)
        sem.relations = rels
        if text:
            sem.semantic_text = text
    if not sem.semantic_text:
        text, kv = parse_chart(page, region)
        sem.semantic_text = text
        sem.key_values = kv
    return sem


def analyze_figures(regions: Sequence[FigureRegion]) -> List[FigureSemantics]:
    """批量解析图形区域（按源文档分组，每份 PDF 只打开一次）。"""
    by_source: Dict[str, List[FigureRegion]] = {}
    for r in regions:
        by_source.setdefault(r.source, []).append(r)

    out: List[FigureSemantics] = []
    for source, figs in by_source.items():
        path = None
        for p in config.PDF_PATHS:
            if Path(p).name == source:
                path = p
                break
        if path is None or not Path(path).exists():
            for r in figs:
                out.append(FigureSemantics(
                    source=r.source, page=r.page, bbox=r.bbox, kind=r.kind,
                    image_path=r.image_path, caption=r.caption,
                    figure_type="统计图",
                    semantic_text=f"【图形】{r.caption or '图形'}（{r.page_label}）",
                ))
            continue
        with pymupdf.open(str(path)) as doc:
            for r in figs:
                try:
                    page = doc.load_page(r.page - 1)
                    out.append(analyze_figure(page, r))
                except Exception as exc:
                    logger.warning("图形语义解析失败 %s: %s", r.key, exc)
    logger.info("图像语义解析完成：%d 个图形", len(out))
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO)
    from src.figure_extractor import extract_all_figures

    figs = extract_all_figures()
    sems = analyze_figures(figs)
    print(f"共解析 {len(sems)} 个图形")
    for s in sems:
        if s.relations or s.key_values:
            print("-" * 60)
            print(s.key, s.figure_type)
            print(s.semantic_text[:600])