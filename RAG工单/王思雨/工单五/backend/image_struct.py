# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：流程图/组织结构图的层级还原。工单五 Q5 问的是"组织结构图里哪个销售部的销售处最多"，
#           这类问题必须知道方框之间的父子关系，单纯 OCR 只能得到一堆散落的标签。
#           做法：几何方法检出方框与连线 → 按连线还原父子关系 → 对每个方框单独 OCR 取标签 → 输出树文本。
import json                                                                     # 打印/调试用
from typing import Dict, List, Tuple, Optional                                  # 类型标注

import numpy as np                                                              # 图像矩阵运算
import cv2                                                                      # 形态学与轮廓：检出框线与连线
from PIL import Image                                                           # 读取图片

from . import config as C                                                       # 复用集中配置（OCR 相关参数）
from .ocr_image import ocr_box, ocr_text                                        # 复用 OCR 能力

MIN_BOX_PX = 40                                                                 # 方框最小边长（像素），滤掉水印碎片与项目符号
MIN_BOX_AREA = 3000                                                             # 方框最小面积，进一步滤噪
SOLID_RATIO = 0.85                                                              # 轮廓面积/外接矩形面积的下限：越接近 1 越像实心矩形
DARK = 100                                                                      # 二值化阈值：水印是灰色(~180)，只保留近黑线条
H_KERNEL = 60                                                                   # 横向长线的最小长度，用于形态学开运算
V_KERNEL = 40                                                                   # 纵向长线的最小长度
EDGE_TOL = 6                                                                    # 端点与框边对齐的容差（像素）


def _detect_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:         # 检出图中的方框
    """用轮廓法找矩形方框；返回去重后的 [(x, y, w, h)]。水印为灰色，被阈值挡在门外。"""
    binary = ((gray < DARK).astype(np.uint8) * 255)                             # 二值化：近黑为前景
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 取全部轮廓（含内轮廓）
    raw: List[Tuple[int, int, int, int]] = []                                   # 候选框
    for c in contours:                                                          # 逐个轮廓判断
        x, y, w, h = cv2.boundingRect(c)                                        # 外接矩形
        if w < MIN_BOX_PX or h < MIN_BOX_PX or w * h < MIN_BOX_AREA:            # 太小的一律不是方框
            continue                                                            # 跳过
        if cv2.contourArea(c) / float(w * h) < SOLID_RATIO:                     # 非矩形轮廓（文字笔画等）
            continue                                                            # 跳过
        raw.append((x, y, w, h))                                                # 收下
    merged: List[Tuple[int, int, int, int]] = []                                # 去重结果（内外轮廓会重复报同一框）
    for x, y, w, h in sorted(set(raw), key=lambda t: (t[1], t[0])):             # 按位置排序，保证结果稳定
        if any(abs(x - a) < 8 and abs(y - b) < 8 and abs(w - c) < 8 and abs(h - d) < 8
               for a, b, c, d in merged):                                       # 与已有框几乎重合
            continue                                                            # 视为重复
        merged.append((x, y, w, h))                                             # 保留
    return merged                                                               # 返回全部方框


def _detect_segments(gray: np.ndarray) -> Tuple[List, List]:                     # 检出横线与竖线
    """形态学开运算抽出长线段；横线返回 (y, x0, x1)，竖线返回 (x, y0, y1)。"""
    binary = (gray < DARK).astype(np.uint8)                                     # 二值化
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,                            # 横线：宽核腐蚀掉短笔画
                              cv2.getStructuringElement(cv2.MORPH_RECT, (H_KERNEL, 1)))  # 只在水平方向保留长条
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,                            # 竖线：高核腐蚀
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, V_KERNEL)))  # 只在垂直方向保留长条
    def unpack(mask: np.ndarray, horizontal: bool) -> List:                     # 把连通域压成线段
        """每个连通域取中位坐标作为线段位置，避免线宽造成抖动。"""
        n, labels = cv2.connectedComponents(mask)                               # 连通域标记
        out = []                                                                # 结果
        for i in range(1, n):                                                   # 跳过背景 0
            ys, xs = np.where(labels == i)                                      # 取该连通域像素
            if horizontal:                                                      # 横线
                out.append((int(np.median(ys)), int(xs.min()), int(xs.max())))  # (y, x0, x1)
            else:                                                               # 竖线
                out.append((int(np.median(xs)), int(ys.min()), int(ys.max())))  # (x, y0, y1)
        return out                                                              # 返回
    return unpack(h_mask, True), unpack(v_mask, False)                          # 横线与竖线


def _box_edges(boxes: List[Tuple[int, int, int, int]]) -> List[tuple]:          # 列出方框自身的四条边
    """方框的四条边也会被线段检测抓到；先把它们登记出来，剩下的线段才是"连线"。"""
    edges = []                                                                  # 结果
    for x, y, w, h in boxes:                                                    # 逐框
        edges.append(("h", y, x, x + w))                                        # 上边
        edges.append(("h", y + h, x, x + w))                                    # 下边
        edges.append(("v", x, y, y + h))                                        # 左边
        edges.append(("v", x + w, y, y + h))                                    # 右边
    return edges                                                                # 返回


def _matches_edge(seg: tuple, kind: str, edges: List[tuple]) -> bool:           # 判断线段是否为某方框的边
    """位置与长度都吻合才算方框边；容差内认为重合，避免线宽造成误判。"""
    a, b, c = seg                                                               # 线段参数（横线为 y,x0,x1；竖线为 x,y0,y1）
    for k, a2, b2, c2 in edges:                                                 # 遍历已知框边
        if k == kind and abs(a - a2) <= 4 and abs(b - b2) <= EDGE_TOL and abs(c - c2) <= EDGE_TOL:  # 三要素都吻合
            return True                                                         # 是框边
    return False                                                                # 不是框边（即连线）


def _touch(box: Tuple[int, int, int, int], px: int, py: int) -> str:            # 判断线段端点与方框的接触类型
    """返回 'parent'（端点接框的下边，说明框在上方）、'child'（接上边，框在下方）、
    'side-right'/'side-left'（接左右边，横向连接）或 ''（不接触）。"""
    bx, by, bw, bh = box                                                        # 拆解方框
    tol = EDGE_TOL + 4                                                          # 端点对齐容差
    if by - tol <= py <= by + bh + tol and bx - tol <= px <= bx + bw + tol:     # 端点要落在框的邻域内才算接触
        if abs(py - (by + bh)) <= tol:                                          # 贴下边 → 框在上，作父
            return "parent"                                                     # 父类型
        if abs(py - by) <= tol:                                                 # 贴上边 → 框在下，作子
            return "child"                                                      # 子类型
        if abs(px - bx) <= tol:                                                 # 贴左边
            return "side-right"                                                 # 框在右侧
        if abs(px - (bx + bw)) <= tol:                                          # 贴右边
            return "side-left"                                                  # 框在左侧
    return ""                                                                   # 未接触


def _segment_endpoints(seg: tuple, kind: str) -> List[Tuple[int, int]]:         # 取线段两端点坐标
    """横线 (y,x0,x1) 的两端是 (x0,y)、(x1,y)；竖线 (x,y0,y1) 的两端是 (x,y0)、(x,y1)。"""
    a, b, c = seg                                                               # 线段参数
    return [(b, a), (c, a)] if kind == "h" else [(a, b), (a, c)]                # 按类型返回两个端点


def _touching(a: tuple, ka: str, b: tuple, kb: str) -> bool:                    # 判断两条线段是否相接
    """横竖相交、或同向且共线重叠，才算接在一起（导线连成一张网）。"""
    if ka == "h" and kb == "h":                                                 # 两条横线：同一高度且区间重叠
        return abs(a[0] - b[0]) <= EDGE_TOL and not (a[2] < b[1] - EDGE_TOL or b[2] < a[1] - EDGE_TOL)  # 判断
    if ka == "v" and kb == "v":                                                 # 两条竖线：同一列且区间重叠
        return abs(a[0] - b[0]) <= EDGE_TOL and not (a[2] < b[1] - EDGE_TOL or b[2] < a[1] - EDGE_TOL)  # 判断
    hy, hx0, hx1 = a if ka == "h" else b                                        # 取出横线的 y 与 x 区间
    vx, vy0, vy1 = a if ka == "v" else b                                        # 取出竖线的 x 与 y 区间
    return hx0 - EDGE_TOL <= vx <= hx1 + EDGE_TOL and vy0 - EDGE_TOL <= hy <= vy1 + EDGE_TOL  # 竖线的 x 落在横线区间内、横线的 y 落在竖线区间内


def build_edges(boxes: List, h_segs: List, v_segs: List) -> List[Tuple[int, int]]:  # 还原父子关系
    """把"框—导线—框"的图还原成 (父框下标, 子框下标) 列表。

    做法：把非框边的线段按相接关系聚成"导线段组"，再统计每组接触到的方框：
    贴下边的是父，贴上边的是子；若没有贴下边的（如专门委员会用横向支线挂在董事会的竖干上），
    则取贴在导线段组左侧的那个框（它在图的右边）作父。
    """
    edges = _box_edges(boxes)                                                   # 方框自身的四条边
    segs: List[Tuple[str, tuple]] = [(kind, s) for kind, group in (("h", h_segs), ("v", v_segs))
                                     for s in group if not _matches_edge(s, kind, edges)]  # 剔除框边，剩下才是导线
    groups: List[List[Tuple[str, tuple]]] = []                                  # 导线段组
    for kind, s in segs:                                                        # 逐个线段归组
        placed = False                                                          # 是否已并入某个组
        for g in groups:                                                        # 遍历已有组
            if any(_touching(s, kind, s2, k2) for k2, s2 in g):                 # 与组内任一线段相接
                g.append((kind, s)); placed = True; break                       # 并入并跳出
        if not placed:                                                          # 与现有组都不相接
            groups.append([(kind, s)])                                          # 新开一组
    for _ in range(len(groups)):                                                # 归组可能因传递性漏并，做一轮合并收敛
        merged: List[List[Tuple[str, tuple]]] = []                              # 合并结果
        for g in groups:                                                        # 遍历每组
            for m in merged:                                                    # 与已收下的组比较
                if any(_touching(s, k, s2, k2) for k, s in g for k2, s2 in m):  # 有相接即合并
                    m.extend(g); break                                          # 合并后跳出
            else:                                                               # 与任何组都不相接
                merged.append(list(g))                                          # 独立成组
        groups = merged                                                         # 收敛一轮
    # 先把每组的"接触情况"算出来，再分两轮定向：
    #   第一轮处理"贴下边"的明确层级；第二轮处理纯横向的组，用"已挂进树的那个框"当父，避免把方向判反。
    groups_contacts: List[Dict[int, str]] = []                                  # 每组：框下标 → 接触类型
    for g in groups:                                                            # 逐组统计
        contacts: Dict[int, str] = {}                                           # 该组的接触表
        for kind, s in g:                                                       # 组内每条线段的两端
            for px, py in _segment_endpoints(s, kind):                          # 逐个端点
                for idx, box in enumerate(boxes):                               # 与每个框比对
                    t = _touch(box, px, py)                                     # 接触类型
                    if t and (idx not in contacts or t == "parent"):            # 记录（父类型优先保留）
                        contacts[idx] = t                                       # 写入
        groups_contacts.append(contacts)                                        # 收下该组
    pairs: List[Tuple[int, int]] = []                                           # 父子对
    for contacts in groups_contacts:                                            # 第一轮：层级明确的组（有框贴在上方导线的下边）
        if len(contacts) < 2:                                                   # 只碰到一个框，不是连接关系
            continue                                                            # 跳过
        parents = [i for i, t in contacts.items() if t == "parent"]             # 贴下边的作父
        if not parents:                                                         # 无明确父，留给第二轮
            continue                                                            # 跳过
        for p in parents:                                                       # 每个父
            for c, t in contacts.items():                                       # 其余框挂到它下面
                if c != p and t != "parent":                                    # 排除自身与其它父
                    pairs.append((p, c))                                        # 记一条父子边
    pending = [c for c in groups_contacts                                       # 第二轮待处理的组：纯横向布局，方向需推断
               if len(c) >= 2 and not any(t == "parent" for t in c.values())]   # 排除第一轮已处理的
    has_parent = {c for _, c in pairs}                                          # 已有父的框：它在树里的位置是确定的
    for _ in range(len(pending) + 1):                                           # 反复迭代直到不再有新进展（顺序无关）
        changed = False                                                         # 本轮是否有更新
        for contacts in pending:                                                # 逐个待定组
            known = [i for i in contacts if i in has_parent]                    # 组内"位置已知"的框
            if len(known) == 1:                                                 # 恰好一个 → 它一定是父
                parents = known                                                 # 定向成功
            elif known:                                                         # 多个已知，暂不处理，等后续轮次
                continue                                                        # 跳过
            else:                                                               # 组内没有任何已知框
                parents = [i for i, t in contacts.items() if t == "side-right"]  # 退回"右边的框作父"的兜底规则
            if not parents:                                                     # 仍然定不出父
                continue                                                        # 跳过
            for p in parents:                                                   # 逐个子关系
                for c in contacts:                                              # 其余框
                    if c != p and (p, c) not in pairs:                          # 排除自身与重复
                        pairs.append((p, c))                                    # 记边
                        changed = True                                          # 标记有更新
            has_parent = {c for _, c in pairs}                                  # 刷新已知表
        if not changed:                                                         # 本轮无更新
            break                                                               # 收敛退出
    return pairs                                                                # 返回全部父子边


def is_flowchart(image_path: str, min_boxes: int = 4, min_edges: int = 2) -> bool:  # 判断是否为流程图类图片
    """方框够多且连线能还原出层级，才值得走结构化还原；否则按普通插图 OCR 处理。"""
    try:                                                                        # 任何一步失败都不阻断解析
        gray = np.array(Image.open(image_path).convert("L"))                    # 转灰度
        boxes = _detect_boxes(gray)                                             # 找方框
        if len(boxes) < min_boxes:                                              # 方框太少
            return False                                                        # 判定为非流程图
        h_segs, v_segs = _detect_segments(gray)                                 # 找线
        return len(set(build_edges(boxes, h_segs, v_segs))) >= min_edges        # 能还原出足够多的父子边才算
    except Exception:                                                           # 图片损坏等
        return False                                                            # 保守判定为普通图


def flowchart_to_text(image_path: str, max_boxes: int = 60) -> str:             # 流程图 → 层级文本
    """输出"父框：子框1、子框2"的显式层级描述 + 缩进树，让检索与大模型都能直接读懂组织关系。"""
    gray = np.array(Image.open(image_path).convert("L"))                        # 转灰度
    boxes = _detect_boxes(gray)[:max_boxes]                                     # 方框（上限保护，避免超大图拖慢）
    h_segs, v_segs = _detect_segments(gray)                                     # 线段
    pairs = sorted(set(build_edges(boxes, h_segs, v_segs)))                     # 去重后的父子边
    labels: Dict[int, str] = {}                                                 # 框下标 → 标签
    for i, (x, y, w, h) in enumerate(boxes):                                    # 逐个方框 OCR 取标签
        labels[i] = ocr_box(image_path, (x, y, w, h))                           # 单框识别（内缩避开框线）
    children: Dict[int, List[int]] = {}                                         # 父 → 子下标列表
    has_parent = set()                                                          # 有父的框，用于找根
    for p, c in pairs:                                                          # 遍历父子边
        children.setdefault(p, []).append(c)                                    # 登记子节点
        has_parent.add(c)                                                       # 标记有父
    lines: List[str] = ["【图片内容：组织结构/流程图】"]                            # 输出头，让检索能识别块性质
    for p, cs in sorted(children.items()):                                      # 逐个父节点输出显式层级
        pname = labels.get(p, "")                                               # 父标签
        cnames = [labels.get(c, "") for c in cs]                                # 子标签
        cnames = [n for n in cnames if n]                                       # 丢掉空标签
        if pname and cnames:                                                    # 两边都有名才输出
            lines.append(f"{pname} 下辖：{'、'.join(cnames)}")                    # 显式关系句，最利于关键词与向量双路召回
    roots = [i for i in range(len(boxes)) if i not in has_parent and labels.get(i)]  # 没有父的框即根
    def dump(i: int, depth: int, seen: set) -> None:                            # 深度优先输出缩进树
        """递归输出，seen 防止图里出现环时无限递归。"""
        if i in seen or depth > 6:                                              # 防环 + 深度保护
            return                                                              # 直接返回
        seen.add(i)                                                             # 标记已访问
        lines.append("  " * depth + labels.get(i, ""))                          # 按层级缩进输出
        for c in sorted(children.get(i, [])):                                   # 递归子节点
            dump(c, depth + 1, seen)                                            # 下一层
    for r in sorted(roots):                                                     # 从每个根出发
        dump(r, 0, set())                                                       # 输出整棵树
    return "\n".join([l for l in lines if l.strip()])                           # 去掉空行后返回


def image_to_text(image_path: str) -> str:                                      # 图像 → 文本的统一入口
    """先判是否流程图：是则做层级还原，否则退回普通整图 OCR。"""
    if not image_path:                                                          # 没有图片路径
        return ""                                                               # 返回空
    try:                                                                        # 结构化还原可能失败
        if is_flowchart(image_path):                                            # 判定为流程图
            txt = flowchart_to_text(image_path)                                 # 层级还原
            if txt and len(txt) > 20:                                           # 有实质内容
                return txt                                                      # 采用
            print(f"[图像] {image_path} 层级还原结果过短，退回整图 OCR")           # 记录降级
    except Exception as exc:                                                    # 任何异常都退回 OCR
        print(f"[图像] 结构化还原失败({type(exc).__name__}: {exc})，退回整图 OCR")  # 明确报因
    try:                                                                        # 整图 OCR 也可能失败
        return ocr_text(image_path)                                             # 普通 OCR
    except Exception as exc:                                                    # OCR 不可用（缺模型等）
        print(f"[图像] OCR 不可用：{type(exc).__name__}: {exc}")                 # 明确报因
        return ""                                                               # 返回空，不影响其它块
