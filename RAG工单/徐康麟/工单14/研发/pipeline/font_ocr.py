# 工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单
"""
附图标记识别引擎（纯像素方案，不依赖任何 OCR 库）

原理
----
低质量图片型专利 PDF 的附图里，附图标记（4、5、6、6'、6"、X1、X2、X3、h1、h2、D、7、10 …）
是以「印刷字符笔画」直接绘制在位图上的。本模块用两种互补手段恢复它们：

1) 字体模板匹配（FontTemplateOCR）
   用系统字体把候选字符（0-9、A-Z、a-z、'、"）渲染成位图模板，
   与页面上切出的字符连通域做「归一化 + 掩码 IoU / 像素余弦」匹配，取最高分。
   —— 不需要 onnxruntime / tesseract / cv2，只需要 numpy + Pillow。

2) 版面文本行恢复（band detection）
   通过行/列墨迹投影切出文本行与字符块，得到字符的坐标 (x, y)。

两份输出合并后即可回答「部件 A 相对部件 B 的位置关系」「尺寸 X1 属于哪两个部件之间」等问题。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 数据结构

@dataclass
class GlyphHit:
    """一个被识别出来的字符及其在图页上的像素坐标。"""
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    source: str = "font_template"

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cx"] = round(self.cx, 1)
        d["cy"] = round(self.cy, 1)
        return d


# ---------------------------------------------------------------- 字体模板

# 常见 Windows 字体；缺失也不报错，逐个尝试
_CJK_FONTS = [
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
]
_LATIN_FONTS = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def _load_font(cands: Iterable[str], size: int):
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_glyph(ch: str, size: int, font_path: str | None = None) -> np.ndarray:
    """把单个字符渲染成 size×size 的二值位图（1=墨迹）。"""
    if font_path:
        font = ImageFont.truetype(font_path, size)
    else:
        font = _load_font(_LATIN_FONTS, size)
    img = Image.new("L", (size * 2, size * 2), 255)
    d = ImageDraw.Draw(img)
    d.text((size // 2, size // 2), ch, font=font, fill=0)
    a = np.array(img)
    ink = a < 160
    if not ink.any():
        return np.zeros((size, size), dtype=np.uint8)
    ys, xs = np.nonzero(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return crop.astype(np.uint8)


def _normalize(mask: np.ndarray, target: int = 32) -> np.ndarray:
    """把字符位图缩放到 target×target（保持长宽比，居中填充）。

    内存友好：使用 uint8，必要时才转 float32（由调用方决定）。
    """
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros((target, target), dtype=np.uint8)
    s = target / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    im = Image.fromarray((mask * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    a = (np.array(im) > 100).astype(np.uint8)
    out = np.zeros((target, target), dtype=np.uint8)
    oy, ox = (target - nh) // 2, (target - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = a
    return out


class FontTemplateOCR:
    """用系统字体模板识别「数字 / 拉丁字母 / 撇号 / 双撇号」字符。

    只识别工程图纸标注会用到的字符集，因此误识率远低于通用 OCR。
    """

    CHARSET = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abcdefghijklmnopqrstuvwxyz") + ["'", '"']

    def __init__(self, target: int = 32, font_paths: list[str] | None = None):
        self.target = target
        self.font_paths = font_paths or (_LATIN_FONTS[:3] + _CJK_FONTS[:1])
        # 模板存成单个 (N, target, target) float32 张量 + 对应字符表，避免大量小对象
        self.tpl: np.ndarray | None = None
        self.tpl_chars: list[str] = []
        self._build()

    def _build(self):
        mats, chars = [], []
        seen: set[bytes] = set()
        for ch in self.CHARSET:
            for fp in self.font_paths:
                if not os.path.exists(fp):
                    continue
                for size in (34, 48):
                    try:
                        m = render_glyph(ch, size, fp)
                    except Exception:
                        continue
                    if m.size == 0 or int(m.sum()) < 4:
                        continue
                    n = _normalize(m, self.target)
                    key = n.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    mats.append(n)
                    chars.append(ch)
                    del m, n
        if mats:
            self.tpl = np.stack(mats).astype(np.float32)
            # 预归一化，识别时只做一次点积
            norms = np.linalg.norm(self.tpl.reshape(len(mats), -1), axis=1)
            norms[norms == 0] = 1.0
            self.tpl /= norms[:, None, None]
            self.tpl_flat = self.tpl.reshape(len(mats), -1)
        self.tpl_chars = chars

    # ------------------------------------------------------------ 匹配
    def classify(self, mask: np.ndarray) -> tuple[str, float]:
        """识别单个字符位图，返回 (字符, 置信度)。"""
        if mask.sum() < 4 or self.tpl is None:
            return "", 0.0
        q = _normalize(mask, self.target).astype(np.float32).ravel()
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return "", 0.0
        sims = self.tpl_flat @ (q / qn)      # 余弦相似度（模板已单位化）
        i = int(np.argmax(sims))
        return self.tpl_chars[i], float(sims[i])


# ---------------------------------------------------------------- 字符切分

def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """把 0/1 序列切成连续 1 的区间。"""
    out, s = [], None
    for i, v in enumerate(flags.tolist()):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(flags) - 1))
    return out


def segment_glyphs(mask: np.ndarray, min_h: int = 6, max_h: int = 200,
                   max_w: int = 200, min_px: int = 6) -> list[tuple[int, int, int, int, np.ndarray]]:
    """在二值图里切出字符候选连通块（用列投影 + 行投影的轻量近似）。

    返回 [(x0, y0, x1, y1, submask), ...]，坐标相对于传入的 mask。
    """
    if not mask.any():
        return []
    rowflags = mask.any(axis=1)
    out = []
    for y0, y1 in _runs(rowflags):
        if y1 - y0 + 1 < min_h:
            continue
        strip = mask[y0:y1 + 1]
        colflags = strip.any(axis=0)
        for x0, x1 in _runs(colflags):
            h, w = y1 - y0 + 1, x1 - x0 + 1
            if h < min_h or h > max_h or w > max_w:
                continue
            sub = strip[:, x0:x1 + 1]
            if int(sub.sum()) < min_px:
                continue
            out.append((x0, y0, x1, y1, sub))
    return out


# ---------------------------------------------------------------- 页面级 API

def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    sum_all = float(np.dot(np.arange(256), hist))
    sumB = 0.0; wB = 0.0; best = -1.0; thr = 127
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > best:
            best, thr = var, t
    return int(thr)


def binarize(gray: np.ndarray, thr: int | None = None, invert: bool = False) -> np.ndarray:
    t = otsu_threshold(gray) if thr is None else thr
    ink = gray < t
    if invert:
        ink = ~ink
    return ink.astype(np.uint8)


def extract_numerals(gray: np.ndarray, ocr: FontTemplateOCR,
                     min_score: float = 0.55,
                     min_h: int = 14, max_h: int = 70,
                     min_w: int = 3, max_w: int = 70) -> list[GlyphHit]:
    """从整页灰度图里抽取「附图标记」字符候选。

    只保留字符高度落在 [min_h, max_h] 的连通块（图纸标注字号），
    并用模板匹配打分过滤。
    """
    ink = binarize(gray)
    hits: list[GlyphHit] = []
    for x0, y0, x1, y1, sub in segment_glyphs(ink, min_h=min_h, max_h=max_h, max_w=max_w):
        w = x1 - x0 + 1
        if w < min_w:
            continue
        ch, sc = ocr.classify(sub)
        if ch and sc >= min_score:
            hits.append(GlyphHit(text=ch, x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1), score=round(sc, 4)))
    return hits


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    """8连通域标记（两遍扫描 + 并查集，整型数组，内存占用 ≈ 4×W×H 字节）。

    返回 (labels, n_components, stats)，stats[i] 对应 label=i+1：
        {"label","x0","y0","x1","y1","area","w","h"}
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    nxt = 1
    m = mask.astype(bool)
    for y in range(h):
        row = m[y]
        if not row.any():
            continue
        lab = labels[y]
        lab_prev = labels[y - 1] if y else None
        xs = np.nonzero(row)[0]
        for x in xs.tolist():
            neigh = []
            if x and lab[x - 1]:
                neigh.append(int(lab[x - 1]))
            if lab_prev is not None:
                if lab_prev[x]:
                    neigh.append(int(lab_prev[x]))
                if x and lab_prev[x - 1]:
                    neigh.append(int(lab_prev[x - 1]))
                if x + 1 < w and lab_prev[x + 1]:
                    neigh.append(int(lab_prev[x + 1]))
            if not neigh:
                lab[x] = nxt
                parent.append(nxt)
                nxt += 1
            else:
                mn = min(neigh)
                lab[x] = mn
                for l in neigh:
                    union(l, mn)

    # 扁平化 + 统计
    remap = np.zeros(nxt, dtype=np.int32)
    stats: list[dict] = []
    root2new: dict[int, int] = {}
    ys, xs = np.nonzero(labels)
    for y, x in zip(ys.tolist(), xs.tolist()):
        r = find(int(labels[y, x]))
        ni = root2new.get(r)
        if ni is None:
            ni = len(root2new) + 1
            root2new[r] = ni
            stats.append({"label": ni, "x0": x, "y0": y, "x1": x, "y1": y, "area": 0})
        remap[r] = ni
        s = stats[ni - 1]
        if x < s["x0"]: s["x0"] = x
        if y < s["y0"]: s["y0"] = y
        if x > s["x1"]: s["x1"] = x
        if y > s["y1"]: s["y1"] = y
        s["area"] += 1
    for lab_id in np.unique(labels):
        if lab_id:
            labels[labels == lab_id] = remap[find(int(lab_id))]
    for s in stats:
        s["w"] = s["x1"] - s["x0"] + 1
        s["h"] = s["y1"] - s["y0"] + 1
    return labels, len(stats), stats


def extract_isolated_glyphs(gray: np.ndarray, ocr: FontTemplateOCR,
                            min_h: int = 10, max_h: int = 45,
                            min_w: int = 4, max_w: int = 45,
                            min_area: int = 25, max_area: int = 1200,
                            max_aspect: float = 3.2,
                            min_score: float = 0.42,
                            bbox: tuple[int, int, int, int] | None = None) -> list[GlyphHit]:
    """抽取「孤立且尺寸像字符」的连通域并识别。

    关键点：图纸里的线条会与大连通域粘连，数值标注则多为**孤立小连通域**，
    因此用 面积 / 高宽 / 长宽比 三重过滤先把线条排除，再交给模板匹配。

    bbox: 只在该像素矩形 (x0,y0,x1,y1) 内搜索（用于区域级定向识别）。
    """
    ink = binarize(gray)
    if bbox:
        x0, y0, x1, y1 = bbox
        sub = np.zeros_like(ink)
        sub[y0:y1, x0:x1] = ink[y0:y1, x0:x1]
        ink = sub
    labels, n, stats = connected_components(ink)
    hits: list[GlyphHit] = []
    for s in stats:
        if not (min_h <= s["h"] <= max_h and min_w <= s["w"] <= max_w):
            continue
        if not (min_area <= s["area"] <= max_area):
            continue
        if s["h"] / max(1, s["w"]) > max_aspect or s["w"] / max(1, s["h"]) > max_aspect:
            continue
        # 填充率：字符大约 0.2~0.7；纯斜线填充率很低
        fill = s["area"] / float(s["w"] * s["h"])
        if fill < 0.12:
            continue
        sub = ink[s["y0"]:s["y1"] + 1, s["x0"]:s["x1"] + 1]
        ch, sc = ocr.classify(sub)
        if ch and sc >= min_score:
            hits.append(GlyphHit(text=ch, x0=int(s["x0"]), y0=int(s["y0"]),
                                 x1=int(s["x1"]), y1=int(s["y1"]),
                                 score=round(sc, 4), source="isolated_cc"))
    del labels, ink
    return hits


# 兼容旧名
def extract_numerals(gray, ocr, min_score=0.55, min_h=14, max_h=70, min_w=3, max_w=70):
    """(旧接口) 用独立连通域法抽取标记字符。"""
    return extract_isolated_glyphs(gray, ocr, min_h=min_h, max_h=max_h,
                                   min_w=min_w, max_w=max_w, min_score=min_score)


def merge_into_tokens(hits: list[GlyphHit], gap_x: int = 18, gap_y: int = 10) -> list[dict]:
    """把相邻字符合并成「标记 token」（如 X1、h2、6''、10）。"""
    if not hits:
        return []
    hs = sorted(hits, key=lambda h: (h.cy // 40, h.cx))
    used = [False] * len(hs)
    tokens = []
    for i, a in enumerate(hs):
        if used[i]:
            continue
        used[i] = True
        grp = [a]
        changed = True
        while changed:
            changed = False
            for j, b in enumerate(hs):
                if used[j]:
                    continue
                for g in grp:
                    if abs(b.cy - g.cy) <= gap_y and abs(b.x0 - g.x1) <= gap_x and b.x0 >= g.x0:
                        used[j] = True; grp.append(b); changed = True; break
                    if abs(b.cy - g.cy) <= gap_y and abs(g.x0 - b.x1) <= gap_x and b.x1 <= g.x1:
                        used[j] = True; grp.append(b); changed = True; break
        grp = sorted(grp, key=lambda h: h.x0)
        tokens.append({
            "text": "".join(g.text for g in grp),
            "x0": min(g.x0 for g in grp), "y0": min(g.y0 for g in grp),
            "x1": max(g.x1 for g in grp), "y1": max(g.y1 for g in grp),
            "score": round(float(np.mean([g.score for g in grp])), 4),
            "n": len(grp),
        })
    for t in tokens:
        t["cx"] = round((t["x0"] + t["x1"]) / 2, 1)
        t["cy"] = round((t["y0"] + t["y1"]) / 2, 1)
    tokens.sort(key=lambda t: (t["cy"], t["cx"]))
    return tokens
