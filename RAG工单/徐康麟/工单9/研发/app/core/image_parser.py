# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单4 图像资产提取与去重（image_parser）。

任务：**人工智能NLP-RAG-图像内容解析及检索优化**。

本模块是「图像内容解析链路」的第 1 环，产出「可回溯引用的图像资产清单」：
    PDF 每页每张位图 → xref / 像素尺寸 / bbox / 字节 sha1 / 是否水印装饰 → JSONL。

设计依据：``设计/图像解析方案.md`` §1~§2（去重规则 R1~R6）。
所有页码一律 **1-based 物理页码**（``doc[i]`` → ``i + 1``），
与 ``设计/验收标准.md`` 的引用口径一致。

------------------------------------------------------------------------
实测坑（务必保留，换 PyMuPDF 版本前先复测）
------------------------------------------------------------------------
1. ``page.get_image_info()`` **不传** ``xrefs=True`` 时 ``it["xref"]`` 可能为 ``None``，
   于是「位置 ↔ xref」的映射会丢。本模块一律用
   ``page.get_image_info(xrefs=True)``，并在 ``xref`` 缺失时用
   ``(width, height, bbox)`` 三元组回退匹配；回退命中会记 ``xref_source="bbox-fallback"``。
2. 水印**不是**「纯色小图」那么简单：实测水印底图是**纯黑 143×127 PNG（154 B）**，
   文字在同 xref 的 smask 里（PDF1 112173↔112174、PDF2 2338↔2339）。
   因此**禁止**用「纯色/近空白」启发式判水印，唯一可靠判据是 **xref 跨页重复度（R1）**。
3. 不能用 ``get_text()`` 的最后一行当页脚（会取到别的行）；页脚要取该页 **y 最大的 span**。
   本模块的 ``page_footer_number()`` 按此实现，且**只用于自检**，不参与图像判重。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import fitz  # PyMuPDF（1.28.2 实测可用；fitz 别名有 DeprecationWarning，属正常）

__all__ = [
    "PAGE_BASIS",
    "ImageAsset",
    "WatermarkInfo",
    "extract_assets_for_document",
    "scan_xref_frequency",
    "page_footer_number",
    "write_jsonl",
    "read_jsonl",
    "asset_to_record",
]

# ---------------------------------------------------------------------------
# 常量：去重规则阈值（对应 设计/图像解析方案.md §2.2 的 R1~R6）
# ---------------------------------------------------------------------------
PAGE_BASIS = "1-based-physical"

R1_WATERMARK_XREF_RATIO = 0.5      # R1：xref 出现次数 ≥ 页数 × 0.5 → 水印
R2_MIN_SIDE = 4                    # R2：min(w,h) ≤ 4 px → 装饰线条
R2_MAX_ASPECT = 20.0               # R2：max/min ≥ 20 → 装饰长条
R3_MIN_AREA = 4096                 # R3：w×h ≤ 4096 → 可疑小图标
R4_MIN_PAGE_AREA_RATIO = 0.005     # R4：bbox 面积 < 页面面积 × 0.5% → 页眉页脚装饰
R6_CONTAIN_RATIO = 0.95            # R6：被包含且面积比 > 0.95 → 只留大的

# 章的标题/来源关键词（用于把图与其上方/下方的说明文字关联）
_CAPTION_KEYS = ("图", "表", "资料来源", "数据来源", "单位：", "单位:", "注：", "注:")
_SOURCE_RE = re.compile(r"(资料来源|数据来源)\s*[:：]\s*(.+)")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class WatermarkInfo:
    """R1 命中的水印信息（xref 黑名单 + 其 smask）。"""

    corpus: str
    page_count: int
    xref_hits: dict[str, int] = field(default_factory=dict)
    blacklist: list[int] = field(default_factory=list)
    smask_blacklist: list[int] = field(default_factory=list)
    threshold_ratio: float = R1_WATERMARK_XREF_RATIO

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageAsset:
    """一张位图的**一次出现**（同 xref 在多页出现 = 多条记录，便于页码引用）。"""

    corpus: str
    page: int                      # 1-based 物理页码
    occurrence: int                # 该页内第几次出现（0-based）
    xref: int
    smask: int | None
    width: int
    height: int
    colorspace: str
    bpc: int
    filt: str
    size_bytes: int
    sha1: str
    bbox: list[float]
    keep: bool = True
    drop_reason: str = ""
    xref_source: str = "get_image_info(xrefs=True)"
    sha1_shared_pages: list[int] = field(default_factory=list)
    caption: str = ""
    source_note: str = ""
    unit: str = ""

    @property
    def pixel_area(self) -> int:
        return self.width * self.height

    @property
    def bbox_area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    """写 JSONL（UTF-8 无 BOM，``ensure_ascii=False`` 保留中文），返回行数。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """读 JSONL；文件不存在返回空列表（不抛异常，便于下游「缺失即降级」）。"""

    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _sha1_of(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def page_footer_number(page: "fitz.Page") -> int | None:
    """取该页 **y 最大** 的 span 文本并尝试解析为数字（印刷页码）。

    实测口径：两份 PDF 均为「印刷页码 = 物理页码 − 1」。
    反面做法（已实测踩坑）：用 ``get_text().splitlines()[-1]`` 会取到别的行，
    这正是任务书早期「PDF2 页码不吻合」误判的根因。
    """

    best_y = -1.0
    best_txt = ""
    d = page.get_text("dict")
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if not txt:
                    continue
                y = float(span["bbox"][3])
                if y > best_y:
                    best_y, best_txt = y, txt
    m = re.fullmatch(r"[-–—]?\s*(\d{1,4})", best_txt)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 全量 xref 频次扫描（R1 水印判据）
# ---------------------------------------------------------------------------
def scan_xref_frequency(doc: "fitz.Document") -> dict[int, int]:
    """逐页统计每个 xref 的**出现页次数**（不是图片张数：同页重复只算 1 次）。

    实测：PDF2 每页 15 张同 xref（2338）水印，若按「张数」统计会到 5250；
    按「页数」统计得 350/350 = 100%，判据更稳。
    """

    freq: dict[int, int] = {}
    for pno in range(doc.page_count):
        page = doc[pno]
        seen: set[int] = set()
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            infos = []
        for it in infos:
            xref = it.get("xref")
            if isinstance(xref, int) and xref > 0:
                seen.add(xref)
        for xref in seen:
            freq[xref] = freq.get(xref, 0) + 1
    return freq


def build_watermark_blacklist(
    corpus: str,
    page_count: int,
    freq: dict[int, int],
    *,
    ratio: float = R1_WATERMARK_XREF_RATIO,
) -> WatermarkInfo:
    """按 R1 生成水印黑名单，并把黑名单 xref 的 smask 一并登记（R1b）。"""

    threshold = max(2, int(page_count * ratio))
    hits = {str(k): v for k, v in sorted(freq.items()) if v >= threshold}
    blacklist = sorted(int(k) for k in hits)
    info = WatermarkInfo(corpus=corpus, page_count=page_count, xref_hits=hits, blacklist=blacklist)
    # R1b：登记 smask（需要 doc 才能读；在 extract_assets_for_document 里补齐）
    return info


# ---------------------------------------------------------------------------
# 单页图像提取
# ---------------------------------------------------------------------------
def _image_infos(page: "fitz.Page") -> list[dict[str, Any]]:
    """取该页的位图占位信息（含 bbox），并保证每条都有可用的 ``xref``。"""

    try:
        infos = list(page.get_image_info(xrefs=True))
    except Exception:
        infos = list(page.get_image_info())

    # 回退：用 (width,height) 与 get_images(full=True) 的 xref 逐个对齐。
    # 只在 xref 缺失时才启动，且回退结果会被标记，便于发现 PyMuPDF 行为变化。
    missing = [it for it in infos if not isinstance(it.get("xref"), int) or it.get("xref") <= 0]
    if missing:
        try:
            raw = list(page.get_images(full=True))
        except Exception:
            raw = []
        for it in missing:
            cand = None
            for g in raw:
                if int(g[2]) == int(it.get("width", -1)) and int(g[3]) == int(it.get("height", -1)):
                    cand = int(g[0])
                    break
            if cand is None and raw:
                cand = int(raw[0][0])
            if cand is not None:
                it["xref"] = cand
                it["_xref_source"] = "bbox-fallback"
    for it in infos:
        it.setdefault("_xref_source", "get_image_info(xrefs=True)")
    return infos


def extract_assets_for_document(
    pdf_path: Path | str,
    *,
    pages: Sequence[int] | None = None,
    watermark: WatermarkInfo | None = None,
) -> tuple[list[ImageAsset], WatermarkInfo]:
    """提取一份 PDF 的图像资产。

    参数
    ----
    pdf_path   : PDF 路径。
    pages      : 只处理这些 **1-based 物理页码**（``None`` = 全量）。注意：
                 R1 水印判据**永远按全量页频次**计算（即使只处理几页），
                 否则「只跑 4 页」会把水印误判成内容图。
    watermark  : 复用已算好的水印信息（批量跑多个 PDF 时避免重复扫描）。

    返回
    ----
    (assets, watermark_info)
    """

    pdf_path = Path(pdf_path)
    corpus = pdf_path.name
    doc = fitz.open(str(pdf_path))
    try:
        page_count = doc.page_count
        if watermark is None:
            freq = scan_xref_frequency(doc)
            watermark = build_watermark_blacklist(corpus, page_count, freq)
        wl = set(watermark.blacklist)

        selected = set(pages) if pages else None
        assets: list[ImageAsset] = []
        # sha1 → 记录索引，用于 R5 合并（记录全部出现页码）
        sha1_index: dict[str, list[ImageAsset]] = {}
        smask_seen: set[int] = set()

        for pno in range(page_count):
            page_no = pno + 1
            if selected is not None and page_no not in selected:
                continue
            page = doc[pno]
            try:
                infos = _image_infos(page)
            except Exception:
                infos = []
            page_area = float(page.rect.width * page.rect.height)

            for occ, it in enumerate(infos):
                xref = int(it.get("xref") or 0)
                if xref <= 0:
                    continue
                w = int(it.get("width") or 0)
                h = int(it.get("height") or 0)
                bbox = [float(v) for v in (it.get("bbox") or (0, 0, 0, 0))]
                cs = str(it.get("cs-name") or it.get("colorspace") or "")
                bpc = int(it.get("bpc") or 8)
                rect = None

                # 读原始字节（含滤镜与 smask）
                try:
                    raw = doc.extract_image(xref)
                    data = raw.get("image") or b""
                    smask = raw.get("smask")
                    smask = int(smask) if isinstance(smask, int) and smask > 0 else None
                    filt = str(raw.get("ext") or "")
                    if not w:
                        w = int(raw.get("width") or 0)
                    if not h:
                        h = int(raw.get("height") or 0)
                except Exception as exc:  # 读不出来必须留痕，不得静默跳过
                    data, smask, filt = b"", None, ""
                    assets.append(
                        ImageAsset(
                            corpus=corpus, page=page_no, occurrence=occ, xref=xref, smask=None,
                            width=w, height=h, colorspace=cs, bpc=bpc, filt="",
                            size_bytes=0, sha1="", bbox=bbox, keep=False,
                            drop_reason="extract_image 失败: %s" % type(exc).__name__,
                            xref_source=str(it.get("_xref_source")),
                        )
                    )
                    continue

                if smask:
                    smask_seen.add(smask)

                sha1 = _sha1_of(data) if data else ""
                a = ImageAsset(
                    corpus=corpus, page=page_no, occurrence=occ, xref=xref, smask=smask,
                    width=w, height=h, colorspace=cs, bpc=bpc, filt=filt,
                    size_bytes=len(data), sha1=sha1, bbox=bbox,
                    xref_source=str(it.get("_xref_source")),
                )

                # ---- 规则判定（顺序固定：先杀水印） ----
                reason = ""
                if xref in wl or (smask and smask in wl):
                    reason = "R1:水印(xref/page=%d/%d)" % (
                        watermark.xref_hits.get(str(xref), 0), page_count
                    )
                elif h and w and min(w, h) <= R2_MIN_SIDE:
                    reason = "R2:装饰线条(min side=%d)" % min(w, h)
                elif h and w and (max(w, h) / max(1, min(w, h))) >= R2_MAX_ASPECT:
                    reason = "R2:装饰长条(aspect=%.1f)" % (max(w, h) / max(1, min(w, h)))
                elif w and h and w * h <= R3_MIN_AREA:
                    reason = "R3:可疑小图标(area=%d)" % (w * h)
                elif page_area > 0 and a.bbox_area < page_area * R4_MIN_PAGE_AREA_RATIO:
                    reason = "R4:页眉页脚装饰(bbox面积=%.0fpt²)" % a.bbox_area
                else:
                    reason = ""
                a.keep = not reason
                a.drop_reason = reason
                assets.append(a)
                if sha1:
                    sha1_index.setdefault(sha1, []).append(a)

        # ---- R5：同 sha1 合并（记录全部出现页），保留首个为 keep，其余标记 ----
        for sha1, group in sha1_index.items():
            pages_all = sorted({g.page for g in group})
            for g in group:
                g.sha1_shared_pages = pages_all
            if len(group) > 1:
                for g in group[1:]:
                    if g.keep:
                        g.keep = False
                        g.drop_reason = "R5:与第 %d 页同 sha1 重复" % group[0].page

        # ---- R6：bbox 完全包含且面积比 > 0.95 → 只留大的 ----
        for a in assets:
            if not a.keep:
                continue
            for b in assets:
                if b is a or not b.keep or b.page != a.page or b.sha1 == a.sha1:
                    continue
                if b.bbox_area > a.bbox_area and _bbox_contains(a.bbox, b.bbox):
                    ratio = a.bbox_area / max(1.0, b.bbox_area)
                    if ratio > R6_CONTAIN_RATIO:
                        a.keep = False
                        a.drop_reason = "R6:被同页更大图(xref=%d)包含" % b.xref
                        break

        # ---- 语义关联：标题 / 资料来源 / 单位 ----
        for a in assets:
            page = doc[a.page - 1]
            cap, src, unit = _associate_caption(page, a.bbox)
            a.caption, a.source_note, a.unit = cap, src, unit

        # 补齐水印 smask 黑名单
        if smask_seen:
            watermark.smask_blacklist = sorted(smask_seen | set(watermark.smask_blacklist))
        return assets, watermark
    finally:
        doc.close()


def _bbox_contains(inner: Sequence[float], outer: Sequence[float], tol: float = 1.0) -> bool:
    """outer 是否包含 inner（容差 tol pt）。"""

    return (
        outer[0] - tol <= inner[0]
        and outer[1] - tol <= inner[1]
        and outer[2] + tol >= inner[2]
        and outer[3] + tol >= inner[3]
    )


def _associate_caption(page: "fitz.Page", bbox: Sequence[float], *, window: float = 46.0) -> tuple[str, str, str]:
    """把图像与其上方的标题、下方的「资料来源」关联起来。

    规则（几何，不依赖页面模板）：
      * 标题：图 bbox **上方** ``window`` pt 内、且水平区间与图重叠的文本块；
      * 来源：图 bbox **下方** ``window`` pt 内匹配 ``资料来源[:：]...`` 的文本块；
      * 单位：标题里 ``(亿元)`` / ``（%）`` / ``单位：xx`` 的括号内容。
    """

    try:
        blocks = [b for b in page.get_text("blocks") if b[4] and b[4].strip()]
    except Exception:
        return "", "", ""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    above: list[tuple[float, str]] = []
    below: list[tuple[float, str]] = []
    for b in blocks:
        bx0, by0, bx1, by1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
        # 水平重叠
        if bx1 < x0 - 2 or bx0 > x1 + 2:
            continue
        t = re.sub(r"\s+", " ", txt).strip()
        if not t:
            continue
        if by1 <= y0 + 1 and (y0 - by1) <= window:
            above.append((y0 - by1, t))
        elif by0 >= y1 - 1 and (by0 - y1) <= window:
            below.append((by0 - y1, t))

    caption = ""
    for _, t in sorted(above, key=lambda z: z[0]):
        if any(k in t for k in _CAPTION_KEYS) or re.search(r"\d{4}\s*年", t):
            caption = t
    if not caption and above:
        caption = sorted(above, key=lambda z: z[0])[0][1]

    source_note = ""
    for _, t in sorted(below, key=lambda z: z[0]):
        m = _SOURCE_RE.search(t)
        if m:
            source_note = t
            break

    unit = ""
    m = re.search(r"[（(]([^）)]{1,12})[）)]\s*$", caption)
    if m:
        unit = m.group(1)
    if not unit:
        m = re.search(r"单位\s*[:：]\s*([^\s，,。；;]{1,10})", caption or "")
        if m:
            unit = m.group(1)
    return caption, source_note, unit


def asset_to_record(a: ImageAsset) -> dict[str, Any]:
    """转成落盘用的 dict（附带便于检索阅读的中文摘要字段）。"""

    rec = a.to_dict()
    rec["page_basis"] = PAGE_BASIS
    rec["pixel_area"] = a.pixel_area
    rec["bbox_area"] = a.bbox_area
    return rec
