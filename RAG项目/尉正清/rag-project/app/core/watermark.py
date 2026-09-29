# app/core/watermark.py
"""PDF 水印检测与擦除。

针对三类常见水印：
  1. 文字水印 —— 多页相近位置重复出现的文本（含斜向旋转的），如机构名、网址、"机密"。
  2. 图片水印 —— 同一图片对象铺在多数页面上，或覆盖面积大且带透明通道的图。
  3. 重复铺版的 logo —— 同一图片在多页同一位置反复出现。

仅凭「重复」不足以判定：正文页眉、表头同样会重复。因此还要求命中片段
具备水印的形态特征（旋转 / 大字号 / 浅灰低对比 / **独立的**极短文本），
避免误删正文。「极短」之所以要加「独立」，见 `_watermark_like` 的说明。
"""
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pymupdf
import logging

logger = logging.getLogger(__name__)

# 判定阈值
MIN_PAGE_RATIO = 0.5        # 出现在半数以上页面
MIN_PAGES = 3               # 且至少出现在 3 页
POS_GRID = 15               # 位置归一化网格（磅），容忍轻微偏移
LARGE_IMAGE_RATIO = 0.35    # 单图覆盖页面面积超过该比例且带透明通道
MAX_WM_CHARS = 60           # 超过该长度的文本不视作水印（正文段落通常更长）
MIN_WM_CHARS = 2
MIN_IMAGE_BYTES = 200       # 纯色 logo 压缩后只有几百字节，阈值放低
SHORT_TEXT = 12             # 不超过该长度直接视为可疑水印


class WatermarkReport:
    """PDF 解析报告：去水印统计 + 走的哪条解析路径，供接口回显与排查。

    OCR / 表格的计数只在**兜底路径**下有意义（那是本项目的 PyMuPDF +
    PaddleOCR + pdfplumber 三件套在干活）；走 MinerU 主路径时它们保持 0，
    因为 OCR 与表格由 MinerU 内部完成，不在这一层统计。
    """

    def __init__(self):
        # 去水印
        self.text_removed = 0
        self.image_removed = 0
        self.text_samples: List[str] = []
        self.notes: List[str] = []
        # 解析路径：mineru（主）/ pymupdf（兜底）
        self.path = ""
        # MinerU 的统计（档位、字数、剥掉的内嵌图数）；走兜底路径时为 {}
        self.mineru: Dict = {}
        # 兜底路径下 OCR 的产出
        self.ocr_scanned = 0    # 实际跑了 OCR 的页数
        self.ocr_pages = 0      # 其中真正贡献了内容的页数
        self.ocr_lines = 0      # 补充的文本行数
        # 兜底路径下表格的产出
        self.table_pages = 0    # 含表格的页数
        self.table_count = 0    # 提取到的表格总数

    def to_dict(self):
        return {
            "text_removed": self.text_removed,
            "image_removed": self.image_removed,
            "text_samples": self.text_samples[:10],
            "path": self.path,
            "mineru": self.mineru,
            "ocr_scanned": self.ocr_scanned,
            "ocr_pages": self.ocr_pages,
            "ocr_lines": self.ocr_lines,
            "table_pages": self.table_pages,
            "table_count": self.table_count,
            "notes": self.notes,
        }

    def __str__(self):
        base = "文字水印 %d 处 / 图片水印 %d 处" % (self.text_removed,
                                                    self.image_removed)
        if self.path:
            base += " / 解析路径 %s" % self.path
        if self.mineru:
            base += " / %s 档 %d 字" % (self.mineru.get("tier", "?"),
                                        self.mineru.get("text_chars", 0))
            if self.mineru.get("images_dropped"):
                base += "（剥掉 %d 张内嵌图）" % self.mineru["images_dropped"]
        # 兜底路径才有这两项
        if self.ocr_lines:
            base += " / OCR 补充 %d 行（%d 页）" % (self.ocr_lines,
                                                    self.ocr_pages)
        if self.table_count:
            base += " / 表格 %d 个（%d 页）" % (self.table_count,
                                                self.table_pages)
        return base


class WatermarkRemover:

    def remove(self, doc, report: WatermarkReport) -> None:
        page_count = doc.page_count
        if page_count < MIN_PAGES:
            report.notes.append("页数过少，跳过水印统计")
            return

        text_hits, img_hits, samples = self._detect(doc, page_count)
        report.text_samples = samples

        for page_no, rects in text_hits.items():
            page = doc[page_no]
            for rect in rects:
                try:
                    page.add_redact_annot(rect)
                except Exception:                           # pragma: no cover
                    continue
            try:
                self._apply_redactions(page)
                report.text_removed += len(rects)
            except Exception as e:                          # pragma: no cover
                logger.warning("第 %s 页文字水印擦除失败: %s", page_no + 1, e)

        for page_no, rects in img_hits.items():
            page = doc[page_no]
            for rect in rects:
                try:
                    page.add_redact_annot(rect)
                except Exception:                           # pragma: no cover
                    continue
            try:
                self._apply_redactions(page, remove_images=True)
                report.image_removed += len(rects)
            except Exception as e:                          # pragma: no cover
                logger.warning("第 %s 页图片水印擦除失败: %s", page_no + 1, e)

        if report.text_removed or report.image_removed:
            logger.info("去水印完成: %s", report)

    @staticmethod
    def _apply_redactions(page, remove_images: bool = False) -> None:
        """兼容不同 PyMuPDF 版本的 apply_redactions 参数。"""
        kwargs = {}
        try:
            if remove_images:
                kwargs["images"] = pymupdf.PDF_REDACT_IMAGE_REMOVE
            kwargs["graphics"] = pymupdf.PDF_REDACT_LINE_ART_NONE
            page.apply_redactions(**kwargs)
        except TypeError:                                   # 老版本
            page.apply_redactions()

    def _detect(self, doc, page_count: int
                ) -> Tuple[Dict[int, list], Dict[int, list], List[str]]:
        """统计全篇，返回 ({页号: [rect]}, {页号: [rect]}, 水印文本样本)。"""
        text_seen: Dict[tuple, List[Tuple[int, tuple]]] = defaultdict(list)
        img_seen: Dict[object, List[Tuple[int, tuple]]] = defaultdict(list)
        img_xref_page = defaultdict(set)
        img_area = {}

        for pno in range(page_count):
            page = doc[pno]
            pw, ph = page.rect.width or 1, page.rect.height or 1

            try:
                d = page.get_text("dict")
            except Exception:                               # pragma: no cover
                d = {}
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                lines = block.get("lines", [])
                # 整块只有一行 = 独立文本对象；判定「极短」时要靠它区分
                # 真水印与正文折行的续行，见 _watermark_like
                standalone = len(lines) == 1
                for line in lines:
                    for span in line.get("spans", []):
                        txt = (span.get("text") or "").strip()
                        if not (MIN_WM_CHARS <= len(txt) <= MAX_WM_CHARS):
                            continue
                        if not self._watermark_like(span, line, standalone):
                            continue
                        bbox = span.get("bbox") or line.get("bbox")
                        if not bbox:
                            continue
                        text_seen[(txt, self._grid(bbox))].append(
                            (pno, pymupdf.Rect(bbox)))

            try:
                infos = page.get_image_info(xrefs=True)
            except Exception:                               # pragma: no cover
                infos = []
            for info in infos:
                bbox = info.get("bbox")
                if not bbox:
                    continue
                rect = pymupdf.Rect(bbox)
                area_ratio = (rect.width * rect.height) / (pw * ph)
                xref = info.get("xref") or 0
                if xref:
                    img_xref_page[xref].add(pno)
                img_area[xref] = max(img_area.get(xref, 0), area_ratio)

                if area_ratio >= LARGE_IMAGE_RATIO and self._has_alpha(doc, xref):
                    img_seen[("alpha", xref)].append((pno, rect))
                else:
                    digest = self._image_digest(doc, xref)
                    if digest:
                        img_seen[("hash", digest)].append((pno, rect))

        threshold = max(MIN_PAGES, int(page_count * MIN_PAGE_RATIO))

        text_hits = defaultdict(list)
        samples = []
        for (txt, _), occurrences in text_seen.items():
            if len({p for p, _ in occurrences}) >= threshold:
                samples.append(txt)
                for pno, rect in occurrences:
                    text_hits[pno].append(rect)

        img_hits = defaultdict(list)
        for key, occurrences in img_seen.items():
            if len({p for p, _ in occurrences}) >= threshold:
                for pno, rect in occurrences:
                    img_hits[pno].append(rect)
        for xref, pages_set in img_xref_page.items():
            if xref and len(pages_set) >= threshold and img_area.get(xref, 0) > 0.05:
                for pno in pages_set:
                    for info in doc[pno].get_image_info(xrefs=True):
                        if info.get("xref") == xref and info.get("bbox"):
                            img_hits[pno].append(pymupdf.Rect(info["bbox"]))

        return dict(text_hits), dict(img_hits), sorted(set(samples))

    @staticmethod
    def _watermark_like(span, line, standalone: bool) -> bool:
        """判断文本片段是否具备水印的形态特征。

        水平、黑色、小字号的正文会被排除；只认这几种：
        旋转（斜向水印）、大字号、浅灰低对比，以及**独立的**极短文本（如「机密」）。

        「极短」必须配合 `standalone` 一起看——长度本身说明不了什么，
        正文折行的续行、表格里的短单元格同样很短。实测一段 11 字的正文续行
        在 12 页上重复出现，被当成水印整段删除，与 §6.3 记的失败同类。
        真正的水印是单独画上去的文本对象，在 `get_text("dict")` 里自成一个块。
        """
        txt = (span.get("text") or "").strip()
        if len(txt) <= SHORT_TEXT:
            return standalone
        direction = line.get("dir") or (1, 0)
        if abs(direction[0] - 1) > 0.01 or abs(direction[1]) > 0.01:
            return True                                    # 旋转文字
        if float(span.get("size") or 0) >= 20:
            return True                                    # 大字号
        color = span.get("color")
        if isinstance(color, int):
            r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
            if min(r, g, b) > 140 and (max(r, g, b) - min(r, g, b)) < 40:
                return True                                # 浅灰低对比
        return False

    @staticmethod
    def _grid(bbox) -> tuple:
        """把坐标量化到网格，容忍水印在各页的轻微偏移。"""
        x0, y0, x1, y1 = bbox[:4]
        g = POS_GRID
        return (round(x0 / g), round(y0 / g), round(x1 / g), round(y1 / g))

    @staticmethod
    def _has_alpha(doc, xref: int) -> bool:
        if not xref:
            return False
        try:
            img = doc.extract_image(xref)
            return bool(img.get("smask")) or img.get("colorspace", 0) in (4, 5)
        except Exception:                                   # pragma: no cover
            return False

    @staticmethod
    def _image_digest(doc, xref: int) -> Optional[str]:
        """按像素数据给图片取指纹，用于识别多页重复铺版的 logo。"""
        if not xref:
            return None
        try:
            img = doc.extract_image(xref)
            data = img.get("image")
            if not data or len(data) < MIN_IMAGE_BYTES:
                return None
            return hashlib.md5(data).hexdigest()
        except Exception:                                   # pragma: no cover
            return None
