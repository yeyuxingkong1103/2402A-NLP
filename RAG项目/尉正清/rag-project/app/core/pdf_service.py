# app/core/pdf_service.py
"""PDF 解析编排：**总体用 MinerU，兜底用自研三件套**。

## 两条路径

    主路径（默认）  PyMuPDF 擦水印 → 另存干净 PDF → MinerU
                    版面 / OCR / 公式 / 表格一次解析成整篇 Markdown

    兜底路径        MinerU 不可用或解析失败 →
                    PyMuPDF 取文本层 + PaddleOCR 补图表文字 + pdfplumber 补表格

兜底路径是刻意保留的，不是历史残留：

  · MinerU 是独立 venv 里的重依赖（venv 7.4GB + 模型 1.4GB）。开发环境不装它
    也要能把项目跑起来 —— 没有兜底就寸步难行。
  · 上传时遇到 MinerU 出问题（模型损坏、子进程超时、显存不足被拒），
    有兜底至少能拿到文本层，而不是让接口直接 500。
  · 兜底的能力弱于 MinerU（阅读顺序要自己拼、公式拿不到），但**能用**。

## 为什么两条路径都必须先去水印

MinerU 只做解析，不做水印擦除；PyMuPDF 文本层同理，水印会被当成正文读进来。
所以水印在两条路径之前统一擦掉。

## 输出契约

`extract_pages()` 返回 `([{page, text}], 报告)`。

  · 主路径：MinerU 产出的是**整篇 Markdown**，返回单元素列表
    `[{page: 1, text: Markdown}]`，由 `chunk_service.semantic_split` 按语义切块。
    这比按页码拆更好 —— MinerU 的价值就是连贯的阅读顺序。
  · 兜底路径：沿袭逐页返回，因为 PyMuPDF 本来就是逐页取文本。

## 归一化只对兜底路径做

`_normalize()` 做的是「去空行 + 合并硬换行」，那是给 PyMuPDF 纯文本层写的。
套在 Markdown 上会毁掉标题层级、列表与表格结构，所以主路径的 Markdown
**原样直通**。
"""
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import pymupdf

from app.core import mineru_service
from app.core.ocr_service import get_ocr_service, merge_ocr_text
from app.core.table_service import get_table_service
from app.core.watermark import WatermarkRemover, WatermarkReport
import logging

logger = logging.getLogger(__name__)


class PDFService:

    def __init__(self):
        self.remover = WatermarkRemover()

    # ---------------- 对外入口 ----------------
    def extract_pages(self, file_path: str, remove_watermark: bool = True,
                      clean_metadata: bool = True, use_ocr: bool = True,
                      use_tables: bool = True
                      ) -> Tuple[List[Dict], WatermarkReport]:
        """解析 PDF，返回 ([{page, text}], 解析报告)。

        `use_ocr` / `use_tables` **只作用于兜底路径**：走 MinerU 时它自带这两项
        能力、无法单独关闭，两个开关会被忽略（取值记进报告的 notes）。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("文件不存在: %s" % file_path)

        report = WatermarkReport()

        # 1) 先去水印，另存一份干净 PDF（MinerU 拿不到内存里的 doc 对象）
        with tempfile.TemporaryDirectory(prefix="pdfclean_") as tmp:
            clean_path = os.path.join(tmp, "clean.pdf")
            try:
                self._clean_copy(file_path, clean_path, remove_watermark,
                                 clean_metadata, report)
            except ValueError:
                raise                       # 加密等致命问题，直接上抛
            except Exception as e:
                logger.warning("预处理失败，改用原文件: %s", e)
                clean_path = file_path

            # 2) 主路径：MinerU
            if mineru_service.available():
                try:
                    text, stat = mineru_service.parse(clean_path)
                    if not text:
                        raise RuntimeError("MinerU 返回空文本")
                    report.mineru = stat
                    report.path = "mineru"
                    if not (use_ocr and use_tables):
                        report.notes.append(
                            "MinerU 自带 OCR 与表格识别，use_ocr/use_tables 在主路径下不生效")
                    return [{"page": 1, "text": text}], report
                except Exception as e:
                    logger.warning("MinerU 解析失败，转兜底三件套: %s", e)
                    report.notes.append("MinerU 解析失败，已转兜底路径：%s" % e)
            else:
                report.notes.append("MinerU 未安装，走兜底三件套")

            # 3) 兜底路径：PyMuPDF + PaddleOCR + pdfplumber
            #
            # 两个「必须」：
            #   · 必须读 clean_path 而非 file_path —— 水印是擦在 clean_path 上的，
            #     读原文件会让去水印整步白做，而报告的 text_removed 照样计数，
            #     表现为「报告说删了、正文里还在」。
            #   · 必须留在 with 内调用 —— 临时目录一退出，clean_path 就被删了。
            return self._extract_fallback(clean_path, report, use_ocr, use_tables)

    # ---------------- 主路径辅助 ----------------
    @staticmethod
    def _clean_copy(src: str, dst: str, remove_watermark: bool,
                    clean_metadata: bool, report: WatermarkReport) -> None:
        """擦水印 + 清元数据后另存，供 MinerU 使用。"""
        doc = pymupdf.open(src)
        try:
            if doc.needs_pass:
                raise ValueError("PDF 已加密，请先解除密码保护")
            if remove_watermark:
                PDFService._remove_watermarks(doc, report)
            if clean_metadata:
                PDFService._clean_metadata(doc, report)
            doc.save(dst, garbage=3, deflate=True)
        finally:
            doc.close()

    @staticmethod
    def _remove_watermarks(doc, report: WatermarkReport) -> None:
        from app.core.watermark import WatermarkRemover
        WatermarkRemover().remove(doc, report)

    # ---------------- 兜底路径 ----------------
    def _extract_fallback(self, pdf_path: str, report: WatermarkReport,
                          use_ocr: bool, use_tables: bool
                          ) -> Tuple[List[Dict], WatermarkReport]:
        """PyMuPDF 取文本层，PaddleOCR 补图表文字，pdfplumber 补表格。

        `pdf_path` 必须是**去过水印的干净副本**，不是调用方传进来的原文件。
        """
        report.path = "pymupdf"

        # OCR 模型不可用时静默退回纯文本层
        ocr = get_ocr_service() if use_ocr else None
        if ocr is not None and not ocr.available:
            report.notes.append("OCR 不可用，仅提取文本层")
            ocr = None

        # 表格提取用 pdfplumber，与 PyMuPDF 是两个库，按文件路径调用
        tables = get_table_service() if use_tables else None
        if tables is not None and not tables.available():
            report.notes.append("pdfplumber 不可用，跳过表格提取")
            tables = None

        pages: List[Dict] = []
        doc = pymupdf.open(pdf_path)
        try:
            if doc.needs_pass:
                raise ValueError("PDF 已加密，请先解除密码保护")
            # 全篇表格一次性提取：pdfplumber 每次 open 都要解析整份文件，
            # 逐页调用等于把同一份 PDF 重复打开 N 次
            doc_tables = (tables.extract_document_tables(pdf_path, doc.page_count)
                          if tables is not None else {})

            for i, page in enumerate(doc):
                text = self._normalize(page.get_text("text") or "")

                if ocr is not None and self._needs_ocr(page, len(text)):
                    report.ocr_scanned += 1
                    lines = ocr.recognize_pdf_page(page)
                    text, added = merge_ocr_text(text, lines)
                    if added:
                        report.ocr_pages += 1
                        report.ocr_lines += added

                found = doc_tables.get(i + 1)
                if found:
                    text, added = tables.merge_into_text(text, found)
                    if added:
                        report.table_pages += 1
                        report.table_count += added

                if text:
                    pages.append({"page": i + 1, "text": text})
        finally:
            doc.close()

        if report.ocr_lines:
            logger.info("兜底路径：OCR 补充 %d 行图表文字（%d 页）",
                        report.ocr_lines, report.ocr_pages)
        return pages, report

    @staticmethod
    def _needs_ocr(page, text_len: int) -> bool:
        """判断这一页值不值得跑 OCR。

        OCR 很贵（CPU 上约 10~40 秒/页），无差别跑会把一份 30 页的 PDF
        拖到十分钟。实测只有「文字少 + 图大」的页才有增量，
        封面那种整页背景图跑出来全是噪声，反而污染正文。

        条件：
          · 文本层几乎没有 + 整页是图  → 可能是扫描页，必跑
          · 图片占页面 15%~95%        → 图表页，跑（铺满全页的背景图排除在外）
        """
        infos = page.get_image_info(xrefs=True)
        if not infos:
            return False

        pw = page.rect.width or 1
        ph = page.rect.height or 1
        page_area = pw * ph
        biggest = 0.0
        for info in infos:
            b = info.get("bbox")
            if not b:
                continue
            ratio = ((b[2] - b[0]) * (b[3] - b[1])) / page_area
            biggest = max(biggest, ratio)
        if biggest <= 0:
            return False

        # 扫描页：文本层几乎为空但整页是图
        if text_len < 50 and biggest > 0.5:
            return True
        # 图表页：有实质图片，但不是铺满整页的背景
        return 0.15 <= biggest <= 0.95

    # ---------------- 工具 ----------------
    @staticmethod
    def _clean_metadata(doc, report: WatermarkReport) -> None:
        """清掉生产者、作者等附带信息。"""
        try:
            doc.set_metadata({
                "title": "", "author": "", "subject": "",
                "keywords": "", "creator": "", "producer": "",
            })
        except Exception:                                   # pragma: no cover
            report.notes.append("元数据清除失败")

    @staticmethod
    def _normalize(text: str) -> str:
        """去掉空行，并把版面硬换行合并回整段。

        只用于兜底路径 —— Markdown 走这条路会毁掉标题与表格结构。

        中文行直接相接；英文行若首尾都是字母数字，则补一个空格，
        否则会把单词粘成 "Thisworkhasbeen"。
        """
        lines = [raw.strip() for raw in text.split("\n") if raw.strip()]
        out = ""
        for line in lines:
            if (out and out[-1].isascii() and out[-1].isalnum()
                    and line[0].isascii() and line[0].isalnum()):
                out += " "
            out += line
        return out.strip()


_pdf_service: Optional[PDFService] = None


def get_pdf_service() -> PDFService:
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service
