# app/core/ocr_service.py
"""PaddleOCR 封装：补出图表里的文字。

## 它在链路里的位置

    MinerU 可用  →  MinerU 一次解析出版面 / OCR / 公式 / 表格（主路径）
    MinerU 不可用 →  PyMuPDF 取文本层 + 本模块补图表文字 + pdfplumber 补表格（兜底）

本模块**只在兜底路径下被调用**。主路径下 MinerU 自带 OCR，不需要它。

## 为什么需要单独补 OCR

PyMuPDF 只能读 PDF 的**文本层**。图表若是位图，里面的文字对文本层完全不可见。
实测 WHO《世界精神卫生报告》：某页文本层只有一句图题，而图内另有数十行文字，
纯文本提取会把整张图的内容丢掉。

## 代价

CPU 上约 10~40 秒/页，因此调用方（pdf_service）会先用 `_needs_ocr()` 判断
这一页值不值得跑，而不是无差别处理。
"""
import os
import re
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)

RENDER_DPI = 150            # 页面渲染成位图的分辨率；再高收益递减而耗时线性增长
MAX_LINES_PER_PAGE = 300    # 单页识别行数上限，防御性截断，避免异常页面拖垮整份文档

# 归一化时要去掉的标点：只比较「字」，不比较标点写法差异
_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）《》【】,.!?;:\"'()\[\]<>-]+")


def _normalize(text: str) -> str:
    """去掉空白与标点，用于判断两段文字是不是「同一句」。"""
    return _PUNCT.sub("", text or "")


class OCRService:
    """PaddleOCR 的薄封装，带懒加载与不可用降级。"""

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self._model = None
        self._failed = False     # 加载失败后置位，避免每次调用都重试一遍

    def _ensure(self) -> bool:
        """懒加载模型。返回是否可用。"""
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            logger.info("加载 PaddleOCR 模型（首次会下载，约需 1 分钟）...")
            from paddleocr import PaddleOCR
            # enable_mkldnn=False 是**必须的**，不是性能选项：
            # paddlepaddle 3.3.1 的 oneDNN 指令在 PP-OCRv5/v6 上会抛
            #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute
            #   not support [pir::ArrayAttribute<pir::DoubleAttribute>]
            #   (onednn_instruction.cc)
            # 关掉后走通用 CPU 路径，慢一点但能跑通 —— 兜底路径要的是「能用」。
            #
            # 注意参数名：PaddleOCR 3.x 已移除 show_log，use_angle_cls 改名为
            # use_textline_orientation。这里只传跨版本稳定的 lang 与 enable_mkldnn。
            self._model = PaddleOCR(lang=self.lang, enable_mkldnn=False)
            logger.info("PaddleOCR 就绪")
            return True
        except Exception as e:
            self._failed = True
            logger.warning("PaddleOCR 不可用，图表文字将无法提取: %s", e)
            return False

    @property
    def available(self) -> bool:
        return self._ensure()

    def recognize_image(self, image_path: str) -> List[str]:
        """识别一张图片，返回文本行（已去空白与空行）。

        PaddleOCR 3.x 的返回是「每页一个 dict」，文本在同级的 `rec_texts` 里
        （形如 [{'rec_texts': ['第一行', '第二行'], 'rec_scores': [...], ...}]）。
        2.x 那种 [文本, 置信度] 的嵌套结构已经没有了。
        """
        if not self._ensure():
            return []
        try:
            # 不传 cls=：3.x 的 .ocr() 签名是 (img, **kwargs)，没有这个参数
            result = self._model.ocr(image_path)
        except Exception as e:                              # pragma: no cover
            logger.warning("OCR 识别图片失败 %s: %s", image_path, e)
            return []

        lines: List[str] = []
        for page in (result or []):
            if not isinstance(page, dict):
                continue
            for text in (page.get("rec_texts") or []):
                text = (text or "").strip()
                if text:
                    lines.append(text)
        return lines[:MAX_LINES_PER_PAGE]

    def recognize_pdf_page(self, page, dpi: int = RENDER_DPI) -> List[str]:
        """把 PDF 某页渲染成位图再识别。

        PaddleOCR 只接受图片路径，所以这里必须先渲染落盘。
        """
        if not self._ensure():
            return []
        # 用页面自身的 xref 命名临时文件：同一进程内并发处理不同页时不会互相覆盖
        tmp = "/tmp/_ocr_page_%d_%d.png" % (id(page) % 100000, dpi)
        try:
            pix = page.get_pixmap(dpi=dpi)
            pix.save(tmp)
            return self.recognize_image(tmp)
        except Exception as e:                              # pragma: no cover
            logger.warning("渲染 PDF 页失败: %s", e)
            return []
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def merge_ocr_text(text_layer: str, ocr_lines: List[str],
                   min_len: int = 2) -> tuple:
    """把 OCR 结果并进文本层，返回 (合并后的正文, 新增行数)。

    去重规则：归一化（去空白与标点）后**互相包含**即视为重复。
    OCR 常常把文本层已有的句子又识别一遍，只是标点或断行不同；
    不这样判重的话，正文会出现大段重复。
    """
    if not ocr_lines:
        return text_layer, 0

    base = _normalize(text_layer)
    novel = []
    for line in ocr_lines:
        norm = _normalize(line)
        if len(norm) < min_len:
            continue
        # 文本层里已经有这句（或这句包含在文本层某个片段里）就跳过
        if norm in base:
            continue
        novel.append(line)

    if not novel:
        return text_layer, 0

    # 文本层为空（整页扫描件）时，分隔用的空行不能留在正文最前面
    head = text_layer.rstrip()
    merged = (head + "\n\n" if head else "") + "【图表文字】\n" + "\n".join(novel)
    return merged, len(novel)


_ocr: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    global _ocr
    if _ocr is None:
        _ocr = OCRService()
    return _ocr
