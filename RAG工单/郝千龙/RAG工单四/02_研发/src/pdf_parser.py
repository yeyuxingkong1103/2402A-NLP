# -*- coding: utf-8 -*-
# 【PDF解析模块 · pdf_parser.py】双引擎文本/表格解析，并输出供图像层复用的页面清单
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""PDF解析层（在工单二成熟实现上扩展为多文档、并暴露页面级几何信息）：

1. PdfInventory：一次性解析每页文本行（带bbox）、绘图指令、图片引用，
   文本解析与图像抽取共享同一份清单，避免重复打开PDF；
2. parse_inventory：按文本块输出正文/表格Block（页眉清洗、硬断行合并、
   标题路径维护、pdfplumber表格回退），每个Block带文档标签与页码。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pymupdf

try:
    import pdfplumber
except Exception:  # pragma: no cover - pdfplumber缺失时仅禁用表格回退
    pdfplumber = None

from config import CONFIG


class PDFParseError(Exception):
    """文档解析失败时抛出的自定义异常（损坏/加密PDF等）。"""


@dataclass
class Block:
    """结构化文档块：一段正文或一张表格。"""

    type: str                  # text / table
    text: str
    page_no: int               # 1起始物理页码
    doc: str = ""              # 文档标签（招股说明书1/招股说明书2）
    heading_path: str = ""
    table_title: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Line:
    """页面文本行（保留坐标，供图注/邻近正文关联）。"""

    text: str
    bbox: Tuple[float, float, float, float]


# 中文终止符：行尾出现这些符号时不做跨行拼接
_END_PUNCT = tuple("。！？；：”’）)】」.!?;")


def clean_line(line: str) -> str:
    """清洗单行文本：行内剔除页眉页脚标记、水印与噪声词。

    :param line: 原始文本行
    :return: 清洗后的文本行（可能为空串）
    """
    text = line.replace("\u3000", " ").replace("\xa0", " ").strip()
    text = re.sub(r"[\u200b\ufeff\u2000-\u200a\u202f\u205f]", " ", text)
    for pattern in CONFIG.header_footer_patterns:
        text = re.sub(pattern, "", text)
    for word in CONFIG.noise_words:
        text = text.replace(word, "")
    text = re.sub(r"^[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司)\s*", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def is_heading(text: str) -> bool:
    """判断文本行是否为章节标题。

    :param text: 待判断文本行
    :return: 命中标题模式且长度不超过40字返回True
    """
    return bool(re.match(CONFIG.heading_pattern, text)) and len(text) <= 40


def looks_tabular(page_text: str) -> bool:
    """启发式判断页面是否含强表格特征：出现“单位：”且含较多数字簇。

    :param page_text: 该页PyMuPDF提取的全文
    :return: 命中表格特征返回True
    """
    if "单位：" not in page_text:
        return False
    return len(re.findall(r"\d[\d,\.]{2,}", page_text)) >= 8


class PdfInventory:
    """单个PDF的页面级清单：文本行（带坐标）、绘图指令、图片引用与文档句柄。"""

    def __init__(self, path: str, doc_tag: str) -> None:
        """打开PDF并逐页构建轻量清单。

        :param path: PDF绝对路径
        :param doc_tag: 文档标签
        """
        self.path = path
        self.doc_tag = doc_tag
        try:
            self.doc = pymupdf.open(path)
        except Exception as exc:
            raise PDFParseError(f"无法打开PDF文件: {path}，原因: {exc}") from exc
        self.pages_lines: List[List[Line]] = []
        self.pages_drawings: List[list] = []
        self.pages_images: List[list] = []
        for page in self.doc:
            lines: List[Line] = []
            page_h = page.rect.height
            data = page.get_text("dict")
            for item in data.get("blocks", []):
                if item.get("type") != 0:
                    continue
                line_text = "".join(
                    "".join(span.get("text", "") for span in ln.get("spans", []))
                    for ln in item.get("lines", [])
                )
                x0, y0, x1, y1 = item.get("bbox", (0, 0, 0, 0))
                # 页眉页脚坐标区（顶部/底部5%内）短文本直接剔除
                in_margin = (y1 < page_h * 0.06) or (y0 > page_h * 0.95)
                if in_margin and len(line_text.strip()) < 40:
                    continue
                cleaned = clean_line(line_text)
                if cleaned:
                    lines.append(Line(cleaned, (x0, y0, x1, y1)))
            self.pages_lines.append(lines)
            self.pages_drawings.append(page.get_drawings())
            self.pages_images.append(page.get_images(full=True))

    @property
    def page_count(self) -> int:
        """返回PDF总页数。"""
        return len(self.doc)

    def page(self, index: int):
        """按0基下标返回PyMuPDF页面对象。

        :param index: 0基页码
        """
        return self.doc[index]

    def close(self) -> None:
        """关闭文档句柄。"""
        self.doc.close()

    def __enter__(self) -> "PdfInventory":
        """进入with上下文，返回清单自身。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出with上下文时自动关闭PDF句柄。"""
        self.close()


def _extract_tables(plumber_page, page_no: int, doc_tag: str) -> List[Block]:
    """用pdfplumber提取当前页全部表格并序列化为Markdown。

    :param plumber_page: pdfplumber页面对象
    :param page_no: 1起始页码
    :param doc_tag: 文档标签
    :return: 表格Block列表
    """
    blocks: List[Block] = []
    for table in plumber_page.extract_tables() or []:
        rows = [[(cell or "").replace("\n", " ").strip() for cell in row]
                for row in table if row]
        if len(rows) < 2:
            continue
        md_lines = ["| " + " | ".join(rows[0]) + " |",
                    "| " + " | ".join(["---"] * len(rows[0])) + " |"]
        md_lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        blocks.append(Block(type="table", text="\n".join(md_lines),
                            page_no=page_no, doc=doc_tag))
    return blocks


def parse_inventory(inv: PdfInventory) -> List[Block]:
    """把页面清单解析为有序正文/表格Block列表。

    :param inv: 已构建的PdfInventory
    :return: Block列表（携带页码、文档标签、标题路径）
    """
    plumber = None
    if pdfplumber is not None:
        try:
            plumber = pdfplumber.open(inv.path)
        except Exception:
            plumber = None  # 表格回退不可用时不影响正文解析

    blocks: List[Block] = []
    heading_stack: List[str] = []
    try:
        for index in range(inv.page_count):
            page_no = index + 1
            page_lines = list(inv.pages_lines[index])
            # 硬断行合并：中文段落被PDF换行切断，行尾非终止符则拼接
            merged: List[str] = []
            buf = ""
            for line in page_lines:
                text = line.text
                if is_heading(text):
                    if buf:
                        merged.append(buf)
                        buf = ""
                    merged.append(text)
                    if text.startswith("第") and ("章" in text or "节" in text):
                        heading_stack = [text]
                    elif re.match(r"^[一二三四五六七八九十]+、", text):
                        heading_stack = heading_stack[:1] + [text]
                elif buf and not buf.endswith(_END_PUNCT):
                    buf += text
                else:
                    if buf:
                        merged.append(buf)
                    buf = text
            if buf:
                merged.append(buf)

            page_text = "".join(merged)
            current_heading = " > ".join(heading_stack)

            table_blocks: List[Block] = []
            if plumber is not None:
                need_table = (len(page_text) < CONFIG.table_fallback_chars
                              or looks_tabular(page_text))
                if need_table:
                    table_blocks = _extract_tables(
                        plumber.pages[index], page_no, inv.doc_tag)
            table_chars = sum(len(b.text) for b in table_blocks)
            if table_blocks and table_chars > len(page_text) * 0.6:
                blocks.extend(table_blocks)
                continue

            for seg in merged:
                if is_heading(seg):
                    continue
                blocks.append(Block(type="text", text=seg, page_no=page_no,
                                    doc=inv.doc_tag, heading_path=current_heading))
            blocks.extend(table_blocks)
    except Exception as exc:
        raise PDFParseError(f"解析PDF失败: {inv.path}，原因: {exc}") from exc
    finally:
        if plumber is not None:
            plumber.close()
    return blocks


def parse_pdf(path: str, doc_tag: str) -> Tuple[PdfInventory, List[Block]]:
    """一步完成“打开清单→解析Block”（解析层对外便捷入口）。

    :param path: PDF绝对路径
    :param doc_tag: 文档标签
    :return: (清单对象, Block列表)，调用方负责在用完后close清单
    """
    inv = PdfInventory(path, doc_tag)
    return inv, parse_inventory(inv)
