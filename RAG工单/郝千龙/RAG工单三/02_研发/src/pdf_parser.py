# -*- coding: utf-8 -*-
# 【PDF解析模块 · pdf_parser.py】PyMuPDF正文解析 + pdfplumber表格抽取 + 表格区正文去重 + 标题结构识别
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""PDF 解析层：输出带标题路径、页码与公司主体的结构化 Block。

解析策略（对应设计文档第 3 章）：
1. PyMuPDF 提取正文文本块，清洗页眉页脚、拼接硬断行、维护章节标题栈；
2. 每页调用 pdfplumber 做表格识别，经 table_parser 清洗还原为 Markdown 表；
3. 以“二元组覆盖率”删除已被表格单元格承载的正文行（不做粗暴的矩形覆盖，
   避免错版卡式表丢失“注册资本/法定代表人”等键值行）；
4. 正文块与表格块均携带 1 起始页码、标题路径与发行人主体。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pymupdf

try:
    import pdfplumber
except Exception:  # pdfplumber 缺失时仅禁用表格解析
    pdfplumber = None

from config import CONFIG
from table_parser import ParsedTable, parse_page_tables

# 中文终止符：行尾出现这些符号时不做跨行拼接
_END_PUNCT = tuple("。！？；：”’）)】」.!?;")


class PDFParseError(Exception):
    """文档解析失败时抛出的自定义异常（损坏/加密 PDF 等）。"""


@dataclass
class Block:
    """结构化文档块：一段正文或一张表格。"""

    type: str               # text / table
    text: str               # 正文清洗文本或表格 Markdown
    page_no: int            # 1 起始页码
    heading_path: str = ""  # 所属标题路径
    company: str = ""       # 发行人主体全称
    table_title: str = ""   # 表格标题（仅 type=table）
    meta: dict = field(default_factory=dict)


def clean_line(line: str) -> str:
    """清洗单行文本：行内剔除页眉页脚标记与噪声词（不丢弃整行正文）。

    :param line: 原始文本行
    :return: 清洗后的文本行（可能为空串）
    """
    text = line.replace(" ", " ").replace("　", " ").replace("\xa0", " ")
    text = re.sub(r"[​﻿ -   ]", " ", text)
    for pattern in CONFIG.header_footer_patterns:
        text = re.sub(pattern, "", text)
    for word in CONFIG.noise_words:
        text = text.replace(word, "")
    text = re.sub(r"^[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司)\s*",
                  "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def is_heading(text: str) -> bool:
    """判断文本行是否为章节标题。

    :param text: 待判断文本行
    :return: 命中标题模式且长度不超过 40 字时返回 True
    """
    return bool(re.match(CONFIG.heading_pattern, text)) and len(text) <= 40


def _bigrams(text: str) -> set:
    """取文本去除空白标点后的相邻二元组集合（用于覆盖率去重）。

    :param text: 文本
    :return: 二元组集合
    """
    chars = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


def _line_covered_by_tables(line: str, table_bigrams: set) -> bool:
    """判断正文行是否已被本页表格单元格语料覆盖。

    :param line: 正文行
    :param table_bigrams: 本页表格单元格全部二元组集合
    :return: 覆盖率达到阈值返回 True
    """
    if len(line) < CONFIG.text_cover_min_len:
        return False
    grams = _bigrams(line)
    if not grams:
        return False
    hit = sum(1 for g in grams if g in table_bigrams)
    return hit / len(grams) >= CONFIG.text_cover_ratio


def _merge_lines(page_lines: List[str]) -> List[str]:
    """硬断行合并：中文段落被 PDF 换行切断，行尾非终止符则拼接。

    :param page_lines: 清洗后的文本行
    :return: 合并后的段落列表
    """
    merged: List[str] = []
    buf = ""
    for line in page_lines:
        if is_heading(line):
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(line)
        elif buf and not buf.endswith(_END_PUNCT):
            buf += line
        else:
            if buf:
                merged.append(buf)
            buf = line
    if buf:
        merged.append(buf)
    return merged


def _table_bigrams(tables: List[ParsedTable]) -> set:
    """汇总本页全部表格单元格文本的二元组集合。

    :param tables: 本页解析出的表格
    :return: 二元组集合
    """
    grams: set = set()
    for tb in tables:
        for row in tb.rows:
            for cell in row:
                grams |= _bigrams(cell)
    return grams


def parse_pdf(path: str, company: str,
              enable_tables: bool = True) -> List[Block]:
    """解析 PDF 为结构化 Block 列表（解析层唯一对外入口）。

    :param path: PDF 文件绝对路径
    :param company: 该 PDF 对应的发行人主体全称
    :param enable_tables: 是否启用 pdfplumber 表格解析（基线评测可关闭）
    :return: 有序 Block 列表（正文/表格）
    :raises PDFParseError: 文件无法打开或解析失败时抛出
    """
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PDFParseError(f"无法打开PDF文件: {path}，原因: {exc}") from exc

    plumber = None
    if enable_tables and pdfplumber is not None and \
            CONFIG.parse_tables_every_page:
        try:
            plumber = pdfplumber.open(path)
        except Exception:
            plumber = None

    blocks: List[Block] = []
    heading_stack: List[str] = []
    try:
        for index, page in enumerate(doc):
            page_no = index + 1
            raw_dict = page.get_text("dict")
            page_h = page.rect.height
            page_lines: List[str] = []
            page_headings: List[str] = []
            for item in raw_dict.get("blocks", []):
                if item.get("type") != 0:
                    continue
                line_text = "".join(
                    "".join(span.get("text", "")
                            for span in line.get("spans", []))
                    for line in item.get("lines", []))
                x0, y0, _x1, y1 = item.get("bbox", (0, 0, 0, 0))
                # 页眉页脚坐标区短文本直接跳过
                if (y1 < page_h * 0.06 or y0 > page_h * 0.95) and \
                        len(line_text.strip()) < 40:
                    continue
                cleaned = clean_line(line_text)
                if not cleaned:
                    continue
                if is_heading(cleaned):
                    if cleaned.startswith("第") and (
                            "章" in cleaned or "节" in cleaned):
                        heading_stack = [cleaned]
                    elif re.match(r"^[一二三四五六七八九十]+、", cleaned):
                        heading_stack = heading_stack[:1] + [cleaned]
                    page_headings.append(cleaned)
                page_lines.append(cleaned)

            # 表格解析（失败不影响正文）
            tables: List[ParsedTable] = []
            if plumber is not None:
                try:
                    tables = parse_page_tables(plumber.pages[index], page_no)
                except Exception:
                    tables = []
            table_grams = _table_bigrams(tables)

            # 表格覆盖行去重：标题行保留，被表格单元格承载的正文行剔除
            if table_grams:
                page_lines = [
                    ln for ln in page_lines
                    if is_heading(ln) or not _line_covered_by_tables(
                        ln, table_grams)]

            current_heading = " > ".join(heading_stack)
            for seg in _merge_lines(page_lines):
                if is_heading(seg):
                    continue
                blocks.append(Block(
                    type="text", text=seg, page_no=page_no,
                    heading_path=current_heading, company=company))

            for tb in tables:
                table_text = ((tb.caption + "\n") if tb.caption else "") + \
                             tb.markdown
                blocks.append(Block(
                    type="table", text=table_text, page_no=page_no,
                    heading_path=current_heading, company=company,
                    table_title=tb.caption,
                    meta={"headers": tb.headers, "rows": tb.rows,
                          "claims": tb.claims, "caption": tb.caption}))
    except PDFParseError:
        raise
    except Exception as exc:
        raise PDFParseError(f"解析PDF失败: {path}，原因: {exc}") from exc
    finally:
        doc.close()
        if plumber is not None:
            plumber.close()
    return blocks
