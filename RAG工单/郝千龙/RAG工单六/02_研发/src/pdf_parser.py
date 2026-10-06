# -*- coding: utf-8 -*-
# 【PDF解析模块 · pdf_parser.py】双引擎解析两份招股书、页眉清洗、表格回退、标题结构识别、来源文档标注
# 工单编号：人工智能NLP-RAG-混合检索任务

"""PDF 解析层：输出带标题路径、页码与来源文档名的结构化 Block。

解析策略：
1. PyMuPDF 按文本块提取正文，识别章节标题并拼接硬断行；
2. 对文本稀疏页/强表格特征页回退 pdfplumber 提取表格，序列化为 Markdown；
3. 清洗两份招股书各自的页眉页脚、纯页码、硬回车；
4. 每个 Block 携带 ``doc_name``（语料文件名），支撑多文档公司实体消歧。
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pymupdf

try:
    import pdfplumber
except Exception:  # pdfplumber 缺失时仅禁用表格回退，不影响正文解析
    pdfplumber = None

from config import CONFIG


class PDFParseError(Exception):
    """文档解析失败时抛出的自定义异常（损坏/加密PDF等）。"""


@dataclass
class Block:
    """结构化文档块：一段正文或一张表格。"""

    type: str               # text / table
    text: str               # 清洗后的文本（表格为 Markdown 形式）
    page_no: int            # 1 起始页码（各文档独立编号）
    doc_name: str = ""      # 来源语料文件名（招股说明书1 / 招股说明书2）
    heading_path: str = ""  # 所属标题路径
    table_title: str = ""   # 表格标题（仅 type=table）
    meta: dict = field(default_factory=dict)


# 中文终止符：行尾出现这些符号时不做跨行拼接
_END_PUNCT = tuple("。！？；：”’）)】」！？.!?;")
# 页眉公司名模式（两份招股书通用，不写死具体公司）
_COMPANY_HEADER_RE = re.compile(
    r"^[一-龥]{2,20}(?:股份有限公司|有限责任公司)\s*")


def _clean_line(line: str) -> str:
    """清洗单行文本：行内剔除页眉页脚标记与署名公司名（不丢弃整行正文）。

    招股书页眉常与表格首行黏连（如“武汉力源信息技术股份有限公司 招股意向书 304”），
    因此采用行内 sub 删除，而非整行丢弃。

    :param line: 原始文本行
    :return: 清洗后的文本行（可能为空串）
    """
    text = line.replace("　", " ").replace("\xa0", " ").strip()
    # 归一化零宽字符与非常用空白，避免页眉正则因隐藏字符失配
    text = re.sub(r"[​﻿ -   ]", " ", text)
    for pattern in CONFIG.header_footer_patterns:
        text = re.sub(pattern, "", text)
    for word in CONFIG.noise_words:
        text = text.replace(word, "")
    # 行首残留的孤立页眉公司名统一剔除
    text = _COMPANY_HEADER_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _looks_tabular(page_text: str) -> bool:
    """启发式判断页面是否为表格页：出现“单位：”且含较多数字簇。

    :param page_text: 该页 PyMuPDF 提取的全文
    :return: 命中表格特征返回 True
    """
    if "单位：" not in page_text:
        return False
    return len(re.findall(r"\d[\d,\.]{2,}", page_text)) >= 8


def _is_heading(text: str) -> bool:
    """判断文本行是否为章节标题。

    :param text: 待判断的文本行
    :return: 命中标题模式且长度不超过 40 字时返回 True
    """
    if not (re.match(CONFIG.heading_pattern, text) and len(text) <= 40):
        return False
    # 数字编号列举项（如“4、法定代表人：赵马克”“5、公司成立日期：
    # 2001年8月9日”）是概况卡正文事实行，不是章节标题，不得丢弃
    if re.match(r"^\d+[、.]", text) and ("：" in text or ":" in text):
        return False
    return True


def _extract_tables(page_no: int, doc_name: str, pdf_page) -> List[Block]:
    """使用 pdfplumber 提取当前页全部表格并序列化为 Markdown。

    :param page_no: 1 起始页码
    :param doc_name: 来源文档名
    :param pdf_page: pdfplumber 页面对象
    :return: 表格 Block 列表
    """
    blocks: List[Block] = []
    for table in pdf_page.extract_tables() or []:
        rows = [[(cell or "").replace("\n", " ").strip() for cell in row]
                for row in table if row]
        if len(rows) < 2:
            continue
        md_lines = ["| " + " | ".join(rows[0]) + " |",
                    "| " + " | ".join(["---"] * len(rows[0])) + " |"]
        md_lines += ["| " + " | ".join(r) for r in rows[1:]]
        blocks.append(Block(type="table", text="\n".join(md_lines),
                            page_no=page_no, doc_name=doc_name))
    return blocks


def parse_pdf(path: str, doc_name: Optional[str] = None) -> List[Block]:
    """解析单个 PDF 为结构化 Block 列表（解析层唯一对外入口）。

    :param path: PDF 文件绝对路径
    :param doc_name: 来源文档名，缺省取文件名（去扩展名）
    :return: 有序 Block 列表（正文/表格），携带页码、标题路径与来源文档
    :raises PDFParseError: 文件无法打开或解析失败时抛出
    """
    import os
    doc_name = doc_name or os.path.splitext(os.path.basename(path))[0]
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PDFParseError(f"无法打开PDF文件: {path}，原因: {exc}") from exc

    plumber = None
    if pdfplumber is not None:
        try:
            plumber = pdfplumber.open(path)
        except Exception:
            plumber = None  # 表格回退不可用时不影响正文解析

    blocks: List[Block] = []
    heading_stack: List[str] = []
    try:
        for index, page in enumerate(doc):
            page_no = index + 1
            raw_dict = page.get_text("dict")
            page_h = page.rect.height
            page_lines: List[str] = []
            for item in raw_dict.get("blocks", []):
                if item.get("type") != 0:  # 0=文本块，1=图片块
                    continue
                line_text = "".join(
                    "".join(span.get("text", "") for span in line.get("spans", []))
                    for line in item.get("lines", []))
                # 页眉/页脚坐标区过滤：顶部5%或底部5%内的短文本直接跳过
                _x0, y0, _x1, y1 = item.get("bbox", (0, 0, 0, 0))
                in_margin = (y1 < page_h * 0.06) or (y0 > page_h * 0.95)
                if in_margin and len(line_text.strip()) < 40:
                    continue
                cleaned = _clean_line(line_text)
                if not cleaned:
                    continue
                if _is_heading(cleaned):
                    # 维护两级标题栈：“第X节”置顶，“一、”置第二级
                    if cleaned.startswith("第") and ("章" in cleaned or "节" in cleaned):
                        heading_stack = [cleaned]
                    elif re.match(r"^[一二三四五六七八九十]+、", cleaned):
                        heading_stack = heading_stack[:1] + [cleaned]
                page_lines.append(cleaned)

            # 硬断行合并：中文段落被 PDF 换行切断，行尾非终止符则拼接
            merged, buf = [], ""
            for line in page_lines:
                if _is_heading(line):
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

            page_text = "".join(merged)
            page_text_len = len(page_text)
            current_heading = " > ".join(heading_stack)

            # 表格提取：稀疏页直接回退 pdfplumber；强表格特征财务页同样提取；
            # 表格单元格文本覆盖率超过 60% 时以表格为准，避免表格被打散成噪声文本
            table_blocks: List[Block] = []
            if plumber is not None:
                need_table = (page_text_len < CONFIG.table_fallback_chars
                              or _looks_tabular(page_text))
                if need_table:
                    table_blocks = _extract_tables(
                        page_no, doc_name, plumber.pages[index])
            table_chars = sum(len(b.text) for b in table_blocks)
            if table_blocks and table_chars > page_text_len * 0.6:
                blocks.extend(table_blocks)
                continue

            for seg in merged:
                if _is_heading(seg):
                    continue
                blocks.append(Block(type="text", text=seg, page_no=page_no,
                                    doc_name=doc_name, heading_path=current_heading))
            blocks.extend(table_blocks)
    except PDFParseError:
        raise
    except Exception as exc:
        raise PDFParseError(f"解析PDF失败: {path}，原因: {exc}") from exc
    finally:
        doc.close()
        if plumber is not None:
            plumber.close()
    return blocks
