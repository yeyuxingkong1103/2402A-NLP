# -*- coding: utf-8 -*-
"""
PDF 解析模块（文字层）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：基于 PyMuPDF 提取 PDF 每页文字，供后续分块与向量化使用。
     表格解析见 table_parser.py（工单03），图像解析见 image_parser.py（工单04）。
"""
import re   # 编译页眉/页脚匹配的正则
import fitz  # PyMuPDF：PDF 解析库，import 名固定为 fitz（不是 pymupdf）

# 页眉/页脚模式：公司名+文档名行、1-1-XX 编码行、独立数字页码行
# ^ 与 $ 锚定整行，防止正文中恰好包含这些词的行被误删
_HEADER_PATTERNS = [
    re.compile(r"^武汉兴图新科电子股份有限公司\s+招股意向书\s*$"),  # 页眉：公司名+文档名
    re.compile(r"^1-1-\d+\s*$"),        # 页脚编号行，如 "1-1-23"
    re.compile(r"^\d{1,4}\s*$"),        # 独立数字页码行（1~4 位数字）
    re.compile(r"^武汉兴图新科电子股份有限公司\s*$"),  # 仅公司名的页眉变体
    re.compile(r"^招股意向书\s*$"),     # 仅文档名的页眉变体
]


def clean_page_text(text):
    """清洗页眉页脚噪音，避免其稀释分块语义（提升向量检索精度）"""
    lines = text.split("\n")  # PDF 提取文本按行处理，页眉页脚都是独立行
    kept = []
    for ln in lines:
        s = ln.strip()  # 先去首尾空白再匹配，防止行首缩进导致匹配失败
        # 任一页眉/页脚模式命中则丢弃该行；match 只从行首匹配
        if any(p.match(s) for p in _HEADER_PATTERNS):
            continue
        kept.append(ln)
    # 重新用换行拼回，保持原有段落结构
    return "\n".join(kept)


def parse_pdf_text(pdf_path, clean_headers=True):
    """逐页提取 PDF 文字（可选清洗页眉页脚）
    参数 clean_headers: False 时不清洗（仅工单02"优化前"基线对比使用）
    返回: [{"page": 页码(1起), "text": 页面文本}, ...]
    """
    doc = fitz.open(pdf_path)  # 打开 PDF 文档句柄，用完必须 close 释放文件
    pages = []
    for i, page in enumerate(doc):  # i 从 0 起，返回时 +1 转为 1 起页码
        # get_text("text") 提取纯文本模式；strip() 去掉页首尾空白
        text = page.get_text("text").strip()
        if text and clean_headers:
            text = clean_page_text(text)  # 非空页才做页眉页脚清洗
        if text:  # 空白页跳过
            # 清洗后可能变空（整页只有页眉），所以要二次判空再入库
            pages.append({"page": i + 1, "text": text})
    doc.close()  # 及时关闭文件句柄，避免 Windows 下文件占用
    return pages


def pdf_page_count(pdf_path):
    """返回 PDF 总页数（工单01 统计文档规模用）"""
    doc = fitz.open(pdf_path)
    n = doc.page_count  # page_count 属性直接读页数，无需遍历
    doc.close()
    return n
