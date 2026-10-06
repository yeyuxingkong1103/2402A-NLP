# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
PDF 解析模块：用 PyMuPDF (fitz) 逐页提取 PDF 文本。

在系统中的位置：
    上游是入库脚本（ingest_pdf.py）和上传接口（main.py 的 /api/upload）——
    它们把 PDF 文件路径交给本模块；
    下游是 text_splitter——本模块只做「PDF -> 文本/Document」的转换，不负责切分。
    本模块调用的外部库：PyMuPDF (fitz)。

关键设计取舍：
    1. fitz 在函数内部惰性 import：库体积大，只在解析 PDF 时加载，避免拖慢
       服务启动；
    2. 用 page.get_text("text") 按阅读顺序吐纯文本，比 blocks/dict 模式省事，
       切分阶段再按段落/句子处理；
    3. 解析结果按页返回，metadata 里带 source 与 page_number，便于检索命中
       后向前端展示「出自哪个文件第几页」。
"""

import os  # 路径操作（取文件名做 source）
from typing import List  # 列表类型标注

from langchain_core.documents import Document  # 统一文档结构

from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger


def parse_pdf(file_path: str) -> List[str]:
    """
    用 PyMuPDF (fitz) 逐页提取文本，返回按页的文本列表

    参数：
        file_path：PDF 文件的绝对/相对路径。
    返回：
        List[str]，每个元素是一页的纯文本（按页码顺序，下标 0 = 第 1 页）；
        打不开文件或提取为空时返回空列表 []（提取空时只打 warning 不抛异常，
        让调用方决定是否走 OCR，本函数保持单一职责）。
    说明：
        "text" 模式按阅读顺序吐纯文本，比 blocks/dict 模式省事；
        每页文本里保留了原始换行，便于后续段落切分时按空行/换行识别边界。
    """
    import fitz  # PyMuPDF 惰性导入：库体积大，只在解析 PDF 时加载，避免拖慢服务启动

    if not os.path.isfile(file_path):  # 文件不存在
        logger.error(f"PDF 文件不存在：{file_path}")
        return []

    pages: List[str] = []  # 存放每页文本
    try:
        doc = fitz.open(file_path)  # 按路径打开 PDF
    except Exception as e:  # 打不开（损坏、加密、权限不足）
        logger.error(f"PDF 打开失败：{file_path}（{e}）")
        return []

    try:
        for page in doc:  # 逐页
            pages.append(page.get_text("text"))  # "text" 模式按阅读顺序吐纯文本
    finally:
        doc.close()  # 必须显式释放：PyMuPDF 持有底层文件句柄，否则文件多时句柄会泄漏

    text_all = "".join(pages).strip()
    if not text_all:  # 整本 PDF 都是空的（典型情况：扫描件没有文字层）
        logger.warning(f"PDF 文本提取为空，可能是扫描件：{file_path}（需要 OCR 处理）")
    logger.info(f"PDF 解析完成：{file_path}，共 {len(pages)} 页")
    return pages


def parse_pdf_to_document(file_path: str) -> List[Document]:
    """
    用 PyMuPDF (fitz) 逐页提取文本，返回 LangChain Document 列表

    与 parse_pdf 的区别：本函数把每页文本包成 Document，metadata 里带：
        - source：文件名（不带目录，便于前端展示和后续幂等去重）
        - page_number：页码（从 1 开始，给前端「第 N 页」用）

    参数：
        file_path：PDF 文件路径。
    返回：
        List[Document]，每个 Document 对应一页；提取为空时返回空列表 []。
    说明：
        这里只按「页」粗粒度组织 Document，真正的切分由 text_splitter
        在入库前进一步处理（一页可能切成多个 chunk）；不在这里切是为了
        让切分逻辑统一收口在 text_splitter，便于切换切分策略。
    """
    pages = parse_pdf(file_path)  # 复用纯文本解析
    source = os.path.basename(file_path)  # 只取文件名作为 source（带后缀，不带目录）

    docs: List[Document] = []
    for i, text in enumerate(pages):  # 逐页包装
        if not text.strip():  # 空白页跳过（入库空内容只会污染检索结果）
            continue
        docs.append(Document(
            page_content=text,  # 一页的原文
            metadata={
                "source": source,  # 来源文件名
                "page_number": i + 1,  # 页码从 1 开始（人类习惯，避免「第 0 页」）
            },
        ))
    logger.info(f"PDF 转 Document 完成：{file_path}，有效页 {len(docs)} 个")
    return docs
