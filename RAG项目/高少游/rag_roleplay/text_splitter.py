# -*- coding: utf-8 -*-
"""
文本切分模块：5 种切分策略
1. 固定长度切分
2. 句子切分
3. 段落切分
4. 标题结构切分（Markdown）
5. 语义切分（进阶，需调向量模型）
"""

from typing import List  # 列表类型标注

from langchain_core.documents import Document  # 统一文档结构
from langchain_text_splitters import (  # 切分器
    RecursiveCharacterTextSplitter,  # 递归字符切分（支持自定义分隔符优先级）
    MarkdownHeaderTextSplitter,  # Markdown 标题切分
)

from config import CHUNK_SIZE, CHUNK_OVERLAP  # 切分参数
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# 自然结构分隔符：按顺序递退切分（能整段切就整段切，切不动退一级）
NATURAL_SEPARATORS = [
    "\n\n",  # 空行（段落之间）
    "\n",  # 换行（标题/条目之间）
    "。",  # 中文句号
    "！",  # 感叹号
    "？",  # 问号
    "；",  # 分号
    ". ",  # 英文句号+空格
    " ",  # 空格
    "",  # 兜底：硬切
]


def split_by_fixed_length(text: str, source: str, size: int = 500, overlap: int = 80) -> List[Document]:
    """策略一：固定长度切分（最简单，适合无结构纯文本快速处理）"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=[""],  # 只用空串 → 纯按长度硬切
    )
    chunks = splitter.create_documents([text])
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    return chunks


def split_by_sentence(text: str, source: str, size: int = 500, overlap: int = 80) -> List[Document]:
    """策略二：句子级切分（以句号/问号/感叹号为分隔，尽量不切断句子）"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["。", "！", "？", "；", "\n", ""],  # 优先在句子边界切
    )
    chunks = splitter.create_documents([text])
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    return chunks


def split_by_paragraph(text: str, source: str, size: int = 500, overlap: int = 80) -> List[Document]:
    """策略三：段落级切分（优先按空行切段，段内再按句子递退切）"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=NATURAL_SEPARATORS,  # 段落 > 换行 > 句子 > 兜底
    )
    chunks = splitter.create_documents([text])
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    return chunks


def split_by_markdown_header(text: str, source: str, size: int = 500, overlap: int = 80) -> List[Document]:
    """策略四：Markdown 标题切分（按 #/##/### 层级，标题信息保留在 section 里）"""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],  # 三级标题
    )
    header_docs = header_splitter.split_text(text)  # 先按标题切大块
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=NATURAL_SEPARATORS,
    )
    chunks = recursive_splitter.split_documents(header_docs)  # 超长块再细切
    # 统一 metadata：标题路径合并成 section，写进正文增强语义
    for i, c in enumerate(chunks):
        title_parts = [c.metadata.get(k, "") for k in ("h1", "h2", "h3") if c.metadata.get(k)]
        section = " / ".join(title_parts)  # 如 "民法典 / 相邻关系 / 第二百八十八条"
        if section:
            c.page_content = f"{section}\n{c.page_content}"  # 标题写进正文
        c.metadata = {"source": source, "chunk_index": i, "section": section}  # 统一三键
    return chunks


def split_by_semantic(text: str, source: str, embedding_func=None) -> List[Document]:
    """
    策略五：语义切分（用向量模型对每句算向量，语义相近的句子聚成一块）
    需要传 embedding 函数；不传则回退到段落切分
    """
    if embedding_func is None:  # 没传向量函数
        logger.warning("语义切分需要 embedding 函数，未提供，回退到段落切分")
        return split_by_paragraph(text, source)
    try:
        from langchain_experimental.text_splitters import SemanticChunker  # 语义切分器（实验性）
    except ImportError:
        logger.warning("langchain_experimental 未安装，回退到段落切分")
        return split_by_paragraph(text, source)
    splitter = SemanticChunker(  # 语义切分
        embeddings=embedding_func,  # 向量函数
        breakpoint_threshold_type="percentile",  # 用百分位数法找断点
        breakpoint_threshold_amount=95,  # 前 95% 的语义距离才切分（保守，避免切太碎）
    )
    chunks = splitter.create_documents([text])
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    logger.info(f"语义切分完成：{len(chunks)} 块")
    return chunks


# 策略选择映射表：外部按名字调用
SPLIT_STRATEGIES = {
    "fixed": split_by_fixed_length,
    "sentence": split_by_sentence,
    "paragraph": split_by_paragraph,
    "markdown": split_by_markdown_header,
    "semantic": split_by_semantic,
}


def split_text(raw_text: str, source: str, strategy: str = "paragraph", is_markdown: bool = False,
               embedding_func=None) -> List[Document]:
    """
    统一入口：按策略名切分文本

    Args:
        raw_text:  原始纯文本
        source:    来源文件名（写入 metadata.source）
        strategy:  切分策略名（fixed/sentence/paragraph/markdown/semantic）
        is_markdown: 是否 Markdown 文档（True 时强制用 markdown 策略）
        embedding_func: 语义切分需要的向量函数（仅 strategy=semantic 时用）
    Returns:
        Document 列表
    """
    if is_markdown:  # Markdown 文档强制用标题结构切
        return split_by_markdown_header(raw_text, source)
    func = SPLIT_STRATEGIES.get(strategy, split_by_paragraph)  # 找不到策略默认段落切
    if strategy == "semantic":  # 语义切分需要传向量函数
        return func(raw_text, source, embedding_func)
    return func(raw_text, source)
