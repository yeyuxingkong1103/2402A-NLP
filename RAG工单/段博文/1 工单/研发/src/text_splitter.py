# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
文本切分模块：用 RecursiveCharacterTextSplitter 把长文本切成检索友好的小块。

在系统中的位置：
    上游是 pdf_parser（按页解析出的 Document 列表，或纯文本）；
    下游是入库逻辑（本模块产出的 Document 列表会被算向量写进 Milvus，
    同时用于重建 BM25）。

职责与关键取舍：
    把长文本切成检索友好的小块。切分粒度直接决定检索质量——
    块太大：检索命中后噪声多、向量语义被稀释；
    块太小：一句完整的话被切碎，答案不完整。
    因此默认 chunk_size=500、overlap=80：500 字左右能装下一个完整段落，
    80 字重叠用来兜住「关键句正好落在切口上」的情况。

    RecursiveCharacterTextSplitter 的行为是「能用上层分隔符就在上层切」，
    所以 separators 的顺序 = 优先级：先按段落切，段落太长才退到下一级，
    直到 "" 兜底硬切，因此正常情况下一句话不会被劈开。
"""

from typing import List  # 列表类型标注

from langchain_core.documents import Document  # 统一文档结构
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符切分

from config import CHUNK_SIZE, CHUNK_OVERLAP  # 切分参数
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# 自然结构分隔符：按顺序递退切分（能整段切就整段切，切不动退一级）
# 顺序即优先级：越靠前的分隔符「语义边界」越强（空行=段落 > 换行 > 句子 > 空格 > 硬切）
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


def _make_splitter(chunk_size: int = CHUNK_SIZE,
                   chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """
    内部工具：构造一个递归字符切分器

    参数：
        chunk_size：每块最大字符数
        chunk_overlap：相邻块重叠字符数
    返回：
        RecursiveCharacterTextSplitter 实例（已配置分隔符优先级）。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=NATURAL_SEPARATORS,  # 段落 > 换行 > 句子 > 兜底
    )


def split_text(text: str, source: str = "") -> List[Document]:
    """
    切分一段纯文本，返回 Document 列表

    参数：
        text：待切分的全文（由 pdf_parser 解析得到，或调用方直接给的字符串）。
        source：来源文件名，写进每个块的 metadata["source"]，检索命中后可
                向前端展示出处；不传则为空串（仍可入库，但溯源能力下降）。
    返回：
        List[Document]，每个 Document 的 metadata 固定为三键：
        {"source": 来源文件名, "chunk_index": 块序号（从 0 递增）, "page_number": 0}。
    说明：
        page_number 这里给 0 表示「未知页」（split_text 接收的是整篇文本，
        已经丢失了页码信息；若需要保留页码请用 split_documents 走 Document
        列表那条路径，那里的 page_number 会被 split_documents 自动继承）。
    """
    if not text or not text.strip():  # 空文本直接返回空列表
        logger.warning("切分输入为空，跳过")
        return []

    splitter = _make_splitter()  # 用 config 的默认参数
    chunks = splitter.create_documents([text])  # 传列表：该 API 支持一次切多篇，这里只喂一篇

    for i, c in enumerate(chunks):  # 统一 metadata 三键格式
        c.metadata = {
            "source": source,
            "chunk_index": i,  # 块序号（同源内唯一，配合 source 做幂等去重）
            "page_number": 0,  # 整篇切分不知道页码，用 0 占位
        }
    logger.info(f"切分完成：source={source or '(未指定)'}，共 {len(chunks)} 块")
    return chunks


def split_documents(documents: List[Document]) -> List[Document]:
    """
    对已有的 Document 列表做切分（保留原有 metadata）

    与 split_text 的区别：本函数接收的是 Document 列表（如 pdf_parser
    按页产出的），切分时 split_documents 会自动继承原有 metadata（包括
    source 和 page_number），这样每个切出来的小块都能正确溯源到原文档
    的某一页。

    参数：
        documents：待切分的 Document 列表（通常来自 pdf_parser.parse_pdf_to_document）。
    返回：
        List[Document]，metadata 在原有基础上补一个 chunk_index（块序号，
        从 0 递增），用于幂等去重。
    说明：
        原有 metadata 不会被覆盖，只追加 chunk_index；如果原 metadata 里
        本来就有 chunk_index，会被这里覆盖（按「全局递增」重新编号，便于
        跨页统一做幂等键）。
    """
    if not documents:  # 空列表
        logger.warning("切分文档列表为空，跳过")
        return []

    splitter = _make_splitter()
    chunks = splitter.split_documents(documents)  # split_documents 会继承原 metadata

    # 给每个小块补一个全局递增的 chunk_index（原有 page_number/source 保留）
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    logger.info(f"切分 Document 完成：输入 {len(documents)} 篇，输出 {len(chunks)} 块")
    return chunks
