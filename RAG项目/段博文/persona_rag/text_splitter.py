# -*- coding: utf-8 -*-
"""
文本切分模块：5 种切分策略
1. 固定长度切分
2. 句子切分
3. 段落切分
4. 标题结构切分（Markdown）
5. 语义切分（进阶，需调向量模型）

在系统中的位置：
    上游是 doc_parser（解析出的纯文本 + 是否 Markdown）；
    下游是入库逻辑（本模块产出的 Document 列表会被算向量写进 Milvus，同时用于重建 BM25）。

职责与关键取舍：
    把长文本切成检索友好的小块。切分粒度直接决定检索质量——
    块太大：检索命中后噪声多、向量语义被稀释；块太小：一句完整的话被切碎，答案不完整。
    因此默认 size=500、overlap=80：500 字左右能装下一个完整法条或一段完整论述，
    80 字重叠用来兜住"关键句正好落在切口上"的情况。
    五种策略从粗到细，让使用方按文档类型选：结构化的用 markdown、散文用段落/句子、
    无结构纯文本用 fixed；semantic 效果最好但最慢（要逐句算向量），且是实验性 API。
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
# 顺序即优先级：越靠前的分隔符"语义边界"越强（空行=段落 > 换行 > 句子 > 空格 > 硬切），
# 递归切分器会先尝试用最靠前的分隔符，只有当某一段仍然超过 chunk_size 时才降级用后面的，
# 所以这个列表决定了"块内语义完整性"的上限——想让块更完整就往前面加更强的边界
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
    """
    策略一：固定长度切分（最简单，适合无结构纯文本快速处理）

    Args:
        text:    待切分的全文（由 doc_parser 解析得到）
        source:  来源文件名，写进每个块的 metadata["source"]，检索命中后可向前端展示出处
        size:    每块最大字符数，默认 500；调大→上下文更完整但检索噪声多，调小→更精准但语义易被截断
        overlap: 相邻块重叠的字符数，默认 80；设为 0 会让切口处的信息被撕成两半、两边都检索不到
    Returns:
        List[Document]，每个 Document 的 metadata 固定为三键：
        {"source": 来源文件名, "chunk_index": 块序号（从 0 递增）, "section": ""（本策略没有章节信息）}
    取舍:
        separators=[""] 是"空串兜底分隔符"，效果等于逐字符判断长度上限，
        所以速度最快、块长最均匀，代价是会从句中硬切（可能把一个法条劈成两半）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )

    chunks = splitter.create_documents([text])  # 传列表：该 API 支持一次切多篇文本，这里只喂一篇
    # 统一重写 metadata：langchain 默认只会带很简单的来源信息，这里统一成入库需要的三键格式
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    return chunks


def split_by_sentence(text: str, source: str, size: int = 500, overlap: int = 80) -> List[Document]:
    """
    策略二：句子级切分（以句号/问号/感叹号为分隔，尽量不切断句子）

    Args:
        text:    待切分全文
        source:  来源文件名，写进 metadata["source"]
        size:    每块最大字符数，默认 500
        overlap: 相邻块重叠字符数，默认 80
    Returns:
        List[Document]，metadata 三键 {"source","chunk_index","section"}（本策略 section 恒为空串）
    说明:
        RecursiveCharacterTextSplitter 的行为是"能用上层分隔符就在上层切"，
        所以 separators 的顺序 = 优先级：先按句号切，句子太长才退到下一级，
        直到 "" 兜底硬切，因此正常情况下一句话不会被劈开
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["。", "！", "？", "；", "\n", ""],  # 优先在句子边界切（分隔符会保留在上一块末尾）
    )
    chunks = splitter.create_documents([text])
    # 统一 metadata：与其它策略保持同样的三键结构，方便入库和前端展示共用一套逻辑
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    return chunks


def split_by_paragraph(text: str, source: str, size: int = 500, overlap: int = 80) -> List[Document]:
    """
    策略三：段落级切分（优先按空行切段，段内再按句子递退切）

    Args:
        text:    待切分全文
        source:  来源文件名，写进 metadata["source"]
        size:    每块最大字符数，默认 500（超过就继续往句号层级切）
        overlap: 相邻块重叠字符数，默认 80
    Returns:
        List[Document]，metadata 三键 {"source","chunk_index","section"}（本策略 section 恒为空串）
    说明:
        这是默认策略（split_text 里 strategy 找不到时回退到它），
        因为中文文档绝大多数以"空行分段、段内成句"组织，按这个层级切最贴合原文语义单元
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=NATURAL_SEPARATORS,  # 段落 > 换行 > 句子 > 兜底（共用模块顶部定义，保证各策略分隔符一致）
    )
    chunks = splitter.create_documents([text])
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    return chunks


def split_by_markdown_header(text: str, source: str, size: int = 500, overlap: int = 80) -> List[Document]:
    """
    策略四：Markdown 标题切分（按 #/##/### 层级，标题信息保留在 section 里）

    Args:
        text:    待切分全文（必须是 Markdown 语法，doc_parser 判定为 .md 时才走这里）
        source:  来源文件名，写进 metadata["source"]
        size:    每块最大字符数，默认 500（只用于"章节太长时再细切"这一步）
        overlap: 相邻块重叠字符数，默认 80
    Returns:
        List[Document]，每个 Document 的 metadata 固定为三键 {"source","chunk_index","section"}；
        section 是形如 "民法典 / 相邻关系 / 第二百八十八条" 的标题路径，没有标题的块为空串
    关键取舍:
        1. 两段式切分：先按标题切大块（保结构），超长的块再用递归字符切分（保长度上限），
           若只做前者会出现超长块、只做后者会丢掉标题层级；
        2. 标题被拼进正文开头，是因为向量模型只看 page_content、看不到 metadata ——
           光看"本条所称相邻关系……"不知道出自哪部法，带上标题路径检索命中率会明显提高
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],  # 三级标题，再深的层级不切（避免块过碎）
    )
    header_docs = header_splitter.split_text(text)  # 先按标题切大块（标题会写进 metadata 的 h1/h2/h3）
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=NATURAL_SEPARATORS,  # 与其它策略复用同一套分隔符
    )
    chunks = recursive_splitter.split_documents(header_docs)  # 超长块再细切（split_documents 会自动继承原有 metadata）
    # 统一 metadata：标题路径合并成 section，写进正文增强语义
    for i, c in enumerate(chunks):
        title_parts = [c.metadata.get(k, "") for k in ("h1", "h2", "h3") if c.metadata.get(k)]  # 只保留存在的那几级标题
        section = " / ".join(title_parts)  # 如 "民法典 / 相邻关系 / 第二百八十八条"
        if section:
            c.page_content = f"{section}\n{c.page_content}"  # 标题写进正文（不改 page_content 的话向量只认正文）
        c.metadata = {"source": source, "chunk_index": i, "section": section}  # 统一三键（顺带丢掉 h1/h2/h3 等中间键）
    return chunks


def split_by_semantic(text: str, source: str, embedding_func=None) -> List[Document]:
    """
    策略五：语义切分（用向量模型对每句算向量，语义相近的句子聚成一块）
    需要传 embedding 函数；不传则回退到段落切分

    Args:
        text:           待切分全文
        source:         来源文件名，写进 metadata["source"]
        embedding_func: 向量函数（项目里是本地 bge-m3，1024 维、CUDA 上跑）；
                        传 None 会直接回退，所以调用方必须显式传入才算启用语义切分
    Returns:
        List[Document]，metadata 三键 {"source","chunk_index","section"}（本策略 section 恒为空串）
    降级行为（两级兜底）:
        1. 没传 embedding_func：打 warning 后回退到段落切分；
        2. 没装 langchain_experimental：打 warning 后回退到段落切分。
        两种情况都保证"一定有结果返回"，不会让入库流程因为可选依赖缺失而中断
    取舍:
        效果最好的策略（切出来的块内部语义最连贯），但代价也最大：
        要先对每个句子算一次向量，长文档会很慢；而且要用和检索同一个向量模型，否则语义空间不一致
    """
    if embedding_func is None:  # 没传向量函数
        logger.warning("语义切分需要 embedding 函数，未提供，回退到段落切分")
        return split_by_paragraph(text, source)
    try:
        from langchain_experimental.text_splitters import SemanticChunker  # 语义切分器（实验性 API，放在 try 里防版本变动）
    except ImportError:
        logger.warning("langchain_experimental 未安装，回退到段落切分")
        return split_by_paragraph(text, source)
    splitter = SemanticChunker(  # 语义切分
        embeddings=embedding_func,  # 向量函数（负责把句子转成向量用于比较相邻句距离）
        breakpoint_threshold_type="percentile",  # 用百分位数法找断点：距离超过阈值分位数就切
        breakpoint_threshold_amount=95,  # 取 95 分位，即只有最"不相关"的那 5% 相邻句之间才切（偏保守，避免切太碎）
    )
    chunks = splitter.create_documents([text])
    for i, c in enumerate(chunks):
        c.metadata = {"source": source, "chunk_index": i, "section": ""}
    logger.info(f"语义切分完成：{len(chunks)} 块")  # 语义切分耗时长，数量日志便于评估是否切得合理
    return chunks


# 策略选择映射表：外部按名字调用
# 用"字符串 -> 函数对象"的字典做分发（而不是一长串 if/elif），
# 好处是新增策略只要在表里加一行，调用方传的 strategy 名字不用改代码
SPLIT_STRATEGIES = {
    "fixed": split_by_fixed_length,  # 固定长度：最快，适合无结构文本
    "sentence": split_by_sentence,  # 句子级：保证句子完整
    "paragraph": split_by_paragraph,  # 段落级：默认策略，最贴合中文文档
    "markdown": split_by_markdown_header,  # 标题结构：适合 .md 等有层级的文档
    "semantic": split_by_semantic,  # 语义切分：效果最好但最慢，需要向量模型
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
    说明:
        is_markdown 的优先级最高，会直接覆盖调用方传的 strategy：
        因为 Markdown 文档按标题切几乎是必然更优的选择，没必要让使用方再手动传一次
    降级行为:
        strategy 传了表里没有的名字（含拼写错误）时，静默回退到段落切分，
        不抛异常——入库是批量流程，不能因为一个策略名写错就整批失败
    """
    if is_markdown:  # Markdown 文档强制用标题结构切
        return split_by_markdown_header(raw_text, source)
    func = SPLIT_STRATEGIES.get(strategy, split_by_paragraph)  # 找不到策略默认段落切
    if strategy == "semantic":  # 语义切分需要传向量函数
        return func(raw_text, source, embedding_func)  # 位置参数注意：对应 split_by_semantic 的 embedding_func
    return func(raw_text, source)  # 其余策略签名一致，直接透传
