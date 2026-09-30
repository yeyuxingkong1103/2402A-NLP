"""src/offline/chunkers.py —— 文本分块策略集合。

在链路中的位置（离线侧中间环节）：
    src/offline/parsers.py 的按页文本 → 【本文件】 → src/offline/embedder.py 向量化
上游调用方：src/offline/pipeline.py 的 build_role()

为什么分块是 RAG 质量的第一道关口：
    chunk 太大 → 一个问题会被大量无关内容淹没，检索精度下降、还浪费上下文预算
    chunk 太短 → 一句话被切断，检索到的片段缺少上下文，模型无法据此回答
    五种策略就是给不同文档类型准备的选择：结构清晰的用 heading，纯叙述的用 sentence。

输出统一为 Chunk 对象，携带 content/summary/page/parent_id/metadata 五个字段 ——
其中 page 和 metadata 是从解析阶段一路带下来的，最终会成为检索结果的"出处"信息。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class Chunk:
    """一个分块。

    字段：
        content:   分块正文，最终会被向量化并写入 Milvus
        summary:   截断后的摘要（默认 256 字），供列表展示，避免列表接口拖回全文
        page:      所在页码，用于答案引用（"第 N 页"）
        parent_id: 父块序号，配合 add_parent_child 做"小块检索、大块喂给模型"
        metadata:  解析器信息 + 分块策略等附加信息
    """

    content: str
    summary: str
    page: int
    parent_id: int | None
    metadata: dict[str, Any]


def _summary(text: str, limit: int = 256) -> str:
    """生成摘要：先把空白压成单空格，再截断。

    参数：
        text: 原文
        limit: 最大字符数
    返回：
        长度不超过 limit 的单行摘要。

    先把换行压成空格再截断：
        否则截断点可能落在换行上，摘要里带着半截换行，在列表里显示会很难看。
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def fixed_chunks(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """固定长度切分（最简单的基线策略）。

    参数：
        text: 待切文本
        size: 每块字符数
        overlap: 相邻块的重叠字符数
    返回：
        文本块列表。
    异常：
        size <= overlap -> ValueError（否则步长为 0 或负，会死循环）。

    overlap 的作用：
        切点恰好落在一句话中间时，前后两块各拿到半句，都检索不到完整语义。
        重叠 80 字符让跨越切点的那句话至少在一个块里是完整的。

    兜底 `or [text]`：
        文本短于 size 时循环体一次都不执行，返回空列表 ——
        那会让内容直接丢失，所以空结果时返回原文本身。
    """
    if size <= overlap:
        raise ValueError("size 必须大于 overlap")
    result = []
    start = 0
    while start < len(text):
        result.append(text[start : start + size])
        start += size - overlap  # 步长 = size - overlap，即每块前进这么多
    return result or [text]


def sentence_chunks(text: str, max_chars: int = 500) -> list[str]:
    """按句子切分，再把相邻句子打包到接近 max_chars。

    参数：
        text: 待切文本
        max_chars: 每块的目标上限
    返回：
        文本块列表。

    为什么按句号切而不是按固定长度切：
        句子是语义的天然边界。按句切分能保证每一块的语义是完整的，
        检索命中时给模型看到的是完整的几句论述，而不是半句话。

    切句正则 (?<=[。！？.!?])\\s* 用后行断言：
        在标点之后断开但不吃掉标点（断言是零宽的），
        所以"第一句。第二句。"会切成 ["第一句。", "第二句。"]，标点留在前一句里。

    超长单句的特殊处理：
        如果一句话本身就超过 max_chars，它会被单独成块（不会硬切），
        因为硬切会造成半句话 —— 宁可这一块大一点。
    """
    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s*", text) if part.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        # +1 是算上拼接时那个空格的长度，避免拼完刚好超出上限
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)  # 收尾：最后一段没攒满也要成块
    return chunks or [text]


def paragraph_chunks(text: str, max_chars: int = 700) -> list[str]:
    """按自然段切分，再把相邻段落打包到接近 max_chars。

    参数：
        text: 待切文本
        max_chars: 每块的目标上限
    返回：
        文本块列表。

    段落是比句子更粗的语义单元：
        一个段落通常只在讲一件事，所以按段分块能让每块的主题更集中，
        向量也更"纯"—— 这直接提升检索命中率。
    """
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return _pack(paragraphs, max_chars)


def heading_chunks(text: str, max_chars: int = 700) -> list[str]:
    """按标题切分（结构化文档的首选策略）。

    参数：
        text: 待切文本
        max_chars: 每块的目标上限
    返回：
        文本块列表。

    识别两种标题：
        1. Markdown 标题  # / ## / ###（1~6 级）
        2. 数字编号标题   1 / 3.2 / 3.2.1
    用 (?m) 多行模式配合零宽先行断言 (?=...) 切分：
        断言不消耗标题本身，所以标题留在后一块的开头（而不是被切到前一块尾），
        这样每块都自带章节名，检索命中后能直接告诉用户"来自 3.2 设备要求"。

    这是国标/规范类文档的最佳策略 —— 它们的章节结构本身就是权威的语义边界。
    """
    sections = re.split(r"(?m)(?=^#{1,6}\s+|^\d+(?:\.\d+)*\s+)", text)
    return _pack([part.strip() for part in sections if part.strip()], max_chars)


def semantic_chunks(text: str, max_chars: int = 700) -> list[str]:
    """在段落分块的基础上，给每块补上"上一块的尾部"作为上下文。

    参数：
        text: 待切文本
        max_chars: 每块的目标上限（实际会略超，因为要额外拼接上文）
    返回：
        文本块列表。

    为什么这样做：
        段落分块切得太"干净"，丢失了跨段的指代关系 ——
        比如第 3 段说"该装置"，第 2 段才说明"该装置"指什么。
        单独把第 3 段给模型，它无法判断"该装置"是谁。
        在块首粘上上一段的最后 100 字，就能保住这个指代链。

    这是"语义分块"在本项目里的实现方式：
        不引入额外的 embedding 模型去判断语义边界（成本高、不可控），
        而是用"带上下文前缀"的方式低成本缓解跨段指代问题。

    只取上一段最后 100 字而不是整段：
        整段会让每块体积翻倍、大量重复内容进入向量库，性价比很低。
    """
    paragraphs = paragraph_chunks(text, max_chars)
    if len(paragraphs) <= 1:
        return paragraphs  # 只有一段时没有"上文"可拼，直接返回
    result = []
    for index, paragraph in enumerate(paragraphs):
        previous = paragraphs[index - 1][-100:] if index else ""
        result.append(f"{previous}\n{paragraph}" if previous else paragraph)
    return result


def _pack(parts: Iterable[str], max_chars: int) -> list[str]:
    """把若干文本片段（段落/章节）贪心打包到接近 max_chars。

    参数：
        parts: 待打包的片段序列
        max_chars: 每包的目标上限
    返回：
        打包后的文本列表。

    为什么是"打包"而不是"一片一块"：
        切出的小段落可能只有二十几个字，单独成块的话向量里几乎没有信息量，
        检索时容易被误召回。把相邻小段落合并到几百字，信息量才足够。

    +2 是算上拼接用的两个换行符 \\n\\n。
    贪心策略（装不下就换下一个包）不追求最优解，但简单、稳定、可预测。
    """
    result, current = [], ""
    for part in parts:
        if current and len(current) + len(part) + 2 > max_chars:
            result.append(current)
            current = ""
        current = f"{current}\n\n{part}".strip()
    if current:
        result.append(current)
    return result


def build_chunks(pages: list[dict[str, Any]], strategy: str = "heading", max_chars: int = 700) -> list[Chunk]:
    """按指定策略把按页文本切成 Chunk 列表（对外主入口）。

    参数：
        pages: 解析器的输出（每项含 page / text / metadata）
        strategy: 策略名，可选 fixed / sentence / paragraph / heading / semantic
        max_chars: 每块目标长度
    返回：
        Chunk 列表。
    异常：
        策略名不在表内 -> ValueError。

    注意是逐页切分（外层 for page），不跨页合并：
        跨页拼接会让 chunk 的 page 字段失去意义（一个块同时属于两页），
        而页码是本项目答案引用的核心依据，必须保持"一块一页码"。

    metadata 里记录 strategy：
        同一份文档换策略重建后，能从元数据看出每块是用哪种策略切出来的，
        这对归因"指标变化是策略导致的还是数据导致的"很关键。
    """
    strategies = {"fixed": fixed_chunks, "sentence": sentence_chunks, "paragraph": paragraph_chunks, "heading": heading_chunks, "semantic": semantic_chunks}
    if strategy not in strategies:
        raise ValueError(f"未知分块策略：{strategy}")
    chunks: list[Chunk] = []
    for page in pages:
        texts = strategies[strategy](page.get("text", ""), max_chars)
        for text in texts:
            chunks.append(Chunk(content=text, summary=_summary(text), page=int(page.get("page", -1)), parent_id=None, metadata={**page.get("metadata", {}), "strategy": strategy}))
    return chunks


def add_parent_child(chunks: list[Chunk], parent_size: int = 1800) -> list[Chunk]:
    """为每个块生成一个"父块"，形成父子对（小块检索、大块喂模型）。

    参数：
        chunks: 原始小块列表
        parent_size: 父块保留的字符数上限
    返回：
        长度翻倍的列表，父子块交替出现（父在前、子在紧随其后）。

    这个设计要解决的是 RAG 里的一个经典矛盾：
        用小块做向量检索 → 命中准（信息密度高、噪声少）
        但把小块喂给模型 → 上下文不足，模型答不好
        解法就是两者兼顾：用小子块去检索、命中后改用它的父块内容去生成答案。

    父块的 content 是 child.content[:parent_size]：
        注意这里实现上父块只是子块的截断副本（不是"合并相邻若干块"），
        所以当 parent_size 大于子块本身长度时，父块内容与子块相同。
        它的价值在于统一了数据结构、给上层留下了替换实现的位置。

    父块的 parent_id 为 None，子块的 parent_id 指向父块的序号 index：
        因为父子块交替插入（父在 2i、子在 2i+1），
        所以子块的 parent_id 就是它在 result 中"父块所在的位置"。
        上层按这个序号回查即可拿到父块内容。
    """
    result: list[Chunk] = []
    for index, chunk in enumerate(chunks):
        parent = Chunk(content=chunk.content[:parent_size], summary=_summary(chunk.content, 512), page=chunk.page, parent_id=None, metadata={**chunk.metadata, "chunk_type": "parent"})
        result.append(parent)
        result.append(Chunk(content=chunk.content, summary=chunk.summary, page=chunk.page, parent_id=index, metadata={**chunk.metadata, "chunk_type": "child"}))
    return result
