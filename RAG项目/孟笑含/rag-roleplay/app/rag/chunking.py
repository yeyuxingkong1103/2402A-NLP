# -*- coding: utf-8 -*-
"""分块器：按句切分，贪心聚合到目标大小，块间保留重叠。"""
import re
# 解析：正则模块（断句与标题识别）

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|(?<=\.)(?=\s)|(?<=\S)(?=\n(?!\n))")
# 解析：断句正则三规则——①中文标点后切 ②英文句号+空白后切 ③单个换行前切（连续换行=段落间隔不切）


def split_sentences(text: str) -> list[str]:
    """按中文标点切句，保留标点；空白句丢弃。"""
    parts = _SENTENCE_END.split(text)
    # 解析：按断句正则切分（后向断言保留标点）
    return [p for p in parts if p.strip()]
    # 解析：过滤纯空白句返回


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 80) -> list[str]:
    """把文本切成不超过 chunk_size 的块，相邻块尾部/头部重叠 overlap 个字符。"""
    if chunk_size <= overlap:
        # 解析：参数合法性校验
        raise ValueError("chunk_size 必须大于 overlap")
        # 解析：块大小必须大于重叠量（否则死循环）
    sentences = split_sentences(text)
    # 解析：先断句
    chunks: list[str] = []
    # 解析：结果块列表
    current = ""
    # 解析：当前聚合中的块

    for sentence in sentences:
        # 解析：逐句聚合
        if len(sentence) > chunk_size:
            # 解析：单句超长（无法聚合）
            # 单句超长：先落当前块，再按窗口硬切
            if current:
                # 解析：有未落的聚合块
                chunks.append(current)
                # 解析：先落块
                current = ""
                # 解析：清空
            pos = 0
            # 解析：硬切窗口起点
            while pos < len(sentence):
                # 解析：按窗口循环切
                chunks.append(sentence[pos : pos + chunk_size])
                # 解析：切出一块
                pos += chunk_size - overlap
                # 解析：窗口前移（减去重叠量）
            continue
            # 解析：本句处理完毕

        if len(current) + len(sentence) <= chunk_size:
            # 解析：装得下
            current += sentence
            # 解析：并入当前块
        else:
            # 解析：装不下
            chunks.append(current)
            # 解析：当前块封口
            current = current[-overlap:] + sentence
            # 解析：新块开头带上旧块尾部 overlap 字（保持语义连续）

    if current:
        # 解析：最后剩余的聚合块
        chunks.append(current)
        # 解析：封口
    return [c for c in chunks if c.strip()]
    # 解析：过滤空白块返回


_HEADING = re.compile(
    # 解析：标题行正则
    r"^(?:第[一二三四五六七八九十百\d]+[章节部分]|[一二三四五六七八九十]+、|\d+(?:\.\d+)*[、.]?)\s*.+"
    # 解析：匹配三类标题——"第X章/节/部"、"一、"、"1."、"1.2.3" 开头且后面有内容
)


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """按标题行切分：返回 [(标题, 该节内容)]；无标题时整篇一节（标题为空串）。"""
    lines = text.splitlines()
    # 解析：按行处理
    sections: list[tuple[str, str]] = []
    # 解析：节列表（标题, 内容）
    current_heading = ""
    # 解析：当前节标题
    current_body: list[str] = []
    # 解析：当前节正文行

    for line in lines:
        # 解析：逐行扫描
        stripped = line.strip()
        # 解析：去空白
        # 标题行特征：匹配标题模式且较短（≤ 40 字）
        if stripped and len(stripped) <= 40 and _HEADING.match(stripped):
            # 解析：判定为标题行（非空、短、匹配标题模式）
            if current_body or current_heading:
                # 解析：上一节有内容
                sections.append((current_heading, "\n".join(current_body).strip()))
                # 解析：保存上一节
            current_heading = stripped
            # 解析：开始新节（记标题）
            current_body = []
            # 解析：清空正文
        else:
            # 解析：普通正文行
            current_body.append(line)
            # 解析：并入当前节
    sections.append((current_heading, "\n".join(current_body).strip()))
    # 解析：保存最后一节
    return [(h, b) for h, b in sections if h or b]
    # 解析：过滤空节返回


def chunk_by_headings(text: str, chunk_size: int = 700, overlap: int = 80) -> list[str]:
    """标题分块：按标题切节，节内句子聚合；标题前置注入每个块（带上下文检索）。"""
    chunks: list[str] = []
    # 解析：结果块列表
    for heading, body in split_by_headings(text):
        # 解析：逐节处理
        prefix = f"{heading}\n" if heading else ""
        # 解析：标题前缀（无标题节为空串）
        budget = max(chunk_size - len(prefix), 100)  # 标题占用计入块大小
        # 解析：正文可用大小 = 块大小 - 标题长度（保底 100 防负数）
        for piece in chunk_text(body, chunk_size=budget, overlap=overlap):
            # 解析：节内按句子聚合分块
            chunks.append(prefix + piece)
            # 解析：标题前置注入每块——检索时块自带章节上下文
    return chunks
    # 解析：返回标题分块结果


def parent_child_chunk(
    # 解析：父子块切分
    text: str,
    parent_size: int = 2400,
    # 解析：父块大小（给大模型看的完整上下文）
    child_size: int = 700,
    # 解析：子块大小（向量化检索用）
    overlap: int = 80,
    # 解析：子块间重叠
) -> list[tuple[str, str, list[str]]]:
    """父子块切分：大块切父块 → 父块内切子块。

    返回 [(parent_id, parent_text, [child_text, ...]), ...]；
    parent_id 由父块内容 sha256 生成（同内容稳定唯一）。
    """
    import hashlib
    # 解析：哈希模块（生成稳定 parent_id）

    parents = chunk_text(text, chunk_size=parent_size, overlap=0)
    # 解析：大块切父块（父块间不重叠）
    pairs = []
    # 解析：父子对列表
    for parent in parents:
        # 解析：逐父块
        parent_id = hashlib.sha256(parent.encode("utf-8")).hexdigest()[:16]
        # 解析：父块内容哈希前 16 位作 ID（同内容稳定、全局唯一）
        children = chunk_text(parent, chunk_size=child_size, overlap=overlap)
        # 解析：父块内切子块（带重叠）
        pairs.append((parent_id, parent, children))
        # 解析：记录父子对
    return pairs
    # 解析：返回全部父子对


def _cosine(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    # 解析：点积
    na = sum(x * x for x in a) ** 0.5
    # 解析：a 的模长
    nb = sum(y * y for y in b) ** 0.5
    # 解析：b 的模长
    if na == 0 or nb == 0:
        # 解析：零向量防御
        return 0.0
        # 解析：相似度定义为 0
    return dot / (na * nb)
    # 解析：余弦相似度


def semantic_chunk(
    # 解析：语义分块
    text: str,
    embedder,
    # 解析：向量化模型（BGE-m3）
    max_chunk_size: int = 700,
    # 解析：块大小上限
    threshold: float = 0.5,
    # 解析：相邻句相似度阈值
) -> list[str]:
    """语义分块：逐句向量化，相邻句相似度低于 threshold 处切分，块大小兜底。"""
    sentences = split_sentences(text)
    # 解析：先断句
    if not sentences:
        # 解析：空文本
        return []
        # 解析：返回空列表
    if len(sentences) == 1:
        # 解析：只有一句
        return [sentences[0]]
        # 解析：直接返回单句块
    vectors = embedder.embed_documents(sentences)
    # 解析：逐句 BGE-m3 向量化

    chunks, current, last_vec = [], sentences[0], vectors[0]
    # 解析：结果列表、当前块、上一句向量
    for sent, vec in zip(sentences[1:], vectors[1:]):
        # 解析：从第二句开始逐句判断
        similar = _cosine(last_vec, vec) >= threshold
        # 解析：本句与上一句语义是否连贯
        overflow = len(current) + len(sent) > max_chunk_size
        # 解析：是否超长
        if not similar or overflow:
            # 解析：话题漂移或超长
            chunks.append(current)
            # 解析：封当前块
            current = sent
            # 解析：开始新块
        else:
            # 解析：语义连贯且不超长
            current += sent
            # 解析：并入当前块
        last_vec = vec
        # 解析：更新上一句向量
    if current:
        # 解析：最后剩余块
        chunks.append(current)
        # 解析：封口
    return chunks
    # 解析：返回语义分块结果
