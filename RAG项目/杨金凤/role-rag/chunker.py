"""纯文本分块层：空白归一、父子分块、清洗、摘要。

从 ingest.py 迁出（ingest.py 超 300 行），供 ingest.py 经 `from chunker import ...` 复用。
父子分块：split_parents 把一段文本切成 ~1500 字父块，再对每个父块用 split_chunks 切成 ~400 字子块，
子块冗余记录 parent_content（父块全文）供检索时替换。
"""
from __future__ import annotations

import hashlib
import re

CHUNK_SIZE = 400        # 子块目标大小（中文字符）
CHUNK_OVERLAP = 60      # 相邻子块重叠字符数
PARENT_SIZE = 1500      # 父块目标大小（中文字符）
PARENT_HARD_CUT = 800   # 超长单段硬切大小


def normalize_whitespace(text: str) -> str:#归一化空白字符
    """压缩空白、合并连续换行，去掉 PDF 抽取常见的断裂空格。"""
    text = text.replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def split_parents(text: str, size: int = PARENT_SIZE) -> list[str]:#1500 字父块切分
    """把一段文本切成 ~size 字父块。

    优先按空行（\\n\\n）切段落；parser 输出无空行时按单 \\n 行累加。
    段落累加到接近 size 就切一块；超长单段（> size）按 PARENT_HARD_CUT 硬切。
    必须在 normalize_whitespace 之前调用，否则空行边界会被压成单换行。
    """
    text = text.strip()
    if not text:
        return []
    if "\n\n" in text:
        units = [u.strip() for u in re.split(r"\n\n+", text) if u.strip()]
        sep = "\n\n"
    else:
        units = [u.strip() for u in text.split("\n") if u.strip()]
        sep = "\n"

    parents: list[str] = []
    current = ""
    for u in units:
        if len(u) > size:  # 超长单段硬切
            if current:
                parents.append(current)
                current = ""
            for i in range(0, len(u), PARENT_HARD_CUT):
                parents.append(u[i:i + PARENT_HARD_CUT])
            continue
        if current and len(current) + len(sep) + len(u) > size:
            parents.append(current)
            current = u
        else:
            current = (current + sep + u) if current else u
    if current:
        parents.append(current)
    return parents


def split_chunks(text: str, page_no: int, start: int = 0) -> list[dict]:#400 字/60 重叠分块（子块）
    """把一段文本切成带重叠的块，返回 [{"id","content","page"}, ...]。

    start 为页内子块起始序号，供同一页多次调用（每个父块一次）时保证 id 全局唯一。
    """
    sentences = re.split(r"(?<=[。；！？!?])|\n", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > CHUNK_SIZE:  # 单句超长则硬切
            if current:
                chunks.append(current)
                current = ""
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for i in range(0, len(sent), step):
                chunks.append(sent[i:i + CHUNK_SIZE])
            continue
        if len(current) + len(sent) > CHUNK_SIZE:
            chunks.append(current)
            current = sent
        else:
            current = (current + sent) if current else sent
    if current:
        chunks.append(current)

    return [
        {"id": f"p{page_no}_c{i + start}", "content": c, "page": page_no}
        for i, c in enumerate(chunks)
    ]


MIN_CHUNK_LEN = 20           # 低于该长度视为低质量
_VALID_CHAR_RE = re.compile(r"[一-鿿A-Za-z0-9]")
_GARBLE_THRESHOLD = 0.5      # 有效字符占比低于该值视为乱码


def clean_chunks(chunks: list[dict]) -> tuple[list[dict], dict]:#MD5 去重 + 删低质量
    """清洗 chunk：按 content 的 MD5 去重 + 丢弃低质量（过短/纯符号/乱码）。

    返回 (清洗后 chunks, 统计 dict)，并日志打印清洗前后数量。
    """
    seen: set[str] = set()
    cleaned: list[dict] = []
    stats = {"before": len(chunks), "dedup_removed": 0, "low_quality_removed": 0}

    for c in chunks:
        content = c["content"].strip()
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()
        if digest in seen:
            stats["dedup_removed"] += 1
            continue
        seen.add(digest)
        if _is_low_quality(content):
            stats["low_quality_removed"] += 1
            continue
        cleaned.append(c)

    stats["after"] = len(cleaned)
    return cleaned, stats


def _is_low_quality(text: str) -> bool:
    """过短、纯符号（无任何 CJK/字母/数字）、乱码（替换符或有效字符占比过低）。"""
    if len(text) < MIN_CHUNK_LEN:
        return True
    if not _VALID_CHAR_RE.search(text):
        return True  # 纯符号/空白，无实际内容
    if "�" in text:
        return True  # 含替换符，抽取乱码
    if len(_VALID_CHAR_RE.findall(text)) / len(text) < _GARBLE_THRESHOLD:
        return True
    return False


def make_summary(content: str) -> str:#取前 100 字做摘要
    """取 content 前 100 字作为摘要（不调 LLM）。"""
    return content[:100]
