"""父子切分：条=父块、款/项=子块、超长切叶块。

存在的理由（技术方案 4.1）：子块负责被找到，父块负责被理解。
overlap=0——法律文本条款边界天然清晰，重叠只会造成同款被重复引用（FR-2.4）。
父块与子块同集合存放，用 chunk_type 区分，回填时按 parent_id 取。
不用 LangChain 的 ParentDocumentRetriever：它默认要额外配 docstore，会让父块二次落库
造成父子不一致（技术方案 2.2 摩擦点 3）。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.ingest.structure import Article

# 技术方案 4.1 定的阈值
MAX_ARTICLE_CHARS = 800
MAX_PARAGRAPH_CHARS = 500

# 项标记：（一）（二）… 或 (一)(二)…
ITEM_PATTERN = re.compile(r"^[（(][一二三四五六七八九十]+[）)]")

# 句末标点，用于超长文本按语义边界切分
SENTENCE_END = re.compile(r"(?<=[。；])")


@dataclass
class Chunk:
    """一个可检索块。chunk_type 取值见技术方案 4.2。"""
    chunk_id: str
    parent_id: str | None
    chunk_type: str
    text: str
    article_no: int
    paragraph_no: int | None
    path: str


def _make_id(article_no: int, suffix: str) -> str:
    """块 id 用条号+suffix 的哈希。

    用哈希而不是自增列，是为了让 id 在重跑入库时保持稳定——增量 upsert 依赖这一点，
    自增 id 每次重跑都会变，会导致重复写入而不是覆盖。
    """
    raw = f"law_article:{article_no}:{suffix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _split_long(text: str, limit: int) -> list[str]:
    """按句号/分号切超长文本，尽量不破坏语义边界。"""
    pieces: list[str] = []
    buffer = ""
    for sentence in SENTENCE_END.split(text):
        if buffer and len(buffer) + len(sentence) > limit:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer += sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def build_chunks(articles: list[Article]) -> list[Chunk]:
    """把条列表切成父子块。顺序为父块紧跟其子块，便于人工核对。"""
    chunks: list[Chunk] = []
    for art in articles:
        father_id = _make_id(art.number, "father")
        chunks.append(Chunk(chunk_id=father_id, parent_id=None, chunk_type="father",
                            text=art.text, article_no=art.number,
                            paragraph_no=None, path=art.path))
        oversized = len(art.text) > MAX_ARTICLE_CHARS
        for para_no, para in enumerate(art.paragraphs, start=1):
            if oversized or len(para) > MAX_PARAGRAPH_CHARS:
                # 超长：切叶块，每个叶块仍回指同一个父块
                for leaf_no, leaf in enumerate(_split_long(para, MAX_PARAGRAPH_CHARS), start=1):
                    chunks.append(Chunk(
                        chunk_id=_make_id(art.number, f"leaf:{para_no}:{leaf_no}"),
                        parent_id=father_id, chunk_type="leaf", text=leaf,
                        article_no=art.number, paragraph_no=para_no, path=art.path))
                continue
            is_item = bool(ITEM_PATTERN.match(para.strip()))
            kind = "item" if is_item else "paragraph"
            chunks.append(Chunk(
                chunk_id=_make_id(art.number, f"{kind}:{para_no}"),
                parent_id=father_id, chunk_type=kind, text=para,
                article_no=art.number, paragraph_no=para_no, path=art.path))
    return chunks
