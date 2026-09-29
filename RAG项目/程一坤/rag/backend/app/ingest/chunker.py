"""法规切块主流程：把解析后的全文切成「条（父块）+ 款/项（子块）」两级块。

职责拆分（本文件只保留块的数据结构与组装主流程）：
- 条（一级）边界切分 → app/ingest/article_splitter.py
- 款/项（二三级）识别 → app/ingest/paragraph_items.py
- 文本工具（归一化/检索文本/文档标识）→ app/ingest/chunk_text.py
- 切块指纹与版本（增量判定）→ app/ingest/chunk_fingerprint.py
"""
from dataclasses import dataclass
from pathlib import Path

from app.ingest.article_splitter import _split_articles
from app.ingest.chunk_text import _build_retrieval_content, _document_key, _normalize_text
from app.ingest.parser import ParsedDocument
from app.ingest.paragraph_items import _extract_items, _split_into_paragraphs


# 保存一个法律父块或子块
@dataclass(frozen=True)
class DocumentChunk:
    # 保存块的唯一标识
    chunk_id: str

    # 保存父块标识，父块自身为空
    parent_id: str | None

    # 保存块层级：parent 或 child
    chunk_level: str

    # 保存原文内容
    content: str

    # 保存用于检索的上下文增强内容
    retrieval_content: str

    # 保存法律条文号
    article_no: str | None

    # 保存款号（条内自然段序号，从 1 起；父块与项子块存所属款号以外的说明见各构造处）
    paragraph_no: str | None

    # 保存项号（款内「（一）（二）」分项的序号；无项为空）
    item_no: str | None

    # 保存块顺序
    sequence: int

    # 保存来源文件路径
    source_path: Path

    # 保存输入内容格式
    content_format: str


# 按法律条文生成父块和子块
def chunk_document(
    document: ParsedDocument,
    document_title: str = "",
    document_id: str | None = None,
) -> list[DocumentChunk]:
    """把解析后的法规全文切成「条（父块）+ 款/项（子块）」两级块。

    三级语义（docs/CONTEXT.md）：
    - 条：第 X 条，父块，正文保留段落换行；
    - 款：条内无编号自然段，paragraph_no 从 1 依次编号；
    - 项：款内的「（一）（二）」分项，item_no 按款内顺序编号。
    """
    # 只去除文档首尾空白，保留条文内部段落边界
    content = document.content.strip()

    # 空文档不生成无意义块
    if not content:
        return []

    # 按法律条文编号切出独立父块，并保留内部段落边界
    article_parts = _split_articles(content)

    # 没有条文号时，将全文作为一个文档级父块
    if not article_parts:
        article_parts = [(None, content)]

    # 保存所有父块和子块
    chunks: list[DocumentChunk] = []
    sequence = 1
    document_key = _document_key(document.source_path, document_id)

    # 逐个生成父块及其子块
    for article_index, (article_no, article_content) in enumerate(
        article_parts,
        start=1,
    ):
        # 清理父块文本（保留段落换行，只压行内空白）
        normalized_article_content = _normalize_text(article_content)

        # 使用文档标识和条文序号创建稳定块 ID
        parent_id = f"{document_key}-a{article_index}"

        # 保存完整条文作为父块
        parent_chunk = DocumentChunk(
            chunk_id=parent_id,
            parent_id=None,
            chunk_level="parent",
            content=normalized_article_content,
            retrieval_content=_build_retrieval_content(
                document_title,
                article_no,
                normalized_article_content,
            ),
            article_no=article_no,
            paragraph_no=None,
            item_no=None,
            sequence=sequence,
            source_path=document.source_path,
            content_format=document.content_format,
        )
        chunks.append(parent_chunk)
        sequence += 1

        # 条内自然段即款；款内「（一）」行归属当前款
        paragraphs = _split_into_paragraphs(article_content)

        # 每条独立编号子块，保证同条内 chunk_id 不重复
        child_counter = 0

        # 逐款生成检索子块
        for kuan_index, paragraph_content in enumerate(paragraphs, start=1):
            # 款内识别项（（一）（二）…）；无项返回 None
            items = _extract_items(paragraph_content)

            # 无项：整款一个子块，款号即条内自然段序号
            if items is None:
                normalized_content = _normalize_text(paragraph_content)
                child_counter += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{parent_id}-c{child_counter}",
                        parent_id=parent_id,
                        chunk_level="child",
                        content=normalized_content,
                        retrieval_content=_build_retrieval_content(
                            document_title,
                            article_no,
                            normalized_content,
                        ),
                        article_no=article_no,
                        paragraph_no=str(kuan_index),
                        item_no=None,
                        sequence=sequence,
                        source_path=document.source_path,
                        content_format=document.content_format,
                    )
                )
                sequence += 1
                continue

            # 有项：项为子块（引导段不单独成子块，仅保留在父块正文中）
            for item_no, item_content in items:
                child_counter += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{parent_id}-c{child_counter}",
                        parent_id=parent_id,
                        chunk_level="child",
                        content=item_content,
                        retrieval_content=_build_retrieval_content(
                            document_title,
                            article_no,
                            item_content,
                        ),
                        article_no=article_no,
                        paragraph_no=str(kuan_index),
                        item_no=item_no,
                        sequence=sequence,
                        source_path=document.source_path,
                        content_format=document.content_format,
                    )
                )
                sequence += 1

    # 返回父块和子块组成的完整结果
    return chunks
