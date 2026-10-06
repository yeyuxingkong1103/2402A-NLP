"""离线测试：分块（``app/core/chunker.py``）。

测试目标（工单 9.1 / 5.2）：
1. 分块数量 > 0，且 chunk_id 全局唯一；
2. 每个 chunk 都带齐 ``chunk_id / page / section / type / content`` 元数据；
3. 文本 chunk 长度落在配置区间（``min_chunk_chars`` ≤ 长度 ≤ ``chunk_size``）；
4. 表格 chunk 的 ``type == "table"``，且带 ``table_id``、整表不切分；
5. 关键词标注只包含领域词典内的词；
6. 分块结果可落盘为 JSONL 并回读；
7. 空文档不得抛异常，应返回空列表。
"""

from __future__ import annotations

import json
import re

import pytest

from app.core.chunker import DOMAIN_KEYWORDS, Chunker
from app.models.schemas import Chunk, ParsedDocument

CHUNK_ID_PATTERN = re.compile(r"^c\d{6}$")


@pytest.fixture(scope="module")
def chunks(sample_document):
    """前 30 页（含表格）的分块结果，本模块内复用。"""
    return Chunker().split(sample_document)


# --------------------------------------------------------------------------
# 基本形态
# --------------------------------------------------------------------------
def test_split_returns_chunks(chunks) -> None:
    """前 30 页必须切出足量分块，否则检索没有可召回的单元。"""
    assert len(chunks) > 0, "分块结果为空：PDF 可能没有可提取文本（扫描件需先做 OCR）"
    assert len(chunks) >= 30, f"前 30 页仅切出 {len(chunks)} 个分块，分块粒度可能过粗"


def test_chunk_ids_unique_and_well_formed(chunks) -> None:
    """chunk_id 必须形如 c000123 且全局唯一（引用与落库都依赖它）。"""
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(set(ids)) == len(ids), "存在重复的 chunk_id，会导致引用与落库串号"
    for chunk_id in ids:
        assert CHUNK_ID_PATTERN.match(chunk_id), f"chunk_id 格式应为 c000123，实际为 {chunk_id!r}"


def test_every_chunk_has_required_metadata(chunks, sample_document) -> None:
    """每个 chunk 都必须齐备 chunk_id/page/section/type/content 五项元数据。"""
    for chunk in chunks:
        assert isinstance(chunk, Chunk), "split 返回的元素必须是 Chunk 模型"
        assert chunk.chunk_id, "存在缺少 chunk_id 的分块"
        assert chunk.doc_id == sample_document.doc_id, f"{chunk.chunk_id} 的 doc_id 与文档不一致"
        assert isinstance(chunk.page, int) and 1 <= chunk.page <= sample_document.page_count, (
            f"{chunk.chunk_id} 的页码 {chunk.page} 超出 1..{sample_document.page_count}"
        )
        assert isinstance(chunk.section, str), f"{chunk.chunk_id} 的 section 必须是字符串"
        assert chunk.type in {"text", "table"}, f"{chunk.chunk_id} 的 type 非法：{chunk.type!r}"
        assert chunk.content.strip(), f"{chunk.chunk_id} 的 content 为空"
        assert chunk.char_count == len(chunk.content), (
            f"{chunk.chunk_id} 的 char_count({chunk.char_count}) 与 content 长度({len(chunk.content)}) 不一致"
        )


# --------------------------------------------------------------------------
# 长度区间
# --------------------------------------------------------------------------
def test_text_chunk_length_within_configured_range(chunks, settings) -> None:
    """文本 chunk 长度必须落在配置区间（chunk_size=700，min_chunk_chars=30）。"""
    min_chars = settings.chunk.min_chunk_chars
    max_chars = settings.chunk.chunk_size
    text_chunks = [chunk for chunk in chunks if chunk.type == "text"]
    assert text_chunks, "没有生成任何文本分块"

    for chunk in text_chunks:
        assert len(chunk.content) >= min_chars, (
            f"{chunk.chunk_id} 长度 {len(chunk.content)} 小于最小分块长度 {min_chars}，噪声片段不应入库"
        )
        assert len(chunk.content) <= max_chars, (
            f"{chunk.chunk_id} 长度 {len(chunk.content)} 超过 chunk_size={max_chars}，滑窗切分失效"
        )


def test_long_page_is_split_into_multiple_chunks(chunks) -> None:
    """同一页的正文若超过 chunk_size，必须被切成多个 chunk（滑窗生效）。"""
    by_page: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk.page, []).append(chunk)
    multi = [page for page, items in by_page.items() if len(items) > 1]
    assert multi, "前 30 页没有任何一页被切分成多个 chunk，滑窗切分可能未生效"


# --------------------------------------------------------------------------
# 表格分块
# --------------------------------------------------------------------------
def test_table_chunks_are_typed_as_table(chunks, sample_document) -> None:
    """表格必须整表成块，type=='table' 且带 table_id。"""
    table_chunks = [chunk for chunk in chunks if chunk.type == "table"]
    assert table_chunks, "前 30 页存在表格，但没有生成任何 type=='table' 的分块"
    assert len(table_chunks) == len(sample_document.tables), (
        f"表格分块数({len(table_chunks)})与解析到的表格数({len(sample_document.tables)})不一致，整表成块策略被破坏"
    )
    for chunk in table_chunks:
        assert chunk.table_id, f"表格分块 {chunk.chunk_id} 缺少 table_id"
        assert chunk.table_id in chunk.content, f"表格分块 {chunk.chunk_id} 的正文中未包含表格 ID"
        assert "|" in chunk.content, f"表格分块 {chunk.chunk_id} 未保留 Markdown 表格结构"


def test_table_chunk_is_not_split(chunks) -> None:
    """整表不切分：表格 chunk 内部必须包含完整的分隔行（表头结构完整）。"""
    table_chunks = [chunk for chunk in chunks if chunk.type == "table"]
    for chunk in table_chunks:
        assert "| --- |" in chunk.content, (
            f"表格分块 {chunk.chunk_id} 缺少 Markdown 分隔行，表头结构被破坏"
        )


# --------------------------------------------------------------------------
# 章节与关键词
# --------------------------------------------------------------------------
def test_section_metadata_is_populated(chunks) -> None:
    """章节路径应尽量被填充，供引用展示与按章节检索使用。"""
    with_section = [chunk for chunk in chunks if chunk.section.strip()]
    assert len(with_section) >= len(chunks) // 2, (
        f"仅 {len(with_section)}/{len(chunks)} 个分块带章节信息，标题切分可能失效"
    )


def test_keywords_come_from_domain_dictionary(chunks) -> None:
    """chunk.keywords 只能来自领域词典，且必须真实出现在正文中。"""
    tagger = Chunker()
    for chunk in chunks:
        for keyword in chunk.keywords:
            assert keyword in DOMAIN_KEYWORDS, f"{chunk.chunk_id} 标注了词典外的关键词：{keyword!r}"
            assert keyword in chunk.content, f"{chunk.chunk_id} 标注了未出现在正文中的关键词：{keyword!r}"
    # 招股书正文里必然出现"公司/注册资本"等高频领域词，至少要有分块被打上标签
    tagged = [chunk for chunk in chunks if chunk.keywords]
    assert tagged, "没有任何分块命中领域关键词，关键词标注逻辑可能未生效"
    assert tagger._tag_keywords("公司注册资本为 5,520 万元") == ["注册资本"], "关键词标注结果不符合预期"


# --------------------------------------------------------------------------
# 落盘与边界
# --------------------------------------------------------------------------
def test_save_jsonl_and_read_back(chunks, tmp_path) -> None:
    """分块结果必须能落盘为 JSONL 并原样回读（人工核对与离线测试用）。"""
    target = Chunker().save(chunks, output_dir=tmp_path)
    assert target.exists(), f"分块结果文件未生成：{target}"

    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(chunks), f"落盘行数({len(lines)})与分块数({len(chunks)})不一致"

    restored = Chunk(**json.loads(lines[0]))
    assert restored.chunk_id == chunks[0].chunk_id, "回读后的 chunk_id 与原分块不一致"
    assert restored.content == chunks[0].content, "回读后的正文与原分块不一致"


def test_split_empty_document_returns_empty_list() -> None:
    """空文档不得抛异常，应返回空列表（由上层决定如何提示用户）。"""
    document = ParsedDocument(doc_id="doc_empty", source_path="", page_count=0, pages=[], tables=[])
    assert Chunker().split(document) == [], "空文档应返回空分块列表，而不是抛异常或返回占位分块"
