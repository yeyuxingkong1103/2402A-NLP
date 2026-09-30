"""分块测试：标题路径、表格完整性、重叠、编号连续。"""

from __future__ import annotations

from role_rag.ingest.chunking import chunk_document, chunk_summary
from role_rag.ingest.loaders import LoadedDoc


def make_doc(text: str, title: str = "测试文档") -> LoadedDoc:
    return LoadedDoc(doc_id="doc1", title=title, text=text, source="mem://test",
                     role="scientist", scope="scientist")


def test_short_document_single_chunk():
    chunks = chunk_document(make_doc("# 标题\n\n这是一段足够长的正文内容，用于验证分块结果。" * 2),
                            chunk_size=700, chunk_overlap=120, min_chunk_chars=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].section == "标题"
    assert chunks[0].chunk_id == "doc1#0"


def test_sections_split_by_heading():
    text = "# 一\n\n" + "甲" * 60 + "\n\n## 二\n\n" + "乙" * 60
    chunks = chunk_document(make_doc(text), chunk_size=50, chunk_overlap=0, min_chunk_chars=10)
    sections = [chunk.section for chunk in chunks]
    assert "一" in sections and "一 › 二" in sections


def test_table_kept_atomic():
    table = "| 情形 | 处理 |\n|---|---|\n| 未约定利息 | 视为无息 |\n| 约定不明 | 视为无息 |" * 3
    chunks = chunk_document(make_doc("# 表\n\n" + table), chunk_size=120, chunk_overlap=0,
                            min_chunk_chars=10, max_chunk_chars=4000)
    assert any("| 未约定利息 |" in chunk.text for chunk in chunks)
    # 表格行不应被拆成孤立片段：包含表头与数据行的块内换行结构完整
    for chunk in chunks:
        if "未约定利息" in chunk.text:
            assert "| 情形 | 处理 |" in chunk.text or "| 约定不明 |" in chunk.text


def test_overlap_present_between_neighbours():
    paragraphs = [f"第{i}段内容，用于验证相邻知识块之间的重叠是否生效。" + "补" * 60 for i in range(4)]
    text = "# 一\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_document(make_doc(text), chunk_size=130, chunk_overlap=40, min_chunk_chars=10)
    assert len(chunks) >= 2
    # 上一块的尾部片段应出现在下一块的开头
    assert any(
        chunks[index].text[-12:] and chunks[index].text[-12:] in chunks[index + 1].text
        for index in range(len(chunks) - 1)
    )


def test_single_paragraph_not_split_by_chunk_size():
    """单段略超 chunk_size 时保留完整段落，只在超过 max_chunk_chars 时才二次切分。"""

    paragraph = "这是一个完整的段落，" * 8       # 约 80 字
    chunks = chunk_document(make_doc("# 一\n\n" + paragraph), chunk_size=60, chunk_overlap=0,
                            min_chunk_chars=10, max_chunk_chars=400)
    assert len(chunks) == 1
    assert chunks[0].text == paragraph


def test_oversized_paragraph_split_by_sentence():
    long_paragraph = "。".join(["超过上限的段落需要按句子切分" * 4] * 12) + "。"
    chunks = chunk_document(make_doc("# 一\n\n" + long_paragraph), chunk_size=200,
                            chunk_overlap=0, min_chunk_chars=10, max_chunk_chars=300)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_chunk_index_is_continuous():
    text = "\n\n".join(["# 一"] + ["段落" + str(i) * 40 for i in range(6)])
    chunks = chunk_document(make_doc(text), chunk_size=120, chunk_overlap=20, min_chunk_chars=10)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_long_paragraph_split_by_sentence():
    text = "# 一\n\n" + "。".join(["这是一个很长的句子用于测试按句切分" * 3] * 20) + "。"
    chunks = chunk_document(make_doc(text), chunk_size=200, chunk_overlap=0,
                            min_chunk_chars=10, max_chunk_chars=300)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 320 for chunk in chunks)


def test_chunk_summary_shape():
    chunks = chunk_document(make_doc("# 一\n\n" + "内容" * 100), chunk_size=100, min_chunk_chars=10)
    summary = chunk_summary(chunks)
    assert summary["chunks"] == len(chunks)
    assert summary["max_chars"] >= summary["min_chars"]


def test_min_chunk_merged_or_dropped():
    text = "# 一\n\n" + "甲" * 200 + "\n\n乙"  # 尾块只有一个字
    chunks = chunk_document(make_doc(text), chunk_size=100, chunk_overlap=0, min_chunk_chars=50)
    assert all(len(chunk.text) >= 50 for chunk in chunks)
