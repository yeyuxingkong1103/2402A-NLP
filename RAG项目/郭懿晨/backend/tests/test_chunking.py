import pytest

from backend.app.chunking import build_chunks, clean_blocks, clean_text
from backend.app.mineru import MinerUBlock


def test_clean_text_removes_extra_space():
    assert clean_text(" 第一行\n\n\n 第二行 ") == "第一行\n第二行"


def test_clean_blocks_drops_empty_and_trims_text():
    blocks = [
        MinerUBlock(page=1, label="正文", text=" 首块  ", source_span="page=1:block=1"),
        MinerUBlock(page=1, label="正文", text="\n\n", source_span="page=1:block=2"),
    ]

    cleaned = clean_blocks(blocks)

    assert len(cleaned) == 1
    assert cleaned[0].text == "首块"
    assert cleaned[0].source_span == "page=1:block=1"


def test_build_chunks_keeps_page_and_label():
    blocks = [
        MinerUBlock(page=2, label="正文", text="第一段内容。", source_span="page=2:block=1"),
        MinerUBlock(page=2, label="表格", text="表格内容。", source_span="page=2:block=2"),
    ]

    chunks = build_chunks(document_id="doc-1", blocks=blocks, max_chars=100)

    assert [chunk.category for chunk in chunks] == ["正文", "表格"]
    assert [chunk.page for chunk in chunks] == [2, 2]


def test_build_chunks_rejects_missing_page():
    blocks = [MinerUBlock(page=0, label="正文", text="内容", source_span="page=0:block=1")]

    with pytest.raises(ValueError, match="页码"):
        build_chunks(document_id="doc-1", blocks=blocks, max_chars=100)
