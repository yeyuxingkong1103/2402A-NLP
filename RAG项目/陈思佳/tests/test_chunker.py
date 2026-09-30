from src.edu_rag_ingest.ingestion.chunker import TextChunker
from src.edu_rag_ingest.config.config import ChunkingConfig


def test_chunker_splits_text_into_chunks(tmp_path):
    config = ChunkingConfig(
        chunk_size=12,
        chunk_overlap=5,
        min_chunk_chars=5,
        output_path=tmp_path / "chunks.jsonl",
    )
    chunker = TextChunker(config)

    chunks = chunker.split("doc-1", "第一段内容很长。\n\n第二段内容也很长。\n\n第三段内容。", {"subject": "语文"})

    assert len(chunks) >= 2
    assert chunks[0].document_id == "doc-1"


def test_chunker_adds_heading_metadata(tmp_path):
    config = ChunkingConfig(
        chunk_size=200,
        chunk_overlap=0,
        min_chunk_chars=5,
        output_path=tmp_path / "chunks.jsonl",
    )
    chunker = TextChunker(config)

    text = "## 三、课程目标\n\n### 第四学段（7～9年级）\n\n阅读与鉴赏要求。"
    chunks = chunker.split("doc-1", text, {"subject": "语文", "grade": "九年级"})

    assert chunks[-1].metadata["section"] == "三、课程目标 > 第四学段（7～9年级）"
    assert chunks[-1].metadata["topic"] == "第四学段（7～9年级）"
    assert chunks[-1].metadata["stage"] == "第四学段（7～9年级）"
