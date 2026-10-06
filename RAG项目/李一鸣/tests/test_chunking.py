from app.rag.chunking import ChunkingConfig, DocumentChunker
from app.rag.types import ParsedDocument


def test_chunker_keeps_content_and_metadata():
    text = "第一章\n\n" + "高血压治疗需要结合风险分层、生活方式管理和规范随访。" * 30
    chunks = DocumentChunker(ChunkingConfig(chunk_size=100, overlap=20)).split(
        ParsedDocument(text=text, pages=[text]),
        document_id="doc-1",
        source="guide.txt",
        base_metadata={"role_id": "global"},
    )
    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.metadata["role_id"] == "global" for chunk in chunks)
    assert "高血压" in "".join(chunk.text for chunk in chunks)
