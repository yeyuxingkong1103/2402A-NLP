"""文本分块单元测试"""
from src.rag.text_chunking import (
    get_chunker, FixedLengthChunker, SentenceChunker,
    ParagraphChunker, SemanticChunker,
)


def test_fixed_length_chunker():
    text = "第一句。第二句。第三句。第四句。"
    chunks = FixedLengthChunker(chunk_size=8, chunk_overlap=0).chunk(text)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)


def test_sentence_chunker():
    text = "第一句。第二句。第三句！"
    chunks = SentenceChunker().chunk(text)
    assert len(chunks) == 3


def test_paragraph_chunker():
    text = "段落一\n\n段落二\n\n段落三"
    chunks = ParagraphChunker().chunk(text)
    assert len(chunks) == 3


def test_semantic_chunker_splits_large():
    text = "## 标题\n" + "句子。" * 500
    chunks = SemanticChunker(chunk_size=100, chunk_overlap=0).chunk(text)
    assert len(chunks) > 1


def test_get_chunker_fallback():
    assert isinstance(get_chunker("unknown_method"), SemanticChunker)
    assert isinstance(get_chunker("fixed"), FixedLengthChunker)
