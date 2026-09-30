"""入库侧的三个增强步骤：去重指纹、低质量过滤、摘要。

都是 ingest / 上传接口的前置环节，所以这里的**失败语义**比返回值更要紧：
指纹失真会放过重复数据，摘要少一项会让后面的 chunk 拿到别人的摘要。
"""
from app.core.document.enhance import chunk_hash, filter_low_quality, summarize_chunks


def test_chunk_hash_normalizes_whitespace():
    # 指纹必须对空白不敏感：同一段文本换个换行重传就绕过去重，库里会留两份。
    assert chunk_hash("a  b\nc") == chunk_hash("a b c")
    assert chunk_hash("x") != chunk_hash("y")


def test_filter_low_quality():
    # 长度按去空白后算：全空格的块不能靠凑空格混进库。
    assert filter_low_quality(["abc", "a", "     ", "abcdef"], 3) == ["abc", "abcdef"]


def test_summarize_chunks_uses_llm():
    # 一一对应：调用方按位置取摘要（summaries[i]），长度必须与入参相等。
    class _LLM:
        def chat(self, messages):
            return "一句话摘要"

    assert summarize_chunks(_LLM(), ["t1", "t2"]) == ["一句话摘要", "一句话摘要"]


def test_summarize_chunks_llm_failure_keeps_empty():
    # 失败要留空串占位而不是跳过这一项——少一项会让后面的 chunk 索引错位、拿到别人的摘要。
    class _Boom:
        def chat(self, messages):
            raise RuntimeError("boom")

    assert summarize_chunks(_Boom(), ["t1"]) == [""]
