from app.core.document.chunker import Chunker


def test_fixed_chunking_with_no_overlap():
    c = Chunker(chunk_size=10, overlap=0, strategy="fixed")
    chunks = c.chunk("abcdefghijklmnop")
    assert chunks == ["abcdefghij", "klmnop"]


def test_fixed_chunking_with_overlap():
    c = Chunker(chunk_size=10, overlap=4, strategy="fixed")
    chunks = c.chunk("abcdefghijklmnop")
    # step = 6，所有 chunk 长度 <= 10
    assert all(len(x) <= 10 for x in chunks)
    assert chunks[0] == "abcdefghij"


def test_paragraph_merges_small_paragraphs():
    c = Chunker(chunk_size=1000, overlap=0, strategy="paragraph")
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = c.chunk(text)
    assert len(chunks) == 1
    assert "第一段" in chunks[0] and "第三段" in chunks[0]


def test_sentence_strategy():
    c = Chunker(chunk_size=1000, overlap=0, strategy="sentence")
    chunks = c.chunk("句子一。句子二！句子三？")
    assert len(chunks) == 1
    assert "句子一" in chunks[0]


def test_empty_and_blank_text():
    # 必须是 0 块而不是一个空块：空块照样进索引，白占召回名额还污染 prompt。
    assert Chunker().chunk("") == []
    assert Chunker().chunk("   \n\n  ") == []


def test_unknown_strategy_raises():
    # 策略名写错要当场炸：静默退回某种默认策略的话，配置错误只在检索质量上慢慢显形。
    import pytest

    with pytest.raises(ValueError):
        Chunker(strategy="nope")


# ===== chunk_size 是硬上界 =====
#
# chunk_size 会一路影响上下文预算：检索把 TOP_K 条 chunk 整段塞进系统消息，而系统消息
# 不受 HISTORY_MAX_CHARS 约束。参数校验（接口 / CLI）只拦得住"传了多大的 chunk_size"，
# 拦不住分块器自己产出超长 chunk —— 下面两条就是后者。


def test_oversized_paragraph_is_split():
    """单独超过 chunk_size 的段落必须切开。

    PDF 解析出来的长文常常整节没有空行，一段就是几千字：不切的话它就是**一个** chunk，
    chunk_size 完全不生效。实测 3600 字的单段落原样产出 3800 字的 chunk。
    """
    text = "高血压患者的血压应当控制在合理范围内。" * 200  # 3600 字，无空行
    chunks = Chunker(chunk_size=1000, overlap=50).chunk(text)
    assert max(len(c) for c in chunks) <= 1000, f"仍有超长 chunk：{[len(c) for c in chunks]}"


def test_overlap_larger_than_chunk_size_does_not_inflate_chunks():
    """overlap >= chunk_size 时必须收口。

    重叠是把上一个 chunk 的尾部接到下一个前面，所以 overlap 比 chunk_size 还大时
    chunk 会被越撑越大——实测 chunk_size=50 / overlap=500 产出 536 字的 chunk。
    """
    text = "\n\n".join(f"第{i}段内容，讲的是限盐与随访。" * 5 for i in range(40))
    for overlap in (50, 500, 1000):
        chunks = Chunker(chunk_size=50, overlap=overlap).chunk(text)
        assert max(len(c) for c in chunks) <= 50, f"overlap={overlap} 撑出超长 chunk"


def test_every_chunk_respects_chunk_size():
    """把各种策略和各种参数组合扫一遍，保证上界处处成立。"""
    text = "\n\n".join(f"第{i}段。{'内容' * (i % 7 + 1)}" for i in range(60))
    for strategy in ("paragraph", "sentence", "fixed"):
        for chunk_size in (10, 50, 200):
            for overlap in (0, 9, 200, 400):
                chunks = Chunker(chunk_size=chunk_size, overlap=overlap, strategy=strategy).chunk(text)
                assert max(len(c) for c in chunks) <= chunk_size, (
                    f"{strategy}/chunk_size={chunk_size}/overlap={overlap}: "
                    f"最大 {max(len(c) for c in chunks)}"
                )


def test_splitting_oversized_paragraph_keeps_content():
    """硬切是为了收口，不是丢内容。"""
    text = "".join(f"第{i}节：限盐、运动、随访。" for i in range(200))
    chunks = Chunker(chunk_size=100, overlap=0).chunk(text)
    joined = "".join(chunks).replace("\n", "")
    assert joined == text.replace("\n", "")
