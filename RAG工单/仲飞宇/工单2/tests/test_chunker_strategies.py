from app.core.document.chunker import Chunker


def test_title_chunks_carry_heading():
    # 标题拼进每个 chunk 是为了保住章节语境：单独看「限盐 5 克」不知道说的是哪一节。
    text = "# 饮食\n限盐 5 克。\n\n# 运动\n每周 150 分钟。"
    chunks = Chunker(chunk_size=1000, overlap=0, strategy="title").chunk(text)
    assert len(chunks) == 2
    assert "饮食" in chunks[0] and "限盐" in chunks[0]
    assert "运动" in chunks[1] and "分钟" in chunks[1]


def test_title_strategy_respects_chunk_size():
    text = "# 一\n" + "长内容" * 200 + "\n\n# 二\n" + "长内容" * 200
    chunks = Chunker(chunk_size=100, overlap=0, strategy="title").chunk(text)
    # 严格不超过上界：以前这里留了「+10 余量」的宽限，恰好把「标题拼在切完之后、
    # 于是整块超出 chunk_size」这个 bug 放了过去（chunk_size 是 prompt 预算的硬约束）。
    assert all(len(c) <= 100 for c in chunks)


def test_title_with_long_heading_still_respects_chunk_size():
    """长标题不能把 chunk 撑破上界——实测 chunk_size=1000 + 300 字标题曾切出 1303 字符。

    PDF 转 markdown 时标题行常常很长（整行无换行），而标题是在切完之后拼到每个
    chunk 前面的，所以必须先把标题的位置留出来。
    """
    long_heading = "# " + "标题" * 150  # 302 字
    doc = long_heading + "\n" + ("正文内容。" * 300)  # 正文无空行，会被当一整块硬切
    for overlap in (0, 50):
        chunks = Chunker(chunk_size=1000, overlap=overlap, strategy="title").chunk(doc)
        assert chunks, "不该切出空结果"
        assert max(len(c) for c in chunks) <= 1000, f"overlap={overlap} 时超了上界"


def test_title_longer_than_budget_is_truncated():
    """标题长到能独自吃光预算时，截断标题而不是让正文没地方放。"""
    doc = "# " + "超长标题" * 400 + "\n" + ("正文内容。" * 200)
    chunks = Chunker(chunk_size=600, overlap=50, strategy="title").chunk(doc)

    assert max(len(c) for c in chunks) <= 600
    assert all(c.startswith("#") for c in chunks), "截断后仍要保留标题前缀（不然丢了章节语境）"


def test_semantic_without_embed_falls_back_to_paragraph():
    text = "段落一。\n\n段落二。"
    chunks = Chunker(chunk_size=1000, strategy="semantic").chunk(text)
    assert len(chunks) == 1  # 退化段落分块：两段被合并


def test_semantic_splits_on_topic_shift():
    sents = ["苹果很好吃很甜。", "香蕉也很好吃很香。", "股票市场波动很大。", "债券收益相对稳定。"]

    def fake_embed(texts):
        def vec(s):
            return [1.0, 0.0, 0.0] if ("苹果" in s or "香蕉" in s) else [0.0, 1.0, 0.0]

        return [vec(t) for t in texts]

    # chunk_size 够小，话题断点两侧的句子才会分成两个 chunk（否则会被合并回一个）
    chunks = Chunker(chunk_size=20, strategy="semantic").chunk("\n".join(sents), embed=fake_embed)
    assert len(chunks) == 2
    assert "苹果" in chunks[0] and "香蕉" in chunks[0]
    assert "股票" in chunks[1] and "债券" in chunks[1]
