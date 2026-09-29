"""单元测试：src.rag.chunker 文本分块、token 估算与边界情况。"""
import pytest

from src.rag.chunker import (Chunk, chunk_by_heading, chunk_fixed, chunk_parent_child,
                             chunk_text, estimate_tokens, split_paragraphs,
                             split_sentences, summarize_chunk)

SAMPLE = (
    "焦虑是对未来威胁的一种正常情绪反应，适度的焦虑可以提升表现。\n\n"
    "当焦虑持续时间过长、强度过高时，就会影响睡眠、注意力和人际关系。\n\n"
    "认知行为疗法认为，影响情绪的不是事件本身，而是我们对事件的解释。\n\n"
    "通过识别自动思维、检验证据、寻找替代解释，可以降低情绪的强度。\n\n"
    "呼吸放松与渐进式肌肉放松可以帮助身体从紧张状态回到平静。\n"
)

HEADING_DOC = (
    "第一章 认识焦虑\n"
    "焦虑是一种常见的情绪体验，几乎每个人都会经历。\n\n"
    "第二章 认知重构\n"
    "识别自动思维是认知重构的第一步。\n\n"
    "第三章 行为激活\n"
    "行为激活强调通过增加积极活动来改善情绪。\n"
)


# ---------------- estimate_tokens ----------------
def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_cjk_and_ascii():
    assert estimate_tokens("你好世界") == 4
    # 8 个 ASCII 字符 -> 8 // 4 = 2
    assert estimate_tokens("abcdefgh") == 2
    # 少于 4 个 ASCII 字符时至少算 1 个 token
    assert estimate_tokens("ab") == 1
    # 混合文本：2 个中文字 + 4 个 ASCII
    assert estimate_tokens("中文abcd") == 3


def test_estimate_tokens_monotonic():
    assert estimate_tokens(SAMPLE) > estimate_tokens(SAMPLE[:20])


# ---------------- 切句 / 切段 ----------------
def test_split_sentences_and_paragraphs():
    sentences = split_sentences("第一句。第二句！第三句？")
    assert sentences == ["第一句。", "第二句！", "第三句？"]
    assert split_paragraphs("段落一。\n\n段落二。\n\n\n段落三。") == ["段落一。", "段落二。", "段落三。"]


# ---------------- 边界情况 ----------------
@pytest.mark.parametrize("strategy", ["fixed", "sentence", "paragraph", "heading", "semantic"])
def test_chunk_text_empty_input(strategy):
    assert chunk_text("", strategy=strategy) == []
    assert chunk_text(None, strategy=strategy) == []
    assert chunk_text("   \n\n  ", strategy=strategy) == []


@pytest.mark.parametrize("strategy", ["fixed", "sentence", "paragraph", "heading", "semantic"])
def test_chunk_text_all_strategies(strategy):
    chunks = chunk_text(SAMPLE, strategy=strategy, chunk_size=60, overlap=10)
    assert chunks, f"{strategy} 未产出分块"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.content.strip() for c in chunks)
    # chunk_index 连续且从 0 开始
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.tokens > 0 for c in chunks)


def test_chunk_text_unknown_strategy_falls_back_to_paragraph():
    fallback = chunk_text(SAMPLE, strategy="not-exist", chunk_size=60, overlap=10)
    expected = chunk_text(SAMPLE, strategy="paragraph", chunk_size=60, overlap=10)
    assert [c.content for c in fallback] == [c.content for c in expected]


def test_chunk_text_default_strategy():
    chunks = chunk_text(SAMPLE)
    assert chunks
    assert all(c.content for c in chunks)


# ---------------- fixed ----------------
def test_chunk_fixed_size_and_overlap():
    text = "测试文本" * 60  # 240 字符
    chunks = chunk_fixed(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.content) <= 50 for c in chunks)
    # 重叠窗口：step = 40
    assert chunks[1].content[:10] in chunks[0].content


def test_chunk_fixed_empty():
    assert chunk_fixed("", 100, 10) == []


# ---------------- sentence ----------------
def test_chunk_by_sentences_keeps_sentence_boundary():
    text = "第一句内容。第二句内容。第三句内容。第四句内容。"
    chunks = chunk_text(text, strategy="sentence", chunk_size=8, overlap=0)
    assert len(chunks) >= 2
    joined = "".join(c.content for c in chunks)
    for sentence in ["第一句内容。", "第四句内容。"]:
        assert sentence in joined


# ---------------- paragraph ----------------
def test_chunk_by_paragraph_preserves_paragraph():
    chunks = chunk_text(SAMPLE, strategy="paragraph", chunk_size=40, overlap=0)
    assert len(chunks) >= 2
    # 段落内容不被截断（短段落原样保留）
    assert any("焦虑是对未来威胁的一种正常情绪反应" in c.content for c in chunks)


# ---------------- heading ----------------
def test_chunk_by_heading_splits_on_heading():
    chunks = chunk_by_heading(HEADING_DOC, 200, 20)
    assert len(chunks) == 3
    assert chunks[0].content.startswith("第一章")
    assert chunks[1].content.startswith("第二章")
    assert chunks[2].content.startswith("第三章")


def test_chunk_by_heading_no_heading():
    chunks = chunk_by_heading("只有一段普通文字，没有标题。", 200, 20)
    assert len(chunks) == 1


# ---------------- semantic / parent_child ----------------
def test_chunk_semantic_non_empty():
    chunks = chunk_text(HEADING_DOC, strategy="semantic", chunk_size=40, overlap=5)
    assert chunks
    assert all(c.content.strip() for c in chunks)


def test_chunk_parent_child_links():
    children = chunk_parent_child(SAMPLE, parent_size=60, child_size=30)
    assert children
    assert all(c.parent_index is not None for c in children)
    assert all(c.extra.get("parent_content") for c in children)
    # 子块 parent_index 递增且不超过父块数量
    assert sorted({c.parent_index for c in children}) == list(range(len({c.parent_index for c in children})))


def test_chunk_text_parent_child_strategy():
    chunks = chunk_text(SAMPLE, strategy="parent_child")
    assert chunks
    assert all(c.parent_index is not None for c in chunks)


# ---------------- summarize_chunk ----------------
def test_summarize_chunk_takes_first_sentence():
    assert summarize_chunk("第一句话。第二句话。") == "第一句话。"
    assert len(summarize_chunk("很长的一句话" * 50, max_len=20)) == 20