# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pytest
from rag04.config import get_settings, PROJECT_ROOT
from rag04.schema import TextBlock, TableBlock, FigureBlock
from rag04.ingest.chunker import (
    chunk_text_blocks, chunk_fixed, chunk_tables, chunk_figures,
    build_chunks, detect_lang, merge_line_blocks, chunk_figure_texts,
)


def _tb(i, text, page=1):
    return TextBlock(doc_id="d", page=page, bbox=(0, i * 10, 100, i * 10 + 10), text=text)


def test_detect_lang_zh_and_en():
    assert detect_lang("武汉力源信息技术股份有限公司") == "zh"
    assert detect_lang("What is the company name?") == "en"


def test_fixed_chunk_respects_size():
    s = get_settings("baseline_03")
    blocks = [_tb(i, "字" * 300) for i in range(5)]
    chunks = chunk_fixed(blocks, s)
    assert chunks
    for c in chunks:
        assert len(c.text) <= s.fixed_chunk_size * 1.5
        assert c.block_type == "text"


def test_semantic_chunk_respects_max():
    s = get_settings("full_04")
    blocks = [_tb(i, "内容" * 400) for i in range(3)]
    chunks = chunk_text_blocks(blocks, s)
    for c in chunks:
        assert len(c.text) <= s.semantic_max_chars * 1.6


def test_chunks_have_required_metadata():
    s = get_settings()
    chunks = chunk_text_blocks([_tb(0, "武汉力源信息")], s)
    c = chunks[0]
    assert c.chunk_id and c.doc_id == "d" and c.page == 1
    assert c.block_type == "text" and c.source_id
    assert c.lang == "zh"


def test_chunk_ids_are_unique():
    s = get_settings()
    blocks = [_tb(i, f"段落{i}" * 50) for i in range(10)]
    chunks = chunk_text_blocks(blocks, s)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 必须唯一"


def test_table_chunk_is_whole_table():
    s = get_settings()
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    t = TableBlock(doc_id="d", page=3, bbox=(0, 0, 10, 10), markdown=md, n_rows=2, n_cols=2)
    chunks = chunk_tables([t], s)
    assert len(chunks) == 1
    assert chunks[0].block_type == "table"
    assert md in chunks[0].text
    assert chunks[0].page == 3


def test_figure_chunk_carries_description_and_clip():
    s = get_settings()
    f = FigureBlock(doc_id="d", page=38, bbox=(0, 0, 10, 10),
                    image_path="a.png", caption="公司组织结构图",
                    description="销售部下设渠道销售部等4个部门",
                    clip_vector=[0.1] * 512)
    chunks = chunk_figures([f], s)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.block_type == "image"
    assert "渠道销售部" in c.text
    assert "公司组织结构图" in c.text, "题注必须进正文，否则检索不到"
    assert c.extra["clip_vector"] == [0.1] * 512
    assert c.extra["image_path"] == "a.png"


# ---------- RC2：图像 VLM 描述进入文本通道（重建第二阶段） ----------

def test_figure_text_chunk_links_to_image_chunk_for_dedup():
    """图描述文本块必须携带 dedup_key=对应 image 块 chunk_id（同图只占一席）。"""
    s = get_settings("full_04")
    f = FigureBlock(doc_id="d", page=39, bbox=(0, 0, 10, 10),
                    image_path="p39.png", caption="组织结构图",
                    description="销售部下设渠道销售部等4个部门",
                    clip_vector=[0.1] * 512)
    img = chunk_figures([f], s)[0]
    txt = chunk_figure_texts([f], s)[0]

    assert txt.block_type == "text"
    assert txt.chunk_id != img.chunk_id
    assert txt.extra["dedup_key"] == img.chunk_id
    assert txt.extra["figure_chunk_id"] == img.chunk_id
    assert txt.text == img.text, "文本块与被去重的图像块正文必须一致"
    assert "渠道销售部" in txt.text and "组织结构图" in txt.text


def test_figure_text_chunks_unique_ids_same_page():
    s = get_settings("full_04")
    figs = [FigureBlock("d", 12, (0, i * 10, 100, i * 10 + 10), f"p12_{i}.png",
                        caption=f"图{i}", description=f"描述{i}")
            for i in range(3)]
    chunks = chunk_figure_texts(figs, s)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) == 3
    assert {c.extra["dedup_key"] for c in chunks} == \
        {c.chunk_id for c in chunk_figures(figs, s)}


def test_build_chunks_full_04_puts_figure_description_in_text_channel():
    s = get_settings("full_04")
    f = FigureBlock("d", 39, (0, 0, 1, 1), "i.png", caption="组织结构图",
                    description="大客户销售部下设六个销售处")
    chunks = build_chunks([], [], [f], s)
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c.block_type, []).append(c)
    assert len(by_type["image"]) == 1
    assert any("大客户销售部" in c.text and c.extra.get("dedup_key")
               for c in by_type["text"]), "描述必须同时进入 text 通道"


def test_build_chunks_accepts_figure_generator():
    """复核 Minor：figures 被遍历两次（描述文本块 + 图像块），生成器调用方不得丢图。"""
    s = get_settings("full_04")
    fig = FigureBlock("d", 39, (0, 0, 1, 1), "i.png", caption="组织结构图",
                      description="销售部下设4个部门")
    chunks = build_chunks([], [], (f for f in [fig]), s)
    kinds = [c.block_type for c in chunks]
    assert kinds.count("image") == 1, "生成器传入时图像块不得静默消失"
    assert kinds.count("text") == 1


def test_build_chunks_baseline_excludes_tables_and_figures():
    s = get_settings("baseline_03")
    chunks = build_chunks(
        [_tb(0, "正文内容")],
        [TableBlock("d", 1, (0, 0, 1, 1), "| a |", 1, 1)],
        [FigureBlock("d", 1, (0, 0, 1, 1), "i.png", description="图描述")],
        s,
    )
    kinds = {c.block_type for c in chunks}
    assert kinds == {"text"}, f"baseline_03 必须只有文本块，实际={kinds}"


def test_build_chunks_full_includes_all_three():
    s = get_settings("full_04")
    chunks = build_chunks(
        [_tb(0, "正文内容")],
        [TableBlock("d", 1, (0, 0, 1, 1), "| a | b |\n| --- | --- |\n| 1 | 2 |", 2, 2)],
        [FigureBlock("d", 1, (0, 0, 1, 1), "i.png", description="图描述")],
        s,
    )
    kinds = {c.block_type for c in chunks}
    assert kinds == {"text", "table", "image"}


def test_pathological_long_sentence_is_hard_split_within_max():
    """3000 字无句读句（OCR 乱码/列表）必须硬切，任何块不得超过 semantic_max_chars。"""
    s = get_settings("full_04")
    text = "字" * 3000
    chunks = chunk_text_blocks([_tb(0, text)], s)
    assert chunks
    assert len(chunks) > 1, "超长无标点句不得整句成块"
    assert max(len(c.text) for c in chunks) <= s.semantic_max_chars
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "硬切产生的 chunk_id 必须唯一"
    assert chunks[0].text.startswith(text[:20]), "首片必须从句子开头开始，不能丢头部"
    assert chunks[-1].text.endswith(text[-20:]), "末片必须覆盖句子结尾，不能丢尾部"
    assert sum(len(c.text) for c in chunks) >= len(text), "硬切不得丢失内容"


def test_long_sentence_after_499_char_buffer_stays_within_max():
    """跨边界病态：499 字前缀 + 无标点长句，缓冲结清与硬切后仍不得超上限。"""
    s = get_settings("full_04")
    text = "前" * 499 + "。" + "长" * 3000
    chunks = chunk_text_blocks([_tb(0, text)], s)
    assert chunks
    assert max(len(c.text) for c in chunks) <= s.semantic_max_chars
    joined = "".join(c.text for c in chunks)
    assert "前" * 499 in joined, "499 字缓冲前缀必须保留"
    assert "长" * 3000 in joined, "长句内容必须完整覆盖"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_multi_tables_and_figures_on_same_page_have_unique_ids():
    """Task 7 每页可产出多张表/多张图，同页多模态 chunk_id 必须互不相同。"""
    s = get_settings("full_04")
    page = 12
    tables = [
        TableBlock("d", page, (0, i * 10, 100, i * 10 + 10),
                   f"| h{i} | b |\n| --- | --- |\n| {i} | 2 |", 2, 2)
        for i in range(3)
    ]
    figs = [
        FigureBlock("d", page, (0, i * 10, 100, i * 10 + 10), f"p{page}_{i}.png",
                    caption=f"图{i}", description=f"描述{i}")
        for i in range(3)
    ]
    chunks = build_chunks([_tb(0, "正文" * 200, page=page)], tables, figs, s)
    kinds = [c.block_type for c in chunks]
    assert kinds.count("table") == 3
    assert kinds.count("image") == 3
    ids = [c.chunk_id for c in chunks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert len(ids) == len(set(ids)), f"同页多表多图 chunk_id 重复：{dupes}"


# ---------- RC5：PDF 行级块 → 段落级合并（重建第二阶段） ----------

def test_merge_line_blocks_joins_mid_sentence_lines():
    """答案句被行块切断（id260 金额续句）时必须合并为同一块。"""
    blocks = [
        _tb(0, "报告期内，公司来自军用领域的"),
        _tb(1, "收入分别为6,464.51 万元、14,414.16 万元，占"),
        _tb(2, "主营业务收入比重分别为82.10%、97.31%。"),
        _tb(3, "下一段落独立成句。"),
    ]
    out = merge_line_blocks(blocks)
    assert [b.text for b in out] == [
        "报告期内，公司来自军用领域的收入分别为6,464.51 万元、14,414.16 万元，"
        "占主营业务收入比重分别为82.10%、97.31%。",
        "下一段落独立成句。",
    ]


def test_merge_line_blocks_does_not_cross_page_or_doc():
    blocks = [
        _tb(0, "本页结尾没有句号的续句", page=1),
        _tb(1, "下一页开头的正文。", page=2),
        TextBlock(doc_id="d2", page=1, bbox=(0, 0, 1, 1),
                  text="另一文档的续句"),
    ]
    out = merge_line_blocks(blocks)
    assert len(out) == 3, "跨页/跨文档不得合并"
    assert out[0].page == 1 and out[1].page == 2


def test_merge_line_blocks_ascii_boundary_keeps_space():
    out = merge_line_blocks([_tb(0, "Wuhan Xingtu Xinke Electronics"),
                             _tb(1, "Co.,Ltd. and more")])
    assert out[0].text == "Wuhan Xingtu Xinke Electronics Co.,Ltd. and more"


def test_merge_line_blocks_sentence_enders_and_closers():
    """句末标点（含后置引号/括号）之后的块不得吞并。"""
    blocks = [
        _tb(0, "第一句结束。”"),
        _tb(1, "第二句结束。）"),
        _tb(2, "第三句结束。"),
        _tb(3, "第四段还没结束，"),
        _tb(4, "续完。"),
    ]
    out = merge_line_blocks(blocks)
    assert [b.text for b in out] == [
        "第一句结束。”", "第二句结束。）", "第三句结束。", "第四段还没结束，续完。"]


def test_merge_line_blocks_or_rotated_flag_and_keep_first_bbox():
    a = TextBlock(doc_id="d", page=3, bbox=(1, 2, 3, 4), text="上半句，")
    b = TextBlock(doc_id="d", page=3, bbox=(5, 6, 7, 8), text="下半句。",
                  has_rotated_text=True)
    (m,) = merge_line_blocks([a, b])
    assert m.has_rotated_text is True
    assert m.bbox == (1, 2, 3, 4), "合并块以首块定位，保持页码/坐标可溯源"


def test_build_chunks_full_04_merges_lines_baseline_keeps_them():
    s_full = get_settings("full_04")
    blocks = [_tb(0, "本次募集资金拟投资以下项目："),
              _tb(1, "仓储及物流中心、研发中心、电子商务平台。")]
    chunks = build_chunks(blocks, [], [], s_full)
    joined = "".join(c.text for c in chunks)
    assert "本次募集资金拟投资以下项目：仓储及物流中心、研发中心、电子商务平台。" in joined

    s_base = get_settings("baseline_03")
    baseline_chunks = build_chunks(blocks, [], [], s_base)
    assert len(baseline_chunks) == 2, "baseline_03 是对照组，不做 RC5 合并"


@pytest.mark.skipif(not (PROJECT_ROOT / "招股说明书1.pdf").exists(),
                    reason="语料缺失")
def test_merge_line_blocks_real_corpus_rejoins_split_amount_sentence():
    """真实语料 p129：金额续句与占比句必须落在同一块（id260/id33 证据）。"""
    from rag04.ingest.loader import doc_id_of, iter_text_blocks, open_pdf

    pdf = PROJECT_ROOT / "招股说明书1.pdf"
    doc = open_pdf(pdf)
    try:
        blocks = [b for b in iter_text_blocks(doc, doc_id_of(pdf)) if b.page == 129]
    finally:
        doc.close()
    merged = [b.text for b in merge_line_blocks(blocks)]
    assert any("6,464.51" in t and "94.34%" in t and "军用领域" in t
               for t in merged), f"p129 金额句未合并完整：{merged}"
