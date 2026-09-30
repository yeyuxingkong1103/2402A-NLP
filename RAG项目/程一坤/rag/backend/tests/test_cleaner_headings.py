"""批次 22：cleaner 短行规则专项测试。

背景：`cleaner._is_navigation_line` 有一条规定「< 6 字符且不含标点」的孤立短行
按导航栏剔除。法规里的章节标题（「第一章」「总则」「第一章 总则」）恰好满足这个
特征，会被误删 —— 标题没了，整章条文就失去归属。

本批次把结构标题加入白名单（不参与导航判定），同时保证真实的导航噪音仍被剔除。
"""

from app.ingest.cleaner import clean_text


def test_chapter_heading_alone_is_kept():
    assert clean_text("第一章") == "第一章"


def test_section_name_alone_is_kept():
    assert clean_text("总则") == "总则"


def test_chapter_heading_with_title_is_kept():
    assert clean_text("第一章 总则") == "第一章 总则"


def test_article_heading_alone_is_kept():
    assert clean_text("第五条") == "第五条"


def test_article_heading_variant_is_kept():
    assert clean_text("第五十条之一") == "第五十条之一"


def test_other_structural_headings_are_kept():
    for line in ("第二节", "第三编", "第一部分", "第二章 劳动合同的订立", "附则", "序言"):
        assert clean_text(line) == line, line


def test_table_of_contents_heading_is_kept():
    assert clean_text("目录") == "目录"
    assert clean_text("目 录") == "目 录"  # PDF 排版常见形态


def test_headings_survive_inside_a_document():
    raw = "第一章 总则\n\n第一条 为了规范。\n\n第二章\n\n总则\n\n第二条 用人单位。"
    cleaned = clean_text(raw)
    for expected in ("第一章 总则", "第一条 为了规范。", "第二章", "总则", "第二条 用人单位。"):
        assert expected in cleaned, expected


def test_real_navigation_noise_still_removed():
    # 导航关键词开头
    assert clean_text("首页") == ""
    assert clean_text("机构设置 联系我们") == ""
    # 不含结构特征的孤立短行（乱码/栏目标签）仍被剔除
    assert clean_text("某某") == ""
    assert clean_text("abc") == ""
    assert clean_text("12") == ""


def test_menu_keyword_dense_line_removed():
    assert clean_text("首页 资讯 公告 政策 法规") == ""


def test_normal_sentences_untouched():
    line = "第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务。"
    assert clean_text(line) == line


def test_repeated_header_and_footer_rules_still_apply():
    raw = "第一章\n页眉文本\n第一条 正文。\n页脚文本"
    cleaned = clean_text(raw, repeated_headers=["页眉文本"], repeated_footers=["页脚文本"])
    assert cleaned == "第一章\n第一条 正文。"


def test_watermark_removal_still_applies():
    raw = "第一条 内部资料 正文。"
    assert clean_text(raw, watermarks=["内部资料"]) == "第一条 正文。"


def test_blank_lines_collapse_and_edges_trimmed():
    # 空行按原文保留（段落边界不能压掉），只裁掉首尾空行
    assert clean_text("\n\n第一章\n\n\n第一条 正文。\n\n") == "第一章\n\n\n第一条 正文。"
