"""切块器测试：条/款/项三级语义 + 父块换行保留。

术语依据 docs/CONTEXT.md：
- 条：基本单位（第 X 条）
- 款：条内相对独立的自然段（无编号）→ paragraph_no
- 项：款内的分项（（一）（二）…）→ item_no
"""
from pathlib import Path

from app.ingest.chunker import chunk_document
from app.ingest.parser import ParsedDocument


def _make_doc(content: str) -> ParsedDocument:
    return ParsedDocument(
        source_path=Path("test.html"),
        content=content,
        content_format="text/plain",
    )


def test_parent_content_preserves_newlines():
    """父块正文必须保留段落换行（只压行内空白，不压换行）。"""
    content = (
        "第四十七条 经济补偿按劳动者在本单位工作的年限，"
        "每满一年支付一个月工资的标准向劳动者支付。\n"
        "劳动者月工资高于用人单位所在直辖市、设区的市级人民政府"
        "公布的本地区上年度职工月平均工资三倍的，向其支付经济补偿的标准按职工月平均工资三倍的数额支付。\n"
        "本条所称月工资是指劳动者在劳动合同解除或者终止前十二个月的平均工资。"
    )
    chunks = chunk_document(_make_doc(content), document_title="劳动合同法")

    parents = [c for c in chunks if c.chunk_level == "parent"]
    assert len(parents) == 1
    # 父块正文保留两个段间换行
    assert parents[0].content.count("\n") == 2
    # 行内连续空格/制表符仍被压缩
    content_with_spaces = "第八条\t测试行内\t\t空白与   连续空格。"
    chunks2 = chunk_document(_make_doc(content_with_spaces), document_title="测试")
    parent2 = [c for c in chunks2 if c.chunk_level == "parent"][0]
    assert "\t" not in parent2.content
    assert "  " not in parent2.content


def test_items_are_item_no_not_paragraph_no():
    """「（一）（二）」是项（item_no），不是款；款是无编号自然段。"""
    content = (
        "第四十条 有下列情形之一的，用人单位提前三十日以书面形式通知劳动者本人"
        "或者额外支付劳动者一个月工资后，可以解除劳动合同：\n"
        "（一）劳动者患病或者非因工负伤，在规定的医疗期满后不能从事原工作，"
        "也不能从事由用人单位另行安排的工作的；\n"
        "（二）劳动者不能胜任工作，经过培训或者调整工作岗位，仍不能胜任工作的；\n"
        "（三）劳动合同订立时所依据的客观情况发生重大变化，"
        "致使劳动合同无法履行，经用人单位与劳动者协商，未能就变更劳动合同内容达成协议的。"
    )
    chunks = chunk_document(_make_doc(content), document_title="劳动合同法")

    parents = [c for c in chunks if c.chunk_level == "parent"]
    children = [c for c in chunks if c.chunk_level == "child"]

    # 1 个父块 + 3 个子块（3 个项）
    assert len(parents) == 1
    assert len(children) == 3

    # 项号 1/2/3；它们都属于第一款（引导段所在款）
    assert [c.item_no for c in children] == ["1", "2", "3"]
    assert all(c.paragraph_no == "1" for c in children)
    assert "劳动者患病" in children[0].content


def test_unnumbered_paragraphs_are_kuan():
    """条内无编号自然段是款：paragraph_no 依次 1/2/3，item_no 为空。"""
    content = (
        "第四十七条 经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。\n"
        "劳动者月工资高于用人单位所在直辖市、设区的市级人民政府公布的本地区上年度职工月平均工资三倍的，"
        "向其支付经济补偿的标准按职工月平均工资三倍的数额支付，"
        "向其支付经济补偿的年限最高不超过十二年。\n"
        "本条所称月工资是指劳动者在劳动合同解除或者终止前十二个月的平均工资。"
    )
    chunks = chunk_document(_make_doc(content), document_title="劳动合同法")

    children = [c for c in chunks if c.chunk_level == "child"]
    assert len(children) == 3
    assert [c.paragraph_no for c in children] == ["1", "2", "3"]
    assert all(c.item_no is None for c in children)


def test_chunk_ids_are_unique():
    """同一条文内的多个子块 chunk_id 不得重复。"""
    content = (
        "第二条 测试条款：\n"
        "（一）第一项内容；\n"
        "（二）第二项内容。\n"
        "第三条 另一条款，分两段。\n"
        "第二段内容。"
    )
    chunks = chunk_document(_make_doc(content), document_title="测试法")
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_chinese_item_number_conversion():
    """项号中文数字转阿拉伯数字：十、十一、十二。"""
    content = (
        "第一条 测试：\n"
        "（十）第十项；\n"
        "（十一）第十一项；\n"
        "（十二）第十二项。"
    )
    chunks = chunk_document(_make_doc(content), document_title="测试法")
    children = [c for c in chunks if c.chunk_level == "child"]
    assert [c.item_no for c in children] == ["10", "11", "12"]


def test_half_width_brackets_also_items():
    """半角括号 (一)(二) 同样识别为项。"""
    content = (
        "第二十八条 测试条款：\n"
        "(一)甲情形；\n"
        "(二)乙情形。"
    )
    chunks = chunk_document(_make_doc(content), document_title="测试法")
    children = [c for c in chunks if c.chunk_level == "child"]
    assert len(children) == 2
    assert [c.item_no for c in children] == ["1", "2"]
    assert all(c.paragraph_no == "1" for c in children)


def test_multiple_paragraphs_with_items_get_separate_kuan():
    """引导段之后的独立自然段开启新款；项归属各自所在款。"""
    content = (
        "第九条 引导情形一：\n"
        "（一）甲项；\n"
        "（二）乙项。\n"
        "这是第二款，无编号自然段。\n"
        "第三款开头：\n"
        "（一）丙项。"
    )
    chunks = chunk_document(_make_doc(content), document_title="测试法")
    children = [c for c in chunks if c.chunk_level == "child"]

    # 款 1 的两项 + 款 2 本身 + 款 3 的一项
    assert len(children) == 4
    assert children[0].paragraph_no == "1" and children[0].item_no == "1"
    assert children[1].paragraph_no == "1" and children[1].item_no == "2"
    assert children[2].paragraph_no == "2" and children[2].item_no is None
    assert children[3].paragraph_no == "3" and children[3].item_no == "1"


def test_single_paragraph_article_child_not_duplicated():
    """单段条文的子块正文与父块一致，不产生重复行拼接。"""
    content = "第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，制定本法。"
    chunks = chunk_document(_make_doc(content), document_title="劳动合同法")

    assert len(chunks) == 2
    parent, child = chunks
    assert parent.chunk_level == "parent"
    assert child.chunk_level == "child"
    assert child.content == parent.content
    assert child.paragraph_no == "1"


def test_articles_without_separator_after_tiao():
    """法院网页常见格式：条号后无分隔符直接接正文（"第一条劳动者…"）。

    现行边界正则要求"条"后必须是空白/标点/行尾，导致该格式整篇降级为
    一个文档级父块（article_no 全空）。修复后应正常切条：
    - 行首"第X条"+正文汉字 → 条边界
    - 行首"第X条第Y款…"（引用续接）→ 不算新条边界
    - 行首"第X条之N"变体 → 照常识别
    """
    content = (
        "为正确审理劳动争议案件，制定本解释。\n"
        "第一条劳动者与用人单位之间发生的下列纠纷，属于劳动争议：\n"
        "（一）劳动者与用人单位在履行劳动合同过程中发生的纠纷；\n"
        "第二条下列纠纷不属于劳动争议：\n"
        "第三十二条劳动者在用人单位安排下参加业余活动，" 
        "参照本解释第一条执行。\n"
    )
    chunks = chunk_document(_make_doc(content), document_title="解释一")

    parents = [c for c in chunks if c.chunk_level == "parent"]
    nos = [p.article_no for p in parents]
    assert nos == ["第一条", "第二条", "第三十二条"]
    # 子块保留条号归属
    children = [c for c in chunks if c.chunk_level == "child"]
    assert all(c.article_no in {"第一条", "第二条", "第三十二条"} for c in children)
    # （一）项子块正常识别
    item_children = [c for c in children if c.item_no]
    assert len(item_children) == 1 and item_children[0].item_no == "1"


def test_article_reference_second_paragraph_not_a_boundary():
    """行首"第二十三条第二款规定的情形"是引用续接，不是新条文边界。"""
    content = (
        "第四十七条 经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资。\n"
        "第二十三条第二款规定的情形，用人单位无需支付经济补偿。"
    )
    chunks = chunk_document(_make_doc(content), document_title="测试法")
    parents = [c for c in chunks if c.chunk_level == "parent"]
    # "第二十三条第二款" 不能被切成新条
    assert [p.article_no for p in parents] == ["第四十七条"]


def test_chunker_version_bumped_for_boundary_change():
    """边界识别行为变更必须递增 CHUNKER_VERSION（触发切块指纹变化）。"""
    from app.ingest.chunk_fingerprint import CHUNKER_VERSION

    assert CHUNKER_VERSION >= "3"
