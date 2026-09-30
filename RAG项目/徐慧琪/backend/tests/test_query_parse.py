# 抽号是 AC-2（条款号直问 100% 命中）的第一道关：抽不出来就置顶不了。
# 全是纯函数，不需要任何服务。
from app.retrieval.query_parse import extract_article_nos


def test_extracts_chinese_numeral_form():
    # 评估集直问题多半这么写，法条自身也是这个写法
    assert extract_article_nos("第五百八十四条规定的违约责任是什么") == [584]


def test_extracts_arabic_form():
    assert extract_article_nos("第584条怎么规定的") == [584]


def test_extracts_fullwidth_arabic():
    # PDF 与网页复制来的问句常带全角数字，不认就会静默漏掉置顶
    assert extract_article_nos("第５８４条怎么规定的") == [584]


def test_extracts_ten_without_prefix_digit():
    # 「第十条」的十前面没有数字，cn2int 专门为它写了分支
    assert extract_article_nos("第十条") == [10]


def test_extracts_thousand_form():
    # 民法典 1000~1260 条（共 261 条）正文写法全是千字头（如「第一千零四十条」），
    # 字符类漏「千」时这类问句全部静默返回空——不报错、不置顶，261 条宽的洞。
    # 上界取 1260 是因为它是民法典最后一条，顺带钉住千位串的解析
    assert extract_article_nos("第一千零四十条") == [1040]
    assert extract_article_nos("第一千二百六十条") == [1260]


def test_keeps_order_and_dedupes():
    text = "第577条和第584条有什么关系？第577条我已经看过了"
    assert extract_article_nos(text) == [577, 584]


def test_ignores_non_article_references():
    # 「第三款」「第二章」不是条号，误抽会让精确通路返回错条
    assert extract_article_nos("第二章第三款说了什么") == []


def test_ignores_article_internal_cross_reference():
    # 中册 md 里有 42 处条内引用（如"依照本法第五百八十四条"），它们也是真条号，
    # 抽出来是对的——置顶不会有害，漏掉反而可能漏掉用户真正想问的条
    assert extract_article_nos("依照本法第五百八十四条的规定") == [584]


def test_returns_empty_for_plain_question():
    assert extract_article_nos("租房押金不退怎么办") == []


def test_extracts_out_of_range_number():
    # 抽号不管边界：第9999条也抽出来，查库查不到自然就不置顶（见设计文档 4.1）
    assert extract_article_nos("第9999条") == [9999]
