# OCR 是兜底路径。三册语料文本层都干净，默认应零触发——
# 但判定逻辑必须可测，否则无从证明"它确实没触发"而不是"它坏了"
from app.ingest.ocr_paddle import low_quality_pages


def test_clean_text_triggers_nothing():
    # 三册语料的实际情况：有文本层、无乱码，不应触发任何 OCR
    md = "第一条为了保护民事主体的合法权益，调整民事关系。\n第二条民法调整平等主体。\n"
    assert low_quality_pages(md) == []


def test_no_page_marker_returns_empty():
    # 没有页标记时不做猜测性触发——宁可漏触发也不能无故全量跑 VLM 抢显存
    assert low_quality_pages("正文内容" * 100) == []


def test_page_with_almost_no_text_is_flagged():
    # 空页/图版页特征：单页字符数远低于正常页（正常页约 400 字）
    md = "<!-- page 5 -->\n\n\n<!-- page 6 -->\n" + "字" * 500
    assert 5 in low_quality_pages(md)


def test_replacement_chars_are_flagged():
    # 乱码页特征：字符数够，但替换字符占比超标。
    # 这一条必须与"字符数不足"分开验，否则长度规则会先命中、掩盖掉本规则是否真的生效
    md = "<!-- page 3 -->\n" + "�" * 30 + "字" * 70
    assert 3 in low_quality_pages(md)


def test_clean_page_after_bad_page_is_not_flagged():
    # 触发是按页的，不能一页坏就整册标记
    md = ("<!-- page 1 -->\n\n\n"
          "<!-- page 2 -->\n" + "字" * 500)
    assert low_quality_pages(md) == [1]
