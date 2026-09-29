from app.core.document.cleaner import clean_text, remove_watermark_lines


def test_clean_text_normalizes_whitespace():
    # 段落之间要留一个空行：paragraph 策略靠 \n\n 找边界，全压掉整篇就并成一段。
    assert clean_text("  a   b \n\n\n c ") == "a b\n\nc"


def test_remove_watermark_lines():
    # 页脚噪音（URL、页码、扫码）会顶掉 chunk 预算却不含任何知识。
    lines = ["正文内容", "扫描二维码关注我们", "http://example.com", "第 1 页", "仅供参考"]
    assert remove_watermark_lines(lines) == ["正文内容"]


def test_clean_text_drops_watermark():
    text = "正文内容\n扫描二维码关注我们\n"
    assert clean_text(text) == "正文内容"


# ---- 正文里"提了一嘴"水印词，不能连整行一起删 ----
def test_body_lines_mentioning_watermark_words_survive():
    """水印特征以前是裸 re.search，正文行只要含这些词就整行蒸发（实测 6 条删 5 条）。

    这类丢失最阴的地方是**不报错**：入库 200 OK、列表里也看得到文档，只是检索答非所问。
    所以现在只认「整行几乎就是这句水印」的行。
    """
    body = [
        "更多关于高血压的饮食建议，可访问我们的健康专栏",
        "关注公众号后回复关键词即可获取随访表",
        "扫描二维码可加入患者互助群，群内有医生答疑",
        "本资料仅供内部培训参考，请勿外传至患者",
        "更多内容请查看附录 A",
        "仅供临床医生参考使用，具体用药请遵医嘱",
    ]
    assert remove_watermark_lines(body) == body


def test_real_watermark_lines_still_dropped():
    """反过来的误报：收紧正则不能把真水印放进来（页脚噪音会污染 chunk）。"""
    watermarks = [
        "扫描二维码关注我们",
        "扫描二维码",
        "扫码关注公众号：健康小站",
        "关注公众号",
        "更多内容请访问 www.example.com",
        "更多精彩内容请查看 https://x.cn/a",
        "仅供内部参考",
        "仅供参考",
        "第 1 页",
        "第 2 页 共 8 页",
        "http://example.com",
        "www.example.com",
    ]
    assert remove_watermark_lines(watermarks) == []
