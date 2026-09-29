from app.core.postprocess import postprocess
from helpers import T_OPEN, think_block


def test_strip_trailing_reference():
    assert postprocess("答案是少吃盐。\n参考来源：[1]") == "答案是少吃盐。"


def test_collapse_blank_lines():
    # 折叠到最多一个空行：成串空行对模型没有信息量，白烧 token 预算。
    assert postprocess("a\n\n\n\nb") == "a\n\nb"


def test_empty_input():
    # 空答案的判据是 `if not answer`（见 pipeline），所以空输入必须原样返回 falsy，
    # 一旦被洗成非空的空白字符，空回答就会被当成有效答案写进记忆。
    assert postprocess("") == ""
    assert postprocess(None) is None


# ---- 回归：行内引用不该把正文整段截断 ----
def test_inline_reference_does_not_truncate_body():
    """正文里的「（参考来源：资料N）」是正常引用，不能从这里把后面全删掉。

    早先的正则 `\\n*\\s*(参考来源|...).*$`（配合 re.S）会匹配上行内引用，
    并把该位置之后的整篇正文一起吃掉——实测 1110 字的回答被截成小半段。
    """
    src = (
        "高血压患者每日食盐应控制在5克以内（参考来源：资料1）。\n\n"
        "## 二、运动建议\n每周至少150分钟中等强度有氧运动。"
    )
    assert postprocess(src) == src


def test_inline_reference_mid_paragraph_keeps_following_lines():
    src = "先看这一段（参考资料：资料2）。\n后面这一行必须保留。"
    assert postprocess(src) == src


def test_trailing_reference_only_drops_its_own_line():
    """文末的参考来源行去掉，但它后面的正文不能被牵连。"""
    src = "答案在上一段。\n参考来源：资料1\n\n补充说明：以上仅供参考。"
    out = postprocess(src)
    assert "参考来源" not in out
    assert "补充说明：以上仅供参考。" in out


def test_multiline_reference_block_is_removed():
    """多行幻觉引用：表头 + 紧随的条目行要一起去掉。

    只删表头的话，剩下的 `[资料1] …` 看起来和正文一样，还会被写进短期记忆——
    清理前至少还有「参考来源」四个字提示这是引用，清理后反而更像事实。
    """
    src = (
        "高血压的诊断标准是 140/90 mmHg。\n\n"
        "参考来源：\n[资料1] 中国高血压防治指南\n[资料2] 国家基层高血压防治手册"
    )
    out = postprocess(src)

    assert "参考来源" not in out
    assert "[资料1]" not in out and "[资料2]" not in out
    assert "140/90" in out


def test_bracketed_numbered_list_in_body_survives():
    """正文里正常的编号列表（[1] 第一步）不能被当成引用条目误删。"""
    src = "按这个顺序来：\n[1] 限盐\n[2] 减油\n[3] 增钾"
    assert postprocess(src) == src


# ---- 回归：思维链清洗（pipeline 注释声称「去 think」，此前并无实现）----
# 「结束标记认不出时不能连正文一起删」那条不变式在 tests/test_think_stripping.py
# 用真实推理文本钉得更死（同一个 │end▁of▁thinking│ 标记），此处不再重复一份。
def test_strip_think_block():
    src = think_block("用户问的是限盐，我先想想。\n再想想。") + "\n高血压患者每日食盐应控制在5克以内。"
    assert postprocess(src) == "高血压患者每日食盐应控制在5克以内。"


def test_plain_angle_brackets_are_kept():
    """正文里正常的尖括号（如血压范围）不能被误伤。"""
    assert postprocess("血压 < 140/90 mmHg 为正常范围。") == "血压 < 140/90 mmHg 为正常范围。"
