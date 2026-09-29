"""postprocess() 单元测试：5 条清洗规则 + 边界/幂等。"""
import rag


def test_postprocess_strips_code_fence():
    """去掉 Markdown 代码块围栏行（含语言标记），保留围栏内内容。"""
    assert rag.postprocess('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_postprocess_collapses_five_plus_newlines():
    """5 个及以上连续换行压缩为 3 个换行。"""
    assert rag.postprocess("a" + "\n" * 5 + "b") == "a\n\n\nb"


def test_postprocess_keeps_four_newlines():
    """4 个连续换行不触发压缩（正则阈值为 5）。"""
    text = "a" + "\n" * 4 + "b"
    assert rag.postprocess(text) == text


def test_postprocess_strips_trailing_whitespace():
    """去掉每行行尾的空格与制表符。"""
    assert rag.postprocess("第一行  \n第二行\t\n") == "第一行\n第二行"


def test_postprocess_strips_edges():
    """去掉文本首尾的空白字符。"""
    assert rag.postprocess("  \n  内容  \n  ") == "内容"


def test_postprocess_normalizes_reference_brackets():
    """把【资料1】/「资料2」统一为资料1/资料2。"""
    assert rag.postprocess("见【资料1】与「资料2」") == "见资料1与资料2"


def test_postprocess_normalizes_reference_space():
    """把「资料 3」统一为「资料3」（去括号内空格）。"""
    assert rag.postprocess("见资料 3 说明") == "见资料3 说明"


def test_postprocess_empty_string():
    """空字符串原样返回，不抛异常。"""
    assert rag.postprocess("") == ""


def test_postprocess_whitespace_only():
    """纯空白文本清洗后为空字符串。"""
    assert rag.postprocess("   \n  ") == ""


def test_postprocess_idempotent():
    """已清洗文本再次清洗结果不变（幂等）。"""
    dirty = "```markdown\n结论：血压高\n\n\n\n\n\n详见资料 2 与【资料1】\n```  "
    once = rag.postprocess(dirty)
    assert rag.postprocess(once) == once
