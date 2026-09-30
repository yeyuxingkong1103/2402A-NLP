"""单元测试：src.rag.parser 文本清洗、去重、文件解析与异常。"""
import pytest

from src.rag import parser

DOC_TXT = """第 3 页
焦虑的自我调节

焦虑是对未来威胁的一种正常情绪反应。适度的焦虑可以提升表现。

焦虑是对未来威胁的一种正常情绪反应。适度的焦虑可以提升表现。

www.example.com
扫码关注公众号获取更多资料

当焦虑持续时间过长、强度过高时，就会影响睡眠与注意力。
"""


# ---------------- clean_text ----------------
def test_clean_text_empty():
    assert parser.clean_text("") == ""


def test_clean_text_removes_watermark_and_page_number():
    cleaned = parser.clean_text(DOC_TXT)
    assert "第 3 页" not in cleaned
    assert "www.example.com" not in cleaned
    assert "扫码关注" not in cleaned
    assert "焦虑的自我调节" in cleaned


@pytest.mark.parametrize("line", ["Page 12 of 30", "- 42 -", "仅供个人学习使用，请勿传播"])
def test_clean_text_removes_watermark_lines(line):
    assert parser.clean_text(f"正文内容\n{line}\n结尾内容") == "正文内容\n结尾内容"


def test_clean_text_removes_invisible_chars_and_blank_lines():
    cleaned = parser.clean_text("第一行\u200b有零宽字符\n\n\n\n第二行")
    assert "\u200b" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "第一行有零宽字符\n\n第二行"


def test_clean_text_normalizes_full_width():
    assert parser.clean_text("ＡＢＣ１２３") == "ABC123"


# ---------------- 去重 ----------------
def test_deduplicate_paragraphs():
    text = "重复段落。\n\n重复段落。\n\n另一个段落。"
    deduped = parser.deduplicate_paragraphs(text)
    assert deduped.count("重复段落。") == 1
    assert "另一个段落。" in deduped


def test_deduplicate_paragraphs_ignores_whitespace_diff():
    deduped = parser.deduplicate_paragraphs("同样的内容\n\n同 样 的 内 容")
    assert deduped == "同样的内容"


def test_remove_repeated_lines():
    pages = ["页眉\n第一页正文", "页眉\n第二页正文", "页眉\n第三页正文"]
    cleaned = parser.remove_repeated_lines(pages)
    assert all("页眉" not in page for page in cleaned)
    assert "第一页正文" in cleaned[0]


# ---------------- parse_file ----------------
def test_parse_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parser.parse_file(str(tmp_path / "nope.txt"))


def test_parse_file_unsupported_type(tmp_path):
    target = tmp_path / "bad.xyz"
    target.write_text("内容", encoding="utf-8")
    with pytest.raises(ValueError):
        parser.parse_file(str(target))


def test_parse_file_txt(tmp_path):
    target = tmp_path / "情绪管理.txt"
    target.write_text(DOC_TXT, encoding="utf-8")
    doc = parser.parse_file(str(target))
    assert doc.file_type == "txt"
    assert doc.title == "情绪管理"
    assert doc.pages == 1
    assert doc.char_count > 0
    assert "第 3 页" not in doc.text
    assert "扫码关注" not in doc.text  # 水印行被剔除
    assert doc.text.count("焦虑是对未来威胁的一种正常情绪反应") == 1  # 段落去重


def test_parse_file_md(tmp_path):
    target = tmp_path / "笔记.md"
    target.write_text("# 正念呼吸\n\n吸气四秒，呼气六秒。", encoding="utf-8")
    doc = parser.parse_file(str(target))
    assert doc.file_type == "md"
    assert "吸气四秒" in doc.text


def test_parse_file_txt_without_dedup(tmp_path):
    target = tmp_path / "dup.txt"
    target.write_text("重复段落。\n\n重复段落。", encoding="utf-8")
    doc = parser.parse_file(str(target), do_dedup=False)
    assert doc.text.count("重复段落。") == 2


def test_parse_file_gbk_encoding(tmp_path):
    target = tmp_path / "gbk.txt"
    target.write_bytes("中文编码测试内容".encode("gbk"))
    doc = parser.parse_file(str(target))
    assert "中文编码测试内容" in doc.text


def test_parse_directory(tmp_path):
    (tmp_path / "a.txt").write_text("第一份文档内容", encoding="utf-8")
    (tmp_path / "b.md").write_text("第二份文档内容", encoding="utf-8")
    (tmp_path / "c.xyz").write_text("不支持的类型", encoding="utf-8")
    docs = parser.parse_directory(str(tmp_path), recursive=False)
    assert len(docs) == 2
    assert {d.title for d in docs} == {"a", "b"}


def test_parsed_document_char_count():
    doc = parser.ParsedDocument(title="t", text="12345", file_type="txt", file_path="/tmp/t")
    assert doc.char_count == 5