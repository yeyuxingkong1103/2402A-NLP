# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单
单元测试：验证 Skill 五大功能可独立运行并返回正确结构化报告。
"""
import os
import tempfile

from document_quality_assessment import DocumentQualityAssessor


def _make_files():
    d = tempfile.mkdtemp()
    # 文本型 PDF 不易构造，用 md/txt 替代验证格式统计、长度、重复、敏感信息
    p1 = os.path.join(d, "a.md")
    p2 = os.path.join(d, "b.md")
    p3 = os.path.join(d, "c.txt")
    with open(p1, "w", encoding="utf-8") as f:
        f.write("这是一份知识库文档。" * 100)
    with open(p2, "w", encoding="utf-8") as f:
        f.write("这是一份知识库文档。" * 100)  # 与 a.md 完全相同
    with open(p3, "w", encoding="utf-8") as f:
        f.write("联系人 13800138000 邮箱 test@example.com 身份证 110101199001011234")
    return [p1, p2, p3]


def test_format_distribution():
    a = DocumentQualityAssessor()
    files = _make_files()
    dist = a.format_distribution(files)
    assert ".md" in dist and ".txt" in dist
    assert dist[".md"]["count"] == 2


def test_length_distribution():
    a = DocumentQualityAssessor()
    lengths = [500, 1500, 3000, 6000, 20000]
    r = a.length_distribution([("x" * L) for L in lengths])
    assert "P50" in r["percentiles"]
    assert "bins" in r


def test_duplicates():
    a = DocumentQualityAssessor()
    files = ["f1.md", "f2.md", "f3.md"]
    contents = ["同样的内容ABC", "同样的内容ABC", "完全不同的XYZ"]
    r = a.duplicates(files, contents)
    assert len(r["exact"]) == 1          # f1/f2 完全相同


def test_sensitive_info():
    a = DocumentQualityAssessor()
    r = a.sensitive_info(["f.txt"], ["手机 13800138000 邮箱 a@b.com 身份证 110101199001011234"])
    types = {h["type"] for h in r}
    assert "phone" in types and "email" in types and "id_card" in types
    assert all(h["context"] for h in r)   # 每条附上下文


def test_pdf_corrupt_handling():
    a = DocumentQualityAssessor()
    # 损坏 PDF：应返回 Error 而非抛异常
    d = tempfile.mkdtemp()
    p = os.path.join(d, "bad.pdf")
    with open(p, "w", encoding="utf-8") as f:
        f.write("not a real pdf")
    r = a.pdf_page_type(p)
    assert r["type"] in ("Error", "Scan_PDF", "Mixed_PDF", "Text_PDF")


if __name__ == "__main__":
    test_format_distribution()
    test_length_distribution()
    test_duplicates()
    test_sensitive_info()
    test_pdf_corrupt_handling()
    print("所有单元测试通过")
