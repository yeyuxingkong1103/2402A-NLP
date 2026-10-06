"""批次 22：PDF 解析编排分支单测（app/ingest/pdf_parser.py）。

契约：MinerU 完整 → 直接用；不完整 → Qwen-VL 兜底；都没有 → PdfParsingError。
用替身客户端覆盖全部三条分支，以及输入文件缺失、空白正文等边界。
"""

from pathlib import Path

import pytest

from app.ingest.pdf_parser import PdfParsingError, parse_pdf


class StubMineru:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls = 0

    def parse(self, source_path: Path) -> dict:
        self.calls += 1
        return self.result


class StubQwenVl:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def parse(self, source_path: Path) -> str:
        self.calls += 1
        return self.text


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def test_missing_file_raises_file_not_found(tmp_path: Path):
    mineru = StubMineru({"content": "", "page_count": 0, "complete": False, "error": "x"})
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "nope.pdf", mineru, StubQwenVl("x"))


def test_complete_mineru_result_short_circuits(pdf_file):
    mineru = StubMineru(
        {"content": "第一条 正文。", "page_count": 10, "complete": True, "error": ""}
    )
    qwen = StubQwenVl("不该被调用")

    parsed = parse_pdf(pdf_file, mineru, qwen)

    assert parsed.parser_name == "mineru"
    assert parsed.content == "第一条 正文。"
    assert parsed.page_count == 10
    assert qwen.calls == 0  # 完整时不浪费 Qwen-VL 额度


def test_incomplete_mineru_falls_back_to_qwen_vl(pdf_file):
    mineru = StubMineru(
        {"content": "", "page_count": 27, "complete": False, "error": "任务超时"}
    )
    qwen = StubQwenVl("OCR 出来的文字")

    parsed = parse_pdf(pdf_file, mineru, qwen)

    assert parsed.parser_name == "qwen-vl"
    assert parsed.content == "OCR 出来的文字"
    # MinerU 报了非 0 页数时沿用它的值（页数来源随分支变化，优先 MinerU）
    assert parsed.page_count == 27
    assert qwen.calls == 1


def test_fallback_uses_qwen_page_count_when_mineru_reports_zero(pdf_file):
    """MinerU 失败到页数都是 0 时，页数改用 Qwen-VL 渲染出的真实页数。"""
    mineru = StubMineru({"content": "", "page_count": 0, "complete": False, "error": "任务超时"})
    qwen = StubQwenVl("OCR 出来的文字")
    qwen.last_trace = {"page_count": 10}

    parsed = parse_pdf(pdf_file, mineru, qwen)

    assert parsed.parser_name == "qwen-vl"
    assert parsed.page_count == 10  # 不再报 0


def test_fallback_page_count_is_zero_when_qwen_reports_nothing(pdf_file):
    """Qwen-VL 没给出页数（渲染前就失败）时，退回 0 而不是抛异常。"""
    mineru = StubMineru({"content": "", "page_count": 0, "complete": False, "error": "任务超时"})
    qwen = StubQwenVl("OCR 出来的文字")
    qwen.last_trace = {}

    parsed = parse_pdf(pdf_file, mineru, qwen)

    assert parsed.page_count == 0


def test_fallback_tolerates_broken_qwen_trace(pdf_file):
    """Qwen-VL 的解析痕迹不是字典（替身/异常实现）时不能把主流程带崩。"""
    mineru = StubMineru({"content": "", "page_count": 0, "complete": False, "error": "任务超时"})
    qwen = StubQwenVl("OCR 出来的文字")
    qwen.last_trace = None

    parsed = parse_pdf(pdf_file, mineru, qwen)

    assert parsed.page_count == 0


def test_blank_mineru_content_treated_as_incomplete(pdf_file):
    mineru = StubMineru(
        {"content": "   ", "page_count": 3, "complete": True, "error": ""}
    )
    qwen = StubQwenVl("兜底文字")

    parsed = parse_pdf(pdf_file, mineru, qwen)

    assert parsed.parser_name == "qwen-vl"
    assert qwen.calls == 1


def test_both_empty_raises_pdf_parsing_error(pdf_file):
    mineru = StubMineru(
        {"content": "", "page_count": 0, "complete": False, "error": "MinerU 任务失败：x"}
    )
    qwen = StubQwenVl("")

    with pytest.raises(PdfParsingError) as excinfo:
        parse_pdf(pdf_file, mineru, qwen)

    # 报错必须带出 MinerU 的失败原因，否则只剩一句"解析失败"没法排查
    assert "MinerU 任务失败" in str(excinfo.value)


def test_qwen_blank_without_mineru_error_uses_default_message(pdf_file):
    mineru = StubMineru({"content": "", "page_count": 0, "complete": False})
    qwen = StubQwenVl("   ")

    with pytest.raises(PdfParsingError) as excinfo:
        parse_pdf(pdf_file, mineru, qwen)

    assert "远程解析未返回正文" in str(excinfo.value)
