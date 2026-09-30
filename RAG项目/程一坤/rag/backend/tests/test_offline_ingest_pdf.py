"""批次 22：离线入库的 PDF 通道单测（app/ingest/offline_ingest.py）。

本批次修的两个缺口：
1. `build_package` 调 `parse_document` 时只传 1 个参数 → PDF 必抛 ValueError；
   现在把 MinerU / Qwen-VL 客户端注入下去（未注入时按配置懒装配）。
2. `collect_raw_files` 只收 .html/.htm → 扩展名补上 .pdf。

另外校验 version.media_type 不再一律写 text/html（PDF 应落 application/pdf）。
"""

from pathlib import Path

import pytest

from app.ingest.offline_ingest import (
    build_package,
    collect_raw_files,
    process_raw_directory,
)

PDF_FIXTURE_BYTES = b"%PDF-1.4 fake pdf bytes"

MARKDOWN_BODY = (
    "第一章 总则\n\n"
    "第一条 为了规范劳动合同制度，制定本条例。\n\n"
    "第二条 用人单位与劳动者建立劳动关系，应当订立书面劳动合同。\n\n"
    "第二章 劳动合同的订立\n\n"
    "第三条 订立劳动合同，应当遵循合法、公平、平等自愿的原则。"
)

# MinerU 会把部分条文行提升为 Markdown 标题（批次 22 真实输出形态）
MARKDOWN_BODY_WITH_HEADINGS = (
    "# 中华人民共和国劳动合同法\n\n"
    "## 目 录\n\n"
    "第一章 总则\n\n"
    "## 第一章 总则\n\n"
    "第一条 为了完善劳动合同制度。\n\n"
    "## 第十六条 劳动合同由用人单位与劳动者协商一致签订。\n\n"
    "## 第十七条 劳动合同应当具备以下条款:\n\n"
    "（一）用人单位的名称、住所和法定代表人或者主要负责人；\n\n"
    "（二）劳动者的姓名、住址和居民身份证或者其他有效身份证件号码。\n\n"
    "## 第二章 劳动合同的订立\n\n"
    "第十八条 劳动合同对劳动报酬和劳动条件等标准约定不明确的，适用本条例。"
)


class StubMineru:
    def __init__(self, content: str = MARKDOWN_BODY, page_count: int = 27) -> None:
        self.content = content
        self.page_count = page_count
        self.calls = 0

    def parse(self, source_path: Path) -> dict:
        self.calls += 1
        return {
            "content": self.content,
            "page_count": self.page_count,
            "complete": True,
            "error": "",
        }


class StubQwenVl:
    def __init__(self, text: str = "兜底文字") -> None:
        self.text = text
        self.calls = 0

    def parse(self, source_path: Path) -> str:
        self.calls += 1
        return self.text


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "contract_law.pdf"
    path.write_bytes(PDF_FIXTURE_BYTES)
    return path


# ------------------------------------------------------- collect_raw_files


def test_collect_raw_files_includes_pdf(tmp_path: Path):
    (tmp_path / "b.html").write_text("<p>x</p>", encoding="utf-8")
    (tmp_path / "a.htm").write_text("<p>x</p>", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(PDF_FIXTURE_BYTES)
    (tmp_path / "d.txt").write_text("x", encoding="utf-8")  # 仍不在收集范围

    collected = collect_raw_files(tmp_path)

    assert collected == [
        tmp_path / "a.htm",
        tmp_path / "b.html",
        tmp_path / "c.pdf",
    ]


def test_collect_raw_files_extension_case_insensitive(tmp_path: Path):
    (tmp_path / "A.PDF").write_bytes(PDF_FIXTURE_BYTES)
    assert collect_raw_files(tmp_path) == [tmp_path / "A.PDF"]


# ----------------------------------------------------------- build_package


def test_build_package_pdf_uses_injected_clients(pdf_file: Path):
    mineru = StubMineru()
    qwen = StubQwenVl()

    package = build_package(
        pdf_file,
        "https://example.test/law.pdf",
        "example-law",
        "劳动合同法",
        mineru_client=mineru,
        qwen_vl_client=qwen,
    )

    assert mineru.calls == 1
    assert qwen.calls == 0
    # PDF 的媒体类型必须落 application/pdf，不能再是 text/html
    assert package.version.media_type == "application/pdf"
    # 章节标题经清洗后仍在（cleaner 白名单生效）
    assert "第一章 总则" in package.version.cleaned_content
    assert "第二章 劳动合同的订立" in package.version.cleaned_content
    assert package.chunks
    assert package.document.source_url == "https://example.test/law.pdf"
    assert package.document.title == "劳动合同法"


def test_build_package_pdf_chunks_keep_article_numbers(pdf_file: Path):
    package = build_package(
        pdf_file,
        "https://example.test/law.pdf",
        "example-law",
        "劳动合同法",
        mineru_client=StubMineru(),
        qwen_vl_client=StubQwenVl(),
    )

    article_numbers = [c.article_number for c in package.chunks if c.chunk_type == "parent"]
    assert article_numbers == ["第一条", "第二条", "第三条"]


def test_build_package_pdf_without_clients_assembles_from_settings(
    pdf_file: Path, monkeypatch
):
    """未显式注入时，按应用配置懒装配（这里替换装配函数，避免真实联网）。"""
    mineru = StubMineru()
    qwen = StubQwenVl()
    monkeypatch.setattr(
        "app.ingest.offline_ingest._default_pdf_clients", lambda: (mineru, qwen)
    )

    package = build_package(
        pdf_file, "https://example.test/law.pdf", "example-law", "劳动合同法"
    )

    assert mineru.calls == 1
    assert package.version.media_type == "application/pdf"


def test_build_package_pdf_without_title_is_rejected(pdf_file: Path):
    """PDF 无内嵌标题，禁止编造：没给人工核对标题就报错，且提示语要点明 PDF。"""
    with pytest.raises(ValueError) as excinfo:
        build_package(
            pdf_file,
            "https://example.test/law.pdf",
            "example-law",
            mineru_client=StubMineru(),
            qwen_vl_client=StubQwenVl(),
        )

    assert "PDF 没有内嵌文档标题" in str(excinfo.value)


def test_pdf_markdown_headings_do_not_break_article_boundary(pdf_file: Path):
    """MinerU 把条文行提升成 '## 第十七条 …' 时，第十七条仍必须是独立父块。

    真实回归：不加 Markdown 标记剥离，PDF 侧会漏掉这一条（被并进第十六条），
    与库内 HTML 的 98 条对不上。
    """
    package = build_package(
        pdf_file,
        "https://example.test/law.pdf",
        "example-law",
        "劳动合同法",
        mineru_client=StubMineru(MARKDOWN_BODY_WITH_HEADINGS),
        qwen_vl_client=StubQwenVl(),
    )

    article_numbers = [c.article_number for c in package.chunks if c.chunk_type == "parent"]
    assert article_numbers == ["第一条", "第十六条", "第十七条", "第十八条"]
    # Markdown 标记不应残留到正文里
    assert "#" not in package.version.cleaned_content
    # 第十七条 的两个项要识别到
    items = [c.item_number for c in package.chunks if c.item_number]
    assert items == ["1", "2"]


def test_pdf_markdown_heading_markers_removed(pdf_file: Path):
    package = build_package(
        pdf_file,
        "https://example.test/law.pdf",
        "example-law",
        "劳动合同法",
        mineru_client=StubMineru(MARKDOWN_BODY_WITH_HEADINGS),
        qwen_vl_client=StubQwenVl(),
    )

    content = package.version.cleaned_content
    assert "## 第一章 总则" not in content
    assert "第一章 总则" in content
    assert "目 录" in content


def test_build_package_html_still_works_without_clients(tmp_path: Path):
    """HTML 通道不受影响：不注入客户端也能打包。"""
    html = tmp_path / "law.html"
    html.write_text(
        "<html><body><p>第一条 为了规范劳动合同制度。</p></body></html>",
        encoding="utf-8",
    )

    package = build_package(html, "https://example.test/law.html", "example-law", "测试法")

    assert package.version.media_type == "text/html"
    assert "第一条" in package.version.cleaned_content


def test_build_package_pdf_failure_is_reported_not_silent(pdf_file: Path):
    """两个客户端都给不出正文时，必须抛错（而不是产出空包）。"""

    class DeadMineru:
        def parse(self, source_path: Path) -> dict:
            return {"content": "", "page_count": 0, "complete": False, "error": "tasks failed"}

    class DeadQwen:
        def parse(self, source_path: Path) -> str:
            return ""

    with pytest.raises(RuntimeError):
        build_package(
            pdf_file,
            "https://example.test/law.pdf",
            "example-law",
            mineru_client=DeadMineru(),
            qwen_vl_client=DeadQwen(),
        )


# ------------------------------------------------- process_raw_directory


def test_process_raw_directory_handles_pdf(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "law.pdf").write_bytes(PDF_FIXTURE_BYTES)
    output_root = tmp_path / "processed"

    result = process_raw_directory(
        raw_root,
        output_root,
        {"law.pdf": "https://example.test/law.pdf"},
        {"law.pdf": "example-law"},
        {"law.pdf": "劳动合同法"},
        mineru_client=StubMineru(),
        qwen_vl_client=StubQwenVl(),
    )

    assert result.failure_count == 0, result.errors
    assert result.success_count == 1


def test_process_raw_directory_mixes_html_and_pdf(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "law.html").write_text(
        "<html><body><p>第一条 正文。</p></body></html>", encoding="utf-8"
    )
    (raw_root / "law.pdf").write_bytes(PDF_FIXTURE_BYTES)
    output_root = tmp_path / "processed"

    result = process_raw_directory(
        raw_root,
        output_root,
        {
            "law.html": "https://example.test/law.html",
            "law.pdf": "https://example.test/law.pdf",
        },
        {"law.html": "example-html", "law.pdf": "example-pdf"},
        {"law.html": "劳动合同法（HTML）", "law.pdf": "劳动合同法（PDF）"},
        mineru_client=StubMineru(),
        qwen_vl_client=StubQwenVl(),
    )

    assert result.failure_count == 0, result.errors
    assert result.success_count == 2
