"""文档加载测试：front-matter、标题回退、清洗、知识库发现。"""

from __future__ import annotations

from pathlib import Path

import pytest

from role_rag.errors import IngestError
from role_rag.ingest.loaders import discover_kb_files, load_document


def test_front_matter_parsed(tmp_path: Path):
    file = tmp_path / "a.md"
    file.write_text(
        "---\ntitle: 自定义标题\ntags: 资产配置, 风险\n---\n\n# 正文标题\n\n内容内容内容。",
        encoding="utf-8",
    )
    doc = load_document(file, role="financial_planner", scope="financial_planner")
    assert doc.title == "自定义标题"
    assert doc.tags == ["资产配置", "风险"]
    assert doc.text.startswith("# 正文标题")
    assert doc.role == "financial_planner" and doc.scope == "financial_planner"
    assert len(doc.doc_id) == 16


def test_title_falls_back_to_heading(tmp_path: Path):
    file = tmp_path / "b.md"
    file.write_text("# 一级标题\n\n正文。", encoding="utf-8")
    doc = load_document(file, role="scientist", scope="scientist")
    assert doc.title == "一级标题"


def test_title_falls_back_to_filename(tmp_path: Path):
    file = tmp_path / "c.md"
    file.write_text("没有标题的正文内容。", encoding="utf-8")
    doc = load_document(file, role="scientist", scope="scientist")
    assert doc.title == "c"


def test_text_cleaning(tmp_path: Path):
    file = tmp_path / "d.md"
    file.write_text("# 标题\n\n\n\n段落一   \n\n\n\n\n段落二\u200b", encoding="utf-8")
    doc = load_document(file, role="lawyer", scope="lawyer")
    assert "\n\n\n" not in doc.text
    assert "\u200b" not in doc.text
    assert "段落一\n\n段落二" in doc.text


def test_unsupported_suffix_rejected(tmp_path: Path):
    file = tmp_path / "e.json"
    file.write_text("{}", encoding="utf-8")
    with pytest.raises(IngestError):
        load_document(file, role="lawyer", scope="lawyer")


def test_empty_file_rejected(tmp_path: Path):
    file = tmp_path / "f.md"
    file.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(IngestError):
        load_document(file, role="lawyer", scope="lawyer")


def test_missing_file_rejected(tmp_path: Path):
    with pytest.raises(IngestError):
        load_document(tmp_path / "nope.md", role="lawyer", scope="lawyer")


def test_discover_kb_files(project_root: Path):
    files = discover_kb_files(project_root / "data" / "kb", "lawyer")
    assert len(files) == 4
    assert all(file.suffix == ".md" for file in files)
    assert discover_kb_files(project_root / "data" / "kb", "not_exists") == []


def test_doc_id_is_stable_and_scope_dependent(tmp_path: Path):
    file = tmp_path / "g.md"
    file.write_text("# 标题\n\n内容。", encoding="utf-8")
    first = load_document(file, role="lawyer", scope="lawyer")
    again = load_document(file, role="lawyer", scope="lawyer")
    other_scope = load_document(file, role="shared", scope="shared")
    assert first.doc_id == again.doc_id
    assert first.doc_id != other_scope.doc_id
