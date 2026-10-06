import pytest

from backend.app.files import pdf_file_sha256, pdf_sha256, sanitize_pdf_name, save_pdf_bytes


def test_pdf_sha256_is_stable_and_content_sensitive():
    assert pdf_sha256(b"%PDF-1.7") == pdf_sha256(b"%PDF-1.7")
    assert pdf_sha256(b"%PDF-1.7") != pdf_sha256(b"%PDF-1.8")


def test_pdf_file_sha256_matches_content_hash(tmp_path):
    target = tmp_path / "a.pdf"
    target.write_bytes(b"%PDF-1.7")

    assert pdf_file_sha256(target) == pdf_sha256(b"%PDF-1.7")


def test_sanitize_pdf_name_keeps_pdf_extension():
    assert sanitize_pdf_name("报告 2026.pdf").endswith(".pdf")


def test_sanitize_pdf_name_rejects_non_pdf():
    with pytest.raises(ValueError, match="PDF"):
        sanitize_pdf_name("a.txt")


def test_save_pdf_bytes_writes_under_data_dir(tmp_path):
    saved = save_pdf_bytes(tmp_path, "a.pdf", b"%PDF-1.7")

    assert saved.exists()
    assert saved.read_bytes() == b"%PDF-1.7"
