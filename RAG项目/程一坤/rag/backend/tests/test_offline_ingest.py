from pathlib import Path

import pytest

from app.ingest.offline_ingest import (
    build_package,
    collect_raw_files,
    process_raw_directory,
)
from app.pipeline.package_models import PackageData
from app.pipeline.package_validator import MANIFEST_FILE, validate_package


def test_collect_raw_files_returns_sorted_html_files(tmp_path: Path) -> None:
    (tmp_path / "b.html").write_text("<p>第二条</p>", encoding="utf-8")
    (tmp_path / "a.htm").write_text("<p>第一条</p>", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")

    assert collect_raw_files(tmp_path) == [tmp_path / "a.htm", tmp_path / "b.html"]


def test_build_package_contains_parse_and_chunk_results(tmp_path: Path) -> None:
    source = tmp_path / "law.html"
    source.write_text(
        "<html><head><title>中华人民共和国劳动合同法实施条例</title></head>"
        "<body><h1>中华人民共和国劳动合同法实施条例</h1><p>第一条 适用范围。</p></body></html>",
        encoding="utf-8",
    )

    package = build_package(
        source,
        source_url="https://example.test/law.html",
        source_id="example",
    )

    assert package.document.source_url == "https://example.test/law.html"
    assert package.version.raw_file_path == "raw/law.html"
    assert package.version.cleaned_content
    assert package.chunks
    assert package.crawl.http_status == 200


def test_process_raw_directory_publishes_validated_packages_and_continues_after_failure(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "packages"
    raw_root.mkdir()
    (raw_root / "good.html").write_text(
        "<html><head><title>中华人民共和国劳动法</title></head>"
        "<body><p>中华人民共和国劳动法</p><p>第一条 有效内容。</p></body></html>",
        encoding="utf-8",
    )
    (raw_root / "bad.html").write_bytes(b"\xff\xfe")

    result = process_raw_directory(
        raw_root,
        output_root,
        {"good.html": "https://example.test/good"},
        document_titles={},
    )

    package_root = next(output_root.iterdir())
    validated = validate_package(package_root)
    assert result.success_count == 1
    assert result.failure_count == 1
    assert (package_root / MANIFEST_FILE).is_file()
    assert (package_root / "raw" / "good.html").is_file()
    assert PackageData.read_records(package_root).chunks
    assert validated.data.document.source_url == "https://example.test/good"
    assert not list(output_root.glob("*.tmp"))
    assert result.errors[0].filename == "bad.html"


def test_process_raw_directory_uses_unique_output_directories_for_colliding_urls(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "packages"
    raw_root.mkdir()
    (raw_root / "first.html").write_text(
        "<html><head><title>中华人民共和国劳动法</title></head>"
        "<body><p>中华人民共和国劳动法</p><p>第一条 内容一。</p></body></html>",
        encoding="utf-8",
    )
    (raw_root / "second.html").write_text(
        "<html><head><title>中华人民共和国劳动合同法</title></head>"
        "<body><p>中华人民共和国劳动合同法</p><p>第一条 内容二。</p></body></html>",
        encoding="utf-8",
    )

    result = process_raw_directory(
        raw_root,
        output_root,
        {
            "first.html": "https://example.test/law.html?version=one",
            "second.html": "https://other.test/law.html?version=two",
        },
        document_titles={},
    )

    package_roots = [path for path in output_root.iterdir() if path.is_dir()]
    assert result.success_count == 2
    assert len(package_roots) == 2
    assert all((path / MANIFEST_FILE).is_file() for path in package_roots)


def test_process_raw_directory_rejects_missing_raw_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        process_raw_directory(
            tmp_path / "missing",
            tmp_path / "packages",
            {},
            document_titles={},
        )
