import io
import zipfile
from pathlib import Path

import pytest

from src.edu_rag_ingest.config.config import MinerUConfig
from src.edu_rag_ingest.ingestion.mineru_client import MinerUClient


def make_config(tmp_path: Path) -> MinerUConfig:
    return MinerUConfig(
        enabled=True,
        api_url="https://mineru.example.com",
        api_key="test-key",
        timeout_seconds=10,
        upload_field="file",
        extra_form={},
        response_text_fields=[],
        parsed_output_dir=tmp_path / "parsed",
        artifact_output_dir=tmp_path / "mineru",
    )


def build_zip(entries: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            archive.writestr(name, data)
    return buffer.getvalue()


def test_save_artifacts_from_zip_keeps_complete_mineru_outputs(tmp_path):
    client = MinerUClient(make_config(tmp_path))
    source_path = tmp_path / "测试文档.pdf"
    archive_bytes = build_zip(
        {
            "result.md": "# 标题\n\n解析正文",
            "layout/result.json": '{"pages": 1}',
            "images/page-1.png": b"png-bytes",
        }
    )

    markdown = client._save_artifacts_from_zip(source_path, archive_bytes)
    parsed_path = client.save_parsed_text(source_path, markdown)

    assert markdown == "# 标题\n\n解析正文"
    assert parsed_path.read_text(encoding="utf-8") == markdown
    assert (tmp_path / "mineru" / "测试文档.zip").is_file()
    assert (tmp_path / "mineru" / "测试文档" / "result.md").read_text(encoding="utf-8") == markdown
    assert (tmp_path / "mineru" / "测试文档" / "layout" / "result.json").read_text(encoding="utf-8") == '{"pages": 1}'
    assert (tmp_path / "mineru" / "测试文档" / "images" / "page-1.png").read_bytes() == b"png-bytes"


def test_save_artifacts_from_zip_rejects_unsafe_paths(tmp_path):
    client = MinerUClient(make_config(tmp_path))
    archive_bytes = build_zip({"../evil.md": "bad"})

    with pytest.raises(RuntimeError, match="不安全路径"):
        client._save_artifacts_from_zip(tmp_path / "测试文档.pdf", archive_bytes)
