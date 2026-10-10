# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from pathlib import Path

import pytest
import requests

from rag04.config import get_settings
from rag04.ingest.clip_dl import CLIP_FILES, download_file, ensure_clip, ModelDownloadError


class _FakeHead:
    """HEAD 响应桩：实现只用 headers 与 raise_for_status。"""

    def __init__(self, length: int | None = None):
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.status_code = 200

    def raise_for_status(self):
        pass


def _seed_clip_dir(tmp_path: Path) -> Path:
    """造一个「看起来完整」的 CLIP 目录：权重 1.1MB（大于旧的 1MB 下限）。"""
    dest = tmp_path / "clip-vit-base-patch32"
    dest.mkdir(parents=True)
    for f in CLIP_FILES:
        # 权重须大于 1MB，否则会被体积下限判为半截文件而重下
        (dest / f).write_bytes(b"\0" * 1_100_000 if f.endswith(".bin") else b"x")
    return dest


def test_clip_files_use_bin_not_safetensors():
    assert "pytorch_model.bin" in CLIP_FILES
    assert "model.safetensors" not in CLIP_FILES, \
        "实测 model.safetensors 返回 404，必须用 pytorch_model.bin"
    for f in ("config.json", "preprocessor_config.json", "vocab.json", "merges.txt"):
        assert f in CLIP_FILES


def test_ensure_clip_returns_existing_dir_without_download(tmp_path, monkeypatch):
    """已存在完整模型时必须直接返回，不触发网络下载。

    远端大小探测走 HEAD（与 get 不同的调用），故毒丸同时覆盖 get 与 head；
    HEAD 不通即模拟离线，退回体积下限后完整目录仍应零下载返回。
    """
    import rag04.ingest.clip_dl as m

    def _no_download(*a, **k):
        raise AssertionError("完整模型已存在，不应触发网络下载")

    probed: list[str] = []

    def _offline_head(url, *a, **k):
        probed.append(url)
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(m.requests, "get", _no_download)
    monkeypatch.setattr(m.requests, "head", _offline_head)

    s = get_settings()
    object.__setattr__(s, "models_dir", tmp_path)
    dest = _seed_clip_dir(tmp_path)
    assert ensure_clip(s) == dest
    assert len(probed) == len(CLIP_FILES), "应对每个已存在文件做远端大小探测（HEAD）"


def test_ensure_clip_accepts_size_verified_files_without_download(tmp_path, monkeypatch):
    """HEAD 报出的远端大小与本地一致时，零下载返回（在线幂等常用路径）。"""
    import rag04.ingest.clip_dl as m

    s = get_settings()
    object.__setattr__(s, "models_dir", tmp_path)
    dest = _seed_clip_dir(tmp_path)

    def _head(url, *a, **k):
        name = url.rsplit("/", 1)[-1]
        return _FakeHead((dest / name).stat().st_size)

    def _no_download(*a, **k):
        raise AssertionError("远端大小一致，不应触发网络下载")

    monkeypatch.setattr(m.requests, "head", _head)
    monkeypatch.setattr(m.requests, "get", _no_download)
    assert ensure_clip(s) == dest


def test_ensure_clip_redownloads_truncated_weight(tmp_path, monkeypatch):
    """截断到 1.1MB（可通过旧体积下限）的权重必须按远端大小判为不完整并重下。"""
    import rag04.ingest.clip_dl as m

    s = get_settings()
    object.__setattr__(s, "models_dir", tmp_path)
    dest = _seed_clip_dir(tmp_path)
    binp = dest / "pytorch_model.bin"
    assert binp.stat().st_size == 1_100_000

    remote_sizes = {f: (dest / f).stat().st_size for f in CLIP_FILES}
    remote_sizes["pytorch_model.bin"] = 605_247_071   # 真实 CLIP 权重字节数
    downloaded: list[str] = []

    def _head(url, *a, **k):
        return _FakeHead(remote_sizes[url.rsplit("/", 1)[-1]])

    class FakeResp:
        status_code = 200
        headers = {}

        def iter_content(self, chunk_size=1):
            yield b"fresh-weight"

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _get(url, *a, **k):
        downloaded.append(url)
        return FakeResp()

    monkeypatch.setattr(m.requests, "head", _head)
    monkeypatch.setattr(m.requests, "get", _get)

    assert ensure_clip(s) == dest
    expected = f"{s.clip_hf_mirror.rstrip('/')}/{s.clip_repo}/resolve/main/pytorch_model.bin"
    assert downloaded == [expected], "只有截断的权重文件应被重下"
    assert binp.read_bytes() == b"fresh-weight"


def test_download_file_writes_content(tmp_path, monkeypatch):
    class FakeResp:
        status_code = 200
        headers = {"Content-Length": "5"}

        def iter_content(self, chunk_size=1):
            yield b"hello"

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import rag04.ingest.clip_dl as m
    monkeypatch.setattr(m.requests, "get", lambda *a, **k: FakeResp())
    dest = tmp_path / "f.bin"
    out = download_file("https://example.com/f.bin", dest)
    assert out.read_bytes() == b"hello"


def test_download_file_restarts_from_zero_on_416(tmp_path, monkeypatch):
    """HTTP 416（.part >= 远端）应在同一次尝试内删 .part 并从 0 重下。"""
    import rag04.ingest.clip_dl as m

    dest = tmp_path / "f.bin"
    part = tmp_path / "f.bin.part"
    stale = b"stale-part-larger-than-remote"
    part.write_bytes(stale)
    calls: list[dict] = []

    class FakeResp:
        def __init__(self, status, body):
            self.status_code = status
            self.headers = {}
            self._body = body

        def iter_content(self, chunk_size=1):
            yield self._body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _get(url, headers=None, *a, **k):
        calls.append(dict(headers or {}))
        if len(calls) == 1:
            return FakeResp(416, b"")
        return FakeResp(200, b"fresh")

    monkeypatch.setattr(m.requests, "get", _get)
    # retries=1：若实现靠「换一次重试」而非「同一次内重启」，此处必然 ModelDownloadError
    out = download_file("https://example.com/f.bin", dest, retries=1)

    assert out.read_bytes() == b"fresh"
    assert calls[0] == {"Range": f"bytes={len(stale)}-"}
    assert "Range" not in calls[1], "416 重下必须去掉 Range 头"
    assert not part.exists()


def test_download_file_raises_on_persistent_failure(tmp_path, monkeypatch):
    import rag04.ingest.clip_dl as m

    def boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(m.requests, "get", boom)
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    with pytest.raises(ModelDownloadError):
        download_file("https://example.com/f.bin", tmp_path / "f.bin", retries=2)


@pytest.mark.integration
def test_real_clip_download_and_file_sizes():
    """联网实测：确认权重约 605MB。"""
    s = get_settings()
    p = ensure_clip(s)
    binp = p / "pytorch_model.bin"
    assert binp.exists()
    assert binp.stat().st_size > 500_000_000, "权重体积异常，疑似下载不完整"
