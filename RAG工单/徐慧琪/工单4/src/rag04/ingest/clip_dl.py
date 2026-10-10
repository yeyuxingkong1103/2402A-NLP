# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""CLIP / reranker 权重下载。

实测约束：
  1. huggingface.co 不通（HTTP 000），hf-mirror.com 通（HTTP 200）
  2. huggingface_hub 设 HF_ENDPOINT 仍报 LocalEntryNotFoundError
     → 一律用 requests 直连 resolve/main/<file>
  3. CLIP 权重文件是 pytorch_model.bin（605MB），model.safetensors 返回 404
  4. 完整性以远端 Content-Length 为准（HEAD 探测）；探测不到才退回体积下限，
     保证离线且模型完整的用户仍可运行
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from rag04.config import Settings

logger = logging.getLogger("rag04.clip_dl")

CLIP_FILES = [
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "pytorch_model.bin",          # 实测：不是 model.safetensors
]

RERANKER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",    # 实测：XLM-R 仓库无 vocab.txt（404）
    "pytorch_model.bin",
]

_CHUNK = 1 << 20  # 1MB
_MIN_WEIGHT_BYTES = 1_000_000   # 离线回退时 .bin 的体积下限
_HEAD_TIMEOUT = 30.0


class ModelDownloadError(Exception):
    """模型下载失败（重试后仍不通）。"""


def _stream_to_file(resp, path: Path, mode: str) -> None:
    """响应体流式落盘（mode='ab' 续传 / 'wb' 覆盖）。"""
    with open(path, mode) as f:
        for chunk in resp.iter_content(chunk_size=_CHUNK):
            if chunk:
                f.write(chunk)


def _remote_size(url: str, timeout: float = _HEAD_TIMEOUT) -> int | None:
    """HEAD 探测远端字节数；离线/拿不到 Content-Length 时返回 None。"""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        raw = r.headers.get("Content-Length")
        if raw is None:
            return None
        size = int(raw)
        return size if size > 0 else None
    except (requests.RequestException, ValueError) as e:
        logger.warning("远端大小探测失败（%s）：%s", url, e)
        return None


def _is_complete(target: Path, url: str) -> bool:
    """本地文件是否与远端大小一致。

    远端可探测时以 Content-Length 为准（截断文件不再被体积下限放过）；
    探测不到（离线）则退回体积下限，只拦截明显半截的文件。
    """
    local = target.stat().st_size
    remote = _remote_size(url)
    if remote is not None:
        if local == remote:
            return True
        logger.warning("%s 不完整：本地 %d 字节 / 远端 %d 字节，重新下载",
                       target.name, local, remote)
        return False
    # 离线回退：保证「离线且模型完整」的用户仍可运行
    if target.name.endswith(".bin") and local < _MIN_WEIGHT_BYTES:
        logger.warning("%s 体积异常偏小（%d 字节），重新下载",
                       target.name, local)
        return False
    logger.warning("无法探测 %s 的远端大小，按体积下限判定为完整（离线回退）",
                   target.name)
    return True


def download_file(url: str, dest: Path, retries: int = 3,
                  timeout: float = 60.0) -> Path:
    """下载到 dest，支持断点续传与重试。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            have = tmp.stat().st_size if tmp.exists() else 0
            wrote_body = False
            if have:
                headers = {"Range": f"bytes={have}-"}
                with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                    if r.status_code == 416:
                        # .part 已 >= 远端（崩溃残留 / 上游文件变了），Range 永不可满足：
                        # 删除后在同一次尝试内从 0 重下，而不是反复重试同一 Range。
                        logger.warning(
                            "HTTP 416：%s 的断点 %d 字节不可用，删除 .part 从头下载",
                            dest.name, have)
                        tmp.unlink(missing_ok=True)
                        have = 0
                    else:
                        # 206=续传, 200=服务端忽略 Range 返回全量
                        if r.status_code == 200:
                            have = 0
                            tmp.unlink(missing_ok=True)
                        r.raise_for_status()
                        _stream_to_file(r, tmp, "ab" if have else "wb")
                        wrote_body = True
            if not wrote_body:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    _stream_to_file(r, tmp, "wb")
            tmp.replace(dest)
            logger.info("下载完成：%s（%.1f MB）", dest.name, dest.stat().st_size / 1e6)
            return dest
        except Exception as e:
            last_err = e
            logger.warning("第 %d/%d 次下载失败：%s（%s）", attempt, retries, dest.name, e)
            if attempt < retries:
                time.sleep(2 ** attempt)

    raise ModelDownloadError(f"下载失败：{url}（{type(last_err).__name__}: {last_err}）")


def _download_repo(repo: str, files: list[str], dest_dir: Path,
                   mirror: str, *, skip_big_if_present: bool = True) -> Path:
    """按文件清单下载整个仓库到 dest_dir。"""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = dest_dir / name
        url = f"{mirror.rstrip('/')}/{repo}/resolve/main/{name}"
        if skip_big_if_present and target.exists() and target.stat().st_size > 0:
            if _is_complete(target, url):
                continue
        download_file(url, target)
    return dest_dir


def ensure_clip(s: Settings) -> Path:
    """确保 CLIP 就位，返回本地目录。"""
    return _download_repo(
        s.clip_repo, CLIP_FILES, Path(s.models_dir) / "clip-vit-base-patch32",
        s.clip_hf_mirror,
    )


def ensure_reranker(s: Settings) -> Path:
    """确保 reranker 就位，返回本地目录。失败由调用方降级。"""
    return _download_repo(
        s.rerank_model, RERANKER_FILES,
        Path(s.models_dir) / "bge-reranker-base", s.clip_hf_mirror,
    )
