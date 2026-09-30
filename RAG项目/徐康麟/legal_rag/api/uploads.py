# -*- coding: utf-8 -*-
"""上传校验与落盘（**内容 MD5 命名 = 幂等**）。

为什么按内容 MD5 命名
---------------------
1. **幂等**：同一份 PDF 反复上传，落盘路径与 ``doc_id`` 完全一致，
   ``KnowledgeBasePipeline`` 的 MD5 指纹去重会直接跳过，不会重复入库；
2. **天然防路径穿越**：文件名由内容派生，**不含用户输入**，``../`` 之类无从生效；
3. **doc_id 可推导**：ingest 用 ``path.stem`` 当 ``doc_id``，于是 ``doc_id == 内容 MD5``，
   ``DELETE /documents/{doc_id}`` 能直接定位到 Milvus 里的分块与磁盘上的文件。

即便如此，**用户提供的原始文件名仍然要净化并校验**（纵深防御 + 可读性）：
``sanitize_filename()`` 只保留最后一段路径、去掉控制字符与危险字符、限制长度。

被拒的上传一律计入 ``upload_rejected_total{reason}``，原因分类见
:data:`REJECT_REASONS`（便于按类别 grep / 告警）。
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

#: 允许的扩展名（与 ``legal_rag.ingest.loaders.SUPPORTED_SUFFIXES`` 对齐）：
#: PDF 为主力，兼容 txt/md/json。
ALLOWED_EXTENSIONS: tuple[str, ...] = (
    ".pdf", ".txt", ".md", ".markdown", ".json", ".jsonl", ".ndjson",
)
PDF_SUFFIXES = frozenset({".pdf"})
TEXT_SUFFIXES = frozenset(ALLOWED_EXTENSIONS) - PDF_SUFFIXES

#: ``RagConfig.upload.allowed_extensions`` 的默认值（config.py 已冻结，这里显式镜像）。
#: 若配置仍是这个「只允许 PDF」的默认值，就用上面更宽的 :data:`ALLOWED_EXTENSIONS`
#: （否则 t4 要求的「兼容 txt/md/json」会被配置默认值直接挡掉）；
#: **一旦运维显式配了别的组合，就完全以配置为准**（可收窄到只允许 .pdf）。
CONFIG_DEFAULT_EXTENSIONS: tuple[str, ...] = (".pdf",)

#: PDF 魔数（MIME 校验不只信 content_type，直接看内容头）
PDF_MAGIC = b"%PDF-"

#: 拒绝原因分类（upload_rejected_total 的 reason 标签取值）
REJECT_REASONS = (
    "no_files",           # 一个文件都没传
    "too_many_files",     # 超过 config.upload.max_upload_files
    "empty_file",         # 0 字节
    "too_large",          # 超过 config.upload.max_upload_mb
    "bad_extension",      # 扩展名不在白名单
    "bad_filename",       # 文件名净化后为空 / 非法
    "mime_mismatch",      # content_type 与扩展名明显不符
    "not_pdf",            # 声明 .pdf 但内容没有 %PDF- 魔数
    "not_text",           # 声明文本格式但内容不是可解码文本
    "unsupported_format",  # 其它（含 ingest 侧不支持的后缀）
)

_ILLEGAL_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_MULTI_UNDERSCORE = re.compile(r"_{2,}")
MAX_NAME_LENGTH = 120


class UploadRejected(ValueError):
    """上传被拒（带原因分类，便于计数与定位）。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        assert reason in REJECT_REASONS, f"未知的拒绝原因: {reason}"
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def effective_allowed_extensions(config: Any = None) -> tuple[str, ...]:
    """算出真正生效的扩展名白名单（见 :data:`CONFIG_DEFAULT_EXTENSIONS` 的说明）。"""
    if config is None:
        return ALLOWED_EXTENSIONS
    upload_cfg = getattr(config, "upload", None)
    configured = tuple(
        str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in (getattr(upload_cfg, "allowed_extensions", None) or ())
    )
    if not configured or configured == CONFIG_DEFAULT_EXTENSIONS:
        return ALLOWED_EXTENSIONS
    # 显式配置：与代码白名单取交集，防止配出 .exe 这种越界项
    narrowed = tuple(ext for ext in configured if ext in ALLOWED_EXTENSIONS)
    return narrowed or ALLOWED_EXTENSIONS


def sanitize_filename(name: str, *, fallback: str = "upload") -> str:
    """净化文件名，返回**仅文件名**（不含任何目录成分）。

    处理：Unicode 归一化 -> 反斜杠转正斜杠 -> 只取最后一段 -> 去控制/危险字符
    -> 折叠连续下划线 -> 限制长度。净化后为空则返回 ``fallback``。
    """
    text = unicodedata.normalize("NFKC", str(name or ""))
    text = text.replace("\\", "/")
    text = text.rsplit("/", 1)[-1]          # 丢掉所有目录成分（含 ../ 与 C:\）
    text = _ILLEGAL_CHARS.sub("_", text).strip().strip(".")
    text = _MULTI_UNDERSCORE.sub("_", text)
    if text in ("", ".", "..") or not text:
        return fallback
    if len(text) > MAX_NAME_LENGTH:
        stem, dot, ext = text.rpartition(".")
        if dot and len(ext) <= 10:
            keep = MAX_NAME_LENGTH - len(ext) - 1
            text = f"{stem[:max(keep, 1)]}.{ext}"
        else:
            text = text[:MAX_NAME_LENGTH]
    return text


def _looks_like_text(data: bytes) -> bool:
    """内容能否按 UTF-8（或常见中文编码）解码 —— 文本格式的轻量校验。"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            data.decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
    return False


def validate_upload(
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
    config: Any = None,
) -> tuple[str, str]:
    """校验单个上传，返回 ``(净化后的文件名, 扩展名)``；不合法则抛 :class:`UploadRejected`。

    ==================  ==========================================================
    校验项              规则
    ==================  ==========================================================
    空文件              0 字节 -> ``empty_file``
    体积                超过 ``config.upload.max_upload_mb`` -> ``too_large``
    文件名              净化后为空/非法 -> ``bad_filename``
    扩展名              不在白名单 -> ``bad_extension``
    MIME                声明与扩展名明显不符 -> ``mime_mismatch``
    内容                ``.pdf`` 必须有 ``%PDF-`` 魔数 -> ``not_pdf``；
                        文本格式必须可解码 -> ``not_text``
    ==================  ==========================================================
    """
    safe_name = sanitize_filename(filename)
    if not safe_name or safe_name == "upload":
        raise UploadRejected("bad_filename", f"净化后为空: {filename!r}")

    if not data:
        raise UploadRejected("empty_file", f"{safe_name} 是空文件")

    upload_cfg = getattr(config, "upload", None) if config is not None else None
    max_mb = int(getattr(upload_cfg, "max_upload_mb", 50) or 50)
    if len(data) > max_mb * 1024 * 1024:
        raise UploadRejected(
            "too_large", f"{safe_name} {len(data)} 字节 > {max_mb}MB")

    suffix = Path(safe_name).suffix.lower()
    allowed = effective_allowed_extensions(config)
    if suffix not in allowed:
        raise UploadRejected(
            "bad_extension", f"{safe_name} 后缀 {suffix or '(无)'} 不在 {list(allowed)}")
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadRejected("unsupported_format", f"{suffix} 无法被 ingest 解析")

    declared = (content_type or "").split(";")[0].strip().lower()
    if suffix in PDF_SUFFIXES:
        if declared and declared not in ("application/pdf", "application/x-pdf",
                                         "application/octet-stream", "binary/octet-stream"):
            raise UploadRejected("mime_mismatch", f"{safe_name} 声明 {declared}")
        if not data.lstrip()[:5].startswith(PDF_MAGIC):
            raise UploadRejected("not_pdf", f"{safe_name} 内容没有 %PDF- 魔数")
    else:
        if declared and not (declared.startswith("text/")
                             or declared in ("application/json",
                                             "application/x-ndjson",
                                             "application/octet-stream")):
            raise UploadRejected("mime_mismatch", f"{safe_name} 声明 {declared}")
        if not _looks_like_text(data):
            raise UploadRejected("not_text", f"{safe_name} 不是可解码文本")

    return safe_name, suffix


def content_md5(data: bytes) -> str:
    """内容 MD5（doc_id 与落盘文件名的来源）。"""
    return hashlib.md5(data).hexdigest()


def stored_name(digest: str, suffix: str) -> str:
    """落盘文件名：``<md5><后缀>``（不含任何用户输入）。"""
    return f"{digest}{suffix}"


class UploadStore:
    """把上传内容按 MD5 落到 ``config.upload_path``（幂等）。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.directory: Path = Path(getattr(config, "upload_path",
                                            Path("uploads")))

    def ensure_dir(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    def path_for(self, digest: str, suffix: str) -> Path:
        return self.directory / stored_name(digest, suffix)

    def save(self, data: bytes, suffix: str) -> tuple[Path, str, bool]:
        """写入 ``uploads/<md5><后缀>``；返回 ``(路径, md5, 是否新建)``。

        已存在且大小一致 => 视为命中（``created=False``），不重复写盘 —— 这就是「幂等」。
        """
        digest = content_md5(data)
        target = self.ensure_dir() / stored_name(digest, suffix)
        if target.is_file() and target.stat().st_size == len(data):
            logger.info("上传幂等命中（内容相同，跳过写盘）：%s", target.name)
            return target, digest, False
        tmp = target.with_name(f".{target.name}.part")
        tmp.write_bytes(data)
        os.replace(tmp, target)     # 原子落盘，避免半截文件被 ingest 读到
        logger.info("上传落盘：%s（%d 字节）", target.name, len(data))
        return target, digest, True

    def find_by_doc_id(self, doc_id: str) -> Path | None:
        """按 ``doc_id``（= 内容 MD5）找到已落盘的文件。"""
        if not self.directory.is_dir():
            return None
        for candidate in sorted(self.directory.glob(f"{doc_id}.*")):
            if candidate.is_file() and not candidate.name.startswith("."):
                return candidate
        return None

    def list_files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(p for p in self.directory.iterdir()
                      if p.is_file() and not p.name.startswith("."))

    def delete(self, path: Path) -> bool:
        try:
            path.unlink()
            logger.info("已删除上传文件：%s", path.name)
            return True
        except FileNotFoundError:
            return False


def summarize_supported(config: Any = None) -> dict[str, Any]:
    """给 ``/documents/upload`` 的自描述信息（也便于排障）。"""
    upload_cfg = getattr(config, "upload", None) if config is not None else None
    return {
        "allowed_extensions": list(effective_allowed_extensions(config)),
        "max_upload_mb": int(getattr(upload_cfg, "max_upload_mb", 50) or 50),
        "max_upload_files": int(getattr(upload_cfg, "max_upload_files", 10) or 10),
        "upload_dir": str(getattr(config, "upload_path", "uploads")),
    }


__all__ = [
    "ALLOWED_EXTENSIONS", "PDF_SUFFIXES", "TEXT_SUFFIXES", "PDF_MAGIC",
    "REJECT_REASONS", "CONFIG_DEFAULT_EXTENSIONS", "MAX_NAME_LENGTH",
    "UploadRejected", "UploadStore", "sanitize_filename", "validate_upload",
    "content_md5", "stored_name", "effective_allowed_extensions",
    "summarize_supported",
]
