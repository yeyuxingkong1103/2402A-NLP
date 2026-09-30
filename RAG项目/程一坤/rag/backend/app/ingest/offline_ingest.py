from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

from app.ingest.chunk_fingerprint import compute_chunk_fingerprint
from app.ingest.chunker import chunk_document
from app.ingest.parser import parse_document
from app.ingest.title_normalizer import _extract_document_title
from app.pipeline.package_models import (
    ChunkEntry,
    CrawlEntry,
    DocumentEntry,
    PackageData,
    PackageFormatError,
    VersionEntry,
    stable_document_id,
    stable_version_id,
)
from app.pipeline.package_writer import publish_package


@dataclass(frozen=True)
class OfflineIngestError:
    filename: str
    message: str


@dataclass(frozen=True)
class OfflineIngestResult:
    success_count: int
    failure_count: int
    errors: tuple[OfflineIngestError, ...]


def collect_raw_files(raw_root: Path) -> list[Path]:
    """收集待入库的原始文件：HTML 与 PDF。

    PDF 于批次 22 接入（此前只收 .html/.htm，PDF 根本进不了入库流程）。
    """
    return sorted(
        path
        for path in raw_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".html", ".htm", ".pdf"}
    )


# 内容格式 → HTTP 媒体类型（version 记录里落库，供前端/导出侧判断载体）
_MEDIA_TYPES = {
    "html": "text/html",
    "pdf": "application/pdf",
    "txt": "text/plain",
    "markdown": "text/markdown",
}


def _default_pdf_clients() -> tuple[object, object]:
    """按应用配置装配 PDF 解析客户端（MinerU 主路径 + Qwen-VL 兜底）。

    只在这一刻才 import，避免非 PDF 入库流程被迫依赖 PDF 相关配置。
    """
    from app.models.mineru import build_mineru_client_from_settings
    from app.models.qwen_vl import build_qwen_vl_client_from_settings

    return build_mineru_client_from_settings(), build_qwen_vl_client_from_settings()


def build_package(
    source_path: Path,
    source_url: str,
    source_id: str,
    explicit_title: str | None = None,
    mineru_client: object | None = None,
    qwen_vl_client: object | None = None,
) -> PackageData:
    raw_content = source_path.read_bytes()
    raw_content_hash = sha256(raw_content).hexdigest()
    # PDF 走 MinerU/Qwen-VL 远程解析，客户端未显式传入时按配置装配
    if source_path.suffix.lower() == ".pdf" and (
        mineru_client is None or qwen_vl_client is None
    ):
        mineru_client, qwen_vl_client = _default_pdf_clients()
    parsed_document = parse_document(
        source_path,
        mineru_client=mineru_client,
        qwen_vl_client=qwen_vl_client,
    )
    # 语义哈希：对清洗后正文计算，避免页面动态时间戳、导航栏等无关内容导致版本误判
    semantic_hash = sha256(parsed_document.content.encode("utf-8")).hexdigest()
    document_id = stable_document_id(source_url)
    title = _extract_document_title(source_path, parsed_document.content, explicit_title)
    package_raw_path = f"raw/{source_path.name}"
    document_chunks = chunk_document(
        parsed_document,
        document_title=title,
        document_id=document_id,
    )

    if not document_chunks:
        raise ValueError("解析结果未生成任何 chunk")

    # 切块指纹：切块器版本 + 分块结果（条/款/项结构）的哈希，供导入侧
    # 双比较判定"切块是否变化"（content_hash 只反映正文，感知不到切块变更）
    chunk_fingerprint = compute_chunk_fingerprint(
        [
            (
                chunk.chunk_level,
                chunk.article_no,
                chunk.paragraph_no,
                chunk.item_no,
                chunk.content,
            )
            for chunk in sorted(document_chunks, key=lambda c: c.sequence)
        ]
    )
    # version_id 纳入指纹：正文相同但切块变化时，新版本必须有新的
    # version_key（该列全局唯一），否则同一 version_key 插入必撞唯一约束
    version_id = stable_version_id(document_id, f"{semantic_hash}:{chunk_fingerprint}")

    chunks = tuple(
        ChunkEntry(
            chunk_id=chunk.chunk_id,
            version_id=version_id,
            parent_chunk_id=chunk.parent_id,
            chunk_type=chunk.chunk_level,
            article_number=chunk.article_no,
            paragraph_number=chunk.paragraph_no,
            item_number=chunk.item_no,
            sequence=chunk.sequence,
            content=chunk.content,
            retrieval_text=chunk.retrieval_content,
        )
        for chunk in document_chunks
    )

    return PackageData(
        document=DocumentEntry(
            document_id=document_id,
            source_url=source_url,
            title=title,
        ),
        version=VersionEntry(
            version_id=version_id,
            document_id=document_id,
            content_hash=semantic_hash,
            cleaned_content=parsed_document.content,
            raw_file_path=package_raw_path,
            # 媒体类型随实际解析出的格式走（PDF 接入后不能再一律写 text/html）
            media_type=_MEDIA_TYPES.get(
                parsed_document.content_format, "text/plain"
            ),
            # 切块指纹随包落盘，导入侧据此双比较
            chunk_fingerprint=chunk_fingerprint,
        ),
        chunks=chunks,
        crawl=CrawlEntry(
            crawl_id=f"crawl-{raw_content_hash[:24]}",
            source_id=source_id,
            source_url=source_url,
            http_status=200,
            content_hash=raw_content_hash,
            raw_file_path=package_raw_path,
            collected_at=datetime.now(UTC).isoformat(),
        ),
    )


def process_raw_directory(
    raw_root: Path,
    output_root: Path,
    source_urls: dict[str, str],
    source_ids: dict[str, str] | None = None,
    document_titles: dict[str, str | None] = None,
    force: bool = False,
    mineru_client: object | None = None,
    qwen_vl_client: object | None = None,
) -> OfflineIngestResult:
    """离线解析 raw 目录全部 HTML/PDF 并打包到 output_root。

    PDF 客户端可显式注入（测试/自定义装配）；未注入时遇到 PDF 才按配置装配。

    force 参数（默认 False，行为与历史版本一致）：
    - False：目标包目录已存在 → 该文件记为 failed（"package directory already exists"），
      禁止静默覆盖
    - True：显式覆盖已存在的包目录——覆盖前打印"即将覆盖"，并把旧包整体搬到
      输出根目录旁边的备份目录（<output_root 名>_backup_<时间戳>，与 output_root
      同级而非其子目录，避免被包发现逻辑误扫），不直接删除任何数据
    """
    errors: list[OfflineIngestError] = []
    success_count = 0
    resolved_source_ids = source_ids or {}
    resolved_document_titles = document_titles or {}
    # 备份目录后缀整轮取一次：同一轮 --force 的全部旧包进同一个备份目录，
    # 避免打包跨秒时散落多个时间戳目录
    force_backup_suffix = f"{datetime.now(UTC):%Y%m%d_%H%M%S}"

    for raw_file in collect_raw_files(raw_root):
        try:
            source_url = source_urls[raw_file.name]
            source_id = resolved_source_ids.get(
                raw_file.name,
                _safe_output_name(source_url),
            )
            explicit_title = resolved_document_titles.get(raw_file.name)
            # Lazy 装配：只有真的遇到 PDF 才去读 PDF 相关配置
            if raw_file.suffix.lower() == ".pdf" and (
                mineru_client is None or qwen_vl_client is None
            ):
                mineru_client, qwen_vl_client = _default_pdf_clients()
            package = build_package(
                raw_file,
                source_url,
                source_id,
                explicit_title,
                mineru_client=mineru_client,
                qwen_vl_client=qwen_vl_client,
            )
            package_root = output_root / _package_directory_name(
                source_url,
                package.document.document_id,
            )
            if package_root.exists():
                # 默认拒绝覆盖（与 package_writer 的保护一致，提前给出同样错误）
                if not force:
                    raise PackageFormatError("package directory already exists")
                # --force 路径：先备份旧包再覆盖，旧数据永不直接删除
                backup_directory = output_root.parent / (
                    f"{output_root.name}_backup_{force_backup_suffix}"
                )
                backup_target = backup_directory / package_root.name
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                print(f"即将覆盖 {package_root}")
                print(f"旧包备份到 {backup_target}")
                shutil.move(str(package_root), str(backup_target))
            publish_package(
                package_root,
                package,
                raw_file.read_bytes(),
                package_id=package_root.name,
            )
            success_count += 1
        except KeyError:
            errors.append(
                OfflineIngestError(
                    filename=raw_file.name,
                    message="缺少已验证 source_url 映射",
                )
            )
        except (OSError, UnicodeError, ValueError, PackageFormatError) as error:
            errors.append(
                OfflineIngestError(
                    filename=raw_file.name,
                    message=str(error),
                )
            )

    return OfflineIngestResult(
        success_count=success_count,
        failure_count=len(errors),
        errors=tuple(errors),
    )


def _package_directory_name(source_url: str, document_id: str) -> str:
    return f"{_safe_output_name(source_url)}-{document_id}"


def _safe_output_name(source_url: str) -> str:
    parsed_url = urlparse(source_url)
    last_path_part = Path(parsed_url.path).stem or parsed_url.netloc or "document"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", last_path_part).strip("-")
    return safe_name or "document"

