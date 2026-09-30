"""数据包增量判定：决定导入走 new / unchanged / updated 哪条路径。

判定是纯决策（只读库 + 纯比较），不写任何行；从 import_service.py
抽出，让"导入编排"与"增量规则"各自独立、规则可被单测直接覆盖。
判定契约（双比较：content_hash + chunk_fingerprint）：
- content_hash 不同 → 正文变化，new/updated 建新版本
- content_hash 相同、指纹相同（或一方未知）→ unchanged，跳过
- content_hash 相同、指纹都已知且不同 → 切块变了，updated
  （原因 chunking_changed），重建分块，旧版本保留为历史
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.sql_models import Document, DocumentVersion


@dataclass(frozen=True)
class VersionDecision:
    """增量判定结果；unchanged 时附带既有版本行供编排层直接引用。"""

    version_status: str  # new / unchanged / updated（导入审计语义，非审核状态）
    chunking_changed: bool  # True = 切块方式变化（正文未变），导入结果 reason 标记用
    unchanged_version: DocumentVersion | None = None  # 仅 unchanged 分支非空


def decide_version_status(
    session: Session,
    document: Document,
    incoming_version: Any,
) -> VersionDecision:
    """按双比较规则判定数据包相对当前文档的增量状态。

    incoming_version 是数据包中的版本载荷（package.data.version），
    需要 content_hash 与 chunk_fingerprint 两个属性。
    """
    # 先按 content_hash 找同正文版本，再比对切块指纹
    same_content_versions = session.scalars(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.content_hash == incoming_version.content_hash,
        )
    ).all()
    if same_content_versions:
        incoming_fingerprint = incoming_version.chunk_fingerprint
        # 库内已有的非空指纹（老版本行该列为 NULL，代表升级前产出，指纹未知）
        known_fingerprints = [
            version.chunk_fingerprint
            for version in same_content_versions
            if version.chunk_fingerprint is not None
        ]
        # 存量兼容规则：只要存在"未知"（NULL 指纹的库内版本，或无指纹的新包），
        # 就按 unchanged 处理——不得因为指纹为空就一律判 updated，
        # 否则升级后第一次重跑会对全部存量文档触发全量重建。
        # 注：若库内同时存在已知指纹且均不匹配，仍判 chunking_changed。
        fingerprint_known_match = incoming_fingerprint in known_fingerprints
        has_unknown = (
            incoming_fingerprint is None
            or len(known_fingerprints) < len(same_content_versions)
        )
        if has_unknown or fingerprint_known_match:
            # 已有版本中优先取指纹精确匹配的；否则取最新一条（存量兼容路径）
            unchanged_version = next(
                (
                    version
                    for version in same_content_versions
                    if version.chunk_fingerprint == incoming_fingerprint
                ),
                same_content_versions[-1],
            )
            return VersionDecision(
                version_status="unchanged",
                chunking_changed=False,
                unchanged_version=unchanged_version,
            )
        # 走到这里：content_hash 相同、双方指纹都已知且不同 → 切块方式变化
        return VersionDecision(version_status="updated", chunking_changed=True)
    # 正文变化（或首次导入）→ 保持现有行为
    version_status = "new" if document.current_version_id is None else "updated"
    return VersionDecision(version_status=version_status, chunking_changed=False)
