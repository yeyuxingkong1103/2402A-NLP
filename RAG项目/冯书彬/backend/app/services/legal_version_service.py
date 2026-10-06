from dataclasses import dataclass
from datetime import date, datetime
import logging
from typing import Any

from backend.app.models.knowledge_base import KnowledgeMaterial

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegalFactTimeline:
    # 事实时间线仅表达案件事实发生日和关系类型，不保存当事人敏感信息。
    event_date: date | None
    relationship_type: str


@dataclass(frozen=True)
class ApplicableMaterial:
    # 命中的适用材料只暴露材料和版本标识，原因用于审计解释。
    material_id: str
    version_id: str
    reason: str


def _as_date(value: Any) -> date | None:
    # None 表示缺少事实日期，由调用方按现行版本处理。
    if value is None:
        return None
    # date 类型可直接用于闭区间比较。
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    # datetime 去掉时间部分，避免时区差异影响法律生效日判断。
    if isinstance(value, datetime):
        return value.date()
    # ISO 字符串兼容 Task 5 既有 effective_from 字段定义。
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("effective date must be date, datetime, ISO string, or None")


def _material_version_id(material: KnowledgeMaterial) -> str:
    # 版本 ID 优先使用 Task 5 后续动态字段，缺省回退材料 ID并记录脱敏告警。
    version_id = getattr(material, "version_id", None)
    if version_id:
        return str(version_id)
    logger.warning("材料缺少版本标识，使用材料 ID 兜底", extra={"material_id": material.id})
    return str(material.id)


def _matches_relationship(material: KnowledgeMaterial, relationship_type: str) -> bool:
    # 未标注关系类型的正式材料默认可参与匹配，兼容已有数据。
    relationship_types = getattr(material, "relationship_types", None)
    if not relationship_types:
        return True
    return relationship_type in relationship_types


def _is_published_material(material: KnowledgeMaterial) -> bool:
    # 只允许已发布且可检索的材料进入适用版本候选。
    return material.status == "published" and material.searchable is True


def _is_effective_on(material: KnowledgeMaterial, query_date: date) -> bool:
    # 生效开始日必须存在且不晚于事实日期。
    effective_from = _as_date(material.effective_from)
    if effective_from is None or effective_from > query_date:
        return False
    # 失效日为空表示持续有效；否则事实日期必须不晚于失效日。
    effective_to = _as_date(getattr(material, "effective_to", None))
    return effective_to is None or query_date <= effective_to


def _is_current_version(material: KnowledgeMaterial, today: date) -> bool:
    # 缺少事实日期时选择当前有效版本，必须已生效且没有失效日期。
    effective_from = _as_date(material.effective_from)
    return effective_from is not None and effective_from <= today and _as_date(getattr(material, "effective_to", None)) is None


def _to_applicable(material: KnowledgeMaterial, reason: str) -> ApplicableMaterial:
    # 构造输出结构，不携带正文，避免日志和上游响应泄露全文。
    return ApplicableMaterial(material_id=material.id, version_id=_material_version_id(material), reason=reason)


def find_applicable_materials(query_date: date | datetime | str | None, relationship_type: str, materials: list[KnowledgeMaterial]) -> list[ApplicableMaterial]:
    normalized_query_date = _as_date(query_date)
    today = date.today()
    # 记录脱敏摘要日志，只包含数量、关系类型和日期。
    logger.info(
        "开始匹配法律适用版本",
        extra={"query_date": str(normalized_query_date) if normalized_query_date else None, "relationship_type": relationship_type, "candidate_count": len(materials)},
    )
    # 先过滤发布状态和关系类型，避免草稿、案例草稿或无关材料被引用。
    candidates = [material for material in materials if _is_published_material(material) and _matches_relationship(material, relationship_type)]
    if normalized_query_date is None:
        matched = [_to_applicable(material, "current_version_due_to_missing_fact_date") for material in candidates if _is_current_version(material, today)]
    else:
        matched = [_to_applicable(material, "effective_on_fact_date") for material in candidates if _is_effective_on(material, normalized_query_date)]
    logger.info(
        "完成法律适用版本匹配",
        extra={"query_date": str(normalized_query_date) if normalized_query_date else None, "relationship_type": relationship_type, "matched_count": len(matched)},
    )
    return matched
