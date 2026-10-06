import logging

from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.schemas.retrieval import Citation

logger = logging.getLogger(__name__)

_LEGAL_MATERIAL_TYPES = {"law", "administrative_regulation", "department_rule"}
_CASE_MATERIAL_TYPES = {"guiding_case", "typical_case"}


def _citation_kind(material_type: str) -> str:
    # 法律、行政法规和部门规章统一作为法律规定引用。
    if material_type in _LEGAL_MATERIAL_TYPES:
        return "法律规定"
    # 司法解释必须独立标注，不能混同为普通法律条文。
    if material_type == "judicial_interpretation":
        return "司法解释"
    # 指导案例和典型案例只能作为案例参考，不能替代法律条文。
    if material_type in _CASE_MATERIAL_TYPES:
        return "案例参考"
    raise ValueError("unsupported material_type for citation")


def _material_title(material: KnowledgeMaterial) -> str:
    # Task 5 材料模型未固定 title 字段，先兼容动态属性。
    return str(getattr(material, "title", material.publisher))


def _material_version(material: KnowledgeMaterial) -> str:
    # 优先使用明确版本描述，否则使用生效日期生成可读版本。
    version = getattr(material, "version", None)
    if version:
        return str(version)
    if material.effective_from:
        return f"{material.effective_from}起施行"
    return "版本未知"


def build_citation(material: KnowledgeMaterial, article: str | None, paragraph: str | None, excerpt: str) -> Citation:
    # 构建引用前先判定类别，确保案例不会被当作法律规定。
    kind = _citation_kind(material.material_type)
    logger.info(
        "构建法律知识引用",
        extra={"material_id": material.id, "material_type": material.material_type, "citation_kind": kind, "has_article": article is not None},
    )
    return Citation(
        kind=kind,
        title=_material_title(material),
        article=article,
        paragraph=paragraph,
        excerpt=excerpt,
        version=_material_version(material),
        official_url=material.source_url,
        material_id=material.id,
    )
