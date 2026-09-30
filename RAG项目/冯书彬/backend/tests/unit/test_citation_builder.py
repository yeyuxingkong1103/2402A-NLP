from datetime import date, datetime, timezone

import pytest

from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.rag.citation_builder import build_citation


@pytest.fixture
def material_factory():
    # 引用构造只需要材料元数据，不读取正文全文。
    def _factory(
        material_id="material-1",
        material_type="law",
        title="中华人民共和国民法典",
        source_url="https://example.gov.cn/law",
        version="2021-01-01现行有效",
    ):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        material = KnowledgeMaterial(
            id=material_id,
            snapshot_id="snapshot-1",
            source_url=source_url,
            publisher="全国人大常委会",
            material_type=material_type,
            raw_text="脱敏材料正文",
            attachments=[],
            status="published",
            searchable=True,
            created_at=now,
            updated_at=now,
            effective_from=date(2021, 1, 1),
        )
        material.title = title
        material.version = version
        material.version_id = "version-1"
        return material

    return _factory


def test_case_citation_is_marked_as_case_reference(material_factory):
    material = material_factory(material_type="typical_case", title="某离婚财产典型案例")

    citation = build_citation(material, article=None, paragraph="裁判规则", excerpt="案例不能替代法律条文")

    assert citation.kind == "案例参考"
    assert citation.title == "某离婚财产典型案例"
    assert citation.article is None
    assert citation.paragraph == "裁判规则"


def test_law_citation_keeps_article_and_official_url(material_factory):
    material = material_factory(material_id="law-1", material_type="law")

    citation = build_citation(material, article="第一千零六十二条", paragraph="第一款", excerpt="夫妻共同财产范围")

    assert citation.kind == "法律规定"
    assert citation.material_id == "law-1"
    assert citation.article == "第一千零六十二条"
    assert citation.official_url == "https://example.gov.cn/law"


def test_judicial_interpretation_citation_has_separate_kind(material_factory):
    material = material_factory(material_type="judicial_interpretation", title="民法典婚姻家庭编解释")

    citation = build_citation(material, article="第二条", paragraph=None, excerpt="司法解释内容")

    assert citation.kind == "司法解释"
    assert citation.title == "民法典婚姻家庭编解释"
