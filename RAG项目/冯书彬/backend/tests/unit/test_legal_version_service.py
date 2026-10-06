from datetime import date, datetime, timezone

import pytest

from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.services.legal_version_service import find_applicable_materials


@pytest.fixture
def material_version_factory():
    # 构造已发布材料版本，不依赖真实仓储，避免测试污染。
    def _factory(
        version_id="current",
        material_id="law-1",
        material_type="law",
        effective_from=date(2021, 1, 1),
        effective_to=None,
        relationship_types=None,
        status="published",
        searchable=True,
    ):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        material = KnowledgeMaterial(
            id=material_id,
            snapshot_id=f"snapshot-{version_id}",
            source_url="https://example.gov.cn/law",
            publisher="全国人大常委会",
            material_type=material_type,
            raw_text="脱敏法律材料正文",
            attachments=[],
            status=status,
            searchable=searchable,
            created_at=now,
            updated_at=now,
            effective_from=effective_from,
        )
        material.version_id = version_id
        material.effective_to = effective_to
        material.relationship_types = relationship_types or ["property"]
        return material

    return _factory


def test_selects_version_effective_on_fact_date(material_version_factory):
    old = material_version_factory(version_id="old", effective_from=date(2010, 1, 1), effective_to=date(2020, 12, 31))
    current = material_version_factory(version_id="current", effective_from=date(2021, 1, 1), effective_to=None)

    result = find_applicable_materials(date(2019, 6, 1), "property", [old, current])

    assert [item.version_id for item in result] == ["old"]
    assert result[0].reason == "effective_on_fact_date"


def test_missing_fact_date_selects_current_version(material_version_factory):
    old = material_version_factory(version_id="old", effective_from=date(2010, 1, 1), effective_to=date(2020, 12, 31))
    current = material_version_factory(version_id="current", effective_from=date(2021, 1, 1), effective_to=None)

    result = find_applicable_materials(None, "property", [current, old])

    assert [item.version_id for item in result] == ["current"]
    assert result[0].reason == "current_version_due_to_missing_fact_date"


def test_missing_fact_date_excludes_future_open_ended_version(material_version_factory):
    future = material_version_factory(version_id="future", effective_from=date(2099, 1, 1), effective_to=None)

    result = find_applicable_materials(None, "property", [future])

    assert result == []


def test_normalizes_query_date_values(material_version_factory):
    current = material_version_factory(version_id="current", effective_from=date(2021, 1, 1), effective_to=None)

    from_datetime = find_applicable_materials(datetime(2022, 1, 1, 9, 30, tzinfo=timezone.utc), "property", [current])
    from_iso_string = find_applicable_materials("2022-01-01", "property", [current])

    assert [item.version_id for item in from_datetime] == ["current"]
    assert [item.version_id for item in from_iso_string] == ["current"]


def test_missing_version_id_falls_back_to_material_id_with_warning(material_version_factory, caplog):
    material = material_version_factory(material_id="law-fallback", version_id="")

    result = find_applicable_materials(date(2022, 1, 1), "property", [material])

    assert [item.version_id for item in result] == ["law-fallback"]
    assert "材料缺少版本标识" in caplog.text


def test_filters_unpublished_and_unrelated_materials(material_version_factory):
    published = material_version_factory(version_id="published", relationship_types=["property"])
    draft = material_version_factory(version_id="draft", status="reviewed")
    unrelated = material_version_factory(version_id="unrelated", relationship_types=["labor"])

    result = find_applicable_materials(date(2022, 1, 1), "property", [draft, unrelated, published])

    assert [item.version_id for item in result] == ["published"]
