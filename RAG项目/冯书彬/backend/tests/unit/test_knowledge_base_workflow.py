import pytest

from backend.app.services.knowledge_base_service import (
    KnowledgeBaseWorkflowError,
    create_crawl_snapshot,
    create_source_whitelist_entry,
    deprecate_material,
    disable_source_whitelist_entry,
    publish_material,
    review_material,
    submit_for_review,
    update_source_whitelist_entry,
)


@pytest.fixture(autouse=True)
def reset_knowledge_store():
    # 每个用例清空内存仓储，避免状态互相污染。
    from backend.app.repositories.knowledge_base_repository import reset_store

    reset_store()


@pytest.fixture
def whitelist_entry():
    # 有效白名单是创建抓取快照的前置条件。
    return create_source_whitelist_entry(
        url="https://example.gov.cn/laws",
        publisher="最高人民法院",
        material_type="judicial_interpretation",
        actor_id="admin-1",
    )


@pytest.fixture
def pending_material(whitelist_entry):
    snapshot = create_crawl_snapshot(
        source_url=whitelist_entry.url,
        raw_html="<html><body>司法解释正文</body></html>",
        attachments=[{"name": "notice.pdf", "url": "https://example.gov.cn/laws/notice.pdf"}],
    )
    return submit_for_review(snapshot.id)


def test_content_reviewer_review_does_not_publish(pending_material):
    reviewed = review_material(
        pending_material.id,
        reviewer_id="reviewer-1",
        decision="approved",
        reason="source verified",
    )

    assert reviewed.status == "reviewed"
    assert reviewed.reviewed_by == "reviewer-1"
    assert reviewed.searchable is False


def test_super_admin_publish_makes_material_searchable(pending_material):
    reviewed = review_material(
        pending_material.id,
        reviewer_id="reviewer-1",
        decision="approved",
        reason="source verified",
    )

    published = publish_material(reviewed.id, publisher_id="admin-1")

    assert published.status == "published"
    assert published.published_by == "admin-1"
    assert published.searchable is True


def test_rejected_material_cannot_be_published(pending_material):
    rejected = review_material(
        pending_material.id,
        reviewer_id="reviewer-1",
        decision="rejected",
        reason="source mismatch",
    )

    with pytest.raises(ValueError, match="reviewed"):
        publish_material(rejected.id, publisher_id="admin-1")


def test_published_material_can_only_be_deprecated(pending_material):
    reviewed = review_material(pending_material.id, "reviewer-1", "approved", "source verified")
    published = publish_material(reviewed.id, "admin-1")

    deprecated = deprecate_material(published.id, actor_id="admin-1", reason="new version released")

    assert deprecated.status == "deprecated"
    assert deprecated.searchable is False


def test_crawl_snapshot_is_not_searchable(whitelist_entry):
    snapshot = create_crawl_snapshot(
        source_url=whitelist_entry.url,
        raw_html="<html><body>未审核正文</body></html>",
        attachments=[],
    )

    assert snapshot.searchable is False
    assert snapshot.status == "crawled"


def test_snapshot_requires_active_whitelist(whitelist_entry):
    whitelist_entry.active = False

    with pytest.raises(ValueError, match="active whitelist"):
        create_crawl_snapshot(
            source_url=whitelist_entry.url,
            raw_html="<html><body>正文</body></html>",
            attachments=[],
        )


def test_super_admin_can_update_and_disable_whitelist(whitelist_entry):
    updated = update_source_whitelist_entry(
        whitelist_entry.id,
        actor_id="admin-1",
        publisher="最高人民检察院",
        material_type="typical_case",
    )

    assert updated.publisher == "最高人民检察院"
    assert updated.material_type == "typical_case"

    disabled = disable_source_whitelist_entry(updated.id, actor_id="admin-1", reason="source retired")

    assert disabled.active is False


def test_backup_source_must_have_same_publisher(whitelist_entry):
    create_source_whitelist_entry(
        url="https://backup.gov.cn/laws",
        publisher="国务院",
        material_type="judicial_interpretation",
        actor_id="admin-1",
    )

    with pytest.raises(ValueError, match="publisher"):
        create_crawl_snapshot(
            source_url=whitelist_entry.url,
            raw_html="<html><body>正文</body></html>",
            attachments=[],
            backup_source_urls=["https://backup.gov.cn/laws"],
        )


def test_backup_source_allows_same_publisher(whitelist_entry):
    backup = create_source_whitelist_entry(
        url="https://backup.gov.cn/laws",
        publisher="最高人民法院",
        material_type="judicial_interpretation",
        actor_id="admin-1",
    )

    snapshot = create_crawl_snapshot(
        source_url=whitelist_entry.url,
        raw_html="<html><body>正文</body></html>",
        attachments=[],
        backup_source_urls=[backup.url],
    )

    assert snapshot.backup_source_urls == [backup.url]


def test_published_and_rejected_materials_forbid_invalid_transitions(pending_material):
    rejected = review_material(pending_material.id, "reviewer-1", "rejected", "source mismatch")

    with pytest.raises(KnowledgeBaseWorkflowError, match="pending_review"):
        review_material(rejected.id, "reviewer-1", "approved", "retry")
    with pytest.raises(KnowledgeBaseWorkflowError, match="reviewed"):
        publish_material(rejected.id, "admin-1")

    snapshot = create_crawl_snapshot(
        source_url="https://example.gov.cn/laws",
        raw_html="<html><body>另一正文</body></html>",
        attachments=[],
    )
    material = submit_for_review(snapshot.id)
    reviewed = review_material(material.id, "reviewer-1", "approved", "source verified")
    published = publish_material(reviewed.id, "admin-1")

    with pytest.raises(KnowledgeBaseWorkflowError, match="pending_review"):
        review_material(published.id, "reviewer-1", "rejected", "late reject")
    with pytest.raises(KnowledgeBaseWorkflowError, match="reviewed"):
        publish_material(published.id, "admin-1")
