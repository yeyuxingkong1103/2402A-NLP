from backend.app.services.knowledge_base_service import create_crawl_snapshot


def create_crawl_snapshot_task(
    source_url: str,
    raw_html: str,
    attachments: list[dict],
    failure_reason: str | None = None,
    backup_source_urls: list[str] | None = None,
):
    # Worker 仅创建抓取快照，不执行审核、发布或检索索引写入。
    return create_crawl_snapshot(source_url, raw_html, attachments, failure_reason, backup_source_urls)
