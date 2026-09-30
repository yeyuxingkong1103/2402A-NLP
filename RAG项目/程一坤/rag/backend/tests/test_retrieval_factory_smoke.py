"""真实检索服务装配冒烟测试。

批次 16-B 任务 0：本用例依赖真实外部服务（SiliconFlow Embedding API、Milvus、
MySQL、Redis），外部 API 抖动会使其变红，故打 @pytest.mark.real_api 标记：
- 默认 `pytest tests -q` 仍会运行本用例（全量口径不变）；
- 需要稳定快速回归时用 `pytest tests -q -m "not real_api"` 跳过本用例；
- 上线前或改动检索链路后，必须跑全量（含 real_api）。

容错约定：外部 API 瞬断允许"失败即重试一次"，但重试后仍失败必须报错，
且每次失败都打印重试证据（第几次、错误类型），不掩盖真实失败。
"""

import pytest

from app.memory.short_term import ShortTermMemoryStore
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.assembly import build_default_retrieval_service


def _run_smoke() -> None:
    """单次冒烟：真实工厂构造 + 短期记忆读写删 + 真实检索断言。"""
    service = build_default_retrieval_service()

    assert service.keyword_searcher is not None
    assert service.keyword_searcher.indexed_chunk_count > 0
    assert isinstance(service.short_term_memory, ShortTermMemoryStore)
    assert isinstance(service.query_rewriter, QueryRewriter)
    assert service.reranker is not None

    user_id = "retrieval-factory-smoke-user"
    session_id = "retrieval-factory-smoke-session"
    service.short_term_memory.write_summary(user_id, session_id, "经济补偿咨询")
    assert service.short_term_memory.read_summary(user_id, session_id) == "经济补偿咨询"
    assert isinstance(service.short_term_memory.read_summary(user_id, session_id), str)
    service.short_term_memory.delete_session_memory(user_id, session_id)

    result = service.retrieve("经济补偿怎么算", rerank_top_n=8)

    assert len(result.articles) >= 5
    assert result.stats["keyword_recall_count"] > 0
    assert len({article.chunk_key for article in result.articles}) == len(result.articles)
    parent_articles = {
        (article.document_title, article.article_number)
        for article in result.articles
    }
    assert len(parent_articles) == len(result.articles)
    assert all(
        article.rerank_score is not None and 0 <= article.rerank_score <= 1
        for article in result.articles
    )
    assert all(
        article.fusion_score is not None
        and article.rerank_score != article.fusion_score
        for article in result.articles
    )


@pytest.mark.real_api
def test_default_retrieval_service_builds_real_dependencies_and_retrieves(
    capsys,
) -> None:
    """真实 MySQL、Redis、Milvus 配置下，默认工厂应能构造并返回检索结果。

    失败即重试一次；重试后仍失败则抛出最后一次异常（真实失败不掩盖），
    并通过打印留下重试证据。
    """
    last_error: BaseException | None = None
    for attempt in (1, 2):
        try:
            _run_smoke()
            if attempt == 2:
                print(
                    "[real_api] 第 1 次冒烟失败后重试成功（第 1 次失败证据见上一行），"
                    "最终结果有效"
                )
            return
        except Exception as exc:  # noqa: BLE001 - 重试证据需要打印真实错误
            last_error = exc
            suffix = "，重试一次" if attempt == 1 else "，已重试一次仍失败（真实失败）"
            print(
                f"[real_api] 冒烟第 {attempt} 次失败："
                f"{type(exc).__name__}: {exc}{suffix}"
            )
    assert last_error is not None  # 防御：循环必然写入
    raise last_error
