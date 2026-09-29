"""集成测试（integration）：真实 Milvus 插入/检索/删除 + 长期记忆 Collection。

使用专属 doc_id / user_id（9999xx）写入 persona_knowledge 与 user_long_term_memory，
用例结束统一清理，不污染真实数据。
"""
import time

import pytest

pytestmark = pytest.mark.integration

TEST_DOC_ID = 999901
TEST_USER_ID = 999901
VECTOR = [0.05 * ((i % 20) + 1) for i in range(1024)]  # 固定 1024 维假向量


@pytest.fixture
def knowledge_records(require_milvus, require_mysql):
    from src.core.config import settings
    from src.db import milvus as milvus_db

    records = [
        {
            "persona_id": 1, "doc_id": TEST_DOC_ID, "chunk_id": i,
            "text": text, "summary": "测试摘要", "source": "pytest",
            "created_at": int(time.time()), "updated_at": int(time.time()),
            "vector": VECTOR,
        }
        for i, text in enumerate([
            "认知重构是识别自动负性思维并用合理认知替代的技术。",
            "腹式呼吸可以在焦虑发作时快速平复自主神经。",
            "每周三次三十分钟的有氧运动能显著改善轻度抑郁情绪。",
        ])
    ]
    ids = milvus_db.insert_chunks(records)
    assert len(ids) == 3
    yield records
    milvus_db.delete_knowledge_by_doc(TEST_DOC_ID, 1)  # 清理


def test_insert_and_dense_search(require_milvus, knowledge_records):
    from src.db import milvus as milvus_db

    hits = milvus_db.dense_search_knowledge(1, VECTOR, top_k=5)
    assert hits, "稠密检索应至少命中刚插入的向量"
    texts = {h["text"] for h in hits}
    assert "认知重构是识别自动负性思维并用合理认知替代的技术。" in texts
    top = max(hits, key=lambda h: h["score"])
    assert top["score"] > 0.99          # 同向量余弦相似度 ≈ 1


def test_hybrid_search(require_milvus, knowledge_records):
    from src.db import milvus as milvus_db

    hits = milvus_db.hybrid_search_knowledge(1, VECTOR, "腹式呼吸 缓解焦虑", top_k=5)
    assert isinstance(hits, list)       # BM25 函数 + RRFRanker 正常执行
    assert all(h.get("persona_id") == 1 for h in hits)


def test_count_and_delete_by_doc(require_milvus, knowledge_records):
    from src.db import milvus as milvus_db

    assert milvus_db.count_knowledge(1) >= 3
    deleted = milvus_db.delete_knowledge_by_doc(TEST_DOC_ID, 1)
    assert deleted >= 3
    remaining = [h for h in milvus_db.dense_search_knowledge(1, VECTOR, top_k=10)
                 if h.get("doc_id") == TEST_DOC_ID]
    assert remaining == []              # 删除后不再命中


def test_long_term_memory_insert_and_search(require_milvus):
    from src.db import milvus as milvus_db

    memory_id = milvus_db.insert_memory(
        persona_id=1, user_id=TEST_USER_ID, conversation_id=999901,
        summary="用户长期存在考试焦虑，倾向灾难化解读成绩。", vector=VECTOR,
    )
    assert memory_id
    try:
        hits = milvus_db.search_memory(1, TEST_USER_ID, VECTOR, top_k=3)
        assert hits and "考试焦虑" in hits[0]["summary"]
    finally:
        milvus_db.get_client().delete(
            settings_milvus_memory_collection(),
            filter=f"user_id == {TEST_USER_ID}",
        )


def settings_milvus_memory_collection() -> str:
    from src.core.config import settings
    return settings.milvus_memory_collection
