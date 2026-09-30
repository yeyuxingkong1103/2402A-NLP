"""长期记忆存储层测试（批次 14，需求 3.6 / 架构文档 9.3）。

约束（用户裁决）：
- Milvus 独立集合 legal_long_term_memory，严禁与 legal_documents 混用
- 写入：先摘要、与该用户近期记忆去重（高于阈值更新原记录不新增）、importance 0~1
- 读取：必须带 user_id 过滤（缺失直接报错，不允许查全量）、按相关度取前 5、
  排除已软删与已过期
- 删除：软删除；非本人返回"不存在"语义（False → API 层 404）

替身说明：
- FakeMilvusClient：内存实现 search/query/upsert/delete，用可控表达式求值
  （只评估本存储层生成的受控表达式）
- FakeEmbedding：字符词袋向量，相同文本相似度 1.0，不同事实相似度低
"""

import time

import pytest

from app.memory.long_term import (
    LongTermMemoryStore,
)
from app.memory.memory_summarize import summarize_memory_fact


class FakeEmbedding:
    """字符词袋向量：相同文本 → 相同向量（相似度 1.0）。"""

    VOCAB = "社保经济补偿金解除劳动合同终止试用期工资年假用人单位"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vectors.append([float(text.count(ch)) for ch in self.VOCAB])
        return vectors


class FakeMilvusClient:
    """内存 Milvus 替身：只实现本存储层用到的接口与受控表达式。"""

    def __init__(self) -> None:
        self.collections: dict[str, list[dict]] = {}
        self._loaded: set[str] = set()

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_schema(self, auto_id: bool = False):
        return FakeSchema()

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, name, schema=None, index_params=None) -> None:
        self.collections[name] = []

    def load_collection(self, name) -> None:
        self._loaded.add(name)

    def _match(self, row: dict, expr: str) -> bool:
        expr = expr.replace(" false", " False").replace(" true", " True")
        try:
            return bool(eval(expr, {"__builtins__": {}}, dict(row)))  # noqa: S307
        except Exception:  # noqa: BLE001
            return False

    def upsert(self, name, rows):
        collection = self.collections.setdefault(name, [])
        count = 0
        for row in rows:
            for i, existing in enumerate(collection):
                if existing["memory_id"] == row["memory_id"]:
                    collection[i] = dict(row)
                    break
            else:
                collection.append(dict(row))
            count += 1
        return {"upsert_count": count}

    def query(self, name, filter: str, output_fields=None, limit=16384, offset=0):
        collection = self.collections.get(name, [])
        matched = [dict(row) for row in collection if self._match(row, filter)]
        return matched[offset : offset + limit]

    def search(self, name, data, filter: str, limit=10, output_fields=None):
        import math

        collection = self.collections.get(name, [])
        vec = data[0]
        hits = []
        for row in collection:
            if not self._match(row, filter):
                continue
            rvec = row["dense_vector"]
            dot = sum(a * b for a, b in zip(vec, rvec))
            norm_a = math.sqrt(sum(a * a for a in vec)) or 1.0
            norm_b = math.sqrt(sum(b * b for b in rvec)) or 1.0
            hits.append(
                {"id": row["memory_id"], "distance": dot / (norm_a * norm_b), "entity": dict(row)}
            )
        hits.sort(key=lambda h: -h["distance"])
        return [hits[:limit]]


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[str] = []

    def add_field(self, name, *args, **kwargs) -> None:
        self.fields.append(name)


class FakeIndexParams:
    def add_index(self, *args, **kwargs) -> None:
        pass


@pytest.fixture()
def store():
    client = FakeMilvusClient()
    store = LongTermMemoryStore(
        milvus_client=client,
        embedding_client=FakeEmbedding(),
        collection_name="legal_long_term_memory",
        dimension=24,
    )
    store.ensure_collection()
    return store


# --------------------------------------------------------------------------
# 集合隔离（docs 9.3 / 需求 3.6：严禁与知识库集合混用）
# --------------------------------------------------------------------------


def test_uses_dedicated_collection_not_knowledge_base() -> None:
    client = FakeMilvusClient()
    store = LongTermMemoryStore(
        milvus_client=client,
        embedding_client=FakeEmbedding(),
        collection_name="legal_long_term_memory",
        dimension=24,
    )
    store.ensure_collection()

    assert "legal_long_term_memory" in client.collections
    assert "legal_documents" not in client.collections
    # 字段按 docs 9.3 全量落位
    fields = client.collections and store.schema_fields
    for required in (
        "memory_id",
        "user_id",
        "character_id",
        "content",
        "summary",
        "importance",
        "created_at",
        "updated_at",
        "expire_at",
        "source_session_id",
        "deleted",
        "dense_vector",
    ):
        assert required in fields, f"缺字段 {required}"


# --------------------------------------------------------------------------
# 写入与去重
# --------------------------------------------------------------------------


def test_add_memory_inserts_new_record(store: LongTermMemoryStore) -> None:
    memory_id, updated = store.add_memory(
        user_id="u1",
        content="用户在 A 公司工作，担任程序员。",
        summary="用户是 A 公司的程序员",
        character_id="character_001",
        importance=0.6,
        source_session_id="s1",
    )

    assert not updated
    records = store.list_memories(user_id="u1")["items"]
    assert len(records) == 1
    assert records[0]["memory_id"] == memory_id
    assert records[0]["importance"] == 0.6
    assert records[0]["deleted"] is False


def test_duplicate_fact_updates_original_record(store: LongTermMemoryStore) -> None:
    """同一事实问两遍只留一条：相似度高于阈值 → 更新原记录而非新增。"""
    first_id, _ = store.add_memory(
        user_id="u1",
        content="用户在 A 公司工作，担任程序员。",
        summary="用户是 A 公司的程序员",
    )
    _, updated = store.add_memory(
        user_id="u1",
        content="用户在 A 公司当程序员，已经三年。",
        summary="用户是 A 公司的程序员",
    )

    assert updated
    records = store.list_memories(user_id="u1")["items"]
    assert len(records) == 1  # 没有新增
    assert records[0]["memory_id"] == first_id  # 还是原来那条
    assert "三年" in records[0]["content"]  # 内容已更新


def test_different_facts_do_not_merge(store: LongTermMemoryStore) -> None:
    store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")
    store.add_memory(user_id="u1", content="y", summary="用户关注工伤保险待遇标准")

    assert len(store.list_memories(user_id="u1")["items"]) == 2


def test_dedup_is_scoped_to_same_user(store: LongTermMemoryStore) -> None:
    """用户 B 写入与用户 A 相同的事实，不得合并进 A 的记录。"""
    store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")
    _, updated_b = store.add_memory(user_id="u2", content="y", summary="用户是 A 公司的程序员")

    assert not updated_b
    assert len(store.list_memories(user_id="u1")["items"]) == 1
    assert len(store.list_memories(user_id="u2")["items"]) == 1


# --------------------------------------------------------------------------
# 读取：user_id 强制过滤 + 排除软删/过期
# --------------------------------------------------------------------------


def test_search_without_user_id_raises(store: LongTermMemoryStore) -> None:
    """缺少 user_id 过滤直接报错，不允许"查全量"。"""
    with pytest.raises(ValueError):
        store.search("", "社保")


def test_search_returns_top_k_by_relevance(store: LongTermMemoryStore) -> None:
    for i in range(7):
        store.add_memory(user_id="u1", content=f"fact {i}", summary=f"用户关注社保话题{i}")
    hits = store.search("u1", "社保", top_k=5)

    assert len(hits) <= 5


def test_search_excludes_soft_deleted(store: LongTermMemoryStore) -> None:
    memory_id, _ = store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")
    store.soft_delete(user_id="u1", memory_id=memory_id)

    assert store.search("u1", "程序员") == []
    assert store.list_memories(user_id="u1")["items"] == []


def test_search_excludes_expired(store: LongTermMemoryStore) -> None:
    store.add_memory(
        user_id="u1",
        content="x",
        summary="用户是 A 公司的程序员",
        expire_at=int(time.time()) - 10,  # 已过期
    )

    assert store.search("u1", "程序员") == []


def test_search_filters_by_user_strictly(store: LongTermMemoryStore) -> None:
    """跨用户隔离：A 的记忆在 B 的检索里查不到。"""
    store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")
    store.add_memory(user_id="u2", content="y", summary="用户在 B 公司做财务")

    hits_a = store.search("u1", "程序员")
    hits_b = store.search("u2", "程序员")

    assert len(hits_a) == 1
    assert hits_a[0].user_id == "u1"
    assert all(r.user_id == "u2" for r in hits_b)


# --------------------------------------------------------------------------
# 软删除：所有权校验
# --------------------------------------------------------------------------


def test_soft_delete_by_non_owner_returns_false(store: LongTermMemoryStore) -> None:
    memory_id, _ = store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")

    assert store.soft_delete(user_id="u2", memory_id=memory_id) is False  # 非本人
    # 原记录不受影响
    assert len(store.list_memories(user_id="u1")["items"]) == 1


def test_soft_delete_missing_memory_returns_false(store: LongTermMemoryStore) -> None:
    assert store.soft_delete(user_id="u1", memory_id="no-such-id") is False


def test_soft_delete_by_owner_succeeds(store: LongTermMemoryStore) -> None:
    memory_id, _ = store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")

    assert store.soft_delete(user_id="u1", memory_id=memory_id) is True
    assert store.list_memories(user_id="u1", include_deleted=True)[
        "items"
    ][0]["deleted"] is True


# --------------------------------------------------------------------------
# LLM 摘要
# --------------------------------------------------------------------------


class FakeLlmJson:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def test_summarize_memory_fact_parses_llm_json() -> None:
    llm = FakeLlmJson('{"summary": "用户是 A 公司的程序员", "importance": 0.7}')

    result = summarize_memory_fact(llm, "我在 A 公司上班，做程序员", "好的，了解了你的情况。")

    assert result is not None
    assert result["summary"] == "用户是 A 公司的程序员"
    assert result["importance"] == 0.7
    # 问题与回答都进了提示词
    assert "程序员" in llm.calls[0][1]


def test_summarize_memory_fact_survives_llm_failure() -> None:
    """LLM 挂了必须返回 None（写入侧跳过），不能让记忆写入拖垮回答。"""

    class Broken:
        def chat(self, *args, **kwargs):
            raise RuntimeError("llm down")

    assert summarize_memory_fact(Broken(), "q", "a") is None
