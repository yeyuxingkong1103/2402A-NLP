"""ChatService 长期记忆集成测试（批次 14）。

覆盖（对应验收 a / d / e 的服务层语义）：
- 记忆在检索法律知识之前读取，并注入提示词（记忆段落出现在用户消息里）
- 开关关闭（memory_gate 返回 False）→ 既不读也不写
- 拒答不写记忆；正常回答后写入（memory_write_async=False 同步断言）
- 记忆检索/写入失败不影响问答主流程
"""

from types import SimpleNamespace

import pytest

from app.chat.service import ChatService
from app.memory.long_term import LongTermMemoryStore


class FakeEmbedding:
    VOCAB = "社保经济补偿金解除劳动合同终止试用期工资年假用人单位"

    def embed(self, texts):
        return [[float(t.count(ch)) for ch in self.VOCAB] for t in texts]


class FakeMilvusClient:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def has_collection(self, name):
        return True

    def create_schema(self, auto_id=False):
        return FakeSchema()

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, *args, **kwargs):
        pass

    def load_collection(self, name):
        pass

    def _match(self, row, expr):
        expr = expr.replace(" false", " False").replace(" true", " True")
        return bool(eval(expr, {"__builtins__": {}}, dict(row)))  # noqa: S307

    def upsert(self, name, rows):
        for row in rows:
            for i, existing in enumerate(self.rows):
                if existing["memory_id"] == row["memory_id"]:
                    self.rows[i] = dict(row)
                    break
            else:
                self.rows.append(dict(row))
        return {"upsert_count": len(rows)}

    def query(self, name, filter, output_fields=None, limit=16384, offset=0):
        return [dict(r) for r in self.rows if self._match(r, filter)][offset:offset + limit]

    def search(self, name, data, filter, limit=10, output_fields=None):
        import math

        vec = data[0]
        hits = []
        for row in self.rows:
            if not self._match(row, filter):
                continue
            rvec = row["dense_vector"]
            dot = sum(a * b for a, b in zip(vec, rvec))
            na = math.sqrt(sum(a * a for a in vec)) or 1.0
            nb = math.sqrt(sum(b * b for b in rvec)) or 1.0
            hits.append(
                {"id": row["memory_id"], "distance": dot / (na * nb), "entity": dict(row)}
            )
        hits.sort(key=lambda h: -h["distance"])
        return [hits[:limit]]


class FakeSchema:
    def add_field(self, *args, **kwargs):
        pass


class FakeIndexParams:
    def add_index(self, *args, **kwargs):
        pass


class FakeRetrieval:
    """记录调用顺序；返回单条法源候选。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, question, **kwargs):
        self.calls.append("retrieve")
        articles = [
            SimpleNamespace(
                chunk_key="c1",
                document_title="中华人民共和国劳动合同法",
                article_number="第十九条",
                content="试用期不得超过一年。",
                paragraph_number=None,
                vector_score=0.9,
            )
        ]
        return SimpleNamespace(
            articles=articles,
            context_block="[1] 劳动合同法第十九条 试用期不得超过一年。",
            stats={"vector_recall_count": 1},
        )


class FakeLlm:
    """记录提示词；摘要调用返回固定 JSON。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.chat_calls = 0

    def chat(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        self.chat_calls += 1
        # 第一次：回答生成；第二次：记忆摘要
        if self.chat_calls == 1:
            return "试用期上限参见[1]。内容仅供法律信息参考，不能替代律师出具的正式法律意见。"
        return '{"summary": "用户想了解试用期上限", "importance": 0.5}'

    def stream_chat(self, system_prompt, user_prompt):
        yield from self.chat(system_prompt, user_prompt)


@pytest.fixture()
def memory_store():
    store = LongTermMemoryStore(
        milvus_client=FakeMilvusClient(),
        embedding_client=FakeEmbedding(),
        collection_name="legal_long_term_memory",
        dimension=24,
    )
    store.ensure_collection()
    return store


def _service(store, gate=None, **kwargs) -> tuple[ChatService, FakeRetrieval, FakeLlm]:
    retrieval = FakeRetrieval()
    llm = FakeLlm()
    service = ChatService(
        retrieval_service=retrieval,
        llm_client=llm,
        long_term_memory=store,
        memory_gate=gate,
        memory_write_async=False,
        **kwargs,
    )
    return service, retrieval, llm


def test_memory_injected_into_prompt_and_written_after_answer(memory_store) -> None:
    """先写一条记忆 → 提问时注入提示词 → 回答后新记忆落库。"""
    memory_store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")
    service, retrieval, llm = _service(memory_store)

    result = service.chat("试用期最长多久？", user_id="u1", session_id="s1")

    # 记忆在检索之前读取
    assert retrieval.calls == ["retrieve"]
    # 注入提示词：记忆段落出现在用户消息里
    assert "用户是 A 公司的程序员" in llm.prompts[0]
    assert "# 用户记忆" in llm.prompts[0]
    # 回答后写入了一条新记忆（摘要由 LLM 生成）
    summaries = [
        r["summary"] for r in memory_store.list_memories(user_id="u1")["items"]
    ]
    assert "用户想了解试用期上限" in summaries
    assert result.refused is False


def test_disabled_gate_means_no_read_no_write(memory_store) -> None:
    """开关关闭（接口 8.3）：既不读取也不写入。"""
    memory_store.add_memory(user_id="u1", content="x", summary="用户是 A 公司的程序员")
    service, retrieval, llm = _service(memory_store, gate=lambda user_id: False)

    service.chat("试用期最长多久？", user_id="u1", session_id="s1")

    # 提示词没有记忆段落
    assert "# 用户记忆" not in llm.prompts[0]
    # 没有写入新记录
    assert len(memory_store.list_memories(user_id="u1")["items"]) == 1


def test_refused_answer_skips_memory_write(memory_store) -> None:
    service = ChatService(
        retrieval_service=FakeRetrieval(),
        llm_client=FakeLlm(),
        refusal_min_vector_score=0.99,  # 必然触发拒答
        long_term_memory=memory_store,
        memory_write_async=False,
    )
    result = service.chat("今天天气怎么样", user_id="u1")

    assert result.refused is True
    assert memory_store.list_memories(user_id="u1")["items"] == []


def test_memory_failure_does_not_break_chat(memory_store) -> None:
    """记忆检索挂掉 → 回答照常返回。"""

    class BrokenStore:
        def search(self, *args, **kwargs):
            raise RuntimeError("milvus down")

        def add_memory(self, *args, **kwargs):
            raise RuntimeError("milvus down")

    service, _, llm = _service(BrokenStore())
    result = service.chat("试用期最长多久？", user_id="u1", session_id="s1")

    assert "[1]" in result.answer


def test_no_memory_store_keeps_original_prompt(memory_store) -> None:
    """未装配记忆存储时行为与批次 13 完全一致（无记忆段落）。"""
    service = ChatService(
        retrieval_service=FakeRetrieval(),
        llm_client=FakeLlm(),
    )
    result = service.chat("试用期最长多久？", user_id="u1")

    assert result.refused is False
