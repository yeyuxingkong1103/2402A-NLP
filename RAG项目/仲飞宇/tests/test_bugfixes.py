"""回归测试：覆盖本轮修掉的几个 bug。

每个用例都对应一个真实踩过的坑，注释里记了「怎么坏的」，避免以后被改回去。

按故障分组，组标题写的是「坏掉的形态 → 现在的口径」。各组的完整实测记录（怎么复现、
修前修后各是什么数）在 git log 的「深度查 bug 第 N 批」那几条 commit message 里，
拿组标题搜就能对上——这里的用例是那几轮的固化。
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.core.pipeline import EmptyAnswerError
from app.core.store.milvus_store import quote_literal, role_filter
from app.main import app
from helpers import parse_sse


# ---- 流式与非流式口径必须一致（postprocess 只在流末尾生效）----
def test_stream_done_carries_same_answer_as_chat(client):
    """流式路径此前完全绕过 postprocess，导致同一个问题两个接口给出不同答案。"""
    q = "高血压饮食注意什么"
    stream_resp = client.post("/chat/stream", json={"question": q, "role_id": "psychologist"})
    assert stream_resp.status_code == 200

    events = parse_sse(stream_resp.text)
    kinds = list(events)  # dict 保序，即事件到达顺序
    assert kinds[0] == "sources"
    assert "delta" in kinds
    assert kinds[-1] == "done"

    streamed = "".join(p["delta"] for p in events.get("delta", []))
    done_answer = events["done"][-1]["answer"]
    # done 里带的必须是后处理过的文本，而不是原始增量拼接
    assert done_answer == done_answer.strip()  # postprocess 会 strip
    assert done_answer  # 不能是空的
    assert streamed  # 流式增量本身也照常吐

    # 非流式走的是同一条后处理，两边应当一致
    chat_resp = client.post("/chat", json={"question": q, "role_id": "psychologist"})
    assert chat_resp.json()["answer"] == done_answer


def test_stream_stores_same_answer_as_done_event(client):
    """写进短期记忆的必须是清洗后的文本，否则下一轮模型会读到被截断的自述。"""
    sid = "stream-consistency"
    resp = client.post(
        "/chat/stream",
        json={"question": "高血压饮食注意什么", "role_id": "psychologist", "session_id": sid},
    )
    done_answer = parse_sse(resp.text)["done"][-1]["answer"]
    hist = app.state.pipeline.memory.get_history(
        app.state.pipeline._memory_key(sid, "psychologist")
    )
    assert hist[-1]["role"] == "assistant"
    assert hist[-1]["content"] == done_answer


# ---- 未知角色：404，而不是 500 / 静默半截流 ----
def test_chat_unknown_role_returns_404(client):
    resp = client.post("/chat", json={"question": "你好", "role_id": "no_such_role"})
    assert resp.status_code == 404


def test_chat_stream_unknown_role_returns_404_not_half_stream(client):
    """此前会先发 200 + sources，再在生成器里抛错断流，用户只看到「空回复」。"""
    resp = client.post("/chat/stream", json={"question": "你好", "role_id": "no_such_role"})
    assert resp.status_code == 404


# ---- 同一会话并发不能写脏对话史（深度查 bug 第三批）----
def test_concurrent_turns_same_session_do_not_interleave(make_fake_pipeline):
    """两个并发请求打同一个 session 时，历史必须是成对的「问-答」，不能交错。

    实测过的坏形态：[user Q_A][user Q_B, assistant A_B, assistant A_A] ——
    下一轮模型读到的是错位对话史，Q_A 成了"没有回答的问题"、A_A 挂到 Q_B 名下。
    """
    import threading
    import time

    class _SlowLLM:
        """拖慢生成，把交错窗口放大到必现。"""

        def chat(self, messages):
            time.sleep(0.05)
            return "回答内容" * 20

        def stream(self, messages):
            yield "回答内容" * 20

    p = make_fake_pipeline(_SlowLLM())
    key = p._memory_key("concurrent", "psychologist")

    def ask(q):
        p.answer(q, "psychologist", "concurrent")

    threads = [threading.Thread(target=ask, args=(f"问题{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    roles = [h["role"] for h in p.memory.get_history(key)]
    assert roles == ["user", "assistant", "user", "assistant"], f"历史被写乱了: {roles}"
    # 且每题后面跟的必须是自己的答案（顺序上成对即可，这里再确认内容不改动）
    assert p.memory.get_history(key)[0]["content"].startswith("问题")


def test_concurrent_first_connect_and_role_upsert(dummy_settings):
    """多线程同时首次连接 + 注册同一个角色：都不许抛错。

    这是真实场景：nginx 后面 3 个 worker 启动时都会跑一遍预设角色注册，全新库上
    并发插入同一个 role_id。以前两处都会炸——create_all 撞 `table users already exists`、
    角色插入撞 `UNIQUE constraint failed: roles.id`（ensure_role 是先查再插）。
    """
    import threading

    from app.core.store.sql_store import SQLStore

    sql = SQLStore(dummy_settings)
    errors: list[Exception] = []

    def worker():
        try:
            sql.connect()
            sql.upsert_role("psychologist", "高血压健康助手", "描述", "你是医生。", "🩺")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发首次连接/注册角色不该报错: {errors}"
    assert len(sql.list_roles()) == 1, "同一个 role_id 只该有一条记录"


# ---- 上游空 choices 要走"模型没答"而不是"依赖掉线"（深度查 bug 第三批）----
def test_empty_choices_raises_empty_answer_error():
    """HTTP 200 + `choices: []`（中转/内容过滤常见）不能是 IndexError。

    IndexError 会被接口层归成 503「依赖掉线」+ detail 里一个 IndexError，
    把排查方向带到"服务挂了"；同一响应在流式路径是走 EmptyAnswerError（502）的，
    两条路径口径必须一致。
    """
    import pytest

    from app.core.errors import EmptyAnswerError
    from app.core.llm import LLMClient
    from app.core.config import Settings

    client = LLMClient(Settings(llm_provider="openai_compat", llm_api_key="k"))

    class _Completions:
        def create(self, **_kw):
            return type("R", (), {"choices": []})()

    client._client = type("C", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()

    with pytest.raises(EmptyAnswerError):
        client.chat([{"role": "user", "content": "你好"}])


# ---- 记忆后端的边界语义（深度查 bug 第三批）----
def test_memory_max_turns_zero_means_one_not_unbounded(monkeypatch):
    """`MEMORY_MAX_TURNS=0` 在两个后端必须同义。

    进程内是 deque(maxlen=0)（一条不存），Redis 侧算出 ltrim(0,-1) 即"不裁剪"
    （反而无界增长）——恰在本该最省内存的配置上最容易膨胀。统一夹到至少 1 轮。
    """
    import redis

    from app.core.store.memory import InMemoryMemoryStore, RedisMemoryStore

    s = InMemoryMemoryStore(0)
    assert s.max_turns == 1
    s.add("sess", "user", "问")
    assert len(s.get_history("sess")) == 1

    monkeypatch.setattr(redis.Redis, "from_url", lambda *_a, **_kw: _StubRedis([]))
    assert RedisMemoryStore("redis://x", 0).max_turns == 1


class _StubRedis:
    """只实现记忆后端用到的那几个命令。"""

    def __init__(self, rows: list[str]):
        self.rows = rows
        self.includes: list[str] = []

    def lrange(self, *_a):
        return list(self.rows)

    def delete(self, *_a):
        return 0

    def ping(self):
        return True


def test_redis_history_skips_bad_elements_instead_of_blanking(monkeypatch, caplog):
    """一条坏 JSON 不能让整个会话历史静默消失（以前整个列表推导被一个 except 吞掉，返回 []）。"""
    import json
    import logging

    import redis

    from app.core.store.memory import RedisMemoryStore

    rows = ["{坏掉的 JSON", json.dumps({"role": "user", "content": "你好"})]
    monkeypatch.setattr(redis.Redis, "from_url", lambda *_a, **_kw: _StubRedis(rows))

    store = RedisMemoryStore("redis://x")
    with caplog.at_level(logging.WARNING, logger="memory"):
        hist = store.get_history("sess")

    assert hist == [{"role": "user", "content": "你好"}], "坏元素应被跳过，其余照常返回"
    assert "非法 JSON" in caplog.text, "跳过坏元素要留痕，否则查不出历史为什么少了一段"


def test_in_memory_store_bounds_session_count():
    """进程内实现没有 TTL：会话数必须有上限，否则轮换 session_id 的客户端能让它无限涨。"""
    from app.core.store.memory import InMemoryMemoryStore

    store = InMemoryMemoryStore(10)
    store.MAX_SESSIONS = 3
    for i in range(10):
        store.add(f"s{i}", "user", "问")

    assert len(store._store) == store.MAX_SESSIONS
    assert store.get_history("s9") == [{"role": "user", "content": "问"}], "最近用过的会话要留着"


# ---- 配置读错时要看得见（深度查 bug 第二批）----
def test_int_config_error_names_the_key(monkeypatch):
    """类型写错必须报出键名：以前是裸的 `invalid literal for int()`，还发生在 import 期。"""
    import pytest

    from app.core.config import _int

    monkeypatch.setenv("TOP_K", "abc")
    with pytest.raises(ValueError, match="TOP_K"):
        _int("TOP_K", 5)


def test_bool_config_rejects_typo_loudly(monkeypatch, capsys):
    """`OCR_ENABLED=enabled` 这种写法以前静默取默认值（开关写错＝功能静默关掉）。"""
    from app.core.config import _bool

    monkeypatch.setenv("OCR_ENABLED", "enabled")
    assert _bool("OCR_ENABLED", True) is True  # 取值仍是默认
    assert "OCR_ENABLED" in capsys.readouterr().err  # 但必须喊一声

    monkeypatch.setenv("OCR_ENABLED", "off")
    assert _bool("OCR_ENABLED", True) is False  # 认得的写法照常生效


# ---- 登记失败时不留孤儿向量（深度查 bug 第一批）----
def test_store_chunks_rolls_back_vectors_when_register_fails():
    """写进向量库后、登记关系库失败时必须补偿删除。

    否则留下的是「检索能召回、/knowledge/list 里看不到」的孤儿数据：列表显示没入库，
    再跑一次 ingest 又会重复灌一份（SQL 里没有这条登记，文档级去重拦不住）。
    """
    import pytest

    from app.core.document.ingest import store_chunks

    deleted: list[tuple] = []

    class _M:
        def insert(self, role_id, chunks):
            self.inserted = chunks

        def delete_by_source(self, role_id, source):
            deleted.append((role_id, source))
            return 1

    class _S:
        def register_document(self, *a, **kw):
            raise RuntimeError("MySQL 掉线")

    with pytest.raises(RuntimeError):
        store_chunks(
            [{"text": "t", "title": "标题", "source": "s.md", "chunk_index": 0, "summary": "", "vector": [0.0]}],
            role_id="psychologist",
            source="s.md",
            title="标题",
            milvus=_M(),
            sql=_S(),
        )

    assert deleted == [("psychologist", "s.md")], "登记失败后必须按 (角色, 来源) 回滚向量"


# ---- 重灌不产生重复登记（合并入库管线那轮挖出来的：seed 有自己的清理，CLI 没有）----
def test_reingest_does_not_duplicate_document_rows(dummy_settings):
    """同一份文档重灌两次，关系库里只能有一条登记。

    实测（2026-09-21）：`scripts/ingest.py --re-ingest` 只删了 Milvus 旧向量，
    关系库这边一直在加，重灌两次 /knowledge/list 里就有两条一模一样的记录。
    seed 有自己的清理，CLI 没有——所以把清理挪进了 SQLStore.dedup_document。
    """
    from app.core.store.sql_store import SQLStore

    sql = SQLStore(dummy_settings)
    sql.connect()
    role, src = "psychologist", "/tmp/reingest.md"

    sql.register_document(role, src, "标题", 3)
    sql.register_document(role, src, "标题", 3)  # 没清理的情况
    assert len(sql.list_documents(role)) == 2, "先复现出重复登记"

    assert sql.dedup_document(role, src) == 2
    assert sql.list_documents(role) == []

    sql.register_document(role, src, "标题", 3)
    assert len(sql.list_documents(role)) == 1


# ---- Milvus 过滤表达式注入 ----
# 名字刻意不叫 test_role_filter_rejects_expression_injection：tests/test_milvus_store.py
# 里已有一个同名用例，但两者**参数集不同**（那边管中文/空串/超长，这边管复合注入
# 与反斜杠），不是重复。改名是为了别让人误以为其中一个可以删。
@pytest.mark.parametrize(
    "bad",
    [
        'x" or role_id != "',
        'doc" or true or role_id == "doc',
        "back\\slash",
    ],
)
def test_role_filter_rejects_additional_injection_payloads(bad):
    """role_id 是被拼进 Milvus 过滤表达式字符串的：载荷一旦能闭合引号再接一个 or，条件就
    恒真——多角色共用一张向量表，过滤被打开等于跨角色召回别人的资料。
    """
    with pytest.raises(ValueError):
        role_filter(bad)


def test_role_filter_accepts_normal_role_ids():
    """反向锁住合法集合：加固注入时很容易顺手收严，而内置 id 带下划线、自建角色可能带点与
    横线，拒错的后果是入库/检索直接抛 ValueError。
    """
    assert role_filter("psychologist") == 'role_id == "psychologist"'
    assert role_filter("role.v2-test") == 'role_id == "role.v2-test"'


def test_quote_literal_rejects_quote_and_backslash():
    """引号与反斜杠都能改写表达式里的字符串边界，必须拒；中文路径必须放行——source 来自真实
    文件名，拒错会让整轮 ingest 在第一个文件上中断（入库侧现在统一转正斜杠，正是这条的后续）。
    """
    with pytest.raises(ValueError):
        quote_literal('data/corpus/a" or role_id != "')
    assert quote_literal("data/corpus/劳动法.md") == '"data/corpus/劳动法.md"'


def test_chat_rejects_malicious_role_id_with_422(client):
    """schema 层先挡一道，非法 id 直接 422，不会走到 Milvus。"""
    resp = client.post("/chat", json={"question": "你好", "role_id": 'x" or role_id != "'})
    assert resp.status_code == 422


def test_role_create_rejects_malicious_role_id(client):
    """创建接口是同一个 id 的另一条入口，schema 层要一起挡——只堵 /chat 是堵不住的。"""
    resp = client.post(
        "/role",
        json={"role_id": 'x"', "name": "x", "system_prompt": "x"},
    )
    assert resp.status_code == 422


# ---- BM25 全空语料不再除零 ----
def test_bm25_all_blank_corpus_does_not_crash():
    """语料整段是空白时 token 总数为 0、平均长度也是 0，rank_bm25 内部除零。

    与 test_retriever.py 的「IDF 恰好为 0」不是一回事：那条的输入是「词出现在半数文档上」，
    这条的输入是「语料本身没内容」。
    """
    from app.core.retrieve.hybrid_retriever import BM25Index

    idx = BM25Index([{"id": "1", "text": "   "}, {"id": "2", "text": "\n\n"}])
    assert idx.search("高血压", top_k=5) == []


# ---- BGEReranker 不该原地改写调用方的列表 ----
def test_bge_reranker_does_not_mutate_candidates(monkeypatch):
    """两个断言各锁一件事：排序看的是精排分（融合分只有 0.1 的候选该排到 0.9 的前面），
    以及返回新列表——就地覆盖 score 会让上游拿到两种量纲混着的分数。
    """
    from app.core.config import Settings
    from app.core.reranker import BGEReranker

    candidates = [
        {"id": "1", "text": "低相关", "score": 0.9},
        {"id": "2", "text": "高相关", "score": 0.1},
    ]
    before = [dict(c) for c in candidates]

    r = BGEReranker(Settings(reranker="bge"))

    class FakeResp:
        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"results": [{"index": 1, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.05}]}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @staticmethod
        def post(*a, **kw):
            return FakeResp()

    monkeypatch.setattr(r, "_get_client", lambda: FakeClient())
    out = r.rerank("query", candidates)

    assert [c["id"] for c in out] == ["2", "1"]
    assert candidates == before  # 入参没被就地改掉


# ---- 空答案不能静默通过（模型在思考阶段就把长度用光）----
#
# 实测形态（探针：给 /v1 设小 max_tokens，卡在思考阶段）：
#     finish_reason='length'  content=0 字  reasoning=214 字
# 修复前：/chat 返回 200 + answer:""，/chat/stream 发完 sources 直接 done(answer:"")，
# 没有 error 事件 —— 用户看到一片空白，且空答案被写进短期记忆污染下一轮。


class _EmptyLLM:
    """一个字都不吐的模型（截断在思考阶段）。"""

    def chat(self, messages) -> str:
        return ""

    def stream(self, messages):
        return iter(())


class _BlankOnlyLLM:
    """content 只有空白：postprocess 会 strip 成空，同样属于「没答」。"""

    def chat(self, messages) -> str:
        return "   \n\n \t "

    def stream(self, messages):
        yield "   \n\n \t "


@contextmanager
def _client_with(pipeline):
    """这几条要把「坏模型」装进管线，而 conftest 的 client 夹具只能给默认 dummy LLM，
    没法在夹具层面换掉——所以在这里手工组一次。"""
    with TestClient(app) as c:
        app.state.pipeline = pipeline
        yield c


def test_chat_empty_answer_returns_502_not_200(make_fake_pipeline):
    """空答案必须是明确的错误状态，不能是 200 + 空串。"""
    with _client_with(make_fake_pipeline(_EmptyLLM())) as c:
        resp = c.post("/chat", json={"question": "高血压注意什么？", "role_id": "psychologist"})

    assert resp.status_code == 502
    assert "未产出回答" in resp.json()["detail"]


def test_chat_blank_only_answer_also_returns_502(make_fake_pipeline):
    """只有空白的 content 经 postprocess 后为空，同样要报错。"""
    with _client_with(make_fake_pipeline(_BlankOnlyLLM())) as c:
        resp = c.post("/chat", json={"question": "高血压注意什么？", "role_id": "psychologist"})

    assert resp.status_code == 502


def test_chat_stream_empty_answer_emits_error_not_done(make_fake_pipeline):
    """流式必须发 error 事件，且不能同时发一个空答案的 done。"""
    with _client_with(make_fake_pipeline(_EmptyLLM())) as c:
        resp = c.post(
            "/chat/stream", json={"question": "高血压注意什么？", "role_id": "psychologist"}
        )

    assert resp.status_code == 200  # 响应头在生成器之前就发出去了，改不了
    kinds = list(parse_sse(resp.text))

    assert "error" in kinds, "空答案没有发 error 事件"
    assert "done" not in kinds, "空答案不该再发 done —— 那会让前端把空串当成权威答案"
    msg = parse_sse(resp.text)["error"][-1]["message"]
    assert "未产出回答" in msg
    assert "EmptyAnswerError" not in msg, "异常类名不该给用户看"


def test_empty_answer_not_written_to_memory(make_fake_pipeline):
    """失败的那一轮不能进短期记忆。

    否则历史里会多一条空的 assistant 消息，下一轮模型读到的上下文变成
    「用户问了一句，自己什么都没答」——比空答案本身更伤。
    """
    p = make_fake_pipeline(_EmptyLLM())
    with _client_with(p) as c:
        c.post("/chat/stream", json={"question": "高血压注意什么？", "role_id": "psychologist"})

    hist = p.memory.get_history(p._memory_key("default", "psychologist"))
    assert hist == [], f"失败的一轮被写进了记忆: {hist}"


def test_pipeline_raises_empty_answer_error_directly(make_fake_pipeline):
    """不带异常类名给用户看，但类型必须是 EmptyAnswerError（接口层靠它分流）。"""
    p = make_fake_pipeline(_EmptyLLM())

    with pytest.raises(EmptyAnswerError) as e1:
        p.answer("高血压注意什么？", "psychologist", "z1")

    final: list[str] = []
    with pytest.raises(EmptyAnswerError):
        list(p.answer_stream("高血压注意什么？", "psychologist", "z2", final=final))

    assert "未产出回答" in str(e1.value)
    assert final == [], "报错时不该往 final 里塞东西"


def test_partial_answer_is_still_returned(make_fake_pipeline):
    """只有「完全为空」才报错。被截断但有内容的部分答案要照常返回。

    这是刻意的取舍：截断到一半的答案对用户仍有价值，而空答案没有任何价值。
    """

    class _PartialLLM:
        def chat(self, messages) -> str:
            return "高血压患者每日食盐应控制在 5 克以内，后续内容被截断"

        def stream(self, messages):
            yield from ["高血压患者", "每日食盐应控制在", " 5 克以内，后续内容被截断"]

    p = make_fake_pipeline(_PartialLLM())
    out = p.answer("高血压注意什么？", "psychologist", "z3")

    assert out["answer"].startswith("高血压患者每日食盐应控制在")
    assert len(p.memory.get_history(p._memory_key("z3", "psychologist"))) == 2


def test_chat_dependency_down_returns_503_not_500(make_fake_pipeline):
    """依赖掉线（LLM/SQL/Milvus 等）应返回 503 而非裸 500。

    曾经只有 EmptyAnswerError 被转成 502，其它异常（如 Ollama 运行中掉线的
    APIConnectionError）直接冒到 FastAPI 变成 500 + "Internal Server Error"，
    调用方分不清「模型没答」和「服务挂了」。统一成 503 并带类名。
    """

    class _DownLLM:
        def chat(self, messages):
            raise RuntimeError("模拟 Ollama 掉线")

        def stream(self, messages):
            raise RuntimeError("模拟 Ollama 掉线")
            yield  # pragma: no cover

    with _client_with(make_fake_pipeline(_DownLLM())) as c:
        resp = c.post("/chat", json={"question": "高血压注意什么？", "role_id": "psychologist"})

    assert resp.status_code == 503
    assert "暂时不可用" in resp.json()["detail"]
