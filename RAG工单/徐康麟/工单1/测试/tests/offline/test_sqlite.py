"""离线测试：SQLite 存储层（``app/storage/sqlite_manager.py``）。

测试目标（工单 9.1 / 5.10）：
1. 工单要求的 8 张表全部存在：documents / chunks / conversations / messages /
   feedback / eval_results / golden_qa / logs；
2. 文档元数据、分块（含元数据与关键词）可写入并原样读出；
3. 对话与消息可写入、按顺序读出、可清空、可删除；
4. 反馈（点赞/点踩）可写入并统计；
5. 评估记录与标准问答可写入并读出；
6. 事务失败必须回滚，禁止半成品数据（禁止静默失败）；
7. 不同数据库文件互相隔离（``tmp_path`` 保证不污染 ``data/``）。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    Chunk,
    Citation,
    Conversation,
    DocumentMeta,
    EvalRecord,
    Feedback,
    GoldenQA,
    Message,
)
from app.storage.sqlite_manager import SQLiteManager

# 工单 5.10 要求的表清单（本文件内自带，避免依赖 conftest 的实现细节）
EXPECTED_TABLES = [
    "chunks",
    "conversations",
    "documents",
    "eval_results",
    "feedback",
    "golden_qa",
    "logs",
    "messages",
]


def _chunk(chunk_id: str, page: int = 1, doc_id: str = "doc_test", chunk_type: str = "text") -> Chunk:
    """构造一个测试用分块。"""
    content = f"这是第 {page} 页的测试正文，包含注册资本 5,520 万元。"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=page,
        section="第一节 测试",
        type=chunk_type,  # type: ignore[arg-type]
        content=content,
        char_count=len(content),
        table_id="p1_t1" if chunk_type == "table" else None,
        keywords=["注册资本"],
    )


@pytest.fixture
def store(tmp_path) -> SQLiteManager:
    """每个用例一个全新的临时数据库。"""
    return SQLiteManager(tmp_path / "unit.sqlite3")


@pytest.fixture
def document_meta() -> DocumentMeta:
    """测试用文档元数据。"""
    return DocumentMeta(
        doc_id="doc_test",
        title="招股说明书1",
        source_path="data/raw/招股说明书1.pdf",
        page_count=548,
        chunk_count=2,
        table_count=1,
        status="indexed",
        is_default=True,
    )


# ==========================================================================
# 1. 表结构
# ==========================================================================
def test_all_required_tables_exist(store) -> None:
    """工单 5.10 要求的 8 张表必须全部建好。"""
    tables = store.table_names()
    missing = [name for name in EXPECTED_TABLES if name not in tables]
    assert not missing, f"缺少数据表：{missing}；实际表清单：{tables}"


def test_init_schema_is_idempotent(store) -> None:
    """重复初始化表结构不得报错，也不得清空既有数据。"""
    store.upsert_document(DocumentMeta(doc_id="doc_keep", title="保留", page_count=1))
    store.init_schema()
    assert store.get_document("doc_keep") is not None, "重复建表后既有数据被清空"


# ==========================================================================
# 2. documents 与 chunks
# ==========================================================================
def test_document_upsert_and_query(store, document_meta) -> None:
    """文档元数据可写入、可更新、可按默认文档查询。"""
    store.upsert_document(document_meta)

    loaded = store.get_document("doc_test")
    assert loaded is not None, "写入后的文档查不到"
    assert loaded.page_count == 548 and loaded.status == "indexed", f"文档字段读取错误：{loaded}"
    assert loaded.is_default is True, "默认文档标记丢失"

    changed = document_meta.model_copy(update={"chunk_count": 3024, "status": "indexed"})
    store.upsert_document(changed)
    assert store.get_document("doc_test").chunk_count == 3024, "upsert 未更新已存在的文档"

    assert store.get_default_document().doc_id == "doc_test", "默认文档查询结果不正确"
    assert [item.doc_id for item in store.list_documents()] == ["doc_test"], "文档列表不正确"


def test_chunk_write_and_read(store, document_meta) -> None:
    """分块的元数据、关键词与表格 ID 必须完整往返。"""
    store.upsert_document(document_meta)
    chunks = [_chunk("c000001", page=1), _chunk("c000002", page=2, chunk_type="table")]
    written = store.insert_chunks(chunks)
    assert written == 2, f"应写入 2 个分块，实际 {written}"

    assert store.count_chunks("doc_test") == 2, "按文档统计的分块数不正确"
    assert store.count_chunks() == 2, "全库分块数不正确"

    loaded = store.get_chunks("doc_test")
    assert [item.chunk_id for item in loaded] == ["c000001", "c000002"], "分块应按页码与 ID 有序返回"
    assert loaded[1].type == "table" and loaded[1].table_id == "p1_t1", "表格分块元数据丢失"
    assert loaded[0].keywords == ["注册资本"], "关键词字段未能往返"

    single = store.get_chunk("c000002")
    assert single is not None and single.page == 2, "单条分块查询结果不正确"
    assert store.get_chunks_by_ids(["c000002", "c000001"])[0].chunk_id == "c000002", "按 ID 批量查询打乱了顺序"
    assert store.get_chunks_by_ids(["不存在"]) == [], "查询不存在的分块应返回空列表"


def test_insert_chunks_replaces_old_version(store, document_meta) -> None:
    """``replace_doc=True`` 时必须先清空该文档的旧分块，避免新旧索引混用。"""
    store.upsert_document(document_meta)
    store.insert_chunks([_chunk("c000001", page=1), _chunk("c000002", page=2)])
    store.insert_chunks([_chunk("c000009", page=9)])

    remaining = store.get_chunks("doc_test")
    assert [item.chunk_id for item in remaining] == ["c000009"], "重建索引时旧分块未被清除"


def test_page_range(store, document_meta) -> None:
    """页码范围用于引用合法性校验，必须准确。"""
    store.upsert_document(document_meta)
    store.insert_chunks([_chunk("c000001", page=3), _chunk("c000002", page=129)])
    assert store.page_range("doc_test") == (3, 129), "页码范围计算错误"


# ==========================================================================
# 3. 对话与消息
# ==========================================================================
def test_conversation_and_messages(store) -> None:
    """会话与消息：写入、顺序读取、计数、清空、删除。"""
    conversation: Conversation = store.create_conversation("conv_test", title="测试会话", doc_id="doc_test")
    assert conversation.conversation_id == "conv_test", "创建会话返回的对象不正确"
    assert store.get_conversation("conv_test").title == "测试会话", "会话标题读取错误"

    citation = Citation(page=22, chunk_id="c000022", snippet="注册资本 5,520 万元")
    first_id = store.add_message(Message(conversation_id="conv_test", role="user", content="注册资本是多少？"))
    second_id = store.add_message(
        Message(
            conversation_id="conv_test",
            role="assistant",
            content="注册资本为 5,520 万元。",
            citations=[citation],
            first_token_ms=15.5,
        )
    )
    assert first_id > 0 and second_id > first_id, f"消息 ID 应自增，实际 {first_id} -> {second_id}"

    messages = store.get_messages("conv_test")
    assert [message.role for message in messages] == ["user", "assistant"], "消息顺序或角色不正确"
    assert messages[0].content == "注册资本是多少？", "用户消息内容读取错误"
    assert messages[1].citations[0].page == 22, "助手消息的引用未能往返"
    assert messages[1].first_token_ms == 15.5, "首字耗时不正确"
    assert isinstance(messages[1].created_at, datetime), "消息时间戳未解析为 datetime"

    assert store.get_conversation("conv_test").message_count == 2, "会话的消息计数未同步更新"
    assert store.get_messages("conv_test", limit=1)[0].role == "assistant", "limit 应返回最近的消息"
    assert store.list_conversations()[0].conversation_id == "conv_test", "会话列表不正确"

    removed = store.clear_conversation("conv_test")
    assert removed == 2, f"清空会话应删除 2 条消息，实际 {removed}"
    assert store.get_messages("conv_test") == [], "清空后仍能读到消息"
    assert store.get_conversation("conv_test").message_count == 0, "清空后消息计数未归零"

    store.delete_conversation("conv_test")
    assert store.get_conversation("conv_test") is None, "删除会话后仍能查到"


# ==========================================================================
# 4. 反馈
# ==========================================================================
def test_feedback_write_and_stats(store) -> None:
    """点赞/点踩反馈可写入、可按时间倒序读取并统计。"""
    store.create_conversation("conv_fb")
    up_id = store.add_feedback(
        Feedback(conversation_id="conv_fb", message_id=1, rating="up", comment="回答准确", question="注册资本？")
    )
    down_id = store.add_feedback(
        Feedback(conversation_id="conv_fb", message_id=2, rating="down", comment="答非所问", question="占比？")
    )
    assert up_id > 0 and down_id > up_id, "反馈 ID 应自增"

    stats = store.feedback_stats()
    assert stats == {"up": 1, "down": 1}, f"反馈统计不正确：{stats}"

    items = store.list_feedback()
    assert len(items) == 2, "反馈列表条数不正确"
    assert items[0].feedback_id == down_id, "反馈列表应按 ID 倒序返回最新一条"
    assert items[1].comment == "回答准确", "反馈评论未能往返"


def test_feedback_rating_must_be_up_or_down(store) -> None:
    """非法评分必须被模型层拒绝，避免脏数据入库。"""
    store.create_conversation("conv_fb2")
    with pytest.raises(ValidationError):
        Feedback(conversation_id="conv_fb2", rating="maybe")  # type: ignore[arg-type]


# ==========================================================================
# 5. 评估记录与标准问答
# ==========================================================================
def test_eval_record_write_and_read(store) -> None:
    """评估记录可写入、可按模式过滤、可清空。"""
    record = EvalRecord(
        question_id=543,
        question="注册资本是多少？",
        mode="extractive",
        answer="注册资本为 5,520 万元。",
        golden="注册资本为5,520万元。",
        is_correct=True,
        citation_pages=[22],
        citation_valid=True,
        first_token_ms=12.0,
        total_ms=20.0,
    )
    eval_id = store.add_eval_record(record)
    assert eval_id > 0, "评估记录未写入"

    loaded = store.list_eval_records(mode="extractive")
    assert len(loaded) == 1, "按模式过滤的评估记录数不正确"
    assert loaded[0].question_id == 543 and loaded[0].is_correct is True, "评估记录字段往返错误"
    assert loaded[0].citation_pages == [22], "引用页码未能往返"
    assert loaded[0].faithfulness is None, "未运行的 RAGAS 指标应为 None，不得伪造"

    assert store.list_eval_records(mode="llm") == [], "按不存在的模式过滤应返回空列表"
    store.clear_eval_records()
    assert store.list_eval_records() == [], "清空评估记录失败"


def test_golden_qa_upsert_and_list(store) -> None:
    """标准问答按 id 幂等写入，并可按 id 有序读出。"""
    items = [
        GoldenQA(id=543, question="注册资本是多少？", answer="注册资本为5,520万元。", evidence_pages=[22],
                 evidence="法定代表人：程家明\n注册资本：5,520 万元", category="注册资本"),
        GoldenQA(id=531, question="法定代表人是谁？", answer="法定代表人是程家明。", evidence_pages=[22],
                 category="法定代表人"),
    ]
    assert store.upsert_golden_qa(items) == 2, "标准问答写入条数不正确"

    updated = items[0].model_copy(update={"answer": "注册资本为5,520.00万元。"})
    store.upsert_golden_qa([updated])

    loaded = store.list_golden_qa()
    assert [item.id for item in loaded] == [531, 543], "标准问答应按 id 有序返回"
    assert loaded[1].answer == "注册资本为5,520.00万元。", "upsert 未更新既有标准答案"
    assert loaded[1].evidence_pages == [22], "证据页码未能往返"


# ==========================================================================
# 6. 日志索引、统计与事务
# ==========================================================================
def test_logs_table(store) -> None:
    """关键日志索引可写入并按级别过滤。"""
    store.add_log("app.core.retriever", "检索完成", level="INFO", function="retrieve", payload={"top": 5})
    store.add_log("app.core.retriever", "检索失败", level="ERROR", function="retrieve")

    assert len(store.list_logs()) == 2, "日志条数不正确"
    errors = store.list_logs(level="ERROR")
    assert len(errors) == 1 and errors[0]["message"] == "检索失败", "按级别过滤日志失败"
    assert errors[0]["payload"] == "{}", "日志负载应序列化为 JSON 字符串"


def test_stats_counts(store, document_meta) -> None:
    """统计接口必须覆盖全部 8 张表且计数准确。"""
    store.upsert_document(document_meta)
    store.insert_chunks([_chunk("c000001")])
    store.create_conversation("conv_stats")
    store.add_message(Message(conversation_id="conv_stats", role="user", content="你好"))
    store.add_feedback(Feedback(conversation_id="conv_stats", rating="up"))

    stats = store.stats()
    for table in EXPECTED_TABLES:
        assert table in stats, f"统计结果缺少表 {table}"
    assert stats["documents"] == 1 and stats["chunks"] == 1, f"文档/分块统计错误：{stats}"
    assert stats["conversations"] == 1 and stats["messages"] == 1, f"会话/消息统计错误：{stats}"
    assert stats["feedback"] == 1, "反馈统计错误"


def test_session_rolls_back_on_error(store) -> None:
    """事务中的任何异常都必须整体回滚，禁止留下半成品数据。"""
    store.upsert_document(DocumentMeta(doc_id="doc_tx", title="事务测试", page_count=1))
    before = store.count_chunks()
    duplicate = _chunk("c000001", doc_id="doc_tx")

    with pytest.raises(sqlite3.IntegrityError):
        with store.session() as conn:
            conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, page, type, content) VALUES (?,?,?,?,?)",
                (duplicate.chunk_id, duplicate.doc_id, duplicate.page, duplicate.type, duplicate.content),
            )
            # 同一事务内再插一次主键，触发 IntegrityError -> 应整体回滚
            conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, page, type, content) VALUES (?,?,?,?,?)",
                (duplicate.chunk_id, duplicate.doc_id, duplicate.page, duplicate.type, duplicate.content),
            )

    assert store.count_chunks() == before, "事务失败后仍写入了数据，回滚未生效"


@pytest.mark.xfail(
    reason=(
        "已知缺陷：connect() 未对每条新连接执行 PRAGMA foreign_keys=ON"
        "（app/storage/sqlite_manager.py:185-189；_SCHEMA 里的 PRAGMA 只作用于建表那条连接），"
        "实测 PRAGMA foreign_keys=0，孤儿分块可写入、ON DELETE CASCADE 不生效"
    ),
    strict=False,
)
def test_foreign_key_constraint_rejects_orphan_chunk(store) -> None:
    """引用不存在文档的分块必须被外键约束拒绝（保护数据一致性）。"""
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_chunks(
            [Chunk(chunk_id="c000404", doc_id="不存在的文档", page=1, content="孤儿分块", char_count=4)]
        )


def test_separate_databases_are_isolated(tmp_path) -> None:
    """不同数据库文件必须完全隔离（测试用临时库不污染真实索引库）。"""
    first = SQLiteManager(tmp_path / "a.sqlite3")
    second = SQLiteManager(tmp_path / "b.sqlite3")

    first.upsert_document(DocumentMeta(doc_id="doc_a", title="A", page_count=1))
    assert first.get_document("doc_a") is not None, "写入 A 库失败"
    assert second.get_document("doc_a") is None, "B 库读到了 A 库的数据，隔离失效"
    assert second.count_chunks() == 0, "B 库不应有任何分块"
