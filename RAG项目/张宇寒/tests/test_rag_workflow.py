"""问答主流程的小范围回归测试，不调用外部模型或数据库。"""

from types import SimpleNamespace

import pytest

from backend.app.memory.api import MemoryOrchestrator
from backend.app.memory.resolver import CoreferenceResolver
from backend.app.models.prompt import build_answer_messages
from backend.app.rag import api as rag_api
from backend.app.rag.answer import AnswerGenerator
from backend.app.rag.understand import QueryUnderstanding
from backend.app.rag.workflow import RagWorkflow
from backend.app.storage.mysql import ConversationHistoryStore, LongTermMemoryStore
from backend.app.workspace.service import WorkspaceService


def test_uploaded_material_supplements_the_case_instead_of_replacing_the_question():
    """防止普通案情咨询因为存在附件而被擅自改写成文件摘要任务。"""
    messages = build_answer_messages(
        "朋友借款到期不还，我应该怎么办？",
        "[1] 【用户上传材料】借条.pdf\n借款金额三万元。",
        has_private_material=True,
        use_compact=True,
    )
    system_prompt = messages[0]["content"]
    assert "用户当前陈述是本轮要解决的问题，上传材料只用于补充事实和证据线索" in system_prompt
    assert "不得擅自改成文件摘要或独立文档阅读任务" in system_prompt


def test_understanding_internal_type_error_is_not_retried():
    calls = []

    class Understanding:
        def understand(self, question, memory_context):
            calls.append(question)
            raise TypeError("问题理解函数内部错误")

    with pytest.raises(TypeError, match="问题理解函数内部错误"):
        RagWorkflow._understand_query(Understanding(), "追问", None)
    assert calls == ["追问"]


def test_generation_internal_type_error_is_not_retried():
    calls = []

    class Generator:
        def generate(self, question, evidence, history, thinking_enabled=None):
            calls.append(question)
            raise TypeError("生成函数内部错误")

    with pytest.raises(TypeError, match="生成函数内部错误"):
        RagWorkflow._generate_answer(Generator(), "问题", [], [], False)
    assert calls == ["问题"]


def test_streaming_internal_type_error_does_not_restart_answer():
    calls = []

    class Generator:
        def stream_answer_text(self, question, evidence, history, thinking_enabled=None):
            calls.append(question)
            yield "已输出的一段"
            raise TypeError("流式生成函数内部错误")

    response = RagWorkflow._stream_answer_text(Generator(), "问题", [], [], False)
    assert next(response) == "已输出的一段"
    with pytest.raises(TypeError, match="流式生成函数内部错误"):
        next(response)
    assert calls == ["问题"]


def test_case_analysis_reuses_answer_without_another_model_call():
    """主回答已生成后，补充展示信息不能再请求一次大模型。"""

    class Model:
        def __init__(self):
            self.calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            return "不应生成这段内容"

    model = Model()
    evidence = [{"title": "民法典相关条文", "source_type": "public", "content": "依法履行合同义务。"}]

    analysis = AnswerGenerator(model).build_case_analysis(
        "对方不履行合同怎么办？",
        evidence,
        "可以先固定合同和催告记录，再依法主张责任。",
    )

    assert model.calls == 0
    assert len(analysis) >= 3


def test_word_yinggai_is_not_mistaken_for_a_reference():
    result = CoreferenceResolver().resolve(
        "我应该怎么办？",
        {"current_topic": "上一次咨询的借款纠纷"},
        [],
    )

    assert result["resolved_query"] == "我应该怎么办？"
    assert result["resolved_references"] == {}


def test_query_understanding_uses_generic_resolver_result_without_inventing_labor_facts():
    resolved_memory = SimpleNamespace(
        resolved_query="关于房屋租赁合同，这个能解除吗？",
        resolved_references={"target": "房屋租赁合同"},
        legacy_case_memory={},
    )
    assert QueryUnderstanding.resolve_reference("这个能解除吗？", resolved_memory) == {"target": "房屋租赁合同"}
    assert QueryUnderstanding.rewrite_query("这个能解除吗？", resolved_memory) == "关于房屋租赁合同，这个能解除吗？"

    labor_memory = SimpleNamespace(
        legacy_case_memory={
            "case_type": "劳动争议",
            "parties": {"employee": "张某", "employer": "某公司"},
            "facts": {"salary": 8000},
        }
    )
    assert QueryUnderstanding.rewrite_query("赔多少？", labor_memory) == "赔多少？"


def test_conversation_history_delete_removes_messages_and_session_derivatives():
    class Database:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            self.calls.append((" ".join(sql.split()), params))

    database = Database()
    ConversationHistoryStore(database).delete_session("user-1", "session-1")

    tables = [sql.split("DELETE FROM ", 1)[1].split(" ", 1)[0] for sql, _params in database.calls]
    assert tables == ["conversation_messages", "conversation_summaries", "case_memories"]
    assert all(params == ("user-1", "session-1") for _sql, params in database.calls)


def test_memory_session_delete_clears_short_history_and_session_long_memory():
    deleted = []
    memory = object.__new__(MemoryOrchestrator)
    memory.short_memory = SimpleNamespace(delete=lambda user, session: deleted.append(("short", user, session)))
    memory.history = SimpleNamespace(delete_session=lambda user, session: deleted.append(("history", user, session)))
    memory.long_memory_store = SimpleNamespace(delete_session=lambda user, session: deleted.append(("long", user, session)))
    memory._long_memory_cache = {"enable_long_memory:user-1": True}

    memory.delete_session("user-1", "session-1")

    assert deleted == [
        ("short", "user-1", "session-1"),
        ("history", "user-1", "session-1"),
        ("long", "user-1", "session-1"),
    ]


def test_long_memory_delete_removes_only_rows_derived_from_the_session():
    class Database:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            self.calls.append((" ".join(sql.split()), params))

    database = Database()
    LongTermMemoryStore(database).delete_session("user-1", "session-1")

    assert len(database.calls) == 2
    assert "DELETE c FROM memory_conflicts" in database.calls[0][0]
    assert "DELETE FROM long_term_memory" in database.calls[1][0]
    assert all(params[-1] == "session-1" for _sql, params in database.calls)


def test_delete_session_endpoint_clears_memory_and_workspace(monkeypatch):
    calls = []
    memory = SimpleNamespace(delete_session=lambda user, session: calls.append(("memory", user, session)))
    workspace = SimpleNamespace(
        delete_session_files=lambda user, session: calls.append(("workspace", user["user_id"], session))
        or {"file_count": 0, "warnings": []}
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services={"memory": memory, "workspace": workspace})))
    monkeypatch.setattr(rag_api, "current_user", lambda _request: {"user_id": "user-1"})

    result = rag_api.delete_session("session-1", request)

    assert calls == [("memory", "user-1", "session-1"), ("workspace", "user-1", "session-1")]
    assert result["success"] is True
    assert "retained" not in result["data"]


def test_workspace_record_delete_removes_private_vectors():
    calls = []
    workspace = object.__new__(WorkspaceService)
    workspace.local = SimpleNamespace(delete=lambda path: calls.append(("file", path)))
    workspace.vectors = SimpleNamespace(delete=lambda user, document: calls.append(("vector", user, document)))
    workspace.files = SimpleNamespace(delete=lambda document: calls.append(("database", document)))

    warnings = workspace.delete_record(
        {"user_id": "user-1"},
        {"document_id": "doc-1", "storage_path": "safe/path.pdf"},
    )

    assert warnings == []
    assert calls == [
        ("file", "safe/path.pdf"),
        ("vector", "user-1", "doc-1"),
        ("database", "doc-1"),
    ]
