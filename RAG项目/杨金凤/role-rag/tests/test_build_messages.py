"""build_messages() 单元测试：有/无 history、system 内容。"""
import rag

CONTEXTS = [
    {"content": "血压分级标准", "page": 3, "similarity": 0.9},
    {"content": "用药注意事项", "page": 7, "similarity": 0.8},
]
QUERY = "血压多少算高"


def test_build_messages_no_history():
    """无 history 时返回 [system, user] 两条。"""
    messages = rag.build_messages(QUERY, CONTEXTS)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": QUERY}


def test_build_messages_with_history():
    """有 history 时返回 [system, ...history, user]。"""
    history = [
        {"role": "user", "content": "之前的问题"},
        {"role": "assistant", "content": "之前的回答"},
    ]
    messages = rag.build_messages(QUERY, CONTEXTS, history)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": QUERY}


def test_build_messages_none_history_same_as_empty():
    """history=None 与空列表结果等价。"""
    assert rag.build_messages(QUERY, CONTEXTS, None) == rag.build_messages(
        QUERY, CONTEXTS, []
    )


def test_build_messages_system_has_persona_and_references():
    """system 内容包含 DOCTOR_PERSONA 与【参考资料】。"""
    system = rag.build_messages(QUERY, CONTEXTS)[0]["content"]
    assert rag.DOCTOR_PERSONA in system
    assert "【参考资料】" in system


def test_build_messages_context_format():
    """参考资料按 [资料i｜第page页] 从 1 编号，多条用空行分隔。"""
    system = rag.build_messages(QUERY, CONTEXTS)[0]["content"]
    assert "[资料1｜第3页]" in system
    assert "[资料2｜第7页]" in system
    assert "血压分级标准" in system
    assert "用药注意事项" in system
