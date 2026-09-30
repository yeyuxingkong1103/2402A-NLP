"""测试提示词组装模块。

任务书 4-A 要求：
- 把《回答提示词 v1.0》正式搬进代码
- 含铁律、输出结构、免责声明、防注入条款
- 提示词可单独改而不动流程代码
"""
from app.chat.prompt_builder import build_chat_messages


def test_build_chat_messages_structure():
    """测试消息结构：系统提示词 + 用户消息"""
    messages = build_chat_messages(
        question="公司解除劳动合同需要提前多久通知？",
        context_block="[1] 《劳动合同法》第 40 条\n内容...",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_prompt_contains_rules():
    """测试系统提示词包含铁律"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    # 铁律关键点
    assert "只能使用用户提供的【法源清单】" in system_content or "只能使用" in system_content
    assert "引用编号" in system_content or "[1]" in system_content
    assert "禁止" in system_content or "不得" in system_content
    assert "免责" in system_content or "不能替代律师" in system_content


def test_system_prompt_contains_output_structure():
    """测试系统提示词包含输出结构指引"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    assert "直接回答" in system_content or "依据" in system_content
    assert "引用" in system_content


def test_system_prompt_contains_disclaimer():
    """测试系统提示词包含固定免责声明"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    assert "内容仅供" in system_content or "不能替代律师" in system_content


def test_system_prompt_anti_injection():
    """测试系统提示词包含防注入条款"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    # 防注入条款
    assert "数据" in system_content or "指令" in system_content or "忽略" in system_content


def test_user_message_format():
    """测试用户消息格式：法源清单 + 用户问题"""
    question = "试用期有多长？"
    context = "[1] 《劳动合同法》第 19 条\n试用期不得超过六个月。"

    messages = build_chat_messages(question, context)
    user_content = messages[1]["content"]

    # 法源清单在前
    assert "法源清单" in user_content or "[1]" in user_content
    # 用户问题在后
    assert question in user_content


def test_prompt_is_independent():
    """测试提示词可单独修改：返回纯数据结构，不依赖流程代码"""
    messages = build_chat_messages("问题", "上下文")

    # 验证返回的是标准消息格式
    assert isinstance(messages, list)
    assert all(isinstance(m, dict) for m in messages)
    assert all("role" in m and "content" in m for m in messages)


# ---------- v2.0 亲民化提示词 ----------


def test_v2_prompt_is_friendly_not_robotic():
    """v2.0：要求口语化亲民表达，禁止公文腔五段结构。"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    # 亲民要求：通俗、像人话
    assert "通俗易懂" in system_content or "口语" in system_content
    # 不再强制五段公文结构（v1.0 的「输出结构（按此顺序…」已删除）
    assert "按此顺序" not in system_content
    # 篇幅控制：避免一次输出两千字论文
    assert "简洁" in system_content or "啰嗦" in system_content


def test_v2_prompt_keeps_citation_and_disclaimer_rules():
    """v2.0 亲民化不许牺牲引用铁律、免责与防注入。"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    assert "引用编号" in system_content
    assert "内容仅供" in system_content or "不能替代律师" in system_content
    assert "数据" in system_content or "指令" in system_content or "忽略" in system_content


def test_industrial_prompt_understands_workers_plain_language():
    """工业级提示词必须把普通劳动者的口语当作待理解的问题。"""
    messages = build_chat_messages("我被公司开了怎么办", "测试上下文")
    system_content = messages[0]["content"]

    assert "普通劳动者" in system_content
    assert "口语" in system_content
    assert "不要求用户使用专业法律术语" in system_content or (
        "不是要求用户使用专业法律术语" in system_content
    )
    assert "我被公司开了怎么办" in system_content


def test_industrial_prompt_handles_missing_facts_without_forced_refusal():
    """工业级提示词必须在事实不足时澄清关键信息而非机械拒答。"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    assert "事实" in system_content
    assert "最关键" in system_content
    assert "1～3" in system_content or "1-3" in system_content
    assert "没有检索结果" in system_content
    assert "不能编造" in system_content or "不得编造" in system_content


def test_industrial_prompt_preserves_multiturn_context_rules():
    """工业级提示词必须维护多轮事实并避免重复追问。"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    assert "多轮对话" in system_content
    assert "不重复询问" in system_content or "不要重复询问" in system_content
    assert "前后事实矛盾" in system_content


def test_industrial_prompt_covers_safe_next_steps_and_high_risk_cases():
    """工业级提示词必须覆盖证据保存、高风险和不当行为边界。"""
    messages = build_chat_messages("测试问题", "测试上下文")
    system_content = messages[0]["content"]

    assert "保存证据" in system_content or "保存劳动合同" in system_content
    assert "高风险" in system_content
    assert "伪造证据" in system_content or (
        "伪造、篡改、倒签" in system_content
    )
    assert "不保证胜诉" in system_content
