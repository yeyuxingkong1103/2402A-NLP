"""src/online/prompt_builder.py —— 五层提示词拼装。

在链路中的位置：
    src/online/chain.py → 【本文件】 → src/online/llm.py

这是"五层 Prompt"这个说法的实现处。把提示词按信息性质分成五层，
比把一堆文本随便拼接更强的地方在于：**模型能分清每段内容的地位**。

    L0 安全合规层   —— 来自 prompts/L0_safety.txt，全局底线，优先级最高
    L1 角色人设层   —— 来自 prompts/L1_role.txt，模板化，用角色卡字段填充
    L2 长期记忆层   —— 该用户在该角色下的历史相关发言
    L3 短期记忆层   —— 当前会话最近几轮对话
    L4 检索上下文层 —— 本轮检索到的知识库片段（唯一的事实依据来源）
    L5 当前输入层   —— 用户本轮问题（放在 user 消息里，与上面五层分开）

L0~L4 全部进 system 消息、L5 单独进 user 消息，这个划分是有意的：
    把"系统设定"和"用户输入"放在不同角色通道里，
    是让模型守住设定的最有效手段 —— 用户在消息里试图改写指令时，
    system 通道的内容不会被 user 通道的内容覆盖。

为什么配置文件缺失时要内置一份兜底文本（`or "..."`）：
    prompts/ 目录被误删或部署时漏拷时，服务不应该直接崩 ——
    有一份等效的内置文案，功能就还能正常工作（只是不方便改文案而已）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"  # 上溯两级到项目根，再进 prompts/


def _read(name: str) -> str:
    """读取提示词模板文件。

    参数：
        name: 文件名（如 L0_safety.txt）
    返回：
        文件内容；文件不存在时返回空串（由调用方决定用什么兜底）。

    返回空串而不是抛异常：
        提示词模板是"可替换的配置"，不是程序的必要组成。
        缺了它应该降级，不该让整个聊天功能不可用。
    """
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_prompt(role: dict[str, Any], long_memory: list[dict], short_memory: list[dict], contexts: list[dict], user_message: str) -> list[dict[str, str]]:
    """拼装五层提示词。

    参数：
        role: 角色卡（含 name 以及 L1 模板需要的 identity/tone/taboo_topics/disclaimer 等字段）
        long_memory: 长期记忆条目
        short_memory: 最近几轮对话
        contexts: 本轮检索到的知识片段
        user_message: 用户本轮问题
    返回：
        [{"role": "system", ...}, {"role": "user", ...}]

    每一层都给了中文占位文案（如"（暂无长期记忆）""（无检索上下文）"）：
        这比留空更有用 —— 留空时模型不知道自己"没有记忆"，
        容易把上下文里其他内容误当成记忆。
        显式告诉它"这一层是空的"，它才会在需要依据时选择说不知道。

    L1 的 role_text.format(**_safe_role(role))：
        L1_role.txt 是个带 {identity} {tone} 占位符的模板，
        用角色卡的字段填充它。这样改角色表现不用改代码，改模板或改角色卡即可。

    L4 的片段带序号/来源/页码：
        让模型在答案里标注引用时有据可依。
    """
    safety = _read("L0_safety.txt") or "遵守安全边界；知识不足时拒答；专业领域必须提示咨询专业人士。"
    # 兜底的 L1 模板同样带占位符，保证 format 调用在两种情况下都能工作
    role_text = _read("L1_role.txt") or "你是{identity}，语气：{tone}。禁忌：{taboo_topics}。免责声明：{disclaimer}。"
    long_text = "\n".join(f"- {item.get('text', item.get('content', ''))}" for item in long_memory) or "（暂无长期记忆）"
    short_text = "\n".join(f"{item.get('speaker')}: {item.get('content')}" for item in short_memory) or "（新会话）"
    # enumerate(..., 1) 从 1 开始编号，片段序号与人读的习惯一致
    context_text = "\n\n".join(_format_context(hit, idx) for idx, hit in enumerate(contexts, 1)) or "（无检索上下文）"
    system = "\n\n".join([
        f"【L0 安全合规层】\n{safety}",
        "【L1 角色人设层】\n" + role_text.format(**_safe_role(role)),
        f"【L2 长期记忆层】\n{long_text}",
        f"【L3 短期记忆层】\n{short_text}",
        f"【L4 检索上下文层】\n{context_text}",
    ])
    user = f"【L5 当前输入层】\n{user_message}\n请保持角色一致，引用来源，无法依据知识库回答时说明边界。"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _safe_role(role: dict[str, Any]) -> dict[str, Any]:
    """为 L1 模板补全必需的占位符字段。

    参数：
        role: 角色卡
    返回：
        补齐了 identity/tone/taboo_topics/disclaimer 的字典。

    为什么必须有这个函数：
        L1 模板里有四个占位符，只要角色卡缺任何一个，
        str.format 就会抛 KeyError 让整轮对话失败。
        这里先铺一层默认值，再用 defaults.update(role) 让角色卡的真实字段覆盖它们 ——
        既不会因为缺字段而崩，也不会丢掉角色自己的设定。

    注意 update 的方向（role 覆盖 defaults）：
        反过来的话默认值会盖掉真实角色设定，所有角色都变成同一个"可靠助手"。
    """
    defaults = {"identity": role.get("name", "可靠助手"), "tone": "清晰、谨慎", "taboo_topics": "编造事实", "disclaimer": "信息仅供参考"}
    defaults.update(role)
    return defaults


def _format_context(hit: dict[str, Any], idx: int) -> str:
    """把一个检索片段格式化成"带出处的一段文本"。

    参数：
        hit: 一条召回结果
        idx: 片段序号（从 1 开始）
    返回：
        形如 "[片段1 来源:manual.pdf 页码:12]\\n正文…" 的文本。

    关键在那一行头信息：
        它是"答案可溯源"的基础 —— 模型看到来源和页码，
        才能在回答末尾正确标注 [来源:manual.pdf 第12页]。

    三层兜底：
        hit.get("entity") or hit    兼容 Milvus 的嵌套返回结构
        doc_source / source         兼容新架构和 backend 两条主线的字段名
        page 默认 "?"              页码缺失时显示问号，而不是显示 -1 这种内部值
    """
    entity = hit.get("entity") or hit
    source = entity.get("doc_source") or entity.get("source") or "unknown"
    page = entity.get("page", "?")
    content = entity.get("content") or entity.get("text") or ""
    return f"[片段{idx} 来源:{source} 页码:{page}]\n{content}"
