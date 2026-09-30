# app/core/prompt.py
"""提示词模板。

角色人设与规则由 MySQL 的 roles 表驱动（见 RoleService），
本模块负责把它们和检索到的参考资料、双层记忆组装成最终 messages。

分层结构：
    system  = 角色人设 + 回答规则 + 用户长期记忆 + 本轮参考资料
    history = 最近 N 轮短期记忆（真实多轮消息）
    user    = 当前问题
"""
from typing import Any, Dict, List, Optional

# 所有角色共用的基础规则，角色自身的 rules 会追加在后面
BASE_RULES = """1. 严格依据【参考资料】作答，不得编造资料中不存在的事实、条款或数据。
2. 参考资料不足以回答时，按下面的要求回复：{fallback}
3. 始终保持角色设定中的身份、语气与立场，不要跳出角色。
4. 回答要条理清晰；涉及专业结论时优先引用资料原文。
5. 不要向用户暴露"参考资料""知识库""检索"等系统实现细节。"""


def build_system_prompt(persona: str,
                        rules: Optional[str] = None,
                        fallback: Optional[str] = None,
                        memories: Optional[List[str]] = None,
                        contexts: Optional[List[Dict[str, Any]]] = None) -> str:
    """拼装 system 提示词。"""
    fallback = fallback or "坦诚说明你不掌握该信息，并建议用户补充细节或咨询专业人士。"
    parts = [persona.strip()]

    rule_text = BASE_RULES.format(fallback=fallback)
    if rules:
        rule_text += "\n" + rules.strip()
    parts.append("【回答规则】\n" + rule_text)

    if memories:
        mem = "\n".join("- %s" % m for m in memories if m)
        if mem:
            parts.append("【你对该用户的长期记忆】\n"
                         "以下是你之前与这位用户交流时记下的信息，可用来保持连贯：\n" + mem)

    if contexts:
        blocks = []
        for i, c in enumerate(contexts, 1):
            title = c.get("title") or c.get("source") or ""
            head = "资料%d" % i + ("（%s）" % title if title else "")
            blocks.append("%s：\n%s" % (head, (c.get("text") or "").strip()))
        parts.append("【参考资料】\n" + "\n\n".join(blocks))
    else:
        parts.append("【参考资料】\n（本轮没有检索到相关资料）")

    return "\n\n".join(parts)


# messages 的组装已改由 LangChain 的 ChatPromptTemplate + MessagesPlaceholder
# 负责（见 app/core/chain_service.py），本模块只保留 system 提示词的拼装。
