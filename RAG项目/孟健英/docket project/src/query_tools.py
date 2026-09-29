# -*- coding: utf-8 -*-
"""查询侧工具：输入完整性、多轮改写、角色检索偏向、澄清选项。"""
import re  # 正则：解析 LLM 输出的编号选项

from langchain_core.messages import HumanMessage  # 消息封装：把提示词包成用户消息送 LLM

from src import prompts  # 提示词模板层：改写/澄清模板统一维护

# 澄清兜底选项：模型没给出可用选项时使用。
_DEFAULT_OPTIONS = [  # 低相关度澄清的兜底选项
    "高血压的用药问题",  # 方向一：用药
    "高血压的饮食与生活方式",  # 方向二：饮食
    "高血压的症状与危害",  # 方向三：症状
]  # 固定 3 个方向，覆盖知识库主要内容

# 输入不完整（断句/缺宾语）时的追问选项。
_INCOMPLETE_OPTIONS = [  # 断句追问选项
    "我在吃降压药，想问问有没有影响",  # 引导用户补充"是否用药"
    "我没有吃药，想问平时怎么调理",  # 引导用户补充"调理诉求"
    "我说的是别的意思，我重新描述一下",  # 给用户重新表述的出口
]  # 可点击选项，降低用户输入成本

# 这些词结尾说明话没说完，先追问再检索。
_DANGLING = ("有吃", "在吃", "有在", "吃的", "用的", "是的", "然后", "还有", "但是", "而且", "就是", "我想", "我")  # 断句信号词：以这些词结尾说明话没说完（如"我在吃"）

# 疾病锚点：知识库只覆盖高血压。情绪/症状类口语问句缺主体时补上锚点，
# 否则重排分过低（实测"我最近很焦虑"仅 0.087），会被误判成无关问题而错误追问。
_ANCHOR = "高血压"  # 疾病锚点：本项目知识库只覆盖高血压
_ANCHOR_HINTS = (  # 需要补锚点的信号词
    "情绪", "焦虑", "紧张", "压力", "睡眠", "失眠", "心情", "抑郁", "烦躁",  # 情绪类：口语化、缺疾病主体
    "头晕", "头痛", "心慌", "乏力", "耳鸣",  # 症状类
)  # 命中任一信号且不含"高血压"才考虑补锚点

# 纯症状词：这些词单独出现时不补"高血压"锚点（知识库无鉴别诊断内容，补了反而降分）
_SYMPTOM_ONLY = ("头晕", "头痛", "心慌", "乏力", "耳鸣")  # 纯症状词单独出现时不补锚点


def needs_clarify(query: str) -> bool:  # 判断输入是否明显不完整（断句/缺宾语）
    """输入明显断句/缺宾语时先追问，不要硬答。"""
    q = query.strip().rstrip("。.！!~ ")  # 去掉首尾空白和句末标点，再看结尾词
    if not q or len(q) > 14:  # 空串或长句不追问：长句一般已表达完整
        return False  # 不需要澄清
    return q.endswith(_DANGLING)  # 以断句信号词结尾才算不完整


def incomplete_options(query: str) -> list:  # 断句时的追问选项
    """输入不完整时的追问选项。"""
    return list(_INCOMPLETE_OPTIONS)  # 拷贝一份返回，防调用方改坏全局常量


def anchor_query(query: str) -> str:  # 给重排打分用的查询补疾病锚点
    """给缺疾病主体的情绪/症状类问句补锚点（仅用于重排打分，不改用户原话）。

    纯症状词（头晕/头痛/心慌等）不补锚点——知识库没有鉴别诊断内容，
    补了"高血压"反而让 reranker 拿"高血压 头晕"去比高血压指南，分更低。
    情绪类问句仍补锚点（心理医生角色需要匹配高血压情绪相关片段）。
    """
    if _ANCHOR in query or not any(w in query for w in _ANCHOR_HINTS):  # 已含"高血压"或不含情绪/症状信号：不用补
        return query  # 原样返回
    # 纯症状问句不补锚点
    if any(w in query for w in _SYMPTOM_ONLY):  # 纯症状问句：补了反而拉低 rerank 分
        return query  # 原样返回
    return f"{query} {_ANCHOR}"  # 拼上锚点，让重排打分有疾病上下文


def role_queries(role: dict, query: str) -> list:  # 生成角色偏向的额外检索路
    """角色偏向检索：如心理医生补上情绪/睡眠类关键词。"""
    boost = role.get("boost") or []  # 角色检索增强词（如心理医生的情绪/睡眠关键词）
    if boost:  # 有增强词才加一路
        return [f"{query} {' '.join(boost)}"]  # 增强词拼进查询，提高角色相关片段的召回
    return []  # 无增强词就不加路


def role_filter(role: dict, candidates: list) -> list:  # 角色后置过滤
    """含角色禁用术语的片段直接移除（西医/心理医生不复用中医辨证资料）。"""
    banned = role.get("banned_terms") or []  # 该角色的禁用术语表
    if not banned:  # 没配置禁用词
        return candidates  # 原样返回
    return [c for c in candidates if not any(t in c.get("text", "") for t in banned)]  # 含任一禁用词的片段整条移除


def rewrite_query(llm, query: str, history: list, force_join: bool = False) -> str:  # 多轮查询改写：把省略句补全成可独立检索的问句
    """把"一直持续了一天"这类省略句补全为可独立检索的问句。

    force_join：澄清/追问后用户点了选项，此时必须并上上一轮的问题一起检索。
    """
    user_turns = [m for m in history if m["role"] == "user"]  # 只看用户说过的话
    if not user_turns:  # 首轮对话没有历史
        return query  # 原样返回
    if force_join:  # 用户点了澄清选项：必须把上一轮问题和这次选择拼一起检索
        return f"{user_turns[-1]['content']} {query}"[:80]  # 直接拼接并截 80 字：确定场景要确定性行为，不再过 LLM
    if len(query) > 12:  # 问句够长说明信息已完整
        return query  # 不用改写
    fallback = f"{user_turns[-1]['content']} {query}"  # 兜底策略：LLM 改写失败就机械拼接上一轮问题
    try:  # 调 LLM 改写可能失败，包 try
        context = prompts.format_history(history[-4:], limit=4)  # 只取最近 4 轮做上下文：够理解指代又不爆 token
        content = llm.invoke([HumanMessage(content=prompts.rewrite_prompt(context, query))]).content.strip()  # 让 LLM 按模板把省略句补全
        rewritten = content.splitlines()[0].strip()  # 只取第一行：防模型啰嗦输出解释
        if 2 < len(rewritten) <= 60:  # 长度合理性校验：太短没意义，太长大概率跑偏
            return rewritten  # 改写合格就用新查询
        return fallback  # 不合格用兜底拼接
    except Exception:  # LLM 超时/报错
        return fallback  # 改写失败也要能检索：降级为机械拼接


def clarify_options(llm, query: str, snippets: list) -> list:  # 低相关度时让 LLM 生成澄清选项
    """低相关度时生成 2-3 个澄清选项。"""
    joined = "\n".join(f"- {s}" for s in snippets[:3]) or "（无）"  # 最多取 3 条候选片段给 LLM 参考
    try:  # LLM 调用包 try
        content = llm.invoke([HumanMessage(content=prompts.clarify_prompt(query, joined))]).content  # 按澄清模板生成选项
        options = [  # 从输出里解析编号行
            re.sub(r"^\s*\d+[.、)]\s*", "", line).strip()  # 去掉"1.""2、"等编号前缀
            for line in content.splitlines()  # 逐行扫描
            if re.match(r"^\s*\d+[.、)]", line)  # 只收编号开头的行
        ]  # 解析出的选项列表
        return options[:3] or _DEFAULT_OPTIONS  # 最多 3 个；解析不出来就用兜底选项
    except Exception:  # LLM 失败
        return _DEFAULT_OPTIONS  # 也保证有选项可点
