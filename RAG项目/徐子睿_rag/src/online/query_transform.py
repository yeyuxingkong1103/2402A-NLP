"""src/online/query_transform.py —— 查询改写：指代消解 + 术语扩展。

在链路中的位置：
    src/online/retriever.py → 【本文件】 → 改写后的查询用于向量化和 Milvus 检索

与 backend/retrieval.py 的 rewrite_query 是同一套思路的新架构版本：
    都使用**规则**而不是让大模型改写。原因不变 —— 规则快、零成本、
    可解释、可复现，而且答辩时能准确说出"是哪条规则救回了这题"。

本文件比 backend 那个多了一步"指代消解"：
    backend 那条主线的对话是一问一答式的知识问答；
    新架构是角色扮演式的连续对话，用户会说"它有什么要求？""这个怎么处理？"，
    这类追问单看一句完全无法检索，必须借助历史把代词还原成具体名词。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryBundle:
    """一次查询改写的结果。

    字段：
        original:   用户原始问题（保留下来便于对比和展示）
        rewritten:  最终用于检索的查询串（原查询 + 扩展词）
        expansions: 新补充进来的术语（前端可高亮"系统替你补了什么"）
    """

    original: str
    rewritten: str
    expansions: list[str]


def resolve_coreference(query: str, history: list[dict] | None = None) -> str:
    """用对话历史消解查询开头的指代词。

    参数：
        query: 用户本轮问题
        history: 历史消息列表（每项含 speaker / content）
    返回：
        消解后的查询；无需消解时原样返回。

    做法（保守的字符串拼接，不是真正的指代消解）：
        如果问题以"它/这个/那这个/上述"开头，就把上一条用户消息拼在前面，
        形成"上一句；追问：本轮问题"的形式。

    为什么是这种做法：
        真正的指代消解需要模型，成本和不确定性都高。
        而拼接能达到同样的检索效果 —— 检索系统看到的是完整语义，
        上一句提供了"它"指的是什么，本轮问题提供了真正要问的内容。
        这恰好印证了本项目的整体取向：用简单可靠的规则解决工程问题。

    三个保守设计，都是为了"宁可不错，不可误伤"：
        1. 只在**开头**出现代词时才处理（startswith）——
           句中出现的"它"可能指代别的东西，贸然拼接会引入错误上下文
        2. 用 next(...) 取**最近一条用户消息**而不是最后一条消息 ——
           最后一条可能是助手回答，而"它"通常指用户自己刚说过的话题
        3. 找不到历史用户消息（last_user 为空）就原样返回
    """
    query = (query or "").strip()
    if not history:
        return query
    # reversed 从最近往前找；next 带默认值，找不到时为空串而不是抛 StopIteration
    last_user = next((item.get("content", "") for item in reversed(history) if item.get("speaker") == "user"), "")
    if query.startswith(("它", "这个", "那这个", "上述")) and last_user:
        return f"{last_user}；追问：{query}"
    return query


def expand_query(query: str) -> QueryBundle:
    """按规则表扩展专业术语。

    参数：
        query: 查询串（通常是已做指代消解的）
    返回：
        QueryBundle(original, rewritten, expansions)

    规则表把"用户的口语词"映射到"文档里的实际术语"：
        赔偿 → 责任 / 损失 / 证据 / 法律依据
        合同 → 违约 / 条款 / 履行 / 解除
        焦虑 → 情绪 / 呼吸 / 支持 / 求助
        抑郁 → 情绪低落 / 睡眠 / 风险 / 专业帮助
    这正是"字面匹配"路的补救措施：
    稀疏检索只能匹配原样出现的词，用户说"焦虑"而文档写"情绪低落"就搜不到，
    所以要在查询侧把这些同义表达补进去。

    rewritten 的构造用 dict.fromkeys 去重且保序：
        原查询必须放在最前面（它是最重要的检索意图），
        扩展词跟在后面；同一批输入两次运行得到的 rewritten 完全相同 ——
        可复现性对评测归因至关重要（否则无法判断指标变化来自哪次改动）。

    注意这里没有做 normalize：
        backend 那边需要统一 SF6/标准号写法，是因为它面向国标文档；
        本文件面向的是角色扮演场景（法律/心理类问答），
        没有那类术语形态问题，所以不做多余处理。
    """
    normalized = re.sub(r"\s+", " ", query).strip()
    expansions = []
    rules = {
        "赔偿": ["责任", "损失", "证据", "法律依据"],
        "合同": ["违约", "条款", "履行", "解除"],
        "焦虑": ["情绪", "呼吸", "支持", "求助"],
        "抑郁": ["情绪低落", "睡眠", "风险", "专业帮助"],
    }
    for key, values in rules.items():
        if key in normalized:
            expansions.extend(values)
    # dict.fromkeys 去重保序：[normalized, *expansions] 保证原查询排第一
    rewritten = " ".join(dict.fromkeys([normalized, *expansions]))
    return QueryBundle(query, rewritten, expansions)
