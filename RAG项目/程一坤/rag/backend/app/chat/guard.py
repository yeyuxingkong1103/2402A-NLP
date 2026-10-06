"""回答护栏模块：拒答判定、绝对化表述拦截、免责声明、引用失败统一处置。

三大护栏：
1. 对低相关候选在服务流程中拒答，避免把无关法源硬套进回答
2. 拦截绝对化表述（"保证胜诉"、"必然"、"确定违法"等）
3. 确保每个回答都有固定免责声明

另含 resolve_citation_outcome：引用校验失败后的统一处置阶梯（批次 32 抽取），
service.py（同步）与 streaming.py（流式）共用，替换原先两份逐字平行的 except 实现。
"""
import re
from typing import Any

from app.chat.citation_check import (
    NoCitationError,
    UncitedConclusionError,
    UnknownLawCitationError,
)


# 拒答文本（无候选条文时返回）
REFUSAL_ANSWER = """现有资料无法确认这个问题的答案。

已检索范围：知识库中的中国大陆劳动法法规与司法解释文本。
未找到与该问题直接相关的法条。

建议：
1. 补充事实信息（发生时间、涉及主体、具体行为）
2. 若涉及具体纠纷或权益，请携带完整材料咨询执业律师

内容仅供法律信息参考，不能替代律师出具的正式法律意见。"""


# 低分寒暄引导语：仅由低分处置模块对白名单寒暄调用，不替代真实问题拒答。
GREETING_ANSWER = """你好！我是法律知识助手，主要帮助你了解中国大陆劳动法相关问题。

你可以问我：劳动合同、工资和加班、经济补偿、违法解除、工伤、社保、劳动仲裁等。
例如：
1. 公司单方面解除劳动合同，可能涉及哪些补偿？
2. 加班费通常按照什么标准计算？

内容仅供法律信息参考，不能替代律师出具的正式法律意见。"""

# 引用校验失败的整段替换文本（回答零引用/引用越界等，回答不可信时用）
CITATION_FALLBACK_NOTICE = """抱歉，本次回答未通过引用校验：回答缺少可核查的法条引用，无法保证内容可靠性。

建议：
1. 重新提问，或补充具体事实信息（发生时间、涉及主体、具体行为）
2. 若涉及具体纠纷或权益，请携带完整材料咨询执业律师

内容仅供法律信息参考，不能替代律师出具的正式法律意见。"""


# 回答只是提到清单外法规时的追加警示行（回答本身保留，不整段替换）
UNKNOWN_LAW_WARNING = (
    "⚠️ 提示：回答中提到的部分法规名称不在本次检索到的法源清单内，"
    "相关表述未经原文核对，请以官方发布文本为准。"
)


# 有候选条文但回答零引用时的追加警示行（批次 10 分级：回答保留，不整段替换）。
# 与零引用整段替换的旧行为相比，模型未标 [n] 不等于回答没有依据——
# 内容可能来自注入的法条，只是没标引用编号，整段替换会浪费一次可用的回答。
NO_CITATION_WARNING = (
    "⚠️ 提示：本回答未引用具体法条编号，内容未经原文逐条核对，"
    "请对照下方法源清单或官方发布文本核实。"
)


# 回答整体有引用、但个别句子是无引用的确定性结论时的追加警示行。
# 比整篇零引用轻：主体结论已可核查，只提示"个别结论没标引用"，
# 因此保留回答、不整段替换（严重度与处置对齐）。
UNCITED_CONCLUSION_WARNING = (
    "⚠️ 提示：本次回答有部分结论未标注引用，请以引用条目为准，"
    "未标注部分请对照法源清单或官方发布文本核实。"
)


def append_unknown_law_warning(answer: str) -> str:
    """在回答末尾追加清单外法规警示行（已有则不重复）。"""
    if UNKNOWN_LAW_WARNING in answer:
        return answer
    return f"{answer.rstrip()}\n\n{UNKNOWN_LAW_WARNING}"


def append_no_citation_warning(answer: str) -> str:
    """在回答末尾追加零引用警示行（已有则不重复）；回答本身保留。"""
    if NO_CITATION_WARNING in answer:
        return answer
    return f"{answer.rstrip()}\n\n{NO_CITATION_WARNING}"


def append_uncited_conclusion_warning(answer: str) -> str:
    """追加"个别结论未标引用"警示行（已有则不重复）；回答本身保留。"""
    if UNCITED_CONCLUSION_WARNING in answer:
        return answer
    return f"{answer.rstrip()}\n\n{UNCITED_CONCLUSION_WARNING}"


# 事件名前缀（批次 32 收拢为唯一出处；service/streaming 不再各自拼事件字符串）
_EVENT_UNKNOWN_LAW = "citation_warning_unknown_law"
_EVENT_NO_CITATION = "citation_warning_no_citation"
_EVENT_UNCITED_CONCLUSION = "citation_warning_uncited_conclusion"
_EVENT_CHECK_FAILED = "citation_check_failed"


def resolve_citation_outcome(answer: str, error: Exception) -> tuple[str, str]:
    """引用校验失败后的统一处置：返回 (处置后回答, 护栏事件名)。

    service.chat（同步）与 chat_stream（流式）共用的唯一实现；处置分级由
    citation_check 的异常子类决定，调用方不再各自维护 except 阶梯：
    - UnknownLawCitationError / NoCitationError / UncitedConclusionError：
      回答保留，末尾追加对应警示行，事件名带 citation_warning_ 前缀；
    - 其余 CitationError（空回答 / 引用越界等不可核查硬违规）：
      整段替换为 CITATION_FALLBACK_NOTICE，事件名 citation_check_failed。

    事件名格式与抽取前逐字一致（`前缀: {error}`），历史日志/评测口径不受影响。
    """
    if isinstance(error, UnknownLawCitationError):
        return append_unknown_law_warning(answer), f"{_EVENT_UNKNOWN_LAW}: {error}"
    if isinstance(error, NoCitationError):
        return append_no_citation_warning(answer), f"{_EVENT_NO_CITATION}: {error}"
    if isinstance(error, UncitedConclusionError):
        return (
            append_uncited_conclusion_warning(answer),
            f"{_EVENT_UNCITED_CONCLUSION}: {error}",
        )
    return CITATION_FALLBACK_NOTICE, f"{_EVENT_CHECK_FAILED}: {error}"


# 绝对化表述模式（需要拦截的词汇）
ABSOLUTE_PATTERNS = [
    r"保证胜诉",
    r"必然胜诉",
    r"肯定胜诉",
    r"一定胜诉",
    r"必然败诉",
    r"肯定败诉",
    r"一定败诉",
    r"确定违法",
    r"必然违法",
    r"肯定违法",
    r"绝对违法",
    r"百分之百",
    r"100%",
    r"必然会",
    r"一定会",
    r"肯定会",
    r"必然",  # 通用的"必然"
]


# 绝对化表述到审慎表述的映射
ABSOLUTE_TO_CAUTIOUS = {
    "保证胜诉": "较大可能胜诉",
    "必然胜诉": "有较大胜算",
    "肯定胜诉": "胜诉可能性较高",
    "一定胜诉": "胜诉可能性较高",
    "必然败诉": "败诉可能性较高",
    "肯定败诉": "可能败诉",
    "一定败诉": "可能败诉",
    "确定违法": "可能构成违法",
    "必然违法": "通常认定为违法",
    "肯定违法": "一般视为违法",
    "绝对违法": "通常构成违法",
    "百分之百": "很大概率",
    "100%": "很大概率",
    "必然会": "通常会",
    "一定会": "一般会",
    "肯定会": "可能会",
    "必然": "通常",  # 通用替换，放最后避免误替换
}


def should_refuse(candidates: list[Any], *, min_vector_score: float = 0.0) -> bool:
    """判断是否应该拒答。

    两条拒答条件（满足其一即拒答）：
    1. 一条候选都没有（历史行为）；
    2. 候选中最高的向量相似度低于 min_vector_score（阶段 8.4 校准出的阈值）。

    为什么用向量相似度而不是重排分：评测集实测两个信号的分辨力——
    向量分（余弦）拒答题最高 0.6281 / 非拒答题最低 0.6328，可完全分离；
    重排分两簇重叠（拒答题最高 0.2326 > 非拒答题最低 0.0597），做门限会误拒 6 条。
    详见 reports/refusal_calibration_*.md。

    Args:
        candidates: 检索到的候选条文列表
        min_vector_score: 向量相似度下限；0 表示不启用阈值（默认行为不变）。
            阈值只对"有向量分"的候选生效：候选全部来自关键词召回（无向量分）时
            不作答阈值判断，避免把缺分误判成"查不到"。

    Returns:
        True 表示应该拒答，False 表示可以尝试回答
    """
    if len(candidates) == 0:
        return True
    if min_vector_score <= 0:
        return False
    scores = [
        float(score)
        for score in (getattr(candidate, "vector_score", None) for candidate in candidates)
        if score is not None
    ]
    # 没有向量分（例如全部候选来自关键词召回）→ 不凭阈值拒答
    if not scores:
        return False
    return max(scores) < min_vector_score


def check_absolute_expressions(answer: str) -> list[str]:
    """检测回答中的绝对化表述。

    Args:
        answer: 大模型生成的回答

    Returns:
        问题列表，每项描述一个绝对化表述。空列表表示没有问题。
    """
    issues = []

    for pattern in ABSOLUTE_PATTERNS:
        matches = re.findall(pattern, answer)
        if matches:
            issues.append(f"包含绝对化表述：{pattern}（出现 {len(matches)} 次）")

    return issues


def rewrite_absolute_expressions(answer: str) -> str:
    """重写绝对化表述为审慎表述。

    Args:
        answer: 包含绝对化表述的回答

    Returns:
        重写后的回答
    """
    rewritten = answer

    # 按长度降序排序，确保先替换长的词组（避免"必然败诉"被"必然"误替换）
    sorted_pairs = sorted(ABSOLUTE_TO_CAUTIOUS.items(), key=lambda x: len(x[0]), reverse=True)

    for absolute, cautious in sorted_pairs:
        if absolute == "100%":
            # 例外：「50%以上100%以下」是法条里的法定比例区间（如加付赔偿金条款），
            # 不是绝对化承诺，替换会篡改法条原文。仅当 100% 不夹在"以上…以下"之间才改写。
            rewritten = re.sub(r"(?<!以上)100%(?!以下)", cautious, rewritten)
        else:
            rewritten = rewritten.replace(absolute, cautious)

    return rewritten


def ensure_disclaimer(answer: str) -> str:
    """确保回答包含固定免责声明。

    Args:
        answer: 回答文本

    Returns:
        带免责声明的回答（如果已有则不重复添加）
    """
    disclaimer = "内容仅供法律信息参考，不能替代律师出具的正式法律意见。"

    # 检查是否已有免责声明
    if disclaimer in answer or "内容仅供" in answer:
        return answer

    # 添加免责声明（用空行分隔）
    return f"{answer.rstrip()}\n\n{disclaimer}"


def apply_guardrails(answer: str, candidates: list[Any]) -> str:
    """应用所有护栏规则。

    Args:
        answer: 大模型生成的回答
        candidates: 检索到的候选条文列表

    Returns:
        经过护栏处理后的回答
    """
    # 护栏 1：无候选时保留模型的事实梳理与一般性引导。
    # 服务流程会把"本次没有可引用法源"明确注入提示词，模型不得编造具体法律依据。
    # 低相关候选仍在 service/chat_stream 的拒答分支中提前拦截。

    # 护栏 2：检测并重写绝对化表述
    issues = check_absolute_expressions(answer)
    if issues:
        answer = rewrite_absolute_expressions(answer)

    # 护栏 3：确保免责声明
    answer = ensure_disclaimer(answer)

    return answer
