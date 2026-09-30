CRISIS_TERMS = (
    "想自杀",
    "想死",
    "不想活",
    "结束生命",
    "自杀",
    "自残",
    "伤害自己",
    "伤害别人",
)

CRISIS_RESPONSE = (
    "听起来你现在可能正面临紧急风险。请先尽量不要独处，并远离可能用来伤害自己或他人的物品。"
    "请立即联系当地急救或警方、前往最近的急诊，并联系一位你信任的人陪着你。"
    "如果你愿意，可以告诉我你现在是否安全、身边是否有人；我会继续陪你，但我不能替代紧急救助或专业人员。"
)


def is_crisis_message(message: str) -> bool:
    normalized = "".join(message.lower().split())
    return any(term in normalized for term in CRISIS_TERMS)
