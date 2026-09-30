from dataclasses import dataclass


@dataclass(frozen=True)
class RiskResult:
    # 风险检测结果只表达等级和命中类别，不记录原文。
    level: str
    categories: list[str]


_RISK_KEYWORDS = {
    "family_violence": ["家暴", "殴打", "打我", "暴力"],
    "active_threat": ["威胁", "杀", "砍", "跟踪", "堵门"],
    "ongoing_danger": ["正在", "现在", "马上", "危险"],
    "minor": ["孩子", "未成年", "小孩", "儿童"],
}

_MINOR_RISK_KEYWORDS = ["危险", "伤害", "威胁", "家暴", "报警", "殴打", "打", "受伤", "虐待"]


def detect_risk(text: str) -> RiskResult:
    # 用户明确说明无危险或安全时，不因“危险”二字误入紧急路径。
    if "无危险" in text or "没有危险" in text or "安全" in text:
        return RiskResult(level="normal", categories=[])
    # 逐类扫描关键词，MVP 用确定性规则优先保证高风险场景不漏过。
    categories: list[str] = []
    for category, keywords in _RISK_KEYWORDS.items():
        # 任一关键词命中即记录该风险类别。
        if any(keyword in text for keyword in keywords):
            categories.append(category)
    if "minor" in categories and any(keyword in text for keyword in _MINOR_RISK_KEYWORDS):
        categories.append("minor_danger")
    # 家暴、现时威胁、正在发生危险、未成年人危险均按紧急安全路径处理。
    if {"family_violence", "active_threat", "ongoing_danger", "minor_danger"} & set(categories):
        return RiskResult(level="emergency", categories=categories)
    return RiskResult(level="normal", categories=categories)


def emergency_guidance() -> str:
    # 安全指引固定、可审计，不依赖 RAG 或模型生成。
    return "请先确保人身安全：如正在遭受暴力、威胁或有未成年人危险，请立即拨打110；有伤情请及时拨打120或就近急诊。可联系当地妇联、居委会/村委会、法律援助中心或12348公共法律服务热线。尽量在安全前提下保存报警记录、伤情照片、诊断证明、聊天记录、录音录像和证人信息，并尽快远离施暴者。"
