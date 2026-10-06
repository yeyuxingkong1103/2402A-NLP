"""危机干预：危机词检测 + 转介提示 + 敏感词后处理。

职责：安全兜底层，独立于角色提示词——不论哪个心理医生角色、模型输出什么，
只要用户消息命中危机信号，最终回答都必须在末尾包含转介信息（见下"为什么兜底"）。

被谁调用：rag_service 问答前 detect_crisis、问答后 post_process。

为什么要兜底追加转介提示：
LLM 存在"不听话"的可能（漏讲热线、被特殊表述绕过安全提示），而危机场景漏提示的代价极高，
故这里不信任模型——只要标记了 crisis，就在回答尾部强制拼接转介文案；
宁可出现重复提示，也不能出现"该提示没提示"。
注意：本模块只做"提醒式"处理，不做内容审查拦截，避免误伤正常倾诉。
"""
import re
from typing import Dict, List, Tuple

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("crisis")

# 高危危机词（自伤/自杀/伤人）
CRISIS_KEYWORDS: List[str] = [
    "自杀", "自尽", "自残", "自伤", "不想活", "不想活了", "活着没意思", "活着没有意思",
    "结束生命", "结束自己", "了结自己", "伤害自己", "割腕", "跳楼", "跳下去", "吃安眠药",
    "上吊", "轻生", "寻死", "想死", "去死", "死了算了", "解脱了", "杀人", "报复社会",
    "伤害别人", "杀了", "同归于尽",
    # 委婉高危表述（测试发现"我想结束这一切"漏检后补充）
    "结束这一切", "了结这一切", "离开这个世界",
]

# 需弱化处理的高危表述（用于判断语气强度，不直接拦截）
RISK_PATTERNS = [
    re.compile(r"(活|存在)(着)?(好)?(累|没意义|没意思|痛苦)"),
    re.compile(r"(撑不下去|熬不下去|坚持不住|活不下去)"),
]

# 敏感词（仅做提醒式脱敏，不做内容审查拦截）
SENSITIVE_WORDS: List[str] = ["毒品", "赌博", "色情", "裸聊", "枪支", "办证", "刷单"]

_CRISIS_NOTICE_TEMPLATE = (
    "\n\n——\n**请优先关注安全**：如果此刻你有伤害自己或他人的想法，请立即联系"
    "心理援助热线 {hotline}、急救 120 或报警 110，也可以前往当地精神卫生中心急诊。"
    "如果身边有人可以陪伴，请告诉他们你现在的状态。我会继续在这里陪着你。"
)


def detect_crisis(text: str) -> Tuple[bool, List[str]]:
    """检测危机词，返回 (是否命中, 命中词列表)。"""
    if not text:
        return False, []
    hits = [kw for kw in CRISIS_KEYWORDS if kw in text]
    if not hits:
        for pattern in RISK_PATTERNS:
            matched = pattern.search(text)
            if matched:
                hits.append(matched.group(0))
    if hits:
        logger.warning("检测到危机表达：%s", hits)
    return bool(hits), hits


def crisis_notice() -> str:
    """渲染转介文案（热线号从配置读取，不硬编码，换地区只改配置即可）。"""
    return _CRISIS_NOTICE_TEMPLATE.format(hotline=settings.crisis_hotline)


def mask_sensitive_words(text: str, replacement: str = "*") -> str:
    """敏感词脱敏（按词长等量替换为 *），只做提醒式弱化，不做审核拦截。"""
    result = text
    for word in SENSITIVE_WORDS:
        if word in result:
            result = result.replace(word, replacement * len(word))
    return result


def ensure_crisis_notice(answer: str) -> str:
    """确保危机提示存在，避免大模型遗漏转介。"""
    if settings.crisis_hotline in answer or "120" in answer or "110" in answer:
        return answer
    return answer + crisis_notice()


def post_process(answer: str, crisis: bool = False) -> str:
    """后处理：正则清理 + 敏感词脱敏 + 危机转介兜底。"""
    if not answer:
        return answer
    cleaned = re.sub(r"\n{3,}", "\n\n", answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    cleaned = mask_sensitive_words(cleaned)
    # 合规兜底：模型偶尔会"自称能诊断/开药"，这里用正则替换成免责表述
    cleaned = re.sub(r"(作为(一个)?(心理|精神)(医生|咨询师)?[，,]?\s*我可以(为你)?(诊断|开药))", "我不能进行诊断或开药", cleaned)
    if crisis:
        cleaned = ensure_crisis_notice(cleaned)
    return cleaned