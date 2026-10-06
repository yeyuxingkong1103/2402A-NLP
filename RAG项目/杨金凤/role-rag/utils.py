"""通用纯函数：文本后处理 + 医疗安全兜底（急症检测 / 免责声明）。"""
import logging
import re

logger = logging.getLogger(__name__)

# 医疗安全兜底：急症关键词（命中即拦截，不走 LLM / 检索）
EMERGENCY_KEYWORDS = (
    "胸痛", "胸闷痛", "呼吸困难", "喘不上气", "意识模糊", "晕倒",
    "昏迷", "剧烈头痛", "视物模糊", "抽搐", "半身不遂", "说话不清",
)

EMERGENCY_MESSAGE = (
    "⚠️ 您描述的症状可能是急症，请立即拨打 120 或前往最近的急诊科，不要等待在线回复。"
)

DISCLAIMER = "\n\n---\n*以上内容仅供健康科普参考，不能替代医生面诊。如有不适，请及时就医。*"


def check_emergency(query: str) -> str | None:
    """命中急症关键词返回固定提示，否则 None。"""
    if any(k in query for k in EMERGENCY_KEYWORDS):
        return EMERGENCY_MESSAGE
    return None


def add_disclaimer(answer: str) -> str:
    """答案已含「不能替代」或「及时就医」则不追加，否则追加免责声明。"""
    if not answer.strip():  # 空答案不追加，避免只输出一条孤零零的免责声明
        return answer
    if "不能替代" in answer or "及时就医" in answer:
        return answer
    return answer + DISCLAIMER


def postprocess(text: str) -> str:
    """对 DeepSeek 生成结果做正则清洗：去代码块围栏、压缩空行、去行尾空格、去首尾空白、统一引用格式。"""
    if not text:
        return text
    before = len(text)
    # a. 去掉 ```json / ```markdown / ``` 等围栏行（保留围栏内内容）
    text = re.sub(r'^[ \t]*```[ \t]*[a-zA-Z0-9_+-]*[ \t]*\n?', '', text, flags=re.MULTILINE)
    # b. 4 个及以上连续空行 -> 2 个空行（5+ 个换行 -> 3 个换行）
    text = re.sub(r'(?:\n[ \t]*){5,}', '\n\n\n', text)
    # c. 去每行行尾空格/制表符
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # d. 去首尾空白
    text = text.strip()
    # e. 统一引用格式：资料1 / 资料 1 / 【资料1】 / 「资料1」 -> 资料1
    text = re.sub(r'[【「]?资料\s*(\d+)[】」]?', r'资料\1', text)
    after = len(text)
    logger.info("postprocess 长度变化 %d -> %d", before, after)
    return text
"""
这是文本后处理与医疗安全兜底模块。分为三部分。第一，`check_emergency()`做急症关键词检测，配置急症关键词列表，如果用户提问里面包含急症词，直接返回急症提醒消息；没有命中急症关键词就返回空。急症检测放在最前面，一旦识别急症，直接拦截，不再调用检索和大模型。第二，`add_disclaimer()`负责追加免责声明。如果回答文本已经自带 “不能替代” 或者 “及时就医” 相关文字，就不再重复追加免责；空答案也不追加，其余情况在回答末尾拼接免责提示。第三，`postprocess()`对大模型返回的原始文本做清洗。使用正则表达式移除 markdown 代码块标记，把大量连续空行压缩，删除每行末尾多余空格，去除首尾空白，统一资料引用标记格式，最后打印日志记录文本处理前后长度。
"""