# -*- coding: utf-8 -*-
"""问题意图识别 + 多路检索路由（知识科普 / 个人病情 / 信息不完整）。

知识科普类必须直接作答（不追问），个人病情类才允许追问 1-2 个关键信息。
"""
from src.query_tools import needs_clarify  # 复用断句判断：话没说完算"信息不完整"意图

KNOWLEDGE, PERSONAL, INCOMPLETE = ("knowledge", "personal", "incomplete")  # 三种基础意图：知识科普/个人病情/信息不完整
EMOTIONAL, SYMPTOM = "emotional", "symptom"  # 另两种意图：情绪安抚/症状咨询

# 情绪表达信号：先接住情绪，再给建议。
_EMOTION_HINTS = ("心里堵", "憋屈", "难受", "烦", "焦虑", "抑郁", "压力大", "不开心", "心情差", "情绪低落")  # 情绪信号词：命中则走"先共情后建议"模板

# 身体症状信号：先追问关键信息，再给一般性建议。
_SYMPTOM_HINTS = ("心悸", "头晕", "胸闷", "头疼", "头痛", "乏力", "失眠", "心慌", "气短", "恶心")  # 症状信号词

# 个人病情信号：出现"我/家人/时间"等词，才允许走追问链路。
_PERSONAL_HINTS = (  # 个人病情信号词
    "我", "本人", "自己", "家人", "妈", "爸", "老公", "老婆", "孩子",  # 人称词：说到具体的人
    "小时", "天了", "几天", "一直", "最近", "刚才", "昨天", "今天", "现在",  # 时间词：说到具体的病程
)  # 两类词都是"在讲自己的病情"的信号

# 清单型问题信号：能不能吃、有哪些、该注意什么。
_LIST_HINTS = ("不能吃", "可以吃", "吃什么", "哪些", "有哪些", "清单", "注意什么", "建议什么")  # 清单型信号：这类问题要拆多路召回保证找全

# 用药类问题信号：需要按五大类降压药多路检索。
_DRUG_HINTS = ("什么药", "吃什么药", "降压药", "用药", "药物", "怎么吃", "有哪几类")  # 用药信号词

# 饮食清单的拆分口径：盐/脂/酒/糖/甘草。
_FOOD_ASPECTS = [  # 饮食清单拆分口径：一个清单问题变 5 路检索，保证各类都召回
    "盐 腌制 加工食品 酱油",  # 控盐方向
    "脂肪 胆固醇 油炸 动物内脏",  # 控脂方向
    "酒精 饮酒",  # 限酒方向
    "糖 含糖饮料 甜食",  # 控糖方向
    "甘草 甘草类制品",  # 甘草方向（会升压，指南特别提示）
]  # 五路饮食口径

# 五大类降压药 + 联合用药，避免只召回一两条。
_DRUG_ASPECTS = [  # 用药拆分口径：一路专找一类药，避免只召回一两条
    "钙通道阻滞剂 CCB 氨氯地平 硝苯地平",  # CCB 类（地平类）
    "ACEI 血管紧张素转换酶抑制剂 普利类",  # ACEI 类（普利类）
    "ARB 血管紧张素受体拮抗剂 沙坦类",  # ARB 类（沙坦类）
    "利尿剂 嗪类 氢氯噻嗪 吲达帕胺",  # 利尿剂
    "β受体阻滞剂 美托洛尔 比索洛尔",  # β受体阻滞剂
    "联合用药 复方制剂 起始剂量",  # 联合用药
]  # 六路用药口径


def detect_intent(query: str) -> str:  # 意图识别主函数：决定走哪条应答路径
    """判断问题类型：有"我/家人/时间"信号算个人病情，否则按知识科普直接答。

    宁可多答一点知识，也不要无缘无故地反问用户（这正是"固定开场白"的根源）。
    """
    q = query.strip()  # 去空白再判断
    if needs_clarify(q):  # 话没说完优先级最高
        return INCOMPLETE  # 走追问链路
    if any(h in q for h in _EMOTION_HINTS):  # 情绪信号其次：先接住情绪
        return EMOTIONAL  # 情绪安抚路径
    if any(h in q for h in _SYMPTOM_HINTS):  # 再看症状
        return SYMPTOM  # 症状咨询路径
    if any(m in q for m in _PERSONAL_HINTS):  # 有"我/家人/时间"才算个人病情
        return PERSONAL  # 允许追问 1-2 个关键信息
    return KNOWLEDGE  # 默认知识科普：直接作答不追问


def is_drug_question(query: str) -> bool:  # 是否用药类问题
    """是否问用药（需要按五大类降压药多路检索）。"""
    return any(h in query for h in _DRUG_HINTS)  # 关键词命中即算


def is_list_question(query: str) -> bool:  # 是否清单型问题
    """是否清单型问题（能不能吃/有哪些）。"""
    return any(h in query for h in _LIST_HINTS)  # 关键词命中即算


def is_multi_route(query: str) -> bool:  # 是否需要多路召回
    """是否需要多路检索（药物分类 / 饮食清单），需要时给更大的 top_k。"""
    return is_drug_question(query) or is_list_question(query)  # 用药或清单都要拆路，主链会给更大的 top_k 保证找全


def multi_queries(query: str) -> list:  # 生成多路检索口径
    """多路检索口径：药物类按五大类拆，饮食类按盐/脂/酒/糖/甘草拆。"""
    if is_drug_question(query):  # 用药类
        return [f"{query} {a}" for a in _DRUG_ASPECTS]  # 原问句逐类拼接，一路专找一类药
    if is_list_question(query):  # 饮食清单类
        return [f"{query} {a}" for a in _FOOD_ASPECTS]  # 按盐/脂/酒/糖/甘草拆 5 路
    return []  # 其他问题不拆路


# 化验单关键词：OCR 拼入的检验报告内容。
_LAB_HINTS = ("检验项目", "参考范围", "检验结果", "化验", "项目名称", "参考区间")  # 化验单特征词：OCR 文本里出现才算化验单场景


def lab_report_query(query: str, fallback: str) -> tuple:  # 化验单检索词提取
    """化验单场景：返回 (True, 异常项检索词)；否则 (False, fallback)。

    从 OCR 文本提取含 +/↑/↓/阳性 的异常项目名作为检索词。
    """
    if "[图片识别内容：" not in query or not any(k in query for k in _LAB_HINTS):  # 双条件：是 OCR 图片内容 + 含化验单特征词
        return False, fallback  # 不是化验单：返回原检索词
    import re  # 函数内导入：只有化验单场景才用正则
    items = re.findall(r"([一-鿿\w()（）·]+)\s*(?:\+{1,3}|↑|↓|阳性)", query)  # 抓异常项：项目名后跟 +/↑/↓/阳性 的就是异常指标
    return True, (f"{' '.join(items[:5])} 临床意义" if items else fallback)  # 最多取 5 个异常项拼"临床意义"作检索词；抓不到就回退
