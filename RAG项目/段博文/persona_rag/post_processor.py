# -*- coding: utf-8 -*-
"""
后处理模块：对 LLM 生成的回答做清洗、校验、格式化
- 正则表达式替换（去掉多余标记、统一标点）
- 引用校验（回答中引用的法条/案号是否在参考资料中存在）
- 格式规范化（Markdown 格式清理）

在系统中的位置：
    上游是对话接口，它在拿到 LLM 的原始输出、但还没落库/推给前端之前调用 post_process；
    下游不依赖任何本项目模块（只依赖 re 和 logger），所以能独立单测。

职责：
    把大模型的"毛坯回答"加工成可以直接展示的成品：洗掉啰嗦的套话与身份自述、
    统一标点空白、清理多余的 Markdown 标记，并顺手统计回答里出现了多少处法条/案号引用。

关键设计取舍：
    全部用正则而不是再调一次小模型来改写——因为后处理必须快（在流式回答结束后立刻执行）
    且必须确定（同样输入必须同样输出，否则无法调试）。代价是规则只能覆盖常见脏格式，
    所以引用校验这里刻意做成"只统计不拦截"，避免误杀正常回答。
"""

import re  # 正则表达式库：所有清洗规则都用 re.sub 实现，无需引入额外的文本处理依赖
from typing import List  # 列表类型标注

from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# ==================== 正则替换规则 ====================
# 每条规则：(正则, 替换文本, 说明)
# 用三元组而不是字典，是因为同一个正则理论上可能重复出现，且"说明"只给读代码的人看
# 规则按列表顺序依次执行，顺序会影响结果：例如先压缩换行、再去行尾空格，
# 能保证"行尾空格清理"看到的是已经规整过的文本；调整顺序前请先想清楚依赖
REPLACEMENT_RULES = [
    # 去掉 LLM 常见的多余前缀（如"根据参考资料，"开头太啰嗦）
    (r"^根据参考资料[，,]\s*", "", "去掉开头的'根据参考资料，'"),
    # 去掉"作为一个AI/作为一个大模型"等泄露身份的表述
    (r"作为一个(?:AI|大语言模型|人工智能|语言模型)[，,]?\s*", "", "去掉AI身份自述"),
    # 统一全角空格为半角
    (r"\u3000", " ", "全角空格转半角"),
    # 去掉连续 3 个以上换行（保持段落间隔但不留大片空白）
    (r"\n{3,}", "\n\n", "连续换行压缩为两个"),
    # 去掉行尾多余空格
    (r"[ \t]+\n", "\n", "去掉行尾空格"),
    # 统一引号：把英文引号在中文上下文中换成中文引号（粗略版）
    # 这条规则较保守，只在中文字符后面才转
    # (r'([\u4e00-\u9fff])"', r'\1"', "英文引号转中文引号"),
    # 上面这条之所以默认注释掉：引号成对出现，单边替换容易把成对的 "..." 改成不匹配的半对引号，
    # 反而破坏 Markdown/代码片段；要启用就先改成"成对匹配"的正则
]


def apply_replacements(text: str) -> str:
    """
    依次执行所有正则替换规则

    Args:
        text: LLM 返回的原始回答文本
    Returns:
        清洗后的文本（长度通常略变短：套话前缀和多余空白被去掉）
    说明:
        顺序执行、不做提前返回，所以每条规则都会作用在前一条的结果上——
        规则之间是有依赖的（见 REPLACEMENT_RULES 上方的顺序说明），不要随意重排
    异常:
        不抛异常：正则都是固定写死的字面量，不存在"非法正则"的运行时风险
    """
    # 1. 逐条规则做替换，后一条看到的是前一条处理过的文本
    for pattern, replacement, desc in REPLACEMENT_RULES:
        text = re.sub(pattern, replacement, text)
    # 2. 打调试日志：出问题时可对照规则条数确认规则表有没有被改少
    logger.debug(f"正则后处理完成，应用了 {len(REPLACEMENT_RULES)} 条规则")
    return text


# ==================== 引用校验 ====================

def validate_citations(answer: str, sources: List[str]) -> dict:
    """
    校验回答中引用的来源是否真实存在于检索资料中

    Args:
        answer:  LLM 生成的回答（注意：传进来的是清洗后的文本）
        sources: 检索到的资料来源列表（如 ["民法典.md", "判例.md"]）
                 调用方不传时 post_process 会补成空列表，本函数自身不接受 None
    Returns:
        {
            "valid":      bool，是否没有发现编造引用（当前实现恒为 True，见下方说明）,
            "fabricated": list，被判为编造的引用列表（当前实现恒为空）,
            "case_count": int，回答中匹配到"（20xx）xx民初x号"式案号的个数，仅统计不做判断,
        }
    现状说明（重要）:
        这版是"占位实现"：法条检查里算出 found 但既没 break 也没写入 fabricated，
        案号只做正则计数，所以 valid 永远为 True —— 刻意保持"只提示不拦截"，
        避免因为粗略匹配误判而把正常回答标成编造；要做严格校验需拿资料原文做包含匹配
    """
    fabricated = []  # 存放编造的引用

    # 检查回答中是否提到"第X条"但资料里没有对应的
    # 这是个简化版校验，真正做需要更复杂的 NLP
    # 1. 抽出所有 "《法律名》第X条" 形式的引用（条号支持中文数字和阿拉伯数字）
    law_pattern = re.findall(r"《(.+?)》第([一二三四五六七八九十百千零\d]+)条", answer)
    for law_name, article_num in law_pattern:
        # 检查资料里是否提到过这个法律名+条号
        found = False
        for src in sources:
            if law_name in src or article_num in src:  # 命中任一条件就算"资料里有提到"（很宽松，宁可漏判不可误判）
                found = True
                break
        # 这里只能做粗略检查，不做严格断言
        # 如果想要严格校验，需要把每条资料的原文拿来做包含匹配
        # 注：found 目前没有被使用，即"查到也没记、没查到也不记"，所以 fabricated 始终为空

    # 2. 统计案号数量（形如"（2023）京0101民初1234号"，全角半角括号都认）
    case_pattern = re.findall(r"[（(]\d{4}[）)]\w+民初\d+号", answer)

    # 3. 汇总返回：valid 由 fabricated 是否为空推出，保证两者语义永远一致
    return {
        "valid": len(fabricated) == 0,  # 是否全部通过
        "fabricated": fabricated,  # 编造的引用列表
        "case_count": len(case_pattern),  # 引用的案号数量
    }


# ==================== 格式规范化 ====================

def normalize_markdown(text: str) -> str:
    """
    清理 LLM 输出的 Markdown 格式（去掉多余标记）

    Args:
        text: 已经过 apply_replacements 的文本
    Returns:
        清理后的文本；只处理"整段被代码块包住"和"列表缩进"两种情况，其它 Markdown 语法原样保留
    说明:
        模型常把整段回答包在 ```markdown ... ``` 里，直接用前端 Markdown 渲染会显示成一大块代码，
        所以要脱掉这层壳；这里用 ^...$ + DOTALL 精确匹配"整段被包住"，避免误删正文里真正的代码块
    """
    # 去掉代码块包裹（如果整段都被包了）
    text = re.sub(r"^```(?:markdown)?\n(.*?)\n```$", r"\1", text, flags=re.DOTALL)  # \1 只保留代码块内部内容
    # 去掉多余的列表缩进
    text = re.sub(r"^(\s*)[-*]\s+", "- ", text, flags=re.MULTILINE)  # MULTILINE 让 ^ 匹配每一行开头；统一成 "- " 避免层级错乱
    return text


# ==================== 统一后处理入口 ====================

def post_process(answer: str, sources: List[str] = None) -> dict:
    """
    对 LLM 回答做全套后处理

    Args:
        answer:  LLM 生成的原始回答（流式场景下是各 chunk 拼接后的完整文本）
        sources: 检索到的资料来源列表（用于引用校验）；
                 允许传 None（此时内部补成空列表），方便"没有检索结果"的闲聊场景直接调用
    Returns:
        {
            "answer":     清洗后的回答（已做正则替换 + Markdown 清理，可直接回给前端）,
            "citations":  引用校验结果，结构见 validate_citations（含 valid/fabricated/case_count）,
        }
    说明:
        顺序固定为"先清洗、再校验"：校验要在清洗后的文本上做，
        否则被清掉的前缀（如"根据参考资料，"）会干扰正则匹配，导致统计结果偏差
    """
    text = apply_replacements(answer)  # 正则替换（去套话、统一空白）
    text = normalize_markdown(text)  # Markdown 清理（去代码块外壳、规整列表）
    citations = validate_citations(text, sources or [])  # 引用校验（None 兜底为空列表，保证函数签名只处理 List）
    logger.info(f"后处理完成：引用校验={'通过' if citations['valid'] else '有编造'}")  # info 级：引用异常是业务上要关注的问题
    return {
        "answer": text,
        "citations": citations,
    }
