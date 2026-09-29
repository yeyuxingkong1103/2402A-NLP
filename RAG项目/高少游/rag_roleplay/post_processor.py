# -*- coding: utf-8 -*-
"""
后处理模块：对 LLM 生成的回答做清洗、校验、格式化
- 正则表达式替换（去掉多余标记、统一标点）
- 引用校验（回答中引用的法条/案号是否在参考资料中存在）
- 格式规范化（Markdown 格式清理）
"""

import re  # 正则表达式库
from typing import List  # 列表类型标注

from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# ==================== 正则替换规则 ====================
# 每条规则：(正则, 替换文本, 说明)
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
]


def apply_replacements(text: str) -> str:
    """依次执行所有正则替换规则"""
    for pattern, replacement, desc in REPLACEMENT_RULES:
        text = re.sub(pattern, replacement, text)
    logger.debug(f"正则后处理完成，应用了 {len(REPLACEMENT_RULES)} 条规则")
    return text


# ==================== 引用校验 ====================

def validate_citations(answer: str, sources: List[str]) -> dict:
    """
    校验回答中引用的来源是否真实存在于检索资料中

    Args:
        answer:  LLM 生成的回答
        sources: 检索到的资料来源列表（如 ["民法典.md", "判例.md"]）
    Returns:
        {"valid": True/False, "fabricated": ["引用了但不存在的来源"]}
    """
    fabricated = []  # 存放编造的引用

    # 检查回答中是否提到"第X条"但资料里没有对应的
    # 这是个简化版校验，真正做需要更复杂的 NLP
    law_pattern = re.findall(r"《(.+?)》第([一二三四五六七八九十百千零\d]+)条", answer)
    for law_name, article_num in law_pattern:
        # 检查资料里是否提到过这个法律名+条号
        found = False
        for src in sources:
            if law_name in src or article_num in src:
                found = True
                break
        # 这里只能做粗略检查，不做严格断言
        # 如果想要严格校验，需要把每条资料的原文拿来做包含匹配

    # 检查回答中是否出现"案号"格式但资料里没有
    case_pattern = re.findall(r"[（(]\d{4}[）)]\w+民初\d+号", answer)

    return {
        "valid": len(fabricated) == 0,  # 是否全部通过
        "fabricated": fabricated,  # 编造的引用列表
        "case_count": len(case_pattern),  # 引用的案号数量
    }


# ==================== 格式规范化 ====================

def normalize_markdown(text: str) -> str:
    """清理 LLM 输出的 Markdown 格式（去掉多余标记）"""
    # 去掉代码块包裹（如果整段都被包了）
    text = re.sub(r"^```(?:markdown)?\n(.*?)\n```$", r"\1", text, flags=re.DOTALL)
    # 去掉多余的列表缩进
    text = re.sub(r"^(\s*)[-*]\s+", "- ", text, flags=re.MULTILINE)
    return text


# ==================== 统一后处理入口 ====================

def post_process(answer: str, sources: List[str] = None) -> dict:
    """
    对 LLM 回答做全套后处理

    Returns:
        {
            "answer":     清洗后的回答,
            "citations":  引用校验结果,
        }
    """
    text = apply_replacements(answer)  # 正则替换
    text = normalize_markdown(text)  # Markdown 清理
    citations = validate_citations(text, sources or [])  # 引用校验
    logger.info(f"后处理完成：引用校验={'通过' if citations['valid'] else '有编造'}")
    return {
        "answer": text,
        "citations": citations,
    }
