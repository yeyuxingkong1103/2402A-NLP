# -*- coding: utf-8 -*-
"""后处理校验：回答中的数值与知识块对照，无依据的数值给出提醒（规则版）。"""
import re
# 解析：正则模块（提取数值）

# 数值 + 常见单位（可选）
_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:mmHg|mg|kg|g|ml|分钟|小时|次|周|天|%|级)?")
# 解析：数值正则——整数或小数，后面可选跟常见医疗/生活单位（如 140mmHg、5克、150分钟）


def verify_answer(answer: str, contexts: list[str]) -> list[str]:
    """提取回答中的数值，与检索上下文中的数值对照；无依据的数值返回提醒。

    contexts 为空时跳过校验（无 RAG 场景不提示）。
    """
    if not contexts:
        # 解析：没有知识块（无 RAG 场景）
        return []
        # 解析：跳过校验直接返回空提醒列表
    answer_numbers = set(_NUMBER.findall(answer))
    # 解析：提取回答中出现的全部数值（去重）
    context_numbers = set()
    # 解析：收集知识块中的全部数值
    for ctx in contexts:
        # 解析：遍历每个知识块
        context_numbers.update(_NUMBER.findall(ctx))
        # 解析：提取该块的数值并入集合

    missing = sorted(answer_numbers - context_numbers)
    # 解析：差集——回答中有而知识库没有的数值，排序保证输出稳定
    warnings = []
    # 解析：提醒列表
    for number in missing:
        # 解析：逐个无依据数值
        warnings.append(f"回答中的数值「{number}」未在知识库中找到依据，请核对")
        # 解析：生成一条提醒（接口 warnings 字段 / SSE warnings 事件返回）
    return warnings
    # 解析：返回全部提醒
