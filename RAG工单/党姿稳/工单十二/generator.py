# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：把检索到的上下文拼装为提示词并调用 LLM 生成答案（RAG 与 LightRAG 共用生成阶段）

import config                                                   # 导入本工单配置模块
import llm                                                      # 导入 LLM 封装（带磁盘缓存）
from logger import LOGGER                                       # 导入统一日志对象

_SYSTEM = ("你是招股说明书问答助手。请严格依据提供的资料回答问题，"
           "只使用资料中出现的事实与数字，不允许补充资料之外的分析、评价或因果推断。")

_USER = """资料：
%s

问题：%s

回答要求：
1. 只依据上述资料作答，资料中没有的信息直接回答「资料中未提及」，不要凭常识补充。
2. 涉及数字、比例、金额、日期时，必须与资料原文完全一致，并注明出自哪份资料的第几页。
3. 分点作答，条理清晰，避免重复堆砌原文。
4. 如果资料之间出现矛盾，以更具体、更接近原始表格或原文的表述为准。"""


def build_context(hits, max_chars=None):                        # 定义上下文拼装函数
    """把检索结果拼装为带来源标记的上下文文本，超出长度上限时截断。"""
    max_chars = max_chars or config.CONTEXT_MAX_CHARS           # 取配置的上下文字符上限
    blocks = []                                                 # 初始化上下文片段列表
    used = 0                                                    # 已使用字符数
    for h in hits:                                              # 按检索名次遍历上下文块
        head = "【%s 第 %d 页】" % (h["source"], h["page"])        # 生成来源标注
        text = h["text"]                                        # 取出块正文
        piece = head + "\n" + text                              # 拼接为一段
        if used + len(piece) > max_chars:                       # 超过上限时停止追加
            break                                               # 跳出循环
        blocks.append(piece)                                    # 追加片段
        used += len(piece)                                      # 累计字符数
    return "\n\n".join(blocks)                                  # 用空行分隔返回


def generate(question, hits, temperature=0.0):                  # 定义答案生成函数
    """根据检索结果生成答案；检索为空时返回明确的兜底回复（保证流程不中断）。"""
    if not hits:                                                # 无检索结果
        return "检索结果为空，无法回答该问题。"                    # 返回兜底回复
    context = build_context(hits)                               # 拼装上下文
    answer = llm.chat(_SYSTEM, _USER % (context, question),     # 调用 LLM 生成答案
                      temperature=temperature, max_tokens=1200)  # 限制输出长度控制成本
    answer = (answer or "").strip()                             # 去除首尾空白
    if not answer:                                              # 生成失败（接口异常或限流）时兜底
        LOGGER.warning("答案生成失败，使用检索首块摘要兜底")      # 输出告警
        answer = "（生成失败）检索到的最相关片段为：" + hits[0]["text"][:200]  # 用首块摘要兜底
    return answer                                               # 返回答案文本
