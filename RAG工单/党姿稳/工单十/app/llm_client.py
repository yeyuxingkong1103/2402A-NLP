# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：封装 DeepSeek 大模型调用，提供超时、重试与失败降级能力

import time                                                     # 导入 time 用于重试退避
import config                                                   # 导入本工单配置模块
from logging_setup import LOGGER                                # 导入统一日志对象

_CLIENT = None                                                  # 客户端单例

SYSTEM_PROMPT = """你是严谨的金融文档问答助手。回答必须遵守：
1. 只能依据【参考资料】作答，不得使用资料之外的知识，不得编造数字。
2. 每个关键结论后用【资料N】标注来源编号。
3. 资料不足以回答时，明确说明"资料中未提及"，禁止猜测。
4. 涉及数字、比例、金额时原样引用，不得改写或换算。
5. 使用中文作答，结构清晰，语言简洁。"""                          # 服务统一的系统提示词


def get_client():                                               # 定义客户端获取函数
    """懒加载 OpenAI 兼容客户端（DeepSeek 使用该协议）。"""
    global _CLIENT                                              # 声明模块级变量
    if _CLIENT is None:                                         # 首次调用时创建
        from openai import OpenAI                               # 延迟导入以加快服务启动
        _CLIENT = OpenAI(api_key=config.DEEPSEEK_API_KEY,       # 传入密钥（可经环境变量注入）
                         base_url=config.DEEPSEEK_BASE_URL,     # 传入接口地址
                         timeout=config.LLM_TIMEOUT)            # 传入请求超时
    return _CLIENT                                              # 返回客户端


def build_context(chunks):                                      # 定义上下文组装函数
    """把检索到的文本块组装为带编号与来源的结构化上下文。"""
    parts = []                                                  # 片段列表
    for i, c in enumerate(chunks, 1):                           # 遍历文本块并编号
        parts.append("[资料%d]（来源：%s 第%d页）\n%s" %          # 拼装编号、来源与正文
                     (i, c["source"], c["page"], c["text"]))    # 填入具体信息
    text = "\n\n".join(parts)                                   # 用空行分隔
    return text[:config.CONTEXT_MAX_CHARS]                      # 截断到最大长度


def generate_answer(question, chunks):                          # 定义答案生成函数
    """根据检索结果生成答案；调用失败时返回降级文案。"""
    context = build_context(chunks)                             # 组装上下文
    if not context.strip():                                     # 无可用上下文
        return "资料库中未检索到与该问题相关的内容。"              # 直接返回提示
    user = "【参考资料】\n%s\n\n【问题】\n%s\n\n请依据上述资料作答，并标注引用来源：" % (
        context, question)                                      # 拼装用户提示词
    messages = [{"role": "system", "content": SYSTEM_PROMPT},   # 系统角色消息
                {"role": "user", "content": user}]              # 用户角色消息
    last_err = None                                             # 记录最后一次异常
    for attempt in range(config.LLM_MAX_RETRY):                 # 按最大次数重试
        try:                                                    # 尝试调用
            resp = get_client().chat.completions.create(        # 发起对话补全请求
                model=config.DEEPSEEK_MODEL,                    # 指定模型
                messages=messages,                              # 传入消息
                temperature=0.2,                                # 低温保证事实稳定
                max_tokens=1000)                                # 限制输出长度
            return (resp.choices[0].message.content or "").strip()  # 返回清洗后的回复
        except Exception as e:                                  # 捕获异常
            last_err = e                                        # 记录异常
            LOGGER.warning("LLM 调用失败（第 %d 次）：%r", attempt + 1, e)  # 输出重试日志
            time.sleep(1.0 * (attempt + 1))                     # 线性退避
    LOGGER.error("LLM 调用最终失败：%r", last_err)               # 记录最终失败
    return "抱歉，生成服务暂时不可用，请稍后重试。"                  # 返回降级文案
