# 工单编号：人工智能NLP-RAG-功能测试及评估
"""RAG 评估裁判（LLM-as-Judge）

两项核心指标：
  grounded  答案的事实性内容能否在检索到的上下文中找到依据（衡量有没有编造）
  relevant  答案是否正面回答了问题

工单7 里这个模块只承担**裁判**职责：批量评估的入口是 测试/eval_ccf.py
（10 道题 + 确定性检索指标 + RAG/纯 LLM 对照），本文件原本那个
「传一个 PDF 路径跑一遍」的命令行入口已被它取代，故移除 ——
它默认拿招股说明书 PDF 配工单7 的年报题目，留着是个坑。
"""
import json
import re

from rag_engine import chat

JUDGE_PROMPT = """你是 RAG 评估裁判。请依据给定上下文，对回答打分。

【问题】
{question}

【检索到的上下文】
{context}

【待评估回答】
{answer}

只输出一行 JSON，不要任何多余文字：
{{"grounded": 0, "relevant": 0, "reason": "一句话理由"}}

grounded 取 1 表示回答中的事实性内容都能在上下文中找到依据，0 表示存在上下文里没有的内容。
relevant 取 1 表示回答正面回答了问题，0 表示答非所问。
上下文为空时，grounded 一律取 0。"""


def _judge(question, context, answer):
    # max_tokens 不能给小：裁判同样走推理模型，额度被 reasoning 吃光就返回空串。
    raw = chat(JUDGE_PROMPT.format(question=question, context=context or "（无）",
                                   answer=answer), temperature=0.0, max_tokens=2048)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"grounded": 0, "relevant": 0, "reason": "裁判输出无法解析"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"grounded": 0, "relevant": 0, "reason": "裁判输出无法解析"}
    return {
        "grounded": int(bool(data.get("grounded", 0))),
        "relevant": int(bool(data.get("relevant", 0))),
        "reason": str(data.get("reason", ""))[:120],
    }
