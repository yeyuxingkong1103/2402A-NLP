# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：组装提示词、调用 DeepSeek 生成答案，并解析出选项字母用于评估。

import re                                                        # 解析答案字母
import config                                                    # 配置与密钥
from logger import get_logger                                    # 日志

logger = get_logger("rag14.generator")

_CLIENT = None                                                   # OpenAI 兼容客户端单例

SYSTEM_PROMPT = (
    "你是一名工业专利文档问答助手。你只能依据用户提供的【检索资料】作答，"
    "不得引入资料之外的知识，也不得凭常识猜测。\n"
    "资料中标记为【图纸】的片段是对说明书附图的文字描述，涉及部件相对位置、"
    "气流方向、尺寸标注等问题时，必须结合该片段进行空间与顺序推理。\n"
    "判断气流/物料的流经顺序时，要区分'气流所处的空间或容纳它的壳体'与"
    "'气流实际穿过并对其起作用的部件'：题目问'首先经过/紧接着经过'时，"
    "答案应取后者，即气流真正穿过的那个部件，而不是包裹它的外壳或锥体。\n"
    "若资料给出若干部件相对于某个基准面（如台阶形截止面）的距离（如 x1、x2、x3），"
    "而题目恰好问气流经过这些部件的先后顺序，则先判断该基准面处于气流的上游还是下游："
    "气流从管状入口流向出口，距离上游入口越近的部件越先被经过，"
    "即距离下游基准面越远的那个部件反而最先被经过。\n"
    "答题步骤：\n"
    "1) 先用不超过 80 字说明你的判断依据，并引用片段编号；\n"
    "2) 然后另起一行，严格输出格式：答案：X（X 为唯一选项字母，如 A/B/C/D）。"
)


def get_client():
    """惰性创建 DeepSeek 客户端。"""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI                                # 延迟导入
        _CLIENT = OpenAI(api_key=config.DEEPSEEK_API_KEY,
                         base_url=config.DEEPSEEK_BASE_URL)
    return _CLIENT


def build_prompt(question, options, context):
    """组装用户提示词：检索资料 + 问题 + 选项。"""
    opts = "\n".join(options) if options else "（本题为开放问答，无选项）"
    return (f"【检索资料】\n{context}\n\n"
            f"【问题】{question}\n\n"
            f"【选项】\n{opts}\n\n"
            f"请按系统提示的要求作答。")


def call_llm(prompt, system=None, temperature=0.0):
    """调用 DeepSeek 对话接口，返回回复文本。"""
    client = get_client()                                        # 客户端
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,                             # 模型
        messages=[{"role": "system", "content": system or SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],          # 消息
        temperature=temperature,                                 # 0 度保证可复现
        timeout=config.LLM_TIMEOUT)
    return resp.choices[0].message.content.strip()               # 回复正文


_LETTER_RE = re.compile(r"答案\s*[:：]\s*([A-Da-d])")             # 标准答案行


def parse_answer(text):
    """从模型回复中解析选项字母；解析不到返回空串。"""
    m = _LETTER_RE.findall(text)                                 # 所有"答案：X"
    if m:                                                        # 取最后一个（模型可能先草稿后定稿）
        return m[-1].upper()
    m2 = re.findall(r"\b([A-D])\b", text)                         # 退化为找单个大写字母
    return m2[-1] if m2 else ""


def answer_question(question, options, picked, assemble_context):
    """完整问答：检索块 → 上下文 → 提示词 → LLM → (答案字母, 原始回复, 上下文)。"""
    context = assemble_context(picked)                           # 组装上下文
    prompt = build_prompt(question, options, context)            # 组装提示词
    raw = call_llm(prompt)                                       # 调用模型
    letter = parse_answer(raw)                                   # 解析答案
    logger.info("LLM 回答完成，解析答案=%s", letter or "(未解析出)")
    return letter, raw, context
