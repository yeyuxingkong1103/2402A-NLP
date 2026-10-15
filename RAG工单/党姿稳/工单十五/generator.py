# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：组装提示词、调用 DeepSeek 生成答案，并解析出选项字母用于评估。

import re                                                        # 解析答案字母
import config                                                    # 配置与密钥
from logger import get_logger                                    # 日志

logger = get_logger("rag15.generator")

_CLIENT = None                                                   # OpenAI 兼容客户端单例

SYSTEM_PROMPT = (
    "你是一名工业专利文档问答助手。你只能依据用户提供的【检索资料】作答，"
    "不得引入资料之外的知识，也不得凭常识猜测。\n"
    "资料中类型为【图纸】的片段是对说明书附图的文字描述，类型为【正文】的片段是说明书说明文字。\n"
    "当问题涉及图纸时，请结合以下技术图纸的描述进行分析：\n"
    "  - 部件编号指向的部件名称（如紧固机构11、保护管12、链条13、输入管14）；\n"
    "  - 各部件之间的相对位置关系（内/外、上/下、左/右）；\n"
    "  - 物料或气流的流经顺序（先经过谁、再经过谁）；\n"
    "  - 若要调整某个部件，需要操作哪个部件。\n"
    "【图纸】片段中，'说明书正文中关于图N的文字描述'是权威依据；"
    "'视觉模型自动生成的图形描述'由模型自动生成、分辨率受限，仅作辅助参考，"
    "两者冲突时一律以说明书正文的文字描述为准。\n"
    "判断物料/气流流经顺序时，要区分'物料被导入的容器'与'物料实际流经并对其起作用的部件'："
    "说明书中表述为'导入某容器'的部件只是物料进入的场所，"
    "而'通过某部件实现减速/分散/导向'的部件才是物料流经顺序中的下一环节，后者才是答案。\n"
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
