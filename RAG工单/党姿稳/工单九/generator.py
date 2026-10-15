# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：上下文组装与答案生成（工单九『生成层面』优化落点），并支持中英文双语问答

import re                                                       # 导入正则用于语言检测
import config                                                   # 导入本工单配置模块
import llm                                                      # 导入 LLM 封装
from timing import LOGGER                                       # 导入统一日志对象

_ZH_RE = re.compile(r"[一-鿿]")                         # 中日韩统一表意文字范围（用于判断中文）


def detect_lang(text):                                          # 定义语言检测函数
    """含中文字符则判定为中文（zh），否则判定为英文（en）。"""
    return "zh" if _ZH_RE.search(text or "") else "en"           # 返回语言代码


# ---------------------------------------------------------------------------
# 一、上下文组装：优化前（平铺拼接） vs 优化后（结构化 + 关系三元组）
# ---------------------------------------------------------------------------
def build_context_baseline(chunks):                             # 定义基线上下文组装函数
    """优化前：把所有块正文直接拼接，无来源标注、无结构化分组。"""
    text = "\n".join(c["text"] for c in chunks)                 # 顺序拼接全部块文本
    return text[:config.CONTEXT_MAX_CHARS]                      # 截断到最大长度后返回


def build_context_optimized(chunks, triples=None):              # 定义优化后上下文组装函数
    """优化后：按块编号与来源结构化组织，并附上知识图谱关系三元组。"""
    parts = []                                                  # 初始化上下文片段列表
    for i, c in enumerate(chunks, 1):                           # 遍历文本块并编号
        parts.append("[资料%d]（来源：%s 第%d页；块ID：%s）\n%s" %  # 结构化标注来源与页码便于引用
                     (i, c["source"], c["page"], c["id"], c["text"]))  # 填入定位信息与正文
    if triples:                                                 # 若存在图谱关系证据
        parts.append("[知识图谱关系]\n" + "\n".join("- " + t for t in triples))  # 追加关系三元组分组
    text = "\n\n".join(parts)                                   # 用空行分隔各片段
    return text[:config.CONTEXT_MAX_CHARS]                      # 截断到最大长度后返回


# ---------------------------------------------------------------------------
# 二、提示词：优化前（无约束） vs 优化后（强约束 + 引用 + 双语）
# ---------------------------------------------------------------------------
_BASE_SYSTEM_ZH = "你是一个金融问答助手，请根据资料回答问题。"      # 基线中文系统提示

_BASE_USER_ZH = """资料：
%s

问题：%s

请回答："""                                                     # 基线中文用户提示（无引用与拒答约束）

# 经验教训：早期版本把规则写成「资料不足以回答时必须说明未提及」，模型会把
# 「资料中没有逐字出现该措辞」误当成「资料不足」，从而对绝大多数问题退化为拒答，
# 答案相关性直接掉到 0。因此第 3 条改为「只要资料含相关信息就必须作答」，
# 只把真正的完全缺失才允许说明未提及。第 6 条用于处理验收问题中的匿名代称。
_OPT_SYSTEM_ZH = """你是严谨的金融文档问答专家。回答时必须遵守：
1. 只能依据【参考资料】和【知识图谱关系】作答，不得使用资料之外的知识，不得编造数字。
2. 每个关键结论后必须用【资料N】标注其来源编号；若结论来自知识图谱关系，标注【图谱】。
3. 只要资料中含有与问题相关的信息，就必须据此给出明确、完整的答案；资料未逐字出现问题中的措辞，
   但可由资料内容归纳得出时，应当照常作答，禁止因此拒答。
4. 只有当资料确实完全不含任何相关信息时，才允许说明"资料中未提及"；只要资料含部分相关信息，
   就必须直接作答，禁止以"资料中未提及"开头再转折作答（这类表述会被视为拒答）。
5. 涉及数字、比例、金额时，必须原样引用，不得改写或换算。
6. 问题中的"xx银行""某银行"等是待识别的匿名代称：请依据资料内容判断其指向的具体银行，
   并在答案开头直接点明该银行名称，例如"xx银行即招商银行"；若问题中已写明真实机构名，
   则无需指认、直接作答，禁止输出"平安银行即平安银行"这类重复表述。
7. 每一条论断都必须能在资料原文中找到直接依据，禁止在资料之外补充分析、评价或因果推断
   （例如「这表明其具备逆周期调节特征」这类资料中没有的结论一律不要写）。
8. 回答使用中文，结构清晰，分点作答，语言简洁；不要罗列与问题无关的资料。"""  # 优化后中文系统提示（强约束）

_OPT_SYSTEM_EN = """You are a rigorous financial document QA expert. You MUST obey:
1. Answer only from the provided [References] and [Knowledge Graph Relations]; never fabricate figures.
2. Cite the source of every key claim as [Ref N], or [KG] if it comes from graph relations.
3. If the references are insufficient, explicitly state "not mentioned in the references".
4. Quote numbers, ratios and amounts verbatim without rewriting or converting them.
5. Answer in English, concise and well structured."""            # 优化后英文系统提示

_OPT_USER = """【参考资料】
%s

【问题】
%s

请依据上述资料作答，并逐条标注引用来源："""                        # 优化后用户提示（中英文通用结构）


def build_prompt(question, context, optimized=True, lang=None):  # 定义提示词构建函数
    """返回 (system, user, lang) 三元组；optimized=False 时使用基线提示词。"""
    lang = lang or detect_lang(question)                        # 未指定语言时自动检测
    if optimized:                                               # 优化后分支
        system = _OPT_SYSTEM_ZH if lang == "zh" else _OPT_SYSTEM_EN  # 按语言选择系统提示
        user = _OPT_USER % (context, question)                  # 填充结构化上下文与问题
    else:                                                       # 优化前基线分支
        system = _BASE_SYSTEM_ZH                                # 使用基线系统提示
        user = _BASE_USER_ZH % (context, question)              # 使用基线用户提示
    return system, user, lang                                   # 返回提示词与语言标记


# ---------------------------------------------------------------------------
# 三、生成入口
# ---------------------------------------------------------------------------
def generate(question, chunks, triples=None, optimized=True, lang=None):  # 定义答案生成函数
    """组装上下文并调用 LLM 生成答案，返回 (答案文本, 上下文文本, 语言)。"""
    if optimized:                                               # 优化后使用结构化上下文
        context = build_context_optimized(chunks, triples)      # 含来源标注与图谱关系
    else:                                                       # 基线使用平铺上下文
        context = build_context_baseline(chunks)                # 仅拼接正文
    system, user, lang = build_prompt(question, context, optimized, lang)  # 构建提示词
    answer = llm.chat(system, user, temperature=0.2,             # 调用 LLM（低温保证事实稳定）
                      max_tokens=1200)                          # 限制输出长度控制延迟
    if not answer:                                              # 调用失败时给出降级提示（容错）
        answer = "抱歉，生成服务暂时不可用，请稍后重试。" if lang == "zh" else \
                 "Sorry, the generation service is temporarily unavailable."  # 语言自适应降级文案
        LOGGER.warning("生成失败，已返回降级答案")                # 记录告警
    return answer, context, lang                                # 返回结果三元组
