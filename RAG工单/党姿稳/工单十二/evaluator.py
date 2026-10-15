# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：自实现 RAGAS 风格评估指标（上下文精度、上下文召回、忠实度、答案相关性）
# 说明：工单允许「RAGAS 或其他评估框架」，这里用 LLM 判定实现同等语义的四个指标，不引入额外依赖

import re                                                       # 导入正则用于句子切分
import llm                                                      # 导入 LLM 封装（带磁盘缓存）
from logger import LOGGER                                       # 导入统一日志对象

_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*")              # 中英文句末标点后的切分点
_YES = ("1", "true", "True", "是")                              # 判定为「是」的取值集合


def split_sentences(text):                                      # 定义句子切分函数
    """把长文本切分为句子列表并过滤过短片段；无有效切分时回退为整段。"""
    parts = [s.strip() for s in _SENT_SPLIT.split(text or "")]   # 按句末标点切分并去空白
    parts = [s for s in parts if len(s) >= 6]                   # 过滤过短片段（无信息量）
    return parts if parts else ([text.strip()] if text else [])  # 空结果时回退为整段


def _texts_of(contexts):                                        # 定义上下文取文本函数
    """统一取出上下文的正文（支持字符串与检索结果字典两种形式）。"""
    return [c if isinstance(c, str) else c.get("text", "") for c in contexts]  # 逐条取出正文


def _verdicts(system, user, n_expected, max_tokens=800):        # 定义通用判定函数
    """请求 LLM 输出 0/1 判定数组；解析失败或长度不符时全部记 0（保守评估，避免虚高）。"""
    obj = llm.chat_json(system, user, max_tokens=max_tokens)    # 调用 LLM 并解析 JSON
    arr = (obj or {}).get("verdicts") if isinstance(obj, dict) else None  # 安全取出判定数组
    if not isinstance(arr, list) or len(arr) != n_expected:     # 判定缺失或长度不符
        return [0] * n_expected                                 # 保守返回全 0
    return [1 if str(v).strip() in _YES else 0 for v in arr]    # 归一化为 0/1 列表


def context_precision(question, contexts, reference):           # 定义上下文精度计算函数
    """RAGAS 定义：对每段有用上下文计算位置精度（命中数/当前排名）再取平均。"""
    if not contexts:                                            # 无上下文时精度记 0
        return 0.0, []                                          # 返回 0 与空明细
    lines = []                                                  # 待判定上下文列表
    for i, c in enumerate(_texts_of(contexts), 1):              # 遍历上下文并编号
        lines.append("[%d] %s" % (i, c[:600].replace("\n", " ")))  # 渲染为「[序号] 内容」，单条截断 600 字
    system = "你是检索质量评估专家。判断每段资料对回答给定问题、并覆盖参考答案要点是否有帮助。"  # 系统提示
    user = """问题：%s

参考答案：%s

资料列表：
%s

请对每段资料给出判断：有助于回答或覆盖参考答案要点则为 1，完全无关则为 0。
只输出 JSON：{"verdicts": [0 或 1, ...]}，长度必须等于 %d。""" % (
        question, reference, "\n".join(lines), len(contexts))   # 拼装评估请求
    verdicts = _verdicts(system, user, len(contexts))           # 获取逐段判定
    pos = [i for i, v in enumerate(verdicts) if v == 1]         # 有用上下文的位置（从 0 开始）
    if not pos:                                                 # 没有任何有用上下文
        return 0.0, verdicts                                    # 精度为 0
    value = 0.0                                                 # 累加精度
    for i in pos:                                               # 遍历每个有用上下文
        value += sum(1 for j in pos if j <= i) / float(i + 1)   # 命中数 / 当前排名
    return value / len(pos), verdicts                           # 除以有用上下文总数得到指标


def context_recall(question, contexts, reference):              # 定义上下文召回计算函数
    """RAGAS 定义：参考答案的要点句能被检索上下文支持的比例。"""
    sents = split_sentences(reference)                          # 切分参考答案为要点句
    if not sents:                                               # 参考答案为空
        return 0.0, []                                          # 返回 0 与空明细
    joined = "\n---\n".join(t[:800] for t in _texts_of(contexts))  # 拼接上下文，每条截断 800 字
    if not joined.strip():                                      # 上下文为空
        return 0.0, [0] * len(sents)                            # 召回为 0
    system = "你是检索质量评估专家。判断参考答案中的每个要点句能否被给定资料支持或推断出来。"  # 系统提示
    lines = "\n".join("[%d] %s" % (i + 1, s) for i, s in enumerate(sents))  # 渲染要点句列表
    user = """问题：%s

资料：
%s

参考答案要点句：
%s

请判断每个要点句是否能从资料中得到支持（能则 1，不能则 0）。
只输出 JSON：{"verdicts": [0 或 1, ...]}，长度必须等于 %d。""" % (
        question, joined, lines, len(sents))                    # 拼装评估请求
    flags = _verdicts(system, user, len(sents))                 # 获取逐句判定
    return sum(flags) / float(len(flags)), flags                # 覆盖比例即召回率


def faithfulness(answer, contexts):                             # 定义忠实度计算函数
    """答案中的论断句能被检索上下文支持的比例，衡量是否产生幻觉。"""
    claims = split_sentences(answer)                            # 把答案切分为论断句
    if not claims:                                              # 无论断
        return 0.0                                              # 返回 0
    joined = "\n---\n".join(t[:800] for t in _texts_of(contexts))  # 拼接上下文
    system = "你是事实核查专家。判断答案中的每个论断句是否被给定资料支持。"  # 系统提示
    lines = "\n".join("[%d] %s" % (i + 1, s) for i, s in enumerate(claims))  # 渲染论断列表
    user = """资料：
%s

答案论断：
%s

请判断每个论断是否被资料支持（支持为 1，不支持或无法验证为 0）。
只输出 JSON：{"verdicts": [0 或 1, ...]}。""" % (joined, lines)  # 拼装请求
    flags = _verdicts(system, user, len(claims))                # 获取逐句判定
    return sum(flags) / float(len(flags))                       # 返回支持比例


def answer_relevancy(question, answer):                         # 定义答案相关性计算函数
    """答案是否切题：把答案拆为要点句，逐句判断是否直接回应了问题所问。"""
    sents = split_sentences(answer)                             # 切分答案为要点句
    if not sents:                                               # 答案为空
        return 0.0                                              # 返回 0
    system = "你是答案质量评估专家。判断答案中的每个要点是否直接回应了用户问题所问的内容。"  # 系统提示
    lines = "\n".join("[%d] %s" % (i + 1, s) for i, s in enumerate(sents))  # 渲染要点列表
    user = """问题：%s

答案要点：
%s

请判断每个要点是否直接回应了问题（回应则 1，答非所问、跑题或纯免责声明则 0）。
只输出 JSON：{"verdicts": [0 或 1, ...]}。""" % (question, lines)  # 拼装请求
    flags = _verdicts(system, user, len(sents))                 # 获取逐句判定
    return sum(flags) / float(len(flags))                       # 返回切题比例


def evaluate_one(question, answer, contexts, reference):        # 定义单题四指标计算函数
    """对单个问题计算四个 RAGAS 风格指标，返回指标字典与判定明细。"""
    cp, cp_detail = context_precision(question, contexts, reference)  # 上下文精度
    cr, cr_detail = context_recall(question, contexts, reference)     # 上下文召回
    fa = faithfulness(answer, contexts)                         # 忠实度
    ar = answer_relevancy(question, answer)                     # 答案相关性
    LOGGER.info("评估：精度 %.4f 召回 %.4f 忠实 %.4f 相关性 %.4f", cp, cr, fa, ar)  # 输出单题指标
    return {"context_precision": round(cp, 4), "context_recall": round(cr, 4),  # 四个指标
            "faithfulness": round(fa, 4), "answer_relevancy": round(ar, 4),  # 保留四位小数
            "detail": {"precision": cp_detail, "recall": cr_detail}}  # 判定明细便于复核


METRIC_KEYS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]  # 指标顺序


def summarize(per_question):                                    # 定义汇总函数
    """把逐题指标汇总为平均值。"""
    out = {}                                                    # 初始化汇总字典
    for k in METRIC_KEYS:                                       # 遍历四个指标
        vals = [q["metrics"][k] for q in per_question]          # 收集逐题取值
        out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0  # 求平均
    return out                                                  # 返回汇总结果
