# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：自实现 RAGAS 风格评估框架（上下文精度、上下文召回、忠实度、答案相关性）
# 说明：工单允许『RAGAS 或者其他评估框架』，这里用 LLM 判定实现同等语义的指标，无需额外依赖

import re                                                       # 导入正则用于句子切分
import config                                                   # 导入本工单配置模块
import llm                                                      # 导入 LLM 封装（带磁盘缓存）
from timing import LOGGER                                       # 导入统一日志对象

_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*")              # 中英文句末标点后的切分点


def split_sentences(text):                                      # 定义参考答案句子切分函数
    """把参考答案切分为句子列表，过滤过短片段，供上下文召回计算使用。"""
    parts = [s.strip() for s in _SENT_SPLIT.split(text or "")]   # 按句末标点切分并去空白
    parts = [s for s in parts if len(s) >= 6]                   # 过滤过短片段（无信息量）
    return parts if parts else ([text.strip()] if text else [])  # 空结果时回退为整段


# ---------------------------------------------------------------------------
# 一、上下文精度（Context Precision）：相关上下文是否排在前面
# ---------------------------------------------------------------------------
def context_precision(question, contexts, reference):           # 定义上下文精度计算函数
    """RAGAS 定义：对每个「有用」的上下文，计算其位置精度，再取平均。"""
    if not contexts:                                            # 无上下文时精度记 0
        return 0.0, []                                          # 返回 0 与空判定明细
    verdicts = _judge_contexts(question, contexts, reference)    # 调用 LLM 批量判定每段上下文是否有用
    relevant_pos = [i for i, v in enumerate(verdicts) if v == 1]  # 收集有用上下文的位置（从 0 开始）
    if not relevant_pos:                                        # 没有任何有用上下文
        return 0.0, verdicts                                    # 精度为 0
    value = 0.0                                                 # 累加精度
    for i in relevant_pos:                                      # 遍历每个有用上下文
        hit = sum(1 for j in relevant_pos if j <= i)            # 该位置及之前命中的有用上下文数
        value += hit / (i + 1)                                  # 位置精度 = 命中数 / 当前排名
    return value / len(relevant_pos), verdicts                  # 除以有用上下文总数得到指标


def _judge_contexts(question, contexts, reference):             # 定义上下文有用性判定函数
    """一次请求判定所有上下文是否有助于回答该问题，返回 0/1 列表（容错返回 0）。"""
    lines = []                                                  # 初始化待判定上下文列表
    for i, c in enumerate(contexts, 1):                         # 遍历上下文并编号
        snippet = (c if isinstance(c, str) else c.get("text", ""))[:900]  # 取前 900 字（覆盖 500 字分块的完整内容）
        lines.append("[%d] %s" % (i, snippet.replace("\n", " ")))  # 渲染为『[序号] 内容』
    system = "你是检索质量评估专家。判断每段资料对回答给定问题、并覆盖参考答案要点是否有帮助。"  # 系统提示
    user = """问题：%s

参考答案：%s

资料列表：
%s

请对每段资料给出判断：有助于回答或覆盖参考答案要点则为 1，完全无关则为 0。
只输出 JSON：{"verdicts": [0 或 1, ...]}，长度必须等于 %d。""" % (
        question, reference, "\n".join(lines), len(contexts))    # 拼装评估请求
    obj = llm.chat_json(system, user, temperature=0.0, max_tokens=600)  # 调用 LLM 并解析 JSON
    arr = (obj or {}).get("verdicts") if isinstance(obj, dict) else None  # 安全取出判定数组
    if not isinstance(arr, list) or len(arr) != len(contexts):  # 判定结果缺失或长度不符
        return [0] * len(contexts)                              # 全部记 0（保守评估，避免虚高）
    return [1 if str(v).strip() in ("1", "true", "True", "是") else 0 for v in arr]  # 归一化为 0/1


# ---------------------------------------------------------------------------
# 二、上下文召回（Context Recall）：参考答案要点被上下文覆盖的比例
# ---------------------------------------------------------------------------
def context_recall(question, contexts, reference):              # 定义上下文召回计算函数
    """RAGAS 定义：参考答案可被检索上下文支持的比例。"""
    sents = split_sentences(reference)                          # 切分参考答案为要点句
    if not sents:                                               # 参考答案为空
        return 0.0, []                                          # 返回 0 与空明细
    # 第二轮保真修复：此前每条上下文只截取 800 字，而综合分析题的参考要点分散在
    # 块的中后段，截断导致评估器"看不见"实际已检回的证据，召回被系统性低估；
    # 这里放宽到 1500 字（约 3 个分块长度），并控制总长避免提示词超限
    joined = "\n---\n".join((c if isinstance(c, str) else c.get("text", ""))[:1500]  # 拼接上下文
                            for c in contexts)                  # 每条最多 1500 字
    if not joined.strip():                                      # 上下文为空
        return 0.0, [0] * len(sents)                            # 召回为 0
    # 第二轮保真修复：验收问题与参考答案中的『xx银行/yy银行/xx人寿』是脱敏占位符，
    # 对应资料中的真实机构名；且 RAGAS 召回的官方定义允许"可从资料推导"的要点。
    # 在提示词中说明这两点，避免判定模型因字面不一致而漏判
    system = ("你是检索质量评估专家。判断参考答案中的每个要点句能否被给定资料支持。"
              "注意：参考答案中的『xx银行/yy银行/xx人寿』等为脱敏占位符，对应资料中的真实机构名；"
              "要点句能从资料中直接得出或合理推断出都算支持。")     # 系统提示
    lines = "\n".join("[%d] %s" % (i + 1, s) for i, s in enumerate(sents))  # 渲染要点句列表
    user = """问题：%s

资料：
%s

参考答案要点句：
%s

请判断每个要点句是否能从资料中得到支持（能则 1，不能则 0）。
只输出 JSON：{"verdicts": [0 或 1, ...]}，长度必须等于 %d。""" % (
        question, joined, lines, len(sents))                    # 拼装评估请求
    obj = llm.chat_json(system, user, temperature=0.0, max_tokens=600)  # 调用 LLM
    arr = (obj or {}).get("verdicts") if isinstance(obj, dict) else None  # 安全取出判定数组
    if not isinstance(arr, list) or len(arr) != len(sents):     # 判定缺失或长度不符
        return 0.0, [0] * len(sents)                            # 保守返回 0
    flags = [1 if str(v).strip() in ("1", "true", "True", "是") else 0 for v in arr]  # 归一化
    return sum(flags) / float(len(flags)), flags                # 覆盖比例即召回率


# ---------------------------------------------------------------------------
# 三、忠实度（Faithfulness）与答案相关性（Answer Relevancy）
# ---------------------------------------------------------------------------
def _judge_ratio(system, user, n_expected):                     # 定义通用比例判定函数
    """请求 LLM 输出 0/1 判定数组并计算比例；异常时返回 0。"""
    obj = llm.chat_json(system, user, temperature=0.0, max_tokens=600)  # 调用 LLM
    arr = (obj or {}).get("verdicts") if isinstance(obj, dict) else None  # 安全取数组
    if not isinstance(arr, list) or not arr:                    # 判定无效
        return 0.0                                              # 返回 0
    flags = [1 if str(v).strip() in ("1", "true", "True", "是") else 0 for v in arr]  # 归一化
    return sum(flags) / float(len(flags))                       # 返回支持比例


def faithfulness(answer, contexts):                             # 定义忠实度计算函数
    """答案中的论断能被上下文支持的比例，衡量是否产生幻觉。"""
    claims = split_sentences(answer)                            # 把答案切分为论断句
    if not claims:                                              # 无论断
        return 0.0                                              # 返回 0
    joined = "\n---\n".join((c if isinstance(c, str) else c.get("text", ""))[:800]  # 拼接上下文
                            for c in contexts)                  # 每条最多 800 字
    system = "你是事实核查专家。判断答案中的每个论断句是否被给定资料支持。"  # 系统提示
    lines = "\n".join("[%d] %s" % (i + 1, s) for i, s in enumerate(claims))  # 渲染论断列表
    user = """资料：
%s

答案论断：
%s

请判断每个论断是否被资料支持（支持为 1，不支持或无法验证为 0）。
只输出 JSON：{"verdicts": [0 或 1, ...]}。""" % (joined, lines)  # 拼装请求
    return _judge_ratio(system, user, len(claims))              # 返回支持比例
