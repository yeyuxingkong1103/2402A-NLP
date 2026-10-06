# -*- coding: utf-8 -*-
# 【RAGAS离线评估器 · ragas_evaluator.py】无LLM环境下RAGAS四项指标的可复核离线近似，预留官方ragas适配
# 工单编号：人工智能NLP-RAG-功能测试及评估

"""RAGAS 四项指标的离线可运行实现。

官方 RAGAS（faithfulness / answer_relevancy / context_precision /
context_recall）依赖 LLM 对“陈述、反事实、蕴含”做判断。本机无 LLM API、
无外网模型，因此本模块给出**口径明确、可人工复核**的离线近似：以词项/实体
重合、数字一致性、答案-证据蕴含近似（答案陈述中的内容词与数字必须能在检索
上下文中找到依据）替代 LLM 判定。

⚠️ 接入 LLM 后可一键替换为官方实现：调用本模块末尾的
``evaluate_with_official_ragas`` 适配函数（需 pip install ragas 并配置
LLM/embeddings），输入输出字段已按官方 Dataset schema 对齐。

各指标定义（值域均为 0~1）：
- faithfulness（忠实度）：把答案切为若干陈述，能被检索上下文支持
  （内容词覆盖≥阈值且数字全部命中）的陈述占比；
- answer_relevancy（答案相关性）：答案对 gold 要点的覆盖——gold 内容词
  召回 + 指定实体召回 + 数字 F1 的加权和；
- context_precision（上下文精确率）：逐参考证据判断检索块是否相关，
  按排名计算平均精度 AP（排名越靠前扣分越少）；
- context_recall（上下文召回率）：参考证据被检索结果覆盖的比例，与
  gold 实体/数字被检索上下文覆盖的比例取均值。

不可回答题口径：系统正确拒答时四项指标统一记 1.0（“不产生无依据陈述”
本身即为忠实且相关的行为），拒答正确性另由“拒答标记”单列统计；
该口径在设计文档与测试报告中同步声明。
"""
import re
from typing import Dict, List, Optional, Sequence

from rag_pipeline import tokenize  # 复用流水线的 jieba 切词与停用词表

# 陈述切分：中文句读、顿号（枚举项）、逗号（复句内各小句）
_CLAIM_SPLIT_RE = re.compile(r"[。；;！？!?，、,]+")
# 数字（含千分位/小数/四位年份）。长数字模式必须优先整体匹配，
# 否则 \d{4} 会把 15000 截成 1500+0、把 20658.33 截成 2065+8.33
_NUMBER_CORE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# 陈述被上下文支持的内容词覆盖阈值
ENTAIL_TOKEN_MIN = 0.6
# 参考证据与检索块判定为“相关”的方向覆盖阈值
CONTEXT_REL_MIN = 0.72


# ---------------------------------------------------------------------------
# 一、基础工具：数字归一化、内容词集合、文本相似度
# ---------------------------------------------------------------------------

def normalized_numbers(text: str) -> set:
    """抽取文本中的数字并归一化（去年千分位，转 float 字符串规范形式）。

    年份按整数、金额/数量按小数统一比较，例如：
    “1,840.00 万股”→ ``1840.0``；“2019年”→ ``2019.0``。

    :param text: 原始文本
    :return: 归一化数字集合（字符串形式，便于与无法转浮点的串共存）
    """
    nums = set()
    for raw in _NUMBER_CORE_RE.findall(text):
        val = raw.replace(",", "")
        try:
            nums.add(str(float(val)))
        except ValueError:
            nums.add(val)
    return nums


def _content_set(text: str) -> set:
    """内容词集合（去停用词）。

    :param text: 原始文本
    :return: 词项集合
    """
    return set(tokenize(text))


def _directed_overlap(small: str, big: str) -> float:
    """方向覆盖度：small 的内容词在 big 中出现的比例。

    :param small: 较短/参考文本
    :param big: 较长/被检索文本
    :return: 0~1 覆盖比例；small 无内容词时记 1（无约束）
    """
    a, b = _content_set(small), _content_set(big)
    if not a:
        return 1.0
    return len(a & b) / len(a)


def _context_relevant(reference: str, chunk_text: str) -> bool:
    """判断单条检索块是否与单条参考证据相关。

    判据：去空白后子串包含，或参考证据方向覆盖度≥阈值。

    :param reference: 参考证据原文
    :param chunk_text: 检索块文本
    :return: 相关返回 True
    """
    ref_compact = re.sub(r"\s+", "", reference)
    chk_compact = re.sub(r"\s+", "", chunk_text)
    if ref_compact and ref_compact in chk_compact:
        return True
    return _directed_overlap(reference, chunk_text) >= CONTEXT_REL_MIN


def _split_claims(answer: str) -> List[str]:
    """把答案切分为陈述小句（忠实度核验的最小单位）。

    :param answer: 系统答案
    :return: 陈述列表（剔除无内容词的碎片）
    """
    # 先消去金额千分位逗号（如 15,000 / 4,926.50），避免被误当句读切开
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", answer)
    claims = []
    for part in _CLAIM_SPLIT_RE.split(text):
        part = part.strip("：: ")
        if part and _content_set(part):
            claims.append(part)
    return claims


# ---------------------------------------------------------------------------
# 二、四项指标的离线近似
# ---------------------------------------------------------------------------

def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """忠实度：答案陈述被检索上下文支持的比例（蕴含近似）。

    每条陈述同时满足：①内容词在上下文中的覆盖度≥阈值；②陈述中全部
    数字都能在上下文中找到（数字一致性，防止抽取式系统串改数字），
    才计为“被支持”。

    :param answer: 系统答案
    :param contexts: 检索到的证据块文本列表
    :return: 0~1 忠实度
    """
    joined = "\n".join(contexts)
    ctx_nums = normalized_numbers(joined)
    claims = _split_claims(answer)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        token_ok = _directed_overlap(claim, joined) >= ENTAIL_TOKEN_MIN
        claim_nums = normalized_numbers(claim)
        number_ok = claim_nums.issubset(ctx_nums)
        if token_ok and number_ok:
            supported += 1
    return round(supported / len(claims), 4)


def answer_relevancy(answer: str, gold_answer: str,
                     key_entities: Optional[Sequence[str]] = None,
                     key_numbers: Optional[Sequence[str]] = None) -> float:
    """答案相关性：答案对 gold 要点（词项/实体/数字）的覆盖。

    权重：内容词召回 0.5、实体召回 0.25、数字 F1 0.25；
    未显式声明实体或数字时，其权重并入内容词召回。

    :param answer: 系统答案
    :param gold_answer: 标准答案
    :param key_entities: gold 中必须命中的实体（人名/机构名等）
    :param key_numbers: gold 中必须命中的数字（支持 "1840" 这类字符串）
    :return: 0~1 相关性
    """
    # 内容词召回
    gold_tokens = _content_set(gold_answer)
    token_recall = (len(gold_tokens & _content_set(answer)) / len(gold_tokens)
                    if gold_tokens else 1.0)

    # 实体召回（显式声明的实体必须出现在答案原文中）
    ents = [e for e in (key_entities or []) if e]
    if ents:
        ent_recall = sum(1 for e in ents if e in answer) / len(ents)
    else:
        ent_recall = None

    # 数字 F1（以 gold 数字召回为主，惩罚答案中多出的无依据数字）
    gold_nums = set(key_numbers or []) | normalized_numbers(gold_answer)
    ans_nums = normalized_numbers(answer)
    if gold_nums:
        hit = len(gold_nums & ans_nums)
        precision = hit / len(ans_nums) if ans_nums else 0.0
        recall = hit / len(gold_nums)
        number_f1 = (2 * precision * recall / (precision + recall)
                     if precision + recall > 0 else 0.0)
    else:
        number_f1 = None

    # 动态权重组合
    score, weight = 0.0, 0.0
    score += 0.5 * token_recall
    weight += 0.5
    if ent_recall is not None:
        score += 0.25 * ent_recall
        weight += 0.25
    if number_f1 is not None:
        score += 0.25 * number_f1
        weight += 0.25
    return round(score / weight, 4)


def context_precision(reference_contexts: Sequence[str],
                      retrieved_contexts: Sequence[str]) -> float:
    """上下文精确率：按检索排名计算相关块的平均精度（AP）。

    :param reference_contexts: 参考证据列表
    :param retrieved_contexts: 按相关性降序的检索块文本列表
    :return: 0~1 平均精度；无参考证据或无相关块时为 0
    """
    if not reference_contexts or not retrieved_contexts:
        return 0.0
    n_ref = len(reference_contexts)
    # 每条参考证据只计一次（多个检索块命中同一条参考不重复计分），
    # 保证 AP 取值严格落在 0~1
    matched_refs = set()
    hits, precision_sum = 0, 0.0
    for rank, chunk in enumerate(retrieved_contexts, start=1):
        for ref_idx, ref in enumerate(reference_contexts):
            if ref_idx in matched_refs:
                continue
            if _context_relevant(ref, chunk):
                matched_refs.add(ref_idx)
                hits += 1
                precision_sum += hits / rank
                break
    return round(precision_sum / n_ref, 4)


def context_recall(reference_contexts: Sequence[str],
                   retrieved_contexts: Sequence[str],
                   key_entities: Optional[Sequence[str]] = None,
                   key_numbers: Optional[Sequence[str]] = None,
                   gold_answer: str = "") -> float:
    """上下文召回率：参考证据覆盖率与 gold 要点覆盖率的均值。

    分量一：每条参考证据是否至少被一个检索块覆盖；
    分量二：gold 的实体/数字（无显式声明时退化为 gold 内容词）
    是否出现在全部检索块的并集中。

    :param reference_contexts: 参考证据列表
    :param retrieved_contexts: 检索块文本列表
    :param key_entities: gold 关键实体
    :param key_numbers: gold 关键数字
    :param gold_answer: 标准答案（用于数字与内容词兜底）
    :return: 0~1 召回率
    """
    joined = "\n".join(retrieved_contexts)
    # 分量一：参考证据召回
    if reference_contexts:
        hit_refs = sum(
            1 for ref in reference_contexts
            if any(_context_relevant(ref, ch) for ch in retrieved_contexts))
        ref_recall = hit_refs / len(reference_contexts)
    else:
        ref_recall = None

    # 分量二：gold 要点（实体/数字）在检索上下文中的覆盖
    ents = [e for e in (key_entities or []) if e]
    nums = set(key_numbers or []) | normalized_numbers(gold_answer)
    points, point_hits = 0, 0
    for e in ents:
        points += 1
        point_hits += 1 if e in joined else 0
    ctx_nums = normalized_numbers(joined)
    for n in nums:
        points += 1
        point_hits += 1 if n in ctx_nums else 0
    if points == 0:
        # 无显式要点时退化为 gold 内容词覆盖
        gold_tokens = _content_set(gold_answer)
        point_cov = (len(gold_tokens & _content_set(joined)) / len(gold_tokens)
                     if gold_tokens else 1.0)
    else:
        point_cov = point_hits / points

    if ref_recall is None:
        return round(point_cov, 4)
    return round(0.5 * ref_recall + 0.5 * point_cov, 4)


# ---------------------------------------------------------------------------
# 三、单题评估入口（含不可回答题拒答口径）
# ---------------------------------------------------------------------------

def evaluate_one(sample: Dict, answer: str, refused: bool,
                 retrieved_contexts: Sequence[str]) -> Dict[str, float]:
    """对单题计算 RAGAS 四项指标。

    :param sample: questions.json 中的单题定义（含 gold_answer、
        reference_contexts、可回答、评分要点等）
    :param answer: 系统答案（拒答时为拒答话术）
    :param refused: 系统是否拒答
    :param retrieved_contexts: 检索块文本列表
    :return: 四项指标字典
    """
    answerable = bool(sample.get("可回答", True))
    gold = sample.get("gold_answer", "")
    refs = sample.get("reference_contexts", [])
    key_points = sample.get("评分要点", {}) or {}
    entities = key_points.get("entities", [])
    numbers = [str(n) for n in key_points.get("numbers", [])]

    # 不可回答题：正确拒答 → 四项记 1（拒答正确性单列统计）；误答 → 全 0
    if not answerable:
        if refused:
            return {"faithfulness": 1.0, "answer_relevancy": 1.0,
                    "context_precision": 1.0, "context_recall": 1.0,
                    "口径": "不可回答题正确拒答，按拒答口径记1"}
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
                "口径": "不可回答题未拒答（疑似编造），四项记0"}

    # 可回答题却拒答：答案类指标记 0，检索类指标照常计算（定位是检索失败
    # 还是拒答过敏，供问题分析使用）
    if refused:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": context_precision(refs, retrieved_contexts),
            "context_recall": context_recall(
                refs, retrieved_contexts, entities, numbers, gold),
            "口径": "可回答题被误拒答，答案类指标记0，检索指标据实计算"}

    return {
        "faithfulness": faithfulness(answer, retrieved_contexts),
        "answer_relevancy": answer_relevancy(answer, gold, entities, numbers),
        "context_precision": context_precision(refs, retrieved_contexts),
        "context_recall": context_recall(
            refs, retrieved_contexts, entities, numbers, gold),
        "口径": "可回答题常规离线RAGAS口径"}


# ---------------------------------------------------------------------------
# 四、官方 ragas 适配（接入 LLM 后替换用；当前环境不安装即跳过）
# ---------------------------------------------------------------------------

def evaluate_with_official_ragas(records: List[Dict],
                                 llm_wrapper=None,
                                 embeddings_wrapper=None) -> Optional[object]:
    """官方 RAGAS 适配函数（预留）。

    联网/有 LLM 的环境中：``pip install ragas`` 并按官方文档初始化
    LLM 与 embeddings 后，把 run_evaluation 产出的 records 直接传入本
    函数即可获得官方口径分数。字段映射：

    =========== ===========================
    官方字段    本流水线字段
    =========== ===========================
    question    问题
    answer      系统答案（拒答题建议剔除）
    contexts    检索块文本列表
    ground_truth gold_answer
    =========== ===========================

    :param records: run_evaluation 中的逐题记录列表
    :param llm_wrapper: 官方要求的 LLM 包装器（如 LangChainLLM）
    :param embeddings_wrapper: 官方要求的 embeddings 包装器
    :return: ragas 评估结果对象；ragas 未安装时返回 None 并打印指引
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)
    except ImportError:
        print("[官方RAGAS适配] 当前环境未安装 ragas/datasets，"
              "离线近似指标不受影响。接入 LLM 后执行 "
              "`pip install ragas datasets langchain` 再调用本函数。")
        return None

    rows = [{
        "question": r["问题"],
        "answer": r["答案"],
        "contexts": [c["文本"] for c in r["检索证据"]],
        "ground_truth": r.get("gold_answer", ""),
    } for r in records if not r.get("拒答", False)]
    if not rows:
        print("[官方RAGAS适配] 没有可评估的非拒答记录。")
        return None
    dataset = Dataset.from_list(rows)
    kwargs = {"metrics": [faithfulness, answer_relevancy,
                          context_precision, context_recall]}
    if llm_wrapper is not None:
        kwargs["llm"] = llm_wrapper
    if embeddings_wrapper is not None:
        kwargs["embeddings"] = embeddings_wrapper
    return evaluate(dataset, **kwargs)
