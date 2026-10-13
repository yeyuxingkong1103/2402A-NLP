# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
工单编号：人工智能NLP-RAG-Query理解优化
         （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
模块：RAG 全链路编排（查询理解 → 混合召回 → 精排 → 父块上下文 → 生成）

链路（每一步计时，供 ≤3 秒验收与瓶颈定位）：
    question
      └─ coref.resolve               多轮改写（state 非空时，延迟计入 total_ms，
                                     另在 result.rewrite.latency_ms 单独可读）
          └─ query_rewrite.analyze   意图/扩展/子问题              (analyze_ms)
              └─ retriever.search_multi 向量+BM25 多路召回 RRF     (retrieve_ms)
                  └─ reranker.rerank  cross-encoder 精排 top-6     (rerank_ms)
                      └─ context.expand_parents 小块→大块 + 压缩   (context_ms)
                          └─ llm.chat  qwen2.5:3b 生成（带 [n] 引用）(gen_ms)

性能保障（工单 ≤3s）：
  - warmup() 预热 embedding/reranker/Ollama（keep_alive 常驻）；
  - 规则版查询改写零额外延迟；全程单次 embedding（query 只编码一次）；
  - LRU 答案缓存：key = 改写后问题 + 会话焦点公司（多轮下同一代词问题
    在不同会话里指不同公司，只有原问题字符串会把答案串味）；
  - LLM 失败/超时 → 抽取式降级（直接给最相关片段与页码），不返回空白。

多轮（工单5）：`ask(..., state=...)` 先把问题经 `coref.resolve` 改写为独立问题再检索；
`ask_turn(question, state)` 在其上回填 `Turn` 到对话状态。无法解析的指代
（无历史焦点）返回 mode="clarify" 明确反问，绝不检索瞎答。

对外接口：
  ask(question, ..., state=None) -> RagResult（rewrite 字段留痕改写过程）
  ask_turn(question, state, ...) -> (RagResult, dialogue.Turn)
  ask_stream(question, ...) -> Iterator[dict]（事件流：meta → delta → final）
  retrieve(question) -> (hits, analysis, timings)
  warmup() -> dict
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from src import (config, context as ctx_mod, coref, dialogue, doc_router, llm,
                 query_rewrite, reranker, retriever)

# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class RagResult:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)     # 引用页码
    contexts: list[dict] = field(default_factory=list)      # 父块片段
    analysis: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    mode: str = "rag"                                       # rag | extractive | cached | clarify | error
    model: str = ""
    from_cache: bool = False
    rewrite: dict = field(default_factory=dict)             # 改写留痕（mode/resolved/span/latency_ms）

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer,
                "citations": self.citations, "analysis": self.analysis,
                "timings": self.timings, "mode": self.mode, "model": self.model,
                "from_cache": self.from_cache, "rewrite": self.rewrite,
                "n_contexts": len(self.contexts)}


# ---------------------------------------------------------------------------
# 提示词（中英双语）
# ---------------------------------------------------------------------------
_SYSTEM_ZH = """你是招股说明书问答助手，只能依据提供的【文档片段】回答。

本系统检索两份招股说明书（每个片段都标注了来源）：
  ·《招股说明书1.pdf》—— 武汉兴图新科电子股份有限公司
  ·《招股说明书2.pdf》—— 武汉力源信息技术股份有限公司
回答时必须先看清片段来源，只用**问题所问的那家公司**的数据作答，不要把两家公司的数据混淆。

硬性要求：
1. 片段中只要包含与问题相关的信息，就必须直接作答；只有全部片段都完全不含相关信息时，
   才回答"根据已有资料无法确定"。
2. 数字、比例、金额、名称、日期必须与片段**逐字一致**（含千分位与百分号），
   禁止改写、缩写、取整或调换顺序；多条数据按片段中出现的顺序完整列出。
   若片段中出现"…分别为A、B、C和D"这样的句式，请**整句复制**其中的数字与顺序，
   不重不漏；不要从其他句子（尤其是表格的"小计/合计"行）取数。
3. 标有【表格数据】的片段是**结构化表格**（Markdown 表格，"|"分隔列）：
   - 先按表头找到问题问的那一列/那一行，再读取对应的值；
   - 表格题往往需要**同一张表里的多个单元格**（如"发行股数"和"发行后总股本"），
     请把它们都找出来；
   - **必须逐行完整列出**：表中只要是该问题的答案行，一行都不能少。绝不要因为
     若干行"同属一类"（如都写着"持有公司股份5%以上的股东"）就合并成一句、
     只报一个类别名称——把每行的名称都写出来；
   - 若问题要求的是比例、占比等未直接给出的数值，允许用表中的数字做**一步除法/减法**
     计算，并在答案中说明依据的两个数字（如"1,670万股 ÷ 6,670万股 ≈ 25.04%"）；
   - 除上述计算外，不得引入表格以外的数字。
4. 只回答被问的指标：片段里常同时出现多个指标（如营业收入总额、毛利率等），
   不要把与问题无关的其他数字列进答案。
5. 关键表述要完整引用（如"军队视频指挥领域的重要供应商"），
   不要简化成上位词（如"国防军队领域"），也不要截断专有名称。
6. 注意区分主体：文中出现的证券公司、会计师事务所、律师事务所等机构（及其法定代表人、
   签字人）不属于发行人，回答关于发行人的问题时必须忽略这些机构的信息。
7. 若多个片段涉及同一事实但详略不同，以**信息最完整**的那条为准（例如含"由某某牵头、
   联合某某共同制定"的完整句子优于简称），并优先引用该片段。
8. **数值口径冲突时以"完整枚举句"为准**：当某片段出现"报告期内……分别为A、B、C和D"
   这类把各期数值一次列全的句子时，直接用它的数值与顺序。表格里常同时有多个口径的列
   （金额 / 占比 / 上期数 / 另一指标的占比），**不要**改从表格的其他列取数，
   也不要把不同来源的数字混着用。
8.1 **沿革/演变类表格取"最新"一次**：招股书常把同一条历史沿革按次列成多张结构相同的表，
   标题只差日期或序号，例如：
       （一）2017 年 1 月公司注册资本增加至 5,016 万元
       （二）2017 年 7 月公司注册资本增加至 5,225 万元
       （五）2018 年 8 月公司注册资本增加至 5,520 万元
   问"注册资本是多少""股本是多少"这类**当前值**时，必须回答**日期最晚的那一次**
   （上例为 2018 年 8 月的 5,520 万元），绝不能回答中间某次或最早那次。
   同理适用于"历次股权转让""增资""股本演变"等按时间排列的多张同构表。
9. **专有名称逐字照抄**：书名号/引号内的全称要完整复制，不得简写、缩写或改字
   （例：写"《某视频指挥系统技术规范（1.0版）》"，不要写成"《某视频技术规范 1.0》"）。
10. 答案末尾用 [1][2] 标注所依据的片段编号（引用必须真实存在，不要重复标注）。
11. 回答直接给结论，**简洁但完整**：用一句话把"主体 + 指标 + 数值/名称"说清楚
   （如"报告期内，公司来自军用领域的收入分别为A、B、C和D。"），
   不要只甩一串裸数字；不复述问题、不输出分析过程、不使用 Markdown 标题。

示例（注意答案完整保留了片段中的关键表述，且没有添加片段外的数字）：
  片段1：兴图新科目前已经成为国防军队视频指挥领域的重要供应商。
  问：公司在哪个领域已经成为重要供应商？
  答：国防军队视频指挥领域的重要供应商。[1]
  片段2（表格数据｜本次发行概况）：
    | 发行股数 | 1,670万股 |
    | 发行后总股本 | 6,670万股 |
  问：本次发行股数是多少，占发行后总股本的比例是多少？
  答：本次发行股数1,670万股，占发行后总股本6,670万股的比例约25.04%（1,670÷6,670）。[2]"""

_SYSTEM_EN = """You are an assistant answering questions about prospectuses. Answer ONLY from the provided [Document Fragments].
**Write your answer in English.**

Two prospectuses are indexed (every fragment states its source):
  · Prospectus 1 — Wuhan Xingtu Xinke Electronics Co., Ltd.
  · Prospectus 2 — Wuhan P&S Information Technology Co., Ltd.
Always check the source of each fragment and use only the data of the company the question asks about.

Hard requirements:
1. If the fragments contain information relevant to the question, you MUST answer directly; answer "Cannot be determined from the available materials" ONLY when none of the fragments contain any relevant information.
2. Numbers, percentages, amounts, names and dates must be copied EXACTLY as in the fragments (keep thousands separators, percent signs and original order); never rewrite, abbreviate, round or reorder.
   **The sentence around them must still be English.** Keep the figure *and its original Chinese unit* verbatim —
   do not translate it and do not rescale it (write "1,670万股", never "1,670 million shares" and never
   "16.7 million shares"; write "1,526.38万元", never "RMB 15.26 million"). Chinese unit words are part of the
   value; converting them has produced wrong answers before. Proper nouns (person names, document titles,
   technical standards) also stay as written in the document.
   Example: Q "How many shares will be issued and what percentage is that?" →
   A "The company will issue 1,670万股, which is 25.04% of the post-issuance total share capital.[2]"
3. Fragments tagged [表格数据] are STRUCTURED TABLES (Markdown tables, columns separated by "|"):
   - locate the column/row asked about via the header first, then read the value;
   - table questions often need SEVERAL cells of the same table (e.g. shares issued and total shares after issuance) — find them all;
   - if the question asks for a ratio/percentage not given directly, you MAY compute it with one division/subtraction using the table's numbers and state the two numbers used;
   - beyond that computation, introduce no numbers outside the tables.
4. Answer only the asked metric: fragments often contain several metrics; do not list numbers irrelevant to the question.
5. Quote key expressions completely (e.g. "军队视频指挥领域的重要供应商"); do not generalize or truncate proper names.
6. Distinguish the subject: information about other parties (securities firm, accounting firm, law firm, their representatives) is NOT the issuer's information and must be ignored.
7. Cite fragment indices like [1][2] at the end; citations must exist and must not repeat.
8. Be direct and concise: no restating the question, no analysis, no Markdown headings.
9. Follow this style (the answer keeps the fragment's exact key wording and adds no outside numbers):
  Fragment 1: The company has become an important supplier in the military video command field.
  Q: In which field has the company become an important supplier?
  A: The military video command field.[1]"""


# ---------------------------------------------------------------------------
# 数字校验（防幻觉）：LLM 偶尔会写出上下文中不存在的数字（如 4,627.14→4,627.15）
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _fragment_pools(context_text: str) -> list[list[float]]:
    """把上下文按【片段N】切成若干"数字池"（`build_context` 用 --- 分隔片段）。

    为什么要按片段分池：见 `_is_derivable` —— 允许"两个数算出的比例"时，
    参与运算的两个数必须来自**同一段原文**，否则候选空间会膨胀到失去判别力。
    """
    pools: list[list[float]] = []
    for seg in (context_text or "").split("\n\n---\n\n"):
        vals = [v for v in (_to_float(m) for m in _NUM_RE.findall(seg)) if v is not None]
        if vals:
            pools.append(vals)
    return pools


def _is_derivable(value: float, pools: Sequence[Sequence[float]],
                  rel_tol: float = 0.005) -> bool:
    """value 能否由**同一片段内**的两个数经一步除法得到？

    工单3 背景：表格题常问"占……的比例是多少"，而表格只给分子分母
    （如 发行股数 1,670 万股 / 发行后总股本 6,670 万股，比例 25.04%）。
    提示词明确允许 LLM 做一步除法，因此这类"算出来的数字"不是幻觉。

    收敛过程（两次实测踩坑，都是"放太宽"或"收太紧"）：
      ① 最初用**上下文全部数字**两两运算（129 个数字 ≈ 8000 对 × 7 种运算
         ≈ 5.6 万个候选值）→ 几乎任何两位小数都能"碰巧"被算出，防幻觉校验形同虚设；
      ② 收紧为"参与运算的两个数都要出现在**答案里**"→ 误伤 id 1：模型自然句式是
         "发行股数1,670万股，占发行后总股本的比例为25.04%"，只写了分子与结果，
         不会再把分母 6,670 重复一遍，于是正确回答被当成幻觉触发重试；
      ③ 现方案：两个数必须**同处一个片段**，且只认"除法得到百分比/比值"
         （分母 > 分子）。片段内数字量小、且比例型派生天然要求分母更大，
         两个约束叠加后既不误放也不误拦。
    """
    for pool in pools:
        vals = [v for v in pool if v != 0]
        for a in vals:
            for b in vals:
                if b <= a:
                    continue                     # 百分比/比值：分母必须更大
                if abs(a / b * 100 - value) <= max(rel_tol * abs(value), 1e-9):
                    return True
                if abs(a / b - value) <= max(rel_tol * abs(value), 1e-9):
                    return True
    return False


def check_numbers(answer: str, context_text: str,
                  allow_direct: bool = False) -> tuple[list[str], list[str]]:
    """校验答案中的数字是否都出现在上下文中，并检测列表型答案中的重复数字。

    返回 (答案中的数字列表, 可疑数字列表)。
    两类可疑：
      1. 幻觉 —— 数字在上下文中不存在（如 4,627.14 写成 4,627.15）；
         工单3：可由上下文两个数一步运算得到的"派生值"（比例等）不算幻觉。
      2. 重复 —— 列表型答案（≥3 个大数字）中同一数字出现多次，几乎必是复制错误
         （实测案例："6,464.51、18,780.67、14,414.16 和 6,464.51"）。
    """
    ctx_nums = {m.replace(",", "") for m in _NUM_RE.findall(context_text or "")}
    ans_nums = _NUM_RE.findall(answer or "")
    bad: list[str] = []
    need_derive = [n for n in dict.fromkeys(ans_nums)
                   if n.replace(",", "") not in ctx_nums]
    if need_derive:
        # 允许"同一片段内两个数一步算出"的派生值（见表 _is_derivable 的说明）
        pools = _fragment_pools(context_text) if allow_direct else []
        for n in need_derive:
            val = _to_float(n)
            if allow_direct and val is not None and _is_derivable(val, pools):
                continue
            bad.append(n)

    big = [n for n in ans_nums if len(n.replace(",", "").replace(".", "")) >= 4]
    if len(set(big)) >= 3:
        for n in dict.fromkeys(big):
            if big.count(n) > 1 and n not in bad:
                bad.append(n)
    return ans_nums, bad


_QSTOP = re.compile(r"[，。、；：？?！!（）()“”\"'\s　]|是多少|分别是|分别|多少|哪些|哪个|"
                    r"什么|公司|股份|有限|报告期内|的|了|和|与|及")


def _strip_doc_names(text: str) -> str:
    """去掉问题里的公司名/别名。

    为什么要去掉：公司名是**路由用的上下文**，不是问题的实质内容。
    实测踩坑（id 1）：问题里"武汉力源信息技术股份有限公司"占了大半的 4-gram，
    于是一段讲"…武汉力源…注册资本5,000万元…"的无关句子也能拿到 0.32 的词重叠度，
    先后骗过了"重试护栏"与"枚举句校正"，把正确回答顶成了离题答案。
    """
    t = text or ""
    for aliases in config.DOC_ALIASES.values():
        for a in sorted(aliases, key=len, reverse=True):
            t = t.replace(a, "")
    return t


def _ngrams(text: str, n: int = 4) -> set[str]:
    t = _QSTOP.sub("", _strip_doc_names(text))
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 0))}


def _enumeration_numbers(sentence: str) -> list[str]:
    return [n for n in _NUM_RE.findall(sentence)
            if len(n.replace(",", "").replace(".", "")) >= 4]


def question_relevance(question: str, answer: str) -> float:
    """答案与问题的词面贴合度 ∈ [0,1]（问题 4-gram 被答案覆盖的比例）。

    用途：**重试护栏**。工单3 实测踩坑：id 1 首次答案把 25.04% 写成了相近的错误值，
    触发重试后模型答到了完全无关的段落（注册资本、股权演变），而该答案"没有可疑数字"，
    于是被判为"不更差"而采纳 —— 正确答案被一个离题答案顶掉。
    数字层面的判据拦不住这种退化，必须再要求"重试不能偏离问题本身"。
    """
    q_grams = _ngrams(question)
    if not q_grams:
        return 1.0
    return len(q_grams & _ngrams(answer)) / len(q_grams)


# --- 沿革/演变类同构表：问"当前值"时必须取日期最晚的那张 -----------------------
_DATE_IN_TEXT_RE = re.compile(r"((?:19|20)\d{2})\s*年(?:\s*(\d{1,2})\s*月)?")
_PAREN_RE = re.compile(r"[（(][^）)]{0,6}[）)]")


def _caption_template(caption: str) -> str:
    """把表标题归一成"模板"：去掉（一）这类序号、去掉空白、数字统一成 #。"""
    t = _PAREN_RE.sub("", caption or "")
    t = re.sub(r"[\s　]", "", t)
    return re.sub(r"[\d,]+", "#", t)


def _caption_value_numbers(caption: str) -> list[str]:
    """从表标题里取"数值"（排除日期中的年/月数字）。"""
    t = _DATE_IN_TEXT_RE.sub("", caption or "")
    return _NUM_RE.findall(t)


def check_chronology(answer: str, blocks: Sequence[dict],
                     question: str) -> list[str]:
    """同构沿革表检查：问当前值却取了非最新的一次 → 给出提示触发重试。

    工单3 实测（id 543）：上下文里同时有
        （一）2017 年 1 月公司注册资本增加至 5,016 万元
        （二）2017 年 7 月公司注册资本增加至 5,225 万元
        （五）2018 年 8 月公司注册资本增加至 5,520 万元
    三张**标题只差日期**的同构表。问"注册资本是多少"（当前值）应答 5,520 万元，
    但小模型稳定地答成中间某一次 —— 加提示词也压不住（实测改一次答错一次），
    故用确定性判据：按标题模板分组，若同一模板下有多张表且答案里没有**最晚**那张
    的数值，就定向重试并点名最新的那张表。

    问题自身带年份/月份时不做纠正（问的就是历史上某一次）。
    """
    if _DATE_IN_TEXT_RE.search(question or ""):
        return []
    groups: dict[str, list[tuple[tuple[int, int], str, dict]]] = {}
    for b in blocks:
        if not b.get("is_table"):
            continue
        cap = b.get("table_caption") or ""
        m = _DATE_IN_TEXT_RE.search(cap)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2) or 0))
        groups.setdefault(_caption_template(cap), []).append((key, cap, b))

    for items in groups.values():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[0])
        _latest_key, latest_cap, _latest_b = items[-1]
        nums = _caption_value_numbers(latest_cap)
        if not nums:
            continue
        ans_norm = (answer or "").replace(",", "")
        if any(n.replace(",", "") in ans_norm for n in nums):
            continue                        # 已采用最新那张的值
        return [f"上下文里有 {len(items)} 张**标题只差日期**的同构表格："
                f"{'；'.join(c for _k, c, _b in items)}。"
                f"问的是**当前**值，必须取日期最晚的「{latest_cap}」中的数值作答。"]
    return []


def _best_enumeration_sentence(question: str, context_text: str,
                               min_score: float = 0.45
                               ) -> tuple[str, float]:
    """在上下文里找与问题**词面最贴合**的枚举句，返回 (句子, 重叠度)。

    工单3 实测（id 33）背景：招股书里存在两句结构几乎一样的话——
      · "报告期内，公司**来自军用领域的收入**…占主营业务收入**比重**分别为 82.10%、97.31%、94.84%和 94.34%"
      · "报告期内，公司**视频指挥控制类产品的销售收入**…占主营业务收入**比例**分别为 78.39%、95.86%、94.37%和 91.92%"
    问的是前者，小模型却把两句的数字混着抄（94.37%、95.86%、78.39%、94.34%）。
    这种"两份结构相同、数值相近的枚举句"是招股书的常见陷阱，靠提示词约束不住，
    故用**问题词重叠度**挑出真正对应的那一句。

    两道闸门（都是实测踩坑后加的）：
      · `min_score=0.45`：阈值太低会误伤。曾经的 0.25 让 id 1（问"本次发行股数…占比"）
        选中了一段讲注册资本/股权演变的枚举句，把正确回答整个替换掉；
      · **显著领先**：最佳句必须比次佳句高 40% 以上，避免两个候选难分伯仲时乱选。
    """
    q_grams = _ngrams(question)
    if not q_grams:
        return "", 0.0
    scored: list[tuple[float, str]] = []
    for sent in re.split(r"[。；\n]", context_text or ""):
        s = sent.strip()
        if not (30 <= len(s) <= 220) or "|" in s or s.endswith(("，", "、", ",", "：")):
            continue
        s = re.sub(r"[，,][^，,。；]{0,6}$", "", s).strip()
        if len(s) < 30 or len(_enumeration_numbers(s)) < 3:
            continue
        s_grams = _ngrams(s)
        if not s_grams:
            continue
        scored.append((len(q_grams & s_grams) / len(q_grams), s))
    if not scored:
        return "", 0.0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < min_score or best_score < second * 1.4:
        return "", best_score
    return best, best_score


def complete_numeric_list(answer: str, context_text: str,
                         question: str = "") -> tuple[str, bool]:
    """数字列表型答案的抽取式校正（防 3B 模型漏抄/重复/抄错口径）。

    场景一（漏抄/重复）：问"…分别是多少"，片段含 4 个数字，模型输出 3 个
    或重复第 1 个。做法：若答案包含 ≥3 个大数字，且上下文中存在一个句子
    覆盖了答案的**全部**数字并包含更多数字，则用该原句替换。
    场景二（抄错口径，工单3 新增）：答案的数字本身都在文档里，但来自**另一句
    结构相同的枚举句**。做法：按问题词重叠度选出最贴合的枚举句，若其数字与
    答案的数字集合不一致且重叠度足够高，则用该原句替换。

    原文永远比模型转述可靠；两级判据都要求"数字数量足够 + 重叠度高"，
    避免在小样本上误替换。
    """
    ans_nums = _NUM_RE.findall(answer or "")
    big = [n for n in ans_nums if len(n.replace(",", "").replace(".", "")) >= 4]
    if len(set(big)) < 3:
        return answer, False
    ans_set = {n.replace(",", "") for n in big}

    best_sent, best_hit = None, 0
    for sent in re.split(r"[。；\n]", context_text or ""):
        s = sent.strip()
        if not (30 <= len(s) <= 170):
            continue
        if "|" in s:                       # 表格行（"小计 | 4,627.15 | …"）不可作答案
            continue
        if s.endswith(("，", "、", ",", "：")):
            continue
        # 去掉结尾的短残句（父块裁剪可能留下"，占"这类尾巴）
        s = re.sub(r"[，,][^，,。；]{0,6}$", "", s).strip()
        if len(s) < 30:
            continue
        s_set = {n.replace(",", "") for n in _enumeration_numbers(s)}
        if not s_set:
            continue
        hit = len(ans_set & s_set)
        # 全覆盖 + 句子数字更全 + 命中最多者胜出
        if hit == len(ans_set) and len(s_set) > len(ans_set) and hit > best_hit:
            best_sent, best_hit = s, hit
    if best_sent:
        return best_sent + "。", True

    # 场景二：按问题词重叠挑枚举句，数字集合不一致则用原文替换
    if question:
        sent, score = _best_enumeration_sentence(question, context_text)
        if sent:
            s_set = {n.replace(",", "") for n in _enumeration_numbers(sent)}
            if s_set and s_set != ans_set:
                return sent + "。", True
    return answer, False


# ---------------------------------------------------------------------------
# 生成后校验（工单3 新增两条，专治实测到的表格题失败模式）
# ---------------------------------------------------------------------------
# 列举型问题的问法（"有哪些/包括哪些/分别是多少"）——只有这类问题才要求"逐行列全"
_LIST_INTENT_RE = re.compile(r"哪些|哪几|有哪|包括|分别是|分别有|都有")

# 表格首列中属于"列标题"而非"答案行"的取值，不参与遗漏检查
_ROW_KEY_STOP = {
    "企业名称", "名称", "项目", "内容", "序号", "合计", "小计", "备注", "说明",
    "与本公司关系", "与公司关系", "关系", "金额", "占比", "比例", "数量",
}
_QUOTE_RE = re.compile(r"《([^》]{2,60})》")


def table_row_keys(parent_text: str) -> list[str]:
    """从表格 Markdown（parent_text）里取首列的"行名"（跳过表头与列标题）。"""
    keys: list[str] = []
    for i, line in enumerate((parent_text or "").splitlines()):
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or i == 0:
            continue                       # 第 0 行是渲染出来的表头
        key = cells[0]
        if not key or key in _ROW_KEY_STOP:
            continue
        if re.fullmatch(r"[\d,\.%\-—]+", key):
            continue                       # 纯数字行名（如序号列）不检查
        if len(key) < 2:
            continue
        keys.append(key)
    return keys


def check_table_rows(answer: str, blocks: Sequence[dict],
                     question: str) -> list[str]:
    """表格片段里有、但答案里没提到的行名（列举型问题才检查）。

    工单3 实测（id 4）：表格 7 家企业，模型因为其中 5 家"同属一类"
    （持有公司股份5%以上的股东）就合并成一句，漏掉了另 2 家。
    提示词约束对小模型不够，故在生成后做一次确定性核对并触发定向重试。
    """
    if not _LIST_INTENT_RE.search(question or ""):
        return []
    missing: list[str] = []
    for b in blocks:
        if not b.get("is_table"):
            continue
        for k in table_row_keys(b.get("parent_text", "")):
            if k not in (answer or "") and k not in missing:
                missing.append(k)
        break                              # 只看上下文里第一张表（避免多表误伤）
    return missing


def complete_row_list(answer: str, blocks: Sequence[dict],
                      question: str) -> tuple[str, list[str]]:
    """表格列举补全：答案漏了同表行名时，**确定性**把漏行补进答案。

    为什么要在提示词 + 定向重试之外再加一层（工单6 实测）：
      id4「不存在控制关系的关联方企业有哪些」——表格 7 家企业，前 5 家关系列
      相同（"持有公司股份5%以上的股东"），模型合并成一句只报 5 家；
      即使重试提示里**点名**"漏掉了：力源贸易、普芯达"，3B 模型仍然照旧。
      这与数字的情况同构（见 complete_numeric_list），故采用同一策略：
      生成侧尽力，抽取侧兜底 —— 补入的行名**全部来自上下文表格**，
      不引入任何表外内容，属于"抽取"而非"编造"。

    保守门控（宁可少补，不可错补）：
      · 仅对列举型问题（哪些/哪几/…）生效；
      · 只看上下文里**第一张**含行名的表；
      · 答案已覆盖该表 ≥50% 的行（说明确实是这张表在回答该问题）；
      · 漏行 ≤5 个（漏太多更像"只有部分行相关"，不猜）。
    """
    if not _LIST_INTENT_RE.search(question or ""):
        return answer, []
    for b in blocks:
        if not b.get("is_table"):
            continue
        keys = table_row_keys(b.get("parent_text", ""))
        if len(keys) < 3:
            break
        present = [k for k in keys if k in (answer or "")]
        missing = [k for k in keys if k not in (answer or "")]
        if not missing or not present:
            break
        if len(present) < max(1, len(keys) // 2) or len(missing) > 5:
            break
        suffix = "；同表其余相关行还有：" + "、".join(missing)
        # 引用标记（如 [1][2]）要留在整句末尾，不能把补语塞到它后面
        m = re.search(r"(\[\d+\](?:\[\d+\])*)\s*$", answer or "")
        if m:
            head = answer[:m.start()].rstrip("。；; ")
            answer = f"{head}{suffix}。{m.group(1)}"
        else:
            answer = (answer or "").rstrip("。；; ") + suffix + "。"
        return answer, missing
    return answer, []


def check_quotes(answer: str, context_text: str) -> list[str]:
    """答案里与片段**不一致**的书名号名称。

    工单3 实测（id 95）：片段写"《某视频指挥系统技术规范（1.0版）》"，
    模型简写成"《某视频技术规范 1.0》"——名称类事实被改写，
    对工单要求的"数字、名称逐字一致"是硬伤。
    """
    ctx_names = {m.strip() for m in _QUOTE_RE.findall(context_text or "")}
    bad: list[str] = []
    for name in _QUOTE_RE.findall(answer or ""):
        n = name.strip()
        if n and n not in ctx_names and n not in bad:
            bad.append(n)
    return bad


def repair_quotes(answer: str, context_text: str) -> tuple[str, list[tuple[str, str]]]:
    """把答案里与片段不一致的书名号名称，确定性替换为片段中**最完整的相近名称**。

    为什么需要（工单6 实测 id95）：
      同一份材料里标准名有两个版本 —— 权威处写
      「《某视频指挥系统技术规范（1.0版）》」，另一处 OCR 噪声写成「《某视频技术规范 1.0》」。
      3B 模型倾向抄**较短**的那个，答案名称因此不是全称。重试提示压不住（与 id4 同因），
      故在生成后用字符覆盖度寻找候选、按"最完整"择优替换（提示词第 7 条同款规则，
      只是改为确定性执行）。替换只发生在**上下文里真实存在的名称**之间，
      不构造新名称 —— 属抽取式校正，不引入幻觉。
    """
    ctx_names = {m.strip() for m in _QUOTE_RE.findall(context_text or "") if m.strip()}
    if not ctx_names or not answer:
        return answer, []

    fixed: list[tuple[str, str]] = []
    for raw in _QUOTE_RE.findall(answer):
        n = raw.strip()
        if not n:
            continue
        # 候选门槛：① 答案名称不在上下文 → 找相近的；② 答案名称虽在上下文，
        # 但上下文存在**同源的更完整名称**（覆盖率 ≥0.9 且更长）→ 升级为全称。
        # ② 是实测补上的：噪声版本「《某视频技术规范1.0》」本身也在语料里，
        # 只判"是否在上下文出现过"会漏掉升级，答案仍是简写。
        in_ctx = n in ctx_names
        best: str | None = None
        for cand in ctx_names:
            if cand == n:
                continue
            cover = len(set(n) & set(cand)) / max(len(set(n)), 1)
            if in_ctx:
                if cover < 0.9 or len(cand) <= len(n):
                    continue          # 只在"更完整"时才升级
            elif cover < 0.6:
                continue
            if best is None or len(cand) > len(best):
                best = cand
        if best and f"《{n}》" in answer:
            answer = answer.replace(f"《{n}》", f"《{best}》")
            fixed.append((n, best))
    return answer, fixed


def answer_lang(question: str) -> str:
    """答案语言 = **用户原始提问**的语言（剥离句中的公司名后复判）。

    不能用 `analysis["lang"]`：那是 `query_rewrite.detect_language` 的纯 ASCII 占比
    口径，问句外壳是英文、里面嵌了中文公司全称时占比被稀释
    （"Who is the legal representative of 武汉力源信息技术股份有限公司?" → 0.58），
    会被判成 zh，于是英文提问回中文答案（工单 P6「支持中文和英文的问答」），
    且 `wrong_lang` 语言守卫要求 `lang == "en"` 才触发，根本不会启动。
    多轮场景下 `analysis["lang"]` 还是**改写后**问题的语言，与"用户提问语言"
    更不是一回事。故这里一律按原始问题判、并按剥离实体后的外壳复判。
    """
    return coref.detect_language(question or "")


def build_messages(question: str, context_text: str, lang: str = "zh") -> list[dict]:
    system = _SYSTEM_EN if lang == "en" else _SYSTEM_ZH
    user = (f"【文档片段】\n{context_text}\n\n【问题】\n{question}"
            if lang != "en" else
            f"[Document Fragments]\n{context_text}\n\n[Question]\n{question}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 检索（含计时）
# ---------------------------------------------------------------------------
def _nearby_table_notes(ranked: Sequence[dict], final: Sequence[dict]) -> list[dict]:
    """精排窗口里「同表多行」却没进最终上下文的表 → 只做**引用定位**，不进生成上下文。

    为什么只定位、不喂表（实测证据，`_scratch/audit_table_modes.py`）：
    p342 收入表与正文存在**源文档级数字冲突**——表里「军用领域/小计/2019年1-6月」
    印的是 4,627.15（= 4,508.18 + 118.97 的舍入），而正文与验收标准答案是 4,627.14。
    把该表放进生成上下文后，模型改用表里的数字，且「数字列表校正」要求答案数字被
    某句原文**全覆盖**，4,627.15 不在任何正文句里 → 校正永不触发、护栏失效。
    实测 5/5 次答成「4,627.15…」并漏掉 14,414.16，验收 Q1 直接不通过。
    触发条件保守：同一张表在窗口内 ≥2 行命中（不同表各 1 行的弱信号不记）。
    """
    in_final = {h.get("table_id") for h in final if h.get("table_id")}
    rows: dict[str, list[dict]] = {}
    for h in ranked:
        tid = h.get("table_id") or ""
        if tid and tid not in in_final:
            rows.setdefault(tid, []).append(h)
    notes = []
    for tid, hs in rows.items():
        if len(hs) < 2:
            continue
        h0 = hs[0]
        notes.append({
            "table_id": tid,
            "table_caption": h0.get("table_caption") or "",
            "source": h0.get("source", config.SOURCE_NAME),
            "page_display": int(h0.get("page_idx", 0)) + 1,
            "row_indices": sorted({int(x.get("row_index", -1)) for x in hs
                                   if int(x.get("row_index", -1)) >= 0}),
        })
    return notes


def _attach_nearby_tables(cmap: Sequence[dict], notes: Sequence[dict] | None) -> None:
    """把定位信息挂到**同章节/同页**的引用条目上（就地修改 cmap）。

    匹配优先用标题：引用块的 `heading_path` 末级就是表名（如
    「4、主营业务收入按行业列示」）——这是两张证据「说的是同一处」的确切判据；
    无表名的表格退化为同页（±1 页，表常跨页）。
    """
    for c in cmap:
        if c.get("is_table"):
            continue                              # 已在上下文里的表格自不必挂
        for n in notes or []:
            if c.get("source") != n["source"]:
                continue
            head = (c.get("heading_path") or [""])[-1]
            same_heading = bool(n["table_caption"]) and head == n["table_caption"]
            same_page = abs(int(c.get("page_display", -99)) - int(n["page_display"])) <= 1
            if same_heading or (same_page and not n["table_caption"]):
                c["nearby_table"] = dict(n)
                break


def retrieve(question: str, analysis: dict | None = None,
             top_k: int | None = None,
             use_rerank: bool = True,
             exclude_block_types: Sequence[str] | None = None
             ) -> tuple[list[dict], dict, dict]:
    """查询理解 + 文档路由 + 混合召回 + 精排。返回 (hits, analysis, timings)。

    工单3 新增**多文档路由**：问题里出现公司名时，把检索限定到对应 PDF
    （"武汉力源"→招股说明书2，"武汉兴图新科"→招股说明书1）；
    若路由后候选过少（可能路由误判），自动放开过滤重检（fallback_open）。

    exclude_block_types：消融对比用（排除 table_row/table_header 即"无表格优化"）。
    """
    timings: dict = {}

    t0 = time.time()
    analysis = analysis or query_rewrite.analyze(question)
    timings["analyze_ms"] = round((time.time() - t0) * 1000, 1)

    t0 = time.time()
    route = doc_router.describe(question)
    analysis["doc_route"] = route
    sources = route["sources"]
    analysis["exclude_block_types"] = list(exclude_block_types or [])

    queries = query_rewrite.multi_queries(analysis)
    subs = analysis.get("sub_questions") or []
    recall_info: dict = {}

    def _recall(srcs):
        # 子问题 + 主查询共同召回：统一融合，避免子问题权重被稀释。
        # 走 strategies 层以便按运行时设置选择 向量/全文/混合 策略（工单06 核心）。
        from src import runtime, strategies

        qs = queries + subs if subs else queries
        cfg = strategies.with_overrides(
            runtime.get(),
            top_k_recall=config.VECTOR_TOP_K * 2,
            sources=list(srcs) if srcs else None,
            exclude_block_types=(list(exclude_block_types)
                                 if exclude_block_types else None))
        res = strategies.retrieve_multi(qs, cfg, do_rerank=False)
        recall_info.clear()
        recall_info.update(res.get("info") or {})
        return res["hits"]

    hits = _recall(sources)
    # 路由兜底：限定文档后命中太少 → 放开过滤全库重检一次（容错机制）
    if sources and config.DOC_ROUTER_FALLBACK_OPEN and \
            len(hits) < config.RERANK_INPUT_TOP_K:
        hits_open = _recall(None)
        if len(hits_open) > len(hits):
            hits = hits_open
            route["fallback_opened"] = True
    timings["retrieve_ms"] = round((time.time() - t0) * 1000, 1)
    analysis["retrieval_info"] = recall_info
    analysis["retrieval_strategy"] = recall_info.get("strategy")

    t0 = time.time()
    if use_rerank and hits:
        from src import rerankers, runtime

        rcfg = runtime.get()
        top_in = hits[: config.RERANK_INPUT_TOP_K]
        # 全量取回精排名次（打分本就覆盖全部候选，不额外花时间）：
        # 「同表多行」要能在 top-k **之外**看到，先截断就拿不到了。
        res = rerankers.rerank_detailed(
            analysis["resolved"], top_in,
            algorithm=rcfg.rerank_algorithm, model=rcfg.reranker_model,
            top_k=len(top_in), score_threshold=rcfg.score_threshold)
        ranked = res["hits"]
        # 工单硬性要求2：记录「召回候选数 / 精排耗时 / 精排前后 Top-K 名次变化」
        analysis["rerank_info"] = res["info"]
        analysis["recall_candidates"] = len(hits)
        hits = ranked[: top_k or config.FINAL_TOP_K]
        analysis["nearby_tables"] = _nearby_table_notes(ranked, hits)
    else:
        hits = hits[: (top_k or config.FINAL_TOP_K)]
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)
    return hits, analysis, timings


def build_qa_context(hits: Sequence[dict]) -> tuple[str, list[dict], list[dict], dict]:
    """命中子块 → (context_text, citation_map, contexts, timings)。"""
    t0 = time.time()
    blocks = ctx_mod.expand_parents(hits, max_chars=config.CONTEXT_MAX_CHARS)
    text = ctx_mod.build_context(blocks)
    cmap = ctx_mod.citation_map(blocks)
    return text, cmap, blocks, {"context_ms": round((time.time() - t0) * 1000, 1)}


# ---------------------------------------------------------------------------
# 引用抽取与校验
# ---------------------------------------------------------------------------
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def extract_citations(answer: str, cmap: Sequence[dict]) -> list[dict]:
    """从答案抽取 [n] 并映射为页码；无标注时返回空列表（界面回退显示全部片段页码）。"""
    idxs: list[int] = []
    for m in _CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if 1 <= n <= len(cmap) and n not in idxs:
            idxs.append(n)
    out = []
    for n in idxs:
        c = dict(cmap[n - 1])
        c["citation"] = f"[{n}]"
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# 抽取式降级
# ---------------------------------------------------------------------------
def extractive_answer(blocks: Sequence[dict]) -> str:
    """LLM 不可用时的降级答案：直接给出最相关片段原文与页码。"""
    if not blocks:
        return "根据已有资料无法确定。"
    b = blocks[0]
    text = (b.get("parent_text") or "").strip()
    if len(text) > 300:
        text = text[:300] + "…"
    page = int(b.get("page_idx", 0)) + 1
    return f"（LLM 暂不可用，以下为检索到的原文）\n{text}\n\n来源：第{page}页"


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
_cache: "OrderedDict[str, RagResult]" = OrderedDict()
_cache_lock = threading.Lock()


def _multiturn_cache_key(question: str, focus_company: str | None) -> str:
    """多轮缓存 key。

    工单3 的 key 只有原问题字符串；多轮下同一句「他参与的哪个工程获奖了？」
    在不同会话里指的是不同公司，沿用会把答案串味。故 key 必须含**改写后问题**
    与**焦点公司**。
    """
    return f"{question}||focus={focus_company or '-'}"


def _cache_get(key: str) -> RagResult | None:
    with _cache_lock:
        r = _cache.get(key)
        if r is not None:
            _cache.move_to_end(key)
        return r


def _cache_put(key: str, value: RagResult) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > config.ANSWER_CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
# clarify 反问里列出的公司名：从 dialogue.COMPANY_BY_SOURCE（口径源自
# config.DOC_ALIASES）单一来源生成，避免以后新增文档时这里漏改（review F5）。
_CLARIFY_COMPANIES = "与".join(dialogue.COMPANY_BY_SOURCE.values())


def ask(question: str, use_cache: bool = True, use_rerank: bool = True,
        top_k: int | None = None,
        exclude_block_types: Sequence[str] | None = None,
        state: "dialogue.DialogueState | None" = None) -> RagResult:
    """完整问答（同步）。检索与生成环节的异常都会降级而不是抛出。

    例外：改写层 `coref.resolve` **有意**放在 try 之外 —— coref 自身已逐层降级
    （L0-L4），若仍抛异常属于编程错误，直接暴露而不是被吞掉后静默降级整条链路。

    state 非空时进入多轮模式：先把问题改写为独立完整问题再检索，
    并把改写留痕写进 result.rewrite（供 UI 展示与评测统计）。
    exclude_block_types：消融对比用；不同模式的缓存互相隔离（见 cache_key）。
    """
    q = (question or "").strip()
    if not q:
        return RagResult(question="", answer="请输入问题。", mode="extractive")

    # 从函数入口计时：total_ms 必须覆盖 coref 改写（约 150-250ms/代词轮），
    # 因为「从用户问题到答案 ≤3s」的验收包含改写延迟（改写自身另有
    # rewrite.latency_ms 字段，其余 timings 字段语义不变）。
    t_start = time.time()
    rewrite_info: dict = {"mode": "passthrough", "resolved": q, "span": None, "latency_ms": 0.0}
    resolved = q
    if state is not None:
        rw = coref.resolve(q, state)
        if rw.mode == "unresolved":
            return RagResult(
                question=q,
                answer="我不确定您指的是哪家公司，请补充公司名称（《招股说明书》中收录的是"
                       f"{_CLARIFY_COMPANIES}）。",
                mode="clarify", analysis={"lang": coref.detect_language(q)},
                rewrite={"mode": "unresolved", "resolved": "", "span": rw.span,
                         "latency_ms": rw.latency_ms})
        resolved = rw.resolved
        rewrite_info = {"mode": rw.mode, "resolved": rw.resolved, "span": rw.span,
                        "candidates": rw.candidates, "degraded": rw.degraded,
                        "latency_ms": rw.latency_ms}

    focus = state.focus_company if state is not None else None
    cache_key = _multiturn_cache_key(resolved, focus)
    if exclude_block_types:
        cache_key += "|x:" + ",".join(sorted(exclude_block_types))
    if use_cache:
        hit = _cache_get(cache_key)
        if hit is not None:
            cached = RagResult(**{**hit.__dict__, "from_cache": True})
            cached.question = q
            # 留痕必须描述**本次**调用：同一 (改写后问题, 焦点) 可能由不同原始问题、
            # 不同改写路径产生（显式点名 → passthrough，代词 → coref），沿用缓存里
            # 首次写入的 rewrite 会让 ask_turn 回填的 rewrite_mode/coref_span 失真。
            cached.rewrite = rewrite_info
            return cached

    try:
        hits, analysis, timings = retrieve(resolved, top_k=top_k, use_rerank=use_rerank,
                                           exclude_block_types=exclude_block_types)
        # 反馈绑定用：记录本次精排第 1 名的块（界面点👍/👎 时归属到它）
        analysis["top_chunk_id"] = (hits[0].get("chunk_id") if hits else "")
        context_text, cmap, blocks, t_extra = build_qa_context(hits)
        _attach_nearby_tables(cmap, analysis.get("nearby_tables"))
        timings.update(t_extra)
    except Exception as exc:  # noqa: BLE001 —— 检索侧失败：明确报错而不是瞎答
        return RagResult(question=q, answer=f"检索失败：{exc}", mode="error",
                         rewrite=rewrite_info,
                         timings={"total_ms": round((time.time() - t_start) * 1000, 1)})

    # 答案语言按**用户原始提问**判定（见 answer_lang），并回写留痕。
    lang = answer_lang(q)
    analysis["lang"] = lang
    if not context_text.strip():
        result = RagResult(question=q, answer="根据已有资料无法确定。",
                           analysis=analysis, contexts=blocks, citations=cmap,
                           mode="rag", timings=timings, rewrite=rewrite_info)
    else:
        try:
            messages = build_messages(resolved, context_text, lang)
            t0 = time.time()
            resp = llm.chat(messages, temperature=0.0)
            answer = (resp.get("answer") or "").strip()

            # 生成后校验（三重）→ 有问题则定向重试一次（只重试一次，控住 ≤3 秒预算）：
            #   ① 数字：答案中出现片段里没有的数字（防幻觉；allow_direct 允许
            #      "表格两个数一步算出的比例"，工单3 表格题需要）
            #   ② 表格行：列举型问题漏列了表格里的行（工单3 实测 id 4）
            #   ③ 书名号名称：与片段不一致的简写/改写（工单3 实测 id 95）
            _, bad_nums = check_numbers(answer, context_text, allow_direct=True)
            miss_rows = check_table_rows(answer, blocks, q)
            bad_quotes = check_quotes(answer, context_text)
            # 多语言（工单验收项）：英文提问时，答案至少要是一句英文。
            # 实测 3B 模型对"简短事实型"答案会整句回中文（如只回"程家明"），
            # 提示词压不住，故做确定性判定：整段不含拉丁字母 → 触发重试。
            # 注意数字与中文单位/专名本就该保留原文，所以只判"完全没英文"这一档。
            wrong_lang = bool(lang == "en" and answer and not re.search(r"[A-Za-z]", answer))
            chrono = check_chronology(answer, blocks, q)
            if answer and (bad_nums or miss_rows or bad_quotes or wrong_lang or chrono):
                hints = []
                if wrong_lang:
                    hints.append(
                        "请用**英文**作答：整句用英文表述，但数字及其原始中文单位、"
                        "专有名词（人名/标准名/书名号内容）保持原文不译、不换算。")
                hints.extend(chrono)
                if bad_nums:
                    hints.append(
                        f"你上次回答中的这些数字有问题：{bad_nums}"
                        f"（在片段中不存在，也不是片段里两个数一步计算得来的，"
                        f"或在列表中重复出现）。请逐字复制片段中的数字"
                        f"（含千分位与小数点），按片段原文顺序完整列出，"
                        f"一个都不能重复、不能遗漏。")
                if miss_rows:
                    hints.append(
                        f"你上次回答**漏掉了表格里的这些行**：{miss_rows}。"
                        f"表格题的答案必须逐行列出表中所有相关行，"
                        f"不要因为若干行同属一类就合并省略——把每个名称都写出来。")
                if bad_quotes:
                    hints.append(
                        f"你上次回答中的这些名称与片段不一致：{bad_quotes}。"
                        f"书名号/引号内的名称必须与片段**逐字一致**，"
                        f"请照抄片段中的全称，不得简写或改字。")
                retry_messages = messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": " ".join(hints) + " 重新作答，只输出答案本身。"},
                ]
                resp2 = llm.chat(retry_messages, temperature=0.0)
                ans2 = (resp2.get("answer") or "").strip()
                if ans2:
                    _, bad2 = check_numbers(ans2, context_text, allow_direct=True)
                    miss2 = check_table_rows(ans2, blocks, q)
                    quote2 = check_quotes(ans2, context_text)
                    chrono2 = check_chronology(ans2, blocks, q)
                    lang2_ok = bool(lang != "en" or not ans2
                                    or re.search(r"[A-Za-z]", ans2))
                    n_before = (len(bad_nums) + len(miss_rows) + len(bad_quotes)
                                + int(wrong_lang) + len(chrono))
                    n_after = (len(bad2) + len(miss2) + len(quote2)
                               + int(not lang2_ok) + len(chrono2))
                    # 采用条件：① 问题层面不更差；② **不得偏离问题**（见
                    # question_relevance 说明——否则模型可能"修好数字"却答到别处）
                    rel_before = question_relevance(q, answer)
                    rel_after = question_relevance(q, ans2)
                    if n_after <= n_before and rel_after >= rel_before * 0.7:
                        answer = ans2
                        timings["retry"] = True
                    else:
                        timings["retry_rejected"] = True
            timings["gen_ms"] = round((time.time() - t0) * 1000, 1)

            if not answer:
                answer = extractive_answer(blocks)
                mode = "extractive"
            else:
                # 数字列表抽取式校正：小模型漏抄/重复数字时用原文句子替换
                answer, patched = complete_numeric_list(answer, context_text, q)
                if patched:
                    timings["numeric_patch"] = True
                # 表格行列举补全（工单6）：列举题漏行时确定性补入（行名取自上下文表格）
                answer, added_rows = complete_row_list(answer, blocks, q)
                if added_rows:
                    timings["row_patch"] = added_rows
                # 书名号名称修复（工单6）：简写/噪声版本 → 片段中最完整的相近名称
                answer, quote_fixes = repair_quotes(answer, context_text)
                if quote_fixes:
                    timings["quote_patch"] = [f"{a}→{b}" for a, b in quote_fixes]
                mode = "rag"
            result = RagResult(question=q, answer=answer, analysis=analysis,
                               contexts=blocks, citations=extract_citations(answer, cmap) or cmap,
                               mode=mode, model=resp.get("model", ""), timings=timings,
                               rewrite=rewrite_info)
        except Exception as exc:  # noqa: BLE001 —— LLM 不可用：抽取式降级
            result = RagResult(question=q, answer=extractive_answer(blocks),
                               analysis=analysis, contexts=blocks, citations=cmap,
                               mode="extractive", rewrite=rewrite_info,
                               timings={**timings, "llm_error": str(exc)})

    timings["total_ms"] = round((time.time() - t_start) * 1000, 1)
    if use_cache:
        _cache_put(cache_key, result)
    return result


def ask_turn(question: str, state: "dialogue.DialogueState",
             use_cache: bool = True, use_rerank: bool = True
             ) -> tuple[RagResult, "dialogue.Turn"]:
    """多轮问答的一轮：改写 → 检索 → 生成 → 回填对话状态。"""
    result = ask(question, use_cache=use_cache, use_rerank=use_rerank, state=state)
    turn = dialogue.Turn(
        idx=state.turn_count + 1,
        raw_question=question,
        resolved_question=result.rewrite.get("resolved", "") or question,
        rewrite_mode=result.rewrite.get("mode", "passthrough"),
        coref_span=result.rewrite.get("span"),
        intent=(result.analysis or {}).get("intent", "fact"),
        lang=(result.analysis or {}).get("lang", "zh"),
        answer=result.answer,
        citations=result.citations,
        timings=result.timings,
    )
    dialogue.append_turn(state, turn)
    return result, turn


def ask_stream(question: str, use_rerank: bool = True,
               top_k: int | None = None,
               exclude_block_types: Sequence[str] | None = None) -> Iterator[dict]:
    """流式问答。事件：
    {"type": "meta", ...} → {"type": "delta", "text": ...} → {"type": "final", "result": RagResult}
    """
    q = (question or "").strip()
    if not q:
        yield {"type": "final", "result": RagResult(question="", answer="请输入问题。")}
        return

    t_start = time.time()
    try:
        hits, analysis, timings = retrieve(q, top_k=top_k, use_rerank=use_rerank,
                                           exclude_block_types=exclude_block_types)
        context_text, cmap, blocks, t_extra = build_qa_context(hits)
        _attach_nearby_tables(cmap, analysis.get("nearby_tables"))
        timings.update(t_extra)
    except Exception as exc:  # noqa: BLE001
        yield {"type": "final",
               "result": RagResult(question=q, answer=f"检索失败：{exc}", mode="error")}
        return

    yield {"type": "meta", "analysis": analysis, "citations": cmap,
           "timings": dict(timings)}

    if not context_text.strip():
        result = RagResult(question=q, answer="根据已有资料无法确定。",
                           analysis=analysis, contexts=blocks, citations=cmap,
                           timings=timings)
        yield {"type": "final", "result": result}
        return

    analysis["lang"] = answer_lang(q)          # 同 ask()：按原始提问定答案语言
    messages = build_messages(q, context_text, analysis["lang"])
    parts: list[str] = []
    t0 = time.time()
    ttft_ms = None
    try:
        for delta in llm.chat_stream(messages):
            if ttft_ms is None:
                ttft_ms = round((time.time() - t_start) * 1000, 1)
            parts.append(delta)
            yield {"type": "delta", "text": delta}
    except Exception as exc:  # noqa: BLE001
        result = RagResult(question=q, answer=extractive_answer(blocks),
                           analysis=analysis, contexts=blocks, citations=cmap,
                           mode="extractive",
                           timings={**timings, "llm_error": str(exc)})
        yield {"type": "final", "result": result}
        return

    timings["gen_ms"] = round((time.time() - t0) * 1000, 1)
    timings["ttft_ms"] = ttft_ms
    timings["total_ms"] = round((time.time() - t_start) * 1000, 1)
    answer = "".join(parts).strip()
    result = RagResult(question=q, answer=answer or extractive_answer(blocks),
                       analysis=analysis, contexts=blocks,
                       citations=extract_citations(answer, cmap) or cmap,
                       mode="rag", timings=timings)
    # 与 ask()（state=None 时）共用同一缓存命名空间：流式路径不接多轮改写，
    # 焦点为 None，故直接复用 _multiturn_cache_key 的同一推导；旧实现写裸问题
    # key，ask() 换成多轮 key 后该条目任何读者都读不到（review F2）。
    cache_key = _multiturn_cache_key(q, None)
    if exclude_block_types:
        cache_key += "|x:" + ",".join(sorted(exclude_block_types))
    _cache_put(cache_key, result)
    yield {"type": "final", "result": result}


# ---------------------------------------------------------------------------
# 预热与状态
# ---------------------------------------------------------------------------
def _ensure_gpu_headroom(need_mb: int = 3600) -> None:
    """加载本地模型前检查显存；不足时请求 Ollama 释放（warmup 末尾会重新预热）。

    本机实测：Ollama 常驻 qwen2.5:3b（~2.4GB）时，新进程加载 bge-reranker-v2-m3
    会因显存竞争触发**进程级段错误**（连 try/except 都来不及执行）。

    阈值口径（2026-10-12 实测收紧）：闸门必须在**加载峰值**前生效，而不是
    按模型稳态大小拍脑袋。原阈值 2800/2000 太松 —— 桌面 + Ollama 占 4.3GB、
    嵌入加载后剩约 2.6GB 仍放行精排加载，结果在第二轮复评时再次段错误（exit 139）。
    现在：
      · 加载嵌入前要求 ≥3600MB（它自己约 1.2GB，留出碎片余量）；
      · 加载精排前要求 ≥3000MB；
      · 任一不满足 → 让 Ollama 卸载模型，再复查一次（仍不足只警告，不阻断 ——
        宁可慢（CPU 兜底）也要给出可解释的行为）。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, _total = torch.cuda.mem_get_info()
        if free / (1024 * 1024) >= need_mb:
            return
        print(f"[warmup] 显存剩余 {free / 1024 / 1024:.0f}MB < {need_mb}MB，"
              f"请求 Ollama 暂时释放…", flush=True)
        from src import llm
        llm.unload()
        torch.cuda.empty_cache()
        time.sleep(2.5)
        free2, _t2 = torch.cuda.mem_get_info()
        if free2 / (1024 * 1024) < need_mb:
            print(f"[warmup] 释放后仍只剩 {free2 / 1024 / 1024:.0f}MB；"
                  f"精排将按显存闸门自动改用 CPU 推理（稳定性优先）", flush=True)
    except Exception:  # noqa: BLE001
        pass


def warmup() -> dict:
    """服务启动预热：embedding / reranker / BM25 / Ollama 全部就绪，避免首题冷启动。

    顺序很重要：**先加载本地模型（embedding/reranker），最后预热 Ollama**，
    并在加载前用 _ensure_gpu_headroom 确保显存干净（见该函数说明）。
    """
    out: dict = {}
    t0 = time.time()
    _ensure_gpu_headroom()
    try:
        from src import embedder
        embedder.get_model()
        out["embedding"] = True
    except Exception as exc:  # noqa: BLE001
        out["embedding"] = f"失败: {exc}"
    # 再查一次：embedding 已占 ~1.5G，reranker 加载前需要足够的连续显存，
    # 否则实测会触发进程级段错误（Windows/CUDA 边界情况，Python 无法捕获）
    _ensure_gpu_headroom(need_mb=3000)
    try:
        reranker.get_model()
        out["reranker"] = True
    except Exception as exc:  # noqa: BLE001
        out["reranker"] = f"失败: {exc}"
    try:
        retriever.bm25_stats()
        out["bm25"] = True
    except Exception as exc:  # noqa: BLE001
        out["bm25"] = f"失败: {exc}"
    # 工单06：全文倒排索引（自研）也需预热 —— 首次构建约 7~8s，
    # 若不预热会计入第一道题的响应时间，破坏 ≤3 秒指标。
    try:
        from src import fulltext
        fulltext.build()
        out["fulltext"] = True
    except Exception as exc:  # noqa: BLE001
        out["fulltext"] = f"失败: {exc}"
    try:
        llm.warmup()
        out["llm"] = True
    except Exception as exc:  # noqa: BLE001
        out["llm"] = f"失败: {exc}"
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def status() -> dict:
    """系统状态（界面侧栏展示）。"""
    from src import embedder, vector_store
    try:
        from src import reranker as rk
        rk_info = rk.info()
    except Exception:  # noqa: BLE001
        rk_info = {}
    return {
        "workorder": config.WORKORDER_ID,
        "vector_store": vector_store.info(),
        "embedding": embedder.info(),
        "reranker": rk_info,
        "llm": llm.info(),
    }
