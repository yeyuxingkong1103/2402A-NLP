# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
工单编号：人工智能NLP-RAG-Query理解优化
         （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
模块：查询改写（意图识别 / 消歧 / 分解与抽象）

工单要求 Query 理解三件事，本模块按"规则优先、LLM 兜底"实现：
  1. 意图识别   —— numeric（数值/比重）、list（列举）、fact（事实）、compare（比较）
  2. 消歧       —— 工单5 起由 `src/coref.py` 基于对话焦点消解（多轮），
                  本模块只做空白规整（工单3 曾在此硬编码替换实体，见下）
  3. 分解与抽象 —— 多疑问结构拆子问题；实体/时间限定词抽取为关键词

历史说明（消歧为何搬走）：工单3 只有一家公司的语料，这里把「该公司/其/公司/发行人」
硬编码替换成武汉兴图新科；工单5 多文档 + 多轮下，该替换会把 Q4「那武汉力源…呢」
错误地拉回兴图新科，故删除，改由 `src/coref.py` 基于对话焦点消解。

延迟约束：规则版零额外耗时（微秒级）；唯一会调 LLM 的改写环节在 `coref.resolve`
（其内部同样规则优先、LLM 兜底），保证端到端 ≤3 秒。

与原查询的关系：
  - `resolved`：消歧后的主查询（仍走混合检索）
  - `expansions`：同义/关键词扩展查询（多路召回，提升 recall）
  - `sub_questions`：子问题（分别检索后 RRF 融合）
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
from functools import lru_cache

from src import config

# 意图关键词
_NUMERIC_WORDS = ("多少", "几", "比重", "占比", "比例", "金额", "数额", "分别是", "增长率", "占比为")
_LIST_WORDS = ("哪些", "有哪", "包括", "主要有哪些")
_COMPARE_WORDS = ("对比", "比较", "区别", "差异", "相比")

# 领域同义/上下位扩展：命中 key → 追加 value 中的检索式
_SYNONYMS: dict[str, list[str]] = {
    "军用领域": ["军用领域 收入", "军工 收入"],
    "主营业务收入": ["主营业务收入", "营业收入 占比"],
    "注册资本": ["注册资本", "股本 总额"],
    "法定代表人": ["法定代表人", "董事长 总经理"],
    "募集资金": ["募集资金 用途", "募集资金投资项目"],
    "补充流动资金": ["补充流动资金", "募集资金 补充流动资金"],
    "技术标准": ["技术标准", "视频指挥系统技术规范"],
    "上游": ["上游 行业 企业", "电子元器件 金属壳体"],
    "下游": ["下游 行业", "应用领域 客户"],
    "重要供应商": ["重要供应商", "供应商 领域"],
    "国家科技进步一等奖": ["国家科技进步一等奖", "科技进步一等奖 工程"],
}


def detect_language(text: str) -> str:
    """中英判定：ASCII 字母占比 > 0.6 视为英文查询。"""
    t = text or ""
    if not t:
        return "zh"
    ascii_letters = sum(1 for c in t if c.isascii() and c.isalpha())
    return "en" if ascii_letters / max(len(t), 1) > 0.6 else "zh"


def disambiguate(text: str) -> str:
    """仅做空白规整。

    历史说明：工单3 这里把「该公司/其/公司/发行人」硬编码替换成「武汉兴图新科」，
    因为当时只有一家公司的语料。工单5 起由 `src/coref.py` 基于对话焦点消解，
    本函数不再做任何实体替换（否则会把 Q4「那武汉力源…呢」错误地拉回兴图新科）。
    """
    return re.sub(r"\s+", " ", (text or "").strip())


def detect_intent(text: str) -> str:
    t = text or ""
    if any(w in t for w in _COMPARE_WORDS):
        return "compare"
    if any(w in t for w in _NUMERIC_WORDS):
        return "numeric"
    if any(w in t for w in _LIST_WORDS):
        return "list"
    return "fact"


def extract_keywords(text: str) -> list[str]:
    """抽取实体/限定词：书名号/引号内容、领域词、年份与数字。

    工单5 起**不再注入公司全称**：知识库收录的是多家公司，实体已不是常量，
    且公司名属路由信息（doc_router 负责），写死进关键词会误导召回。
    """
    t = text or ""
    kws: list[str] = []
    kws += re.findall(r'[《"]([^》"]+)[》"]', t)
    for term in _SYNONYMS:
        if term in t:
            kws.append(term)
    kws += re.findall(r"(?:19|20)\d{2}\s*年(?:度|末)?(?:\s*\d+\s*-\s*\d+\s*月)?", t)
    # 去重保持顺序
    seen: set[str] = set()
    out: list[str] = []
    for k in kws:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def build_expansions(text: str, keywords: list[str]) -> list[str]:
    """生成额外检索式（同义扩展 + 关键词组合），供多路召回。"""
    exps: list[str] = []
    for term, extra in _SYNONYMS.items():
        if term in text:
            exps.extend(extra)
    # 关键词组合式：领域词 + 意图词（如 注册资本 + 多少 已在原句，无需重复）
    if not exps and keywords:
        exps.append(" ".join(keywords))
    # 去重且排除与原句完全相同的
    seen: set[str] = set()
    out: list[str] = []
    for e in exps:
        if e and e != text and e not in seen:
            seen.add(e)
            out.append(e)
    return out[:4]                      # 控制召回路数，保护延迟


_SPLIT_RE = re.compile(r"(?:以及|和|与|、)")


def split_sub_questions(text: str, intent: str) -> list[str]:
    """分解与抽象：仅当出现**多个独立疑问结构**时才拆（避免过度分解）。

    例："...参与制定了哪个技术标准？由谁牵头？" → 两个子问题
    反例："...军用领域的收入分别是多少？"（"分别"指多年份，是单一问题）
    """
    t = (text or "").strip().rstrip("？?")
    parts = [p.strip() for p in re.split(r"[？?]", t) if p.strip()]
    subs = [p + "？" for p in parts if len(p) >= 6]
    if len(subs) >= 2:
        return subs[:3]

    # 单句中"和/以及/与"连接的并列疑问
    if t.count("？") == 0 and ("和" in t or "以及" in t or "与" in t):
        segs = [s.strip() for s in _SPLIT_RE.split(t) if len(s.strip()) >= 8]
        if len(segs) >= 2 and all(len(s) >= 8 for s in segs):
            return [s + "？" for s in segs[:3]]
    return []


@lru_cache(maxsize=config.QUERY_ANALYSIS_CACHE_SIZE)
def analyze_cached(question: str) -> tuple:
    """缓存版（返回 tuple 以便 lru_cache）。见 analyze()。"""
    res = _analyze_impl(question)
    return (res["lang"], res["intent"], res["resolved"],
            tuple(res["keywords"]), tuple(res["expansions"]),
            tuple(res["sub_questions"]), res["mode"])


def _analyze_impl(question: str) -> dict:
    q = (question or "").strip()
    lang = detect_language(q)
    resolved = disambiguate(q)
    intent = detect_intent(resolved)
    keywords = extract_keywords(resolved)
    expansions = build_expansions(resolved, keywords)
    subs = split_sub_questions(resolved, intent)
    return {"lang": lang, "intent": intent, "resolved": resolved,
            "keywords": keywords, "expansions": expansions,
            "sub_questions": subs, "mode": "rule"}


def analyze(question: str, use_cache: bool = True) -> dict:
    """查询理解入口（规则版，零额外延迟）。

    返回：
      {"lang", "intent", "resolved", "keywords", "expansions",
       "sub_questions", "mode"}
    """
    if use_cache:
        (lang, intent, resolved, keywords, expansions, subs,
         mode) = analyze_cached(question or "")
        return {"lang": lang, "intent": intent, "resolved": resolved,
                "keywords": list(keywords), "expansions": list(expansions),
                "sub_questions": list(subs), "mode": mode}
    return _analyze_impl(question)


def core_query(resolved: str) -> str:
    """剥离冗余实体（公司全称/别名），保留查询的核心信息需求。

    为什么需要（本机实测）：
      "武汉兴图新科电子股份有限公司的主要产品包括哪些？" 里公司全称占 19 字，
      向量与 BM25 都被"提及公司名"的块主导（封面、释义表、声明页），
      真正的诉求"主要产品"反而召不回。公司名已由 doc_router 用于路由，
      对"召回什么内容"是零信息量的冗余实体，剥离后检索信号显著变干净。

    工单5：剥离**所有**已知公司全称与别名（不再只针对兴图新科一家），
    否则「武汉力源信息技术股份有限公司注册资本是多少」的长实体依旧压过核心诉求。
    """
    from src import dialogue                   # 延迟导入：避免与 dialogue 的导入顺序耦合
    t = resolved or ""
    for alias, _full in dialogue._ALIASES:     # 长别名在前，避免短别名截断
        t = t.replace(alias, "")
    t = re.sub(r"^[，,、。；：\s]+", "", t).strip()
    return t


def multi_queries(analysis: dict, max_n: int = 3) -> list[str]:
    """由分析结果汇总实际用于召回的查询列表。

    组成：主查询 + **核心查询（去实体）** + 扩展。
    核心查询与主查询等权进入 RRF——实测它常常是长实体查询的唯一有效召回路径。
    """
    qs = [analysis["resolved"]]
    core = core_query(analysis["resolved"])
    if core and len(core) >= 4:
        qs.append(core)
    qs.extend(analysis.get("expansions") or [])
    seen: set[str] = set()
    out: list[str] = []
    for q in qs:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_n]
