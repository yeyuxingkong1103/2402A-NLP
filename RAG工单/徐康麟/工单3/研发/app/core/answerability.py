# -*- coding: utf-8 -*-
"""工单3 可答性闸门（设计/接口设计.md §2.4、§3.19 冻结，v1.4/v1.5/v1.6 条款）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

两件事：
    ① ``decide``：五条否条件（空结果 / 低分 / 低覆盖 / 数字缺失 / 主体不匹配）→ 回「不清楚」；
    ② **主体类型闸门**（题 3/题 4 同页互污染的通用解法，禁止单题特判）：
       发行人全称剥离（可逆）→ person 优先 → 问句意图判 organization → 否则 any；
       允许集合由**证据表列语义**推导，再用表列取值在答案文本里做**包含反查**；
       allowed 为空/expected=any/闸门关闭 → **fail-open**（``counted=False``，不得当作「闸门通过」的证据）。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .config import AppConfig, discover_issuer_names, get_config
from .errors import RagError  # noqa: F401 —— 供上层统一 except
from . import language as language_mod
from .text_utils import extract_numbers, keyword_coverage, squash_text, tokenize

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

UNKNOWN_TEXT = "不清楚"

# ---- 主体类型识别用的常量（设计 §3.19 冻结） ----
ORG_SUFFIXES: tuple[str, ...] = ("公司", "企业", "集团", "投资", "贸易", "中心", "研究所", "厂", "银行", "事务所")
PERSON_COLUMN_HINTS: tuple[str, ...] = ("姓名", "股东名称/姓名")
ORG_COLUMN_HINTS: tuple[str, ...] = ("企业名称", "公司名称")
AMBIGUOUS_COLUMN_HINTS: tuple[str, ...] = ("关联方名称",)
PERSON_QUESTION_HINTS: tuple[str, ...] = ("谁", "姓名", "自然人", "法定代表人", "控股股东", "实际控制人")
ORG_INTENT_HINTS: tuple[str, ...] = (
    "哪些企业", "哪几家企业", "企业有哪些", "公司有哪些", "哪些公司", "哪些机构", "关联方企业",
    "存在控制关系的关联方", "不存在控制关系的关联方",
)

# 关系短语（person 三元组之一）
RELATION_PHRASES: tuple[str, ...] = ("控股股东", "实际控制人", "股东", "董事", "监事", "高级管理人员", "配偶", "亲属")


def _lazy_logger(logger: Any, module: str = "answerability") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class AnswerabilityDecision:
    """可答性判定结果（设计 §2.4 冻结字段）。"""

    is_answerable: bool
    reason: str
    top_score: float
    keyword_coverage: float
    numeric_matched: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SubjectGateResult:
    """主体类型闸门结果（``counted=True`` 才算「闸门真正生效并通过」）。"""

    ok: bool
    expected: str
    allowed: tuple[str, ...]
    found: tuple[str, ...]
    leaked: tuple[str, ...]
    reason: str
    counted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 可答性判定
# ---------------------------------------------------------------------------
def has_supporting_numbers(chunks: Sequence[Any], question: str) -> bool:
    """top-3 片段里是否存在可比对数值（数字类问题的否定条件③）。"""
    for chunk in list(chunks)[:3]:
        if extract_numbers(str(getattr(chunk, "content", ""))):
            return True
    return False


# ---------------------------------------------------------------------------
# t15 三条守卫：主体 / 时间 / 谓词（短语存在性 + 同块共现 + 数值锚点）
# ---------------------------------------------------------------------------
# 取核心名词短语时剔除以疑问词/动词/量词为主的词（**保守**：宁可少判，不可误拦）
_QUESTION_STOPWORDS = (
    "根据", "请问", "请", "查询", "介绍", "说明", "指出", "显示", "披露", "记载", "载明",
    "本次", "这次", "该公司", "发行人", "本公司", "公司", "上市公司",
    "的", "了", "吗", "呢", "是", "为", "有", "在", "和", "与", "及", "以及", "或", "对", "所",
    "中", "里", "内", "上", "下", "前", "后", "之间",
    "多少", "哪些", "哪个", "哪一个", "什么", "几位", "几个", "几", "如何", "怎么",
    "用于", "用来", "计划", "拟", "将", "打算", "申请", "安排", "使用", "投资于", "投入",
    "分别是", "分别", "各", "已经", "目前", "现在", "截至", "期内", "报告期", "金额", "数额",
    "招股意向书", "招股说明书", "募集说明书",
)
# 单字功能词（token 层剔除；**否定词保留**，否则「不存在控制关系」会被破坏）
_TOKEN_STOPWORDS = {
    "的", "了", "吗", "呢", "是", "为", "有", "在", "和", "与", "及", "或", "对", "所",
    "中", "里", "内", "上", "下", "前", "后", "之", "以", "等", "该", "本", "并", "而", "则",
    "就", "也", "都", "很", "更", "最", "会", "要", "能", "可", "由", "从", "向", "到", "于",
    "把", "被", "让", "使", "给", "替", "除", "按", "据", "地", "得", "着", "过", "们", "个",
    "什么", "哪些", "哪个", "多少", "如何", "怎么", "请问", "请", "根据", "本次", "这次",
    "分别", "已经", "目前", "现在", "截至", "用于", "用来", "计划", "拟", "将", "打算", "申请",
    "安排", "使用", "投资于", "投入", "金额", "数额", "合计", "共",
    "来自", "涉及", "包括", "属于", "成为", "达到", "超过", "占", "相关", "有关", "是否", "可否",
    "能否", "以下", "如下", "上述", "该等", "以及", "并且", "或者", "其中", "根据", "依据",
    "几家", "几位", "几个", "哪些", "什么", "名称", "情况", "具体", "分别", "具体是", "共",
    "本公司", "贵公司", "该公司", "其", "这", "那", "哪些", "如何",
}
# 「募投语境」词（判据 A 的显式形式，供日志/后续扩展使用）
_ISSUE_CONTEXT_WORDS = ("募集资金投资项目", "募投", "拟投入募集资金", "募集资金运用", "募集资金用途")
_UNIT_NUMBER = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|万股|元|%|％)")
# 「金额/数值型字段」词：同块共现拦截只在这些词出现（或问题被判数值型）时生效。
# 实测（t15）：否则「不存在控制关系的关联方企业共几家」这类**计数题**会被误拦（N-5-③）。
_AMOUNT_FIELD_TERMS = ("资金", "金额", "金额为", "万元", "亿元", "元", "收入", "营业收入", "注册资本",
                       "股数", "股本", "比例", "比重", "费用", "成本", "利润", "资产", "负债", "现金流",
                       "毛利", "占比", "金额是多少")
_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
_ORG_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,20}?(?:股份有限公司|有限责任公司|有限公司)")

_scoped_cache: dict[tuple[str, ...], list[str]] = {}


def scoped_chunk_texts(file_names: Sequence[str] | None, *, cfg: AppConfig | None = None,
                       logger: Any = None) -> list[str]:
    """取**该题限定语料**的 squash 化块文本列表（按文件缓存，只读 SQLite）。

    ``file_names`` 为空 → 全库。守卫刻意在「限定语料」上判定——这正是
    「同一问法在 PDF1 可答、在 PDF2 不可答」（题 207 vs N-4/N-4b）能够区分的机制。
    """
    log = _lazy_logger(logger)
    config = cfg or get_config()
    key = tuple(sorted(str(n) for n in (file_names or [])))
    if key in _scoped_cache:
        return _scoped_cache[key]
    from .pdf_parser import connect_sqlite
    from .text_utils import squash_text

    conn = connect_sqlite(config.paths.index_dir / "rag.sqlite3")
    try:
        if key:
            marks = ",".join("?" for _ in key)
            rows = conn.execute(f"SELECT content FROM chunks WHERE file_name IN ({marks})",
                                list(key)).fetchall()
        else:
            rows = conn.execute("SELECT content FROM chunks").fetchall()
    finally:
        conn.close()
    texts = [squash_text(str(r[0] or "")) for r in rows]
    _scoped_cache[key] = texts
    log.log_event("answerability.scoped_corpus", files=list(key) or "ALL", chunks=len(texts))
    return texts


def _name_overlap(left: str, right: str, *, minimum: int = 4) -> bool:
    """两个名称是否有 ≥``minimum`` 字的公共子串（主体守卫的别名容忍）。"""
    left, right = str(left or ""), str(right or "")
    if not left or not right:
        return False
    for start in range(0, max(0, len(left) - minimum) + 1):
        if left[start:start + minimum] in right:
            return True
    return False


def core_phrases(question: str, *, issuer_names: Sequence[str] | None = None) -> list[str]:
    """抽问句的**实词集合**（jieba token 层去虚词/疑问词后的 ≥2 字实词，最多 8 个）。

    实测教训（t15 标定，三次迭代）：
        ① 逐字替换停用词 → 把「下游」「持股比例」切碎 → 4/14 误伤；
        ② token 合并成短语 → 跨虚词粘连出「来自军用领域收入主营业务收入比重」这类非原文串 → 11/14 误伤；
        ③ **本实现：只做实词 token 级判定**（存在性 + 同块共现），不做短语拼接 → 无粘连问题。
    """
    text = str(question or "")
    for name in (issuer_names or []):
        text = text.replace(str(name), " ")
    try:
        tokens = tokenize(text, min_len=1)      # min_len=1：单字虚词保留，否则虚词被吞、实词粘连
    except Exception as exc:  # noqa: BLE001 —— 分词不可用时显式降级（不静默）
        _lazy_logger(None).log_event("answerability.terms.degrade", level="WARNING",
                                     error_type=type(exc).__name__, message=str(exc),
                                     fallback="退化为按标点切分的粗粒度词面")
        tokens = re.split(r"[^\u4e00-\u9fffA-Za-z0-9%．.]+", text)
    terms: list[str] = []
    for raw in tokens:
        token = str(raw).strip()
        if len(token) < 2 or token in _TOKEN_STOPWORDS or token in _QUESTION_STOPWORDS:
            continue
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token):     # 纯标点/空白
            continue
        if token not in terms:
            terms.append(token)
    return terms[:8]


def guard_evidence(question: str, file_names: Sequence[str] | None, *, expects_numeric: bool = False,
                   cfg: AppConfig | None = None, logger: Any = None) -> tuple[bool, str, dict[str, Any]]:
    """t15 三条守卫（主体 / 时间 / 谓词），返回 ``(ok, reason, detail)``。

    * **主体守卫**：问句点名的公司必须与语料发行人（动态取，禁硬编码）有 ≥4 字重叠；
    * **时间守卫**：问句中的年份必须在该题语料里出现过；
    * **谓词守卫**：核心短语必须**全部存在**且**同块共现**；数值型问题还要求该块含带单位数值（判据 B）。
      实测动机：N-4/N-4b 的「补充流动资金」在 PDF2 存在（p340 银行借款语境），但与「募集资金」
      **不同块共现** → 不可答；题 207 在 PDF1 的 p479/p490 同块共现且含 `15,000 万元` → 可答。
    """
    log = _lazy_logger(logger)
    config = cfg or get_config()
    with log.enter("guard_evidence", {"question": question[:60], "files": list(file_names or []) or "ALL",
                                      "expects_numeric": expects_numeric}) as span:
        detail: dict[str, Any] = {}
        issuers = [str(n) for n in discover_issuer_names(logger=log)]
        scoped_texts = scoped_chunk_texts(file_names, cfg=config, logger=log)
        joined = "".join(scoped_texts)

        # ① 主体守卫
        scoped_issuers = [name for name in issuers if _name_overlap(name, joined, minimum=4)] if file_names \
            else list(issuers)
        asked = _ORG_PATTERN.findall(question)
        subject_hits = [org for org in asked if not any(_name_overlap(org, name, minimum=4) for name in issuers)]
        # reason 用冻结枚举（issuer_mismatch = 主体不在本题语料内；精确原因见日志/details）
        detail.update({"asked_orgs": asked, "scoped_issuers": scoped_issuers, "subject_hits": subject_hits})
        log.log_event("answerability.guard", guard="subject", hits=subject_hits, asked=asked,
                      scoped_issuers=scoped_issuers, result="block" if subject_hits else "pass")
        if subject_hits:
            span.set_output({"ok": False, "reason": "issuer_mismatch", "hits": subject_hits})
            return False, "issuer_mismatch", detail

        # ② 时间守卫
        years = sorted(set(_YEAR_PATTERN.findall(question)))
        missing_years = [y for y in years if y not in joined]
        detail.update({"years": years, "missing_years": missing_years})
        log.log_event("answerability.guard", guard="time", years=years, missing=missing_years,
                      result="block" if missing_years else "pass")
        if missing_years:
            reason = "numeric_missing" if expects_numeric else "low_coverage"
            log.log_event("answerability.guard", guard="time", inner_reason="time_out_of_range",
                          years=years, missing=missing_years, result="block")
            span.set_output({"ok": False, "reason": reason, "hits": missing_years})
            return False, reason, detail

        # ③ 谓词守卫
        phrases = core_phrases(question, issuer_names=issuers)
        missing = [p for p in phrases if not any(p in text for text in scoped_texts)]
        co_occur = bool(phrases) and any(all(q in text for q in phrases) for text in scoped_texts)
        numeric_anchor = any(all(q in text for q in phrases) and _UNIT_NUMBER.search(text)
                             for text in scoped_texts) if phrases else False
        detail.update({"phrases": phrases, "missing_phrases": missing, "co_occur": co_occur,
                       "numeric_anchor": numeric_anchor})
        # 判据 B（数值锚点）实测在「术语散落在正文块、数值在表块」时会造成误伤（题 3），
        # 而 N-4/N-4b 已由共现判据拦下 → B 降级为**仅记录不拦截**（宁少拦不可错拦）。
        amount_question = bool(expects_numeric) or any(term in phrase for phrase in phrases
                                                       for term in _AMOUNT_FIELD_TERMS)
        blocked = bool(missing) or (len(phrases) >= 2 and not co_occur and amount_question)
        log.log_event("answerability.guard", guard="phrase", phrases=phrases, missing=missing,
                      co_occur=co_occur, numeric_anchor=numeric_anchor, expects_numeric=expects_numeric,
                      amount_question=amount_question, inner_reason=("no_corpus_phrase" if missing else
                      ("phrases_not_co_occurring" if blocked else "ok")),
                      result="block" if blocked else "pass")
        if blocked:
            # ``AnswerabilityDecision.reason`` 用 tester 冻结枚举（out_of_corpus / no_numeric_support），
            # 内部细分原因写入 detail 与日志（两套口径并存，便于审计）。
            # 冻结枚举（设计 §2.4）：缺短语 → low_coverage；要金额却无同块支撑 → numeric_missing
            reason = "low_coverage" if missing else "numeric_missing"
            span.set_output({"ok": False, "reason": reason, "phrases": phrases})
            return False, reason, detail
        span.set_output({"ok": True, "phrases": phrases})
        return True, "ok", detail


# ---------------------------------------------------------------------------
# t21（§24）：英文（拉丁）问句的跨语言可答性判据
# ---------------------------------------------------------------------------
# 英文疑问词/虚词/公司后缀/题目常见名词：抽「专名候选」时剔除，
# 避免把 `What` / `the` / `Co.,Ltd.` / `registered capital` 当成实体或专名。
_LATIN_STOPWORDS = frozenset({
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "does", "do", "did", "will", "would",
    "can", "could", "should", "it", "its", "this", "that", "these", "those", "many", "much", "please",
    "tell", "me", "about", "according", "company", "co", "ltd", "limited", "inc", "corp", "corporation",
    "group", "holdings", "stock", "registered", "capital", "legal", "representative", "share", "shares",
    "issued", "issue", "number", "name", "date", "address", "business", "main", "field", "supplier",
    "important", "become", "raised", "raise", "fund", "funds", "investment", "investments", "project",
    "projects", "statement", "prospectus", "plan", "planned", "use", "used", "using", "does",
})
# 连续大写开头片段（≥2 个词）或单个 ≥4 字的拉丁词 —— 作为「专名候选」的粗筛
_LATIN_NAME_PATTERN = re.compile(r"[A-Z][A-Za-z&.\-]{1,}(?:\s+[A-Z][A-Za-z&.\-]*)*")


def latin_name_spans(question: str) -> list[str]:
    """抽英文问句里的**专名候选**（剔除疑问词/虚词/公司后缀词后的连续大写片段，去重保序）。

    实测口径（t21，§24）：``Wuhan Xingtu Xinke Electronics Co., Ltd.`` → ``Wuhan Xingtu Xinke
    Electronics``；``Tesla Inc.`` → ``Tesla``；``FIFA World Cup`` → ``FIFA World Cup``。
    """
    spans: list[str] = []
    for raw in _LATIN_NAME_PATTERN.findall(str(question or "")):
        words = [word.strip(".,&;:") for word in str(raw).split()]
        keep = [word for word in words if word and word.lower() not in _LATIN_STOPWORDS]
        if not keep:
            continue
        span = " ".join(keep).strip()
        if len(squash_text(span)) >= 4 and span not in spans:
            spans.append(span)
    return spans


_latin_scope_cache: dict[tuple[str, ...], list[str]] = {}


def _scoped_latin_names(file_names: Sequence[str] | None, *, cfg: AppConfig | None = None,
                        logger: Any = None) -> list[str]:
    """该题语料里出现过的**拉丁实体名候选**（多词大写片段；按文件集合缓存）。

    用途：判断「语料是否提供过拉丁实体名」。只有提供了，才允许用「问题点名的拉丁实体不在语料」
    作为拒答依据——否则 `招股说明书2.pdf` 这类**没有拉丁公司名**的语料会把正确的英文问题一并拒掉。
    """
    key = tuple(sorted(str(n) for n in (file_names or [])))
    if key in _latin_scope_cache:
        return _latin_scope_cache[key]
    # 注意：必须用**未 squash** 的原文 —— `scoped_chunk_texts()` 返回的是 squash 化文本（全小写、无空格），
    # 在其上抽「首字母大写的拉丁专名」永远是 0 条（t21 实测踩过这个坑：实体判定被整体跳过）。
    log = _lazy_logger(logger)
    config = cfg or get_config()
    from .pdf_parser import connect_sqlite

    conn = connect_sqlite(config.paths.index_dir / "rag.sqlite3")
    try:
        if key:
            marks = ",".join("?" for _ in key)
            rows = conn.execute(f"SELECT content FROM chunks WHERE file_name IN ({marks})", list(key)).fetchall()
        else:
            rows = conn.execute("SELECT content FROM chunks").fetchall()
    finally:
        conn.close()
    haystack = "\n".join(str(row[0] or "") for row in rows)
    names = sorted({span for span in latin_name_spans(haystack) if len(squash_text(span)) >= 8})
    _latin_scope_cache[key] = names
    log.log_event("answerability.scoped_latin", files=list(key) or "ALL", chunks=len(rows), names=len(names))
    return names


def latin_relevance_ok(question: str, chunks: Sequence[Any], *, file_names: Sequence[str] | None = None,
                       cfg: AppConfig | None = None, logger: Any = None) -> tuple[bool, str, dict[str, Any]]:
    """英文问句的**跨语言相关性**判据（替代中文词面覆盖率），返回 ``(ok, reason, detail)``。

    为什么不复用中文覆盖率：``keyword_coverage`` 是**词面 token 覆盖率**，英文 query 与中文语料
    天然无词面交集 → 恒低于阈值（实测 0.33~0.50，阈值 0.45）→ 全部英文问题被误判 ``low_coverage``
    （这正是本次回归的根因）。英文侧改用**语言无关**的三层判据（词汇映射表在 `language.py`，
    只做领域通用词映射，**不针对任何具体问句写死答案**）：

    ① **实体落地**：问句点名的拉丁专名若既不在**证据**里、也不在**该题语料**里，且该题语料**确实
       提供过拉丁实体名** → ``issuer_mismatch``（与中文主体守卫同源，只是跨语言）。
       语料没有拉丁实体名时（如 `招股说明书2.pdf`）不因「核不出」而拒答 → 继续走 ②。
    ② **主题落地**：英文问句经 ``EN_ZH_TOPIC_TERMS`` 映射到语料词汇，映射到的中文词至少有一个
       **出现在 top-3 证据块**里（等价于中文侧的覆盖率判据，但用领域词而非字面 token）；
       一个词都映射不到（非典型问法）→ 退到 ③。
    ③ **向量底线**（仅 ② 无映射词时启用）：top-3 最大向量相似度 ≥ ``cfg.answer.min_vector_similarity_en``
       （bge-m3 多语言向量；实测英文正样本 0.52~0.57）。

    **中文路径不使用本函数；中文阈值（``min_keyword_coverage`` = 0.45）一字未改。**
    """
    log = _lazy_logger(logger)
    config = cfg or get_config()
    chunk_texts = [str(getattr(c, "content", "")) for c in list(chunks)[:3]]
    chunk_squashed = [squash_text(text) for text in chunk_texts]
    spans = latin_name_spans(question)
    topic_terms = language_mod.match_corpus_topic_terms(question)
    vectors = [float(v) for v in (getattr(c, "vector_score", None) for c in list(chunks)[:3])
               if v is not None]
    vector_max = max(vectors) if vectors else None
    detail: dict[str, Any] = {"latin_spans": spans,
                              "topic_terms": [phrase for phrase, _terms in topic_terms],
                              "vector_max": round(vector_max, 4) if vector_max is not None else None,
                              "min_vector_similarity_en": float(config.answer.min_vector_similarity_en)}
    # ① 实体落地（跨语言主体守卫；语料没有拉丁实体名时不做否决）
    if spans:
        needles = [squash_text(span) for span in spans]
        in_evidence = [span for span, needle in zip(spans, needles)
                       if needle and any(needle in text for text in chunk_squashed)]
        primary_file = [str(getattr(chunks[0], "file_name", ""))] if chunks else []
        evidence_files = primary_file or list(file_names or [])
        scope_latin = _scoped_latin_names(evidence_files or file_names, cfg=config, logger=log)
        scope_join = squash_text(" ".join(scope_latin))
        in_scope = [span for span, needle in zip(spans, needles) if needle and needle in scope_join]
        # 词级容忍：跨语言实体名常只对上一部分（如 `Wuhan Liyuan Information Technology` vs 语料里的
        # 另一处 `Wuhan ...`）——只要问句专名里有一个 ≥4 字的**非停用词**出现在该题语料的拉丁文本里，
        # 就视为「实体可落地」，避免把本体名写法不全相同的正确问题误拒。
        span_tokens = [token for span in spans for token in str(span).split()
                       if len(token) >= 4 and token.lower() not in _LATIN_STOPWORDS]
        token_hits = [token for token in span_tokens if squash_text(token) in scope_join]
        detail["latin_token_hits"] = token_hits
        detail.update({"latin_in_evidence": in_evidence, "latin_in_scope": in_scope,
                       "scope_latin_names": len(scope_latin)})
        log.log_event("answerability.latin_relevance", mode="entity", spans=spans,
                      in_evidence=in_evidence, in_scope=in_scope, scope_latin=len(scope_latin),
                      token_hits=token_hits,
                      result="pass" if (in_evidence or in_scope or token_hits)
                      else ("block" if scope_latin else "skip"))
        if in_evidence or in_scope or token_hits:
            return True, "ok", detail
        if scope_latin:
            # 语料提供过拉丁实体名，但问题点名的实体都不在其列 → 超语料实体（Tesla / FIFA 之类）
            return False, "issuer_mismatch", detail
    # ② 主题落地（问题主题必须真的落在证据里）
    if topic_terms:
        # t22 加固：① 主题词必须出现在 **top-2** 证据块（原为 top-3，映射过宽会让「股票期权归属」这类
        # 只沾主题边的问题也放行）；② 映射到 ≥2 个主题短语时要求**过半命中**（避免只沾一个词就通过）。
        top2_texts = chunk_texts[:2]
        grounded = [(phrase, term) for phrase, terms in topic_terms
                    for term in terms if any(term in text for text in top2_texts)]
        grounded_phrases = {phrase for phrase, _term in grounded}
        half_ok = len(topic_terms) < 2 or len(grounded_phrases) * 2 >= len(topic_terms)
        detail["grounded_terms"] = [f"{phrase}→{term}" for phrase, term in grounded]
        detail["topic_phrases"] = len(topic_terms)
        detail["grounded_phrases"] = sorted(grounded_phrases)
        detail["half_ok"] = half_ok
        passed = bool(grounded) and half_ok
        log.log_event("answerability.latin_relevance", mode="topic",
                      phrases=[phrase for phrase, _terms in topic_terms],
                      grounded=sorted(grounded_phrases), half_ok=half_ok,
                      result="pass" if passed else "block")
        return (True, "ok", detail) if passed else (False, "low_coverage", detail)
    # ③ 向量底线兜底（问句既无实体、也没映射到语料词汇）
    if vector_max is None:
        log.log_event("answerability.latin_relevance", mode="vector", vector_max=None, result="degrade-pass",
                      note="无向量分可用（嵌入降级）→ 不因缺分而拒答，交由后续引用校验兜底")
        return True, "ok", detail
    ok = vector_max >= float(config.answer.min_vector_similarity_en)
    log.log_event("answerability.latin_relevance", mode="vector", vector_max=round(vector_max, 4),
                  threshold=float(config.answer.min_vector_similarity_en), result="pass" if ok else "block")
    return (True, "ok", detail) if ok else (False, "low_coverage", detail)


def guard_evidence_latin(question: str, file_names: Sequence[str] | None, chunks: Sequence[Any], *,
                         expects_numeric: bool = False, cfg: AppConfig | None = None,
                         logger: Any = None) -> tuple[bool, str, dict[str, Any]]:
    """英文（拉丁）问句的守卫（t21，§24）：年份存在 + 专名落地 + （要数值时）证据含数值。

    与中文三条守卫（``guard_evidence``）的对应关系（**中文守卫一字未改**）：

    * **年份守卫**：完全同一判据（年份是语言无关字面量）→ N-GEN 类「超范围年份」仍被拦；
    * **专名守卫**：中文版「中文公司名 ≥4 字重叠」↔ 英文版「拉丁专名出现在该题语料」；
    * **数值守卫**：中文版「中文实词同块共现」↔ 英文版「要数字时 top-3 证据里存在可比对数值」
      （``has_supporting_numbers``，语言无关）。

    因此英文侧既不会被「无词面交集」的判据误杀，也保留了「超语料实体 / 超范围年份 / 无数值支撑」三类拒答。
    """
    log = _lazy_logger(logger)
    config = cfg or get_config()
    with log.enter("guard_evidence_latin", {"question": question[:60], "files": list(file_names or []) or "ALL",
                                            "expects_numeric": expects_numeric}) as span:
        detail: dict[str, Any] = {}
        scoped_texts = scoped_chunk_texts(file_names, cfg=config, logger=log)
        joined = "".join(scoped_texts)
        # ① 专名守卫：**统一到 `latin_relevance_ok`**（t21 实测教训：本处若重复实现、又用「全语料」判定，
        #    会把 `招股说明书2.pdf`（语料里没有拉丁公司名）的正确英文问题误判成 issuer_mismatch）。
        #    实体落地只判一次，且以「证据所在文件」为范围 → 此处不再重复判。
        detail["latin_spans"] = latin_name_spans(question)
        # ② 时间守卫（t22 加固）：英文问句里的年份必须出现在**本次检索到的证据块**里 ——
        # 只查「该题语料任意处」会被语料里的字符串巧合放行（实测：自造负例「2099 月球项目」因语料
        # 某处恰好含 "2099" 而被判可答）。中文侧判据保持不变。
        years = sorted(set(_YEAR_PATTERN.findall(question)))
        evidence_text = "".join(str(getattr(c, "content", "") or "") for c in chunks)
        missing_years = [y for y in years if y not in evidence_text]
        detail.update({"years": years, "missing_years": missing_years})
        log.log_event("answerability.guard_latin", guard="time", years=years, missing=missing_years,
                      result="block" if missing_years else "pass")
        if missing_years:
            reason = "numeric_missing" if expects_numeric else "low_coverage"
            span.set_output({"ok": False, "reason": reason, "hits": missing_years})
            return False, reason, detail
        # ③ 数值守卫（语言无关）：要数字而 top-3 证据里没有任何可比对数值 → 不可答
        numeric_support = has_supporting_numbers(chunks, question)
        detail["numeric_support"] = numeric_support
        log.log_event("answerability.guard_latin", guard="numeric", expects_numeric=expects_numeric,
                      numeric_support=numeric_support, result="pass")
        if expects_numeric and not numeric_support:
            span.set_output({"ok": False, "reason": "numeric_missing"})
            return False, "numeric_missing", detail
        span.set_output({"ok": True})
        return True, "ok", detail


def decide(question: str, retrieval: Any, *, field_type: str = "other", expects_numeric: bool = False,
           cfg: AppConfig | None = None, logger: Any = None) -> AnswerabilityDecision:
    """五条否条件 → ``AnswerabilityDecision``（reason 枚举冻结）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    chunks = list(getattr(retrieval, "chunks", []) or [])
    with log.enter("decide", {"question": question[:60], "chunks": len(chunks),
                              "field_type": field_type, "expects_numeric": expects_numeric}) as span:
        top_score = float(getattr(chunks[0], "score", 0.0)) if chunks else 0.0
        query_tokens = tokenize(question)
        # T7/t12 修正：覆盖率取**返回块里的最好值**（top-3 内），而不是只看第 1 名。
        # 原因：重排一改，第 1 名换人就会让「同一批证据」的闸门判定翻转（实测题 34：正确块升到第 1 后
        # 覆盖率从 0.79 变成 0.43 被误拒）。判据仍是「返回的检索证据里有没有该问题的关键词」，
        # 负例（超语料问题）在 top-3 里同样低覆盖，仍会被拒。
        coverage_samples = [keyword_coverage(query_tokens, tokenize(str(getattr(c, "content", ""))))
                            for c in chunks[:3]]
        coverage = max(coverage_samples) if coverage_samples else 0.0
        numeric_matched = has_supporting_numbers(chunks, question)
        file_names = getattr(retrieval, "file_names", None)
        # t21（§24）：**语言感知**——拉丁（英文）问句与中文语料无词面交集，中文词面覆盖率与
        # 中文三条守卫对其天然失效（照用会把全部英文问题判成 low_coverage，即本次回归）。
        # 中文路径**逐字未变**（阈值与判据都没动）；英文路径改用语言无关判据（专名落地 / 向量底线 /
        # 年份存在 / 数值支撑），见 latin_relevance_ok 与 guard_evidence_latin。
        latin = language_mod.detect_language(question) == "en"
        latin_detail: dict[str, Any] = {}
        reason = "ok"
        if not chunks:
            reason = "empty_retrieval"
        elif file_names and not any(str(getattr(c, "file_name", "")) in set(file_names) for c in chunks):
            reason = "issuer_mismatch"
        elif top_score < float(config.answer.min_score):
            reason = "low_score"
        elif latin:
            latin_ok, latin_reason, latin_detail = latin_relevance_ok(question, chunks,
                                                                                   file_names=file_names, cfg=config, logger=log)
            if not latin_ok:
                reason = latin_reason
        elif coverage < float(config.answer.min_keyword_coverage):
            reason = "low_coverage"
        elif expects_numeric and not numeric_matched:
            reason = "numeric_missing"
        # t15：三条守卫（主体 / 时间 / 谓词存在性+同块共现）。仅在**既有否条件都通过**后才跑
        # （省开销），任一命中即判不可答 → 上层回「不清楚」且零引用。
        # t21：英文问句走对应的**跨语言守卫**（年份 / 专名 / 数值），中文守卫一字未改。
        guard_detail: dict[str, Any] = {}
        if reason == "ok":
            if latin:
                guard_ok, guard_reason, guard_detail = guard_evidence_latin(
                    question, file_names, chunks, expects_numeric=expects_numeric, cfg=config, logger=log)
            else:
                guard_ok, guard_reason, guard_detail = guard_evidence(
                    question, file_names, expects_numeric=expects_numeric, cfg=config, logger=log)
            if not guard_ok:
                reason = guard_reason
        decision = AnswerabilityDecision(is_answerable=(reason == "ok"), reason=reason,
                                          top_score=round(top_score, 6),
                                          keyword_coverage=round(coverage, 4),
                                          numeric_matched=numeric_matched)
        log.log_event("answerability.decide", is_answerable=decision.is_answerable, reason=reason,
                      top_score=decision.top_score, coverage=decision.keyword_coverage,
                      coverage_samples=[round(c, 4) for c in coverage_samples],
                      numeric_matched=numeric_matched, latin=latin,
                      latin_detail=latin_detail or None, guard=guard_detail or None)
        span.set_output(decision.to_dict())
        return decision


# ---------------------------------------------------------------------------
# 主体类型闸门
# ---------------------------------------------------------------------------
def strip_issuer_names_checked(question: str, issuer_names: Sequence[str] | None) -> tuple[str, bool]:
    """剥离发行人全称并做**可逆校验**（失败返回原文 + False，由调用方 fail-open）。"""
    from .query_understanding import strip_issuer_names

    original = str(question or "")
    stripped = strip_issuer_names(original, issuer_names)
    if not issuer_names:
        return stripped, True
    reversible = any(stripped.replace("该公司", n) == original for n in issuer_names)
    return (stripped if reversible else original), reversible


def classify_subject_expectation(question: str, *, issuer_names: Sequence[str] | None = None,
                                 logger: Any = None) -> str:
    """判定期望主体类型：``person`` / ``organization`` / ``any``。

    顺序固定（§3.19）：先把发行人全称换成「该公司」（纯替换、可逆）→ **person 优先** →
    再按**问句意图**判 organization（**禁止**把「含『公司』」当 organization 判据）→ 否则 ``any``。
    """
    log = _lazy_logger(logger)
    names = list(issuer_names) if issuer_names is not None else discover_issuer_names()
    stripped, _ok = strip_issuer_names_checked(question, names)
    expected = "any"
    for hint in PERSON_QUESTION_HINTS:
        if hint in stripped:
            expected = "person"
            break
    if expected == "any":
        for hint in ORG_INTENT_HINTS:
            if hint in stripped:
                expected = "organization"
                break
    log.log_event("answerability.classify_subject", expected=expected,
                  stripped_digest=stripped[:60], issuer_names=len(names))
    return expected


def column_kind(header: str) -> str:
    """列语义：``organization`` / ``person`` / ``ambiguous`` / ``none``。"""
    text = str(header or "").strip()
    if not text:
        return "none"
    for hint in PERSON_COLUMN_HINTS:
        if hint in text:
            return "person"
    for hint in ORG_COLUMN_HINTS:
        if hint in text:
            return "organization"
    for hint in AMBIGUOUS_COLUMN_HINTS:
        if hint in text:
            return "ambiguous"
    return "none"


def _looks_like_org(value: str) -> bool:
    """值是否像企业（后缀判定；不做 NER）。"""
    text = str(value or "").strip()
    return bool(text) and len(text) >= 3 and text.endswith(ORG_SUFFIXES)


def _table_rows_from_markdown(markdown: str) -> tuple[list[str], list[list[str]]]:
    """从表块 Markdown 还原 (表头, 数据行)（表块正本就存 Markdown）。"""
    lines = [line.strip() for line in str(markdown or "").splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return [], []
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]
    header = cells(lines[0])
    rows = [cells(line) for line in lines[2:]]
    return header, rows


def allowed_set_from_tables(tables: Sequence[Any], expected: str, *,
                            logger: Any = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """由证据表**列语义**推导 (allowed, other) 两个取值集合。

    * ``organization`` → allowed = 企业列取值（或 ambiguous 列中带组织后缀的值）；
    * ``person`` → allowed = 人名列取值（或 ambiguous 列中不带组织后缀的值）；
    * other = 对侧桶取值（用于检测夹带/越界）。
    """
    log = _lazy_logger(logger)
    allowed: list[str] = []
    other: list[str] = []
    for table in tables:
        markdown = getattr(table, "markdown", None)
        if markdown is None:
            markdown = getattr(table, "content", "")
        header, rows = _table_rows_from_markdown(str(markdown))
        if not header:
            continue
        kinds = [column_kind(h) for h in header]
        for row in rows:
            for index, kind in enumerate(kinds):
                if index >= len(row):
                    continue
                value = row[index].strip()
                if not value or value in {"-", "未披露"} or len(value) > 40:
                    continue
                is_org = _looks_like_org(value)
                if kind == "organization":
                    bucket_org = True
                elif kind == "person":
                    bucket_org = False
                elif kind == "ambiguous":
                    bucket_org = is_org
                else:
                    continue
                if bucket_org and expected == "organization":
                    allowed.append(value)
                elif (not bucket_org) and expected == "person":
                    allowed.append(value)
                elif bucket_org:
                    other.append(value)
                else:
                    other.append(value)
    allowed_unique = tuple(dict.fromkeys(allowed))
    other_unique = tuple(v for v in dict.fromkeys(other) if v not in allowed_unique)
    log.log_event("answerability.allowed_set", expected=expected, allowed=len(allowed_unique),
                  other=len(other_unique), tables=len(tables))
    return allowed_unique, other_unique


def check_person_triple(answer_text: str, chunks: Sequence[Any], *, logger: Any = None) -> tuple[bool, str]:
    """期望 person 时要求：自然人 + 数值（持股比例/股数）+ 关系短语 三者齐备。"""
    log = _lazy_logger(logger)
    text = str(answer_text or "")
    has_number = bool(extract_numbers(text))
    has_relation = any(phrase in text for phrase in RELATION_PHRASES)
    has_person = any(cite in text for cite in ("（Mark Zhao）", "(Mark Zhao)")) or bool(
        re.search(r"[\u4e00-\u9fff]{2,4}(?:先生|女士)", text)
    )
    # 人物名不易无 NER 判定 → 放宽为「答案中出现了证据表里的人物值」，由 allowed 反查给出 found；
    # 这里只做「数值 + 关系短语」两项硬校验，第三项（自然人）由 found 非空保证（见 subject_gate）。
    ok = has_number and has_relation
    reason = "ok" if ok else ("missing_number_or_relation")
    log.log_event("answerability.person_triple", ok=ok, reason=reason, has_number=has_number,
                  has_relation=has_relation, has_person_hint=has_person)
    return ok, reason


def subject_gate(question: str, answer_text: str, chunks: Sequence[Any], *,
                 table_lookup: Callable[[str], Any] | None = None,
                 issuer_names: Sequence[str] | None = None,
                 cfg: AppConfig | None = None, logger: Any = None) -> SubjectGateResult:
    """主体类型闸门（题 3/题 4 同页互污染）：六条规则见设计 §3.19。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    with log.enter("subject_gate", {"question": question[:60], "answer_chars": len(str(answer_text)),
                                    "chunks": len(chunks)}) as span:
        if not bool(getattr(config.answer, "enable_subject_gate", True)):
            result = SubjectGateResult(True, "any", (), (), (), "disabled", False)
            log.log_event("answerability.subject_gate", **result.to_dict())
            span.set_output(result.to_dict())
            return result
        names = list(issuer_names) if issuer_names is not None else discover_issuer_names()
        stripped, reversible = strip_issuer_names_checked(question, names)
        expected = classify_subject_expectation(stripped if reversible else question,
                                                issuer_names=names, logger=log)
        if expected == "any":
            result = SubjectGateResult(True, expected, (), (), (), "not_applicable", False)
            log.log_event("answerability.subject_gate", **result.to_dict())
            span.set_output(result.to_dict())
            return result
        tables = [c for c in chunks if str(getattr(c, "type", "")) == "table"]
        if table_lookup is not None:
            for table_id in {str(getattr(c, "table_id", "")) for c in chunks if getattr(c, "table_id", None)}:
                extra = table_lookup(table_id)
                if extra is not None:
                    tables.append(extra)
        allowed, other = allowed_set_from_tables(tables, expected, logger=log)
        if not allowed:
            # fail-open：证据里没有可用主体表 → 不判「不清楚」，且 counted=False（不算闸门通过）
            result = SubjectGateResult(True, expected, (), (), (), "no_evidence_table", False)
            log.log_event("answerability.subject_gate", **result.to_dict())
            span.set_output(result.to_dict())
            return result
        text = str(answer_text or "")
        found = tuple(v for v in allowed if v in text)
        leaked = tuple(v for v in other if v in text)
        if expected == "person":
            triple_ok, triple_reason = check_person_triple(text, chunks, logger=log)
            ok = bool(found) and not leaked and triple_ok
            reason = "ok" if ok else ("person_leaked" if leaked else "triple_incomplete")
        else:
            ok = bool(found) and not leaked
            reason = "ok" if ok else ("org_leaked" if leaked else "triple_incomplete")
        result = SubjectGateResult(ok=ok, expected=expected, allowed=allowed[:20], found=found, leaked=leaked,
                                   reason=reason, counted=True)
        log.log_event("answerability.subject_gate", **result.to_dict())
        span.set_output({"ok": ok, "expected": expected, "found": len(found), "leaked": len(leaked),
                         "reason": reason, "counted": True})
        return result
