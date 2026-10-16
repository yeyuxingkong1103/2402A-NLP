# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单4 轻量重排（无模型、确定性、可解释；按需可选 LLM 重排但默认关闭）。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

为什么自实现：本机 pip 通道不通（设计/环境事实.md §3），没有可用的 cross-encoder 重排模型；
任务书要求「FAISS 向量检索 + BM25 混合 + **轻量重排**」且「单次问答 ≤3 秒」，因此重排必须
**确定性、O(候选数)、无额外网络调用**。

重排特征（每项都可单独解释、写进日志与报告）：

    F1 词法覆盖     问句实词在块中的 idf 加权覆盖率（用 BM25 的 idf，避免高频词占优）
    F2 字段命中     问句字段词（注册资本 / 上游 / 关联方 …）在块中的命中比例
    F3 答案形态证据 该块是否含**能承载该题答案形态**的证据句式：
                      hierarchy_enum  → 「由 N 个…构成：…」
                      numeric_series  → 「分别为 …、… 和 …」
                      single_value    → 「字段：数值+单位」或 表行「字段 … 数值」
                      enum_list       → markdown 表 / 顿号长枚举
                      conclusion      → 「最快 / 负增长 / 重要供应商 / 荣获 / 参与制定」
                      comparison      → 「上游/下游 … 涉及/包括 …」
                      relation        → 「持股比例 + 控股股东/实际控制人」
    F4 模态先验     图像题（组织结构图/图表）偏好 image 块；数值题偏好 table 块
    F5 主体路由     问句点名发行人 → 该文件块加分；其它文件且实体几乎不出现 → 扣分
    F6 极性闸门     「不存在…关联方」类问句若块只讲「存在控制关系」→ 扣分（题 3/4 同页互污染）
    F7 数值邻近     问数字的问句：块中「字段词 + 邻近数字带单位」→ 加分（题 207/543 的表格行）

⚠️ 「轻量重排」不做语义生成，**不允许**在这里编造任何事实；它只调整候选顺序。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .text_utils import extract_numbers, normalize_text, tokenize

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

#: 各特征默认权重（可用 RAG_RETRIEVAL__* 覆盖的实验点，见日志 rerank.features）
DEFAULT_WEIGHTS: dict[str, float] = {
    "lexical": 1.00,       # F1
    "field": 0.60,         # F2
    "shape": 1.20,         # F3
    "modality": 0.35,      # F4
    "routing": 0.50,       # F5
    "polarity": 0.45,      # F6
    "numeric": 0.55,       # F7
    "identity": 0.45,      # F8：「发行人身份字段簇」（注册资本/法定代表人/公司名称 同块出现）
}

#: 「发行人身份字段簇」判断用词（与 answer_composer._IDENTITY_CUES 同源语义）
_IDENTITY_CUES = ("公司名称", "中文名称", "英文名称", "法定代表人", "注册地址", "公司住所",
                  "成立日期", "注册资本", "实收资本", "统一社会信用代码")
#: 问句出现这些字段词时 F8 才生效
_IDENTITY_FIELDS = ("注册资本", "法定代表人", "公司名称", "注册地址", "公司住所", "成立日期", "实收资本")
#: 「重复关系短语」：同一 ≥6 字短语在块内出现 ≥3 次 ⇒ 这是「名称-关系」枚举块（如 7 家关联方）
_REPEAT_MIN_COUNT = 3
_REPEAT_MIN_LEN = 6
#: 序列量纲（与 answer_composer 同口径：money 金额 / percent 占比）
_SERIES_RE = re.compile(r"(分别为|依次为|分别是|各为)([^。；;]{2,240})")

_UNIT_NEAR = re.compile(r"(万元|亿元|万股|元|股|%|倍)")
_ENUM_SPLIT = re.compile(r"[、,，；;]")
_SENT_END = re.compile(r"[。；;！!？?]")


def _lazy_logger(logger: Any, module: str = "reranker") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class RerankFeatures:
    """单块的八维特征（全部 0~1，便于跨题比较与写报告）。"""

    lexical: float = 0.0
    field_term: float = 0.0
    shape: float = 0.0
    modality: float = 0.0
    routing: float = 0.0
    polarity: float = 0.0
    numeric: float = 0.0
    identity: float = 0.0
    score: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 特征基元
# ---------------------------------------------------------------------------
def _text_of(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("content") or chunk.get("text") or "")


def _low(chunk: Mapping[str, Any]) -> str:
    return _text_of(chunk).lower().replace(" ", "")


def lexical_coverage(analysis: Any, text_low: str, idf: Mapping[str, float] | None) -> float:
    """F1：问句实词的 idf 加权覆盖率。"""
    keywords = list(getattr(analysis, "keywords", []) or [])
    if not keywords:
        return 0.0
    total = 0.0
    hit = 0.0
    for kw in keywords:
        key = str(kw).lower().replace(" ", "")
        if not key:
            continue
        weight = float((idf or {}).get(str(kw), 1.0)) or 1.0
        total += weight
        if key in text_low:
            hit += weight
    return (hit / total) if total > 0 else 0.0


def field_coverage(analysis: Any, text_low: str) -> float:
    """F2：字段词命中比例。"""
    terms = list(getattr(analysis, "field_terms", []) or [])
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if str(t).lower().replace(" ", "") in text_low)
    return hits / len(terms)


def shape_evidence(analysis: Any, chunk: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """F3：该块是否含承载目标答案形态的证据句式（返回 0~1 与命中说明）。"""
    shape = str(getattr(analysis, "answer_shape", "generic"))
    text = _text_of(chunk)
    low = _low(chunk)
    notes: dict[str, Any] = {}
    score = 0.0
    if shape == "hierarchy_enum":
        hits = re.findall(r"([\u4e00-\u9fff]{2,12})\s*由\s*(\d+)\s*个[\u4e00-\u9fff]{1,6}构成", text)
        notes["hierarchy_pairs"] = hits[:6]
        score = 1.0 if hits else 0.0
    elif shape == "numeric_series":
        # 量纲敏感：问「占比/比重」时必须命中**百分比**序列；问「金额/收入」时必须命中**金额**序列。
        # 依据（T5 实测）：题 33 问占比，而页 39 的「民用市场收入分别为…万元」是金额序列，
        # 若不分量纲就会被错误排到第 1，导致答案答成金额（判错）。
        want = "percent" if _wants_ratio(analysis) else "money"
        best = 0.0
        found_kinds: list[str] = []
        for m in _SERIES_RE.finditer(text):
            kind = _series_kind(m.group(2)[:60])
            found_kinds.append(kind)
            if kind == want:
                best = 1.0
                break
        if best == 0.0 and found_kinds:
            best = 0.4 if "other" in found_kinds else 0.2   # 有序列但量纲不对 → 弱化，不得压过正确块
        notes.update({"series_kinds": found_kinds[:4], "want": want})
        score = best
    elif shape in {"single_value", "dual_value"}:
        terms = list(getattr(analysis, "field_terms", []) or [])
        hit = 0
        for term in terms:
            idx = low.find(str(term).lower().replace(" ", ""))
            if idx >= 0 and _UNIT_NEAR.search(low[idx:idx + 40]):
                hit += 1
        notes["field_value_pairs"] = hit
        score = 1.0 if hit else (0.3 if _UNIT_NEAR.search(text) else 0.0)
    elif shape == "enum_list":
        rows = text.count("|")
        bullets = len(re.findall(r"(?:^|\n)\s*(?:\d+[、.）)]|[①②③④⑤⑥⑦⑧⑨])", text))
        enum_commas = 0
        for line in _SENT_END.split(text):
            if len(_ENUM_SPLIT.split(line)) >= 3:
                enum_commas += 1
        repeated = _max_repeated_phrase(low)
        notes.update({"table_pipes": rows, "bullets": bullets, "enum_sentences": enum_commas,
                      "repeated_phrase": repeated})
        score = 1.0 if (rows >= 4 or bullets >= 3 or enum_commas >= 1
                        or repeated >= _REPEAT_MIN_COUNT) else 0.0
    elif shape == "conclusion":
        cues = ("最快", "最高", "负增长", "重要供应商", "荣获", "国家科技进步", "参与制定", "主要供应商", "位列第")
        found = [c for c in cues if c in text]
        notes["cues"] = found
        score = 1.0 if found else 0.0
    elif shape == "comparison":
        want = "上游" if "上游" in str(getattr(analysis, "question", "")) else "下游"
        other = "下游" if want == "上游" else "上游"
        own = len(re.findall(rf"{want}[^。；;]{{0,60}}(涉及|包括|为)", text))
        opp = len(re.findall(rf"{other}[^。；;]{{0,60}}(涉及|包括|为)", text))
        notes.update({"own_clause": own, "opposite_clause": opp})
        score = 1.0 if own and own >= opp else (0.3 if own else 0.0)
    elif shape == "relation":
        need = ("持股比例", "控股股东", "实际控制人", "关联方名称")
        found = [c for c in need if c in text]
        notes["relation_cues"] = found
        score = 1.0 if len(found) >= 2 else (0.4 if found else 0.0)
    else:
        score = 0.0
    return score, notes


def _wants_ratio(analysis: Any) -> bool:
    """问句是否问「占比/比重/比例」。"""
    question = str(getattr(analysis, "question", ""))
    has = getattr(analysis, "has", None)
    return bool((callable(has) and has("asks_ratio")) or "比重" in question or "占比" in question or "比例" in question)


def _series_kind(segment: str) -> str:
    """序列量纲：``percent`` / ``money`` / ``other``。"""
    if "%" in segment:
        return "percent"
    if re.search(r"(万元|亿元|万股)", segment):
        return "money"
    return "other"


def _max_repeated_phrase(low: str) -> int:
    """块内「重复关系短语」的最大出现次数（用于识别「名称-关系」枚举块）。

    做法：按标点切成片段后，统计长度 ≥ ``_REPEAT_MIN_LEN`` 的片段出现次数，
    返回最大值。例：``持有公司股份5%以上的股东`` 在 7 家关联方块里出现 5 次。
    """
    pieces = [p for p in re.split(r"[|、,，；;。\n（）()\s]+", low) if len(p) >= _REPEAT_MIN_LEN]
    counter: dict[str, int] = {}
    for piece in pieces:
        counter[piece] = counter.get(piece, 0) + 1
    return max(counter.values()) if counter else 0


def identity_cluster(analysis: Any, text_low: str) -> float:
    """F8：发行人身份字段簇（问身份字段时，同块出现多个身份字段 ⇒ 这才是权威证据块）。

    依据（T5 实测）：题 531「法定代表人是谁」若只按词法召回到「声明页」（p534 全体董事名单），
    答案对但引用页不是权威证据页；p22「（一）发行人基本情况」同块含
    中文名称/英文名称/注册资本/法定代表人/控股股东 → 应优先。
    """
    terms = list(getattr(analysis, "field_terms", []) or [])
    if not any(str(t) in _IDENTITY_FIELDS for t in terms):
        return 0.0
    count = sum(1 for cue in _IDENTITY_CUES if cue in text_low)
    if count >= 3:
        return 1.0
    if count == 2:
        return 0.6
    if count == 1:
        return 0.2
    return 0.0


def modality_prior(analysis: Any, chunk: Mapping[str, Any]) -> float:
    """F4：模态先验（图像题偏好 image、数值题偏好 table）。"""
    shape = str(getattr(analysis, "answer_shape", "generic"))
    question = str(getattr(analysis, "question", ""))
    modality = str(chunk.get("type") or chunk.get("modality") or "text")
    if shape == "hierarchy_enum" or "图" in question:
        if modality == "image":
            return 1.0
        if modality == "table":
            return 0.2
        return 0.0
    if shape in {"numeric_series", "single_value", "dual_value", "enum_list"}:
        if modality == "table":
            return 1.0
        if modality == "image":
            return 0.3
        return 0.1
    if modality == "table":
        return 0.4
    return 0.2


def routing_score(analysis: Any, chunk: Mapping[str, Any]) -> float:
    """F5：主体路由（问句点名发行人）。点名文件 +1；其它文件按实体缺失扣分。"""
    hint = getattr(analysis, "corpus_hint", None)
    if not hint:
        return 0.0
    file_name = str(chunk.get("file_name") or "")
    low = _low(chunk)
    issuer = str(getattr(analysis, "issuer_hint", "") or "").lower().replace(" ", "")
    if file_name.startswith(str(hint)) or ("招股说明书1" in hint and "说明书1" in file_name) \
            or ("招股说明书2" in hint and "说明书2" in file_name):
        return 1.0
    if issuer and issuer in low:
        return 0.3      # 另一文件里提到该主体（交叉引用），弱加分
    return -0.6         # 另一文件且未提及该主体 → 明确扣分（跨文件串答的主要来源）


def polarity_score(analysis: Any, chunk: Mapping[str, Any]) -> float:
    """F6：极性闸门（「不存在控制关系的关联方」vs「存在控制关系的关联方」）。"""
    question = str(getattr(analysis, "question", ""))
    if "不存在" not in question and "没有" not in question:
        return 0.0
    low = _low(chunk)
    if "不存在" in low:
        return 1.0
    if "存在控制关系" in low or "控股股东" in low:
        return -1.0
    return 0.0


def numeric_adjacency(analysis: Any, chunk: Mapping[str, Any]) -> float:
    """F7：数值邻近（问数字时，字段词后 40 字内出现带单位数字）。"""
    if not (getattr(analysis, "has", None) and (analysis.has("asks_numeric") or analysis.has("asks_money"))):
        return 0.0
    low = _low(chunk)
    terms = list(getattr(analysis, "field_terms", []) or [])
    if not terms:
        return 0.5 if _UNIT_NEAR.search(low) else 0.0
    best = 0.0
    for term in terms:
        key = str(term).lower().replace(" ", "")
        start = 0
        while True:
            idx = low.find(key, start)
            if idx < 0:
                break
            window = low[idx + len(key): idx + len(key) + 40]
            if re.search(r"\d[\d,]*(?:\.\d+)?", window) and _UNIT_NEAR.search(window):
                best = 1.0
                break
            start = idx + len(key)
        if best >= 1.0:
            break
    return best


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def score_chunk(analysis: Any, chunk: Mapping[str, Any], *, idf: Mapping[str, float] | None = None,
                weights: Mapping[str, float] | None = None) -> RerankFeatures:
    """计算单块八维特征与加权总分。"""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    text_low = _low(chunk)
    feats = RerankFeatures()
    feats.lexical = lexical_coverage(analysis, text_low, idf)
    feats.field_term = field_coverage(analysis, text_low)
    feats.shape, notes = shape_evidence(analysis, chunk)
    feats.modality = modality_prior(analysis, chunk)
    feats.routing = routing_score(analysis, chunk)
    feats.polarity = polarity_score(analysis, chunk)
    feats.numeric = numeric_adjacency(analysis, chunk)
    feats.identity = identity_cluster(analysis, text_low)
    feats.notes = notes
    feats.score = (
        w["lexical"] * feats.lexical + w["field"] * feats.field_term + w["shape"] * feats.shape
        + w["modality"] * feats.modality + w["routing"] * feats.routing
        + w["polarity"] * feats.polarity + w["numeric"] * feats.numeric
        + w["identity"] * feats.identity
    )
    return feats


def rerank(analysis: Any, candidates: Sequence[Mapping[str, Any]], *,
           idf: Mapping[str, float] | None = None, rrf_weight: float = 1.0,
           feature_weight: float = 1.0, weights: Mapping[str, float] | None = None,
           logger: Any = None) -> list[dict[str, Any]]:
    """对候选块重排：``final = rrf_weight × 归一化 RRF + feature_weight × 特征分``。

    返回**新的**候选列表（按 ``final_score`` 降序），每项附 ``rerank`` 特征明细，
    便于日志与报告逐题解释「为什么这个块排在前面」。
    """
    log = _lazy_logger(logger)
    with log.enter("rerank", {"candidates": len(candidates), "shape": getattr(analysis, "answer_shape", None),
                              "rrf_weight": rrf_weight, "feature_weight": feature_weight}) as span:
        if not candidates:
            span.set_output({"count": 0, "note": "无候选"})
            return []
        rrf_values = [float(c.get("rrf_score") or 0.0) for c in candidates]
        max_rrf = max(rrf_values) or 1.0
        scored: list[dict[str, Any]] = []
        for chunk in candidates:
            feats = score_chunk(analysis, chunk, idf=idf, weights=weights)
            rrf_norm = float(chunk.get("rrf_score") or 0.0) / max_rrf
            final = rrf_weight * rrf_norm + feature_weight * feats.score
            item = dict(chunk)
            item["rrf_norm"] = round(rrf_norm, 6)
            item["rerank"] = feats.to_dict()
            item["final_score"] = round(final, 6)
            scored.append(item)
        scored.sort(key=lambda c: (-c["final_score"], str(c.get("chunk_id"))))
        for rank, item in enumerate(scored, 1):
            item["rank"] = rank
        log.log_event("rerank.done", candidates=len(scored),
                      top=[{"chunk_id": c["chunk_id"], "final": c["final_score"],
                            "shape": c["rerank"]["shape"], "lexical": round(c["rerank"]["lexical"], 3)}
                           for c in scored[:5]])
        span.set_output({"count": len(scored), "top1": scored[0]["chunk_id"],
                         "top1_score": scored[0]["final_score"]})
        return scored
