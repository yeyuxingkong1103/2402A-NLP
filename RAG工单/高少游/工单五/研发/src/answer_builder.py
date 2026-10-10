# -*- coding: utf-8 -*-
"""抽取式答案合成：从检索片段中定位并抽取答案句（零幻觉、可溯源）。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

策略（按答案类型分流）：
    - entity   ：在「法定代表人 / 董事长 / 实际控制人」等字段后抽取实体名
    - amount / ratio / number：优先选取数值密度高、且覆盖查询关键词/同义词的句子
    - list     ：优先选取含顿号枚举的句子
    - 极值型（最多/最少/哪个）：对「X 由 N 个 Y 构成：…」结构做确定性聚合，
      直接给出「X 的 Y 最多，共 N 个：…」的结论式答案
    - figure / table / fact  ：按关键词覆盖率与实体命中综合排序选句

所有答案均来自检索片段原文，不做改写式生成，保证可溯源、零幻觉。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src import config
from src.knowledge_base import KnowledgeBase

_SENT_SPLIT = re.compile(r"(?<=[。！？；;])")
_NUM_RE = re.compile(r"[\d,]+(?:\.\d+)?\s*(?:万元|亿元|%|％|万股|股)?")

# 字段型答案的抽取模式（字段名 → 值）
_FIELD_PATTERNS = {
    "法定代表人": r"法定代表人[：:\s]*([\u4e00-\u9fa5]{2,4})",
    "董事长": r"董事长[：:\s]*([\u4e00-\u9fa5]{2,4})",
    "总经理": r"总经理[：:\s]*([\u4e00-\u9fa5]{2,4})",
    "实际控制人": r"实际控制人[：:\s]*([\u4e00-\u9fa5]{2,4})",
}

# 极值型问题（哪个 X 的 Y 最多 / 有哪些 Y）的枚举结构：X由N个Y构成：A、B、C
_ENUM_RE = re.compile(
    r"([\u4e00-\u9fa5A-Za-z]{2,14})由(\d+)个([\u4e00-\u9fa5]{2,8})构成[：:]([^。；]+)")
_EXTREMUM_RE = re.compile(r"(最多|最少|哪个|哪一个|哪家|哪项)")
_UNIT_RE = re.compile(r"[\u4e00-\u9fa5]{2,6}(?:处|部|中心|委员会|分公司|公司)")

# 数值序列型问题（「…分别是多少」）：句中「分别为 / 分别是 / 依次为」引导的并列数值
_SERIES_SENT_RE = re.compile(r"(合计分别为|分别为|分别是|依次为)")
_SERIES_Q_RE = re.compile(r"(分别|各年|依次|每年|各期)")


@dataclass
class Answer:
    """答案对象。"""
    text: str
    answer_type: str = "fact"
    evidence: List[dict] = field(default_factory=list)   # 溯源证据
    doc: str = ""
    confidence: float = 0.0


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\n+", " ", text or "")
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return [p for p in parts if len(p) >= 6]


def _keyword_score(sent: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    return sum(1 for k in keywords if k in sent) / len(keywords)


def _field_name(analysis) -> str:
    """判断问题是否为字段型，返回字段名（否则空串）。"""
    q = f"{analysis.question or ''} {analysis.core or ''}"
    for field_name in _FIELD_PATTERNS:
        if field_name in q:
            return field_name
    return ""


def _field_answer(kb, ranked, analysis) -> tuple[str, int]:
    """字段型问题的定向抽取（如法定代表人）。

    在候选块中检索「字段名 + 值」模式；若问题带实体（公司名），
    则优先选择同时包含该实体的块，避免抽到保荐人 / 律所等其他主体的同名字段。
    返回（答案文本, 命中块 id）。
    """
    field_name = _field_name(analysis)
    if not field_name:
        return "", -1
    pat = _FIELD_PATTERNS[field_name]
    entities = [e for e in (analysis.entities or []) if len(e) >= 3]
    fallback = ("", -1)
    for r in ranked:
        text = kb.chunks[r.chunk_id].text
        m = re.search(pat, text)
        if not m:
            continue
        if fallback[0] == "":
            fallback = (f"{field_name}为{m.group(1)}。", r.chunk_id)
        if not entities or any(e in text for e in entities):
            return f"{field_name}为{m.group(1)}。", r.chunk_id
    return fallback


def _extremum_answer(kb, ranked, analysis) -> tuple[str, int]:
    """极值/枚举型问题的确定性聚合（如「哪个销售部的销售处最多？有哪些销售处？」）。

    在候选块中扫描「X 由 N 个 Y 构成：A、B、C」结构，按问题中出现的单位词
    （销售处 / 部门 / 委员会 …）筛选，取 N 最大的条目，输出结论式答案。
    返回（答案文本, 命中块 id）。
    """
    q = f"{analysis.question or ''} {analysis.core or ''}"
    if not _EXTREMUM_RE.search(q):
        return "", -1
    # 问题中出现的单位词（销售处 / 部门 / 委员会 …）
    units = [m.group(0) for m in _UNIT_RE.finditer(q)]
    best = None
    for r in ranked:
        text = kb.chunks[r.chunk_id].text
        for m in _ENUM_RE.finditer(text):
            parent, cnt, unit, items = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            if units and not any(u == unit or u in unit or unit in u for u in units):
                continue
            items = items.strip().rstrip("。；;")
            if not items:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, parent, unit, items, r.chunk_id)
    if best is None:
        return "", -1
    cnt, parent, unit, items, cid = best
    return f"{parent}的{unit}最多，共{cnt}个：{items}。", cid


def _series_answer(kb, ranked, analysis) -> tuple[str, int]:
    """数值序列型问题（「…分别是多少」）的定向抽取。

    在候选块中扫描「…分别为 A、B、C」这类并列数值句，优先选取同时覆盖查询
    关键词（如「军用 / 收入」）与同义词（如「国防 / 销售额」）、且数值最多的句子，
    直接输出该原句，避免把整段正文拼进答案。
    返回（答案文本, 命中块 id）。
    """
    q = f"{analysis.question or ''} {analysis.core or ''}"
    if not _SERIES_Q_RE.search(q):
        return "", -1
    if analysis.answer_type not in ("amount", "ratio", "number", "table"):
        return "", -1
    keywords = list(analysis.keywords or [])
    synonyms = list(getattr(analysis, "synonyms", []) or [])
    best = None
    for r in ranked:
        text = kb.chunks[r.chunk_id].text
        for sent in _sentences(text):
            if not _SERIES_SENT_RE.search(sent):
                continue
            nums = _NUM_RE.findall(sent)
            if len(nums) < 2:
                continue
            kcov = _keyword_score(sent, keywords)
            scov = _keyword_score(sent, synonyms)
            # 必须命中问题核心词或同义词，否则可能是别的「分别为」句
            if keywords and kcov == 0 and scov == 0:
                continue
            score = 3.0 * kcov + 1.5 * scov + min(len(nums), 8) / 8.0 + r.score
            if best is None or score > best[0]:
                best = (score, sent, r.chunk_id)
    if best is None:
        return "", -1
    return _trim_series_lead(best[1]), best[2]


# 序列句前导噪声：「（1）行业和客户集中度高 ①行业集中度高的影响 发行人产品…，」
_SERIES_LEAD_RE = re.compile(r"^(.*?)(报告期内[，,]|公司来自|发行人来自)")


def _trim_series_lead(sent: str) -> str:
    """裁掉序列句前的章节/小标题噪声，从「报告期内 / 公司来自」等实质内容起算。"""
    t = _clean_answer_text(sent)
    m = _SERIES_LEAD_RE.match(t)
    if m and len(m.group(1)) < 60:
        t = t[len(m.group(1)):]
    return t.strip()


# ---- 奖项类问题（「…参与的哪个工程荣获了…奖？」） ---------------------------------
_AWARD_Q_RE = re.compile(r"(荣获|获得|授予|夺得).{0,8}奖")
_AWARD_SENT_RE = re.compile(r"(荣获|获得|授予|夺得)[^。]{0,50}(一等奖|二等奖|三等奖|科技进步奖|发明奖)")
_QUOTE_ENG_RE = re.compile(r"[“\"]([^”\"]{4,50}工程)[”\"]")

# 版式噪声：章节标题【…】、公司页眉「XX公司 招股意向书」
_HEADER_RE = re.compile(r"【[^】]{0,60}】")
_BOILER_RE = re.compile(r"(武汉兴图新科电子股份有限公司|武汉力源信息技术股份有限公司)\s*招股意向书")


def _clean_answer_text(text: str) -> str:
    """清理答案中的版式噪声（章节标题、页眉、换行空白），使答案可读。"""
    t = text or ""
    t = _HEADER_RE.sub("", t)
    t = _BOILER_RE.sub("", t)
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s*\n\s*", "", t)
    # PDF 换行被转成空格后，可能把中文词从中间劈开（如「一 体化工程」），需并回
    t = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def _award_answer(kb, ranked, analysis) -> tuple[str, int]:
    """奖项类问题的定向抽取：定位「『XX工程』…荣获…奖」的完整句。"""
    q = f"{analysis.question or ''} {analysis.core or ''}"
    if not _AWARD_Q_RE.search(q):
        return "", -1
    best = None
    for r in ranked:
        text = kb.chunks[r.chunk_id].text
        for sent in _sentences(text):
            if not _AWARD_SENT_RE.search(sent):
                continue
            m = _QUOTE_ENG_RE.search(sent)
            if not m:
                continue
            score = len(m.group(1)) + r.score
            if best is None or score > best[0]:
                best = (score, sent, r.chunk_id)
    if best is None:
        return "", -1
    return _clean_answer_text(best[1]), best[2]


def build_answer(kb: KnowledgeBase, ranked, analysis) -> Answer:
    """依据重排结果合成答案。"""
    if not ranked:
        return Answer(text="未在文档中检索到相关内容，请尝试换一种问法。",
                      answer_type=analysis.answer_type)

    keywords = list(analysis.keywords or [])
    entities = list(analysis.entities or [])
    synonyms = list(getattr(analysis, "synonyms", []) or [])
    atype = analysis.answer_type

    # 1) 字段型问题优先定向抽取
    direct, cid = _field_answer(kb, ranked, analysis)
    if direct:
        ev = _evidence(kb, ranked, [cid])
        return Answer(text=direct, answer_type=atype, evidence=ev,
                      doc=ev[0]["source"] if ev else "", confidence=0.95)

    # 2) 极值/枚举型问题（哪个 X 的 Y 最多 / 有哪些 Y）
    direct, cid = _extremum_answer(kb, ranked, analysis)
    if direct:
        ev = _evidence(kb, ranked, [cid])
        return Answer(text=direct, answer_type=atype, evidence=ev,
                      doc=ev[0]["source"] if ev else "", confidence=0.9)

    # 3) 数值序列型问题（「…分别是多少」）定向抽取原句
    direct, cid = _series_answer(kb, ranked, analysis)
    if direct:
        ev = _evidence(kb, ranked, [cid])
        return Answer(text=direct, answer_type=atype, evidence=ev,
                      doc=ev[0]["source"] if ev else "", confidence=0.92)

    # 3b) 奖项类问题（「…参与的哪个工程荣获了…奖」）定向抽取完整句
    direct, cid = _award_answer(kb, ranked, analysis)
    if direct:
        ev = _evidence(kb, ranked, [cid])
        return Answer(text=direct, answer_type=atype, evidence=ev,
                      doc=ev[0]["source"] if ev else "", confidence=0.9)

    # 4) 候选句池（带块来源与得分）
    cands: List[tuple[float, str, int]] = []
    for r in ranked:
        chunk = kb.chunks[r.chunk_id]
        for sent in _sentences(chunk.text):
            s = (r.score * 1.2
                 + 1.2 * _keyword_score(sent, keywords)
                 + 0.9 * _keyword_score(sent, synonyms)
                 + 0.8 * _keyword_score(sent, entities))
            nums = _NUM_RE.findall(sent)
            if atype in ("amount", "ratio", "number"):
                s += 0.5 * min(len(nums), 4) / 4
            if atype == "list" and ("、" in sent or "包括" in sent):
                s += 0.5
            if atype == "figure" and _ENUM_RE.search(sent):
                s += 0.6
            cands.append((s, sent, r.chunk_id))
    if not cands:
        return Answer(text="未在文档中检索到相关内容。", answer_type=atype)
    cands.sort(key=lambda x: -x[0])

    # 4) 通用：选取前若干句，去重拼接（事实型更克制，避免拼进无关长句）
    max_sent = config.EXTRACTIVE_MAX_SENTENCES
    if atype in ("fact", "figure", "table"):
        max_sent = min(max_sent, 2)
    top = cands[0][0]
    picked: List[str] = []
    for s, sent, _cid in cands:
        if s < top * 0.6:          # 与最佳句差距过大者丢弃，抑制夹带
            break
        if any(sent in p or p in sent for p in picked):
            continue
        picked.append(sent)
        if len(picked) >= max_sent:
            break
    # 数值类问题只保留含数值的句子，避免夹带无关内容
    if atype in ("amount", "ratio", "number"):
        with_num = [p for p in picked if _NUM_RE.search(p)]
        if with_num:
            picked = with_num[:2]

    answer_text = _clean_answer_text("".join(picked))
    if not answer_text:
        answer_text = _clean_answer_text(cands[0][1])
    ev = _evidence(kb, ranked, [c[2] for c in cands[:3]])
    return Answer(text=answer_text, answer_type=atype, evidence=ev,
                  doc=ev[0]["source"] if ev else "",
                  confidence=round(min(0.99, 0.5 + 0.1 * len(picked)), 2))


def _evidence(kb: KnowledgeBase, ranked, chunk_ids: List[int]) -> List[dict]:
    """构造溯源证据（源文件 / 页码 / 片段 / 得分）。"""
    score_map = {r.chunk_id: r.score for r in ranked}
    out: List[dict] = []
    seen = set()
    for cid in chunk_ids:
        if cid < 0 or cid in seen:
            continue
        seen.add(cid)
        c = kb.chunks[cid]
        out.append({
            "chunk_id": cid,
            "source": c.source,
            "page": c.page,
            "kind": c.kind,
            "score": round(float(score_map.get(cid, 0.0)), 4),
            "snippet": c.text[:160].replace("\n", " "),
        })
    return out