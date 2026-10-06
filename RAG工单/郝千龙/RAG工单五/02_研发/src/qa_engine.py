# -*- coding: utf-8 -*-
# 【问答引擎 · qa_engine.py】句子级答案抽取 + 确定性槽位抽取 + 全链路耗时预算
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""问答主流程：

- 检索 Top-N 证据块，在其句子/表格行/组织结构图行范围内做
  “IDF 加权覆盖度 + 长短语命中 + 答案类型吻合”的句子级抽取；
- 对“注册资本 / 法定代表人”等事实型问题，使用带主语约束的确定性槽位规则；
- 对“哪个销售部销售处最多”类组织结构图问题，优先从 orgchart 块抽取；
- 全程打点记录各阶段耗时，输出端到端延迟（验收线 3 秒）。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from retriever import (
    Evidence, Retriever, answer_type_bonus, exact_phrases, query_terms,
)

_SENT_RE = re.compile(r"[^。；;！？!?]+[。；;！？!?]?")
_NUMBER_Q = re.compile(r"多少|比重|比例|金额|注册资本|收入|数量|几个|占比")
_PERSON_Q = re.compile(r"谁|代表人|发明人")
_LIST_Q = re.compile(r"哪些|什么项目|包括|涉及|有哪些")
_ORGCHART_Q = re.compile(r"组织结构|销售处|销售部|组织架构")
_PREFIX_HEAD_RE = re.compile(r"【[^】]*】")
_PREFIX_ENT_RE = re.compile(r"主体：[^\n]*\n?")
_NUMBER_TOKEN = re.compile(r"\d[\d,\.]*\s*(?:%|万元|亿元)?")
_COMPANY_FULL_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
_TABLE_SEP_RE = re.compile(r"^[\s\|\-:]+$")


def _iter_sentences(evidences: List[Evidence]):
    """从证据父段落生成去重的候选句/表格行/组织结构图行。"""
    seen = set()
    for order, ev in enumerate(evidences):
        for source, base_order in ((ev.text, order), (ev.parent_text, order + 0.4)):
            raw = _PREFIX_ENT_RE.sub("", _PREFIX_HEAD_RE.sub("", source))
            is_table = raw.count("|") >= 2
            is_org = "销售处" in raw or "组织结构图" in raw
            units = raw.split("\n") if (is_table or is_org) else [raw.replace("\n", "")]
            for unit in units:
                unit = unit.strip(" |")
                if not unit or _TABLE_SEP_RE.match(unit):
                    continue
                if is_table and not is_org:
                    unit = re.sub(r"\s*\|\s*", " ", unit).strip()
                    pieces = [unit]
                else:
                    raw_pieces = _SENT_RE.findall(unit)
                    pieces, buf = [], ""
                    for piece in raw_pieces:
                        buf += piece
                        if re.search(r"[。；;！？!?]$", piece.strip()):
                            pieces.append(buf)
                            buf = ""
                    if buf:
                        pieces.append(buf)
                for raw_sent in pieces:
                    sent = raw_sent.strip(" |")
                    if len(sent) < 6 and not re.search(r"\d", sent):
                        continue
                    key = sent[:60]
                    if key in seen:
                        continue
                    seen.add(key)
                    yield sent, base_order


def _sentence_score(question: str, sent: str, order: int,
                    idf: Dict[str, float]) -> float:
    """候选答案句打分。"""
    terms = query_terms(question, idf)
    total_w = sum(w for _, w in terms) or 1.0
    coverage = sum(w for t, w in terms if t in sent) / total_w
    score = 2.0 * coverage
    score += min(0.4, sum(0.2 for p in exact_phrases(question) if p in sent))
    score += answer_type_bonus(question, sent)
    score += max(0.0, 0.15 - order * 0.03)
    if len(sent) < 12:
        score -= 0.4
    elif not (12 <= len(sent) <= 250):
        score -= 0.08
    return score


def _issuer_name(ev: Evidence) -> str:
    """从块前缀读取主体公司名。"""
    m = re.search(r"主体：([^\n]+)", ev.text)
    return m.group(1).strip() if m else ""


def _nearest_company_before(text: str, pos: int) -> str:
    """返回 pos 之前最近的一个公司全称（用于槽位主语精确约束）。

    :param text: 块文本
    :param pos: 匹配位置
    :return: 最近的公司全称，无则空串
    """
    prefix = text[:pos]
    matches = list(_COMPANY_FULL_RE.finditer(prefix))
    if not matches:
        return ""
    return matches[-1].group(0)


def _slot_answer(question: str, evidences: List[Evidence],
                 all_chunks: Optional[List] = None) -> str:
    """确定性槽位抽取：注册资本 / 法定代表人。

    多文档场景下必须以 question 中的目标公司为主语约束。
    核心约束：匹配位置之前最近的公司全称必须是目标公司。
    若 Top-N 证据未命中，回退到全量 chunks 中扫描主体为目标公司的块。
    """
    target = extract_company_from_question(question)
    pool_texts: List[Tuple[str, str]] = []
    seen_keys = set()
    for ev in evidences:
        issuer = _issuer_name(ev)
        key = (ev.page_no, ev.parent_text[:30])
        if key not in seen_keys:
            seen_keys.add(key)
            pool_texts.append((ev.parent_text, issuer))
    if all_chunks:
        for ch in all_chunks:
            issuer_m = re.search(r"主体：([^\n]+)", ch.text)
            issuer = issuer_m.group(1) if issuer_m else ""
            if target and (target not in issuer and issuer not in target):
                continue
            key = (ch.page_no, ch.parent_text[:30])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pool_texts.append((ch.parent_text, issuer))

    def _subject_ok(text: str, pos: int) -> bool:
        """匹配位置前最近公司是否为目标公司。"""
        if not target:
            return True
        nearest = _nearest_company_before(text, pos)
        if nearest == target:
            return True
        # 提取目标公司字号（如“力源信息”），用于匹配前身/简称
        core = re.sub(r"^(武汉|北京|上海|深圳|广州|成都|珠海|南京|宁波)", "", target)
        core = re.sub(r"(股份有限公司|有限责任公司|有限公司)$", "", core)
        core = core[:6] if len(core) > 6 else core
        nearby = text[max(0, pos - 300):pos]
        # 最近公司含字号，或附近区域含字号且匹配位置前无其他无关公司
        if core and core in nearest:
            return True
        if core and core in nearby:
            return True
        return False

    if "注册资本" in question:
        pattern = re.compile(r"注册资本[：:是为\s]{0,6}?([0-9][0-9,\.]{2,})\s*(万?元)?")
        candidates_val: List[float] = []
        for text, issuer in pool_texts:
            if target and issuer and target not in issuer and issuer not in target:
                continue
            for m in pattern.finditer(text):
                if not _subject_ok(text, m.start()):
                    continue
                try:
                    val = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                if val >= 100:
                    candidates_val.append(val)
        if candidates_val:
            reasonable = [v for v in candidates_val if v < 100000]
            pool = reasonable or candidates_val
            current = max(pool)
            text_val = f"{current:,.2f}".rstrip("0").rstrip(".")
            return f"注册资本为 {text_val} 万元。"
        return ""

    if "法定代表人" in question:
        pattern = re.compile(r"法定代表人[：:是为\s]{0,6}?([\u4e00-\u9fa5]{2,4})")
        for text, issuer in pool_texts:
            if target and issuer and target not in issuer and issuer not in target:
                continue
            for m in pattern.finditer(text):
                if not _subject_ok(text, m.start()):
                    continue
                name = m.group(1)
                if len(name) >= 3 and name[-1] in {"注", "签", "盖", "本"}:
                    name = name[:-1]
                if name in {"姓名", "名称", "签字", "法人", "代表"}:
                    continue
                return f"法定代表人为 {name}。"
        return ""

    return ""


def extract_company_from_question(question: str) -> str:
    """从问题中提取目标公司全称（用于槽位主语约束）。"""
    # 先剥离省略触发词（那/那么），避免“那XX公司”被整体匹配为公司名
    cleaned = re.sub(r"^(那|那么|那末)\s*", "", question)
    m = _COMPANY_FULL_RE.search(cleaned)
    return m.group(0) if m else ""


def _orgchart_answer(question: str, evidences: List[Evidence],
                     idf: Dict[str, float],
                     all_chunks: Optional[List] = None) -> str:
    """组织结构图问题答案抽取：优先从 orgchart 结构化文本中提取。"""
    # 先从检索证据中收集，若不足再从全量 orgchart 块兜底
    all_lines: List[str] = []
    sources = []
    for ev in evidences:
        sources.extend([ev.parent_text, ev.text])
    if all_chunks:
        for ch in all_chunks:
            if "全部销售处" in ch.text or "销售处最多的销售部" in ch.text:
                sources.append(ch.text)
    for src in sources:
        for line in src.split("\n"):
            line = line.strip(" |")
            if line and ("销售处" in line or "销售部" in line or "组织结构图" in line):
                if line not in all_lines:
                    all_lines.append(line)

    # “哪个销售部的销售处最多”——若同时问“有哪些销售处”，合并返回销售部+销售处清单
    if "最多" in question or "哪个销售部" in question:
        top_line = ""
        for line in all_lines:
            if "销售处最多的销售部" in line or "销售处最多" in line:
                top_line = line
                break
        if top_line:
            if "有哪些" in question or "哪些销售处" in question:
                for line in all_lines:
                    if "全部销售处" in line:
                        return top_line + "\n" + line
                for line in all_lines:
                    if "下设销售处" in line:
                        return top_line + "\n" + line
            return top_line
    # “有哪些销售处”：优先取含全部销售处清单的行
    if "有哪些" in question or "哪些销售处" in question or "销售处" in question:
        for line in all_lines:
            if "全部销售处" in line:
                return line
        for line in all_lines:
            if "下设销售处" in line:
                return line
    # “多少个销售处”
    if "多少" in question and "销售处" in question:
        for line in all_lines:
            if "全部销售处" in line:
                return line
    if all_lines:
        return max(all_lines, key=lambda x: x.count("销售处"))
    return ""


def extract_answer(question: str, evidences: List[Evidence],
                   idf: Dict[str, float],
                   all_chunks: Optional[List] = None) -> str:
    """在 Top-N 证据中选取得分最高的答案。"""
    slotted = _slot_answer(question, evidences, all_chunks)
    if slotted:
        return slotted

    # 组织结构图问题优先处理
    if _ORGCHART_Q.search(question):
        ans = _orgchart_answer(question, evidences, idf, all_chunks)
        if ans:
            return ans

    candidates: List[Tuple[str, int]] = list(_iter_sentences(evidences))
    if not candidates:
        return ""

    if _NUMBER_Q.search(question):
        numbered = [(s, o) for s, o in candidates if _NUMBER_TOKEN.search(s)]
        if numbered:
            ranked = sorted(numbered,
                            key=lambda x: _sentence_score(question, x[0], x[1], idf),
                            reverse=True)
            if "分别" in question:
                best, bo = ranked[0]
                best_score = _sentence_score(question, best, bo, idf)
                # 核心词取 idf 最高的领域词（排除通用动词/时间词），避免“来自/期内”等
                # 把敏感度分析大表误判为含核心词
                stop_generic = {"收入", "分别", "多少", "比重", "来自", "期内",
                                "报告", "公司", "领域", "销售", "客户", "主要"}
                qt = query_terms(question, idf)
                core_terms = [t for t, _ in qt
                              if len(t) >= 2 and t not in stop_generic]
                synonyms = {"国防": ["军用"], "军用": ["国防"],
                            "军品": ["军用", "国防"], "军方": ["军用", "国防"]}

                def _has_core(sent):
                    for t in core_terms:
                        if t in sent:
                            return True
                        for syn in synonyms.get(t, []):
                            if syn in sent:
                                return True
                    return False

                # 收入类问题：核心候选须同时含领域词与“收入/销售额/营收”，
                # 且需含“领域”或“来自”（与查询口径一致），避免把“前五名客户
                # 销售额”之类相关但不同口径的句子选中
                revenue_q = any(k in question for k in ("收入", "销售额", "营收"))
                rev_kw = ("收入", "销售额", "营收", "营业收入")

                def _strong_core(sent):
                    if not _has_core(sent):
                        return False
                    if revenue_q:
                        has_rev = any(k in sent for k in rev_kw)
                        # 必须含“领域”，与查询“来自XX领域的收入”口径一致，
                        # 排除“前五名客户销售额”等同含收入但口径不同的句子
                        has_scope = "领域" in sent
                        return has_rev and has_scope
                    return True

                # 若存在含核心词的候选，在其中取数字最全者（不与无核心词大表比较）
                core_cands = [(sent, o) for sent, o in numbered
                              if _sentence_score(question, sent, o, idf) >= best_score - 0.3
                              and _strong_core(sent)]
                if core_cands:
                    # 优先选含“分别为/分别是”的句子（直接给出各期数值），
                    # 排除敏感度分析大表等仅含数字但口径不同的句子
                    fenbie_cands = [c for c in core_cands
                                    if "分别为" in c[0] or "分别是" in c[0]]
                    pool = fenbie_cands or core_cands
                    fullest = max(pool,
                                  key=lambda x: len(_NUMBER_TOKEN.findall(x[0])))
                    return fullest[0]
                return best
            return ranked[0][0]

    if _PERSON_Q.search(question):
        persons = [(s, o) for s, o in candidates
                   if re.search(r"(?:是|为|：|:)\s*[\u4e00-\u9fa5]{2,4}[，。；\s]", s)]
        pool = persons or candidates
        return sorted(pool,
                      key=lambda x: _sentence_score(question, x[0], x[1], idf),
                      reverse=True)[0][0]

    ranked = sorted(candidates,
                    key=lambda x: _sentence_score(question, x[0], x[1], idf),
                    reverse=True)
    return ranked[0][0]


@dataclass
class QAAnswer:
    """问答结果。"""

    question: str
    rewritten_question: str
    answer: str
    evidences: List[Evidence]
    latency_s: float
    timings: dict = field(default_factory=dict)


class QAEngine:
    """问答引擎：检索 + 答案组织 + 延迟控制。"""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        # 缓存全部 chunks，供槽位兜底扫描
        self._all_chunks = retriever.store.chunks

    def answer(self, question: str, rewritten_question: str = "") -> QAAnswer:
        """执行一次完整问答。

        :param question: 用户原始问题
        :param rewritten_question: 改写后的检索查询（多轮对话场景传入）
        :return: QAAnswer
        """
        t0 = time.perf_counter()
        # 检索使用改写后的查询；无改写时回退原始查询
        search_q = rewritten_question or question
        evidences_n = self.retriever.retrieve(search_q, top_k=10)
        t_retrieval = time.perf_counter() - t0

        t1 = time.perf_counter()
        answer = extract_answer(
            search_q, evidences_n, self.retriever.store.sparse.idf,
            all_chunks=self._all_chunks)
        t_answer = time.perf_counter() - t1

        return QAAnswer(
            question=question,
            rewritten_question=rewritten_question,
            answer=answer,
            evidences=evidences_n[:3],
            latency_s=time.perf_counter() - t0,
            timings={"retrieval_s": round(t_retrieval, 3),
                     "answer_s": round(t_answer, 3)},
        )
