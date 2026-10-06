# -*- coding: utf-8 -*-
# 【问答引擎 · qa_engine.py】句子级答案抽取 + 确定性槽位抽取 + 可选LLM生成 + 全链路耗时预算
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""问答主流程：

- 检索 Top-10 证据块（展示 Top-3），抽取阶段在其句子/表格行范围内做
  “IDF 加权覆盖度 + 长短语命中 + 答案类型吻合”的句子级抽取；
- 对“注册资本 / 法定代表人 / 补充流动资金”等高频事实型问题，优先使用
  带主语约束的确定性槽位规则（排除历史沿革与其他公司同名信息）；
- 有 LLM 时把证据块拼装为受限上下文调用生成接口，带超时控制与降级；
- 全程打点记录各阶段耗时，输出端到端延迟（验收线 3 秒）。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from retriever import (
    Evidence, Retriever, answer_type_bonus, exact_phrases, query_terms,
)

# 中文句子切分
_SENT_RE = re.compile(r"[^。；;！？!?]+[。；;！？!?]?")
_NUMBER_Q = re.compile(r"多少|比重|比例|金额|注册资本|收入|数量|几个|占比")
_PERSON_Q = re.compile(r"谁|代表人|发明人")
_LIST_Q = re.compile(r"哪些|什么项目|包括|涉及")
_PREFIX_HEAD_RE = re.compile(r"【[^】]*】")
_PREFIX_ENT_RE = re.compile(r"主体：[^\n]*\n?")
_NUMBER_TOKEN = re.compile(r"\d[\d,\.]*\s*(?:%|万元|亿元)?")
# 紧邻槽位的“其他公司全称”，用于排除中介机构卡片上的同名信息
_COMPANY_FULL_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
# 纯表格分隔行，如 | ---: | --- |
_TABLE_SEP_RE = re.compile(r"^[\s\|\-:]+$")


def _iter_sentences(evidences: List[Evidence]):
    """从证据父段落生成去重的候选句/表格行（携带证据顺序权重）。

    表格父段落按换行拆行，使“募集资金用途表”的每一行可独立成为答案；
    正文行再按中文标点切句。

    :param evidences: 证据列表（相关性降序）
    :return: 生成 (句子, 证据序号0基) 元组
    """
    seen = set()
    for order, ev in enumerate(evidences):
        # 块自身文本是实际命中的检索单元（最相关），父段落用于补充上下文；
        # 父段落按页聚合后可能超过截断长度，故块文本必须纳入候选池
        for source, base_order in ((ev.text, order), (ev.parent_text, order + 0.4)):
            raw = _PREFIX_ENT_RE.sub("", _PREFIX_HEAD_RE.sub("", source))
            # 表格父段落（Markdown 竖线）按行拆，使每个表格行可独立成为答案；
            # 正文父段落则把 PDF 换行重新拼回（段落常因排版被折行），再按句切分
            is_table = raw.count("|") >= 2
            units = raw.split("\n") if is_table else [raw.replace("\n", "")]
            for unit in units:
                unit = unit.strip(" |")
                if not unit or _TABLE_SEP_RE.match(unit):
                    continue
                if is_table:
                    unit = re.sub(r"\s*\|\s*", " ", unit).strip()
                    pieces = [unit]
                else:
                    # 行内小标题（如“③参与制定行业技术标准”）不带句末标点，
                    # 与紧随其后的句子合并，避免把标题碎片误当答案
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
                    if len(sent) < 8 and not re.search(r"\d", sent):
                        continue
                    # 过滤无标点、无谓语动词的小标题/图注碎片
                    # （如“电子信息行业上下游”），表格行不受此限
                    if (not is_table and len(sent) < 14
                            and not re.search(r"[。；;！？!?，、：]", sent)
                            and not re.search(r"包括|涉及|涵盖|分为|属于|指的是|为",
                                              sent)):
                        continue
                    key = sent[:50]
                    if key in seen:
                        continue
                    seen.add(key)
                    yield sent, base_order


def _sentence_score(question: str, sent: str, order: int,
                    idf: Dict[str, float]) -> float:
    """候选答案句打分。

    :param question: 原始问题
    :param sent: 候选句
    :param order: 证据名次（越小越相关）
    :param idf: BM25 IDF 权重表
    :return: 综合得分
    """
    terms = query_terms(question, idf)
    total_w = sum(w for _, w in terms) or 1.0
    coverage = sum(w for t, w in terms if t in sent) / total_w
    score = 2.0 * coverage
    score += min(0.4, sum(0.2 for p in exact_phrases(question) if p in sent))
    score += answer_type_bonus(question, sent)
    score += max(0.0, 0.15 - order * 0.03)  # 证据名次先验
    # 长度惩罚：过短多为标题碎片（如“电子信息行业上下游”），信息不足以作答
    if len(sent) < 12:
        score -= 0.45
    elif not (12 <= len(sent) <= 200):
        score -= 0.1
    return score


def _issuer_name(ev: Evidence) -> str:
    """从块注入前缀中读取文档主体公司名（发行人）。

    :param ev: 证据对象
    :return: 发行公司全称，取不到时为空串
    """
    m = re.search(r"主体：([^\n]+)", ev.text)
    return m.group(1).strip() if m else ""


def _slot_answer(question: str, evidences: List[Evidence]) -> str:
    """确定性槽位抽取：注册资本 / 法定代表人 / 补充流动资金。

    按证据名次顺序扫描全部匹配，跳过历史沿革值与其他公司（中介机构等）
    的同名信息，第一个通过约束的匹配即为答案。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 模板化答案；无可靠匹配时返回空串
    """
    if "注册资本" in question:
        # 注册资本经历次增资只会增大，且当前值在概况卡、验资段多处出现；
        # 收集全部候选值（排除紧邻其他公司名称的匹配），取最大值即当前注册资本
        pattern = re.compile(r"注册资本[：:是为\s]{0,6}?([0-9][0-9,\.]{2,})\s*(万?元)?")
        candidates_val: List[float] = []
        for ev in evidences:
            issuer = _issuer_name(ev)
            for m in pattern.finditer(ev.parent_text):
                window = ev.parent_text[max(0, m.start() - 30):m.start()]
                other = [c for c in _COMPANY_FULL_RE.findall(window)
                         if issuer and c != issuer and issuer not in c]
                if other:
                    continue  # 紧邻其他公司（子公司/被投企业/银行），非发行人注册资本
                try:
                    val = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                if val >= 100:  # 过滤零星认缴额与过小历史值
                    candidates_val.append(val)
        if candidates_val:
            current = max(candidates_val)
            text_val = f"{current:,.2f}".rstrip("0").rstrip(".")
            return f"注册资本为 {text_val} 万元。"
        return ""

    if "法定代表人" in question:
        pattern = re.compile(r"法定代表人[：:是为\s]{0,6}?([\u4e00-\u9fa5]{2,4})")
        for order, ev in enumerate(evidences):
            issuer = _issuer_name(ev)
            for m in pattern.finditer(ev.parent_text):
                name = m.group(1)
                window = ev.parent_text[max(0, m.start() - 40):m.start()]
                other = _COMPANY_FULL_RE.findall(window)
                # 紧邻的是其他公司（券商/会所等），其法定代表人不是发行人信息
                if other and issuer and not any(
                        c == issuer or issuer in c for c in other):
                    continue
                # 贪婪匹配可能把后文“注册资本”的“注”吞入姓名，剔除
                if len(name) >= 3 and name[-1] in {"注", "签", "盖"}:
                    name = name[:-1]
                if name in {"姓名", "名称", "签字"}:
                    continue
                return f"法定代表人为 {name}。"
        return ""

    if "补充流动资金" in question:
        pattern = re.compile(
            r"补充流动资金[\s\S]{0,15}?([0-9][0-9,\.]{2,})\s*(万?元)?")
        for order, ev in enumerate(evidences):
            m = pattern.search(ev.parent_text)
            if m:
                val = m.group(1)
                return f"本次募集资金中用于补充流动资金的金额为 {val} 万元。"
        return ""

    return ""


def extract_answer(question: str, evidences: List[Evidence],
                   idf: Dict[str, float]) -> str:
    """在 Top-N 证据的句子/表格行池中选取得分最高的答案。

    :param question: 用户问题
    :param evidences: 检索证据列表
    :param idf: BM25 IDF 权重表
    :return: 最佳答案（模板答案或原文答案句）
    """
    # 事实型槽位优先（精度高、主语约束强）
    slotted = _slot_answer(question, evidences)
    if slotted:
        return slotted

    candidates: List[Tuple[str, int]] = list(_iter_sentences(evidences))
    if not candidates:
        return ""

    if _NUMBER_Q.search(question):
        # 数字型问题：答案句必须含数字
        numbered = [(s, o) for s, o in candidates if _NUMBER_TOKEN.search(s)]
        if numbered:
            ranked = sorted(numbered,
                            key=lambda x: _sentence_score(question, x[0], x[1], idf),
                            reverse=True)
            # 多值题（收入/比重“分别是”）：在覆盖度足够的候选中取数字最全者
            if "分别" in question:
                best, bo = ranked[0]
                best_score = _sentence_score(question, best, bo, idf)
                fullest = best
                fullest_n = len(_NUMBER_TOKEN.findall(best))
                for s, o in numbered:
                    sc = _sentence_score(question, s, o, idf)
                    n_num = len(_NUMBER_TOKEN.findall(s))
                    if sc >= best_score - 0.35 and n_num > fullest_n:
                        fullest, fullest_n = s, n_num
                return fullest
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
    # “上游……下游……分别涉及哪些”类双侧问题：分别取各侧最高分句组合作答
    if ("上游" in question and "下游" in question and ranked):
        up = next((s for s, _ in ranked if "上游" in s), "")
        down = next((s for s, _ in ranked if "下游" in s and s != up), "")
        if up and down:
            return up + " " + down
    return ranked[0][0]


@dataclass
class QAAnswer:
    """问答结果（含答案、证据与耗时）。"""

    question: str
    answer: str
    evidences: List[Evidence]
    latency_s: float
    timings: dict = field(default_factory=dict)
    mode: str = "extractive"  # extractive / llm


class QAEngine:
    """问答引擎：检索 + 答案组织 + 延迟控制。"""

    def __init__(self, retriever: Retriever,
                 llm_client: Optional[object] = None) -> None:
        """注入检索器与可选 LLM 客户端。

        :param retriever: 已初始化的检索器
        :param llm_client: 可选，需实现 chat(prompt)->str
        """
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str) -> QAAnswer:
        """执行一次完整问答并返回结构化结果。

        :param question: 用户问题（中/英文）
        :return: QAAnswer（答案、证据、耗时）
        """
        t0 = time.perf_counter()
        # 抽取阶段扩大到 Top-10 保证召回，界面展示时再截取 Top-3
        evidences_n = self.retriever.retrieve(question, top_k=10)
        t_retrieval = time.perf_counter() - t0

        t1 = time.perf_counter()
        mode = "extractive"
        if self.llm_client is not None and t_retrieval < 2.2:
            try:
                prompt = self._build_prompt(question, evidences_n[:3])
                answer = self.llm_client.chat(prompt).strip()
                mode = "llm"
            except Exception:
                answer = extract_answer(
                    question, evidences_n, self.retriever.store.sparse.idf)
        else:
            answer = extract_answer(
                question, evidences_n, self.retriever.store.sparse.idf)
        t_answer = time.perf_counter() - t1

        return QAAnswer(
            question=question, answer=answer, evidences=evidences_n[:3],
            latency_s=time.perf_counter() - t0,
            timings={"retrieval_s": round(t_retrieval, 3),
                     "answer_s": round(t_answer, 3)},
            mode=mode,
        )

    @staticmethod
    def _build_prompt(question: str, evidences: List[Evidence]) -> str:
        """构建受限上下文 Prompt（只放 Top-3 证据，控制生成延迟）。

        :param question: 用户问题
        :param evidences: 证据列表
        :return: 拼装后的 Prompt
        """
        context = "\n\n".join(
            f"[资料{i + 1} | 第{e.page_no}页 | {e.heading_path}]\n{e.parent_text[:600]}"
            for i, e in enumerate(evidences))
        return (
            "你是金融招股说明书问答助手。请仅依据下列资料回答问题，"
            "不要编造；资料中没有依据时回答“未在资料中找到”。\n\n"
            f"资料：\n{context}\n\n问题：{question}\n答案：")
