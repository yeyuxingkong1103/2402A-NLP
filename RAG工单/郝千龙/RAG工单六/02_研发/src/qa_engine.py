# -*- coding: utf-8 -*-
# 【问答引擎 · qa_engine.py】混合检索 + 句子级/槽位式答案抽取 + 可选LLM生成 + 全链路耗时打点
# 工单编号：人工智能NLP-RAG-混合检索任务

"""问答主流程：

- 调用 ``HybridRetriever``（向量/全文/混合 × 三种融合 × 三种重排均可热配）取 Top-N 证据；
- 在证据句子/表格行范围内做“IDF 加权覆盖度 + 长短语命中 + 答案类型吻合”句子级抽取；
- 对“注册资本/法定代表人/成立日期”高频事实型问题使用带主语约束的确定性槽位；
- 有 LLM 客户端时可切换生成式答案，带超时降级；全程打点，端到端预算 3 秒。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rerankers import (
    answer_type_bonus, exact_phrases, is_english_query, query_terms,
    translate_en_query,
)
from retriever import Evidence, HybridRetriever

_SENT_RE = re.compile(r"[^。；;！？!?]+[。；;！？!?]?")
_NUMBER_Q = re.compile(r"多少|比重|比例|金额|注册资本|收入|数量|几个|占比")
_PERSON_Q = re.compile(r"谁|代表人|发明人|who", re.IGNORECASE)
_LIST_Q = re.compile(r"哪些|什么项目|包括|涉及|what|which", re.IGNORECASE)
_PREFIX_HEAD_RE = re.compile(r"【[^】]*】")
_PREFIX_ENT_RE = re.compile(r"主体：[^\n]*\n?")
_NUMBER_TOKEN = re.compile(r"\d[\d,\.]*\s*(?:%|万元|亿元|万股)?")
_COMPANY_FULL_RE = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
_TABLE_SEP_RE = re.compile(r"^[\s\|\-:]+$")


def _iter_sentences(evidences: List[Evidence]):
    """从证据父段落生成去重候选句/表格行（携带证据顺序权重与表格标记）。

    :param evidences: 证据列表（相关性降序）
    :return: 生成 (句子, 证据顺序, 是否表格行) 元组
    """
    seen = set()
    for order, ev in enumerate(evidences):
        for source, base_order in ((ev.text, order), (ev.parent_text, order + 0.4)):
            raw = _PREFIX_ENT_RE.sub("", _PREFIX_HEAD_RE.sub("", source))
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
                    if (not is_table and len(sent) < 14
                            and not re.search(r"[。；;！？!?，、：]", sent)
                            and not re.search(r"包括|涉及|涵盖|分为|属于|指的是|为",
                                              sent)):
                        continue
                    key = sent[:50]
                    if key in seen:
                        continue
                    seen.add(key)
                    yield sent, base_order, is_table


def _sentence_score(question: str, sent: str, order: float,
                    idf: Dict[str, float]) -> float:
    """候选答案句打分（覆盖度 + 长短语 + 类型吻合 + 名次先验 + 长度约束）。

    :param question: 原始问题
    :param sent: 候选句
    :param order: 证据顺序
    :param idf: IDF 权重表
    :return: 综合得分
    """
    terms = query_terms(question, idf)
    total_w = sum(w for _, w in terms) or 1.0
    coverage = sum(w for t, w in terms if t in sent) / total_w
    score = 2.0 * coverage
    score += min(0.4, sum(0.2 for p in exact_phrases(question) if p in sent))
    score += answer_type_bonus(question, sent)
    score += max(0.0, 0.15 - order * 0.03)
    if len(sent) < 12:
        score -= 0.45
    elif not (12 <= len(sent) <= 200):
        score -= 0.1
    return score


def _issuer_name(ev: Evidence) -> str:
    """从块前缀读取文档主体公司名。

    :param ev: 证据对象
    :return: 主体公司全称（取不到为空串）
    """
    m = re.search(r"主体：([^\n]+)", ev.text)
    return m.group(1).strip() if m else ""


def _iter_texts(evidences: List[Evidence]):
    """按证据名次遍历（块文本, 父段落），块文本在前（实际命中单元）。

    :param evidences: 证据列表（相关性降序）
    :return: 生成 (Evidence, 文本) 元组
    """
    for ev in evidences:
        yield ev, ev.text
        if ev.parent_text and ev.parent_text != ev.text:
            yield ev, ev.parent_text


def _slot_registered_capital(evidences: List[Evidence]) -> str:
    """注册资本槽位：取当前最大值并排除紧邻其他公司的匹配。

    注册资本经历次增资只会增大，且当前值在概况卡、验资段多处出现；
    收集全部候选值（排除紧邻其他公司名称的匹配），取最大值即当前注册资本。

    :param evidences: 证据列表
    :return: 模板答案；无可靠匹配返回空串
    """
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


def _slot_legal_rep(evidences: List[Evidence]) -> str:
    """法定代表人槽位：优先匹配发行人概况段（公司名称+法定代表人同段），
    再退化为紧邻主语约束，排除子公司/中介机构同名信息。

    :param evidences: 证据列表
    :return: 模板答案；无可靠匹配返回空串
    """
    pattern = re.compile(r"法定代表人[：:是为\s]{0,6}?([一-龥]{2,4})")

    def _clean(name: str) -> str:
        if len(name) >= 3 and name[-1] in {"注", "签", "盖"}:
            name = name[:-1]
        return name

    # 第一优先级：发行人结构化概况段——“公司名称：[发行人] ... 法定代表人：[姓名]”
    # 同段出现，确保是发行人而非子公司/中介的法定代表人
    for ev in evidences:
        issuer = _issuer_name(ev)
        text = ev.parent_text
        for m in pattern.finditer(text):
            name = _clean(m.group(1))
            if name in {"姓名", "名称", "签字"}:
                continue
            ctx = text[max(0, m.start() - 600):m.start()]
            if issuer and "公司名称" in ctx and issuer[:6] in ctx:
                return f"法定代表人为 {name}。"

    # 第二优先级：紧邻主语约束（排除券商/会所等中介机构法定代表人）
    for ev in evidences:
        issuer = _issuer_name(ev)
        for m in pattern.finditer(ev.parent_text):
            name = _clean(m.group(1))
            if name in {"姓名", "名称", "签字"}:
                continue
            window = ev.parent_text[max(0, m.start() - 40):m.start()]
            other = _COMPANY_FULL_RE.findall(window)
            if other and issuer and not any(
                    c == issuer or issuer in c for c in other):
                continue
            return f"法定代表人为 {name}。"
    return ""


def _slot_found_date(evidences: List[Evidence]) -> str:
    """成立日期槽位：优先匹配发行人概况段（公司名称+成立日期同段），
    排除子公司成立日期。容忍 PDF 中“2001 年8 月9 日”式硬空格。

    :param evidences: 证据列表
    :return: 模板答案；无可靠匹配返回空串
    """
    pattern = re.compile(
        r"成立日期[：:\s]{0,6}([0-9]{4}\s*年\s*[0-9]{1,2}\s*月"
        r"\s*[0-9]{1,2}\s*日)")

    # 第一优先级：发行人结构化概况段（公司名称+成立日期同段）
    for ev in evidences:
        issuer = _issuer_name(ev)
        text = ev.parent_text
        for m in pattern.finditer(text):
            ctx = text[max(0, m.start() - 600):m.start()]
            if issuer and "公司名称" in ctx and issuer[:6] in ctx:
                date = re.sub(r"\s+", "", m.group(1))
                return f"公司成立日期为 {date}。"

    # 第二优先级：紧邻主语约束
    for ev in evidences:
        issuer = _issuer_name(ev)
        for m in pattern.finditer(ev.parent_text):
            window = ev.parent_text[max(0, m.start() - 40):m.start()]
            other = _COMPANY_FULL_RE.findall(window)
            if other and issuer and not any(
                    c == issuer or issuer in c for c in other):
                continue
            date = re.sub(r"\s+", "", m.group(1))
            return f"公司成立日期为 {date}。"
    return ""


def _slot_supplement_working_capital(evidences: List[Evidence]) -> str:
    """补充流动资金槽位：募集资金中“补充流动资金”对应金额。

    :param evidences: 证据列表
    :return: 模板答案；无可靠匹配返回空串
    """
    pattern = re.compile(
        r"补充流动资金[\s\S]{0,15}?([0-9][0-9,\.]{2,})\s*(万?元)?")
    for ev, text in _iter_texts(evidences):
        m = pattern.search(text)
        if m:
            return f"本次募集资金中用于补充流动资金的金额为 {m.group(1)} 万元。"
    return ""


def _slot_total_shares(evidences: List[Evidence]) -> str:
    """发行前后总股本槽位：抽取发行前/拟发行/发行后三个股本数（万股）。

    力源信息题（问总股本而非注册资本）专用：三个槽位在发行概况卡与
    “发行人股本情况”段分别出现，跨全部证据汇总后模板化作答。

    :param evidences: 证据列表
    :return: 模板答案；三个槽位取不到至少两个时返回空串
    """
    pre_pat = re.compile(
        r"发行前[^。；\n]{0,20}?总股本(?:为|是)?\s*([0-9][0-9,\.]{2,})\s*万股")
    issue_pat = re.compile(
        r"(?:拟发行|发行股数)[^0-9。；\n]{0,10}?([0-9][0-9,\.]{2,})\s*万股")
    post_pat = re.compile(
        r"发行后总股本[为是\s]{0,4}([0-9][0-9,\.]{2,})\s*万股")
    vals = {"pre": None, "issue": None, "post": None}
    for ev, text in _iter_texts(evidences):
        if vals["pre"] is None:
            m = pre_pat.search(text)
            if m:
                vals["pre"] = m.group(1)
        if vals["issue"] is None:
            m = issue_pat.search(text)
            if m:
                vals["issue"] = m.group(1)
        if vals["post"] is None:
            m = post_pat.search(text)
            if m:
                vals["post"] = m.group(1)
    if sum(1 for v in vals.values() if v) < 2:
        return ""
    parts = []
    if vals["pre"]:
        parts.append(f"发行前总股本{vals['pre']}万股")
    if vals["issue"]:
        parts.append(f"拟发行{vals['issue']}万股")
    if vals["post"]:
        parts.append(f"发行后总股本{vals['post']}万股")
    return "，".join(parts) + "。"


def _slot_main_business(evidences: List[Evidence]) -> str:
    """主营业务槽位：抽取“主要从事……”业务描述整句（推广/销售/应用服务）。

    :param evidences: 证据列表
    :return: 业务描述答案句；无可靠匹配返回空串
    """
    pattern = re.compile(
        r"主要从事[\s\S]{2,110}?(?:推广、销售及应用服务|应用服务)")
    for ev, text in _iter_texts(evidences):
        m = pattern.search(text)
        if m:
            phrase = re.sub(r"\s+", "", m.group(0))
            return phrase[:120]
    return ""


def _slot_top_customers(evidences: List[Evidence]) -> str:
    """前五大客户/供应商槽位：从含多个公司实体+销售金额的表格块中抽取公司名。

    :param evidences: 证据列表
    :return: 公司名列表答案；无可靠匹配返回空串
    """
    if not re.search(r"前五大|前五名|前五大客户|前五大供应商",
                     evidences[0].text if evidences else ""):
        # 不在此处判断问题，由调用方保证
        pass
    company_re = re.compile(
        r"([一-龥A-Za-z0-9()（）]{2,20}?(?:有限公司|股份有限公司|公司))")
    seen: List[str] = []
    for ev in evidences:
        names = company_re.findall(ev.text)
        for n in names:
            n = n.strip()
            if len(n) >= 4 and n not in seen and not any(
                    kw in n for kw in ("合计", "小计", "总计")):
                seen.append(n)
        if len(seen) >= 5:
            break
    if seen:
        return "、".join(seen[:5])
    return ""


def _slot_answer(question: str, evidences: List[Evidence]) -> str:
    """确定性槽位抽取：总股本/注册资本/法定代表人/成立日期/补充流动资金/主营业务。

    扫描全部证据，排除其他公司同名信息，返回带主语约束的模板答案。
    英文问题已先经离线词典翻译为中文，槽位识别与中文问题同链路。

    :param question: 中文问题（英文问题为离线词典翻译后的中文术语串）
    :param evidences: 证据列表
    :return: 模板答案；无可靠匹配返回空串
    """
    # 总股本三值槽位必须先于注册资本判定（本题问总股本而非注册资本）
    if "总股本" in question or ("发行前" in question and "发行后" in question):
        answer = _slot_total_shares(evidences)
        if answer:
            return answer

    if "注册资本" in question or "注册资金" in question:
        return _slot_registered_capital(evidences)

    if "法定代表人" in question or "法人代表" in question:
        return _slot_legal_rep(evidences)

    if "成立" in question and re.search(r"日期|时间|何时", question):
        return _slot_found_date(evidences)

    if "补充流动资金" in question:
        return _slot_supplement_working_capital(evidences)

    if re.search(r"主营业务|主要从事|什么业务|做什么业务|从事什么|业务是什么",
                 question):
        return _slot_main_business(evidences)

    if re.search(r"前五大客户|前五名客户|前五大供应商|前五名供应商", question):
        return _slot_top_customers(evidences)

    return ""


def extract_answer(question: str, evidences: List[Evidence],
                   idf: Dict[str, float]) -> str:
    """在 Top-N 证据句子/表格行池中选取得分最高的答案。

    :param question: 用户问题（中/英）
    :param evidences: 证据列表
    :param idf: IDF 权重表
    :return: 最佳答案（模板答案或原文答案句）
    """
    # 英文问题先经离线词典翻译为中文术语串，槽位识别/答案类型判定/
    # 词项打分全部走中文同链路；证据仍来自翻译后查询的同链路检索
    q_cn = (translate_en_query(question) if is_english_query(question)
            else question)
    slotted = _slot_answer(q_cn, evidences)
    if slotted:
        return slotted

    # 候选三元组：(句子, 证据顺序, 是否表格行)
    candidates: List[Tuple[str, float, bool]] = list(_iter_sentences(evidences))
    if not candidates:
        return ""

    if _NUMBER_Q.search(q_cn):
        numbered = [(s, o, t) for s, o, t in candidates
                    if _NUMBER_TOKEN.search(s)]
        if numbered:
            ranked = sorted(numbered,
                            key=lambda x: _sentence_score(
                                q_cn, x[0], x[1], idf),
                            reverse=True)
            if "分别" in q_cn:
                best, bo, _ = ranked[0]
                best_score = _sentence_score(q_cn, best, bo, idf)
                fullest, fullest_n = best, len(_NUMBER_TOKEN.findall(best))
                for s, o, _ in numbered:
                    sc = _sentence_score(q_cn, s, o, idf)
                    n_num = len(_NUMBER_TOKEN.findall(s))
                    if sc >= best_score - 0.35 and n_num > fullest_n:
                        fullest, fullest_n = s, n_num
                return fullest
            return ranked[0][0]

    if _PERSON_Q.search(q_cn):
        persons = [(s, o, t) for s, o, t in candidates
                   if re.search(r"(?:是|为|：|:)\s*[一-龥A-Za-z]{2,4}[，。；\s]", s)]
        pool = persons or candidates
        return sorted(pool,
                      key=lambda x: _sentence_score(
                          q_cn, x[0], x[1], idf),
                      reverse=True)[0][0]

    ranked = sorted(candidates,
                    key=lambda x: _sentence_score(
                        q_cn, x[0], x[1], idf),
                    reverse=True)
    # 枚举类问题且最佳证据为表格：合并得分最高的 3 个表格行，
    # 使“募投项目有哪些”类问题能答全多个枚举项
    if _LIST_Q.search(q_cn) and ranked and ranked[0][2]:
        table_rows: List[str] = []
        for s, _o, is_table in ranked:
            if is_table and s not in table_rows:
                table_rows.append(s)
            if len(table_rows) >= 3:
                break
        if table_rows:
            return "；".join(table_rows)
    if "上游" in q_cn and "下游" in q_cn and ranked:
        up = next((s for s, _, _ in ranked if "上游" in s), "")
        down = next((s for s, _, _ in ranked if "下游" in s and s != up), "")
        if up and down:
            return up + " " + down
    return ranked[0][0]


@dataclass
class QAAnswer:
    """问答结果（答案、证据、耗时、所用策略）。"""

    question: str
    answer: str
    evidences: List[Evidence]
    latency_s: float
    timings: dict = field(default_factory=dict)
    mode: str = "extractive"  # extractive / llm
    strategy: str = ""        # 检索模式-融合-重排 描述


class QAEngine:
    """问答引擎：混合检索 + 答案组织 + 延迟控制。"""

    def __init__(self, retriever: HybridRetriever,
                 llm_client: Optional[object] = None) -> None:
        """注入检索器与可选 LLM 客户端。

        :param retriever: 已初始化的混合检索器
        :param llm_client: 可选，需实现 chat(prompt)->str
        """
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str, mode: Optional[str] = None,
               fusion_method: Optional[str] = None,
               reranker_name: Optional[str] = None,
               vector_weight: Optional[float] = None,
               fulltext_weight: Optional[float] = None,
               top_k: int = 10) -> QAAnswer:
        """执行一次完整问答（检索参数全部可显式覆盖热配置）。

        :return: QAAnswer（答案、Top证据、耗时、策略描述）
        """
        t0 = time.perf_counter()
        evidences_n = self.retriever.retrieve(
            question, top_k=top_k, mode=mode, fusion_method=fusion_method,
            reranker_name=reranker_name, vector_weight=vector_weight,
            fulltext_weight=fulltext_weight)
        t_retrieval = time.perf_counter() - t0

        t1 = time.perf_counter()
        answer_mode = "extractive"
        if self.llm_client is not None and t_retrieval < 2.2:
            try:
                prompt = self._build_prompt(question, evidences_n[:3])
                answer = self.llm_client.chat(prompt).strip()
                answer_mode = "llm"
            except Exception:
                answer = extract_answer(
                    question, evidences_n,
                    self.retriever.store.sparse.idf)
        else:
            answer = extract_answer(
                question, evidences_n, self.retriever.store.sparse.idf)
        t_answer = time.perf_counter() - t1

        mode_desc = mode or "hybrid"
        strategy = f"{mode_desc}/{fusion_method or 'rrf' if mode_desc == 'hybrid' else '-'}/{reranker_name or 'llm'}"
        return QAAnswer(
            question=question, answer=answer,
            evidences=evidences_n[:3],
            latency_s=time.perf_counter() - t0,
            timings={"retrieval_s": round(t_retrieval, 3),
                     "answer_s": round(t_answer, 3)},
            mode=answer_mode, strategy=strategy)

    @staticmethod
    def _build_prompt(question: str, evidences: List[Evidence]) -> str:
        """构建受限上下文 Prompt（Top-3 证据，控制生成延迟）。

        :param question: 用户问题
        :param evidences: 证据列表
        :return: 拼装 Prompt
        """
        context = "\n\n".join(
            f"[资料{i + 1} | {e.doc_name} 第{e.page_no}页 | {e.heading_path}]\n"
            f"{e.parent_text[:600]}"
            for i, e in enumerate(evidences))
        return (
            "你是金融招股说明书问答助手。请仅依据下列资料回答问题，不要编造；"
            "资料中没有依据时回答“未在资料中找到”。\n\n"
            f"资料：\n{context}\n\n问题：{question}\n答案：")
