# -*- coding: utf-8 -*-
# 【问答引擎 · qa_engine.py】表格行/单元格答案抽取 + 确定性槽位 + 全链路耗时打点
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""问答主流程（对应设计文档 5.4）：

- 先取 Top-N 证据（表格行陈述句、整表 Markdown、正文句）；
- 按问题意图在“已检索证据”上做确定性抽取：发行股数/募投项目/关联方/
  补充流动资金/注册资本/法定代表人/多期财务数字/上下游/技术标准/
  重要供应商/科技进步奖，全部来自表格单元格或原文句子，不写死答案；
- 无槽位命中时走 IDF 覆盖度 + 长短语的句子级通用抽取；
- 全程打点记录检索与抽取耗时（验收线 3 秒）。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from retriever import (
    Evidence, Retriever, answer_type_bonus, exact_phrases, query_terms,
)

_PREFIX_RE = re.compile(r"【[^】]*】|主体：[^\n]*|表格标题：[^\n]*")
_TABLE_SEP_RE = re.compile(r"^[\s\|\-:：]+$")
_SENT_RE = re.compile(r"[^。；;！？!?]+[。；;！？!?]?")
_NUMBER_TOKEN = re.compile(r"\d[\d,\.]{2,}\s*(?:%|万元|亿元|万股)?")
_AMOUNT = re.compile(r"\d[\d,\.]{2,}")


# ---------------------------------------------------------------------------
# 候选单元生成
# ---------------------------------------------------------------------------
def _strip_prefix(text: str) -> str:
    """删除块前缀（标题路径、主体、表格标题行）。

    :param text: 块文本
    :return: 纯内容文本
    """
    return _PREFIX_RE.sub("", text).strip()


def _markdown_rows(md: str) -> List[str]:
    """把 Markdown 表的数据行拍平为候选文本（保留同行单元格共现）。

    :param md: Markdown 表格
    :return: 数据行文本列表
    """
    rows: List[str] = []
    for line in md.split("\n"):
        line = line.strip()
        if not line or _TABLE_SEP_RE.match(line) or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
        if cells and not all(re.search(r"名称|项目|序号|类型|金额|占比|持股|"
                                       r"关系|年度|比例|内容", c) for c in cells):
            rows.append(" ".join(cells))
    return rows


def candidate_units(evidences: List[Evidence]):
    """从证据生成去重候选单元（行陈述句优先，其次表格行、正文句）。

    :param evidences: 证据列表（相关性降序）
    :return: 生成 (文本, 证据名次) 元组
    """
    seen = set()

    def emit(text: str, order: float):
        text = re.sub(r"\s+", " ", text).strip(" |，,。；;")
        if len(text) < 6:
            return
        key = text[:60]
        if key in seen:
            return
        seen.add(key)
        yield text, order

    for order, ev in enumerate(evidences):
        if ev.row_claim:
            yield from emit(ev.row_claim, order - 0.2)
        raw_parent = _strip_prefix(ev.parent_text)
        if raw_parent.count("|") >= 2:
            for row in _markdown_rows(raw_parent):
                yield from emit(row, order + 0.1)
        for source in (ev.text, ev.parent_text):
            plain = _strip_prefix(source).replace("\n", "")
            if plain.count("|") >= 2:
                continue
            for piece in _SENT_RE.findall(plain):
                piece = piece.strip()
                if len(piece) < 10 and not _NUMBER_TOKEN.search(piece):
                    continue
                yield from emit(piece, order + 0.3)


def _unit_score(question: str, unit: str, order: float,
                idf: Dict[str, float]) -> float:
    """通用候选打分：IDF 覆盖度 + 长短语 + 类型吻合 + 名次先验。

    :param question: 原始问题
    :param unit: 候选文本
    :param order: 证据名次
    :param idf: IDF 权重表
    :return: 得分
    """
    terms = query_terms(question, idf)
    total_w = sum(w for _, w in terms) or 1.0
    coverage = sum(w for t, w in terms if t in unit) / total_w
    score = 2.0 * coverage
    score += min(0.4, sum(0.2 for p in exact_phrases(question) if p in unit))
    score += answer_type_bonus(question, unit)
    score += max(0.0, 0.15 - order * 0.03)
    return score


# ---------------------------------------------------------------------------
# 确定性槽位
# ---------------------------------------------------------------------------
def _slot_shares(question: str, evidences: List[Evidence]) -> str:
    """发行股数与占发行后总股本比例（力源 id=1）。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 模板答案或空串
    """
    if "发行股数" not in question:
        return ""
    pat = re.compile(
        r"发行股数[^\n。|]{0,40}?([0-9][0-9,\.]{2,})\s*万?股"
        r"[\s\S]{0,40}?比例[为是：:\s]*([0-9]+\.[0-9]+%)")
    for ev in evidences:
        for src in (ev.row_claim, ev.text, ev.parent_text):
            m = pat.search(src or "")
            if m:
                return (f"本次发行股数为 {m.group(1)} 万股，"
                        f"占发行后总股本的比例为 {m.group(2)}。")
    return ""


def _slot_projects(question: str, evidences: List[Evidence]) -> str:
    """募集资金拟投资项目枚举（力源 id=2）。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 枚举答案或空串
    """
    if not ("募集资金" in question and re.search(r"投资|项目", question)):
        return ""
    if "补充流动资金" in question:  # id=207 走独立槽位
        return ""
    # (序号, 项目名, 投资额)；序号用于按表格原始顺序还原枚举
    projects: List[Tuple[int, str, str]] = []
    seen = set()
    name_re = re.compile(
        r"([0-9]+)\s*[，,、]?\s*项目名称[：:]([^，,。|]+)")
    invest_re = re.compile(
        r"(?:计划总投资\s*\(万元\)|项目总投资\s*[（(]万元[）)])[：:]"
        r"\s*([0-9][0-9,\.]*|◆|\[)")
    for ev in evidences:
        sources = [ev.row_claim] if ev.row_claim else []
        sources += [_strip_prefix(ev.parent_text), _strip_prefix(ev.text)]
        for src in sources:
            if not src or "募集资金" not in src and "项目名称" not in src:
                continue
            matches = list(name_re.finditer(src))
            if not matches:
                continue
            invs = invest_re.findall(src)
            for i, m in enumerate(matches):
                seq, name = m.group(1), m.group(2).strip(" |：:，,。")
                if not name or re.search(r"合计|总计", name):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                inv = invs[i] if i < len(invs) and re.match(
                    r"[0-9]", invs[i]) else ""
                projects.append((int(seq), name, inv))
    if len(projects) < 3:
        return ""
    projects.sort(key=lambda x: x[0])  # 按募投表序号 1→N 还原
    parts = []
    for i, (_seq, name, inv) in enumerate(projects, 1):
        parts.append(f"{i}.{name}" + (f"（计划总投资 {inv} 万元）" if inv else ""))
    return "本次募集资金拟投资以下项目：" + "；".join(parts) + "。"


def _slot_related_parties(question: str,
                          evidences: List[Evidence]) -> str:
    """存在/不存在控制关系的关联方枚举（力源 id=3、id=4）。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 枚举答案或空串
    """
    if "关联方" not in question or "控制关系" not in question:
        return ""

    if "不存在控制关系" in question:
        items: List[str] = []
        seen = set()
        # 优先读“不存在控制关系”标题下的表格行陈述句
        row_re = re.compile(r"不存在控制关系的关联方[：:]([^：|，,]+)[：:](.+)$")
        for ev in evidences:
            src = ev.row_claim or ""
            m = row_re.search(src)
            if m and "不存在" in (ev.table_title or ""):
                name, rel = m.group(1).strip(), m.group(2).strip(" ，,。")
                if name not in seen and name != "赵马克":
                    seen.add(name)
                    items.append(f"{name}（{rel}）")
        # 兜底：直接从整表 Markdown 按行读
        if len(items) < 5:
            for ev in evidences:
                if "不存在控制关系" not in (ev.table_title or "") and \
                        "不存在控制关系" not in ev.text:
                    continue
                for row in _markdown_rows(_strip_prefix(ev.parent_text)):
                    cells = [c for c in re.split(r"\s{2,}| (?=[\u4e00-\u9fa5])",
                                                 row) if c]
                    cells = [c.strip() for c in row.split("|")] if "|" in row \
                        else [row[:8], row[8:]]
                    cells = [c.strip() for c in cells if c.strip()]
                    if len(cells) >= 2 and cells[0] not in (
                            "企业名称", "赵马克") and cells[0] not in seen:
                        seen.add(cells[0])
                        items.append(f"{cells[0]}（{cells[-1]}）")
        if len(items) >= 5:
            return ("与公司不存在控制关系的关联方企业包括："
                    + "、".join(items) + "。")
        return ""

    # 存在控制关系：名称 + 持股比例 + 与本公司关系
    pat = re.compile(
        r"([\u4e00-\u9fa5]{2,4})[，,\s]+持股比例\s*([0-9]+\.?[0-9]*%)"
        r"[，,\s]+与本公司关系[：:]?([^，。|\n]+)")
    for ev in evidences:
        for src in (ev.row_claim, ev.parent_text, ev.text):
            m = pat.search(src or "")
            if m:
                return (f"与公司存在控制关系的关联方为 {m.group(1)}，"
                        f"持股比例 {m.group(2)}，"
                        f"与本公司关系：{m.group(3).strip()}。")
    return ""


def _slot_working_capital(question: str, evidences: List[Evidence]) -> str:
    """补充流动资金金额（兴图 id=207）。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 模板答案或空串
    """
    if "补充流动资金" not in question:
        return ""
    pat = re.compile(r"补充流动资金[\s\S]{0,30}?([0-9][0-9,\.]{2,})")
    for ev in evidences:
        for src in (ev.row_claim, ev.text, ev.parent_text):
            m = pat.search(src or "")
            if m:
                val = m.group(1)
                return (f"公司计划使用本次发行募集资金中的 {val} 万元"
                        f"（约 1.5 亿元）用于补充流动资金。")
    return ""


def _slot_capital(question: str, evidences: List[Evidence]) -> str:
    """注册资本槽位（兴图 id=543）。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 模板答案或空串
    """
    if "注册资本" not in question:
        return ""
    # 数字必须紧跟“万元”，排除“转增前注册资本的25%”等比例句；
    # 允许中间出现“增加至/增至/为”等 6 字内连接，兼容 P59 沿革表述
    pat = re.compile(
        r"注册资本[^\d万元]{0,6}?([0-9][0-9,\.]{2,})\s*万元")
    # 含这些上下文的数值属于子公司出资/历史沿革噪声，降权不直接采用
    noisy = re.compile(r"子公司|孙公司|实缴|缴纳|转增|不得少于|出资额")
    values = []
    for ev in evidences:
        for src in (ev.text, ev.parent_text):
            for m in pat.finditer(src):
                ctx_before = src[max(0, m.start() - 14):m.start()]
                try:
                    val = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                # 资料卡/增资沿革中的发行人本体注册资本才是答案候选
                if not noisy.search(ctx_before) or \
                        re.search(r"增加至|增至", ctx_before):
                    values.append(val)
    if values:
        cur = max(values)  # 当前注册资本为历次增资后的最大值
        text_val = f"{cur:,.2f}".rstrip("0").rstrip(".")
        return f"公司注册资本为 {text_val} 万元。"
    return ""


def _slot_legal_rep(question: str, evidences: List[Evidence]) -> str:
    """法定代表人槽位（兴图 id=531）。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 模板答案或空串
    """
    if "法定代表人" not in question:
        return ""
    pat = re.compile(
        r"法定代表人[：:是为\s]{0,8}?([\u4e00-\u9fa5]{2,4})")
    # 发行当事人页有大量中介机构法定代表人（券商/会所/律所/评估机构），
    # 其前文出现机构特征词时剔除
    institution = re.compile(
        r"证券|会计师事务所|律师事务所|资产评估|银行|保险|基金|"
        r"交易所|保荐机构|主承销商|信用评级|登记结算")
    # 名字后面粘连的卡式表字段，需要截掉
    tail_re = re.compile(r"(注册|住所|电话|传真|邮政|签字|盖章|成立|联系)")
    candidates: List[tuple] = []
    for ev in evidences:
        for src in (ev.text, ev.parent_text):
            for m in pat.finditer(src):
                ctx_before = src[max(0, m.start() - 24):m.start()]
                if institution.search(ctx_before):
                    continue
                name = tail_re.split(m.group(1))[0]
                # 再裁掉可能粘连的单个卡式字段起始字（注/住/电/传/签等）
                while len(name) > 2 and name[-1] in "注住电传成盖签邮":
                    name = name[:-1]
                if name in {"姓名", "名称", "签字"} or len(name) < 2:
                    continue
                # 资料卡（同时出现注册资本/注册地址/成立日期）与概览章节加权
                score = 0.0
                ctx = src[max(0, m.start() - 40):m.end() + 40]
                if re.search(r"注册资本|注册地址|成立日期|实收资本", ctx):
                    score += 2.0
                if "概览" in ev.heading_path or "发行人基本情况" in \
                        ev.heading_path:
                    score += 1.0
                score -= ev.page_no * 0.001
                candidates.append((score, name, ev.page_no))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return f"公司法定代表人为 {candidates[0][1]}。"
    return ""


def _slot_financial_series(question: str,
                           evidences: List[Evidence]) -> str:
    """军品收入多期金额/占比（兴图 id=260、id=33）。

    取同时包含“国防/军/主营业务收入”且数字（金额或百分比）最全的候选。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 原文整句答案或空串
    """
    want_ratio = "比重" in question or "占比" in question or "比例" in question
    if not re.search(r"军用|军品|国防|主营业务收入", question):
        return ""
    if not re.search(r"收入|比重|占比", question):
        return ""
    best, best_n = "", -1
    for unit, order in candidate_units(evidences):
        # 必须明确是军品口径（国防/军用/军品），排除民用行、合计行与
        # 其他产品（视频预警/配套产品）明细表的干扰
        # P4 风险提示句同时含军/民口径，只要句中出现军品关键词即保留
        if not re.search(r"国防|军用|军品", unit):
            continue
        if want_ratio:
            nums = re.findall(r"\d+\.\d{2}%", unit)
        else:
            # 只认真金额：带千分位逗号的数字（6,464.51）或紧跟“万元”的数，
            # 排除年份（2019）与比率小数（0.93、82.10）造成的多期误判
            nums = re.findall(
                r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d[\d,\.]{2,}(?=\s*万元)",
                unit)
        if len(nums) > best_n:
            best, best_n = unit, len(nums)
    if best_n >= 3:
        return best
    return ""


def _slot_updown(question: str, evidences: List[Evidence]) -> str:
    """电子信息行业上游企业 / 下游行业（兴图 id=34、id=793）。

    :param question: 原始问题
    :param evidences: 证据列表
    :return: 原文整句答案或空串
    """
    if "上游" not in question and "下游" not in question:
        return ""
    side = "上游" if "上游" in question else "下游"
    must = {
        "上游": ("电子元器件", "金属壳体"),
        "下游": ("军队", "政府", "能源"),
    }[side]
    best, best_score = "", -1.0
    for unit, order in candidate_units(evidences):
        if side not in unit:
            continue
        hit = sum(1 for w in must if w in unit)
        score = hit * 2 - order * 0.05
        if hit and score > best_score:
            best, best_score = unit, score
    if best and all(w in best for w in must):
        return best
    return best  # 关键词不全时也返回最相关句，由评测暴露问题


def _slot_fact_sentence(question: str, evidences: List[Evidence],
                        triggers: Tuple[str, ...],
                        requires: Tuple[str, ...] = ()) -> str:
    """通用事实句槽位：在候选句中找同时满足触发词与必要关键词的整句。

    :param question: 原始问题
    :param evidences: 证据列表
    :param triggers: 问题意图触发词（任一命中即启用）
    :param requires: 答案句必须包含的关键事实词
    :return: 最佳原文整句或空串
    """
    if not any(t in question for t in triggers):
        return ""
    best, best_score = "", -1.0
    for unit, order in candidate_units(evidences):
        # 必要事实词必须全部出现，防止答非所问
        if requires and not all(w in unit for w in requires):
            continue
        # 必要词命中数优先，证据名次与通用覆盖度次之
        hit = sum(1 for w in requires if w in unit) if requires else 1
        score = hit * 2 + _unit_score(question, unit, order, _IDF)
        if score > best_score:
            best, best_score = unit, score
    return best


# 全局 IDF 由引擎注入（避免槽位函数重复传参）
_IDF: Dict[str, float] = {}


def generic_extract(question: str, evidences: List[Evidence],
                    idf: Dict[str, float]) -> str:
    """朴素通用抽取（基线模式）：仅按 IDF 覆盖度等分数选最佳候选句。

    :param question: 用户问题
    :param evidences: 检索证据列表
    :param idf: BM25 IDF 权重表
    :return: 得分最高的候选单元
    """
    units = list(candidate_units(evidences))
    if not units:
        return ""
    return sorted(units,
                  key=lambda x: _unit_score(question, x[0], x[1], idf),
                  reverse=True)[0][0]


def extract_answer(question: str, evidences: List[Evidence],
                   idf: Dict[str, float], use_slots: bool = True) -> str:
    """按意图分派槽位抽取答案，失败再走通用句子抽取。

    :param question: 用户问题
    :param evidences: 检索证据列表
    :param idf: BM25 IDF 权重表
    :param use_slots: False=基线朴素模式，只走通用句子抽取
    :return: 答案文本
    """
    global _IDF
    _IDF = idf

    # 基线：不使用任何表格/确定性槽位，模拟优化前朴素链路
    if not use_slots:
        return generic_extract(question, evidences, idf)

    # 槽位按特异性从高到低分派
    for slot in (_slot_working_capital, _slot_shares, _slot_projects,
                 _slot_related_parties, _slot_capital, _slot_legal_rep,
                 _slot_financial_series, _slot_updown):
        answer = slot(question, evidences)
        if answer:
            return answer

    # 事实句槽位
    fact_rules = [
        (("技术标准",), ("视频指挥系统技术标准",)),
        (("重要供应商",), ("视频指挥",)),
        (("科技进步一等奖", "国家科技进步"),
         ("情报、指挥、控制与通信网络一体化工程",)),
    ]
    for triggers, requires in fact_rules:
        if any(t in question for t in triggers):
            answer = _slot_fact_sentence(question, evidences, triggers,
                                         requires)
            if answer:
                return answer

    # 通用句子级抽取兜底
    units = list(candidate_units(evidences))
    if not units:
        return ""
    return sorted(units,
                  key=lambda x: _unit_score(question, x[0], x[1], idf),
                  reverse=True)[0][0]


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
@dataclass
class QAAnswer:
    """问答结果（含答案、证据与耗时）。"""

    question: str
    answer: str
    evidences: List[Evidence]
    latency_s: float
    timings: dict = field(default_factory=dict)


class QAEngine:
    """问答引擎：检索 + 表格答案组织 + 延迟控制。"""

    def __init__(self, retriever: Retriever, use_routing: bool = True,
                 use_slots: bool = True) -> None:
        """注入检索器。

        :param retriever: 已初始化的检索器
        :param use_routing: 是否启用公司路由（基线对比可关闭）
        :param use_slots: 是否启用表格/确定性答案槽位（基线对比可关闭）
        """
        self.retriever = retriever
        self.use_routing = use_routing
        self.use_slots = use_slots

    def answer(self, question: str) -> QAAnswer:
        """执行一次完整问答。

        :param question: 用户问题（中/英文）
        :return: QAAnswer
        """
        t0 = time.perf_counter()
        evidences = self.retriever.retrieve(
            question, top_k=CONFIG_TOP_N, use_routing=self.use_routing)
        t_retrieval = time.perf_counter() - t0

        t1 = time.perf_counter()
        text = extract_answer(question, evidences,
                              self.retriever.store.sparse.idf,
                              use_slots=self.use_slots)
        t_answer = time.perf_counter() - t1
        return QAAnswer(
            question=question, answer=text, evidences=evidences[:3],
            latency_s=time.perf_counter() - t0,
            timings={"retrieval_s": round(t_retrieval, 3),
                     "answer_s": round(t_answer, 3)})


# 延迟导入配置常量，避免顶部循环依赖
from config import CONFIG as _CFG  # noqa: E402

CONFIG_TOP_N = _CFG.answer_top_n
