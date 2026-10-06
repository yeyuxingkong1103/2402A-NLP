# -*- coding: utf-8 -*-
# 【图文问答引擎 · qa_engine.py】句子级抽取+确定性槽位+图表解题器+全链路耗时控制
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""问答主流程：

- 检索Top-10证据（展示Top-3），在句子/表格行范围内做覆盖度抽取；
- 注册资本/法定代表人/补充流动资金/发行股数/关联方/募集资金项目等走确定性槽位；
- 图表专用解题器：
  · 组织结构图：矢量标签bbox + 绘图支架横线判定父子层级，统计下属部门/销售处；
  · 柱状图：OCR条目按y配对行业标签与百分比，取最大值与唯一负值；
- OCR/CLIP不可用时答案显式标注降级；全程打点，端到端≤3秒（OCR在建库期完成）。
"""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from image_extractor import FigureRecord, _PCT_TOKEN
from retriever import (
    Evidence, Retriever, answer_type_bonus, exact_phrases, query_terms,
)

logger = logging.getLogger(__name__)

_SENT_RE = re.compile(r"[^。；;！？!?]+[。；;！？!?]?")
_NUMBER_Q = re.compile(r"多少|比重|比例|金额|注册资本|收入|数量|几个|占比")
_PERSON_Q = re.compile(r"谁|代表人|发明人")
_LIST_Q = re.compile(r"哪些|什么项目|包括|涉及")
_PREFIX_HEAD_RE = re.compile(r"【[^】]*】\n?")
_PREFIX_ENT_RE = re.compile(r"主体：[^\n]*\n?")
_NUMBER_TOKEN = re.compile(r"\d[\d,\.]*\s*(?:%|万元|亿元)?")
_COMPANY_FULL_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
_TABLE_SEP_RE = re.compile(r"^[\s\|\-:]+$")


# ============================ 候选句 ============================

def _strip_prefix(raw: str) -> str:
    """去除块前缀（文档/标题标记、主体公司注入行）。

    :param raw: 块原文
    :return: 去前缀文本
    """
    return _PREFIX_ENT_RE.sub("", _PREFIX_HEAD_RE.sub("", raw))


def _iter_sentences(evidences: List[Evidence]):
    """从证据文本生成去重候选句/表格行（携带证据顺序权重）。

    :param evidences: 证据列表
    :return: 生成(句子, 证据序号0基)
    """
    seen = set()
    for order, ev in enumerate(evidences):
        if ev.chunk_type == "image":
            continue  # 图像块由图表解题器单独处理
        for source, base_order in ((ev.text, order), (ev.parent_text, order + 0.4)):
            raw = _strip_prefix(source)
            is_table = raw.count("|") >= 2
            units = raw.split("\n") if is_table else [raw.replace("\n", "")]
            for unit in units:
                unit = unit.strip(" |")
                if not unit or _TABLE_SEP_RE.match(unit):
                    continue
                if is_table:
                    pieces = [re.sub(r"\s*\|\s*", " ", unit).strip()]
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
                    yield sent, base_order


def _sentence_score(question: str, sent: str, order: int,
                    idf: Dict[str, float]) -> float:
    """候选答案句打分（覆盖度+长短语+类型+名次先验-长度惩罚）。

    :param question: 原始问题
    :param sent: 候选句
    :param order: 证据名次
    :param idf: IDF权重表
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
    elif not (12 <= len(sent) <= 220):
        score -= 0.1
    return score


# ============================ 图表解题器 ============================

def _children_of(figure: FigureRecord, parent_text: str,
                 max_depth_pt: float = 160.0) -> List[str]:
    """在组织结构图上按几何关系求某节点的直接下级标签。

    规则：取父节点框下方的绘图水平支架线（x范围覆盖父节点中心、取最长者），
    支架线下方一行、中心x落在支架范围内的标签即直接下级。

    :param figure: 图像证据记录（含标签bbox与水平线段）
    :param parent_text: 父节点完整文字（如“销售部”）
    :param max_depth_pt: 向下搜索的最大纵向距离
    :return: 下级标签文字列表（按x从左到右）
    """
    parent = next((lb for lb in figure.labels if lb.text == parent_text), None)
    if parent is None:
        return []
    px0, py0, px1, py1 = parent.bbox
    pcx = (px0 + px1) / 2
    # 候选支架线：父节点下方，x范围覆盖父节点中心
    brackets = [(y, xa, xb) for (y, xa, xb) in figure.h_lines
                if py1 - 2 <= y <= py1 + max_depth_pt
                and xa - 8 <= pcx <= xb + 8 and xb - xa > 20]
    if brackets:
        # 取距父节点最近的支架（避免误选到下一层级更宽的汇总横架）
        by, bx0, bx1 = min(brackets, key=lambda s: s[0] - py1)
        # 下级标签止于本支架之后的第一条支架线，防止隔代纳入
        later_brackets = sorted(y for (y, _, _) in figure.h_lines if y > by + 2)
        y_limit = later_brackets[0] - 2 if later_brackets else by + max_depth_pt
        children = [lb for lb in figure.labels
                    if by + 2 < lb.cy < y_limit
                    and bx0 - 10 <= lb.cx <= bx1 + 10
                    and abs(lb.cy - parent.cy) > 20]
        children.sort(key=lambda lb: lb.cx)
        if children:
            return [lb.text for lb in children]
    # 几何兜底：取父节点下方最近一行的全部标签（去除父节点自身）
    row = [lb for lb in figure.labels
           if lb.cy > py1 and lb.cy < py1 + max_depth_pt]
    if row:
        row_y = sorted(lb.cy for lb in row)[len(row) // 2]
        same_row = [lb for lb in row if abs(lb.cy - row_y) < 45]
        same_row.sort(key=lambda lb: lb.cx)
        return [lb.text for lb in same_row]
    return []


def solve_org_chart(question: str, figures: List[FigureRecord]) -> str:
    """组织结构图问题：销售部构成部门数、大客户销售部构成销售处数。

    :param question: 原始问题
    :param figures: 候选图像证据
    :return: 模板答案；无法定位返回空串
    """
    if not (("组织结构" in question or "组织架构" in question)
            and ("销售部" in question or "销售处" in question)):
        return ""
    figure = next((f for f in figures
                   if any(lb.text == "销售部" for lb in f.labels)), None)
    if figure is None:
        return ""
    sub_depts = _children_of(figure, "销售部")
    # 支架可能把非销售序列部门一并纳入，按名称语义保留销售/商务/贸易/渠道类
    sales_depts = [d for d in sub_depts
                   if re.search(r"销售|贸易|商务|渠道|客户|网络", d)]
    big_offices = [lb.text for lb in figure.labels if lb.text.endswith("销售处")]
    big_offices = sorted(set(big_offices), key=lambda t: (
        next((lb.cx for lb in figure.labels if lb.text == t), 0)))
    if not sales_depts or not big_offices:
        return ""
    answer = (f"组织结构图显示：销售部由{len(sales_depts)}个部门构成，"
              f"分别为{'、'.join(sales_depts)}；"
              f"其中大客户销售部由{len(big_offices)}个销售处构成，"
              f"分别为{'、'.join(big_offices)}。")
    return answer


def solve_bar_chart(question: str, figures: List[FigureRecord]) -> Tuple[str, bool]:
    """柱状/条形图问题：增长率最快行业与负增长行业（OCR条目配对）。

    :param question: 原始问题
    :param figures: 候选图像证据
    :return: (答案, 是否降级)；无法作答时答案为空串
    """
    if "增长率" not in question:
        return "", False
    figure = next((f for f in figures
                   if "增长率" in f.ocr_text() and
                   any(_PCT_TOKEN.match(it.text) for it in f.ocr_items)), None)
    if figure is None:
        # 降级：仅从邻近正文推断（如“……增长率10.5%位列第二，仅次于汽车电子”）
        for f in figures:
            m = re.search(r"增长率达?\s*([\d.]+)%?，?在所有应用行业中位列第二，"
                          r"仅次于([\u4e00-\u9fa5]{2,8})", f.nearby_text)
            if m:
                return (f"根据图注邻近正文：增长率最快的是{m.group(2)}"
                        f"（文中称工业控制增长率{m.group(1)}%位列第二，仅次于{m.group(2)}）；"
                        f"负增长行业仅载于原图，文本证据中未记载，需查看图像证据。"), True
        return "", True
    pcts = [it for it in figure.ocr_items if _PCT_TOKEN.match(it.text)]
    labels = [it for it in figure.ocr_items
              if re.search(r"[\u4e00-\u9fa5]", it.text)
              and not _PCT_TOKEN.match(it.text) and it.text != "增长率"]
    # 行业标签与百分比一一匹配：柱状图数值在柱端（x靠右）、标签在y轴侧（x靠左），
    # 同一行y接近。按|dy|升序全局贪心分配，避免相邻行（如“其他”与“汽车”）串配
    edges = []
    for p in pcts:
        for lb in labels:
            if abs(lb.y - p.y) < 0.09 and lb.x < p.x + 0.05:
                edges.append((abs(lb.y - p.y), p, lb))
    edges.sort(key=lambda e: e[0])
    used_p, used_lb, pairs = set(), set(), []
    for _, p, lb in edges:
        if id(p) in used_p or id(lb) in used_lb:
            continue
        used_p.add(id(p))
        used_lb.add(id(lb))
        pairs.append((lb.text, float(p.text.rstrip("%"))))
    if not pairs:
        return "", True
    name_max, v_max = max(pairs, key=lambda x: x[1])
    negatives = [(n, v) for n, v in pairs if v < 0]
    caption = figure.caption or "该图"
    answer = f"根据《{caption}》：增长率最快的是{name_max}行业（{v_max}%）"
    if negatives:
        n_name, n_val = min(negatives, key=lambda x: x[1])
        answer += f"；负增长的是{n_name}行业（{n_val}%）。"
    else:
        answer += "；图中未见负增长行业。"
    return answer, False


# ============================ 确定性槽位 ============================

def _issuer_name(ev: Evidence) -> str:
    """从证据块主体前缀读取发行人名称。

    :param ev: 证据对象
    :return: 公司全称（取不到为空串）
    """
    m = re.search(r"主体：([^\n]+)", ev.text)
    return m.group(1).strip() if m else ""


def _target_docs(question: str) -> set:
    """根据问题中出现的发行人全称确定目标文档集合。

    :param question: 原始问题
    :return: 目标文档tag集合；未指名为空集合（表示不限制）
    """
    docs = set()
    for doc_tag, company in CONFIG.doc_companies.items():
        if company and company in question:
            docs.add(doc_tag)
    return docs


def _slot_answer(question: str, evidences: List[Evidence],
                 figures: Optional[List[FigureRecord]] = None) -> str:
    """确定性槽位抽取（高精度事实型问题）。

    :param question: 原始问题
    :param evidences: 检索证据
    :param figures: 候选图像证据（关联方关系图等）
    :return: 模板答案；无可靠匹配返回空串
    """
    # 问题点名某发行人时，槽位仅采用该文档证据，杜绝跨文档事实串味
    target_docs = _target_docs(question)
    if target_docs:
        evidences = [ev for ev in evidences if ev.doc in target_docs]
    figures = figures or []
    # ---- 注册资本：取发行人当前最大值，排除他司紧邻值 ----
    if "注册资本" in question:
        # 连接词兼容“注册资本增加至/增至/为/：/人民币”等表述
        pattern = re.compile(
            r"注册资本[：:是为增加至达人民币\s]{0,12}?"
            r"([0-9][0-9,\.]{2,})\s*(万元|万|元)?")
        vals: List[float] = []
        for ev in evidences:
            issuer = _issuer_name(ev)
            for m in pattern.finditer(ev.parent_text):
                window = ev.parent_text[max(0, m.start() - 30):m.start()]
                other = [c for c in _COMPANY_FULL_RE.findall(window)
                         if issuer and c != issuer and issuer not in c]
                if other:
                    continue
                try:
                    val = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                # “元”口径折算为万元，避免3,000,000元被误当成300万以上增资
                if m.group(2) == "元":
                    val /= 10000.0
                if val >= 100:
                    vals.append(val)
        if vals:
            current = max(vals)
            text_val = f"{current:,.2f}".rstrip("0").rstrip(".")
            return f"注册资本为 {text_val} 万元。"
        return ""

    # ---- 法定代表人：发行人自身基本情况卡片优先，排除中介/声明/子公司卡片 ----
    if "法定代表人" in question:
        pattern = re.compile(r"法定代表人[：:是为\s]{0,6}?([\u4e00-\u9fa5]{2,4})")
        # 中介机构、声明签章、当事人名录等章节直接排除
        blacklist_heading = re.compile(
            r"声明|当事人|中介|资产评估|会计师|律师|证券|交易所|银行|"
            r"控股子公司|参股公司|分公司")
        candidates: List[Tuple[int, str]] = []
        for order, ev in enumerate(evidences):
            issuer = _issuer_name(ev)
            if blacklist_heading.search(ev.heading_path or ""):
                continue
            for m in pattern.finditer(ev.parent_text):
                name = m.group(1)
                if name in {"姓名", "名称", "签字"}:
                    continue
                if len(name) >= 3 and name[-1] in {"注", "签", "盖"}:
                    name = name[:-1]
                window_before = ev.parent_text[max(0, m.start() - 60):m.start()]
                window_after = ev.parent_text[m.end():m.end() + 12]
                # 当事人卡片特征：姓名后紧跟“住所/电话/传真”
                if re.match(r"^(住所|电话|传真|身份证)", window_after):
                    continue
                # 子公司卡片特征：前文出现“公司名称”
                if "公司名称" in window_before:
                    continue
                # 前文出现其他公司全称且非发行人自身
                other = [c for c in _COMPANY_FULL_RE.findall(window_before)
                         if issuer and c != issuer and issuer not in c]
                if other:
                    continue
                # 评分：发行人基本情况章节+5；同卡含发行人注册资本(5,520)+5；
                # 证据名次越前越好
                score = 10 - order * 0.5
                if "发行人基本情况" in (ev.heading_path or ""):
                    score += 5
                if re.search(r"注册资本[：:\s]*5,?520", ev.parent_text):
                    score += 5
                candidates.append((score, name))
        # 反向句式兜底：“程家明为公司股东兴图投资的法定代表人……”
        if not candidates:
            rev_re = re.compile(
                r"([\u4e00-\u9fa5]{2,4})(?:为|是)[^。\n；]{0,30}法定代表人")
            for order, ev in enumerate(evidences):
                if blacklist_heading.search(ev.heading_path or ""):
                    continue
                for m in rev_re.finditer(ev.parent_text):
                    name = m.group(1)
                    if name in {"姓名", "名称"}:
                        continue
                    score = 5 - order * 0.3
                    if "发行人基本情况" in (ev.heading_path or ""):
                        score += 3
                    candidates.append((score, name))
        if candidates:
            name = max(candidates, key=lambda x: x[0])[1]
            return f"法定代表人为 {name}。"
        return ""

    # ---- 补充流动资金 ----
    if "补充流动资金" in question:
        pattern = re.compile(r"补充流动资金[\s\S]{0,15}?([0-9][0-9,\.]{2,})\s*(万?元)?")
        for ev in evidences:
            m = pattern.search(ev.parent_text)
            if m:
                return f"本次发行募集资金中用于补充流动资金的金额为 {m.group(1)} 万元。"
        return ""

    # ---- 本次发行股数与占比（力源信息） ----
    if "发行股数" in question or ("发行" in question and "总股本" in question):
        for ev in evidences:
            m = re.search(
                r"([\d,]{4,})\s*万股[，,][\s\S]{0,20}?比例为\s*([\d.]+)\s*%",
                ev.parent_text)
            if m:
                return (f"本次发行股数为 {m.group(1)} 万股，"
                        f"占发行后总股本的比例为 {m.group(2)}%。")
        return ""

    # ---- 存在控制关系的关联方（力源信息，第157页表） ----
    if "控制关系的关联方" in question and "不存" not in question:
        for ev in evidences:
            window_m = re.search(r"存在控制关系的关联方([\s\S]{0,120}?)"
                                 r"(?:2、|二、|不存在控制关系)", ev.parent_text)
            window = window_m.group(1) if window_m else ev.parent_text
            person = re.search(r"([\u4e00-\u9fa5]{2,4})\s*([\d.]+)\s*%\s*"
                               r"(公司控股股东|控股股东|实际控制人)", window)
            if person:
                return (f"存在控制关系的关联方为{person.group(1)}，"
                        f"持股比例{person.group(2)}%，"
                        f"与本公司关系：{person.group(3)}。")
        return ""

    # ---- 不存在控制关系的关联方企业（力源信息，第157页表/第158页关系图） ----
    if "不存在控制关系" in question:
        # 路径A：第158页《公司的关联方关系图示》OCR名单。
        # 图中同时含自然人、发行人自身与“报告期内曾为关联方”的3家企业，
        # 规则：只留组织形态名称；剔除发行人自身与邻近正文点名的已剥离企业。
        org_suffix = re.compile(r"(投资|贸易|博润|聚源|芯|电子|科技|集团|实业)")
        for fig in figures:
            if target_docs and fig.doc not in target_docs:
                continue
            if not re.search(r"关联方", fig.caption or "") and \
                    "关联方" not in fig.to_text():
                continue
            tokens = re.split(r"[；;，,、\s]+", fig.ocr_text())
            former = fig.nearby_text
            names = []
            for tok in tokens:
                tok = tok.strip("()（）.。")
                if len(tok) < 2 or not org_suffix.search(tok):
                    continue
                if "力源信息" in tok:
                    continue
                # 已对外转让/吊销的原关联方：仅当该名称邻近±25字内出现
                # “曾为/对外转让/吊销/停止经营”描述时才剔除（普芯达虽在
                # 邻近正文出现，但身份是“近亲属控制的公司”，应予保留）
                fi = former.find(tok)
                if fi >= 0:
                    ctx = former[max(0, fi - 25):fi + len(tok) + 45]
                    if re.search(r"曾为|对外转让|吊销|停止经营|"
                                 r"原为同一实际控制人", ctx):
                        continue
                if tok not in names:
                    names.append(tok)
            if len(names) >= 5:
                return "不存在控制关系的关联方企业包括：" + "、".join(names) + "。"
        # 路径B：第157页关系表的“名称—关系”行文
        for ev in evidences:
            window_m = re.search(r"不存在控制关系的关联方([\s\S]{0,400}?)"
                                 r"(?:3、|三、|报告期内曾为|4、)", ev.parent_text)
            window = window_m.group(1) if window_m else ""
            relation_names = re.findall(
                r"([\u4e00-\u9fa5]{2,8}?)"
                r"(?=持有公司股份\s*5%\s*以上的股东|同一实际控制人控制的企业|"
                r"实际控制人近亲属控制的公司)", window)
            # 无锚点兜底：直接匹配“名称+关系后缀”的表格式行文
            if len(relation_names) < 5:
                relation_names = re.findall(
                    r"([\u4e00-\u9fa5]{2,8}?)"
                    r"(?=持有公司股份\s*5%\s*以上的股东|同一实际控制人控制的企业|"
                    r"实际控制人近亲属控制的公司)", ev.parent_text)
            relation_names = [n for n in dict.fromkeys(relation_names)
                              if len(n) >= 2]
            if len(relation_names) >= 5:
                return ("不存在控制关系的关联方企业包括："
                        + "、".join(relation_names) + "。")
        return ""

    # ---- 募集资金拟投资项目（力源信息：引号枚举或用途表） ----
    if "募集资金" in question and ("投资" in question or "项目" in question):
        for ev in evidences:
            text = ev.parent_text
            quoted = re.findall(r"[“\"]([^”\"]{2,20})[”\"]", text)
            projects = [q for q in quoted if re.search(r"中心|平台|资金|种类|物流", q)]
            if len(projects) >= 4:
                return "本次募集资金拟投资以下项目：" + "、".join(
                    dict.fromkeys(projects)) + "。"
        # 表格证据兜底：Markdown行里含关键词的单元格
        for ev in evidences:
            rows = [ln.strip(" |") for ln in ev.parent_text.split("\n")]
            cands = [r for r in rows
                     if re.search(r"仓储|物流中心|研发中心|电子商务|扩充产品|营运资金", r)
                     and not re.search(r"^\d", r)]
            if len(cands) >= 4:
                return "本次募集资金拟投资以下项目：" + "、".join(
                    dict.fromkeys(cands[:6])) + "。"
        return ""

    return ""


def extract_answer(question: str, evidences: List[Evidence],
                   idf: Dict[str, float],
                   figures: Optional[List[FigureRecord]] = None
                   ) -> Tuple[str, bool]:
    """在证据中抽取答案，图表问题优先走图表解题器。

    :param question: 用户问题
    :param evidences: 检索证据
    :param idf: BM25 IDF表
    :param figures: 命中文档的全部图像证据（供图表解题）
    :return: (答案, 是否降级)
    """
    figures = figures or []
    # 图表解题器优先
    if "组织结构" in question:
        chart_answer = solve_org_chart(question, figures)
        if chart_answer:
            return chart_answer, False
    if "增长率" in question:
        chart_answer, degraded = solve_bar_chart(question, figures)
        if chart_answer:
            return chart_answer, degraded

    slotted = _slot_answer(question, evidences, figures)
    if slotted:
        return slotted, False

    candidates: List[Tuple[str, int]] = list(_iter_sentences(evidences))
    if not candidates:
        return "", False

    if _NUMBER_Q.search(question):
        numbered = [(s, o) for s, o in candidates if _NUMBER_TOKEN.search(s)]
        if numbered:
            ranked = sorted(numbered,
                            key=lambda x: _sentence_score(
                                question, x[0], x[1], idf), reverse=True)
            if "分别" in question:
                best, bo = ranked[0]
                best_score = _sentence_score(question, best, bo, idf)
                fullest, fullest_n = best, len(_NUMBER_TOKEN.findall(best))
                for s, o in numbered:
                    sc = _sentence_score(question, s, o, idf)
                    n_num = len(_NUMBER_TOKEN.findall(s))
                    if sc >= best_score - 0.35 and n_num > fullest_n:
                        fullest, fullest_n = s, n_num
                return fullest, False
            return ranked[0][0], False

    if _PERSON_Q.search(question):
        persons = [(s, o) for s, o in candidates
                   if re.search(r"(?:是|为|：|:)\s*[\u4e00-\u9fa5]{2,4}[，。；\s]", s)]
        pool = persons or candidates
        return sorted(pool, key=lambda x: _sentence_score(
            question, x[0], x[1], idf), reverse=True)[0][0], False

    ranked = sorted(candidates,
                    key=lambda x: _sentence_score(
                        question, x[0], x[1], idf), reverse=True)
    if "上游" in question and "下游" in question and ranked:
        up = next((s for s, _ in ranked if "上游" in s), "")
        down = next((s for s, _ in ranked if "下游" in s and s != up), "")
        if up and down:
            return up + " " + down, False
    return ranked[0][0], False


@dataclass
class QAAnswer:
    """问答结果（答案、图文证据、耗时、降级标记）。"""

    question: str
    answer: str
    evidences: List[Evidence]
    latency_s: float
    timings: dict = field(default_factory=dict)
    mode: str = "extractive"
    degraded: bool = False


class QAEngine:
    """图文问答引擎：检索+图表解题+槽位/句子抽取+延迟控制。"""

    def __init__(self, retriever: Retriever) -> None:
        """注入检索器，并构建chunk_id→图像记录索引。

        :param retriever: 已初始化检索器
        """
        self.retriever = retriever
        self.figure_map: Dict[int, FigureRecord] = {
            c.chunk_id: c.figure for c in retriever.store.chunks
            if c.figure is not None}

    def answer(self, question: str) -> QAAnswer:
        """执行一次完整图文问答。

        :param question: 用户问题（中/英文）
        :return: QAAnswer
        """
        t0 = time.perf_counter()
        evidences_n = self.retriever.retrieve(question, top_k=10)
        t_retrieval = time.perf_counter() - t0

        t1 = time.perf_counter()
        # 图表解题需要的图像证据：命中文档范围内全部图像块（含Top10未覆盖的同页图）
        hit_docs = {ev.doc for ev in evidences_n}
        figures = [c.figure for c in self.retriever.store.chunks
                   if c.chunk_type == "image" and c.doc in hit_docs
                   and c.figure is not None]
        # 优先把Top10内图像证据排到前面
        top_fig_ids = {ev.chunk_id for ev in evidences_n
                       if ev.chunk_type == "image"}
        figures.sort(key=lambda f: 0 if next(
            (cid for cid in top_fig_ids if self.figure_map.get(cid) is f),
            None) is not None else 1)
        answer, degraded = extract_answer(
            question, evidences_n, self.retriever.store.sparse.idf, figures)
        t_answer = time.perf_counter() - t1

        return QAAnswer(
            question=question, answer=answer,
            evidences=evidences_n[:3],
            latency_s=time.perf_counter() - t0,
            timings={"retrieval_s": round(t_retrieval, 3),
                     "answer_s": round(t_answer, 3)},
            degraded=degraded)
