# -*- coding: utf-8 -*-
"""抽取式答案合成模块（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

对应本工单的【答案合成优化】，在 02 工单「抽取式合成」基础上，新增
【表格类问题】的专用答案路径：

- 表格类问题（answer_type == "table"）：优先从命中的表格块中提取「键值对行」，
  按“表头：取值”结构化拼装答案（如“发行股数：1,670万股；发行后总股本：6,670万股”），
  保证“字段—取值”一一对应，避免表头与数值分离；
- 正文类问题：沿用 02 工单的句子级打分抽取（关键词覆盖 / 数值线索 / 实体 /
  定义式句式 / 对比词惩罚 / 套话惩罚）。

为什么仍用抽取式？
- 需求要求“准确率 ≥ 90% 且响应 ≤ 3s”，本地生成模型（deepseek-r1:1.5b）为
  推理型，速度慢且易幻觉；抽取式零幻觉、可溯源（附文档与页码）、毫秒级返回。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from src import config
from src.chunking import split_sentences
from src.query_understanding import QueryAnalysis
from src.reranker import ScoredDoc

_NUM_RE = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|元|%|％|股|人|项|家|个|年)?")
_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*[%％]")
_MONEY_RE = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|元)")
# 定义式句式：关键词后紧跟 为/是/：/| 等，表明该句在“给出取值”
_DEFN_RE = re.compile(r"(?:为|是|：|:|＝|=|\|)\s*[\d《（(A-Za-z\u4e00-\u9fa5]")

# 图形语义块里「由图可知」派生出的结论句（增长率最快 / 负增长 / 占比最高）
_CONCL_RE = re.compile(r"增长率最快|负增长|占比最高")
# 结论按主题细分：只有问句主题与结论主题一致时，该结论句才有资格入选答案。否则会出现
# 「问组织结构图，却把别处『获利率』图的『增长率最快的是…』结论排到首位」的图形块串味
# （同一份招股说明书统计图极多，各自都会派生极值结论）。
_GROWTH_CONCL_RE = re.compile(r"增长率最快|负增长")
_SHARE_CONCL_RE = re.compile(r"占比最高")
_GROWTH_Q_KEYS = ("增长", "增速", "最快", "负增长", "增幅")
_SHARE_Q_KEYS = ("占比", "比重", "结构", "份额", "比例")

# 公司基本信息类字段（此类问题优先命中“概览/发行人基本情况”章节）
_BASIC_FIELDS = {
    "注册资本", "法定代表人", "成立日期", "注册地址", "控股股东", "实际控制人",
    "行业分类", "公司名称", "邮政编码", "主营业务", "英文名称", "实收资本",
}
_SUMMARY_SECTIONS = ("概览", "基本情况", "本次发行概况", "发行人基本情况", "发行情况")

# 对比词对：问题只问一侧时，出现“另一侧”的句子应被降权（如问下游却召回上游）
_CONTRAST_PAIRS = [
    ("上游", "下游"), ("军用", "民用"), ("直接", "间接"),
    ("增加", "减少"), ("上升", "下降"), ("境内", "境外"), ("进口", "出口"),
    ("控制关系", "不存在控制关系"),
]
# 标题行（作为候选句时应剔除，避免标题中的关键词造成误命中）
# 枚举型问句：需要“整表罗列”（如募投项目、无控制关系关联方企业）
_ENUM_RE = re.compile(
    r"(哪些|有哪些|主要包括|包括哪些|涉及哪些|分别有|都有|"
    r"哪些企业|哪些行业|哪些项目|哪些公司|哪些产品|哪些标准)"
)
_HEADING_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百零〇\d]+[章节]\s*[^\n]{0,40}"
    r"|[一二三四五六七八九十]+、[^\n]{0,40}"
    r"|（[一二三四五六七八九十]+）[^\n]{0,40}"
    r"|\d+(?:\.\d+){1,3}\s+[^\n]{0,40})\s*$"
)

# 签署页 / 中介机构套话：其中的人名（保荐代表人、项目协办人等）易被误当作答案
_BOILERPLATE_RE = re.compile(
    r"(保荐代表人|项目协办人|保荐机构|证券股份有限公司|会计师事务所|律师事务所"
    r"|签字|盖章|年\s*月\s*日)"
)


@dataclass
class AnswerBundle:
    """答案合成结果。"""

    answer: str
    citations: List[int] = field(default_factory=list)
    evidences: List[dict] = field(default_factory=list)
    source: str = ""


def _kw_cov(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = text.replace(" ", "")
    return sum(1 for k in keywords if k.replace(" ", "") in t) / len(keywords)


def _phrase_terms(keywords: List[str]) -> List[str]:
    """构造“短语线索”：长词 + 相邻双词组合。

    招股说明书问句经分词后，关键限定语常被切成多个短词（如“募集资金”→
    “募集/资金”），单独匹配短词区分度低。这里补充相邻双词组合（“募集资金”），
    使“表头/表题与问句短语一致”的候选获得更高分，避免被“数值密集但无关”的
    表格（如现金流量表）抢占。
    """
    longs = [k for k in keywords if len(k) >= 3]
    bigrams = [keywords[i] + keywords[i + 1] for i in range(len(keywords) - 1)
               if len(keywords[i]) >= 2 and len(keywords[i + 1]) >= 2]
    terms = longs + bigrams
    if not terms:
        terms = [k for k in keywords if len(k) >= 2]
    return list(dict.fromkeys(t for t in terms if t))


def _phrase_cov(text: str, keywords: List[str]) -> float:
    """短语覆盖率：区分“募集资金/补充流动资金/主营业务”等关键限定语。"""
    terms = _phrase_terms(keywords)
    if not terms:
        return 0.0
    t = text.replace(" ", "")
    return sum(1 for k in terms if k.replace(" ", "") in t) / len(terms)


def _num_score(text: str, atype: str, years: List[str]) -> float:
    s = 0.0
    if atype == "ratio":
        s += 0.7 if _PCT_RE.search(text) else 0.0
        s += 0.3 if _NUM_RE.search(text) else 0.0
    elif atype in ("amount", "table"):
        # 金额线索：数值后跟单位（3,393.40万元）或单位写在表头（计划总投资(万元)：3,393.40）
        has_money = bool(_MONEY_RE.search(text)) or (
            bool(re.search(r"(万元|亿元|元)", text)) and bool(_NUM_RE.search(text)))
        s += 0.5 if has_money else 0.0
        s += 0.3 if _NUM_RE.search(text) else 0.0
        s += 0.2 if _PCT_RE.search(text) else 0.0
        # 多数值行加成：募集资金投向、分年度收入等“一行多值”数据更可能是答案所在
        # （权重经调低，避免“数值密集但主题无关”的表格（如现金流量表）抢占答案）
        s += min(0.24, 0.06 * len(_NUM_RE.findall(text)))
    elif atype == "number":
        s += 1.0 if _NUM_RE.search(text) else 0.0
    else:
        s += 0.3 if _NUM_RE.search(text) else 0.0
    if years:
        s += 0.2 * (sum(1 for y in years if y.replace(" ", "") in text.replace(" ", "")) / len(years))
    return min(s, 1.0)


def _ent_score(text: str, entities: List[str]) -> float:
    if not entities:
        return 0.0
    t = text.replace(" ", "")
    return sum(1 for e in entities if e.replace(" ", "") in t) / len(entities)


def _len_score(text: str) -> float:
    n = len(text)
    if n < 12:
        return 0.2
    if n <= 200:
        return 1.0
    if n <= 400:
        return 0.6
    return 0.3


def _defn_score(text: str, keywords: List[str]) -> float:
    """定义式句式得分：关键词附近出现“为/是/：/|”等取值标记。"""
    if not keywords:
        return 0.0
    t = text.replace(" ", "")
    for k in keywords:
        kk = k.replace(" ", "")
        idx = t.find(kk)
        if idx >= 0:
            window = t[idx + len(kk): idx + len(kk) + 8]
            if _DEFN_RE.search(window) or window[:1] in "：:＝=|":
                return 1.0
    return 0.0


def _summary_prior(section: str, keywords: List[str]) -> float:
    """章节先验：公司基本信息类问题优先“概览 / 发行人基本情况”章节。"""
    if not any(k in _BASIC_FIELDS for k in keywords):
        return 0.0
    return 1.0 if any(s in (section or "") for s in _SUMMARY_SECTIONS) else 0.0


def _enum_score(text: str) -> float:
    """列举型问题的“枚举线索”得分。"""
    s = 0.0
    if re.search(r"(主要包括|包括|涉及|分为|覆盖|适用于)", text):
        s += 0.5
    s += min(0.5, 0.1 * text.count("、"))
    return min(s, 1.0)


def _signal_score(text: str, analysis: QueryAnalysis) -> float:
    at = analysis.answer_type
    if at in ("ratio", "amount", "number", "table"):
        return _num_score(text, at, analysis.years)
    if at == "list":
        return _enum_score(text)
    if at == "entity":
        return 1.0 if re.search(r"[为是：:，,、]\s*[\u4e00-\u9fa5]{2,4}", text) else 0.3
    return 0.3 * _num_score(text, at, analysis.years)


def _boilerplate_penalty(text: str) -> float:
    """套话惩罚：签署页/中介机构条款中人名密集，避免其抢占答案位。"""
    return 1.0 if _BOILERPLATE_RE.search(text or "") else 0.0


def _contrast_penalty(text: str, keywords: List[str]) -> float:
    """对比词惩罚：问题只问一侧，句子却出现另一侧时降权。"""
    t = text.replace(" ", "")
    kws = " ".join(keywords)
    for a, b in _CONTRAST_PAIRS:
        if a in kws and b not in kws and b in t and a not in t:
            return 1.0
        if b in kws and a not in kws and a in t and b not in t:
            return 1.0
    return 0.0


def _tidy(text: str) -> str:
    """规整表格行与多余空白，便于阅读。"""
    t = re.sub(r"\s+", " ", text or "").strip()
    if t.startswith("[表格行]"):
        t = t[len("[表格行]"):].strip()
    if t.startswith("|"):
        cells = [c.strip() for c in t.strip("|").split("|")]
        cells = [c for c in cells if c and set(c) - set("- ")]
        return "，".join(cells)
    return t


# ---------------- 表格类问题专用路径（本工单核心） -----------------------------
def _table_answer(analysis: QueryAnalysis, scored: List[ScoredDoc],
                  max_rows: int = 8) -> Tuple[str, List[int], List[dict], str]:
    """从命中的表格块中抽取键值对行，结构化拼装答案。

    策略（本工单核心）：优先聚焦「得分最高且命中线索的表格块」，只从该表（含其
    跨页父块）取行，避免把多张无关表格的行拼在一起；若该表行数不足再顺延到下一
    张表补足。这样既保证“字段—取值”对应准确，又保证列表型问题（募投项目、关联
    方企业）能拿到完整的多行数据。
    """
    table_docs = [sd for sd in scored if sd.doc.metadata.get("ctype") in ("table", "table_kv")]
    if not table_docs:
        return "", [], [], ""

    def _rows_of(sd: ScoredDoc) -> List[Tuple[str, int, str]]:
        """取出单个表格块的候选行（仅用本块内容，避免并入同页其他表格）。

        同时支持两类表格块：
        - `table_kv` 块：正文即“[表格行] 表头：取值；…”，直接解析；
        - `table`（Markdown）块：由首行表头 + 数据行现算“表头：取值”，
          使 Markdown 块也能参与答案拼装（否则只有键值对块可用）。
        """
        md = sd.doc.metadata
        page = int(md.get("page", 0) or 0)
        source = md.get("source", "")
        out: List[Tuple[str, int, str]] = []
        content = sd.doc.page_content

        # 1) Markdown 表格：解析“|”行 → “表头：取值”
        md_rows = [ln.strip() for ln in content.splitlines() if ln.strip().startswith("|")]
        md_rows = [r for r in md_rows if set(r) - set("-| ")]
        if md_rows:
            parsed = [[c.strip() for c in r.strip("|").split("|")] for r in md_rows]
            header = parsed[0] if parsed else []
            for row in parsed[1:]:
                pairs = [f"{header[i]}：{c}" for i, c in enumerate(row)
                         if i < len(header) and c and header[i]]
                if len(pairs) >= 2:
                    out.append(("；".join(pairs), page, source))
            if out:
                return out

        # 2) 键值对块
        for ln in content.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("[表题]") or ln.startswith("[表格]"):
                continue
            if not ln.startswith("[表格行]") and "：" not in ln:
                continue
            body = _tidy(ln)
            if len(body.replace(" ", "")) >= 8:
                out.append((body, page, source))
        return out

    def _row_score(body: str) -> float:
        return (0.30 * _kw_cov(body, analysis.keywords)
                + 0.22 * _phrase_cov(body, analysis.keywords)
                + 0.28 * _num_score(body, analysis.answer_type, analysis.years)
                + 0.20 * _ent_score(body, analysis.entities))

    def _block_section_score(section: str) -> float:
        """表格块的表题/章节与问句的匹配度（表格级消歧的关键信号）。

        同一文档中多张表都可能“数值密集”，仅靠行内数值难以区分；而表题（如
        “本次募集资金拟投资以下项目”）与问句的短语一致性是最可靠的判别依据，
        据此避免现金流量表等无关表格抢占答案。
        """
        if not section:
            return 0.0
        return 0.6 * _kw_cov(section, analysis.keywords) + 0.4 * _phrase_cov(section, analysis.keywords)

    # 字段型 / 枚举型分流：字段型问题（“发行股数是多少”“注册资本是多少”）只需
    # 命中字段的那几行，避免把整张表（含每股净资产、每股收益等无关行）都拼进答案，
    # 造成“答案冗长、可读性差”；枚举型问题（“拟投资哪些项目”）则需整表罗列。
    field_mode = not _ENUM_RE.search(analysis.question or "")
    row_cap = 4 if field_mode else max_rows

    picked: List[Tuple[str, int, str]] = []
    seen: set = set()
    # 1) 先评估每张表格块的相关性：取「该表最佳行得分 + 章节先验 + 表题匹配」作为块得分；
    #    章节先验用于把“注册资本/法定代表人”类问题导向“概览/发行人基本情况”表；
    #    表题匹配用于在数值密集的多张表之间做消歧（本工单核心修复点）。
    block_rank: List[Tuple[float, List[Tuple[str, int, str]], float]] = []
    for sd in table_docs[:8]:
        rows = _rows_of(sd)
        if not rows:
            continue
        section = sd.doc.metadata.get("section", "") or ""
        prior = _summary_prior(section, analysis.keywords)
        best = (max(_row_score(b) for b, _, _ in rows)
                + 0.25 * prior + 0.5 * _block_section_score(section))
        rows.sort(key=lambda x: (-(_row_score(x[0]) + 0.25 * prior), x[1]))
        block_rank.append((best, rows, prior))

    if not block_rank:
        return "", [], [], ""
    block_rank.sort(key=lambda x: -x[0])

    # 2) 聚焦最相关表格：先取该表全部行；不足 row_cap 时顺延到“仍足够相关”的表补足。
    #    仅当后续表格块得分不低于最佳块的 50% 时才继续取行，避免把无关表格（如现金流
    #    量表）的行混入答案，污染“字段—取值”对应关系。
    best_score = block_rank[0][0]
    for score, rows, prior in block_rank:
        if picked and score < 0.5 * best_score:
            break
        cand_rows = rows
        if field_mode:
            # 仅保留命中问句关键词 / 短语的行；无命中则跳过该表（不回退整表，
            # 否则会把无关表格整表混入答案）
            keep = [r for r in rows
                    if _kw_cov(r[0], analysis.keywords) > 0
                    or _phrase_cov(r[0], analysis.keywords) > 0]
            if not keep:
                continue
            # 再按行得分做相对阈值过滤：只保留与最佳行得分接近者，
            # 避免“同表内偶含关键词但语义无关”的行（如每股净资产）混入
            top = _row_score(keep[0][0]) + 0.25 * prior
            near = [r for r in keep if _row_score(r[0]) + 0.25 * prior >= 0.7 * top]
            cand_rows = near or keep[:1]
        for body, page, source in cand_rows:
            key = body.replace(" ", "")
            if key in seen:
                continue
            seen.add(key)
            picked.append((body, page, source))
        if field_mode and picked:
            # 字段型问题只需一个“字段—取值”，取到即止，避免跨表污染
            break
        if len(picked) >= row_cap:
            break

    if not picked:
        # 回退：字段型问题所有表都无关键词命中时，退回最佳表的行，保证不漏答
        picked = list(block_rank[0][1][:row_cap])

    if not picked:
        return "", [], [], ""

    picked = picked[:row_cap]
    cites = sorted({p for _, p, _ in picked if p})
    # 来源取出现次数最多的文档（多文档消歧后的稳定归因）
    src = max({s for _, _, s in picked if s}, key=lambda s: sum(1 for _, _, x in picked if x == s),
              default="")
    page_ref = "、".join(f"第{p}页" for p in cites) if cites else "相关页面"
    doc_ref = f"《{src}》" if src else "招股说明书"
    body_text = "；".join(b for b, _, _ in picked)
    answer = f"根据{doc_ref}{page_ref}：{body_text}"
    evidences = [{"page": p, "text": b, "source": s, "score": round(_row_score(b), 4)}
                 for b, p, s in picked]
    return answer, cites, evidences, src


# ---------------- 图形类问题专用路径（本工单新增） -----------------------------
def _pairing_richness(text: str) -> int:
    """统计图形语义块「图内数值」行的配对条数。

    同一张图在招股说明书中常重复出现（如 id 6 的 IC 市场增长图同时印在第 72、310 页），
    不同页的裁剪与分辨率不同，OCR 可能漏识个别标签，使同一张图派生出**互相矛盾**的结论
    （一处「负增长的是IC卡」、另一处因漏识标签而误判）。配对条数越多说明该副本识别越完整、
    结论越可信，故答案合成时按此加权，实现「同一结论取最完整副本」。
    """
    m = re.search(r"图内数值：([^\n]*)", text)
    body = m.group(1).strip() if m else ""
    return body.count("；") + 1 if body else 0


def _figure_answer(analysis: QueryAnalysis, scored: List[ScoredDoc],
                   max_sentences: int = config.EXTRACTIVE_MAX_SENTENCES
                   ) -> Tuple[str, List[int], List[dict], str]:
    """从命中的图形语义块中抽取答案。

    图形语义块（ctype="figure"）由 `figure_semantics` 生成，其正文已是结构化语句：
    - 组织结构图：「销售部由4个部门构成：…。」「大客户销售部由6个销售处构成：…。」
    - 统计图：「由图可知：增长率最快的是汽车（14.0%）；负增长的是IC卡（-2.0%）」

    本路径优先只在图形块内做句子级打分，避免正文/表格块中"形似但无关"的句子
    （如正文里泛泛提到"销售部"）抢占答案位，从而保证 id 5 / id 6 的答案精准。
    """
    fig_docs = [sd for sd in scored if sd.doc.metadata.get("ctype") == "figure"]
    if not fig_docs:
        return "", [], [], ""

    units: List[Tuple[str, int, float, str, int]] = []
    seen: set = set()
    for sd in fig_docs:
        page = int(sd.doc.metadata.get("page", 0) or 0)
        source = sd.doc.metadata.get("source", "")
        rich = _pairing_richness(sd.doc.page_content)
        for s in split_sentences(sd.doc.page_content):
            key = s.replace(" ", "")
            if key in seen or len(key) < 6:
                continue
            seen.add(key)
            units.append((s, page, sd.score, source, rich))
    if not units:
        return "", [], [], ""

    # 问句主题 → 决定哪类派生结论可用（见 _GROWTH_CONCL_RE 注释）
    q_text = analysis.question or ""
    asks_growth = any(k in q_text for k in _GROWTH_Q_KEYS)
    asks_share = any(k in q_text for k in _SHARE_Q_KEYS)

    ranked: List[Tuple[float, str, int, str]] = []
    for text, page, w, source, rich in units:
        kw = _kw_cov(text, analysis.keywords)
        sig = _signal_score(text, analysis)
        ent = _ent_score(text, analysis.entities)
        s = 0.52 * kw + 0.18 * sig + 0.08 * ent + 0.10 * _len_score(text) + 0.12 * w
        if _CONCL_RE.search(text):
            kind_ok = (asks_growth if _GROWTH_CONCL_RE.search(text) else asks_share)
            if not kind_ok:
                continue            # 结论主题与问句不符 → 丢弃，避免图形块之间串味
            # 派生结论句是图形题的核心答案，给予显著加成，并按副本配对完整度微调排序，
            # 保证同一结论取「OCR 最完整副本」的那一条。
            s += 0.30 + 0.02 * min(rich, 8)
        ranked.append((s, text, page, source))
    ranked.sort(key=lambda x: -x[0])

    picked: List[Tuple[str, int, str]] = []
    concl_seen: set = set()
    for s, text, page, source in ranked:
        if s <= 0.05:
            break
        m = _CONCL_RE.search(text)
        if m:                                   # 同类结论只保留得分最高的一条
            if m.group(0) in concl_seen:
                continue
            concl_seen.add(m.group(0))
        elif concl_seen and text.startswith(("图内文字识别结果", "图内数值")):
            continue                            # 已有结论时不再堆砌原始 OCR 文本
        picked.append((text, page, source))
        if len(picked) >= max_sentences:
            break
    if not picked:
        return "", [], [], ""

    cites = sorted({p for _, p, _ in picked if p})
    src = max({s for _, _, s in picked if s},
              key=lambda x: sum(1 for _, _, y in picked if y == x), default="")
    page_ref = "、".join(f"第{p}页" for p in cites) if cites else "相关页面"
    doc_ref = f"《{src}》" if src else "招股说明书"
    body = "；".join(_tidy(t) for t, _, _ in picked)
    answer = f"根据{doc_ref}{page_ref}（图形内容解析）：{body}"
    evidences = [{"page": p, "text": _tidy(t), "source": s, "score": round(sc, 4)}
                 for sc, t, p, s in ranked[:len(picked)]]
    return answer, cites, evidences, src


# ---------------- 正文类问题路径 -----------------------------------------------
def _candidate_units(scored: List[ScoredDoc]) -> List[Tuple[str, int, float, str, str, str]]:
    """收集候选句 / 表格行：返回 (文本, 页码, 片段得分, 章节, 块类型, 来源)。"""
    units: List[Tuple[str, int, float, str, str, str]] = []
    seen = set()
    for sd in scored:
        md = sd.doc.metadata
        page = int(md.get("page", 0) or 0)
        section = md.get("section", "") or ""
        ctype = md.get("ctype", "text")
        source = md.get("source", "")
        sources = [sd.doc.page_content]
        parent = md.get("parent")
        if parent and parent != sd.doc.page_content:
            sources.append(parent)
        for src in sources:
            if src.strip().startswith("[表格]") or src.strip().startswith("|"):
                cands = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("|")]
            elif src.strip().startswith("[表格行]"):
                cands = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("[表格行]")]
            else:
                cands = split_sentences(src)
            for s in cands:
                key = s.replace(" ", "")
                if key in seen or len(key) < 6:
                    continue
                if _HEADING_RE.match(s.strip()) and len(s.strip()) <= 45:
                    continue
                seen.add(key)
                units.append((s, page, sd.score, section, ctype, source))
    return units


def build_answer(analysis: QueryAnalysis, scored: List[ScoredDoc],
                 max_sentences: int = config.EXTRACTIVE_MAX_SENTENCES) -> AnswerBundle:
    """基于检索结果合成抽取式答案。"""
    if not scored:
        return AnswerBundle(answer="根据招股说明书内容无法回答该问题。")

    # ---- 表格类问题：走表格专用路径 ----
    if analysis.answer_type == "table":
        ans, cites, evidences, src = _table_answer(analysis, scored)
        if ans:
            return AnswerBundle(answer=ans, citations=cites, evidences=evidences, source=src)

    # ---- 图形类问题：走图形语义专用路径（本工单新增） ----
    if analysis.answer_type == "figure":
        ans, cites, evidences, src = _figure_answer(analysis, scored)
        if ans:
            return AnswerBundle(answer=ans, citations=cites, evidences=evidences, source=src)

    # ---- 正文类问题：句子级抽取 ----
    units = _candidate_units(scored)
    if not units:
        return AnswerBundle(answer="根据招股说明书内容无法回答该问题。")

    ranked: List[Tuple[float, str, int, str]] = []
    for text, page, w, section, ctype, source in units:
        kw = _kw_cov(text, analysis.keywords)
        sig = _signal_score(text, analysis)
        ent = _ent_score(text, analysis.entities)
        defn = _defn_score(text, analysis.keywords)
        ln = _len_score(text)
        tbl = 1.0 if ctype in ("table", "table_kv") else 0.0
        sp = _summary_prior(section, analysis.keywords)
        pen = _contrast_penalty(text, analysis.keywords)
        bp = _boilerplate_penalty(text)
        s = (0.34 * kw + 0.16 * sig + 0.09 * ent + 0.09 * defn
             + 0.05 * tbl + 0.04 * ln + 0.05 * w + 0.18 * sp
             - 0.20 * pen - 0.15 * bp)
        ranked.append((s, text, page, source))
    ranked.sort(key=lambda x: x[0], reverse=True)

    picked: List[Tuple[str, int, float, str]] = []
    used_pages: dict[int, int] = {}
    for s, text, page, source in ranked:
        if s <= 0.15:
            break
        if used_pages.get(page, 0) >= 2:
            continue
        picked.append((text, page, s, source))
        used_pages[page] = used_pages.get(page, 0) + 1
        if len(picked) >= max_sentences:
            break
    if not picked:
        picked = [(ranked[0][1], ranked[0][2], ranked[0][0], ranked[0][3])]

    picked.sort(key=lambda x: -x[2])
    cites = sorted({p for _, p, _, _ in picked if p})
    src = picked[0][3] if picked else ""
    body = "；".join(_tidy(t) for t, _, _, _ in picked)
    page_ref = "、".join(f"第{p}页" for p in cites) if cites else "相关页面"
    doc_ref = f"《{src}》" if src else "招股说明书"
    answer = f"根据{doc_ref}{page_ref}：{body}"

    evidences = [
        {"page": p, "text": _tidy(t), "source": s, "score": round(sc, 4)}
        for t, p, sc, s in picked
    ]
    return AnswerBundle(answer=answer, citations=cites, evidences=evidences, source=src)