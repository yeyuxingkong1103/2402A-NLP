# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：第1层抽取 —— 词典 + 正则（硬性要求1、3）

设计取舍：
  · 全量 7757 块都跑本层。词典与正则是**确定性**的，零 LLM 成本、秒级完成、
    零幻觉，因而可作为图谱的**可信骨架**；
  · LLM（第2层）只在不确信的关系上补充，且必须带 evidence（见 kg_extract_llm）。

抽取产出的每一条关系都携带 provenance（source_doc / page / evidence），
支撑工单「返回答案所在的文本块」的核心诉求。
"""
from __future__ import annotations

import re
from typing import Iterator

from src.kg_dict import (COMPANY_TABLE, METRIC_TABLE, SUB_METRIC_QUALIFIERS,
                         find_companies, find_institutions, find_sectors,
                         find_metrics)
from src.kg_schema import Entity, Relation, make_entity_id

# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------
# 金额：可选千分位 + 可选小数 + 单位
_UNIT = r"(?:亿元|万元|百万元|千元|元|亿美元|万美元|万亿|亿|万)"
MONEY_RE = re.compile(rf"(\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*({_UNIT})")

# 百分比：数字 + %（容忍全角％）
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")

# 报告期
PERIOD_RE = re.compile(
    r"(20\d{2}\s*年(?:\s*(?:末|底|度|上半年|中期|第[一二三四]季度))?"
    r"|报告期末|报告期初|本报告期末|上年末|年初|本期|上期|上年同期)")

# 股票代码
CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

# 「较上年末下降0.10个百分点」之类的同比表述
YOY_RE = re.compile(
    r"(?:同比|较上年|比上年|同比增减|增幅|降幅)[^。；\n]{0,12}?"
    r"([+-]?\d+(?:\.\d+)?)\s*(?:个?百分点|[%％])")

# 单位到「元」的换算系数（用于跨文档口径统一）
UNIT_TO_YUAN = {
    "元": 1.0, "千元": 1e3, "万元": 1e4, "百万元": 1e6,
    "亿元": 1e8, "亿": 1e8, "万": 1e4, "万亿": 1e12,
    "亿美元": 1e8, "万美元": 1e4,
}


def parse_amount(text: str, start: int) -> dict | None:
    """从 text 的 start 位置起解析金额。

    返回 {"value": float, "unit": str, "raw": str}；无单位则返回 None
    （裸数字极易误判为页码/编号，不视为金额）。
    """
    m = MONEY_RE.match(text, start)
    if not m:
        return None
    return {"value": float(m.group(1).replace(",", "")),
            "unit": m.group(2), "raw": m.group(0)}


def extract_periods(text: str) -> list[dict]:
    """返回 [{"period": str, "start": int}]。"""
    return [{"period": m.group(1).replace(" ", ""), "start": m.start()}
            for m in PERIOD_RE.finditer(text)]


def _value_after(text: str, pos: int, unit_kind: str) -> dict | None:
    """在 pos 之后的一段窗口内找匹配 unit_kind 的数值。

    窗口取 30 字：年报中的「指标名 …… 数值」通常在同一个短句内。
    """
    window = text[pos:pos + 30]
    if unit_kind == "percent":
        m = PERCENT_RE.search(window)
        if m:
            return {"value": float(m.group(1)), "unit": "%", "raw": m.group(0)}
        return None
    if unit_kind == "money":
        m = MONEY_RE.search(window)
        if m:
            return {"value": float(m.group(1).replace(",", "")),
                    "unit": m.group(2), "raw": m.group(0)}
        return None
    return None


def _yoy_after(text: str, pos: int) -> float | None:
    window = text[pos:pos + 40]
    m = YOY_RE.search(window)
    return float(m.group(1)) if m else None


# 子类限定词的回看窗口（字）：限定词紧跟指标名出现才算同一口径，
# 例如「企业贷款不良率」「境内公司房地产贷款不良贷款率」。
QUALIFIER_LOOKBEHIND = 12

# 子句级规则：子句内出现「限定词 + 贷款/资产/存款/余额」子组合时，
# 该子句后续的裸指标名同属这一子组合口径。语料实例（招行 2019 年报）：
#   · P54「零售不良贷款余额170.54亿元，不良贷款率0.73%」
#   · P56「境内行标大型企业贷款余额12,697.37亿元……不良贷款率1.91%」
# 这类表述里限定词距指标名远超 12 字，紧邻窗口漏检。
def _qualifier_re(q: str) -> str:
    """限定词正则：允许 PDF 换行/断词插入空白（「商 业票据」「研发 投入」）。"""
    return r"\s*".join(re.escape(ch) for ch in q)


SUB_METRIC_SCOPE_RE = re.compile(
    "(?:" + "|".join(_qualifier_re(q) for q in SUB_METRIC_QUALIFIERS) +
    r")[^。！？；]{0,10}?(?:贷款|垫款|资产|存款|余额)")

# 表格块有限回看专用：只用「限定词+贷款/余额」语义明确的强限定词，
# 避免同表列名「零售贷款占比」误伤紧随其后的总口径行。
SUB_METRIC_STRONG_QUALIFIERS: list[str] = [
    "企业贷款", "公司贷款", "个人贷款", "零售贷款", "消费贷款",
    "大型企业", "中型企业", "小企业", "应收账款", "商业票据",
]
SUB_METRIC_TABLE_SCOPE_RE = re.compile(
    "(?:" + "|".join(_qualifier_re(q)
                     for q in SUB_METRIC_STRONG_QUALIFIERS) +
    r")[^。！？；]{0,10}?(?:贷款|垫款|资产|存款|余额)")
SUB_METRIC_CLAUSE_MAX = 200     # 子句回看上限（防超长句拖慢）
SUB_METRIC_TABLE_LOOKBACK = 100  # 表格块有限回看（跨行子类口径折中窗口）

# 子句边界：句末标点与分号。逗号不算边界 ——
# 「其中，小微不良贷款处置加快，不良贷款率0.81%」的限定词在逗号之前；
# 换行也不算 —— PDF 抽取文本会硬折行（「较年\n初下降0.86个百分点」），
# 若按换行切句会把限定词切出子句，造成漏检。
_CLAUSE_BOUNDARY_RE = re.compile(r"[。！？；;]")


def _clause_prefix(text: str, pos: int,
                   max_back: int = SUB_METRIC_CLAUSE_MAX) -> str:
    """取 pos 所在子句的起点到 pos 的文本（以句末/分号/换行为界）。"""
    start = max(0, pos - max_back)
    window = text[start:pos]
    last = None
    for m in _CLAUSE_BOUNDARY_RE.finditer(window):
        last = m
    return window[last.end():] if last else window


# 指标名后紧跟的括号限定：「不良贷款率（包括商业票据）0.51%」
# 括号内出现限定词 → 该数值是子类口径（崩点在指标名之后，前缀规则覆盖不到）。
_POST_PAREN_RE = re.compile(r"[（(]([^）)]{1,16})[）)]")


def _has_qualifying_parenthetical(text: str, end: int) -> bool:
    m = _POST_PAREN_RE.match(text, end)
    if not m:
        return False
    inner = re.sub(r"\s+", "", m.group(1))     # PDF 折行会在词中插空格
    return any(q in inner for q in SUB_METRIC_QUALIFIERS)


def _is_sub_metric_mention(text: str, start: int,
                           window: int = QUALIFIER_LOOKBEHIND,
                           table: bool = False, end: int | None = None) -> bool:
    """匹配点前出现子类限定词 → 该处是子类口径，拒绝入图。

    谓词限定词（企业贷款/公司贷款/信用卡/行业/境内…）与指标名的组合
    定义的是**另一个指标**，其数值不能记到总口径指标名下。

    三条规则：
      1. 紧邻（12 字内）出现限定词；
      2. 非表格块中，当前子句（。！？；\\n 为界）内出现
         「限定词+贷款/资产/存款/余额」子组合；
      3. 指标名后紧跟「（……限定词……）」括号限定（end 为指标名结束位置）。

    表格块（block_type=table）只做有限回看（SUB_METRIC_TABLE_LOOKBACK 字）：
    表格文本没有句读，按子句回看等于整表，跨单元格把「零售贷款占比」
    这类列名误当限定词；但完全不回看又会漏掉「个人贷款不良贷款余额…
    不良贷款率0.99%」这类跨行子类口径，故取有限窗口折中。

    text 传入的应是**父块文本**（chunk.parent_text，若可用）——
    分块可能把限定词切在上一块，只用本块文本会漏检。
    """
    prefix = text[max(0, start - window):start]
    if any(q in prefix for q in SUB_METRIC_QUALIFIERS):
        return True
    prefix_ns = re.sub(r"\s+", "", prefix)      # 去折行空格后的紧邻判定
    if prefix_ns != prefix and any(q in prefix_ns
                                   for q in SUB_METRIC_QUALIFIERS):
        return True
    if end is not None and _has_qualifying_parenthetical(text, end):
        return True
    if table:
        back = text[max(0, start - SUB_METRIC_TABLE_LOOKBACK):start]
        return bool(SUB_METRIC_TABLE_SCOPE_RE.search(back))
    return bool(SUB_METRIC_SCOPE_RE.search(_clause_prefix(text, start)))


def _nearest_company(text: str, pos: int, companies: list[tuple[str, int, int]]):
    """取 pos 之前最近的公司；没有则取之后最近的；都没有返回 None。"""
    before = [c for c in companies if c[2] <= pos]
    if before:
        return before[-1][0]
    after = [c for c in companies if c[1] >= pos]
    return after[0][0] if after else None


def _company_from_source(source: str) -> str | None:
    """从文件名推断公司（语料每份文件即一家公司的年报）。

    文件名形如 `{日期}__{公司简称}__{代码}__{年度}年度报告.pdf`。
    """
    for canon, meta in COMPANY_TABLE.items():
        if canon in source or meta["code"] in source:
            return canon
    return None


# ---------------------------------------------------------------------------
# 组织隶属：文本 → (父组织, [子组织])（服务组织结构图类问题）
# ---------------------------------------------------------------------------
# 触发词：只有这三种「上位→下位」的列举句式才建组织边，
# 不用「包括/设有」——它们在年报里泛指太多（「包括公司章程、股东大会议事规则…」）。
ORG_TRIGGERS = ("下设", "下辖", "内设")

# 组织单元结尾词：父节点必须以此结尾（否则不是组织名）
_ORG_TAILS = ("管理委员会", "委员会", "董事会", "监事会", "总行", "分行",
              "分公司", "支行", "事业部", "事业群", "办公室", "中心",
              "管理部", "部门", "部", "处", "科", "室", "组", "板块")

# 泛指词：单独出现时是类别而非物名，不入图
_GENERIC_ORG = {"部门", "单位", "直属单位", "机构", "委员会", "公司", "分行",
                "支行", "分公司", "中心", "办公室", "组", "部", "室", "板块",
                "人员", "管理人员"}

# 名字片段：触发词前紧邻的、不含分隔符/空白的连续串
_RUN_RE = re.compile(r"[^，。；;：:、（）()\s]{2,15}$")
# 并列连接词：「…董事会和第十一届董事会均下设」取最后一个并列项
_CONJ_RE = re.compile(r"[和与及]")
# 前修饰语（「本行第十届董事会」→「董事会」）与后副词（「董事会均下设」→「董事会」）
_LEAD_NOISE_RE = re.compile(
    r"^(?:本行|本公司|本集团|本企业|我行|公司|该行|该|其|上述|前述|集团|"
    r"第[一二三四五六七八九十百]+[届次]|在|由|经|自|向|从|对|为)+")
_TAIL_NOISE_RE = re.compile(r"(?:均|都|也|亦|并|则|共|各|同时|分别)+$")
# 头里出现这些词说明命中的是句子片段（「各位监事分别列席董事会」），不是组织名
_HEAD_BAD_RE = re.compile(
    r"(列席|各位|分别|要求|作为|负责|通过|根据|由于|对于|关于|同时|其中|"
    r"进行|包括|存在|属于|以及|了)")
# 子节点里出现这些词说明句子在叙述而非列举
_CHILD_BAD_RE = re.compile(
    r"(分为|包括|负责|由|是|进行|通过|以及|其中|主要|相关|下设|下辖|内设|"
    r"作为|要求|各位|分别|列席|了)")
# 组织名里几乎不会出现阿拉伯数字；出现即为规模描述（「42家分公司」）
_CHILD_DIGIT_RE = re.compile(r"[0-9０-９]")


def _ws_norm(s: str) -> str:
    """去空白：PDF 折行会把名字切断（「信息传\\n媒行业组」）。"""
    return re.sub(r"\s+", "", s or "")


def _org_child_ok(s: str) -> bool:
    """判断一个顿号片段是否是合法的组织物名。"""
    if not s or len(s) > 20 or s in _GENERIC_ORG:
        return False
    if _CHILD_DIGIT_RE.search(s) or _CHILD_BAD_RE.search(s):
        return False
    core = re.sub(r"[（(][^）)]*[）)]$", "", s)      # 「综合行业组（北京）」→「综合行业组」
    return core.endswith(_ORG_TAILS)


def _org_head_before(text: str, pos: int) -> tuple[str | None, int]:
    """取触发词之前紧邻的组织名，返回 (head, 起始下标)。

    处理三类真实写法：
      · 「本行第十届董事会下设」        → 董事会（剥前修饰语）
      · 「董事会和第十一届董事会均下设」 → 董事会（取最后一个并列项、剥后副词）
      · 「注1：投资银行管理委员会下设」  → 投资银行管理委员会（标点后取整名）
    """
    win_start = max(0, pos - 20)
    m = _RUN_RE.search(text[win_start:pos])
    if not m:
        return None, -1
    run_start = win_start + m.start()
    cand = _CONJ_RE.split(m.group(0))[-1]
    cand = _ws_norm(_TAIL_NOISE_RE.sub("", _LEAD_NOISE_RE.sub("", cand)))
    if not (2 <= len(cand) <= 15) or _HEAD_BAD_RE.search(cand):
        return None, -1
    if cand not in COMPANY_TABLE and not cand.endswith(_ORG_TAILS):
        return None, -1
    start = text.rfind(cand, run_start, pos)       # 定位真实下标（供 evidence 切片）
    return (cand, start) if start >= 0 else (None, -1)


def _org_children_after(text: str, pos: int) -> tuple[list[str], int]:
    """取触发词之后的列举项，返回 (children, 结束下标)。"""
    end = len(text)
    # 在「等」处截断（「等6个专门委员会」）；在下一个触发词处截断，
    # 否则前一个「下设」会把后一句的列举一并吞进来。
    # 注意：**不**在换行处截断 —— 年报正文按版心折行，「信息传\n媒行业组」
    # 这类断字就在列表中间，按行截断会把长列表砍成前几项。
    for ch in list("。；;等") + list(ORG_TRIGGERS):
        i = text.find(ch, pos)
        if i != -1:
            end = min(end, i)
    items: list[str] = []
    for piece in re.split(r"[、，,]", text[pos:end]):
        parts = [_ws_norm(p) for p in re.split(r"和|以及", piece) if p.strip()]
        if len(parts) > 1 and not all(_org_child_ok(p) for p in parts):
            parts = [_ws_norm(piece)]               # 「和」是名字的一部分，不拆
        for p in parts:
            if _org_child_ok(p) and p not in items:
                items.append(p)
    return items, end


def find_org_relations(text: str) -> list[dict]:
    """从块文本中抽取组织隶属关系。

    返回 [{"parent", "parent_type", "children", "trigger",
           "evidence", "start", "end"}]；
    抽取不到的句式一律返回空列表 —— 宁可漏抽，不可把「高级管理层及其下设委员会」
    这类泛指表述当成组织树写进图谱（错误的树比没有树更糟）。
    """
    out: list[dict] = []
    seen: set[tuple] = set()
    for trig in ORG_TRIGGERS:
        for m in re.finditer(trig, text):
            head, hstart = _org_head_before(text, m.start())
            if not head:
                continue
            children, cend = _org_children_after(text, m.end())
            if len(children) < 2:
                continue
            # 「等N个」是原文自带的对账口径：对不上说明该块文本已损坏
            # （典型是双栏 PDF 行交错把物名打断），整组不抽。
            m_n = re.match(r"等\s*(\d+)\s*个", text[cend:cend + 12])
            if m_n and len(children) < int(m_n.group(1)):
                continue
            key = (head, tuple(children))
            if key in seen:
                continue
            seen.add(key)
            ptype = "Company" if head in COMPANY_TABLE else "Department"
            out.append({"parent": head, "parent_type": ptype,
                        "children": children, "trigger": trig,
                        "evidence": text[hstart:cend],
                        "start": hstart, "end": cend})
    return out


def extract_block(chunk: dict) -> dict:
    """对单个块做第1层抽取。

    返回 {"entities": [Entity], "relations": [Relation], "mentions": [dict]}。
    mentions 记录指标的原文位置，供第2层与可视化定位。
    """
    text = chunk.get("text") or ""
    if not text.strip():
        return {"entities": [], "relations": [], "mentions": []}

    source = chunk.get("source", "")
    page = int(chunk.get("page_idx", -1))
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []
    mentions: list[dict] = []

    def _add_entity(e: Entity) -> None:
        if e.entity_id not in entities:
            entities[e.entity_id] = e
        else:  # 合并别名
            cur = entities[e.entity_id]
            for a in e.aliases:
                if a not in cur.aliases:
                    cur.aliases.append(a)

    companies = find_companies(text)
    for canon, _s, _e in companies:
        meta = COMPANY_TABLE[canon]
        _add_entity(Entity(
            entity_id=make_entity_id("Company", canon), name=canon, type="Company",
            aliases=list(meta["aliases"]),
            attrs={"code": meta["code"], "industry": meta["industry"],
                   "full": meta["full"], "en": meta["en"]}))

    # 公司 → 行业
    for canon, s, e in companies:
        for ind in {COMPANY_TABLE[canon]["industry"]}:
            if ind in text:
                _add_entity(Entity(make_entity_id("Industry", ind), ind, "Industry"))
                relations.append(Relation(
                    make_entity_id("Company", canon), "IN_INDUSTRY",
                    make_entity_id("Industry", ind),
                    evidence=text[max(0, s - 10):e + 10], source_doc=source,
                    page=page))

    for canon, _s, _e in find_institutions(text):
        _add_entity(Entity(make_entity_id("Institution", canon), canon,
                           "Institution"))
    for canon, _s, _e in find_sectors(text):
        _add_entity(Entity(make_entity_id("Department", canon), canon,
                           "Department"))

    # 组织隶属：「董事会下设审计委员会、风险管理委员会…」→ BELONGS_TO（子 → 父），
    # 这是 org_tree 检索模式与组织结构图类问题的图谱依据。
    for org in find_org_relations(text):
        ptype, pname = org["parent_type"], org["parent"]
        pid = make_entity_id(ptype, pname)
        _add_entity(Entity(pid, pname, ptype))
        for child in org["children"]:
            cid = make_entity_id("Department", child)
            _add_entity(Entity(cid, child, "Department"))
            relations.append(Relation(
                cid, "BELONGS_TO", pid,
                evidence=org["evidence"], source_doc=source, page=page))

    # 指标 + 数值 + 报告期 → Metric 节点、HAS_METRIC / REPORTED_IN 边
    periods = extract_periods(text)
    is_table = chunk.get("block_type") == "table"
    # 子类口径守卫用父块文本：分块可能把「限定词+贷款余额」（如
    # 「境内行标中型企业贷款余额…不良贷款率5.16%」）切在上一块。
    parent = chunk.get("parent_text") or ""
    guard_text, guard_shift = text, 0
    if parent:
        _idx = parent.find(text)
        if _idx >= 0:
            guard_text, guard_shift = parent, _idx
    for canon, surface, s, e in find_metrics(text):
        if _is_sub_metric_mention(guard_text, guard_shift + s, table=is_table,
                                  end=guard_shift + e):
            continue                      # 子类口径（企业贷款不良率…）不是该指标
        spec = METRIC_TABLE[canon]
        val = _value_after(text, e, spec["unit_kind"])
        if val is None:
            continue                      # 只提到指标名、没有数值 → 不入图
        yoy = _yoy_after(text, e)
        # 块内没写公司名时（单公司年报的大多数块），按来源文件名归属公司
        company = _nearest_company(text, s, companies) or _company_from_source(source)
        period = "本期"
        for p in periods:
            if p["start"] <= s:
                period = p["period"]

        metric_name = f"{canon}({val['value']}{val['unit']})" + (
            f"@{period}" if period else "")
        metric_id = make_entity_id("Metric", metric_name)
        _add_entity(Entity(
            entity_id=metric_id, name=metric_name, type="Metric",
            aliases=[canon],
            attrs={"metric": canon, "value": val["value"], "unit": val["unit"],
                   "yoy": yoy, "period": period, "raw": val["raw"],
                   "source_doc": source, "page": page,
                   "industry": COMPANY_TABLE[company]["industry"] if company else ""}))

        mentions.append({"metric": canon, "surface": surface, "start": s,
                         "value": val["value"], "unit": val["unit"],
                         "period": period, "company": company})

        if period and period != "本期":
            time_id = make_entity_id("Time", period)
            _add_entity(Entity(time_id, period, "Time"))
            relations.append(Relation(metric_id, "REPORTED_IN", time_id,
                                      evidence=text[max(0, s - 15):e + 25],
                                      source_doc=source, page=page))

        if company:
            # 边上直接带 value/unit：HAS_METRIC 即可单独回答「某公司某指标取值」
            relations.append(Relation(
                make_entity_id("Company", company), "HAS_METRIC", metric_id,
                evidence=text[max(0, s - 15):e + 25], source_doc=source,
                page=page, attrs={"yoy": yoy, "value": val["value"],
                                  "unit": val["unit"]}))

    return {"entities": list(entities.values()), "relations": relations,
            "mentions": mentions}


def iter_extract(chunks: list[dict], limit: int | None = None) -> Iterator[dict]:
    """批量抽取（生成器）。yield {"chunk_id", "entities", "relations", "mentions"}。"""
    for i, chunk in enumerate(chunks):
        if limit is not None and i >= limit:
            break
        got = extract_block(chunk)
        yield {"chunk_id": chunk.get("chunk_id"), **got}
