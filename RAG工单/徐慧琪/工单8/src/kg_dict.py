# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：金融领域词典（硬性要求1）

为什么以词典为主干：
  · 语料是**固定 9 家公司**的年报，公司名/指标名高度可枚举；
  · 词典匹配**零成本、可复现、零幻觉**，比 LLM 更可靠地覆盖高频实体；
  · LLM 只用于补充词典覆盖不到的长尾实体与关系（见 kg_extract_llm）。

所有词典项均为语料中真实出现的表述，可用 scripts/validate_ground_truth.py 复核。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1) 公司：9 家上市公司（语料全量）
# ---------------------------------------------------------------------------
COMPANY_TABLE: dict[str, dict] = {
    "平安银行": {
        "full": "平安银行股份有限公司", "code": "000001", "industry": "银行",
        "en": "Ping An Bank",
        "aliases": ["平安银行股份有限公司", "平安银行", "平安银行股份"],
    },
    "中国平安": {
        "full": "中国平安保险（集团）股份有限公司", "code": "601318",
        "industry": "保险", "en": "Ping An Insurance",
        "aliases": ["中国平安保险（集团）股份有限公司", "中国平安保险集团股份有限公司",
                    "中国平安", "平安集团"],
    },
    "招商银行": {
        "full": "招商银行股份有限公司", "code": "600036", "industry": "银行",
        "en": "China Merchants Bank",
        "aliases": ["招商银行股份有限公司", "招商银行", "招行"],
    },
    "邮储银行": {
        "full": "中国邮政储蓄银行股份有限公司", "code": "601658", "industry": "银行",
        "en": "Postal Savings Bank of China",
        "aliases": ["中国邮政储蓄银行股份有限公司", "中国邮政储蓄银行", "邮储银行"],
    },
    "中信证券": {
        "full": "中信证券股份有限公司", "code": "600030", "industry": "证券",
        "en": "CITIC Securities",
        "aliases": ["中信证券股份有限公司", "中信证券"],
    },
    "中国人寿": {
        "full": "中国人寿保险股份有限公司", "code": "601628", "industry": "保险",
        "en": "China Life Insurance",
        "aliases": ["中国人寿保险股份有限公司", "中国人寿"],
    },
    "中国太保": {
        "full": "中国太平洋保险（集团）股份有限公司", "code": "601601",
        "industry": "保险", "en": "China Pacific Insurance",
        "aliases": ["中国太平洋保险（集团）股份有限公司",
                    "中国太平洋保险集团股份有限公司", "中国太平洋保险", "中国太保"],
    },
    "招商证券": {
        "full": "招商证券股份有限公司", "code": "600999", "industry": "证券",
        "en": "China Merchants Securities",
        "aliases": ["招商证券股份有限公司", "招商证券"],
    },
    "国泰君安": {
        "full": "国泰君安证券股份有限公司", "code": "601211", "industry": "证券",
        "en": "Guotai Junan Securities",
        "aliases": ["国泰君安证券股份有限公司", "国泰君安证券", "国泰君安"],
    },
}

# ---------------------------------------------------------------------------
# 2) 指标：金融核心指标（unit_kind 决定数值解析与答案校验方式）
# ---------------------------------------------------------------------------
METRIC_TABLE: dict[str, dict] = {
    # —— 银行 ——
    # 注意：「不良率」不得作为别名 —— 它是「企业贷款不良率」「公司贷款不良率」
    # 「信用卡应收账款不良率」等**子类指标**的后缀，会把子类取值错误归到总口径。
    "不良贷款率": {"aliases": ["不良贷款率"], "unit_kind": "percent"},
    "拨备覆盖率": {"aliases": ["拨备覆盖率"], "unit_kind": "percent"},
    "拨贷比": {"aliases": ["拨贷比"], "unit_kind": "percent"},
    "资本充足率": {"aliases": ["资本充足率", "资本充足率（集团）"], "unit_kind": "percent"},
    "核心一级资本充足率": {"aliases": ["核心一级资本充足率", "核心一级资本充足率（集团）"],
                          "unit_kind": "percent"},
    "一级资本充足率": {"aliases": ["一级资本充足率"], "unit_kind": "percent"},
    "净息差": {"aliases": ["净息差", "净利息收益率"], "unit_kind": "percent"},
    "成本收入比": {"aliases": ["成本收入比"], "unit_kind": "percent"},
    "不良贷款余额": {"aliases": ["不良贷款余额"], "unit_kind": "money"},
    "贷款总额": {"aliases": ["贷款总额", "发放贷款和垫款总额", "发放贷款及垫款总额"],
                 "unit_kind": "money"},
    "存款总额": {"aliases": ["存款总额", "吸收存款总额"], "unit_kind": "money"},
    "零售业务营业收入": {"aliases": ["零售业务营业收入", "零售金融业务营业收入"],
                         "unit_kind": "money"},
    # —— 保险 ——
    "内含价值": {"aliases": ["内含价值", "集团内含价值"], "unit_kind": "money"},
    "新业务价值": {"aliases": ["新业务价值", "一年新业务价值"], "unit_kind": "money"},
    "保费收入": {"aliases": ["保费收入", "保险业务收入", "已赚保费"], "unit_kind": "money"},
    "总投资收益": {"aliases": ["总投资收益"], "unit_kind": "money"},
    "总投资收益率": {"aliases": ["总投资收益率", "总投资收益率（年化）"],
                     "unit_kind": "percent"},
    "综合偿付能力充足率": {"aliases": ["综合偿付能力充足率"], "unit_kind": "percent"},
    "核心偿付能力充足率": {"aliases": ["核心偿付能力充足率"], "unit_kind": "percent"},
    "归母营运利润": {"aliases": ["归属于母公司股东的营运利润", "归母营运利润"],
                     "unit_kind": "money"},
    "营运ROE": {"aliases": ["营运ROE", "营运 ROE"], "unit_kind": "percent"},
    "投资资产": {"aliases": ["投资资产"], "unit_kind": "money"},
    # —— 证券 ——
    "经纪业务手续费净收入": {
        "aliases": ["经纪业务手续费净收入", "经纪业务手续费及佣金净收入"],
        "unit_kind": "money"},
    "投资银行业务手续费净收入": {
        "aliases": ["投资银行业务手续费净收入", "投资银行业务手续费及佣金净收入",
                    "投资银行业务手续费净收入"],
        "unit_kind": "money"},
    "资产管理业务手续费净收入": {
        "aliases": ["资产管理业务手续费净收入", "资产管理业务手续费及佣金净收入"],
        "unit_kind": "money"},
    "利息净收入": {"aliases": ["利息净收入", "利息净收入（元）"], "unit_kind": "money"},
    "手续费及佣金净收入": {"aliases": ["手续费及佣金净收入"], "unit_kind": "money"},
    # —— 通用财务 ——
    "营业收入": {"aliases": ["营业收入", "营业总收入", "营业收入合计"],
                 "unit_kind": "money"},
    # 注意：不得把裸「净利润」列为别名 —— 它是「子公司净利润」「少数股东的
    # 净利润」等子类表述的子串，且会遮蔽同名的总口径指标「净利润」。
    "归属于母公司股东的净利润": {
        "aliases": ["归属于母公司股东的净利润", "归属于母公司所有者的净利润",
                    "归属于本行股东的净利润", "归属于上市公司股东的净利润",
                    "归母净利润"],
        "unit_kind": "money"},
    "净利润": {"aliases": ["净利润"], "unit_kind": "money"},
    "利润总额": {"aliases": ["利润总额"], "unit_kind": "money"},
    "资产总计": {"aliases": ["资产总计", "资产总额", "总资产"], "unit_kind": "money"},
    "负债合计": {"aliases": ["负债合计", "负债总额"], "unit_kind": "money"},
    "归属于母公司股东的权益": {
        "aliases": ["归属于母公司股东的权益", "归属于母公司股东权益合计",
                    "归属于本行股东权益"], "unit_kind": "money"},
    "经营活动产生的现金流量净额": {
        "aliases": ["经营活动产生的现金流量净额"], "unit_kind": "money"},
    "基本每股收益": {"aliases": ["基本每股收益"], "unit_kind": "money"},
    "加权平均净资产收益率": {
        "aliases": ["加权平均净资产收益率", "加权平均净资产收益率（%）"],
        "unit_kind": "percent"},
    "信息科技投入": {"aliases": ["信息科技投入", "科技投入"], "unit_kind": "money"},
    "现金分红": {"aliases": ["现金分红", "每股现金分红", "股息"], "unit_kind": "money"},
}

# ---------------------------------------------------------------------------
# 2b) 子类指标限定词（限定词 + 指标名 = 另一个指标的取值）
# ---------------------------------------------------------------------------
# 语料中大量出现「<品种/区域/行业> + 指标名」的子类表述，例如
# 「企业贷款不良率」「境内公司房地产贷款不良贷款率」「制造业不良贷款率」。
# 这些是**不同于总口径指标**的取值，抽取时必须拒绝，否则公司总口径指标
# 会被子类数值淹没（如把 6.07% 的压缩退出类行业不良率当成招商银行不良贷款率）。
# 由 kg_extract 在匹配点的前视窗口内做守卫；纯数据，便于随语料扩充。
SUB_METRIC_QUALIFIERS: list[str] = [
    # 贷款品种/科目
    "企业贷款", "公司贷款", "个人贷款", "零售贷款", "零售", "消费贷款",
    "信用卡", "应收账款", "商业票据",
    # 客户规模/标尺口径（招行「境内行标大型/中型/小企业贷款」等）
    "小微", "大型企业", "中型企业", "小企业", "行标",
    # 客户/机构/区域口径
    "对公", "境外", "境内", "子公司",
    # 行业口径（不良贷款按行业披露表）
    "行业", "房地产业", "房地产", "制造业", "批发和零售业", "建筑业",
    "交通运输", "租赁和商务服务", "采矿业", "电力", "水利", "农林牧渔",
]

# ---------------------------------------------------------------------------
# 3) 其他实体词表
# ---------------------------------------------------------------------------
INSTITUTIONS: list[str] = [
    "中国银行保险监督管理委员会", "中国银保监会", "银保监会",
    "中国人民银行", "央行",
    "中国证券监督管理委员会", "中国证监会", "证监会",
    "上海证券交易所", "深圳证券交易所", "香港联合交易所", "上交所", "深交所",
    "财政部", "国务院", "国家外汇管理局",
    "中国银行业协会", "中国证券业协会", "中国保险行业协会",
    "国家金融监督管理总局",
]

INDUSTRIES: list[str] = ["银行", "保险", "证券", "银行业", "保险业", "证券业",
                         "金融", "基金", "信托", "资管", "财富管理"]

SECTORS: list[str] = ["个人金融板块", "普惠金融板块", "公司金融板块", "资金资管板块",
                      "风险管理板块", "信息科技板块", "综合支撑板块"]

# 图谱触发词：块中出现这些词，说明可能含实体间关系，值得送 LLM 抽取
GRAPH_TRIGGERS: list[str] = [
    "控股", "持股", "参股", "股东", "子公司", "全资子公司", "控股子公司",
    "董事", "监事", "高级管理人员", "任职", "聘任", "兼任", "委任",
    "投资", "发行", "回购", "分红", "派息", "并购", "收购", "重组",
    "组织结构", "组织架构", "内设部门", "直属单位", "下设",
    "同业", "竞争", "市场份额", "排名",
]

# LLM 抽取时的章节白名单（匹配 chunk 的 heading_path）
SECTION_WHITELIST: list[str] = [
    "公司治理", "组织结构", "组织架构", "股东", "股本", "董事", "监事",
    "高级管理人员", "主要会计数据", "业务概要", "经营情况讨论与分析",
    "风险管理", "子公司", "关联方", "关联交易", "财务报表附注",
    "董事长致辞", "发展战略", "核心竞争力", "员工",
]

# ---------------------------------------------------------------------------
# 4) 匹配实现
# ---------------------------------------------------------------------------
# 长词优先：先匹配「招商银行股份有限公司」再匹配「招商银行」，
# 避免「招商证券」被截断成「招商」等误匹配。
_COMPANY_SURFACES: list[tuple[str, str]] = sorted(
    [(alias, canonical)
     for canonical, meta in COMPANY_TABLE.items()
     for alias in meta["aliases"]],
    key=lambda x: -len(x[0]),
)

_METRIC_SURFACES: list[tuple[str, str]] = sorted(
    [(alias, canonical)
     for canonical, meta in METRIC_TABLE.items()
     for alias in meta["aliases"]],
    key=lambda x: -len(x[0]),
)

_INSTITUTION_SURFACES: list[tuple[str, str]] = sorted(
    [(s, s) for s in INSTITUTIONS], key=lambda x: -len(x[0]))

# 英文公司名（问句侧链接用；多语言支持硬性要求）。
# 只用于**问题**匹配，不参与语料抽取 —— 避免改变已建图谱的抽取结果。
_EN_COMPANY_SURFACES: list[tuple[str, str]] = sorted(
    [(meta["en"], canonical)
     for canonical, meta in COMPANY_TABLE.items() if meta.get("en")],
    key=lambda x: -len(x[0]),
)


def _scan(text: str, surfaces: list[tuple[str, str]]) -> list[tuple[str, str, int, int]]:
    """在 text 中扫描 surfaces（长词优先），返回不重叠命中。

    返回 [(规范名, 原文片段, start, end)]，按出现位置排序。
    同一位置被长词占用后，短词不再重复命中（避免「招商银行股份有限公司」
    被同时记为长名与「招商银行」两次）。
    """
    taken = [False] * len(text)
    found: list[tuple[str, str, int, int]] = []
    for surface, canonical in surfaces:
        start = 0
        while True:
            i = text.find(surface, start)
            if i < 0:
                break
            j = i + len(surface)
            if not any(taken[i:j]):
                for k in range(i, j):
                    taken[k] = True
                found.append((canonical, surface, i, j))
            start = i + 1
    found.sort(key=lambda x: x[2])
    return found


def find_companies(text: str) -> list[tuple[str, int, int]]:
    """返回 [(规范简称, start, end)]。"""
    return [(c, s, e) for c, _surface, s, e in _scan(text, _COMPANY_SURFACES)]


def find_companies_en(text: str) -> list[tuple[str, int, int]]:
    """英文公司名匹配（大小写不敏感），返回 [(规范简称, start, end)]。

    英文提问若链接不到实体，就会退化成纯文本通道（图谱事实为 0），
    答案质量明显下降 —— 这是多语言支持的关键一环。
    """
    taken = [False] * len(text)
    found: list[tuple[str, int, int]] = []
    lower = text.lower()
    for surface, canonical in _EN_COMPANY_SURFACES:
        needle = surface.lower()
        start = 0
        while True:
            i = lower.find(needle, start)
            if i < 0:
                break
            j = i + len(needle)
            if not any(taken[i:j]):
                for k in range(i, j):
                    taken[k] = True
                found.append((canonical, i, j))
            start = i + 1
    found.sort(key=lambda x: x[1])
    return found


def find_metrics(text: str) -> list[tuple[str, str, int, int]]:
    """返回 [(规范指标名, 原文片段, start, end)]。"""
    return _scan(text, _METRIC_SURFACES)


def find_institutions(text: str) -> list[tuple[str, int, int]]:
    """返回 [(规范机构名, start, end)]。"""
    return [(c, s, e) for c, _surface, s, e in _scan(text, _INSTITUTION_SURFACES)]


def find_sectors(text: str) -> list[tuple[str, int, int]]:
    """返回 [(板块名, start, end)]。"""
    return [(c, s, e) for c, _surface, s, e in _scan(
        text, sorted([(s, s) for s in SECTORS], key=lambda x: -len(x[0])))]


def company_by_code(code: str) -> str | None:
    """股票代码 → 规范简称。"""
    for name, meta in COMPANY_TABLE.items():
        if meta["code"] == code:
            return name
    return None
