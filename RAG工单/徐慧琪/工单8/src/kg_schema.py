# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：知识图谱本体定义与校验（硬性要求1、3）

本体从**语料结构**出发设计：9 份上市公司年报具有固定的章节结构
（公司治理、董监高、股东、组织结构图、主要会计数据、业务概要、风险管理），
据此可稳定抽取的实体/关系如下。

实体类型（10 类）与关系类型（11 类）见下方常量；
每条关系的合法头尾类型在 RELATION_TYPES 中显式声明，供抽取时校验与约束 LLM 输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 实体类型
# ---------------------------------------------------------------------------
ENTITY_TYPES: dict[str, str] = {
    "Company": "公司（上市公司、控股股东、子公司、同业机构）",
    "Department": "部门/板块/分支机构（个人金融部、信用卡中心、个人金融板块）",
    "Person": "人物（董监高、股东）",
    "Institution": "机构（监管机构、交易所、政府部门）",
    "Industry": "行业（银行、保险、证券）",
    "Business": "业务/产品（零售金融、投资银行、财富管理）",
    "Metric": "指标（不良贷款率、拨备覆盖率、营业收入、内含价值）",
    "Event": "事件（发行、分红、回购、并购）",
    "Time": "时间（2019年度、报告期末、本期/上期）",
    "Location": "地区（境内、境外、某省）",
}

# ---------------------------------------------------------------------------
# 关系类型：head/tail 为该关系的合法实体类型集合（空集合表示不限）
# ---------------------------------------------------------------------------
RELATION_TYPES: dict[str, dict] = {
    "HAS_METRIC": {
        "desc": "公司拥有该指标取值", "head": ["Company"], "tail": ["Metric"],
        "directed": True,
    },
    "REPORTED_IN": {
        "desc": "指标所属报告期", "head": ["Metric"], "tail": ["Time"],
        "directed": True,
    },
    "BELONGS_TO": {
        "desc": "组织隶属（部门→部门/公司）",
        "head": ["Department"], "tail": ["Department", "Company"],
        "directed": True,
    },
    "IN_INDUSTRY": {
        "desc": "所属行业", "head": ["Company"], "tail": ["Industry"],
        "directed": True,
    },
    "CONTROLS": {
        "desc": "控股关系", "head": ["Company"], "tail": ["Company"],
        "directed": True,
    },
    "SHAREHOLDER_OF": {
        "desc": "股东关系", "head": ["Company", "Person"],
        "tail": ["Company"], "directed": True,
    },
    "SERVES_AS": {
        "desc": "任职关系（人物在某公司任某职）",
        "head": ["Person"], "tail": ["Company"], "directed": True,
    },
    "OPERATES": {
        "desc": "经营某业务", "head": ["Company"], "tail": ["Business"],
        "directed": True,
    },
    "COMPETES_WITH": {
        "desc": "同业竞争", "head": ["Company"], "tail": ["Company"],
        "directed": False,
    },
    "INVESTS_IN": {
        "desc": "投资关系", "head": ["Company"], "tail": ["Company", "Business"],
        "directed": True,
    },
    "ISSUES": {
        "desc": "发行关系", "head": ["Company"], "tail": ["Event"],
        "directed": True,
    },
}

# ---------------------------------------------------------------------------
# 可视化配色（对齐工单图谱示例：多类型实体用不同颜色节点）
# ---------------------------------------------------------------------------
ENTITY_COLORS: dict[str, str] = {
    "Company":     "#2E6FD9",   # 蓝
    "Department":  "#37A96B",   # 绿
    "Person":      "#D98324",   # 橙
    "Institution": "#8250C4",   # 紫
    "Industry":    "#0E9AA7",   # 青
    "Business":    "#C94F7C",   # 品红
    "Metric":      "#D64545",   # 红
    "Event":       "#8B6F47",   # 棕
    "Time":        "#6B7280",   # 灰
    "Location":    "#4B8C3F",   # 深绿
}

_DEFAULT_COLOR = "#9AA0A6"


def node_color(etype: str) -> str:
    """实体类型 → 十六进制颜色（未知类型用中性灰）。"""
    return ENTITY_COLORS.get(etype, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    """图谱实体节点。

    entity_id 由 make_entity_id 生成，格式 `{类型}:{规范名}`，
    是跨文档融合的**唯一键**（同类型同规范名 = 同一实体）。
    """
    entity_id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"entity_id": self.entity_id, "name": self.name, "type": self.type,
                "aliases": list(self.aliases), "attrs": dict(self.attrs)}


@dataclass
class Relation:
    """图谱关系边。

    evidence / source_doc / page 构成 **provenance 三元组** ——
    这是工单「返回答案所在的文本块」的实现基础：
    LLM 生成的每条结论都能回溯到具体文本块与页码。

    confidence < 1.0 的边视为 LLM 抽取（必须带 evidence）；
    规则/词典抽取的边 confidence = 1.0。
    """
    head_id: str
    rel_type: str
    tail_id: str
    evidence: str = ""
    source_doc: str = ""
    page: int = -1
    confidence: float = 1.0
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"head_id": self.head_id, "rel_type": self.rel_type,
                "tail_id": self.tail_id, "evidence": self.evidence,
                "source_doc": self.source_doc, "page": self.page,
                "confidence": self.confidence, "attrs": dict(self.attrs)}


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def make_entity_id(etype: str, canonical: str) -> str:
    """生成稳定实体 ID：`{类型}:{规范名}`（规范名去首尾空白）。"""
    return f"{etype}:{canonical.strip()}"


def validate_entity(e: Entity) -> list[str]:
    """返回错误列表（空列表 = 合法）。"""
    errors: list[str] = []
    if e.type not in ENTITY_TYPES:
        errors.append(f"未知实体类型: {e.type!r}")
    if not e.name or not e.name.strip():
        errors.append("实体名称为空")
    if not e.entity_id:
        errors.append("entity_id 为空")
    return errors


def validate_relation(r: Relation) -> list[str]:
    """返回错误列表（空列表 = 合法）。"""
    errors: list[str] = []
    spec = RELATION_TYPES.get(r.rel_type)
    if spec is None:
        errors.append(f"未知关系类型: {r.rel_type!r}")
        return errors

    if not r.head_id or not r.tail_id:
        errors.append("关系头/尾实体 ID 为空")

    # 端点类型校验：从 `{类型}:{名称}` 解析类型前缀
    for label, eid in (("head", r.head_id), ("tail", r.tail_id)):
        etype = eid.split(":", 1)[0] if ":" in eid else ""
        allowed = spec[label]
        if allowed and etype and etype not in allowed:
            errors.append(
                f"{r.rel_type} 的 {label} 类型 {etype!r} 不合法，允许 {allowed}")

    # LLM 抽取的边必须有证据片段（防幻觉）
    if r.confidence < 1.0 and not (r.evidence or "").strip():
        errors.append(f"{r.rel_type} 边为 LLM 抽取（confidence<1.0），"
                      f"必须提供 evidence 原文片段")
    return errors
