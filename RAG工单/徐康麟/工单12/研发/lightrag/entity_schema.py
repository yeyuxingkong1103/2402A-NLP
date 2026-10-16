# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""招股说明书实体/关系 Schema 定义与规范化（**核心验收项**）。

设计依据（全部来自两份招股说明书的实际内容，逐条见
``设计/实体关系Schema设计.md`` 的「语料依据」列）：

  实体类型 17 类（与 ENTITY_TYPES 的 len() 实测一致）
    公司 / 子公司 / 参股公司 / 股东 / 自然人 / 机构 / 产品 / 行业 / 资质 / 募投项目
    财务指标 / 金额 / 比例 / 日期 / 专利与技术标准 / 客户 / 供应商
  关系类型 16 类（与 RELATION_TYPES 的 len() 实测一致）
    控股 / 参股 / 任职 / 关联方 / 母子公司 / 投资 / 募投 / 采购 / 销售
    竞争 / 所属行业 / 持有股份 / 披露指标 / 制定标准 / 参与工程 / 获奖

本模块只做「类型定义 + 字符串规范化 + 别名规则」，不调用 LLM。
LLM 抽取见 :mod:`lightrag.graph_extract`。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# 实体类型
# ---------------------------------------------------------------------------
ENTITY_TYPES: dict[str, str] = {
    "公司": "发行主体及其全称/简称（如「武汉力源信息技术股份有限公司」「本公司」「发行人」）",
    "子公司": "发行人控股或全资的下属公司（如「武汉力源(香港)信息技术有限公司」）",
    "参股公司": "发行人参股但不控股的公司",
    "股东": "持有发行人股份的法人股东（如「融冰投资」「联众聚源」）",
    "自然人": "董监高、实际控制人、发起人、核心技术人员等自然人",
    "机构": "中介机构、主管部门、科研院所、银行、交易所等（如「中泰证券股份有限公司」）",
    "产品": "发行人提供的产品或产品线（如「视频指挥控制系统」「IC 分销」）",
    "行业": "所处行业或下游应用行业（如「电子信息行业」「工业控制」）",
    "资质": "经营资质、许可、认证（如「武器装备科研生产许可」）",
    "募投项目": "本次募集资金投资项目（如「补充流动资金」「仓储及物流中心」）",
    "财务指标": "会计科目或指标名（如「营业收入」「军用领域收入」「注册资本」）",
    "金额": "带单位的金额数值（如「15,000.00万元」）",
    "比例": "百分比/比重（如「42.35%」）",
    "日期": "日期或期间（如「2014年12月」「报告期内」）",
    "专利与技术标准": "专利、软件著作权、技术标准（如「《某视频技术规范1.0》」）",
    "客户": "客户或客户群体（如「军方」「总体单位A」）",
    "供应商": "供应商或上游企业类型（如「电子元器件制造企业」）",
}

# ---------------------------------------------------------------------------
# 关系类型
# ---------------------------------------------------------------------------
RELATION_TYPES: dict[str, str] = {
    "控股": "一方控股另一方（持股比例>50% 或明确表述为控股股东）",
    "参股": "一方参股另一方但未控股",
    "任职": "自然人在某公司/机构担任职务（董事长、总经理、董事…）",
    "关联方": "被明确列为关联方（含不存在控制关系的关联方）",
    "母子公司": "母公司与子公司关系",
    "投资": "对外投资行为（含设立、增资、股权转让）",
    "募投": "募集资金投向某项目",
    "采购": "向供应商/关联方采购",
    "销售": "向客户销售产品或服务",
    "竞争": "同业竞争或市场竞争关系",
    "所属行业": "某主体/产品所属行业",
    "持有股份": "持有某公司股份及其比例",
    "披露指标": "某主体的某财务指标取某数值",
    "制定标准": "参与制定技术标准",
    "参与工程": "参与某项工程/项目",
    "获奖": "获得奖项",
}

# 关系类型与其允许的（头实体类型集合, 尾实体类型集合）约束，用于抽取结果校验
RELATION_CONSTRAINTS: dict[str, tuple[set[str], set[str]]] = {
    "控股": ({"公司", "自然人", "股东", "子公司"}, {"公司", "子公司", "参股公司"}),
    "参股": ({"公司", "自然人", "股东"}, {"公司", "参股公司"}),
    "任职": ({"自然人"}, {"公司", "子公司", "机构", "参股公司"}),
    "关联方": ({"公司", "子公司", "参股公司"}, {"公司", "子公司", "参股公司", "股东", "自然人"}),
    "母子公司": ({"公司", "子公司"}, {"子公司", "公司"}),
    "投资": ({"公司", "自然人", "机构"}, {"公司", "子公司", "参股公司", "募投项目"}),
    "募投": ({"公司"}, {"募投项目"}),
    "采购": ({"公司", "子公司"}, {"供应商", "公司", "关联方"}),
    "销售": ({"公司", "子公司"}, {"客户", "公司", "行业"}),
    "竞争": ({"公司", "子公司"}, {"公司", "行业"}),
    "所属行业": ({"公司", "产品", "子公司"}, {"行业"}),
    "持有股份": ({"自然人", "股东", "公司", "机构"}, {"公司", "子公司", "参股公司"}),
    "披露指标": ({"公司", "子公司", "自然人", "行业"},
                 {"财务指标", "金额", "比例", "日期"}),
    "制定标准": ({"公司", "机构"}, {"专利与技术标准"}),
    "参与工程": ({"公司"}, {"机构", "募投项目", "产品"}),
    "获奖": ({"公司", "机构", "产品"}, {"专利与技术标准"}),
}

# ---------------------------------------------------------------------------
# 别名归并规则
# ---------------------------------------------------------------------------
# 指向同一主体的「指代性别名」：命中即归并到 canonical
PRONOUN_ALIASES: dict[str, str] = {
    "本公司": "发行人",
    "公司": "发行人",
    "发行人": "发行人",
    "股份公司": "发行人",
    "兴图新科": "武汉兴图新科电子股份有限公司",
    "兴图新科电子股份有限公司": "武汉兴图新科电子股份有限公司",
    "武汉兴图新科电子股份有限公司": "武汉兴图新科电子股份有限公司",
    "力源信息": "武汉力源信息技术股份有限公司",
    "力源有限": "武汉力源信息技术股份有限公司",
    "武汉力源信息技术有限公司": "武汉力源信息技术股份有限公司",
    "武汉力源信息技术股份有限公司": "武汉力源信息技术股份有限公司",
    "力源贸易": "武汉力源贸易有限公司",
    "香港力源": "武汉力源(香港)信息技术有限公司",
    "武汉力源(香港)信息技术有限公司": "武汉力源(香港)信息技术有限公司",
    "普芯达": "上海普芯达电子科技有限公司",
    "赵马克": "赵马克(Mark Zhao)",
    "Mark Zhao": "赵马克(Mark Zhao)",
    "赵马克(Mark Zhao)": "赵马克(Mark Zhao)",
}

# 需要剥离的尾缀/前后装饰，便于「同实体不同写法」对齐
_STRIP_PREFIX = ("本公司", "贵公司", "该公司")
_STRIP_SUFFIX = ("等", "等等")
# 括号统一：全角括号 → 半角，便于「武汉力源（香港）」与「武汉力源(香港)」归并
_FULLWIDTH = str.maketrans("（）【】〔〕", "()[][]")


def normalize_name(name: str) -> str:
    """实体名规范化：去空白、统一括号、去尾部标点。

    这一步只做「同一字符串的不同写法」对齐，不做语义归并；
    语义归并见 :func:`resolve_alias`。
    """
    if not name:
        return ""
    text = str(name).strip()
    text = text.translate(_FULLWIDTH)
    text = re.sub(r"\s+", "", text)
    text = text.strip("，,。.、；;：:　\"'“”‘’《》")
    for suf in _STRIP_SUFFIX:
        if len(text) > len(suf) + 1 and text.endswith(suf):
            text = text[: -len(suf)]
            break
    return text


def resolve_alias(name: str) -> str:
    """把指代性别名映射到规范名；未命中则返回规范化后的原名。"""
    norm = normalize_name(name)
    if not norm:
        return ""
    if norm in PRONOUN_ALIASES:
        return PRONOUN_ALIASES[norm]
    for pref in _STRIP_PREFIX:
        if norm.startswith(pref) and len(norm) > len(pref) + 1:
            stripped = norm[len(pref):]
            if stripped in PRONOUN_ALIASES:
                return PRONOUN_ALIASES[stripped]
    # 「XX公司」去掉「公司」后仍能命中别名表
    if norm.endswith("公司") and norm[:-2] in PRONOUN_ALIASES:
        return PRONOUN_ALIASES[norm[:-2]]
    return norm


def char_similarity(a: str, b: str) -> float:
    """字符级 Dice 相似度（不依赖第三方库），用于别名二次归并。"""
    a, b = normalize_name(a), normalize_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    bigrams_a = {a[i:i + 2] for i in range(len(a) - 1)}
    bigrams_b = {b[i:i + 2] for i in range(len(b) - 1)}
    if not bigrams_a or not bigrams_b:
        return 0.0
    inter = len(bigrams_a & bigrams_b)
    return 2.0 * inter / (len(bigrams_a) + len(bigrams_b))


def guess_entity_type(name: str) -> str:
    """无 LLM 时的实体类型启发式（用于校验/兜底，不替代 LLM 判定）。"""
    n = normalize_name(name)
    if not n:
        return "公司"
    if re.fullmatch(r"\d{4}\s*年(\s*\d{1,2}\s*月)?", n) or n in ("报告期内", "报告期", "报告期末"):
        return "日期"
    if re.fullmatch(r"-?\d[\d,]*(\.\d+)?\s*%", n):
        return "比例"
    if re.fullmatch(r"-?\d[\d,]*(\.\d+)?\s*(万元|亿元|元|万股|股|万美元)", n):
        return "金额"
    if n.endswith(("股份有限公司", "有限责任公司", "有限公司")):
        return "公司"
    if any(k in n for k in ("标准", "专利", "软件著作权", "技术规范", "规范")):
        return "专利与技术标准"
    if any(k in n for k in ("项目", "工程")) and "工程" not in n[:2]:
        return "募投项目"
    if n.endswith(("率", "收入", "成本", "利润", "资产", "负债", "资本", "股本", "毛利")):
        return "财务指标"
    if any(k in n for k in ("行业", "领域", "市场", "产业")):
        return "行业"
    return "公司"


def relation_allowed(rel_type: str, head_type: str, tail_type: str) -> bool:
    """按 :data:`RELATION_CONSTRAINTS` 校验关系类型是否与两端实体类型相容。

    未登记的 (head_type, tail_type) 组合一律放行（宽松优先，避免误杀真关系）。
    """
    rule = RELATION_CONSTRAINTS.get(rel_type)
    if rule is None:
        return True
    heads, tails = rule
    return (head_type in heads) and (tail_type in tails)


def schema_prompt_block() -> str:
    """生成给 LLM 的 Schema 说明块（抽取提示词用）。"""
    ent = "\n".join(f"  - {k}：{v}" for k, v in ENTITY_TYPES.items())
    rel = "\n".join(f"  - {k}：{v}" for k, v in RELATION_TYPES.items())
    return f"【实体类型（只能从中选）】\n{ent}\n\n【关系类型（只能从中选）】\n{rel}"


def coerce_type(raw: str, *, kind: str = "entity") -> str:
    """把 LLM 返回的任意类型字符串吸附到最接近的合法类型。"""
    table = ENTITY_TYPES if kind == "entity" else RELATION_TYPES
    text = normalize_name(raw)
    if text in table:
        return text
    for key in table:
        if key and (key in text or text in key):
            return key
    # 常见同义映射
    syn = {
        "组织": "机构", "单位": "机构", "科研机构": "机构", "中介机构": "机构",
        "公司名称": "公司", "企业": "公司", "上市公司": "公司", "控股股东": "股东",
        "人物": "自然人", "人名": "自然人", "高管": "自然人", "实际控制人": "自然人",
        "数值": "金额", "货币": "金额", "钱": "金额", "百分比": "比例",
        "时间": "日期", "期间": "日期", "年份": "日期",
        "技术标准": "专利与技术标准", "标准": "专利与技术标准", "专利": "专利与技术标准",
        "指标": "财务指标", "科目": "财务指标",
        "上游": "供应商", "下游": "客户",
        "持股": "持有股份", "控股关系": "控股", "股权关系": "持有股份",
        "供职": "任职", "担任": "任职", "职务": "任职",
        "供给": "采购", "供货": "采购", "购买": "采购",
        "出售": "销售", "供货给": "销售", "营收": "披露指标",
        "关联关系": "关联方", "关联企业": "关联方",
        "投资关系": "投资", "出资": "投资",
        "募集资金投向": "募投", "投入": "募投",
    }
    return syn.get(text, text if text in table else (list(table)[0] if kind == "entity" else "披露指标"))


def all_entity_types() -> list[str]:
    return list(ENTITY_TYPES)


def all_relation_types() -> list[str]:
    return list(RELATION_TYPES)


@dataclass(frozen=True)
class SchemaStats:
    """Schema 规模（用于文档与验收）。"""

    entity_types: int
    relation_types: int

    def to_dict(self) -> dict[str, int]:
        return {"entity_types": self.entity_types, "relation_types": self.relation_types}


def schema_stats() -> SchemaStats:
    return SchemaStats(len(ENTITY_TYPES), len(RELATION_TYPES))


def iter_examples() -> Iterable[tuple[str, str]]:
    """(类型, 说明) 迭代，供文档与提示词复用。"""
    for key, val in ENTITY_TYPES.items():
        yield f"实体:{key}", val
    for key, val in RELATION_TYPES.items():
        yield f"关系:{key}", val
