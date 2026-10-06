# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
"""
文档配置（DocProfile）：一份 PDF 一条。

【为什么要外置】加第二份 PDF 时才暴露出，有**五处**是与具体文档绑死的：
页眉模板、页脚模板、页码格式、查询要抽象掉的实体专名、证据页的抽取正则。
它们原先散在 pdf_parser / config / evaluator 里，全部写死成招股说明书1.pdf。
工单04/05 还要继续加文档，所以收拢到这里，一份文档一行配置。

【两份文档的差异都是实测出来的，不是猜的】
                    招股说明书1.pdf            招股说明书2.pdf
  页眉        武汉兴图新科电子股份有限公司…招股意向书   武汉力源信息技术股份有限公司…招股意向书
  页脚        1-1-21（分节页码）              21（**裸数字**）
  篇幅        548 页                        350 页
  旋转页      无                            8 页（旋转 90° 的横向表，页眉页脚不在常规分带内）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 页眉/页脚所在的纵向比例带（实测页高 842：页眉 y≈44 → 0.05，页脚 y≈780 → 0.93）
HEADER_BAND = 0.12
FOOTER_BAND = 0.85


@dataclass(frozen=True)
class DocProfile:
    """一份 PDF 的解析与检索配置。"""

    key: str                     # 稳定标识
    doc_name: str                # 文件名，与 Milvus 的 doc_name 字段一致
    header_re: re.Pattern        # 页眉：整行模板
    footer_re: re.Pattern        # 页脚：整行模板
    label_fmt: str               # 页码格式，如 "1-1-{n}" 或 "{n}"
    entity_names: list[str] = field(default_factory=list)
    evidence_re: re.Pattern | None = None   # 证据页字符串的抽取正则
    # ---------- 提示词用 ----------
    # 【为什么这两项也必须进配置】原本写死在 generator.SYSTEM_PROMPT 里
    # （"基于《武汉兴图新科电子股份有限公司招股意向书》…"和「（见 1-1-XX 页）」）。
    # 加了第二份文档后，这个写死的 system prompt 会造成**系统性拒答**：
    # 模型读到"这是兴图新科的文件"，于是对力源的问题判定"问题与文档不符"，
    # 直接输出"资料中未提及相关信息" —— 而答案明明就在片段里（实测 id=1）。
    # 同时它还会按 1-1-XX 编造书2 的页码（书2 的页脚是裸数字）。
    title: str = ""              # 文档全称，写进 system prompt
    citation_hint: str = ""      # 页码标注格式示例，如 "1-1-XX" / "XX"
    notes: str = ""              # 该文档特有的作答提醒（拼在规则末尾）

    # ------------------------------------------------------------------
    def label(self, page_no: int, offset: int = 0) -> str:
        """PDF position index → 给人看的页码。

        【为什么格式要按文档走，不统一成 1-1-N】
        招股书2 的页脚印的就是裸数字 `21`。若统一成 `1-1-21`：
          (a) 引用与读者在 PDF 里看到的页码对不上，验收时无法核对；
          (b) 两份文档都产生 `1-1-21`，页级精确率/召回率会**跨文档歧义**。
        按文档走格式，引用可核对，页级指标也不再含糊。
        """
        return self.label_fmt.format(n=page_no - offset)

    def matches_header(self, line: str) -> bool:
        return bool(self.header_re.match(line.strip()))

    def matches_footer(self, line: str) -> bool:
        return bool(self.footer_re.match(line.strip()))


# ----------------------------------------------------------------------
DOC_PROFILES: dict[str, DocProfile] = {
    "xingtu": DocProfile(
        key="xingtu",
        doc_name="招股说明书1.pdf",
        header_re=re.compile(r"^\s*武汉兴图新科电子股份有限公司\s+招股意向书\s*$"),
        footer_re=re.compile(r"^\s*1-1-\d+\s*$"),
        label_fmt="1-1-{n}",
        entity_names=[
            "武汉兴图新科电子股份有限公司",
            "武汉兴图新科电子",
            "兴图新科",
            # 英文全称（工单01 功能验收 4 要求中英双语，英文提问也得能路由）
            "Wuhan Xingtu Xinke Electronics Co.,Ltd.",
            "Wuhan Xingtu Xinke Electronics",
        ],
        # 证据页字符串形如 "1-1-128" 或 "1-1-21（发行人基本情况表）；1-1-51 亦载明"
        evidence_re=re.compile(r"1-1-\d+"),
        title="武汉兴图新科电子股份有限公司招股意向书",
        citation_hint="1-1-XX",
        notes=(
            "7. 【脱敏名称原样】原文用「某」字脱敏的名称（如《某视频技术规范1.0》、\n"
            "   「某情报、指挥、控制与通信网络一体化工程」）必须**原样照抄**，\n"
            "   不得补全、改写或意译成别的名称。"
        ),
    ),
    "liyuan": DocProfile(
        key="liyuan",
        doc_name="招股说明书2.pdf",
        header_re=re.compile(r"^\s*武汉力源信息技术股份有限公司\s+招股意向书\s*$"),
        # 页脚是**裸数字**（实测第 21 页页脚 = "21"，index 21 → 印刷页 21，偏移 0）
        footer_re=re.compile(r"^\s*\d{1,3}\s*$"),
        label_fmt="{n}",
        entity_names=[
            "武汉力源信息技术股份有限公司",
            "武汉力源信息技术",
            "武汉力源",
            "力源信息",
            # 英文全称（书2 封面印的就是这个写法）
            "Wuhan P&S Information Technology Co.,Ltd.",
            "Wuhan P&S Information Technology",
        ],
        # 【注意】这条正则比较宽，所以本文件的 evidence_page 字段必须**只写页码**
        # （如 "21" 或 "21 23"），不要混入年份之类的其它数字，否则会被误抽。
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="武汉力源信息技术股份有限公司招股意向书",
        citation_hint="XX",
        # 书2 大量用「[ ◆ ]」占位（如发行价格、募集资金净额）。不提醒的话，
        # 模型会把空占位当成"有内容"或者自己补一个数出来 —— 而书2 的题目
        # （发行股数、募投项目、关联方）恰好都紧邻这些占位。
        notes=(
            "7. 【占位符原样】原文中印作「[ ◆ ]」的位置是**空缺占位**，"
            "表示该处未披露；照抄「[ ◆ ]」即可，不要臆测或补填其内容。"
        ),
    ),
}

# 未指定文档时的默认配置（保持工单01/02 的既有行为）
DEFAULT_DOC = DOC_PROFILES["xingtu"]


def get_doc_profile(name_or_key: str | None) -> DocProfile | None:
    """按 key 或文件名取配置。取不到返回 None。"""
    if not name_or_key:
        return None
    s = name_or_key.strip()
    if s in DOC_PROFILES:
        return DOC_PROFILES[s]
    for p in DOC_PROFILES.values():
        if p.doc_name == s:
            return p
    return None


def profile_for_path(path: str | Path) -> DocProfile:
    """按 PDF 文件名取配置；取不到**直接报错**，不静默套用别的文档的规则。

    【为什么报错而不是回退】页眉/页脚/页码格式都是文档特有的 ——
    静默套用招股书1 的规则，结果就是"页眉一条都清不掉、页码格式显示错"
    （工单03 加招股书2 时实测踩过：页眉页脚各清 0 条）。
    这种失败不报错、只是结果悄悄变差，所以宁可当场失败。
    """
    name = Path(path).name
    prof = get_doc_profile(name)
    if prof is None:
        known = "、".join(p.doc_name for p in DOC_PROFILES.values())
        raise ValueError(
            f"没有为《{name}》配置解析规则。请在 app/core/doc_profiles.py 的 "
            f"DOC_PROFILES 里加一条（页眉模板 / 页脚模板 / 页码格式 / 实体专名）。"
            f"已配置的文档：{known}"
        )
    return prof


def all_entity_names() -> list[str]:
    """所有文档的实体专名（查询抽象用）。**按长度降序**，保证最长匹配优先。"""
    names: list[str] = []
    for p in DOC_PROFILES.values():
        names.extend(p.entity_names)
    return sorted(set(names), key=len, reverse=True)


def doc_ids_by_entity() -> dict[str, str]:
    """实体专名 → 文档 key。供路由用（最长匹配优先）。"""
    out: dict[str, str] = {}
    for p in DOC_PROFILES.values():
        for n in p.entity_names:
            out[n] = p.key
    return out


def canonical_entity(doc_key: str, *, english: bool = False) -> str:
    """该文档的**规范实体名**（工单05 指代消解用）。取 entity_names[0]（全称）。

    【为什么必须是全称，不能是 route().matched】route() 返回的是**命中的那个专名**，
    可能是短名（「武汉力源」/「兴图新科」）。改写时若代入短名：
      · 路由结果一样（同一个 doc_key），所以**看起来没问题**；
      · 但 `query_views()` 的「原句视角」变了 → 稠密向量变了 → 排序可能漂移。
    这是典型的「数字悄悄变了却查不出原因」的静默回归。所以统一取全称。

    【为什么要 english 分支】工单01 功能验收 4 要求中英双语。英文轮次的指代消解若代入
    中文全称，会产出「Who is 武汉兴图新科电子股份有限公司？」这种中英混句。
    entity_names 里已配了英文全称（如 "Wuhan P&S Information Technology Co.,Ltd."）。

    【注意】本函数是**纯新增**，不改 entity_names、不改 abstract_query 的行为 ——
    后者是已交付 16 题查询视角的事实来源，动它会直接威胁 16/16 的回归闸门。
    """
    prof = DOC_PROFILES[doc_key]
    if english:
        ascii_names = [n for n in prof.entity_names if n.isascii()]
        if ascii_names:
            return max(ascii_names, key=len)
    return prof.entity_names[0]
