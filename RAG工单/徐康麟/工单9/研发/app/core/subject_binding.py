# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单：人工智能NLP-RAG-Query 理解优化

主体绑定（subject binding）—— 把「字段值」绑定到**正确的公司主体**上。

## 为什么需要这个模块
招股说明书的**每一页页眉**都印着发行人全称（如「武汉力源信息技术股份有限公司 招股意向书」），
而同一页/同一块里往往同时出现**多个法律主体**的同类字段。最典型的污染是
「本次发行的有关当事人」与「保荐机构（主承销商）声明」两处签名块：

    (一)发行人：武汉力源信息技术股份有限公司  1、法定代表人：赵马克
    (二)保荐机构(主承销商)：国信证券股份有限公司 1、法定代表人：何如

问「武汉力源信息技术股份有限公司的法定代表人是谁？」时，若不看**局部锚点**，
抽取值可能是「何如」（国信证券的法定代表人）—— 实测工单5 基线即为此错值。

## 判定依据（全部通用，**不硬编码任何公司名**）
1. **邻近机构名**：字段附近窗口内若出现「另一个」公司全称 → 该值属于别人，拒收。
2. **章节角色锚点**：字段之前最近的**章节/小节标题**（形如 ``一、XXX`` / ``(二)XXX：``）
   若命中**非发行人角色**（保荐机构/主承销商/会计师事务所/律师事务所/声明…）→ 拒收；
   命中**发行人角色**（发行人/本公司/公司基本情况…）→ 加分。
3. **主体缺席**：整块里根本没出现问句点名的发行人 → 轻微降权。

本模块**零外部依赖**，可单独单测（见 ``研发/tests/test_subject_binding.py``）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"

#: 公司/机构全称（允许「武汉」「股份」等修饰）
ORG_RE = re.compile(r"[\u4e00-\u9fff]{2,12}(?:股份)?有限公司")

#: 章节标题锚点：``一、发行人的基本情况`` / ``(二)保荐机构(主承销商)：`` / ``（一）发行人：``
LABEL_RE = re.compile(
    r"(?:[（(][一二三四五六七八九十百\d]+[）)]|[一二三四五六七八九十百\d]+[、.])"
    r"\s*([^：:\n，,。；;（(]{2,24})"
)

#: 「这是关于发行人本身」的角色词
ISSUER_ROLE = re.compile(
    r"发行人|本公司|公司概况|公司基本情况|公司的基本情况|发行人基本情况|"
    r"发行人的基本情况|发行人简介|公司简介|本次发行概况"
)

#: 「这是关于中介机构/他人」的角色词 —— 出现即说明该段字段不属于发行人
NON_ISSUER_ROLE = re.compile(
    r"保荐机构|主承销商|承销商|副主承销商|会计师事务所|律师事务所|评估机构|"
    r"验资机构|中介机构|证券服务机构|资信评级|声明|承诺函|承诺|"
    r"董事|监事|高级管理人员|审计机构|发行人律师|发行人会计师"
)

#: 人名型字段（值是人名而非数字/单位）。与 ``query_understanding.FIELD_TERMS`` 同风格：
#: 都是通用招股书字段词，**不含任何语料专有事实**。
PERSON_FIELDS: tuple[str, ...] = (
    "法定代表人", "实际控制人", "董事长", "总经理", "联系人", "保荐代表人",
    "项目协办人", "负责人", "董事会秘书", "独立董事", "签字会计师", "签字律师",
)

#: 人名形态：2~4 个汉字（可带「先生/女士/同志」称谓），且其后**不再紧跟汉字**
#: （用负向先行断言排除被截断的公司名）。例：
#:     「法定代表人：赵马克 5、公司成立日期」→ 捕获「赵马克」
#:     「法定代表人：程家明先生」          → 捕获「程家明」（回溯掉称谓后断言通过）
#:     「法定代表人：武汉力源信息技术股份有限公司」→ **不匹配**（后随汉字，逐级回溯均失败）
PERSON_VALUE_RE = re.compile(r"^[\s:：]*([\u4e00-\u9fff]{2,4})(?:先生|女士|同志)?(?![一-龥])")


#: 页眉/书名（每页都印发行人全称，会把「块里提到过发行人」这个判据骗过去）
HEADER_RE = re.compile(r"^[\u4e00-\u9fff]{2,20}(?:股份)?有限公司\s*招股(?:意向书|说明书)\s*")

#: 公司名「核心」的**结尾虚词**：真实公司名不会以这些字结尾，而句片会
#: （实测踩坑：「6、整体变更为股份有限公司日期…」被 ORG_RE 当成机构「整体变更为股份有限公司」，
#:  进而把正确的「法定代表人：赵马克」误判成串主体）
_BAD_CORE_TAIL = frozenset("为是在于与及等的了和或之其本该此")
#: 句片特征词（出现在核心中即判为非机构名）
_BAD_CORE_SUB = ("变更为", "更名为", "系", "属于")


@dataclass(slots=True)
class BindingVerdict:
    """一次主体绑定的判定结果。"""

    ok: bool
    score_delta: float
    reason: str
    anchor: str = ""
    foreign_org: str = ""
    issuer_seen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "score_delta": round(self.score_delta, 4), "reason": self.reason,
            "anchor": self.anchor, "foreign_org": self.foreign_org, "issuer_seen": self.issuer_seen,
        }


def _looks_like_org(name: str) -> bool:
    """``ORG_RE`` 命中的串**是否真的是机构名**（而非句子片段）。"""
    core = re.sub(r"(?:股份)?有限公司$", "", name or "")
    if len(core) < 2:
        return False
    if core[-1] in _BAD_CORE_TAIL:
        return False
    if any(bad in core for bad in _BAD_CORE_SUB):
        return False
    return True


def issuer_in_body(text: str, issuer: str) -> bool:
    """发行人是否出现在**正文**里（先剔除页眉/书名）。

    页眉每页都印「<发行人全称> 招股意向书」，若只判断「这一块里有没有发行人」，
    任何页都会被判为「提到了发行人」——实测正是这一点让 pdf2 第 52 页（上海博润的
    财务数据）里的「法定代表人：胡桃荣」冒充成了力源的法定代表人。
    """
    if not issuer:
        return False
    body = HEADER_RE.sub("", text or "", count=1)
    return any(f and f in body for f in _short_forms(issuer))


def _short_forms(issuer: str) -> list[str]:
    """发行人的可接受简称（用于判定「这里说的就是它」）。"""
    if not issuer:
        return []
    forms = {issuer}
    core = re.sub(r"(武汉|北京|上海|深圳|股份有限公司|有限公司|电子|信息技术|科技)", "", issuer)
    if len(core) >= 2:
        forms.add(core)
    # 「武汉力源信息技术股份有限公司」→「力源信息」（正文里常见的简称）
    m = re.match(r"^[\u4e00-\u9fff]{2}(\w{2})", issuer)
    if m and len(issuer) >= 6:
        forms.add(issuer[2:6])
    return [f for f in forms if f]


def nearest_label(text: str, pos: int, *, scan: int = 320) -> str:
    """取 ``pos`` 之前最近的章节/小节标题文字（找不到返回空串）。"""
    if not text or pos <= 0:
        return ""
    left = text[max(0, pos - int(scan)):pos]
    best = ""
    for m in LABEL_RE.finditer(left):
        best = m.group(1)
    return best.strip()


def window_orgs(text: str, pos: int, *, before: int = 45, after: int = 50) -> list[str]:
    """取字段位置附近窗口内出现的机构全称（已去重、保序、**剔除截断残片**）。

    窗口是从整块文本里**切片**出来的，左边界可能正好落在某个公司名中间，切出
    「股份有限公司」「限公司」这类没有主体的后缀残片 —— 它们不是「别的机构」，
    必须剔除，否则会把正确的候选误判为串主体（实测踩坑：p23 的「法定代表人：赵马克」
    被左侧切片残片「股份有限公司」误杀）。
    """
    if not text:
        return []
    seg = text[max(0, pos - int(before)): pos + int(after)]
    seen: list[str] = []
    for m in ORG_RE.finditer(seg):
        name = m.group(0)
        if not _looks_like_org(name):
            continue                      # 后缀残片 / 句子片段（如「整体变更为股份有限公司」）
        if name not in seen:
            seen.append(name)
    return seen


def bind_subject(
    text: str,
    pos: int,
    *,
    issuer: str | None,
    section: str = "",
    window_before: int = 45,
    window_after: int = 50,
) -> BindingVerdict:
    """判定 ``text[pos:]`` 处的字段值是否真的属于 ``issuer``。

    判定优先级（先拒收后加分）：
        1. 窗口内出现**别的机构全称** → ``foreign_org``，拒收（-0.45）；
        2. 最近标题命中**非发行人角色** → ``non_issuer_section``，拒收（-0.45）；
        3. 最近标题命中**发行人角色** → 通过并加分（+0.18）；
        4. 其余 → 中性通过（0.0）；
        另：``issuer`` 非空但全块都没出现它 → 额外 -0.15（``issuer_absent``）。
    """
    issuer = (issuer or "").strip()
    flat = text or ""
    forms = _short_forms(issuer)
    issuer_seen = bool(forms) and any(f and f in flat for f in forms)

    orgs = window_orgs(flat, pos, before=window_before, after=window_after)
    foreign = ""
    for name in orgs:
        if forms and any(f and f in name for f in forms):
            continue          # 就是发行人自己（或其简称）
        if issuer and name == issuer:
            continue
        foreign = name
        break

    if foreign:
        return BindingVerdict(False, -0.45, "foreign_org", anchor="", foreign_org=foreign,
                              issuer_seen=issuer_seen)

    anchor = nearest_label(flat, pos) or (section or "").strip()
    if anchor and NON_ISSUER_ROLE.search(anchor):
        return BindingVerdict(False, -0.45, "non_issuer_section", anchor=anchor,
                              issuer_seen=issuer_seen)

    if anchor and ISSUER_ROLE.search(anchor):
        return BindingVerdict(True, 0.18, "issuer_section", anchor=anchor, issuer_seen=issuer_seen)

    delta = 0.0
    reason = "neutral"
    if issuer and not issuer_seen:
        delta = -0.15
        reason = "issuer_absent"
    return BindingVerdict(True, delta, reason, anchor=anchor, issuer_seen=issuer_seen)


def is_person_field(field: str, field_terms: Sequence[str] | None = None) -> bool:
    """问句要的是不是「人名型」字段。"""
    text = str(field or "")
    if any(pf in text for pf in PERSON_FIELDS):
        return True
    return any(pf in str(t) for t in (field_terms or []) for pf in PERSON_FIELDS)


def extract_person_value(after_text: str) -> str:
    """从字段之后的文本里取人名值（取不到返回空串）。"""
    m = PERSON_VALUE_RE.match(after_text or "")
    if not m:
        return ""
    value = m.group(1)
    # 排除明显不是人名的通用词
    if value in {"公司", "本公司", "上述", "如下", "情况", "人民币", "万元", "股份"}:
        return ""
    if ORG_RE.match(value):
        return ""
    return value
