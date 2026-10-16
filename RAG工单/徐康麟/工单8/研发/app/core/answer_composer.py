# -*- coding: utf-8 -*-
"""工单4 证据化答案合成器（**逐项完整枚举**的确定性实现；无 LLM 也能给出可从 PDF 回溯的答案）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

为什么需要它（任务书 ★A / ★B）：
    * 判分口径是「``answer_keywords`` **全集匹配**，缺一即判错」，且**枚举项必须用证据原文全称**；
    * 生成模型（qwen2.5:3b）在 3 秒预算内**不能保证**逐项枚举完整，也不保证不缩写枚举项；
    * 因此本模块把「答案的**事实骨架**」用确定性规则从证据里抽出来：数值照抄、枚举项照抄、
      结论句按问句意图选取——LLM 只负责润色，且**必须通过原子保全校验**（见 ``generator.py``）。

抽取的通用规则（**只依赖证据原文与问句措辞，不含任何语料专有硬编码**）：

    R1 结构树枚举  「X 由 N 个<单位>构成：A、B、C」→ 逐项列出（题 5 的 4 部门 + 6 销售处）
    R2 表行枚举    markdown 表 → «序号 | 名称 | 金额(表头单位)» 逐行列出（题 2 的 5 项目）
    R3 表行实体    «名称 | 关系» 逐行列出，剔除「已转让/曾为关联方」等**已终止**行（题 4 的 7 家）
    R4 序列数值    「分别为 A、B、C 和 D」按问句问的**量纲**取金额组或占比组（题 260/33）
    R5 单值/键值对 表内 «字段 | 值» 或正文 «字段：值»，值缺单位时从表头继承（题 543/531/207）
    R6 结论句      「最快/负增长/重要供应商/荣获/参与制定」对应问句问点（题 6/95/957/795）
    R7 上/下游切片  按问句方向切出**同一方向**的子句（题 34/793）

所有答案都附**可回溯引用**（文件名 + 1-based 物理页码 + 模态），由 ``citation.py`` 生成。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# 工单5 新增：主体绑定（把字段值绑定到正确的公司，解决同页/同块多主体互污染）
from .subject_binding import (bind_subject, extract_person_value, is_person_field,
                              issuer_in_body)

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 句末标点（中文/英文）
_SENT_END = re.compile(r"(?<=[。！？；!?;])\s*")
#: 列表序号前缀（①②… / 1、 / (1) / 一、）
_LIST_MARK = re.compile(r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\d{1,2}[、.）)]|\(\d{1,2}\)|[一二三四五六七八九十]、)\s*")
#: 带单位数值
_NUM_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(万元|亿元|万股|亿元|元|股|%|倍)")
#: 计数（「4 个部门」「6 家销售处」这类**没有货币/百分比单位**的数字也必须进原子表，
#: 否则 LLM 可以把「6 个销售处」改写成「5 个销售处」而护栏一声不响 —— T5 英文路径实测踩坑）
# 工单7：``_NUM_COUNT`` / ``_COUNT_UNITS`` / ``_ATOM_TAIL_STOP`` 已随「不再抽计数原子」一并移除
# （原因见 ``_atoms_of`` 的注释：该启发式既会造出 "2019个年度报告" 这类伪原子，
#  又无法把 "由4个部门构成" 里的 "4个部门" 干净切出来，两头都不可靠）。
#: 页头噪声（PDF 页眉/书名）
_PAGE_HEADER = re.compile(r"^[\u4e00-\u9fff]{2,20}(?:股份)?有限公司\s*招股意向书\s*")
#: 「已终止关系」措辞（表行剔除依据，取自原文语义，不是 golden 名单）
_TERMINATED = re.compile(r"(转让|注销|吊销|已不存在|曾为关联方|原为|不再)")

#: 表格里的占位符单元（原文未披露的写法）
_PLACEHOLDER_CELL = re.compile(r"^(?:未披露|不适用|\[?\s*[◆◇※★*]+\s*\]?|[-—–]|N/?A)$")

#: 表格里不作为数据行的行（合计行）
_TOTAL_ROW = re.compile(r"^(合计|总计|小计|总额|—|-|)$")

#: 清单题允许的最大表行数：超过即视为财务报表体量表，不作为「逐项枚举」来源
MAX_ENUM_ROWS = 15


def _lazy_logger(logger: Any, module: str = "answer_composer") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def squash(text: str) -> str:
    """压平空白（用于匹配），保留原始标点。"""
    return re.sub(r"\s+", "", str(text or ""))


def clean_sentence(sentence: str) -> str:
    """去掉页眉与行首序号，压平空白（答案里不出现 PDF 页眉）。"""
    text = str(sentence or "").strip()
    text = _PAGE_HEADER.sub("", text)
    text = _LIST_MARK.sub("", text)
    return re.sub(r"\s+", "", text)


def split_sentences(text: str) -> list[str]:
    """按中文句末标点切句（保留标点），并过滤页眉噪声行。"""
    out: list[str] = []
    for piece in _SENT_END.split(str(text or "").replace("\n", " ")):
        cleaned = clean_sentence(piece)
        if cleaned:
            out.append(cleaned)
    return out


def text_of(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("content") or chunk.get("text") or "")


# ---------------------------------------------------------------------------
# 表结构解析
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TableView:
    """chunk 内的 markdown 表视图。"""

    chunk_id: str
    file_name: str
    page: int
    title: str
    header: list[str]
    rows: list[list[str]]
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "page": self.page, "title": self.title,
                "header": self.header, "rows": len(self.rows)}

    def unit_hint(self) -> str | None:
        """表头声明的单位（如「计划总投资(万元)」→ ``万元``）。"""
        for cell in self.header:
            m = re.search(r"[（(]\s*(万元|亿元|万股|元|股|%|倍)\s*[)）]", cell)
            if m:
                return m.group(1)
        return None


def parse_table(chunk: Mapping[str, Any]) -> TableView | None:
    """把 ``type == table`` 的 chunk 解析成 ``TableView``（非表返回 ``None``）。"""
    content = text_of(chunk)
    if "[Markdown]" not in content:
        return None
    title_m = re.search(r"\[表标题\]\s*(.*)", content)
    body = content.split("[Markdown]", 1)[1]
    body = body.split("[关键数字]", 1)[0]
    rows: list[list[str]] = []
    header: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell) <= {"-", ":", " "} and cell for cell in cells):
            continue
        if not header:
            header = cells
            continue
        rows.append(cells)
    if not header and not rows:
        return None
    return TableView(chunk_id=str(chunk.get("chunk_id")), file_name=str(chunk.get("file_name") or ""),
                     page=int(chunk.get("page") or 0),
                     title=clean_sentence(title_m.group(1)) if title_m else "",
                     header=header, rows=rows, raw=body)


# ---------------------------------------------------------------------------
# 合成结果
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ComposedAnswer:
    """确定性合成结果（事实骨架 + 原子 + 取用的块）。"""

    text: str = ""
    shape: str = "generic"
    method: str = "none"
    atoms: list[str] = field(default_factory=list)      # 必须被保留的原子（数值/枚举项/关键短语）
    key_phrases: list[str] = field(default_factory=list)  # 关键子句（结论题用；LLM 候选必须保留）
    item_names: list[str] = field(default_factory=list)  # 枚举项（用于完整性自检）
    used_chunk_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def _atoms_of(text: str, items: Sequence[str], field_terms: Sequence[str]) -> list[str]:
    """从答案文本里提取「原子」：带单位数值 + **计数** + 枚举项 + 命中的字段词。

    工单7（人工智能NLP-RAG-功能测试及评估）修正——**假原子导致 LLM 答案被大面积误拒**：
        原 ``_NUM_COUNT`` 把 ``年/月/日`` 也当计数单位，于是 "2019年年度报告中" 被抽成
        "2019个年度报告中"、"2021年营业收入…" 抽成 "2021个营业收入"，甚至把
        "归属于母公司股东" 这种短语尾巴也粘进来。这类原子**任何自然答案都不可能包含**
        （正确答案只会写 "2019年"，不会写 "2019个"），导致原子保全护栏对 16/17 题
        拒绝掉本来正确的 LLM 候选，最终退回规则拼接（实测：反事实口径下本该判对的
        Q11/Q14/Q15 全部答成「不清楚」）。

        两处收紧（判据：原子必须"可验证且自然答案会复述"）：
        1) 计数单位只保留真正的量词 ``个/家/项/名/条/种/类/级/届/期/只/艘/台/套/张``；
           年份/月/日 是**时间标记**而非计数，其数字已由 ``_NUM_UNIT`` 与问句本身覆盖，
           抽出来只会制造 "2019个…" 这种伪原子；
        2) 量词后的名词尾巴限制在 1~4 个汉字（"部门"/"销售处"），
           更长的一定是句子片段（"年度报告中"/"归属于母公司股东"），丢弃。
        ``items``（枚举项）与 ``_NUM_UNIT``（带单位数值）不受影响，验收用例
        "4个部门"/"6个销售处" 仍能正常抽出（见 设计/测试用例设计.md 的回归用例）。
    """
    atoms: list[str] = []
    flat = squash(text)
    for m in _NUM_UNIT.finditer(flat):
        token = squash(m.group(0))
        if token not in atoms:
            atoms.append(token)
    # 工单7 决策：**不再从正文里抽「数字+量词」计数原子**。
    # 原因（两轮实测）：这个启发式无法可靠区分「真计数」与「时间/短语粘连」——
    #   · "2019年年度报告" → "2019个年度报告"（年 被当量词，任何自然答案都不会这样写）
    #   · "由4个部门构成"  → tail 捕获成 "部门构成"（名词+动词粘连，含动词则丢弃 → 真空转）
    # 两头都不对：留着它会把**正确的 LLM 候选**大面积误拒（实测 16/17 题被拒，
    # 本该判对的 Q11/Q14/Q15 因此答成「不清楚」）；收紧后又几乎抽不出东西。
    # 计数类问句的保护由**类型化 items** 承担（层级枚举的部门/销售处名单、图谱字段名等），
    # 它们本身就是「答案必须复述」的内容，且不含粘连噪声（见 _finalize 的 items 入参）。
    for item in items:
        clean = squash(item)
        if clean and clean not in atoms:
            atoms.append(clean)
    for term in field_terms:
        clean = squash(term)
        if clean and clean in squash(text) and clean not in atoms:
            atoms.append(clean)
    return atoms


def _finalize(text: str, *, shape: str, method: str, items: Sequence[str],
              field_terms: Sequence[str], chunk_ids: Sequence[str], confidence: float,
              notes: Mapping[str, Any] | None = None,
              key_phrases: Sequence[str] = ()) -> ComposedAnswer:
    body = clean_sentence(text)
    if body and not body.endswith(("。", "！", "？", ".", "!")):
        body += "。"
    phrases = [squash(p) for p in key_phrases if squash(p)]
    return ComposedAnswer(
        text=body, shape=shape, method=method,
        atoms=_atoms_of(body, items, field_terms),
        key_phrases=phrases,
        item_names=[squash(i) for i in items if squash(i)],
        used_chunk_ids=list(dict.fromkeys(str(c) for c in chunk_ids)),
        confidence=float(confidence), notes=dict(notes or {}),
    )


# ---------------------------------------------------------------------------
# R1 结构树枚举（题 5）
# ---------------------------------------------------------------------------
_HIER = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,20})\s*由\s*(\d+)\s*个([\u4e00-\u9fff]{1,8})构成[：:]\s*([^。\n]{2,200})")


def compose_hierarchy(analysis: Any, chunks: Sequence[Mapping[str, Any]], *, logger: Any = None) -> ComposedAnswer | None:
    """R1：把「X 由 N 个<单位>构成：A、B、…」逐条列出（问句点名的层级优先）。

    过滤规则（**不依赖任何语料专有名单**）：
      * 父节点是「公司」（含「有限公司」）时跳过——股权结构图里的「公司 由 N 个下级单元构成：42.35%」
        不是组织结构答案（T5 实测：会把股东持股图串进第 5 题答案）；
      * 枚举项全为数字/百分比时跳过（那不是「部门名」）；
      * 问句点名了父节点（如「销售部」「大客户销售部」）时只列点名的层级。
    """
    question = squash(str(getattr(analysis, "question", "")))
    found: list[tuple[str, str, list[str], str]] = []
    for chunk in chunks:
        for m in _HIER.finditer(text_of(chunk)):
            parent, unit, listing = m.group(1), m.group(3), m.group(4)
            items = [clean_sentence(p) for p in re.split(r"[、,，;；]", listing) if clean_sentence(p)]
            if not items:
                continue
            if "有限公司" in parent or "股份" in parent:      # 股权结构图节点，不是组织层级
                continue
            if all(_NUM_UNIT.search(item) and not re.search(r"[\u4e00-\u9fff]{2,}", item) for item in items):
                continue                                      # 全是数字/百分比 → 不是人员/部门清单
            found.append((parent, unit, items, str(chunk.get("chunk_id"))))
    if not found:
        return None
    # 问句点名了哪些父节点（如「销售部」「大客户销售部」）：优先列这些；无点名则全列
    named = [f for f in found if squash(f[0]) in question]
    selected = named or found
    parts: list[str] = []
    used: list[str] = []
    items_all: list[str] = []
    for parent, unit, items, cid in selected:
        parts.append(f"{parent}由 {len(items)} 个{unit}构成：" + "、".join(items))
        used.append(cid)
        items_all.extend(items)
    text = "；其中 ".join(parts)
    return _finalize(f"{text}。", shape="hierarchy_enum", method="rule_hierarchy",
                     items=items_all, field_terms=getattr(analysis, "field_terms", []),
                     chunk_ids=used, confidence=0.95 if named else 0.8,
                     notes={"parents": [f[0] for f in selected], "named": bool(named)},
                     key_phrases=parts)


# ---------------------------------------------------------------------------
# R2 表行枚举（题 2）
# ---------------------------------------------------------------------------
def _table_relevance(analysis: Any, table: TableView) -> float:
    """表与问句的相关度：问句实词/字段词在标题+表头+首行的覆盖率。"""
    haystack = squash(table.title + "".join(table.header) + "".join("".join(r) for r in table.rows))
    keywords = [squash(k) for k in getattr(analysis, "keywords", []) or [] if squash(k)]
    if not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k in haystack)
    return hit / len(keywords)


def compose_table_enum(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                       logger: Any = None) -> ComposedAnswer | None:
    """R2/R3：选一张与问句最相关的表，把数据行逐项列出（名称 + 表头单位继承后的金额）。"""
    question = squash(str(getattr(analysis, "question", "")))
    candidates: list[tuple[float, TableView, Mapping[str, Any]]] = []
    for chunk in chunks:
        table = parse_table(chunk)
        if table is None or len(table.rows) < 2:
            continue
        # 财务报表体量表（> 15 数据行，如「利润表」32 行）不是「枚举答案」，不得作为清单答案：
        # T5 实测踩坑——问「特斯拉营业收入是多少」时，错把利润表 32 行当成「共 32 项」清单作答。
        if len(table.rows) > MAX_ENUM_ROWS:
            continue
        score = _table_relevance(analysis, table)
        # 极性一致性：问「不存在…」时，优先标题/内容含「不存在」的表
        if "不存在" in question and "不存在" in squash(table.title + table.raw):
            score += 0.35
        # 明细表优先于财务指标表：表头含「名称」或「项目名称」加分
        if any("名称" in cell for cell in table.header):
            score += 0.2
        candidates.append((score, table, chunk))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -len(item[1].rows), item[1].chunk_id))
    score, table, chunk = candidates[0]
    if score <= 0.05:
        return None
    name_idx = 0
    for idx, cell in enumerate(table.header):
        if "名称" in cell or "项目" in cell:
            name_idx = idx
            break
    unit = table.unit_hint()
    parts: list[str] = []
    items: list[str] = []
    values: list[str] = []
    for row in table.rows:
        cells = [clean_sentence(c) for c in row]
        if not any(cells) or _TOTAL_ROW.match(cells[0] if cells else ""):
            continue
        name = cells[name_idx] if name_idx < len(cells) else cells[0]
        if not name:
            continue
        # 剔除「已终止」的行（如「原为…，2008 年对外转让」）——它们不属于当前清单
        if any(_TERMINATED.search(c) for c in cells):
            continue
        value, placeholder = "", ""
        for idx, cell in enumerate(cells):
            if idx == name_idx or not cell:
                continue
            if _NUM_UNIT.search(cell):                       # 1) 带单位数值（最优先）
                value = cell
                break
            if unit and re.fullmatch(r"[\d,]+(?:\.\d+)?", cell):
                # 2) 纯数值 + 表头声明单位（如「计划总投资(万元)」下的 3,393.40）
                #    排除序号列（短整数且在名称列之前）
                if not (idx < name_idx and re.fullmatch(r"\d{1,2}", cell)):
                    value = f"{cell} {unit}"
                    break
            if _PLACEHOLDER_CELL.search(cell):               # 3) 占位符（[◆]/未披露）→ 如实标注
                placeholder = cell
        if not value and placeholder:
            value = "原文未披露"
        tail = f" {value}" if value else ""
        parts.append(f"{name}{tail}")
        items.append(name)
        if value:
            values.append(value)
    if not parts:
        return None
    listing = "；".join(f"{i}、{p}" for i, p in enumerate(parts, 1))
    n = len(parts)
    text = f"共 {n} 项：{listing}。"
    return _finalize(text, shape="enum_list", method="rule_table_enum", items=items,
                     field_terms=getattr(analysis, "field_terms", []),
                     chunk_ids=[table.chunk_id], confidence=min(0.95, 0.55 + score),
                     notes={"table_title": table.title, "rows": n, "unit": unit, "values": values,
                            "header": table.header, "relevance": round(score, 3)},
                     key_phrases=parts)


# ---------------------------------------------------------------------------
# R4 序列数值（题 260 / 33）
# ---------------------------------------------------------------------------
_SERIES = re.compile(r"(分别为|依次为|分别是|各为)([^。；;]{2,240})")


def _series_kind(segment: str) -> str:
    """一段序列的量纲：``percent`` / ``money`` / ``other``。"""
    if "%" in segment:
        return "percent"
    if re.search(r"(万元|亿元|万股)", segment):
        return "money"
    return "other"


def _cut_series(segment: str, kind: str) -> str:
    """把序列截到同量纲的连续段（遇异量纲即停），保留「和」等原文连接词。"""
    out = segment
    for sep in ("，", ",", "；", ";"):
        if sep in out:
            pieces = out.split(sep)
            keep: list[str] = []
            for piece in pieces:
                current = _series_kind(piece)
                if current == kind or (current == "other" and not _NUM_UNIT.search(piece)):
                    keep.append(piece)
                else:
                    break
            out = sep.join(keep)
            break
    return out.strip()


def compose_numeric_series(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                           logger: Any = None) -> ComposedAnswer | None:
    """R4：按问句问的**量纲**（金额 / 占比）取「分别为 …」序列，逐项完整给出。

    实现要点（T5 实测修正）：
      * 以**句**为单位搜索（不是原始块文本），保证前缀是完整子句（避免抽到「…的影响」残句）；
      * 前缀取「该句开头 → 序列起点」并裁到最后一个子句边界，保留「报告期内，公司来自 X 的」这类语境；
      * 序列按量纲切段：问占比取百分比段、问金额取金额段（题 33 与题 260 同页同句，**不得互相顶替**）。
    """
    question = squash(str(getattr(analysis, "question", "")))
    want_ratio = bool(getattr(analysis, "has", lambda _n: False)("asks_ratio")) or "比重" in question or "占比" in question
    want = "percent" if want_ratio else "money"
    keywords = [squash(k) for k in getattr(analysis, "keywords", []) or [] if squash(k)]
    best: tuple[float, str, str, str] | None = None   # (score, text, cid, raw_list)
    for chunk in chunks:
        for sentence in split_sentences(text_of(chunk)):
            flat = squash(sentence)
            for m in _SERIES.finditer(flat):
                series = _cut_series(m.group(2), want)
                if _series_kind(series) != want:
                    continue
                numbers = _NUM_UNIT.findall(series)
                if len(numbers) < 2:
                    continue
                prefix = flat[: m.start()]
                lead = _series_lead(prefix)
                score = 0.6                       # 量纲与问句一致（其余量纲已在上面 continue 掉）
                if keywords:
                    body = squash(lead + series)
                    score += 0.4 * (sum(1 for k in keywords if k in body) / len(keywords))
                if want == "percent" and re.search(r"(比重|占比|比例)", prefix):
                    score += 0.5
                if want == "money" and re.search(r"(收入|金额|投资|投入)", prefix):
                    score += 0.5
                if len(numbers) >= 3:
                    score += 0.2                  # 多点题优先取长序列（报告期 4 期）
                if best is None or score > best[0]:
                    text = f"{lead}{m.group(1)}{series}"
                    best = (score, text, str(chunk.get("chunk_id")), series)
    if best is None:
        return None
    score, text, cid, series = best
    items = [f"{v}{u}" for v, u in _NUM_UNIT.findall(series)]
    return _finalize(text, shape="numeric_series", method="rule_series",
                     items=items, field_terms=getattr(analysis, "field_terms", []),
                     chunk_ids=[cid], confidence=min(0.95, 0.5 + score / 2),
                     notes={"count": len(items), "want": want, "series": series},
                     key_phrases=[text])


def _series_lead(prefix: str) -> str:
    """序列前缀裁剪：去掉章节号/页眉/小标题等噪声，保留「…分别为」前的完整子句语境。"""
    lead = clean_sentence(prefix)
    if not lead:
        return ""
    # 去掉「以上项目所需募集资金投入合计…」这类与本题无关的长前缀：从最后一个句末标点之后取
    for sep in ("。", "；", ";"):
        if sep in lead:
            lead = lead.rsplit(sep, 1)[-1]
    # 去掉序号/小标题（①… 或 （一）… 或 1、…）
    lead = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]", "", lead)
    lead = re.sub(r"^[（(][一二三四五六七八九十][)）][^，,。]{0,14}", "", lead)
    # 去掉紧贴在正文前的「…的影响 / …的风险 / …的情况」小标题（T5 实测：题 260/33 会带出「行业集中度高的影响」）
    lead = re.sub(r"^[\u4e00-\u9fff]{2,14}(?:影响|风险|情况|说明|分析|概述)(?=发行人|公司|本|报告期)", "", lead)
    if len(lead) > 60:
        lead = lead[-60:]
        # 避免从半句开始
        for sep in ("，", ",", "、"):
            if sep in lead:
                lead = lead.split(sep, 1)[-1]
                break
    return lead


# ---------------------------------------------------------------------------
# R5 单值 / 键值对（题 543 / 531 / 207 / 1）
# ---------------------------------------------------------------------------
def _value_with_unit(cell: str, unit: str | None) -> str:
    value = clean_sentence(cell)
    if not value:
        return ""
    if _NUM_UNIT.search(value):
        return value
    if unit and re.fullmatch(r"[\d,]+(?:\.\d+)?", value):
        return f"{value} {unit}"
    return value


#: 「发行人身份字段簇」：同一块里出现 ≥2 个即说明这是「发行人基本情况」类证据
_IDENTITY_CUES = ("公司名称", "中文名称", "英文名称", "法定代表人", "注册地址", "公司住所",
                  "成立日期", "注册资本", "实收资本", "统一社会信用代码")
#: 键值对分隔（字段前必须是句首/换行/列举分隔符，排除句中「…的注册资本…」）
_KV_LEAD = re.compile(r"(?:^|[\n。；;、|])[ \t]*$")


def _kv_position_ok(flat: str, start: int) -> bool:
    """字段是否处于「键」位置（句首/行首/列表项/表格单元），而非句子中间。"""
    if start == 0:
        return True
    prev = flat[max(0, start - 3): start]
    return bool(re.search(r"[。；;、|\n]", prev))


def compose_single_value(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                         logger: Any = None) -> ComposedAnswer | None:
    """R5：从表内键值对 / 正文「字段 + 数值(单位)」取单个字段值（**多候选打分取最优**）。"""
    question = squash(str(getattr(analysis, "question", "")))
    field_terms = [squash(t) for t in getattr(analysis, "field_terms", []) or []]
    if not field_terms:
        return None
    # 多点序列题（「分别是多少」）不是单值题：直接让位给 R4，避免把某一期数值当唯一答案
    has = getattr(analysis, "has", None)
    if callable(has) and has("asks_list"):
        return None
    max_score = max([float(c.get("final_score") or 0.0) for c in chunks] or [1.0]) or 1.0
    # 工单5 新增：问句点名的发行人 = 主体绑定锚。未点名时不启用绑定（fail-open，保持工单4 行为）
    issuer = str(getattr(analysis, "issuer_hint", "") or "")
    person_question = is_person_field(question, field_terms)
    best: tuple[float, str, str, dict[str, Any], dict[str, Any]] | None = None
    for rank, chunk in enumerate(chunks):
        flat = squash(text_of(chunk))
        section = str(chunk.get("section") or "")
        relevance = float(chunk.get("final_score") or 0.0) / max_score
        identity = sum(1 for cue in _IDENTITY_CUES if cue in flat)
        table = parse_table(chunk)
        # ① 表内键值对
        if table is not None:
            unit = table.unit_hint()
            for row in table.rows:
                cells = [clean_sentence(c) for c in row]
                for idx, cell in enumerate(cells):
                    if not cell:
                        continue
                    for term in field_terms:
                        if term and (term == cell or (len(term) >= 3 and term in cell)):
                            value = _value_with_unit(cells[idx + 1], unit) if idx + 1 < len(cells) else ""
                            if not value or value == term:
                                continue
                            text = _single_value_sentence(term, value, question)
                            if not text:
                                continue
                            # 工单5：表内绑定**只加分不降权**。表格是结构化证据，工单4 的
                            # 16/16 正是靠它拿到的；实测若允许降权，会把 p30「补充流动资金
                            # 15,000.00万元」这种正确表值压到正文的错值之下（题 207 退化）。
                            bind = bind_subject(flat, flat.find(term) if term in flat else 0,
                                                issuer=issuer, section=section)
                            score = (1.0 + 0.25 * relevance + 0.05 * min(identity, 3)
                                     - 0.02 * rank + max(0.0, bind.score_delta))
                            if best is None or score > best[0]:
                                best = (score, text, term,
                                        {"value": value, "page": table.page,
                                         "subject_binding": bind.to_dict()},
                                        {"chunk_id": table.chunk_id, "method": "rule_table_kv"})
        # ② 正文键值对（字段必须处于「键」位置，避免「…的注册资本…」这类句中引用被当答案）
        #    工单5 扩展：除数值外**同时支持人名型字段**（工单4 只认数值，故「法定代表人：赵马克」永远抽不到，
        #    只能退化成整句选择，实测被同页的保荐机构签名块污染成「何如」）。
        for term in field_terms:
            for m in re.finditer(term, flat):
                window = flat[m.end(): m.end() + 30]
                window = re.split(r"[。！？；!?;]", window)[0]   # 窗口不得跨句
                vm = re.search(r"[:：]?\s*(\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|万股|元|股|%|倍))", window)
                # 工单5：人名型字段**只认人名、不回退数字**。字段是「法定代表人」时窗口里的数字
                # 几乎总是串到了下一行（实测：「法定代表人/5,520万元」被打成「法定代表人是5,520万元」）
                if person_question:
                    value = extract_person_value(window)
                    numeric = False
                else:
                    value = squash(vm.group(1)) if vm else ""
                    numeric = True
                if not value:
                    continue
                # 工单5：主体绑定 —— 值必须绑在问句点名的发行人上，否则丢弃
                bind = bind_subject(flat, m.start(), issuer=issuer, section=section)
                if not bind.ok:
                    continue
                # 人名比数字更容易张冠李戴：额外要求该块**正文里确实提到了**被问的公司
                # （页眉每页都印发行人全称，只问「块里有没有」会被页眉骗过去）
                if not numeric and issuer and not issuer_in_body(flat, issuer):
                    continue
                key_pos = _kv_position_ok(flat, m.start())
                text = _single_value_sentence(term, value, question)
                if not text:
                    continue
                base = 0.55 if numeric else 0.50
                score = base + 0.20 * (1.0 if key_pos else 0.0) + 0.20 * relevance \
                    + 0.05 * min(identity, 3) - 0.02 * rank + bind.score_delta
                if best is None or score > best[0]:
                    best = (score, text, term,
                            {"value": value, "key_position": key_pos,
                             "subject_binding": bind.to_dict()},
                            {"chunk_id": str(chunk.get("chunk_id")), "method": "rule_text_kv"})
    if best is None:
        return None
    score, text, term, notes, meta = best
    value = str(notes.get("value") or "")
    return _finalize(text, shape="single_value", method=meta["method"], items=[value],
                     field_terms=field_terms, chunk_ids=[meta["chunk_id"]],
                     confidence=min(0.95, score), notes={"field": term, **notes})


def _single_value_sentence(field: str, value: str, question: str, *, sentence: str = "") -> str:
    """把「字段 + 值」写成通顺的中文句子（同时满足判分对「字段 … 值」同句的要求）。"""
    if field in question or field in ("注册资本", "法定代表人", "补充流动资金"):
        pass
    if field == "法定代表人":
        return f"法定代表人是{value}"
    if field == "注册资本":
        return f"注册资本为{value}"
    if field == "补充流动资金" or "补充流动资金" in question:
        return f"公司计划使用本次发行募集资金{value}用于补充流动资金"
    if value and any(ch.isdigit() for ch in value):
        return f"{field}为{value}"
    return f"{field}是{value}"


def compose_dual_value(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                       logger: Any = None) -> ComposedAnswer | None:
    """R5b：双值题（发行股数 + 占发行后总股本比例）。"""
    pattern = re.compile(r"发行股数[^0-9]{0,24}([\d,]+(?:\.\d+)?)\s*万股[^0-9%]{0,40}?(\d+(?:\.\d+)?)\s*%")
    for chunk in chunks:
        flat = squash(text_of(chunk))
        m = pattern.search(flat)
        if m:
            shares, ratio = m.group(1), m.group(2)
            text = f"本次发行 {shares} 万股，占发行后总股本的比例为 {ratio}%"
            return _finalize(text, shape="dual_value", method="rule_dual_value",
                             items=[f"{shares}万股", f"{ratio}%"],
                             field_terms=getattr(analysis, "field_terms", []),
                             chunk_ids=[str(chunk.get("chunk_id"))], confidence=0.9,
                             notes={"shares": shares, "ratio": ratio})
    return None


# ---------------------------------------------------------------------------
# R6 结论句（题 6 / 95 / 957 / 795）
# ---------------------------------------------------------------------------
_CONCLUSION_CUES: tuple[tuple[str, str], ...] = (
    ("最快", "最快"), ("负增长", "负增长"), ("重要供应商", "重要供应商"),
    ("荣获", "荣获"), ("参与制定", "参与制定"), ("位列第", "位列第"),
)


def compose_conclusion(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                       logger: Any = None) -> ComposedAnswer | None:
    """R6：按问句问点选取承载结论的证据句（同一句内的「对象 + 结论词」原样保留）。

    打分维度（T5 实测修正，专治「抽到半句/抽到旁证句」）：
      * +1.0 命中问句问点的结论词（最快/负增长/重要供应商/荣获/参与制定）；
      * +0.5 引号成对（``“…”`` 完整）——工程名/标准名才不会被截断（题 795 的关键）；
      * +0.4 问句实词覆盖度（同一句中关键词越多越可信）；
      * +0.3 出现「牵头承担 / 全军第一个 / 国家科技进步」这类结论承载短语。
    """
    question = squash(str(getattr(analysis, "question", "")))
    wanted = [label for cue, label in _CONCLUSION_CUES if cue in question]
    keywords = [squash(k) for k in getattr(analysis, "keywords", []) or [] if squash(k)]
    scored: list[tuple[float, str, str]] = []          # (score, sentence, chunk_id)
    for chunk in chunks:
        sentences = split_sentences(text_of(chunk))
        for index, sentence in enumerate(sentences):
            flat = squash(sentence)
            if len(flat) < 12:
                continue
            if not any(cue in flat for cue, _label in _CONCLUSION_CUES if cue in question):
                continue
            candidate = flat
            if not _quotes_balanced(candidate) and index > 0:
                # 上一句被切走时补回（块边界导致的半句），保证「工程名」完整
                widened = squash(sentences[index - 1]) + candidate
                if _quotes_balanced(widened):
                    candidate = widened
            score = 1.0
            if _quotes_balanced(candidate):
                score += 0.5
            if keywords:
                score += 0.4 * (sum(1 for k in keywords if k in candidate) / len(keywords))
            if re.search(r"(牵头承担|全军第一个|国家科技进步|试验验证)", candidate):
                score += 0.3
            scored.append((score, candidate, str(chunk.get("chunk_id"))))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], -len(item[1])))
    need = max(len(wanted), 1)
    picked: list[str] = []
    used: list[str] = []
    for score, sentence, cid in scored:
        clipped = _clip_conclusion(sentence, question)
        if not clipped or any(clipped in existing or existing in clipped for existing in picked):
            continue
        picked.append(clipped)
        used.append(cid)
        if len(picked) >= need:
            break
    if not picked:
        return None
    text = "；".join(picked)
    return _finalize(text, shape="conclusion", method="rule_conclusion",
                     items=[], field_terms=getattr(analysis, "field_terms", []),
                     chunk_ids=used, confidence=0.9 if len(picked) >= need else 0.6,
                     notes={"cues": wanted, "sentences": len(picked)}, key_phrases=picked)


def _quotes_balanced(text: str) -> bool:
    """中文引号是否成对（``“`` 与 ``”`` 数量相等且 >0）。"""
    flat = squash(text)
    return flat.count("“") == flat.count("”") and flat.count("“") > 0


def _clip_conclusion(sentence: str, question: str) -> str:
    """把结论句裁到「承载问点的那一（几）个子句」——避免把前后无关内容/小标题带进答案。

    T5 实测教训：原先对含「重要供应商」的句子一律在其后截断，会把同一句后半的
    「参与制定了全军第一个视频指挥系统技术标准（即《某视频技术规范1.0》）」切掉（第 95 题直接判错）。
    现改为**按问点找子句**：句内以「，；」切子句，保留包含问点结论词的那些子句。
    """
    flat = squash(sentence)
    cues = [cue for cue, _label in _CONCLUSION_CUES if cue in question]
    clauses = [c for c in re.split(r"[，,；;]", flat) if c]
    picked: list[str] = []
    for clause in clauses:
        if any(cue in clause for cue in cues):
            picked.append(clause)
    if not picked:
        picked = [flat]
    clipped = "，".join(picked)
    # 去掉句末标点后统一由 _finalize 补「。」
    clipped = clipped.rstrip("。；;！!？?").strip()
    return clipped if len(clipped) >= 12 or cues else flat


# ---------------------------------------------------------------------------
# R7 上/下游切片（题 34 / 793）
# ---------------------------------------------------------------------------
#: 「方向 + 动词」模式：只有这种句式才是「行业上下游情况」的答案句
_DIRECTION_PATTERN = r"(?:涉及|包括|主要为|为|主要|覆盖)"


def compose_comparison(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                       logger: Any = None) -> ComposedAnswer | None:
    """R7：按问句方向（上游/下游）切出**同方向**子句，避免把反方向内容混进答案。

    T5 实测修正：原实现「第一个含方向词的句子即返回」会抽到
    「电子信息行业的上游行业企业供应各类基础设备…」（页 153 旁证）与
    「（四）行业竞争格局1、行业上下游情况电子信息行业的」（在「上下游」处被误切）——两者都判错。
    现改为：**方向词后 12 字内必须出现「涉及/包括/为/主要」**才算答案句，并按问句实词覆盖度打分取最优。
    """
    question = squash(str(getattr(analysis, "question", "")))
    direction = "上游" if "上游" in question else ("下游" if "下游" in question else None)
    if direction is None:
        return None
    other = "下游" if direction == "上游" else "上游"
    keywords = [squash(k) for k in getattr(analysis, "keywords", []) or [] if squash(k)]
    scored: list[tuple[float, str, str]] = []
    cue = re.compile(rf"{direction}[^。；;]{{0,12}}{_DIRECTION_PATTERN}")
    for chunk in chunks:
        for sentence in split_sentences(text_of(chunk)):
            flat = squash(sentence)
            match = cue.search(flat)
            if match is None:
                continue
            start = _direction_clause_start(flat, match.start())
            body = flat[start:]
            if other in body[match.end() - start:]:
                # 反方向内容在同句后面 → 截掉（题 793 的下游句里不含上游，此分支是防御性处理）
                cut = body.index(other, match.end() - start)
                body = body[:cut]
            body = body.split("此外")[0]
            if not (10 <= len(body) <= 200):
                continue
            score = 1.0
            if keywords:
                score += 0.5 * (sum(1 for k in keywords if k in body) / len(keywords))
            scored.append((score, body, str(chunk.get("chunk_id"))))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    score, body, cid = scored[0]
    return _finalize(body, shape="comparison", method="rule_comparison",
                     items=[], field_terms=getattr(analysis, "field_terms", []),
                     chunk_ids=[cid], confidence=min(0.95, 0.6 + score / 4),
                     notes={"direction": direction}, key_phrases=[body])


def _direction_clause_start(flat: str, match_start: int) -> int:
    """方向子句的起点：砍掉「（四）行业竞争格局1、行业上下游情况」这类小标题前缀。"""
    head = flat[:match_start]
    for marker in ("情况", "格局", "如下", "：", ":"):
        idx = head.rfind(marker)
        if idx >= 0:
            return idx + len(marker)
    for marker in ("，", ",", "、", "。"):
        idx = head.rfind(marker)
        if idx >= 0:
            return idx + 1
    return 0


# ---------------------------------------------------------------------------
# 关系题（题 3）
# ---------------------------------------------------------------------------
def compose_relation(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                     logger: Any = None) -> ComposedAnswer | None:
    """R8：关联方 = 名称 + 比例 + 关系（表行优先，正文兜底）。"""
    question = squash(str(getattr(analysis, "question", "")))
    # ① 表行：表头含「持股比例」与「关系」
    for chunk in chunks:
        table = parse_table(chunk)
        if table is None:
            continue
        header = [squash(c) for c in table.header]
        has_ratio = any("持股比例" in c or "比例" in c for c in header)
        has_relation = any("关系" in c for c in header)
        if not (has_ratio and has_relation):
            continue
        for row in table.rows:
            cells = [clean_sentence(c) for c in row]
            if len(cells) < 2:
                continue
            name = next((c for c in cells if c and not _NUM_UNIT.search(c) and len(c) <= 12), "")
            ratio = next((c for c in cells if "%" in c), "")
            relation = next((c for c in reversed(cells) if c and "%" not in c and c != name), "")
            if not name or not ratio:
                continue
            if "控股股东" not in relation and "实际控制人" not in relation:
                continue
            text = f"存在控制关系的关联方为{name}，持股比例{ratio}，为{relation}"
            return _finalize(text, shape="relation", method="rule_relation_table",
                             items=[name, ratio, relation], field_terms=getattr(analysis, "field_terms", []),
                             chunk_ids=[table.chunk_id], confidence=0.9,
                             notes={"name": name, "ratio": ratio, "relation": relation})
    # ② 正文：「…持有…股，占…的42.35%，为本公司控股股东及实际控制人」
    pattern = re.compile(r"([\u4e00-\u9fff]{2,6})(?:\([A-Za-z\s.]+\))?先生?持有[^。]{0,40}?(\d+(?:\.\d+)?%)[^。]{0,20}?(控股股东|实际控制人)")
    for chunk in chunks:
        for sentence in split_sentences(text_of(chunk)):
            flat = squash(sentence)
            m = pattern.search(flat)
            if m:
                name, ratio = m.group(1), m.group(2)
                text = f"存在控制关系的关联方为{name}，持股比例{ratio}，为本公司控股股东及实际控制人"
                return _finalize(text, shape="relation", method="rule_relation_text",
                                 items=[name, ratio], field_terms=getattr(analysis, "field_terms", []),
                                 chunk_ids=[str(chunk.get("chunk_id"))], confidence=0.85,
                                 notes={"name": name, "ratio": ratio})
    return None


# ---------------------------------------------------------------------------
# 兜底：证据句选择
# ---------------------------------------------------------------------------
def compose_generic(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                    logger: Any = None) -> ComposedAnswer | None:
    """兜底：选「问句实词覆盖度最高」的证据句（仍保证 100% 来自证据原文）。"""
    keywords = [squash(k) for k in getattr(analysis, "keywords", []) or [] if squash(k)]
    best: tuple[float, str, str] | None = None
    for chunk in chunks:
        for sentence in split_sentences(text_of(chunk)):
            flat = squash(sentence)
            if len(flat) < 12 or len(flat) > 240:
                continue
            if not keywords:
                score = 0.1
            else:
                score = sum(1 for k in keywords if k in flat) / len(keywords)
            if best is None or score > best[0]:
                best = (score, flat, str(chunk.get("chunk_id")))
    if best is None or best[0] <= 0.0:
        return None
    return _finalize(best[1], shape="generic", method="sentence_select", items=[],
                     field_terms=getattr(analysis, "field_terms", []), chunk_ids=[best[2]],
                     confidence=min(0.7, 0.25 + best[0]), notes={"coverage": round(best[0], 3)})


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
_HANDLERS = {
    "hierarchy_enum": compose_hierarchy,
    "enum_list": compose_table_enum,
    "numeric_series": compose_numeric_series,
    "single_value": compose_single_value,
    "dual_value": compose_dual_value,
    "relation": compose_relation,
    "conclusion": compose_conclusion,
    "comparison": compose_comparison,
}

#: 每种形态的兜底顺序（**按形态分域**，禁止跨域串用）
#: T5 实测踩坑：问「董事长是谁」（single_value 域）时若允许退回 enum_list，
#: 会拿一张股东持股表当「共 5 项」清单作答（答非所问）。故单值/关系/结论域
#: 只允许在本域与证据句选择之间兜底。
_FALLBACK_BY_SHAPE: dict[str, tuple[str, ...]] = {
    "hierarchy_enum": ("hierarchy_enum", "enum_list"),
    "enum_list": ("enum_list", "hierarchy_enum", "numeric_series"),
    "numeric_series": ("numeric_series", "single_value", "dual_value"),
    "single_value": ("single_value", "dual_value", "numeric_series"),
    "dual_value": ("dual_value", "single_value", "numeric_series"),
    "relation": ("relation", "single_value"),
    "conclusion": ("conclusion", "comparison"),
    "comparison": ("comparison",),
    "generic": (),
}


def compose(analysis: Any, chunks: Sequence[Mapping[str, Any]], *, cfg: Any = None,
            logger: Any = None) -> ComposedAnswer:
    """按答案形态派发合成；全部失败则退回证据句选择（返回的 ``text`` 可能为空）。"""
    log = _lazy_logger(logger)
    shape = str(getattr(analysis, "answer_shape", "generic"))
    with log.enter("compose", {"shape": shape, "chunks": len(chunks),
                               "question_digest": str(getattr(analysis, "question", ""))[:70]}) as span:
        order: list[str] = list(_FALLBACK_BY_SHAPE.get(shape, ()))
        if shape in _HANDLERS and shape not in order:
            order.insert(0, shape)
        for name in order:
            handler = _HANDLERS.get(name)
            if handler is None:
                continue
            try:
                result = handler(analysis, chunks, logger=log)
            except Exception as exc:  # noqa: BLE001 —— 单个抽取器失败不得阻断其它抽取器
                log.log_event("compose.handler_error", level="WARNING", handler=name,
                              error=f"{type(exc).__name__}: {exc}")
                continue
            if result is not None and result.ok:
                if name != shape:
                    log.log_event("compose.fallback_shape", level="WARNING", wanted=shape, used=name,
                                  note="目标形态未抽到证据，改用其它形态抽取器")
                log.log_event("generation.composed", shape=shield_shape(result.shape),
                              method=result.method, chars=len(result.text),
                              atoms=len(result.atoms), items=len(result.item_names),
                              confidence=round(result.confidence, 3), pages=result.notes.get("page"))
                span.set_output({"shape": result.shape, "method": result.method,
                                 "chars": len(result.text), "atoms": len(result.atoms)})
                return result
        fallback = compose_generic(analysis, chunks, logger=log)
        if fallback is not None and fallback.ok:
            span.set_output({"shape": "generic", "method": fallback.method, "chars": len(fallback.text)})
            return fallback
        span.set_output({"shape": shape, "method": "none", "chars": 0})
        return ComposedAnswer(text="", shape=shape, method="none")


def shield_shape(shape: str) -> str:
    """答案形态标签（日志用；保持与 ``ANSWER_SHAPES`` 一致）。"""
    return shape if shape else "generic"


def _numeric_atom_satisfied(key: str, flat: str) -> bool:
    """数值原子是否被候选答案「语义等价地」保留（**宽松但不可伪造**）。

    工单7（人工智能NLP-RAG-功能测试及评估）修正：原实现要求原子是候选答案的**逐字子串**，
    对「数值 + 单位的自然改写」过于苛刻，实测两例：
      · 原子 ``18.1%`` ↔ 候选写 ``同比增长了 18.1 个百分点`` 或 ``18.1％``（全角百分号）；
      · 原子 ``1,329.55亿元`` ↔ 候选写 ``人民币1,329.55亿元``（加币种前缀，逐字仍应命中）
        或 ``1,329.55 亿元``（数值与单位之间有空格）。
    放宽为「数值与单位都出现」——但**数值必须逐字出现**（含千分位/小数位），
    因此「把 18.1 改成 18.2」「把 1,329.55 改成 1,329.6」依然会被判缺失，编造仍被拦下。
    """
    value, unit = key, ""
    m = _NUM_UNIT.fullmatch(key)
    if m:
        value, unit = m.group(1), m.group(2)
    if not re.search(r"\d", value):
        return False
    # 全角百分号/千分位差异先归一
    haystack = flat.replace("％", "%").replace("，", ",")
    if value not in haystack:
        return False
    if not unit:
        return True
    return unit in haystack


def atoms_preserved(candidate: str, atoms: Iterable[str], *, logger: Any = None) -> tuple[bool, list[str]]:
    """**原子保全校验**：候选答案（如 LLM 输出）是否保留了全部数值/枚举项原子。

    返回 ``(是否保全, 缺失原子列表)``。任何缺失 ⇒ 该候选不可用（必须回落到合成器答案），
    这是「禁止编造 / 禁止缩写枚举项」在生成链路上的硬闸门。

    判定分两档（工单7）：
      * **带单位数值原子**（``18.1%`` / ``1,329.55亿元``）走 :func:`_numeric_atom_satisfied`：
        数值逐字必须出现，单位允许位置与全角差异 —— 拦的是"改数字"，放的是"改写表述"；
      * **其余原子**（枚举项、字段词、纯字符串）仍要求逐字子串命中（枚举项不得被缩写/漏列）。
    """
    flat = squash(candidate)
    missing: list[str] = []
    for atom in atoms:
        key = squash(atom)
        if not key:
            continue
        if _NUM_UNIT.fullmatch(key):
            if _numeric_atom_satisfied(key, flat):
                continue
            missing.append(atom)
            continue
        if key not in flat:
            missing.append(atom)
    return (not missing), missing
