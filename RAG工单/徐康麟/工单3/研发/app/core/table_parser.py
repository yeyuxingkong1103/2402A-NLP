# -*- coding: utf-8 -*-
"""工单3 表格归一化与跨页续表合并（设计/接口设计.md §2.2、§3.6；系统架构 §6.2~§6.4 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

实测背景（环境事实 §3.2/§8.8）：
    * ``page.find_tables()`` 存在两个真实缺陷：**合并单元格产生 None**、**格内换行**；
    * 表格**跨页是常态**（PDF1 90% 的含表页是连续段的一部分，最长 28 连页）；
    * 合并单元格的值还可能是符号 ``[ ◆ ]``（PDF2 物理 2/22/24/27/306 共 22 处）→ 归一化为 ``未披露``。

规则（R1~R9）：
    R1 换行归一：输出单元格不含 ``\\n``（``'有限公司成立日\\n期'`` → ``'有限公司成立日期'``）
    R2 None 处理：表头相邻 ``None`` 合并为 1 逻辑列；数据行 ``None`` 按最近**左侧**非空值填充
    R3 全空列删除（notes 记录）
    R4 退化表：逻辑列 < min_cols 或有效数据行 = 0 → ``degenerate=True``、``markdown=""``
    R5 Markdown：首行表头 + ``---`` 分隔行 + 数据行（竖线转义）
    R6 数字保真：``'5,520.00 万元'``、``'25.04%'`` 原样保留
    R7 续表继承：首行首格空/``None`` 且上一页含表 → 并入前块、继承表头
    R8 重复表头去重：本页首行与上一页首行相同 → 丢弃重复表头、只并数据行
    R9 逻辑表块元数据：``page_start``/``page_end``（1-based 闭区间）、``page = page_start``、``row_pages`` 与 ``rows`` 等长

字段扩展说明（设计 §2 允许新增带默认值的字段）：新增 ``raw_first_row`` / ``first_row_is_data`` /
``absorbed_pages`` / ``flat_cols`` / ``header_row_count`` 五个留痕字段，语义见各自注释。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields as dc_fields
from typing import Any, Iterable, Mapping, Sequence

from .errors import TableParseError
from .text_utils import (
    collapse_whitespace,
    extract_numbers,
    normalize_cell,
    normalize_text,
    text_digest,
)

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 表标题提示词：STRONG 是「标题级」短语（表格单元格里不会出现），WEAK 仅作参考
TITLE_STRONG: tuple[str, ...] = (
    "如下表所示", "如下所示", "下表所示", "单位：", "单位:", "情况如下", "明细如下", "概算如下",
    "构成为", "用途如下", "计划如下", "结构如下", "分布如下", "表如下", "具体情况",
)
TITLE_WEAK: tuple[str, ...] = (
    "情况", "明细", "概算", "构成", "用途", "计划", "结构", "分布", "合计", "金额",
)
# 编号式小标题（弱回退用）：1、xxx / 第一章 / 一、xxx / （一）xxx
_HEADING_LIKE = re.compile(
    r"^(?:\d{1,2}[、.．]|第[一二三四五六七八九十百零〇0-9]{1,4}[节章]|[一二三四五六七八九十]{1,3}、|（[一二三四五六七八九十]{1,3}）)\S{1,30}$"
)
# 表头续行（两级表头第二行）允许的最大单元格长度
_SUB_HEADER_MAX_LEN = 12


# ---------------------------------------------------------------------------
# 数据结构（设计 §2.2 + 追加留痕字段）
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TableBlock:
    """一个**逻辑表块**（含跨页合并结果）。"""

    table_id: str
    file_name: str
    page: int                                  # = page_start，对外引用页（1-based 物理页）
    page_start: int
    page_end: int
    bbox: tuple[float, float, float, float]
    n_rows: int                                # 首页原始物理行数
    n_cols: int                                # 首页原始物理列数
    logical_cols: int                          # 归一化后**表头分组**列数（PDF1 物理 130 = 5，锚点）
    header: list[str]                          # 归一化表头（两级表头已压平；续表继承自 page_start）
    rows: list[list[str]]                      # 归一化数据行（无 None、无换行）
    row_pages: list[int]                       # 每行所属物理页（与 rows 等长）
    markdown: str                              # 退化表为空串 ""
    title: str
    key_numbers: list[str]
    degenerate: bool
    continued_from: str | None = None
    header_inherited: bool = False
    none_cells: int = 0
    notes: list[str] = field(default_factory=list)
    # —— 追加字段（带默认值，向后兼容）——
    raw_first_row: list[str] = field(default_factory=list)   # 归一化后的**原始首行**，R7/R8 判据依据
    first_row_is_data: bool = False                          # 首行被当作数据行（续表候选）时为 True
    absorbed_pages: list[int] = field(default_factory=list)  # 被并入本块的续表物理页
    flat_cols: int = 0                                       # markdown/rows 实际列数（两级表头压平后）
    header_row_count: int = 1                                # 被识别为表头的行数（2 = 两级表头）
    sparse_mode: bool = False                                # 稀疏网格压缩模式（合并单元格折叠，不重复填充）

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableBlock":
        names = {f.name for f in dc_fields(cls)}
        payload = {k: v for k, v in data.items() if k in names}
        if "bbox" in payload:
            payload["bbox"] = tuple(payload["bbox"])
        return cls(**payload)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _lazy_logger(logger: Any, module: str) -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def table_to_markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """R5：``| a | b |`` + ``| --- | --- |`` + 数据行；单元格内的竖线转义。"""
    def esc(cell: Any) -> str:
        return str("" if cell is None else cell).replace("|", "\\|").strip()

    head = [str(h) for h in header]
    if not head:
        return ""
    lines = ["| " + " | ".join(esc(h) for h in head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for row in rows:
        cells = [esc(c) for c in row]
        if len(cells) < len(head):
            cells.extend([""] * (len(head) - len(cells)))
        elif len(cells) > len(head):
            cells = cells[: len(head)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def merge_none_header(header: Sequence[str | None]) -> list[str]:
    """R2（表头）：相邻 ``None``/空串与前一个表头同组，合并为 1 列。"""
    out: list[str] = []
    for cell in header:
        text = normalize_cell(cell)
        if text == "" and out:
            continue                        # 与前一个表头同组 → 合并
        out.append(text)
    return out


def forward_fill_merged_rows(rows: Sequence[Sequence[str | None]], *, header_rows: int = 1) -> list[list[str]]:
    """R2（数据行）**细化版**：仅做单元格归一，``None``/空串一律置空（**不做左填充**）；前 ``header_rows`` 行同样归一。

    **为什么改（实测驱动，已上报 captain）**：``find_tables`` 把「合并单元格的后续列」与「本行右侧本来就没有的格」
    都表示为 ``None``；按原 R2 一律「左侧填充」会把同一个逻辑值复制到多列
    （实测 PDF1 物理 22/53/61、PDF2 物理 22 出现 3 次重复 → BM25 词频虚增 3 倍 + 答案抽取出重复文本）。
    置空则保持**列位置正确**：``['合计', None, None, '305,800.00']`` → ``['合计', '', '', '305,800.00']``。

    表头的「相邻 ``None`` 合并为一列」仍由 :func:`merge_none_header` / 稀疏网格压缩负责（未改动）。
    """
    result: list[list[str]] = []
    for row in rows:
        result.append([normalize_cell(c) for c in row])
    return result


def collapse_none_columns(rows: Sequence[Sequence[str]]) -> list[list[str]]:
    """R3：删除全空列（返回新矩阵，列数可能变少）。"""
    if not rows:
        return []
    width = max(len(r) for r in rows)
    keep = [j for j in range(width) if any(j < len(r) and str(r[j]).strip() for r in rows)]
    out: list[list[str]] = []
    for row in rows:
        out.append([str(row[j]) if j < len(row) else "" for j in keep])
    return out


# 「纯数值单元格」：金额/百分比/小数/整数（**不含**年份日期短语，如 `2010 年1-6 月`）
_NUMERIC_CELL = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$|^[-+]?\d+(?:\.\d+)?%$")


def _is_numeric_cell(text: str) -> bool:
    """单元格是否为纯数值（表头里的年份/日期不算）。"""
    return bool(_NUMERIC_CELL.match(text.strip()))


def _row_looks_like_data(row: Sequence[str | None]) -> bool:
    """首行是否像**数据行**：至少 1 个纯数值单元格（金额/百分比/小数）。

    ⚠️ 实测教训：招股书表头大量含年份（``'2010 年1-6 月'``、``'2019 年度'``），
    若用「含任意数字」判定，PDF2 物理 22 的主表头会被误判为数据行，
    真表头被 ``列1..列N`` 合成列名替换（T3 全量审计发现并修复）。
    """
    cells = [normalize_cell(c) for c in row]
    return sum(1 for c in cells if _is_numeric_cell(c)) >= 1


def looks_like_continuation(first_row: Sequence[str | None]) -> bool:
    """R7 判据（契约签名）：首行首格为空/``None`` → 疑似上表续行。

    调用方（``group_continued_tables``）还必须叠加「上一页含表 + 列数匹配 + 首行像数据行」三重校验。
    """
    if not first_row:
        return False
    head = first_row[0]
    return head is None or normalize_cell(head) == ""


def is_repeated_header(prev_block: TableBlock, first_row: Sequence[str | None]) -> bool:
    """R8 判据：本页首行与上一页（表头行）**完全相同** → 重复表头。

    实现要点：归一化后去掉尾部空格再逐格**精确比较**（不给容差）。
    早期版本容忍 1 个格子差异，实测会把数据行误判为重复表头并整行丢弃（数据丢失），已修正。
    """
    if not first_row or not prev_block.raw_first_row:
        return False
    cur = [normalize_cell(c) for c in first_row]
    prev = [normalize_cell(c) for c in prev_block.raw_first_row]
    while cur and cur[-1] == "":
        cur.pop()
    while prev and prev[-1] == "":
        prev.pop()
    return bool(cur) and cur == prev


def is_degenerate(header: Sequence[str], rows: Sequence[Sequence[str]], *, min_cols: int = 2) -> bool:
    """R4：逻辑列 < min_cols 或有效数据行 = 0 → 退化表。

    注：``[ ◆ ]`` 已按 captain 实测裁定归一化为 ``未披露``，因此它的行**算有效行**
    （否则「其他与主营业务相关的营运资金 ↔ 金额」的对应关系会断裂）；退化只由「列不足」与「全空」触发。
    """
    cols = [h for h in header if str(h).strip()]
    if len(cols) < int(min_cols):
        return True
    valid_rows = sum(1 for row in rows if any(str(c).strip() for c in row))
    return valid_rows == 0


def table_title_for(
    page_text: str,
    bbox: tuple[float, float, float, float] | None,
    page: int,
    *,
    max_len: int = 60,
) -> str:
    """取表标题（架构 §6.3）：定位页内**最后一个标题级行**，再向上回并最多 2 行续行。

    判据（避免把表格单元格当成标题，实测教训：PV130 的单元格 ``占比`` 曾误判为标题）：
        ① 只认「标题级」短语（``如下表所示``/``单位：``/``具体情况``…），单元格文本不会命中；
        ② 续行必须长度 ≤ 45 且以 ``，``/``：`` 结尾，最多回并 2 行；
        ③ 取不到→ ``f"{page} 页表格"``（不得为空，避免检索无锚点）。
    """
    lines = [ln.strip() for ln in str(page_text or "").splitlines() if ln.strip()]
    anchor = -1
    for index, line in enumerate(lines):
        if len(line) > 45 or line.endswith("。"):
            continue
        if any(token in line for token in TITLE_STRONG):
            anchor = index
    if anchor < 0:
        # 弱回退：取页内最后一个「编号式小标题」（如「2、行业市场化程度」），优于「N 页表格」
        for index, line in enumerate(lines):
            if len(line) <= 45 and not line.endswith("。") and _HEADING_LIKE.match(line):
                anchor = index
        if anchor < 0:
            return f"{page} 页表格"
        return collapse_whitespace(lines[anchor])[:max_len]
    picked = [anchor]
    cursor = anchor - 1
    while cursor >= 0 and len(picked) < 3:
        prev = lines[cursor]
        if len(prev) <= 45 and not prev.endswith("。") and prev.endswith(("，", "：", ":", ",")):
            picked.insert(0, cursor)
            cursor -= 1
            continue
        break
    title = collapse_whitespace("".join(lines[index] for index in picked))
    if not title:
        return f"{page} 页表格"
    return title[:max_len]


# 单位格（两级表头的单位续行，如 '(万元)'、'万元'、'%'）：必须是「括号短单位」或「纯单位词」
_UNIT_CELL = re.compile(
    r"^[（(][^（()）]{0,8}[)）]$"
    r"|^[%％]?[万亿]?元(?:/[^）)]{1,8})?$"
    r"|^(?:倍|股|万股|次|年|月|日|%|％|人|家|台|个|条|项|平方米|吨)$"
)


def _is_unit_cell(text: str) -> bool:
    """单元格是否为「纯单位」文本（用于识别单位续行）。"""
    return bool(_UNIT_CELL.match(text.strip()))


def _sub_header_kind(row: Sequence[str | None], n_cols: int) -> str | None:
    """判断第二行是否仍是表头：``"full"``（覆盖多数列的两级表头）/ ``"unit"``（只填单位的续行）/ ``None``。

    * full：``['一级单位','二级单位','金额','占比',…]``（PDF1 物理 130）—— 无数字、短文本、覆盖多数列；
    * unit：``[None, None, None, '(万元)', None, None, None]``（PDF2 物理 306）—— 少量**纯单位**格，
      必须并入其所属表头标签（``项目总投资`` + ``(万元)``），否则会被当成数据行破坏列对齐。
    * 实测教训：数据行 ``['承销费和保荐费', None, None, '未披露', …]``（PDF2 物理 24）全是短文本且无数字，
      早期「短文本即单位行」的判据会把它误当表头 —— 因此单位格必须命中 ``_UNIT_CELL`` 单位模式。
    """
    if not row:
        return None
    cells = [normalize_cell(c) for c in row]
    non_empty = [c for c in cells if c]
    if not non_empty:
        return None
    if not all(len(c) <= _SUB_HEADER_MAX_LEN and not re.search(r"\d", c) for c in non_empty):
        return None
    if len(non_empty) >= max(2, min(n_cols, 4)):
        return "full"
    has_gap = any(c == "" for c in cells)
    if has_gap and all(_is_unit_cell(c) for c in non_empty):
        return "unit"
    return None


def _pad_tail(row: Sequence[str], width: int) -> list[str]:
    """把行尾补空到指定列宽（稀疏模式下「行尾合并格不存在」的对齐手段）。"""
    cells = [str(c) for c in row]
    if len(cells) < width:
        cells.extend([""] * (width - len(cells)))
    return cells[:width]


def _header_with_units(row0: Sequence[str | None], sub_row: Sequence[str | None]) -> list[str]:
    """把单位续行并入所属表头标签（``'项目总投资'`` + ``'(万元)'`` → ``'项目总投资(万元)'``）。

    归属规则：单位格属于**其左侧最近的非空表头标签**所在的列组。
    """
    labels: list[str] = []
    owner: dict[int, int] = {}
    current = -1
    for pos, cell in enumerate(row0):
        text = normalize_cell(cell)
        if text:
            labels.append(text)
            current = len(labels) - 1
        owner[pos] = current
    for pos, cell in enumerate(sub_row):
        text = normalize_cell(cell)
        if not text:
            continue
        index = owner.get(pos, -1)
        if index < 0 or index >= len(labels):
            labels.append(text)          # 单位出现在任何标签之前 → 独立成列
        elif labels[index]:
            labels[index] = f"{labels[index]}{text}"
        else:
            labels[index] = text
    return labels


def _group_labels(row0: Sequence[str | None]) -> list[str]:
    """按物理列给出「所属表头分组标签」（``None`` 继承左侧标签）。"""
    labels: list[str] = []
    current = ""
    for cell in row0:
        text = normalize_cell(cell)
        if text:
            current = text
        labels.append(current)
    return labels


def _flatten_header(row0: Sequence[str | None], sub_row: Sequence[str | None]) -> list[str]:
    """两级表头压平：``2019 年1-6 月`` + ``金额`` → ``2019 年1-6 月 金额``。"""
    groups = _group_labels(row0)
    subs = [normalize_cell(c) for c in sub_row]
    flat: list[str] = []
    for index, group in enumerate(groups):
        sub = subs[index] if index < len(subs) else ""
        if not sub or sub == group:
            flat.append(group or sub)
        elif not group:
            flat.append(sub)
        else:
            flat.append(f"{group} {sub}")
    return flat


# ---------------------------------------------------------------------------
# 单表归一化
# ---------------------------------------------------------------------------
def _build_matrix(
    raw_rows: Sequence[Sequence[str | None]],
    *,
    min_cols: int,
) -> dict[str, Any]:
    """把原始行列矩阵归一化：识别表头（含两级）、左填充数据行、删全空列。"""
    if not raw_rows:
        raise TableParseError("表格没有任何行", code="RAG-2100")
    n_rows_raw = len(raw_rows)
    n_cols = max(len(r) for r in raw_rows)
    none_cells = sum(1 for row in raw_rows for cell in row if cell is None)
    notes: list[str] = []

    # 丢弃「全空行」（find_tables 在合并单元格/换页处会产出整行空白），以及首部「页断残缺行」
    # 全空行：find_tables 在合并单元格/换页处会产出整行空白；
    # 残缺行：上一页句子的尾巴被切成一行（例 PDF1 物理 21：``['', '', '家标准。']``），
    #         若不跳过，它会被当成表头 → 删空列后只剩 1 列 → 整张表被误判为退化表（实测 58 个空 markdown 的主因之一）。
    rows_seq: list[list[str | None]] = [list(r) for r in raw_rows]
    kept = [row for row in rows_seq if any(normalize_cell(c) for c in row)]
    if len(kept) != len(rows_seq):
        notes.append(f"丢弃全空行 {len(rows_seq) - len(kept)} 行")
    if kept:
        rows_seq = kept
    while len(rows_seq) > 1:
        cells = [normalize_cell(c) for c in rows_seq[0]]
        non_empty = [c for c in cells if c]
        if not non_empty:
            rows_seq.pop(0)
            notes.append("跳过首部全空行 1 行")
            continue
        fragment = (
            len(non_empty) == 1
            and (non_empty[0][-1] in "。，,；;！？!?" or len(non_empty[0]) > 12)
        )
        if fragment:
            rows_seq.pop(0)
            notes.append(f"跳过首部页断残缺行：{non_empty[0][:14]!r}")
            continue
        break

    row0 = list(rows_seq[0])
    # 首行「疑似续表」的两个信号（R7）：
    #   ① 首格为空/None 且像数据行（含纯数值格）——最典型（PDF1 物理 131）；
    #   ② 首格为空且**第二行首格也为空**——表头被切页时的残缺行（实测 PDF1 物理 66 的关联方表），
    #      单row 不像数据但整表前两行都缺首格，作为独立新表不合常理。
    row0_leading_gap = looks_like_continuation(row0)
    next_row_leading_gap = len(rows_seq) > 1 and looks_like_continuation(rows_seq[1])
    first_row_is_data = bool(row0_leading_gap and (_row_looks_like_data(row0) or next_row_leading_gap))

    if first_row_is_data:
        # 续表候选：首行是数据，表头留给合并阶段继承；未合并时补合成列名
        header = [f"列{i + 1}" for i in range(n_cols)]
        logical_cols = n_cols
        header_row_count = 0
        flat = list(header)
        notes.append("首行疑似续表（首格为空且像数据行），表头待继承")
        data_raw = [list(r) for r in rows_seq]
    else:
        # ---- 表头识别（可能两行：两级表头 full / 单位续行 unit）----
        sub_row: list[str | None] | None = None
        sub_kind: str | None = None
        if len(rows_seq) > 1 and any(normalize_cell(c) == "" for c in row0):
            sub_kind = _sub_header_kind(rows_seq[1], n_cols)
            if sub_kind:
                sub_row = list(rows_seq[1])
        header_row_count = 2 if sub_row is not None else 1
        data_slice = [list(r) for r in rows_seq[header_row_count:]]
        data_has_merge = any(c is None for row in data_slice for c in row)

        if sub_kind == "unit":
            header = _header_with_units(row0, sub_row or [])
            logical_cols = len(header)
            notes.append("检测到单位续行，已并入所属表头标签")
        elif data_has_merge:
            # 稀疏网格：表头压缩为「非空格」（数据行的合并单元格在下面统一折叠）
            header = [normalize_cell(c) for c in row0 if normalize_cell(c) != ""]
            logical_cols = len(header)
        elif sub_kind == "full":
            header = _flatten_header(row0, sub_row or [])
            logical_cols = len(merge_none_header(row0))
            notes.append("检测到两级表头，已压平为单行表头")
        else:
            header = merge_none_header(row0)
            logical_cols = len(header)

        data_raw = [list(r) for r in data_slice]

    # ---- 稀疏网格模式（实测缺陷修复）----
    # 例（PDF2 物理 22 募投表，原始 9 列）：
    #   表头 ['', '序号', '', '', '项目名称', '', '', '计划总投资(万元)', '']
    #   数据 ['1', None, None, '仓储及物流中心', None, None, '3,393.40', None, None]
    # 合并单元格在 find_tables 里被拆成「值 + 若干 None」，表头标签与数据值的**列起点还差 1 格**；
    # 若按「None 左填充」会得到 9 列且每个逻辑值重复 3 次（人读歧义 + BM25 词频虚增 3 倍）。
    # 正确做法：表头与数据行**使用同一套逻辑列映射** —— 两行都压缩为「非空格」，
    # 压缩后列数一致才采用（否则退回等宽左填充，保证不产生错位）。
    sparse_mode = False
    if not first_row_is_data and data_raw and any(c is None for row in data_raw for c in row):
        header_sparse = [normalize_cell(c) for c in header if normalize_cell(c) != ""]
        rows_sparse = [
            [normalize_cell(c) for c in row if normalize_cell(c) != ""] for row in data_raw
        ]
        lengths = [len(r) for r in rows_sparse]
        target = max([len(header_sparse)] + lengths) if lengths else len(header_sparse)
        # 「行尾合并格确实不存在」的行（原始行最后一个单元格为空/None）：允许尾部补空对齐；
        # 行中空洞（多个值缺失）会导致压缩后错位，一律不接受 → 退回等宽左填充。
        short_rows = [(row, cells) for row, cells in zip(data_raw, rows_sparse) if len(cells) < target]
        tail_gap_ok = all(
            len(cells) == target - 1 and normalize_cell(row[-1]) == "" for row, cells in short_rows
        )
        consistent = (
            len(header_sparse) in (target, target - 1)
            and all(n in (target, target - 1) for n in lengths)
            and tail_gap_ok
        )
        if consistent and target >= int(min_cols):
            header = _pad_tail(header_sparse, target)
            data_raw = [_pad_tail(r, target) for r in rows_sparse]
            logical_cols = target
            sparse_mode = True
            notes.append(
                f"稀疏网格：表头与数据行统一为 {target} 个逻辑列"
                f"（原 {n_cols} 列；合并单元格折叠不重复填充，{len(short_rows)} 行按尾部补空对齐）"
            )
        else:
            notes.append(
                f"稀疏网格压缩后列数不一致（表头 {len(header_sparse)}、行 {sorted(set(lengths))}），"
                f"退回等宽左填充模式"
            )

    # 数据行：None/空 → 左侧非空值填充（稀疏模式已压缩，跳过此步）
    filled = [list(r) for r in data_raw] if sparse_mode else forward_fill_merged_rows(data_raw, header_rows=0)
    matrix = [list(header)] + filled
    width_before = max(len(r) for r in matrix)
    matrix = collapse_none_columns(matrix)
    width_after = max((len(r) for r in matrix), default=0)
    removed = width_before - width_after
    if removed > 0:
        notes.append(f"删除全空列 {removed} 列")

    header_out = matrix[0] if matrix else []
    rows_out = matrix[1:] if len(matrix) > 1 else []
    return {
        "header": header_out,
        "rows": rows_out,
        "logical_cols": logical_cols,
        "flat_cols": len(header_out),
        "none_cells": none_cells,
        "notes": notes,
        "first_row_is_data": first_row_is_data,
        "header_row_count": header_row_count,
        "sparse_mode": sparse_mode,
        "raw_first_row": [normalize_cell(c) for c in row0],
        "n_cols": n_cols,
        "n_rows": n_rows_raw,
    }


def normalize_table(
    raw_rows: list[list[str | None]],
    *,
    page: int = 0,
    table_id: str = "",
    min_rows: int = 2,
    min_cols: int = 2,
    header_rows: int = 1,
    logger: Any = None,
) -> TableBlock:
    """把 ``find_tables().extract()`` 的原始矩阵归一化为 ``TableBlock``（单页，不含跨页合并）。"""
    log = _lazy_logger(logger, "table_parser")
    with log.enter("normalize_table", {"table_id": table_id, "page": page, "raw_rows": len(raw_rows)}) as span:
        built = _build_matrix(raw_rows, min_cols=min_cols)
        header: list[str] = built["header"]
        rows: list[list[str]] = built["rows"]
        degenerate = is_degenerate(header, rows, min_cols=min_cols)
        markdown = "" if degenerate else table_to_markdown(header, rows)
        key_numbers = extract_numbers(markdown + " " + " ".join(header))[:24]
        notes: list[str] = list(built["notes"])
        if len(raw_rows) < int(min_rows):
            notes.append(f"原始行数 {len(raw_rows)} < min_rows {min_rows}")
        block = TableBlock(
            table_id=table_id or f"unknown#p{page:04d}#t00",
            file_name="",
            page=page,
            page_start=page,
            page_end=page,
            bbox=(0.0, 0.0, 0.0, 0.0),
            n_rows=int(built["n_rows"]),
            n_cols=int(built["n_cols"]),
            logical_cols=int(built["logical_cols"]),
            header=header,
            rows=rows,
            row_pages=[page] * len(rows),
            markdown=markdown,
            title=f"{page} 页表格",
            key_numbers=key_numbers,
            degenerate=degenerate,
            none_cells=int(built["none_cells"]),
            notes=notes,
            raw_first_row=list(built["raw_first_row"]),
            first_row_is_data=bool(built["first_row_is_data"]),
            flat_cols=int(built["flat_cols"]),
            header_row_count=int(built["header_row_count"]),
            sparse_mode=bool(built.get("sparse_mode", False)),
        )
        log.log_event(
            "table.normalize", table_id=block.table_id, page=page, none_cells=block.none_cells,
            n_cols=block.n_cols, logical_cols=block.logical_cols, flat_cols=block.flat_cols,
            rows=len(block.rows), degenerate=block.degenerate, header_row_count=block.header_row_count,
        )
        if degenerate:
            log.log_event("table.degenerate", level="WARNING", table_id=block.table_id, page=page,
                          reason="逻辑列不足或有效数据行为 0", header_digest=text_digest(" | ".join(header)))
        span.set_output({"table_id": block.table_id, "logical_cols": block.logical_cols,
                         "flat_cols": block.flat_cols, "rows": len(block.rows), "degenerate": degenerate})
        return block


def build_table_block(
    raw_rows: list[list[str | None]],
    *,
    file_name: str,
    page: int,
    idx: int,
    title: str,
    logger: Any = None,
) -> TableBlock:
    """按契约组装带 ``table_id``/标题/文件名的表块（``table_id = stem#p{page:04d}#t{idx:02d}``）。"""
    stem = file_name.rsplit(".", 1)[0]
    table_id = f"{stem}#p{int(page):04d}#t{int(idx):02d}"
    block = normalize_table(raw_rows, page=page, table_id=table_id, logger=logger)
    block.file_name = file_name
    block.title = normalize_text(title) or f"{page} 页表格"
    log = _lazy_logger(logger, "table_parser")
    log.log_event("table.build", table_id=table_id, page=page, title=block.title,
                  rows=len(block.rows), markdown_chars=len(block.markdown))
    return block


# ---------------------------------------------------------------------------
# 跨页续表（R7/R8/R9）
# ---------------------------------------------------------------------------
def merge_continued_tables(
    prev_block: TableBlock,
    cont_block: TableBlock,
    *,
    connect_row: bool = False,
    logger: Any = None,
) -> TableBlock:
    """R7：把续表块的行并入前块，**表头沿用前块**，更新 ``page_end``/``row_pages``/``absorbed_pages``。

    ``connect_row=True`` 时尝试把跨页断行拼接（续表首行首格为空且前块末行尾部无句读时）。
    """
    log = _lazy_logger(logger, "table_parser")
    with log.enter("merge_continued_tables",
                   {"prev": prev_block.table_id, "cont": cont_block.table_id,
                    "prev_rows": len(prev_block.rows), "cont_rows": len(cont_block.rows)}) as span:
        added = [list(r) for r in cont_block.rows]
        if connect_row and prev_block.rows and added:
            # 跨页断行：前块末行尾格为空且续表首行首格为空 → 两行其实是同一逻辑行，按列拼接
            last_row = prev_block.rows[-1]
            first_row = added[0]
            if first_row and not str(first_row[0]).strip():
                width = max(len(last_row), len(first_row))
                joined: list[str] = []
                for j in range(width):
                    left = str(last_row[j]).strip() if j < len(last_row) else ""
                    right = str(first_row[j]).strip() if j < len(first_row) else ""
                    joined.append(left if not right else right if not left else f"{left}{right}")
                prev_block.rows[-1] = joined
                added = added[1:]
        prev_block.rows.extend(added)
        prev_block.row_pages.extend([cont_block.page_start] * len(added))
        prev_block.page_end = max(prev_block.page_end, cont_block.page_end)
        prev_block.absorbed_pages.extend([cont_block.page_start, cont_block.page_end]
                                         if cont_block.page_start != cont_block.page_end
                                         else [cont_block.page_start])
        prev_block.flat_cols = max(prev_block.flat_cols, cont_block.flat_cols)
        prev_block.markdown = "" if prev_block.degenerate else table_to_markdown(prev_block.header, prev_block.rows)
        prev_block.key_numbers = extract_numbers(prev_block.markdown + " " + " ".join(prev_block.header))[:24]
        prev_block.notes.append(f"合并续表 {cont_block.table_id}（物理第 {cont_block.page_start} 页，{len(added)} 行）")
        log.log_event("table.continued", table_id=prev_block.table_id, source=cont_block.table_id,
                      page_start=prev_block.page_start, page_end=prev_block.page_end, rows_added=len(added))
        span.set_output({"table_id": prev_block.table_id, "page_end": prev_block.page_end,
                         "rows": len(prev_block.rows), "rows_added": len(added)})
        return prev_block


def dedupe_repeated_header(
    prev_block: TableBlock,
    dup_block: TableBlock,
    *,
    logger: Any = None,
) -> TableBlock:
    """R8：丢弃重复表头，仅把数据行并入前块；``repeated_headers_removed`` 由调用方计数。"""
    log = _lazy_logger(logger, "table_parser")
    with log.enter("dedupe_repeated_header",
                   {"prev": prev_block.table_id, "dup": dup_block.table_id, "dup_rows": len(dup_block.rows)}) as span:
        added = [list(r) for r in dup_block.rows]
        prev_block.rows.extend(added)
        prev_block.row_pages.extend([dup_block.page_start] * len(added))
        prev_block.page_end = max(prev_block.page_end, dup_block.page_end)
        prev_block.absorbed_pages.append(dup_block.page_start)
        prev_block.markdown = "" if prev_block.degenerate else table_to_markdown(prev_block.header, prev_block.rows)
        prev_block.key_numbers = extract_numbers(prev_block.markdown + " " + " ".join(prev_block.header))[:24]
        prev_block.notes.append(f"重复表头已去重：{dup_block.table_id}（物理第 {dup_block.page_start} 页）")
        log.log_event("table.header_dedup", table_id=prev_block.table_id, page=dup_block.page_start,
                      removed_rows=1, rows_added=len(added))
        span.set_output({"table_id": prev_block.table_id, "rows": len(prev_block.rows), "removed_header_rows": 1})
        return prev_block


def build_table_blocks(
    raw_tables: Sequence[TableBlock],
    *,
    logger: Any = None,
) -> tuple[list[TableBlock], dict[str, int]]:
    """把逐页候选表块按 R7/R8/R9 合并为逻辑表块（跨页是常态）。

    合并条件（三重校验，避免把「首格为空的普通新表」误并）：
        ① 本页首表首行首格为空/``None``（``looks_like_continuation``）且被当作数据行；
        ② 上一页最后一个逻辑块仍然「打开」（``page_end == 当前页 - 1``）；
        ③ 列数与上一块一致（``flat_cols`` 相同），或上一块本身为退化表时跳过合并。
    """
    log = _lazy_logger(logger, "table_parser")
    stats = {"raw_tables": len(raw_tables), "logical_tables": 0, "continued_tables": 0,
             "repeated_headers_removed": 0, "degenerate_tables": 0, "max_run": 0, "merge_rejected": 0}
    if not raw_tables:
        return [], stats

    ordered = sorted(raw_tables, key=lambda b: (b.page_start, b.table_id))
    blocks: list[TableBlock] = []
    open_block: TableBlock | None = None
    prev_input_page: int | None = None

    for block in ordered:
        is_first_of_page = prev_input_page != block.page_start
        prev_input_page = block.page_start
        merged = False
        if is_first_of_page and open_block is not None and open_block.page_end == block.page_start - 1:
            if is_repeated_header(open_block, block.raw_first_row):
                dedupe_repeated_header(open_block, block, logger=log)
                stats["repeated_headers_removed"] += 1
                merged = True
            elif block.first_row_is_data and looks_like_continuation(block.raw_first_row):
                if not open_block.degenerate and open_block.flat_cols == block.flat_cols:
                    merge_continued_tables(open_block, block, logger=log)
                    stats["continued_tables"] += 1
                    merged = True
                else:
                    stats["merge_rejected"] += 1
                    open_block.notes.append(
                        f"疑似续表 {block.table_id}（物理第 {block.page_start} 页）列数不匹配"
                        f"（{open_block.flat_cols} vs {block.flat_cols}），未合并"
                    )
                    log.log_event("table.continue_rejected", level="WARNING", table_id=block.table_id,
                                  page=block.page_start, prev_cols=open_block.flat_cols, cols=block.flat_cols)
        if not merged:
            blocks.append(block)
            open_block = block
        if block.degenerate:
            stats["degenerate_tables"] += 1

    stats["logical_tables"] = len(blocks)
    stats["max_run"] = _max_table_run(ordered)
    log.log_event("table.blocks_done", **stats)
    return blocks, stats


def _max_run(blocks: Sequence[TableBlock]) -> int:
    """最长「连续含表页」长度（按逻辑块覆盖的物理页去重后统计）。"""
    pages = sorted({p for b in blocks for p in range(b.page_start, b.page_end + 1)})
    best = 0
    run = 0
    prev = None
    for page in pages:
        run = run + 1 if prev is not None and page == prev + 1 else 1
        best = max(best, run)
        prev = page
    return best


def _max_table_run(raw_tables: Sequence[TableBlock]) -> int:
    """按**原始候选**页统计 max_run（与全量扫描口径一致：含表即计入）。"""
    pages = sorted({b.page_start for b in raw_tables})
    best = run = 0
    prev = None
    for page in pages:
        run = run + 1 if prev is not None and page == prev + 1 else 1
        best = max(best, run)
        prev = page
    return best


def iter_table_dicts(blocks: Iterable[TableBlock]) -> Iterable[dict[str, Any]]:
    """便捷：逻辑表块 → dict 迭代（写 JSONL/SQLite 用）。"""
    for block in blocks:
        yield block.to_dict()
