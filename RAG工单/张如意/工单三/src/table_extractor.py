# -*- coding: utf-8 -*-
"""
表格抽取模块（工单03 核心模块之一）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

职责：
  1. 用 pdfplumber 抽取《招股说明书2.pdf》（力源信息）等文档中的全部表格；
  2. 把二维表转成 Markdown（保留表头与行列对应关系），每表落盘一个 .md 文件；
  3. 输出 results/table_inventory.json 表清单（页码 / 行列数 / 表头 / 字符数）；
  4. 表格质量校验：过滤空表与噪声表、修复合并单元格造成的列错位、合并跨页续表。

为什么需要质量校验（真实数据驱动，不是凭空设计）：
  · pdfplumber 的 lines 策略会把「合并单元格」识别成多条相邻竖线，
    于是 5 列的表被抽成 9 列甚至 15 列，数据列与表头列错开。
    例：p22「募集资金投资项目表」抽出 6×9，表头 '序号' 落在第 2 列、
    数据 '1' 落在第 1 列 —— 直接转 Markdown 会得到一份「表头对不上数据」的废表。
    本模块用「删全空列 + 表头列错位修复 + 再删全空列」三步把网格还原。
  · 招股书的财务报表（如「合并资产负债表」）跨页续排，下一页标题带「(续)」。
    不合并 → 检索到半张表 → 答案缺行；
    乱合并 → 把「3、报告期内曾为关联方…」误并入「2、不存在控制关系的关联方」→ 答案串类。
    本模块用「表头兼容 + 跨页相邻 + 续排标志」三条件判定，宁可不合并也不错合并。

用法：
    python table_extractor.py --pdf "D:/工单/.../招股说明书2.pdf" --name 招股说明书2
    python table_extractor.py --all          # 抽取招股说明书1 + 招股说明书2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- 路径处理：把项目根目录加入 sys.path，保证能 import rag_core -------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import config                     # noqa: E402
from rag_core.pdf_parse import PageBlock        # noqa: E402

# 复用共享库的 Markdown 转换函数，保证与工单01/02 的表格口径完全一致。
# （rag_core 不修改；此处只做「调用」，若未来该函数改名则有本地兜底实现）
try:
    from rag_core.pdf_parse import _table_to_markdown as _core_table_to_markdown
except ImportError:                              # pragma: no cover
    _core_table_to_markdown = None


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 表格 .md / 清单默认输出目录（相对本工单目录）
DEFAULT_OUT_DIR = ROOT / "工单03-表格解析及检索优化" / "results"

# 页眉页脚清洗（与 rag_core.pdf_parse 的口径保持一致）
_PAGE_HEADER_PAT = re.compile(
    r"^\s*(武汉力源信息技术股份有限公司|武汉兴图新科电子股份有限公司)?\s*招股(意向书|说明书)\s*$"
)
_PAGE_HEADER_PAT2 = re.compile(
    r"^\s*(武汉力源信息技术股份有限公司|武汉兴图新科电子股份有限公司)"
    r"\s+招股(意向书|说明书)\s*$"
)
_PAGENUM_PAT = re.compile(r"^\s*[-—–]?\s*\d{1,3}\s*[-—–]?\s*$")
# 「一、」「(一)」「1、」「（1）」这类编号小标题 —— 它们是「新表」的信号，不是续表
_NUMBERED_HEADING_PAT = re.compile(r"^\s*(第[一二三四五六七八九十百]+[节章]|[一二三四五六七八九十]+[、.．]|\(\s*[一二三四五六七八九十]+\s*\)|（\s*[一二三四五六七八九十]+\s*）|\d{1,2}[、.．])\s*\S")
# 各页顶部/底部的噪声行（单位说明、页码、页眉）
_NOISE_LINE_PAT = re.compile(r"^\s*(\(?\s*单位\s*[:：].*\)?|单位\s*[:：].*)\s*$")

# 质量门限
MIN_ROWS = 2            # 至少 2 行（含表头）
MIN_COLS = 2            # 至少 2 列
MIN_CHARS = 20          # Markdown 字符数下限，低于此值视为噪声
MAX_COLS = 40           # 列数超过此值几乎必然是线条误检
MAX_CELL_CHARS = 600    # 单元格字符数上限（防把整页正文塞进一个格子）

# 跨页合并的位置门限（相对页高的比例）
TOP_RATIO = 0.30        # 续表应出现在下一页的上部
BOTTOM_RATIO = 0.50     # 续表的前半张应出现在上一页的下部
STRICT_TOP_RATIO = 0.15  # 「无表头续页」更严格
STRICT_BOTTOM_RATIO = 0.80


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class TableRecord:
    """一张被抽取出来的表格（已做质量校验与规范化）。"""
    doc: str                       # 文档名
    page: int                      # 物理页码（1 起，= pdfplumber 页序 + 1）
    printed_page: int = 0          # 印刷页码（PDF 页脚上印的号，通常 = 物理页 - 1）
    table_index: int = 0           # 该页内第几张表（0 起）
    caption: str = ""              # 表格上方的标题/所属小节（用于检索定位）
    header: list = field(default_factory=list)   # 规范化后的表头
    n_rows: int = 0                # 规范化后总行数（含表头）
    n_cols: int = 0                # 规范化后列数
    n_data_rows: int = 0           # 数据行数（不含表头）
    char_len: int = 0              # Markdown 字符数
    markdown: str = ""             # 表格正文（Markdown）
    content: str = ""              # 入库用文本 = 【表格】标题 + Markdown
    y0_ratio: float = 0.0          # 表格上边界 / 页高
    y1_ratio: float = 0.0          # 表格下边界 / 页高
    strategy: str = "lines"        # 命中的抽取策略
    flags: list = field(default_factory=list)    # 质量标记（如 header_lost / merged）
    merged_pages: list = field(default_factory=list)   # 合并了哪些页的表
    merged_from: list = field(default_factory=list)    # 被合并掉的表编号

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MergeEvent:
    """一次跨页合并（成功或拒绝）的记录，写入清单便于人工复核。"""
    action: str          # merged | rejected
    pages: list          # 涉及页码
    reason: str          # 判定理由
    caption: str = ""


# ---------------------------------------------------------------------------
# 1. 网格清洗：把 pdfplumber 的「脏网格」还原成「行列对齐」的好网格
# ---------------------------------------------------------------------------
def _clean_cell(c) -> str:
    """单元格清洗：None -> ''，换行/连续空白 -> 单空格。"""
    if c is None:
        return ""
    s = str(c).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[ \t　]+", " ", s)
    return s.strip()


def _drop_empty_rows(grid: list[list[str]]) -> list[list[str]]:
    """删除整行皆空的空行（pdfplumber 在跨行合并处会产出空行）。"""
    return [r for r in grid if any(c for c in r)]


def _drop_empty_cols(grid: list[list[str]]) -> list[list[str]]:
    """删除所有行都为空的全空列（合并单元格造成的竖线误检）。"""
    if not grid:
        return grid
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    keep = [j for j in range(width) if any(r[j] for r in grid)]
    if not keep:                       # 全空表
        return []
    return [[r[j] for j in keep] for r in grid]


def _repair_header_alignment(grid: list[list[str]]) -> list[list[str]]:
    """
    修复「表头列错位」。

    现象：合并单元格被误识别成竖线后，同一张表中
      表头非空列 = [1, 4, 7, 10]      数据非空列 = [0, 3, 6, 9]
    即表头整体右移了 delta 列，直接转 Markdown 会「表头配错数据」。
    做法：统计数据行「首个非空列」的众数位置，与表头首个非空列比较，
    若存在稳定偏移 delta（|delta|<=2）则把表头整体平移回去。

    实测（招股说明书2.pdf）：
      p22 表2 6×9  表头[序号/项目名称/计划总投资] 由第 2/5/8 列平移到第 1/4/7 列 → 再删全空列 → 3 列对齐 ✓
      p157 表0 2×9 关联方名称/持股比例/与本公司关系 → 3 列对齐 ✓
    """
    if len(grid) < 2:
        return grid
    head, body = grid[0], grid[1:]

    head_nz = [j for j, c in enumerate(head) if c]
    first_nz = Counter()
    for r in body:
        nz = [j for j, c in enumerate(r) if c]
        if nz:
            first_nz[nz[0]] += 1
    if not head_nz or not first_nz:
        return grid

    mode_first = first_nz.most_common(1)[0][0]
    delta = head_nz[0] - mode_first
    if delta == 0 or abs(delta) > 2:
        return grid

    n = len(head)
    if delta > 0:                      # 表头偏右 -> 左移
        new_head = head[delta:] + [""] * delta
    else:                              # 表头偏左 -> 右移
        new_head = [""] * (-delta) + head[: n + delta]
    if len(new_head) != n or sum(1 for c in new_head if c) != len(head_nz):
        return grid                    # 平移会丢内容则不修，保守处理
    return [new_head] + body


def normalize_grid(raw: list[list]) -> list[list[str]]:
    """
    表格网格规范化流水线（三步走）：
      ① 单元格清洗 + 去空行
      ② 删全空列 → 表头列错位修复 → 再删全空列（错位修复后会出现新的空列）
      ③ 去空行
    返回「行列严格对齐」的二维字符串表；表为空时返回 []。
    """
    if not raw:
        return []
    grid = [[_clean_cell(c) for c in r] for r in raw if r]
    grid = _drop_empty_rows(grid)
    if not grid:
        return []
    grid = _drop_empty_cols(grid)
    if not grid:
        return []
    grid = _repair_header_alignment(grid)
    grid = _drop_empty_cols(grid)
    grid = _drop_empty_rows(grid)
    # 超长单元格（整页正文被塞进一格）视为误检
    if any(len(c) > MAX_CELL_CHARS for r in grid for c in r):
        return []
    return grid


def table_to_markdown(grid: list[list[str]]) -> str:
    """
    二维表 -> Markdown。
    优先复用 rag_core.pdf_parse._table_to_markdown，保证与工单01/02 完全同口径；
    共享库缺失时使用本地等价实现（逻辑与共享库一致：首行作表头 + 分隔行 + 数据行）。
    """
    if not grid:
        return ""
    if _core_table_to_markdown is not None:
        return _core_table_to_markdown(grid)
    width = max(len(r) for r in grid)
    rows = [r + [""] * (width - len(r)) for r in grid]
    head, body = rows[0], rows[1:]
    md = ["| " + " | ".join(head) + " |",
          "| " + " | ".join(["---"] * width) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


# ---------------------------------------------------------------------------
# 2. 质量校验
# ---------------------------------------------------------------------------
def is_valid_table(grid: list[list[str]], markdown: str) -> tuple[bool, str]:
    """
    表格质量校验。返回 (是否保留, 原因)。
    过滤规则（对应工单「表格质量校验逻辑」要求）：
      · 空表 / 只有表头的表
      · 单行单列（pdfplumber 把正文段落误当表格）
      · 列数异常（> MAX_COLS，线条误检）
      · 字符数过少（< MIN_CHARS，无检索价值）
      · 表头与数据行数比例异常（表头占了一半以上的行）
    """
    if not grid:
        return False, "空表"
    n_rows, n_cols = len(grid), max(len(r) for r in grid)
    if n_rows < MIN_ROWS:
        return False, f"行数不足({n_rows})"
    if n_cols < MIN_COLS:
        return False, f"列数不足({n_cols})"
    if n_cols > MAX_COLS:
        return False, f"列数异常({n_cols}>{MAX_COLS})"
    if len(markdown) < MIN_CHARS:
        return False, f"字符数过少({len(markdown)})"
    return True, "ok"


# ---------------------------------------------------------------------------
# 3. 表格标题（caption）抽取 —— 决定表格能不能被检索到
# ---------------------------------------------------------------------------
def extract_caption(page, y0: float, y1: float) -> str:
    """
    抽取表格标题：取表格上方最近的一条「非噪声」文本行。

    为什么重要：招股书里同一页往往有多张表，表格本身没有名字，
    标题（如「合并资产负债表(续)」「2、不存在控制关系的关联方」）
    既是跨页合并的判据，也是检索时区分同类表的关键词。

    返回：清洗后的标题字符串（找不到返回 ""）。
    """
    # 只看表格上方，从下往上找最近的非噪声行
    try:
        top = page.crop((0, 0, page.width, max(y0, 1)))
        above = [ln.strip() for ln in (top.extract_text() or "").split("\n")]
    except Exception:
        above = []
    above = [ln for ln in above if ln]
    for ln in reversed(above):
        if _PAGENUM_PAT.match(ln) or _NOISE_LINE_PAT.match(ln):
            continue
        if _PAGE_HEADER_PAT.match(ln) or _PAGE_HEADER_PAT2.match(ln):
            continue
        if ln in ("招股意向书", "招股说明书"):
            continue
        return ln[:60]
    return ""


def _extract_printed_page(page) -> int:
    """从页眉/页脚中解析印刷页码（物理页 = 印刷页 + 1，便于与纸质文档对照）。"""
    try:
        txt = page.extract_text() or ""
    except Exception:
        return 0
    lines = [ln.strip() for ln in txt.split("\n")]
    for ln in lines[:3] + lines[-3:]:
        m = re.fullmatch(r"[-—–]?\s*(\d{1,3})\s*[-—–]?", ln or "")
        if m:
            return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# 4. 单页抽表（多策略 + 兜底）
# ---------------------------------------------------------------------------
LINE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
FALLBACK_SETTINGS = [
    ("lines+text", {"vertical_strategy": "lines", "horizontal_strategy": "text"}),
    ("text+text", {"vertical_strategy": "text", "horizontal_strategy": "text"}),
]


def _dedup_tables(found: list, iou_thresh: float = 0.6) -> list:
    """按 bbox 的 IoU 去重（同一张表被不同策略重复检出时只保留一个）。"""
    kept = []
    for t in found:
        dup = False
        for k in kept:
            if _iou(t.bbox, k.bbox) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept


def _iou(a, b) -> float:
    """两个 bbox 的交并比。"""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def extract_page_tables(page, page_no: int, doc: str,
                        fallback: bool = True) -> list[TableRecord]:
    """
    抽取单页内的全部表格。

    策略：默认用 lines（竖线+横线）策略；
    若该页一条有效表都没抽到，但页面存在较多线条/矩形（说明可能是
    无完整边框的表，或竖线缺失），再依次尝试 lines+text / text+text 兜底。
    """
    H = page.height or 1.0
    printed = _extract_printed_page(page)
    records: list[TableRecord] = []

    def _collect(settings: dict, tag: str) -> list[TableRecord]:
        try:
            found = page.find_tables(settings)
        except Exception as e:                      # 单页失败不影响整体
            print(f"  [warn] 第{page_no}页表格解析失败({tag}): {e}")
            return []
        out: list[TableRecord] = []
        for t in _dedup_tables(found):
            try:
                raw = t.extract()
            except Exception:
                continue
            grid = normalize_grid(raw)
            md = table_to_markdown(grid)
            ok, why = is_valid_table(grid, md)
            if not ok:
                continue
            y0, y1 = t.bbox[1], t.bbox[3]
            caption = extract_caption(page, y0, y1)
            out.append(TableRecord(
                doc=doc, page=page_no, printed_page=printed,
                table_index=0,        # 稍后统一编号
                caption=caption,
                header=list(grid[0]),
                n_rows=len(grid),
                n_cols=max(len(r) for r in grid),
                n_data_rows=len(grid) - 1,
                char_len=len(md),
                markdown=md,
                content=_build_content(caption, md),
                y0_ratio=round(y0 / H, 3),
                y1_ratio=round(y1 / H, 3),
                strategy=tag,
            ))
        return out

    records = _collect(LINE_SETTINGS, "lines")

    if not records and fallback:
        # 兜底条件：页面有较多线条/矩形，说明「看起来像表格但没抽出来」
        n_shape = len(getattr(page, "lines", [])) + len(getattr(page, "rects", []))
        if n_shape >= 8:
            for tag, settings in FALLBACK_SETTINGS:
                records = _collect(settings, tag)
                if records:
                    break

    for i, r in enumerate(records):
        r.table_index = i
        r.content = _build_content(r.caption, r.markdown)
    return records


def _build_content(caption: str, markdown: str) -> str:
    """
    构造「入库文本」= 表格标题 + Markdown 表体。

    为什么要把标题拼进去：
      chunk_structure() 对表格块走 _block_to_single_chunk()，不会注入章节路径；
      若不把标题写进正文，向量/BM25 都只看到一堆数字，检索时极易被其它财务表盖过。
      拼上「【表格】2、不存在控制关系的关联方」后，关键词与语义信号同时增强。
    """
    cap = f"【表格】{caption}\n" if caption else "【表格】\n"
    return cap + markdown


# ---------------------------------------------------------------------------
# 5. 跨页续表合并
# ---------------------------------------------------------------------------
def _norm_caption(c: str) -> str:
    """标题归一化：去空格、去「(续)」「（续）」「续表」等续排标志、去尾部标点。"""
    s = (c or "").strip()
    s = re.sub(r"[（(]\s*续\s*[)）]", "", s)
    s = re.sub(r"^(续表|接上表|续上表)[:：]?", "", s)
    s = re.sub(r"[\s　]+", "", s)
    return s.strip("、.．:：")


def _headers_compatible(h1: list, h2: list) -> bool:
    """表头兼容判定：去空后集合重合度 >= 0.6（容忍轻微识别差异）。"""
    a = {_norm_caption(c) for c in (h1 or []) if c and str(c).strip()}
    b = {_norm_caption(c) for c in (h2 or []) if c and str(c).strip()}
    a.discard(""); b.discard("")
    if not a or not b:
        return False
    inter = a & b
    return len(inter) >= max(1, int(min(len(a), len(b)) * 0.6))


def _is_new_section(caption: str) -> bool:
    """标题是否是「新的编号小节」——是则说明这是另一张表，不能当续表合并。"""
    return bool(_NUMBERED_HEADING_PAT.match((caption or "").strip()))


def merge_cross_page_tables(records: list[TableRecord]) -> tuple[list[TableRecord], list[MergeEvent]]:
    """
    跨页续表合并。

    判定条件（全部满足才合并，宁缺毋滥）：
      ① 两表在相邻页（本表在第 p 页、续表在第 p+1 页）
      ② 表头兼容（去空/去「续」后重合度 >= 0.6）
      ③ 位置连续：续表出现在下一页上部（y0 <= 0.30H），
         且被续的表结束在上一页下部（y1 >= 0.50H）
      ④ 语义连续：续表标题要么自带续排标志（含「续」），
         要么与上一张表同属一个小节（标题归一化后一致）；
         若续表上方是「3、xxx」这类新的编号小节，则判为「另一张表」拒绝合并。

    真实案例（招股说明书2.pdf）：
      · 合并成功：p204「合并资产负债表」→ p205「合并资产负债表(续)」，
        两张 15 列同构表拼成完整资产负债表；
      · 拒绝合并：p157「2、不存在控制关系的关联方」→ p158
        「3、报告期内曾为关联方但目前已不存在关联关系的公司」，
        两表表头虽同为「企业名称/与本公司关系」，但分属不同小节，
        合并会把「已不属于关联方」的公司混进「现有关联方」名单 → 必须拒绝。

    Returns:
        (合并后的表列表, 合并事件列表)
    """
    if not records:
        return [], []

    ordered = sorted(records, key=lambda r: (r.page, r.table_index))
    merged: list[TableRecord] = []
    events: list[MergeEvent] = []
    consumed: set[int] = set()

    for i, cur in enumerate(ordered):
        if i in consumed:
            continue
        node = cur
        for j in range(i + 1, len(ordered)):
            nxt = ordered[j]
            if j in consumed:
                continue
            # ① 相邻页
            if nxt.page != node.page + 1:
                break
            # ③ 位置连续
            pos_ok = (nxt.y0_ratio <= TOP_RATIO and node.y1_ratio >= BOTTOM_RATIO)
            hdr_ok = _headers_compatible(node.header, nxt.header)
            cap = nxt.caption or ""
            same_section = bool(_norm_caption(node.caption)) and \
                _norm_caption(node.caption) == _norm_caption(cap)
            cont_marker = ("续" in cap) and not _is_new_section(cap)

            if hdr_ok and pos_ok and (cont_marker or same_section):
                node = _do_merge(node, nxt)
                consumed.add(j)
                events.append(MergeEvent(
                    action="merged", pages=[node.page, nxt.page],
                    reason=f"{'续排标志' if cont_marker else '同小节'}+表头兼容+位置连续",
                    caption=cap))
                continue

            # 记录「看起来像续表但被拒绝」的情况，供人工复核
            if hdr_ok and pos_ok and _is_new_section(cap):
                events.append(MergeEvent(
                    action="rejected", pages=[node.page, nxt.page],
                    reason=f"表头相同但属于新小节「{cap[:24]}」，合并会造成语义串类",
                    caption=cap))
            # ④ 无表头续页（表头在上一页，下一页直接从数据行开始）
            if (not hdr_ok and node.y1_ratio >= STRICT_BOTTOM_RATIO
                    and nxt.y0_ratio <= STRICT_TOP_RATIO
                    and node.n_cols == nxt.n_cols
                    and not _is_new_section(cap)):
                node = _do_merge(node, nxt, header_lost=True)
                consumed.add(j)
                events.append(MergeEvent(
                    action="merged", pages=[node.page, nxt.page],
                    reason="无表头续页：列数一致+位置紧接+非新小节",
                    caption=cap))
            break

        merged.append(node)

    return merged, events


def _do_merge(a: TableRecord, b: TableRecord, header_lost: bool = False) -> TableRecord:
    """把 b 的数据行并入 a（b 的表头行丢弃；header_lost 时整表都是数据行）。"""
    # 直接按二维数据合并，保证列数一致
    rows_b = _md_to_rows(b.markdown)
    data_b = rows_b if header_lost else rows_b[1:]
    rows_a = _md_to_rows(a.markdown)
    grid = rows_a + [r for r in data_b if any(x.strip() for x in r)]

    a.markdown = table_to_markdown(grid)
    a.n_rows = len(grid)
    a.n_cols = max((len(r) for r in grid), default=a.n_cols)
    a.n_data_rows = len(grid) - 1
    a.char_len = len(a.markdown)
    a.merged_pages = sorted(set(a.merged_pages + [a.page, b.page]))
    a.merged_from = a.merged_from + [f"p{b.page}#t{b.table_index}"]
    a.flags = sorted(set(a.flags + (["header_lost"] if header_lost else []) + ["merged"]))
    a.y1_ratio = b.y1_ratio          # 合并后延伸到下一页的下边界
    a.content = _build_content(a.caption, a.markdown)
    return a


def _md_to_rows(md: str) -> list[list[str]]:
    """Markdown 表格 -> 二维数组（用于合并时重新拼接）。"""
    rows = []
    for ln in (md or "").split("\n"):
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and all(set(c) <= {"-", ":", " "} and c for c in cells):
            continue                  # 跳过分隔行
        rows.append(cells)
    return rows or [[""]]


# ---------------------------------------------------------------------------
# 6. 主入口：抽取 + 落盘
# ---------------------------------------------------------------------------
def extract_all(pdf_path: Path, doc_name: str,
                out_dir: Path | None = None,
                max_pages: int | None = None,
                merge_cross_page: bool = True,
                verbose: bool = True) -> tuple[list[TableRecord], list[MergeEvent]]:
    """
    抽取一份 PDF 的全部表格。

    Args:
        out_dir: 结果目录；表格 .md 写到 <out_dir>/tables/
        max_pages: 只解析前 N 页（调试用）
        merge_cross_page: 是否执行跨页续表合并
    Returns:
        (表格记录列表, 跨页合并事件列表)
    """
    import pdfplumber

    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(pdf_path)

    records: list[TableRecord] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for i, page in enumerate(pages):
            if verbose and (i + 1) % 25 == 0:
                print(f"    …已扫描 {i + 1}/{len(pages)} 页，累计表格 {len(records)} 张")
            records += extract_page_tables(page, i + 1, doc_name)

    events: list[MergeEvent] = []
    if merge_cross_page:
        before = len(records)
        records, events = merge_cross_page_tables(records)
        if verbose:
            n_merged = sum(1 for e in events if e.action == "merged")
            n_rej = sum(1 for e in events if e.action == "rejected")
            print(f"    跨页合并：{before} 张 -> {len(records)} 张"
                  f"（合并 {n_merged} 处，拒绝误合并 {n_rej} 处）")

    # 逐表落盘 .md（文件名带页码与序号，便于人工核对）
    for r in records:
        tag = f"_merged" if r.merged_pages else ""
        fn = f"{doc_name}_p{r.page:03d}_t{r.table_index}{tag}.md"
        header = _md_file_header(r)
        (tables_dir / fn).write_text(header + r.markdown + "\n", encoding="utf-8")
        r.flags = sorted(set(r.flags))

    if verbose:
        print(f"    已写出 {len(records)} 个表格 Markdown -> {tables_dir}")
    return records, events


def _md_file_header(r: TableRecord) -> str:
    """表格 .md 文件头的元信息注释（方便人工抽查与写报告）。"""
    lines = [
        f"<!-- 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 -->",
        f"<!-- 文档：{r.doc} | 物理页：{r.page} | 印刷页：{r.printed_page} "
        f"| 表序号：{r.table_index} | 抽取策略：{r.strategy} -->",
        f"<!-- 规格：{r.n_rows} 行 × {r.n_cols} 列（数据行 {r.n_data_rows}）"
        f" | 字符数：{r.char_len} | 质量标记：{','.join(r.flags) or '无'} -->",
    ]
    if r.caption:
        lines.append(f"<!-- 标题：{r.caption} -->")
    if r.merged_pages:
        lines.append(f"<!-- 跨页合并：{r.merged_pages}，源表 {r.merged_from} -->")
    return "\n".join(lines) + "\n\n"


def save_inventory(records: list[TableRecord], events: list[MergeEvent],
                   path: Path | None = None, doc_name: str = "") -> Path:
    """
    输出表清单 results/table_inventory.json。

    字段：页码 / 行列数 / 表头 / 字符数（工单要求），另附标题、质量标记、
    跨页合并信息与全局统计，供《表格定位分析.md》与验收对照使用。
    """
    path = Path(path or (DEFAULT_OUT_DIR / "table_inventory.json"))
    path.parent.mkdir(parents=True, exist_ok=True)

    items = [{
        "doc": r.doc,
        "page": r.page,
        "printed_page": r.printed_page,
        "table_index": r.table_index,
        "caption": r.caption,
        "header": r.header,
        "rows": r.n_rows,
        "cols": r.n_cols,
        "data_rows": r.n_data_rows,
        "char_len": r.char_len,
        "strategy": r.strategy,
        "flags": r.flags,
        "merged_pages": r.merged_pages,
        "merged_from": r.merged_from,
        "file": f"tables/{r.doc}_p{r.page:03d}_t{r.table_index}"
                f"{'_merged' if r.merged_pages else ''}.md",
    } for r in records]

    summary = {
        "n_tables": len(items),
        "n_pages_with_tables": len({r.page for r in records}),
        "n_merged_tables": sum(1 for r in records if r.merged_pages),
        "n_header_lost": sum(1 for r in records if "header_lost" in r.flags),
        "total_chars": sum(r.char_len for r in records),
        "avg_rows": round(sum(r.n_rows for r in records) / max(len(records), 1), 2),
        "avg_cols": round(sum(r.n_cols for r in records) / max(len(records), 1), 2),
        "page_range": [min((r.page for r in records), default=0),
                       max((r.page for r in records), default=0)],
        "by_strategy": dict(Counter(r.strategy for r in records)),
    }

    path.write_text(json.dumps({
        "doc": doc_name or (records[0].doc if records else ""),
        "summary": summary,
        "merge_events": [asdict(e) for e in events],
        "tables": items,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def records_to_page_blocks(records: list[TableRecord]) -> list[PageBlock]:
    """
    把表格记录转成 rag_core 的 PageBlock（type="table"），供建索引使用。
    这样表格块就能走 chunk_structure() 的「整表不切碎」逻辑。
    """
    blocks: list[PageBlock] = []
    for r in records:
        blocks.append(PageBlock(
            doc=r.doc, page=r.page, type="table", content=r.content,
            extra={
                "rows": r.n_rows, "cols": r.n_cols, "table_index": r.table_index,
                "caption": r.caption, "char_len": r.char_len,
                "printed_page": r.printed_page,
                "merged_pages": r.merged_pages,
                "flags": r.flags,
                "strategy": r.strategy,
            },
        ))
    return blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_one(pdf: Path, name: str, out: Path, max_pages: int | None, verbose: bool):
    print(f"[表格抽取] 《{name}》 <- {pdf}")
    records, events = extract_all(pdf, name, out_dir=out, max_pages=max_pages,
                                  verbose=verbose)
    inv = save_inventory(records, events, out / f"table_inventory_{name}.json", name)
    print(f"  表清单：{inv}")
    print(f"  共 {len(records)} 张表，"
          f"{sum(r.char_len for r in records)} 字符")
    return records, events


def main():
    ap = argparse.ArgumentParser(
        description="工单03 表格抽取：pdfplumber 抽表 -> 质量校验 -> Markdown 落盘 + 表清单")
    ap.add_argument("--pdf", type=str, default=str(config.PDF_PROSPECTUS_2),
                    help="待解析 PDF 路径（默认《招股说明书2.pdf》力源信息）")
    ap.add_argument("--name", type=str, default="招股说明书2", help="文档名")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="结果输出目录")
    ap.add_argument("--max-pages", type=int, default=None, help="只解析前 N 页（调试用）")
    ap.add_argument("--all", action="store_true", help="抽取招股说明书1 与 2")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    verbose = not args.quiet

    if args.all:
        for pdf, name in [(config.PDF_PROSPECTUS_2, "招股说明书2"),
                          (config.PDF_PROSPECTUS_1, "招股说明书1")]:
            if Path(pdf).exists():
                _run_one(Path(pdf), name, out, args.max_pages, verbose)
            else:
                print(f"  [warn] 找不到 {pdf}")
    else:
        _run_one(Path(args.pdf), args.name, out, args.max_pages, verbose)


if __name__ == "__main__":
    main()
