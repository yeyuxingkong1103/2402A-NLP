# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：表格解析与结构化（工单3 核心新增）

工单要求「表格作为一等公民，不能把表格当成普通文本」。本模块负责表格链路的
第一段——把 MinerU 输出的 `table_body`（HTML）还原成**结构化二维表格**：

    HTML ──► 单元格矩阵（rowspan/colspan 展开）──► 表头 + 数据行 ──► 质量分
                                                                   │
                                              质量不达标 ──► PaddleOCR-VL 兜底

为什么要自己展开 rowspan/colspan：
  招股说明书的表格大量使用合并单元格（如「项目 | 2018年度 | 2019年度」两级表头，
  或首列跨行的分类列）。不展开就直接按 `<tr>` 取 `<td>`，会导致**每一行的列数不一致**，
  行级描述里的「列名：值」对不上，检索与生成都会错位。

为什么要有质量分：
  实测（工单2 记录）MinerU 对招股1 有 113 个表格块 `table_body` 为空——版面模型把
  双栏正文误判为表格、或 TableRec 失败。这类"假表格"直接入库只会污染检索，
  必须识别出来走兜底或降级。

输出结构（TableStruct）：
    caption / header(列名) / rows(数据行) / n_cols / quality / n_cells / multi_header
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import html as _html
import os
import re
from dataclasses import dataclass, field
from typing import Sequence

from src import config

# --- HTML 解析用正则（MinerU 的 HTML 结构简单且规整，无需引入 bs4）-------------
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S | re.I)
_ROWSPAN_RE = re.compile(r"rowspan\s*=\s*[\"']?(\d+)", re.I)
_COLSPAN_RE = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[\s　]+")


@dataclass
class TableStruct:
    """结构化表格。

    两种形态（mode）：
      · "header"：常规列式表——首行是列名，其余是数据行。
                  如「项目 | 2019年7-9月 | 2018年7-9月 | 同比变动」。
      · "kv"    ：字段-值表——**没有列名**，每行是「字段 → 值」。
                  招股书里极常见，例如"本次发行概况"：
                      发行股数      | 1,670万股
                      发行后总股本  | 6,670万股
                  以及"释义"表（X | 指 | 释义）。
                  这类表若硬把首行当列名，列名会变成"发行股票类型/人民币普通股(A股)"，
                  后续行级描述全部错位——工单3 实测发现并修正。
    """

    caption: str = ""
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    n_cols: int = 0
    quality: float = 0.0
    n_cells: int = 0
    multi_header: bool = False
    mode: str = "header"               # "header" | "kv"
    fields: list[str] = field(default_factory=list)   # kv 模式的字段名列表
    value_cols: list[int] = field(default_factory=list)  # kv 模式的值列索引
    source_html: str = ""
    ocr_text: str = ""                 # 兜底 OCR 得到的纯文本（结构化失败时使用）
    from_ocr: bool = False             # 该表是否由 OCR 兜底产出

    @property
    def is_usable(self) -> bool:
        """是否可用于行级结构化（有列名或字段、且至少一行数据）。"""
        return bool(self.rows) and self.n_cols >= 2 and bool(
            self.header or self.fields)


# ---------------------------------------------------------------------------
# HTML → 单元格矩阵
# ---------------------------------------------------------------------------
def _clean_cell(raw: str) -> str:
    """单元格 HTML → 纯文本（去标签、反转义、压缩空白）。"""
    text = _TAG_RE.sub(" ", raw or "")
    text = _html.unescape(text)
    # MinerU 偶发把换行转义成字面 \n
    text = text.replace("\\n", " ")
    return _WS_RE.sub(" ", text).strip()


def html_to_grid(html_text: str) -> tuple[list[list[str]], int]:
    """HTML 表格 → (单元格矩阵, 展开的单元格总数)，rowspan/colspan 已展开。

    合并单元格的处理：把值**填充到其覆盖的每一个格子**。这样每一行的列数一致，
    「列名: 值」才能对齐；代价是同一值可能出现多次（对检索无害，反而提高召回）。
    """
    rows_raw = _TR_RE.findall(html_text or "")
    if not rows_raw:
        return [], 0

    # 1) 逐行解析出 (值, rowspan, colspan)，记录最大列数
    parsed: list[list[tuple[str, int, int]]] = []
    for row_html in rows_raw:
        cells: list[tuple[str, int, int]] = []
        for attrs, body in _CELL_RE.findall(row_html):
            rs = int(_ROWSPAN_RE.search(attrs).group(1)) if _ROWSPAN_RE.search(attrs) else 1
            cs = int(_COLSPAN_RE.search(attrs).group(1)) if _COLSPAN_RE.search(attrs) else 1
            cells.append((_clean_cell(body), max(rs, 1), max(cs, 1)))
        if cells:
            parsed.append(cells)
    if not parsed:
        return [], 0

    # 2) 先按 col 累计展开后各行的列数，确定总列数
    max_cols = 0
    for cells in parsed:
        max_cols = max(max_cols, sum(cs for _v, _rs, cs in cells))
    if max_cols <= 0:
        return [], 0

    # 3) 逐格填充（rowspan 向下跨行）
    grid: list[list[str]] = [["" for _ in range(max_cols)] for _ in range(len(parsed))]
    n_cells = 0
    for r, cells in enumerate(parsed):
        c = 0
        for value, rs, cs in cells:
            while c < max_cols and grid[r][c] != "":
                c += 1                       # 被上方 rowspan 占用的列，跳过
            for dr in range(rs):
                rr = r + dr
                if rr >= len(grid):
                    break
                for dc in range(cs):
                    cc = c + dc
                    if cc >= max_cols:
                        break
                    if grid[rr][cc] == "":
                        grid[rr][cc] = value
                    n_cells += 1
            c += cs
    return grid, n_cells


# ---------------------------------------------------------------------------
# 表头识别
# ---------------------------------------------------------------------------
# 纯数值单元格：1,918.21 / -226.29 / 206.85% / 12
# 注意**不能用**宽松的 [\d,.\-%—]+，否则 "2010-6-30"、"2009-12-31" 这类日期会被
# 当成数值，导致「项目 | 2010-6-30 | 2009-12-31 | …」这样的日期表头行被误判为数据行，
# 整张表的列名退化成"列1/列2/列3"（工单3 实测招股2 多张财务表踩到，已修正）。
_NUMERIC_CELL_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?%?$")


def _looks_like_header(cells: Sequence[str]) -> bool:
    """判断一行是否像表头：单元格短、且不含纯数值。"""
    vals = [c for c in cells if c]
    if not vals:
        return False
    if any(len(v) > 30 for v in vals):
        return False                    # 表头单元格通常很短
    numeric = sum(1 for v in vals if _NUMERIC_CELL_RE.match(v.strip()))
    return numeric == 0


def split_header(grid: list[list[str]]) -> tuple[list[str], list[list[str]], bool]:
    """矩阵 → (表头, 数据行, 是否多级表头)。

    规则：
      - 首行像表头 → 若第二行也像表头（多级表头）则把前两行拼接为「父/子」列名，
        否则首行即表头；
      - 首行不像表头（整表无表头，如纯数据矩阵）→ 用「列1、列2…」占位，
        并把首行还原为数据行，避免丢数据。
    """
    if not grid:
        return [], [], False

    first = grid[0]
    if not _looks_like_header(first):
        header = [f"列{i + 1}" for i in range(len(first))]
        return header, grid, False

    if len(grid) >= 2 and _looks_like_header(grid[1]):
        parent, child = grid[0], grid[1]
        merged: list[str] = []
        for i in range(max(len(parent), len(child))):
            p = parent[i] if i < len(parent) else ""
            c = child[i] if i < len(child) else ""
            if p and c and p != c:
                merged.append(f"{p}-{c}")
            else:
                merged.append(p or c)
        return merged, grid[2:], True

    return list(first), grid[1:], False


# 「释义」表（术语 | 指 | 释义）中间的连接词列：整列取值几乎都是这些标记
_MARKER_CELLS = {"指", "为", "是", "：", ":", "－", "-", "—", "－", "/", "、"}
_MARKER_RATIO = 0.6


def _marker_columns(grid: list[list[str]]) -> list[int]:
    """找出"连接词列"（整列大多是 指/为/： 之类），用于识别释义型表格。"""
    if not grid:
        return []
    n_cols = max(len(r) for r in grid)
    out: list[int] = []
    for c in range(n_cols):
        vals = [r[c] for r in grid if c < len(r) and r[c].strip()]
        if len(vals) < 3:
            continue
        marker = sum(1 for v in vals if v.strip() in _MARKER_CELLS)
        if marker / len(vals) >= _MARKER_RATIO:
            out.append(c)
    return out


def detect_mode(grid: list[list[str]]) -> tuple[str, list[int]]:
    """判断表格形态：("kv", 值列索引) 或 ("header", [])。

    判据（按优先级）：
      1. 存在"连接词列"（释义表）→ kv，值为其余非键列；
      2. 仅 2 列 → kv（招股书 2 列表几乎都是「字段→值」；即便真是"年度|金额"，
         按 kv 解析也能保留标签，比把首行当列名安全得多）；
      3. 其余 → header（首行即列名）。
    """
    if not grid:
        return "header", []
    n_cols = max(len(r) for r in grid)
    markers = _marker_columns(grid)
    if markers and n_cols >= 3:
        value_cols = [c for c in range(n_cols) if c != 0 and c not in markers]
        return "kv", value_cols
    if n_cols == 2:
        return "kv", [1]
    return "header", []


# ---------------------------------------------------------------------------
# 质量评估
# ---------------------------------------------------------------------------
def score_table(grid: list[list[str]], n_cells: int) -> float:
    """表格质量分 ∈ [0,1]：解析完整性 / 单元格量 / 填充率 / 有效字符率。"""
    if not grid or n_cells <= 0:
        return 0.0

    all_cells = [c for row in grid for c in row]
    if not all_cells:
        return 0.0

    fill_ratio = sum(1 for c in all_cells if c) / len(all_cells)
    joined = "".join(all_cells)
    # 有效字符 = 非空白；空白率过高说明是误判的版面块
    alnum_ratio = len(joined) / max(len(all_cells) * 8, 1)
    size_score = min(1.0, n_cells / 6.0)
    alnum_score = min(1.0, alnum_ratio / config.TABLE_MIN_ALNUM_RATIO)

    quality = 0.25 * 1.0 + 0.20 * size_score + 0.25 * fill_ratio + 0.30 * alnum_score
    return round(min(max(quality, 0.0), 1.0), 4)


def is_low_quality(struct: TableStruct) -> bool:
    """是否低质量（应触发 OCR 兜底或降级）。"""
    if struct.n_cells < config.TABLE_MIN_CELLS:
        return True
    if not struct.is_usable:
        return True
    if struct.quality < config.TABLE_QUALITY_PASS:
        return True
    return False


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def parse_table_html(html_text: str, caption: str = "") -> TableStruct:
    """表格 HTML → TableStruct（永不抛异常；解析失败返回空结构）。"""
    struct = TableStruct(caption=(caption or "").strip(), source_html=html_text or "")
    try:
        grid, n_cells = html_to_grid(html_text or "")
    except Exception:  # noqa: BLE001 —— 容错：解析异常按空表处理
        return struct
    if not grid:
        return struct

    struct.n_cols = max((len(r) for r in grid), default=0)
    struct.n_cells = n_cells
    struct.quality = score_table(grid, n_cells)

    mode, value_cols = detect_mode(grid)
    struct.mode = mode

    if mode == "kv":
        struct.value_cols = value_cols
        struct.rows = [r for r in grid if any(c for c in r)]
        struct.fields = [r[0].strip() for r in struct.rows if r and r[0].strip()]
        # 表头留空：kv 表没有列名，元数据里用 fields 表达"表里有哪些字段"
        struct.header = []
    else:
        header, rows, multi = split_header(grid)
        struct.header = header
        struct.multi_header = multi
        struct.rows = [r for r in rows if any(c for c in r)]
        struct.fields = [h for h in header if h]

    # 保护：超长表截断（招股书财务报表偶有上千行）
    if len(struct.rows) > config.TABLE_MAX_ROWS_PER_TABLE:
        struct.rows = struct.rows[: config.TABLE_MAX_ROWS_PER_TABLE]
    return struct


def table_to_markdown(struct: TableStruct, max_chars: int | None = None) -> str:
    """TableStruct → Markdown 表格（喂 LLM / 作为父块文本）。

    kv 表渲染为「项目 | 内容」两列，让 LLM 明确看到"字段→值"的对应关系。
    """
    if not struct.header and not struct.rows:
        return struct.ocr_text or ""

    n_cols = struct.n_cols
    header = list(struct.header)
    if struct.mode == "kv":
        n_cols = 2
        header = ["项目", "内容"]

    def _row(cells: Sequence[str]) -> str:
        vals = list(cells) + [""] * (n_cols - len(cells))
        return "| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in vals) + " |"

    def _kv_cells(r: Sequence[str]) -> list[str]:
        key = (r[0] if r else "").strip()
        vals = [r[i].strip() for i in (struct.value_cols or [1])
                if i < len(r) and r[i].strip()]
        return [key, " ".join(vals)]

    lines: list[str] = []
    if struct.caption:
        lines.append(struct.caption)
    lines.append(_row(header))
    lines.append("|" + "---|" * max(n_cols, 1))
    for r in struct.rows:
        lines.append(_row(_kv_cells(r) if struct.mode == "kv" else r))

    text = "\n".join(lines)
    limit = max_chars or config.TABLE_MAX_CHARS
    if len(text) > limit:
        # 保留表头 + 前若干行 + 尾部，中间省略（避免长表挤爆上下文）
        head = "\n".join(lines[: max(6, len(lines) // 2)])
        tail = "\n".join(lines[-3:])
        text = (head[: limit - 260] +
                f"\n…（表格过长，中间省略 {len(lines) - len(lines[: max(6, len(lines) // 2)]) - 3} 行）\n" +
                tail)
    return text


def table_heading_text(struct: TableStruct, page_display: int) -> str:
    """表头块文本（供按列名/字段名检索）。

    · header 模式：「表：X | 表头：列A｜列B｜列C | 页码：N」
    · kv 模式    ：「表：X | 包含字段：字段1、字段2、…、字段N | 页码：N」
                   kv 表没有列名，但把全部字段名列出来，等价于一张"字段清单"，
                   问"发行股数"时 BM25/向量都能命中这张表。
    """
    name = struct.caption or "（无标题）"
    if struct.mode == "kv":
        fields = "、".join(struct.fields[:60])
        text = f"表：{name} | 包含字段：{fields} | 页码：{page_display}"
    else:
        cols = "｜".join(c for c in struct.header if c)
        text = f"表：{name} | 表头：{cols} | 页码：{page_display}"
    return text[: config.TABLE_HEADER_MAX_CHARS]


def row_description(struct: TableStruct, row: Sequence[str], row_index: int,
                    page_display: int) -> str:
    """行级自然语言描述 —— **检索的主单元**。

    · header 模式：「表：X | 列A：值 | 列B：值 | 页码：N」
    · kv 模式    ：「表：X | 字段：值 | 页码：N」
                    （第一列本身就是字段名，不再编造列名）
    """
    parts: list[str] = [f"表：{struct.caption or '（无标题）'}"]
    if struct.mode == "kv":
        key = (row[0] if row else "").strip()
        for i in (struct.value_cols or [1]):
            val = row[i].strip() if i < len(row) and row[i] else ""
            if val:
                # 字段名已经在 "字段：值" 里出现一次，不再单独重复（省字数、去噪声）
                parts.append(f"{key}：{val}" if key and key != val else val)
    else:
        for i, col in enumerate(struct.header):
            val = row[i] if i < len(row) else ""
            if not val:
                continue
            col = col or f"列{i + 1}"
            if col == val:                  # 合并单元格展开导致的自我重复，只留一份
                parts.append(val)
            else:
                parts.append(f"{col}：{val}")
    parts.append(f"页码：{page_display}")
    text = " | ".join(p for p in parts if p)
    if len(text) > config.TABLE_ROW_MAX_CHARS:
        text = text[: config.TABLE_ROW_MAX_CHARS - 1] + "…"
    return text


# ---------------------------------------------------------------------------
# 兜底：低质量表格 → PaddleOCR-VL
# ---------------------------------------------------------------------------
def _ocr_cache_path(pdf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(config.TABLE_DIR, f"{stem}_ocr_pages.json")


def ocr_fallback_tables(pdf_path: str, page_indices: Sequence[int],
                        out_dir: str) -> dict[int, str]:
    """低质量表格所在页 → PaddleOCR-VL 识别，返回 {page_idx: 文本}。

    只在 MinerU 表格解析失败/无表格结构时调用（工单要求"识别质量差的表格调用
    PaddleOCR-VL 做兜底"）。任何失败都返回空 dict，不影响主链路。

    结果落盘缓存到 `data/tables/<doc>_ocr_pages.json`：实测 9 页要跑 236s，
    而建库会因调参/改代码反复重跑——不缓存的话每次重建都白等 4 分钟。
    只对**本次缺失**的页发起 OCR，已有缓存的页直接复用。
    """
    if not config.TABLE_OCR_FALLBACK or not page_indices:
        return {}
    cache_file = _ocr_cache_path(pdf_path)
    cached: dict[str, str] = {}
    if os.path.isfile(cache_file):
        try:
            import json

            with open(cache_file, encoding="utf-8") as fh:
                cached = {str(k): str(v) for k, v in json.load(fh).items()}
        except Exception:  # noqa: BLE001 —— 缓存损坏时忽略，重新 OCR
            cached = {}

    todo = sorted({int(p) for p in page_indices if str(int(p)) not in cached})
    if not todo:
        print(f"[table_parser] 表格 OCR 兜底：{len(page_indices)} 页全部命中缓存",
              flush=True)
        return {int(k): v for k, v in cached.items()
                if int(k) in {int(p) for p in page_indices}}

    from src import ocr_fallback

    if not ocr_fallback.is_available():
        print("[table_parser] 低质量表格存在，但 PaddleOCR-VL venv 不可用 → 跳过兜底",
              flush=True)
        return {int(k): v for k, v in cached.items()}

    if len(todo) > config.OCR_MAX_PAGES:
        print(f"[table_parser] 待 OCR 页数 {len(todo)} 超过上限 "
              f"{config.OCR_MAX_PAGES}，仅处理前 {config.OCR_MAX_PAGES} 页", flush=True)
        todo = todo[: config.OCR_MAX_PAGES]

    rendered = ocr_fallback.render_pages(pdf_path, todo, out_dir)
    results = ocr_fallback.run_paddleocr_vl(rendered)
    fresh = {int(r["page_idx"]): r["text"] for r in results if r.get("text")}
    if fresh:
        cached.update({str(k): v for k, v in fresh.items()})
        try:
            import json

            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(cached, fh, ensure_ascii=False, indent=1)
            print(f"[table_parser] 表格 OCR 兜底结果已缓存：{cache_file}", flush=True)
        except Exception:  # noqa: BLE001 —— 缓存写失败不影响主链路
            pass
    return {int(k): v for k, v in cached.items()}


# ---------------------------------------------------------------------------
# 中间产物落盘 / 读取（供报告与调试）
# ---------------------------------------------------------------------------
_TABLE_FIELDS = ("caption", "header", "rows", "n_cols", "quality", "n_cells",
                 "multi_header", "mode", "fields", "from_ocr")


def to_json_dict(struct: TableStruct) -> dict:
    """TableStruct → 可序列化 dict（不含 HTML，避免产物过大）。"""
    return {k: getattr(struct, k) for k in _TABLE_FIELDS}


def save_tables(records: list[dict], path: str) -> None:
    """表格结构化中间产物落盘：data/tables/<doc>_tables.json。"""
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)


def load_tables(path: str) -> list[dict]:
    import json

    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
