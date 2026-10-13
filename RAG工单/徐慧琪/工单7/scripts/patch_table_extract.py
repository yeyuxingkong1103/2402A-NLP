# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 一次性补丁：修复 PyMuPDF 表格抽取在本语料上的**结构性错乱**。

问题（本工单实测，非推测）
--------------------------
上游 06 的 `pymupdf_parser._extract_tables` 用默认策略 `page.find_tables()`
（= strategy="lines"）抽取表格。该策略在**招股说明书**上工作正常，但在
**银行/保险/券商年报**上会把整张表压成 2~3 列：行标签全挤进第 1 个单元格、
数值全挤进第 2 个单元格。实测样例（中国人寿 2020 年报 p8「主要会计数据」）：

    默认 lines 策略 → 36 行 × 3 列
      | 主要会计数据 | 2020年 | 本报告期比\n2019年 上年同期增减 2018年 |
      | 全年业绩\n营业收入\n其中：已赚保费\n营业支出\n... |  | 745,165 10.7% 643,101\n560,... |

行与数值的对应关系完全丢失，**表格题必然取错数**。

根因：年报表格大量使用**合并单元格 + 无框线**排版，`lines` 策略按线条切格子，
合并区域被当成一个大单元格；而 `text` 策略按文字位置聚类，能还原真实网格。

修复
----
1. 抽取策略改为**先 text 后 lines**（text 抽不出再退回原默认，保证不劣化）；
2. 新增 `_clean_grid`：去空行/空列、折叠换行，并修复 text 策略的副产物——
   行标签首个汉字被单独切到第 0 列（如 "营|业收入" → "营业收入"）；
3. 抽出表内「单位：人民币百万元」这类**计量单位**并前置到 Markdown 首行。
   这一步对答案正确性至关重要：年报数字普遍以「百万元/千元」为单位，
   单位丢失会让 LLM 把千元当元、把百万元当亿元。

实测效果（同一张表）：修复后 40 行 × 7 列，
营业收入 2020年=824,961 / 2019年=745,165 / 同比=10.7% / 2018年=643,101，单位=人民币百万元，
与 PDF 原文一致。

本脚本幂等：检测到标记串则跳过。
"""
from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TARGET = os.path.join(ROOT, "src", "pymupdf_parser.py")

MARKER = "工单07 表格修复"

NEW_BLOCK = '''# ---------------------------------------------------------------------------
# 表格：转 Markdown
# ---------------------------------------------------------------------------
# 工单07 表格修复：新增单位识别 + 网格清洗。
# 年报以「百万元/千元」计量，单位若丢失会导致答案差 10^6 量级，故必须随表入库。
_UNIT_RE = re.compile(r"^\\s*单位\\s*[:：]\\s*(?P<u>.+?)\\s*$")


def _clean_grid(rows: list[list[Any]]) -> tuple[list[list[str]], str]:
    """规整二维单元格。返回 (grid, unit)。

    做四件事：
      1. 折叠换行与多余空白、补齐列数、删全空行；
      2. 抽出表内「单位：xxx」行作为 unit（该行不再作为数据行）；
      3. 删全空列；
      4. 合并"被拆开的单字首列"——text 策略常把行标签首字单独切成一列
         （"营|业收入"），当第 0 列**全部**为单字时判定为切分副产物并并回。
    """
    if not rows:
        return [], ""
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return [], ""

    norm: list[list[str]] = []
    for r in rows:
        cells = [re.sub(r"\\s+", " ", (c if c is not None else "")).strip() for c in r]
        cells += [""] * (width - len(cells))
        norm.append(cells)
    norm = [r for r in norm if any(r)]
    if not norm:
        return [], ""

    # ---- 抽单位 ----
    unit = ""
    kept: list[list[str]] = []
    for r in norm:
        joined = " ".join(c for c in r if c)
        m = _UNIT_RE.match(joined)
        if m and not unit:
            unit = m.group("u").strip()
            # 该行可能同时含表标题，保留非单位部分
            rest = [c for c in r if c and not _UNIT_RE.match(c)]
            if rest:
                kept.append(rest + [""] * (width - len(rest)))
            continue
        kept.append(r)
    norm = kept
    if not norm:
        return [], unit

    # ---- 删空列 ----
    keep_idx = [i for i in range(width) if any(i < len(r) and r[i] for r in norm)]
    if not keep_idx:
        return [], unit
    norm = [[r[i] if i < len(r) else "" for i in keep_idx] for r in norm]

    # ---- 合并被切开的单字首列 ----
    if len(norm[0]) >= 2:
        nonempty0 = [r[0] for r in norm if r[0]]
        if nonempty0 and all(len(c) <= 1 for c in nonempty0):
            merged: list[list[str]] = []
            for r in norm:
                merged.append([r[0] + r[1]] + r[2:] if r[0] else r[1:])
            norm = merged

    return norm, unit


def table_to_markdown(rows: list[list[Any]], unit: str = "") -> str:
    """二维单元格 → Markdown 表格；unit 非空时前置一行「单位：xxx」。"""
    grid, unit2 = _clean_grid(rows)
    if not grid:
        return ""
    unit = unit or unit2
    width = len(grid[0])
    lines: list[str] = []
    if unit:
        lines.append(f"单位：{unit}")
    header = grid[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for r in grid[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\\n".join(lines)


def _page_unit_near(page: "fitz.Page", bbox: tuple) -> str:
    """表格正下方 60pt 内查找「单位：xxx」注（部分年报表注在表格下方而非表内）。"""
    try:
        y0 = min(bbox[1] + 60, bbox[3] + 60)
        band = fitz.Rect(bbox[0] - 5, bbox[3] - 5, bbox[2] + 5, y0)
        text = page.get_text("text", clip=band) or ""
    except Exception:
        return ""
    for ln in text.splitlines():
        m = _UNIT_RE.match(ln.strip())
        if m:
            return m.group("u").strip()
    return ""


def _extract_tables(page: "fitz.Page", page_idx: int) -> list[dict]:
    """抽取单页表格，返回 [{bbox, markdown}]。

    工单07 表格修复：先试 strategy="text"（按文字位置聚类，能还原年报的
    合并单元格网格），抽不出再退回默认 strategy（等价于上游行为，不劣化）。
    """
    out: list[dict] = []
    tables = []
    for strat in ("text", None):
        try:
            finder = (page.find_tables(strategy=strat) if strat
                      else page.find_tables())
            tables = finder.tables
        except Exception:
            tables = []
        if tables:
            break
    if not tables:
        return out

    for tb in tables:
        try:
            rows = tb.extract()
        except Exception:
            continue
        unit = _page_unit_near(page, tuple(tb.bbox))
        md = table_to_markdown(rows, unit=unit)
        if not md:
            continue
        # 过滤"伪表格"：用清洗后的网格判定（原实现按未清洗的原始行判定，
        # 在 text 策略下会把大量空行算作有效行而放行伪表格）
        grid, _u = _clean_grid(rows)
        if len(grid) < 2 or len(grid[0]) < 2:
            continue
        if len(re.sub(r"[\\s|:-]", "", md)) < 12:
            continue
        out.append({"bbox": tuple(tb.bbox), "markdown": md, "page_idx": page_idx})
    return out


'''

START = "def table_to_markdown(rows: list[list[Any]]) -> str:"
END = "def _in_any_bbox("


def main() -> int:
    if not os.path.isfile(TARGET):
        raise SystemExit(f"目标文件不存在：{TARGET}")
    text = io.open(TARGET, encoding="utf-8").read()
    if MARKER in text:
        print(f"[跳过] 已含标记「{MARKER}」")
        return 0

    # 连同其上方的注释分隔块一起替换
    marker = "# ---------------------------------------------------------------------------\n# 表格：转 Markdown"
    start = text.find(marker)
    if start < 0:
        start = text.find(START)
    end = text.find(END)
    if start < 0 or end < 0 or end <= start:
        raise SystemExit("定位失败：pymupdf_parser.py 结构可能已变，请人工确认")

    text = text[:start] + NEW_BLOCK + text[end:]
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(text)
    print(f"[完成] 已修复 {TARGET} 的表格抽取（{MARKER}）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
