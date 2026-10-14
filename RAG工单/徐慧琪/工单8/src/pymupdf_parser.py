# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-Query理解优化
         （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
模块：PDF 解析

【选型说明（工单02 实测，2026-10-03）】
本模块是**回退解析引擎**：主解析器为 MinerU 3.4.5（pipeline 后端，
经独立 venv `D:\\model\\mineru-venv` 调用，模型清单与本机 PDF-Extract-Kit
快照完全匹配，实测可用）。仅当 MinerU 失败（依赖/显存/超时）时，
`src/pdf_parser.py::parse_pdf` 才会回退到本模块，保证建库链路不被单点阻塞
（工单"容错机制"要求）。

本模块能力（移植自工单01，含五遍扫描）：
    PyMuPDF (fitz)  —— 正文抽取 + 字号/加粗 → 编号与字号双证据的标题识别
    find_tables     —— 表格提取（转 Markdown）
    页眉/页脚清洗、折行标题合并、目录页识别

MinerU 版本选择记录（均已实测排除）：2.7.6 pipeline 缺 YOLO/MFD 权重；
2.7.6 vlm 单页推理 >1h；4.0.10 本地解析需 GGUF 量化模型。
详见 docs/02-优化方案.md 第 2.1 节。

输出（对齐 MinerU 产物格式，便于将来无缝切换回 MinerU）：
    data/parsed/<stem>.md                 人可读 Markdown
    data/parsed/<stem>_content_list.json  结构化块列表（下游分块直接消费）
      每条: {"type": "text"|"table", "text"/"table_body": str,
             "page_idx": int, "heading_path": [str], "source": str}
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any

import fitz  # PyMuPDF

from src import config

# ---------------------------------------------------------------------------
# 标题识别
# ---------------------------------------------------------------------------
# 中文招股说明书的编号层级（从粗到细）
HEADING_PATTERNS: list[tuple[int, re.Pattern]] = [
    (1, re.compile(r"^第[一二三四五六七八九十百]+[节章]\s*")),          # 第一节 / 第一章
    (2, re.compile(r"^[一二三四五六七八九十]+\s*[、.．]\s*")),           # 一、
    (3, re.compile(r"^[（(]\s*[一二三四五六七八九十]+\s*[)）]\s*")),     # （一）
    # 1、 1． 1. —— 注意排除小数：`2.69%` 不能被当成 "2." 编号
    (4, re.compile(r"^\d{1,2}\s*[、．]\s*\S|^\d{1,2}\s*\.(?!\d)\s*\S")),
    (5, re.compile(r"^[（(]\s*\d{1,2}\s*[)）]\s*")),                    # （1）
]

# 表格标题、图表说明等
CAPTION_RE = re.compile(r"^(表|图|附表)\s*\d+[-－—]?\d*")

# 目录行：大量连续点号 + 结尾页码
TOC_LINE_RE = re.compile(r"[.．·]{4,}\s*\d+\s*$")
# 一页里出现这么多目录行，就认定该页是"目录页"，整页不做标题识别
TOC_PAGE_MIN_LINES = 6


def detect_heading_level(text: str, size: float, body_size: float,
                         is_bold: bool) -> tuple[int, bool]:
    """综合编号规律与字号判定标题层级。

    返回 (层级, 是否由编号规则命中)。层级 0 表示不是标题。
    第二项用于后续抑制"续行误判"——纯靠字号判定的标题更容易是段落续行。
    """
    t = text.strip()
    if not t or len(t) > 100:
        return 0, False

    # 0) 句中含句号 => 是正文句子，不是标题。
    #    例如「1、立项时间远早于中标时间或合同签订时间。为增加订单获取机会，公司」
    #    虽然以 "1、" 开头，但它是正文段落，不是编号标题。
    if "。" in t[:-1]:
        return 0, False

    # 1) 先看编号规律（最可靠）
    regex_level = 0
    for level, pat in HEADING_PATTERNS:
        if pat.match(t):
            regex_level = level
            break

    # 2) 再看视觉特征：明显大于正文，或加粗且不太长
    visual_level = 0
    if size >= body_size + 3:
        visual_level = 1
    elif size >= body_size + 1.2:
        visual_level = 2
    elif is_bold and len(t) <= 40:
        visual_level = 3

    if regex_level:
        # 编号对但字号正常 —— 多为正文里引用的"（一）xxx"句子，要求够短才算标题
        if len(t) > 45:
            return 0, True
        return regex_level, True
    return visual_level, False


# ---------------------------------------------------------------------------
# 表格：转 Markdown
# ---------------------------------------------------------------------------
# 工单07 表格修复：新增单位识别 + 网格清洗。
# 年报以「百万元/千元」计量，单位若丢失会导致答案差 10^6 量级，故必须随表入库。
_UNIT_RE = re.compile(r"^\s*单位\s*[:：]\s*(?P<u>.+?)\s*$")


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
        cells = [re.sub(r"\s+", " ", (c if c is not None else "")).strip() for c in r]
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

    # ---- 合并行内被切散的单字格 ----
    # text 策略按文字 x 坐标聚类，当行标签首字与其余部分字距略大时，会把首字
    # 单独切成一个单元格。实测「主要会计数据」→ ["主", "要会计数据"]、
    # 「营业收入」→ ["营", "业收入"]，导致**精确短语检索整词失配**
    # （`主要会计数据` 检索不到该表块）。这里把"单汉字格 + 紧邻汉字开头格"并回。
    # 安全性：年报表格里不会存在合法的"单汉字单元格紧挨汉字单元格"语义，
    # 且仅当**右侧格以汉字开头**时才合并，不会吞掉数值列。
    for r in norm:
        i = 0
        while i < len(r) - 1:
            if len(r[i]) == 1 and "一" <= r[i] <= "鿿" and r[i + 1][:1] >= "一" \
                    and r[i + 1][:1] <= "鿿":
                r[i] = r[i] + r[i + 1]
                del r[i + 1]
            else:
                i += 1

    # ---- 合并被切散的大额数字 ----
    # 工单07 修复（第二轮）：千分位大数会被切成两格，形如
    #     "1" | ",494.07亿元"      →  应为 "1,494.07亿元"
    #     "1" | ",329.55亿元"      →  应为 "1,329.55亿元"
    # 影响：表块里的金额被读成 "1" 与 ",494.07亿元" 两截，
    # LLM 取数时容易只拿到整数部分或直接错位（实测 中国平安 2019 年报 p20）。
    # 判据刻意取得很窄——**仅当左格是 1~3 位纯数字、右格以「,三位数字」开头**时合并，
    # 这样"2020 | 2019"这类相邻年份列不会被误并（右格不以逗号开头）。
    for r in norm:
        i = 0
        while i < len(r) - 1:
            if re.fullmatch(r"\d{1,3}", r[i]) and re.match(r",\d{3}", r[i + 1]):
                r[i] = r[i] + r[i + 1]
                del r[i + 1]
            else:
                i += 1

    return norm, unit


# ---------------------------------------------------------------------------
# 伪表格判定（双栏正文误判）
# ---------------------------------------------------------------------------
# text 策略按文字位置聚类，会把**双栏排版的正文页**聚成"网格"——左右两栏的句子
# 被交替切进同一行，形成形如下面的假表格（实测 中国平安 2019 年报 p9/p20）：
#
#     | 生态反哺客户经营。我们积极布局“金融服务、医疗健康、汽 | 经营管理，深度转型、实现“四化”。 |
#     | 车服务、房产服务、智慧城市”五大生态圈，通过生态延展获 | 运营数据化、渠道建设精细化、… |
#
# 危害：①`block_type` 被标成 table，引用来源失真；②金额被切碎（见上）；
#      ③假表块参与检索排序，挤占真实表块。
#
# 判据（**两个条件同时成立**才判为伪表格，避免误伤真的无数字表，如董监高名单）：
#   a) 没有任何一列的数值占比 ≥ 0.10 —— 真实财务表必有数字列；
#   b) 长句碎片占比 ≥ 0.30 —— 疑为正文（>20 字且含中文句读的单元格）。
# 命中后该网格**不作为表格返回**，其文字会自动回流为普通正文块
# （见 `parse_page`：表格 bbox 只用于**排除**正文，不返回即不排除），
# 因此**内容不丢失**，只是不再冒充表格。
_NUM_CELL_RE = re.compile(
    r"^[\s\-–—()（）]*[-+]?\d[\d,\.]*\s*"
    r"(%|亿元|万元|千元|百万元|元|股|个|家|人|万|亿|百分点|注\d*)?[\s\d注()（）]*$")
_PROSE_CHAR_RE = re.compile(r"[，。；：、]")


def _max_numeric_column_ratio(grid: list[list[str]]) -> float:
    """返回「数值占比最高的那一列」的数值占比（0 表示无数字列）。"""
    if not grid:
        return 0.0
    width = max(len(r) for r in grid)
    best = 0.0
    for j in range(width):
        col = [r[j] for r in grid if j < len(r) and r[j]]
        if len(col) < 3:          # 列太短不足以判定
            continue
        best = max(best, sum(1 for x in col if _NUM_CELL_RE.match(x)) / len(col))
    return best


def _prose_cell_ratio(grid: list[list[str]]) -> float:
    """长句碎片占比：>20 字且含中文句读的单元格比例。"""
    cells = [c for r in grid for c in r if c]
    if not cells:
        return 0.0
    n = sum(1 for c in cells if len(c) > 20 and _PROSE_CHAR_RE.search(c))
    return n / len(cells)


def _is_prose_grid(grid: list[list[str]]) -> bool:
    """双栏正文被误判成的"表格"？两个条件同时成立才为真。"""
    return (_max_numeric_column_ratio(grid) < 0.10
            and _prose_cell_ratio(grid) >= 0.30)


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
    return "\n".join(lines)


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
        if len(re.sub(r"[\s|:-]", "", md)) < 12:
            continue
        # 工单07 修复（第二轮）：双栏正文被 text 策略误判成的"网格"不作为表格返回。
        # 不返回 → 该 bbox 不进 table_bboxes → parse_page 不会排除其文字，
        # 正文自动回流为普通 text 块，**内容不丢失**，只是不再冒充表格。
        if _is_prose_grid(grid):
            continue
        out.append({"bbox": tuple(tb.bbox), "markdown": md, "page_idx": page_idx})
    return out


def _in_any_bbox(rect: "fitz.Rect", bboxes: list[tuple]) -> bool:
    """判断文本块是否落在任一表格区域内（用于去重，避免表格文字重复进入正文）。"""
    for b in bboxes:
        try:
            other = fitz.Rect(b)
            inter = rect & other
            if inter.is_valid and rect.get_area() > 0:
                if inter.get_area() / rect.get_area() > 0.5:
                    return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# 单页解析
# ---------------------------------------------------------------------------
def parse_page(page: "fitz.Page", page_idx: int) -> tuple[list[dict], float, int]:
    """解析单页，返回 (块列表, 本页最大字号, 本页字符数)。"""
    blocks_out: list[dict] = []
    raw = page.get_text("dict")
    max_size = 0.0
    n_chars = 0

    # 先抽表格，拿到占位 bbox
    tables = _extract_tables(page, page_idx)
    table_bboxes = [t["bbox"] for t in tables]

    # 收集正文行
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:  # 0=文本, 1=图片
            continue
        for line in blk.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            size = max((s.get("size", 0) for s in spans), default=0)
            # flag bit 4 (16) = bold
            is_bold = any(bool(s.get("flags", 0) & 16) for s in spans)
            max_size = max(max_size, size)
            n_chars += len(text)

            try:
                rect = fitz.Rect(line.get("bbox"))
            except Exception:
                rect = None
            if rect is not None and table_bboxes and _in_any_bbox(rect, table_bboxes):
                continue  # 表格内的文字已由表格块承载

            blocks_out.append({
                "text": text, "size": size, "bold": is_bold,
                "y": line.get("bbox", [0, 0, 0, 0])[1],
                "page_idx": page_idx,
            })

    # 合并同段的连续行：这里保持逐行输出，交给下游 chunker 做段落聚合
    for t in tables:
        blocks_out.append({
            "text": t["markdown"], "size": 0.0, "bold": False,
            "y": t["bbox"][1], "page_idx": page_idx, "is_table": True,
        })

    blocks_out.sort(key=lambda b: b["y"])
    return blocks_out, max_size, n_chars


# ---------------------------------------------------------------------------
# 整篇解析
# ---------------------------------------------------------------------------
# 句末标点：某行以此结尾，说明它是完整句子，下一行大概率不是它的续行
TERMINAL_PUNCT = "。：；！？!?;:）)"


def strip_repeated_lines(lines: list[dict], total_pages: int,
                         max_ratio: float = 0.15) -> tuple[list[dict], list[str]]:
    """剔除页眉/页脚。

    招股书每页顶部有「武汉兴图新科电子股份有限公司  招股意向书」、
    底部有「1-1-60」这类页码，逐页重复。它们会被塞进每个 chunk，
    既稀释语义又浪费向量维度。

    判定：把行文本中的数字抹掉后做归一化统计，
    若某个模式出现在超过 `max_ratio` 比例的页面上，即视为页眉/页脚。
    """
    if total_pages <= 0:
        return lines, []

    def norm(t: str) -> str:
        # 抹掉页内变化的数字（页码 1-1-60、年份等），只留骨架
        return re.sub(r"\d+", "#", t).strip()

    page_sets: dict[str, set[int]] = {}
    for b in lines:
        if b.get("is_table"):
            continue
        key = norm(b["text"])
        if len(key) < 4:          # 太短的不参与统计，避免误伤正文
            continue
        page_sets.setdefault(key, set()).add(b["page_idx"])

    threshold = max(3, int(total_pages * max_ratio))
    repeated = {k for k, pages in page_sets.items() if len(pages) >= threshold}

    kept = [b for b in lines
            if b.get("is_table") or norm(b["text"]) not in repeated]
    return kept, sorted(repeated)


def merge_heading_continuations(lines: list[dict], body_size: float) -> list[dict]:
    """把"折行的标题"合并回同一行。

    招股书里长标题常被排版折成两三行，例如：
        「（六）本次募集资金投资项目实施完成后公司折旧费用和摊销费用大幅增」
        「加风险」
    检测时首行会被认成标题、次行变成孤立正文。这里做一次前瞻合并，
    再对合并后的完整文本重新判定层级。
    """
    out: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        cur = dict(lines[i])
        if cur.get("heading_level"):
            while i + 1 < n:
                nxt = lines[i + 1]
                if nxt.get("is_table") or nxt["page_idx"] != cur["page_idx"]:
                    break
                if nxt.get("heading_level"):
                    break  # 下一个本身就是标题，说明当前这条已完整
                txt = cur["text"].rstrip()
                if txt and txt[-1] in TERMINAL_PUNCT:
                    break  # 当前行已句末收尾，不是被折行截断
                nxt_txt = nxt["text"].strip()
                # 只有在"当前行确实像被折断了"时才合并：
                #   - 当前行够长(>=25)，排版上很可能没排完；或
                #   - 续行极短(<=8)，是 "加风险" 这类碎片
                # 否则会误吞下一段正文，例如把「第一节 释义」和
                # 「在本招股意向书中，除非文义另有所指…」粘成一行。
                looks_truncated = len(txt) >= 25 or len(nxt_txt) <= 8
                if not looks_truncated or len(nxt_txt) > 20:
                    break
                cur["text"] = txt + nxt_txt
                i += 1
            cur["heading_level"], cur["by_regex"] = detect_heading_level(
                cur["text"], cur["size"], body_size, cur["bold"])
        out.append(cur)
        i += 1
    return out


def parse_pdf(pdf_path: str, limit_pages: int | None = None,
              progress: bool = True) -> tuple[list[dict], str]:
    """解析 PDF，返回 (content_list, markdown 文本)。"""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    total = doc.page_count
    if limit_pages:
        total = min(total, limit_pages)

    # ---- 第一遍：统计正文字号（按字符数加权取众数）----
    size_counter: Counter = Counter()
    for i in range(total):
        raw = doc[i].get_text("dict")
        for blk in raw.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                for s in line.get("spans", []):
                    txt = s.get("text", "").strip()
                    if txt:
                        size_counter[round(s.get("size", 0), 1)] += len(txt)
    body_size = size_counter.most_common(1)[0][0] if size_counter else 10.5

    # ---- 第二遍：逐页收集行 + 表格 ----
    lines: list[dict] = []
    scan_pages: list[int] = []  # 无可提取文字的页（潜在扫描件）
    for i in range(total):
        page = doc[i]
        blocks, _mx, n_chars = parse_page(page, i)
        if n_chars < 20:
            scan_pages.append(i)
        # 目录页识别：该页若充斥"……. 269"式条目，整页不做标题识别，
        # 否则目录条目会被当成真标题压进标题栈，污染后续正文的 heading_path。
        toc_lines = sum(1 for b in blocks
                        if not b.get("is_table") and TOC_LINE_RE.search(b["text"]))
        is_toc_page = toc_lines >= TOC_PAGE_MIN_LINES

        for b in blocks:
            if b.get("is_table") or is_toc_page:
                b["heading_level"], b["by_regex"] = 0, False
                if is_toc_page:
                    b["is_toc"] = True
            else:
                b["heading_level"], b["by_regex"] = detect_heading_level(
                    b["text"], b["size"], body_size, b["bold"])
            lines.append(b)
        if progress and (i + 1) % 50 == 0:
            print(f"  解析进度 {i + 1}/{total} 页", flush=True)

    doc.close()

    # ---- 第三遍：剔除页眉/页脚 ----
    lines, repeated = strip_repeated_lines(lines, total)
    if repeated:
        print(f"  [清洗] 剔除页眉/页脚模式 {len(repeated)} 条，例如: "
              f"{repeated[0][:40]!r}")

    # ---- 第四遍：合并折行标题 ----
    lines = merge_heading_continuations(lines, body_size)

    # ---- 第五遍：维护标题栈并输出 ----
    content: list[dict] = []
    heading_stack: list[tuple[int, str]] = []
    prev_text = ""

    for b in lines:
        text = b["text"]

        if b.get("is_table"):
            content.append({
                "type": "table", "table_body": text, "page_idx": b["page_idx"],
                "heading_path": [h[1] for h in heading_stack],
                "source": config.SOURCE_NAME,
            })
            prev_text = text
            continue

        level = b.get("heading_level", 0)
        # 续行抑制：上一行没有句末标点，且本行只是"靠字号"判出来的标题
        # —— 那它多半是上一段的折行续写，不是标题。
        if level and not b.get("by_regex") and prev_text and prev_text[-1] not in TERMINAL_PUNCT:
            level = 0

        # 图表标题不参与层级栈
        if level and CAPTION_RE.match(text):
            level = 0

        if level:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            content.append({
                "type": "text", "text": text, "page_idx": b["page_idx"],
                "heading_path": [h[1] for h in heading_stack[:-1]],
                "source": config.SOURCE_NAME, "is_heading": True,
            })
        else:
            content.append({
                "type": "text", "text": text, "page_idx": b["page_idx"],
                "heading_path": [h[1] for h in heading_stack],
                "source": config.SOURCE_NAME,
            })
        prev_text = text

    if scan_pages:
        print(f"[提示] 检测到 {len(scan_pages)} 页缺少文本层，前若干页: {scan_pages[:10]}")
        print("       如需 OCR，可启用 PaddleOCR-VL 兜底（src/ocr_fallback.py）")

    markdown = content_to_markdown(content)
    return content, markdown


def content_to_markdown(content: list[dict]) -> str:
    """把 content_list 渲染成带标题层级的 Markdown。"""
    lines: list[str] = []
    for item in content:
        if item["type"] == "table":
            lines.append("")
            caption = " > ".join(item.get("heading_path") or [])
            if caption:
                lines.append(f"<!-- {caption} (p.{item['page_idx'] + 1}) -->")
            lines.append(item["table_body"])
            lines.append("")
            continue

        if item.get("is_heading"):
            level = len(item.get("heading_path") or []) + 1
            lines.append("")
            lines.append("#" * min(level, 6) + " " + item["text"])
            lines.append("")
        else:
            lines.append(item["text"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="解析 PDF -> Markdown + content_list.json")
    ap.add_argument("pdf", nargs="?", default=config.SOURCE_PDF)
    ap.add_argument("--limit", type=int, default=None, help="只解析前 N 页（调试用）")
    ap.add_argument("--out", default=config.PARSED_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pdf))[0]

    print(f"解析中: {args.pdf}")
    content, markdown = parse_pdf(args.pdf, limit_pages=args.limit)

    md_path = os.path.join(args.out, f"{stem}.md")
    cl_path = os.path.join(args.out, f"{stem}_content_list.json")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    with open(cl_path, "w", encoding="utf-8") as fh:
        json.dump(content, fh, ensure_ascii=False, indent=1)

    n_text = sum(1 for c in content if c["type"] == "text")
    n_table = sum(1 for c in content if c["type"] == "table")
    n_head = sum(1 for c in content if c.get("is_heading"))
    print(f"\n完成：共 {len(content)} 块（正文 {n_text} / 表格 {n_table}，其中标题 {n_head}）")
    print(f"  Markdown      -> {md_path}")
    print(f"  content_list  -> {cl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
