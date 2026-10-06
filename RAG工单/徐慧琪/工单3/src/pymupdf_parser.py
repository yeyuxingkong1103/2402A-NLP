# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：PDF 解析

【选型说明（工单02 实测，2026-10-03）】
本模块是**回退解析引擎**：主解析器为 MinerU 3.4.5（pipeline 后端，
经独立 venv `D:\model\mineru-venv` 调用，模型清单与本机 PDF-Extract-Kit
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
def table_to_markdown(rows: list[list[Any]]) -> str:
    """把二维单元格数组转为 Markdown 表格，保留合并造成的空单元格。"""
    if not rows:
        return ""
    # 统一列数
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return ""
    norm: list[list[str]] = []
    for r in rows:
        cells = [(c if c is not None else "").strip().replace("\n", " ") for c in r]
        cells += [""] * (width - len(cells))
        norm.append(cells)

    # 去掉全空行
    norm = [r for r in norm if any(c for c in r)]
    if not norm:
        return ""

    # 表头占位：首行若非空则作表头，否则补一行空表头（Markdown 语法需要）
    header = norm[0]
    body = norm[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _extract_tables(page: "fitz.Page", page_idx: int) -> list[dict]:
    """抽取单页表格，返回 [{bbox, markdown}]。"""
    out: list[dict] = []
    try:
        finder = page.find_tables()
        tables = finder.tables
    except Exception:
        tables = []

    for tb in tables:
        try:
            rows = tb.extract()
        except Exception:
            continue
        md = table_to_markdown(rows)
        if not md:
            continue
        # 过滤"伪表格"：只有 1 行或 1 列，或内容极短
        data_rows = [r for r in (rows or []) if any((c or "").strip() for c in r)]
        if len(data_rows) < 2:
            continue
        if len(re.sub(r"[\s|:-]", "", md)) < 12:
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
