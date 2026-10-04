# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：PDF 解析（MinerU VLM 主解析 + 质量检测 + PyMuPDF 回退）

本机实测结论（2026-10-03，详见 docs/02-优化方案.md 2.1 节）：
  - **MinerU 3.4.5（pipeline 后端）实测可用**：其模型清单（PP-DocLayoutV2 /
    unimernet_hf_small_2503 / pp_formulanet_plus_m / paddleocr_torch /
    SlanetPlus / UnetStructure / PP-LCNet_x1_0_table_cls）与本机
    PDF-Extract-Kit-1.0 快照**完全匹配**，版面分析约 1.1s/页。
    经独立 venv（`D:\\model\\mineru-venv`，transformers 4.57）调用。
  - 版本排除记录：2.7.6 pipeline 缺 YOLO/MFD 权重；2.7.6 vlm 单页推理 >1h；
    4.0.10 本地解析需 GGUF 模型（均不可行，原因见文档）。
  - 解析失败（依赖/显存/超时）自动回退 PyMuPDF 增强引擎
    （src/pymupdf_parser.py），保证建库链路不被单点阻塞（工单"容错机制"要求）。

输出归一化条目结构：
    {"type": "text|table|equation|image", "text": str, "page_idx": int,
     "text_level": int|None, "caption": list[str], "img_path": str|None}
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入（离线环境 + 栈修复）

import json
import os
import re
import subprocess
import sys
import time

from src import config


def _mineru_env() -> dict:
    """子进程环境：强制本地模型源 + 离线，禁止一切下载路径。"""
    env = dict(os.environ)
    env["MINERU_MODEL_SOURCE"] = "local"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run_mineru(pdf_path: str, out_dir: str, start: int | None = None,
               end: int | None = None, timeout: int | None = None) -> dict:
    """调用 MinerU（独立 venv，pipeline 后端）。不抛异常，返回结构化结果。

    独立 venv 的原因：MinerU 3.4.5 依赖 transformers 4.x 旧 API，
    主环境是 5.15（API 已被移除）——venv 用 --system-site-packages 复用
    主环境 CUDA torch，仅隔离 transformers 及其直接依赖。

    返回：{"ok", "md_path", "content_list_path", "elapsed_s", "error"}
    """
    timeout = timeout or config.MINERU_TIMEOUT_S
    cmd = [config.MINERU_PYTHON, config.MINERU_RUN_SCRIPT,
           "-p", pdf_path, "-o", out_dir, "-b", config.MINERU_BACKEND]
    if start is not None:
        cmd += ["-s", str(start)]
    if end is not None:
        cmd += ["-e", str(end)]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=_mineru_env(), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, cwd=config.ROOT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "md_path": None, "content_list_path": None,
                "elapsed_s": time.time() - t0, "error": f"timeout>{timeout}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "md_path": None, "content_list_path": None,
                "elapsed_s": time.time() - t0, "error": repr(exc)}

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = cl_path = None
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            if name == f"{stem}.md":
                md_path = os.path.join(root, name)
            elif name == f"{stem}_content_list.json":
                cl_path = os.path.join(root, name)
    ok = proc.returncode == 0 and cl_path is not None
    tail = (proc.stderr or proc.stdout or "")[-3000:]
    return {"ok": ok, "md_path": md_path, "content_list_path": cl_path,
            "elapsed_s": time.time() - t0, "error": None if ok else tail}


def _html_table_to_markdown(html: str) -> str:
    """MinerU 的 table_body 是 HTML；转为紧凑的 Markdown 表格。

    好处：① 字符数约减少 60%（HTML 标签开销大，实测 3000 字上限被标签占满）；
    ② LLM 对 Markdown 表格的理解优于 HTML。
    合并单元格（rowspan/colspan）不做网格还原，仅保留单元格文本。
    解析失败时原样返回。
    """
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html or "", re.S | re.I)
    grid: list[list[str]] = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)
        cleaned = [re.sub(r"<[^>]+>", "", c).replace("|", "\\|").strip() for c in cells]
        if cleaned:
            grid.append(cleaned)
    if not grid:
        return html
    width = max(len(r) for r in grid)
    lines: list[str] = []
    for i, r in enumerate(grid):
        padded = r + [""] * (width - len(r))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


def _join_caption(item: dict) -> str:
    caps = list(item.get("image_caption") or []) + list(item.get("table_caption") or [])
    return " ".join(str(c) for c in caps if c).strip()


def normalize_item(item: dict) -> dict | None:
    """MinerU 原始条目 → 内部结构；不可解析返回 None。"""
    t = item.get("type")
    page = int(item.get("page_idx") or 0)
    if t == "text":
        text = (item.get("text") or "").strip()
        if not text:
            return None
        return {"type": "text", "text": text, "page_idx": page,
                "text_level": item.get("text_level"), "caption": []}
    if t == "table":
        body = (item.get("table_body") or "").strip()
        if body.lstrip().lower().startswith("<table"):
            body = _html_table_to_markdown(body)     # 紧凑化（见该函数说明）
        cap = _join_caption(item)
        text = (cap + "\n" + body).strip() if cap else body
        if not text:
            return None
        return {"type": "table", "text": text, "page_idx": page, "text_level": None,
                "caption": [cap] if cap else [], "img_path": item.get("img_path")}
    if t == "equation":
        text = (item.get("text") or "").strip()
        if not text:
            return None
        return {"type": "equation", "text": text, "page_idx": page,
                "text_level": None, "caption": []}
    if t in ("image", "chart"):     # MinerU 3.x 把图表单独标为 chart
        cap = _join_caption(item)
        return {"type": "image", "text": cap or "[图片]", "page_idx": page,
                "text_level": None, "caption": [cap] if cap else [],
                "img_path": item.get("img_path")}
    # header / page_number / footer / page_footnote 是版面噪声，直接丢弃
    return None


def load_content_list(path: str) -> list[dict]:
    """读取 MinerU content_list.json 并归一化。"""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [n for n in (normalize_item(it) for it in raw) if n is not None]


def quality_report(items: list[dict], page_count: int) -> dict:
    """解析质量报告：低文本页（< OCR_MIN_CHARS_PER_PAGE 字符）提示需要 OCR 兜底。"""
    per_page: dict[int, int] = {}
    for it in items:
        if it["type"] == "text":
            per_page[it["page_idx"]] = per_page.get(it["page_idx"], 0) + len(it["text"])
    return {
        "pages_with_text": sum(1 for p in range(page_count) if per_page.get(p, 0) > 0),
        "low_text_pages": [p for p in range(page_count)
                           if per_page.get(p, 0) < config.OCR_MIN_CHARS_PER_PAGE],
        "n_tables": sum(1 for i in items if i["type"] == "table"),
        "n_equations": sum(1 for i in items if i["type"] == "equation"),
        "n_images": sum(1 for i in items if i["type"] == "image"),
    }


def page_count(pdf_path: str) -> int:
    """PDF 页数（PyMuPDF 只读打开，不加载内容）。"""
    import fitz

    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n


def _parse_with_pymupdf(pdf_path: str) -> list[dict]:
    """回退/主用解析：PyMuPDF 增强引擎（编号+字号标题识别 / 表格 / 页眉页脚清洗）。

    见 src/pymupdf_parser.py（移植自工单01，含五遍扫描与标题栈）。
    输出转换为本模块的归一化条目：标题条目的 text_level = len(heading_path)+1。
    """
    from src import pymupdf_parser

    content, _md = pymupdf_parser.parse_pdf(pdf_path, progress=True)
    items: list[dict] = []
    for c in content:
        if c.get("type") == "table":
            items.append({"type": "table", "text": (c.get("table_body") or "").strip(),
                          "page_idx": int(c.get("page_idx", 0)), "text_level": None,
                          "caption": []})
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        level = None
        if c.get("is_heading"):
            level = len(c.get("heading_path") or []) + 1
        items.append({"type": "text", "text": text,
                      "page_idx": int(c.get("page_idx", 0)),
                      "text_level": level, "caption": []})
    return items


def _with_gap_fill(items: list[dict], pdf_path: str) -> list[dict]:
    """统一入口：MinerU 产物 → 混合补齐（两个返回点共用，缓存路径也生效）。"""
    items, gap = merge_pymupdf_gap(items, pdf_path)
    if gap["filled"]:
        print(f"[pdf_parser] 混合补齐 {gap['filled']} 页"
              f"（稀疏页示例：{gap['pages'][:8]}）", flush=True)
    return items


def merge_pymupdf_gap(items: list[dict], pdf_path: str,
                      min_chars: int = 120) -> tuple[list[dict], dict]:
    """混合解析补齐：MinerU 产物中"文本稀疏页"用 PyMuPDF 补正文。

    实测背景（2026-10-04 全项目测试）：MinerU 3.4.5 有 113 个表格块
    `table_body` 为空（PP-DocLayoutV2 把双栏正文/释义页误判为表格，
    TableRec 解析失败且未回退 OCR；content_list_v2 同样为空）。
    这些页在 PyMuPDF 下可正常提取 → 用 PyMuPDF 补齐，保证内容不丢。

    只补"MinerU 文本量 < min_chars"的页，避免与 MinerU 结果重复。
    """
    per_page: dict[int, int] = {}
    for it in items:
        if it.get("type") == "text":
            per_page[it["page_idx"]] = per_page.get(it["page_idx"], 0) + len(it["text"])

    import fitz
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    gap_pages = [p for p in range(n_pages) if per_page.get(p, 0) < min_chars]

    added_items: list[dict] = []
    for p in gap_pages:
        text = doc[p].get_text("text") or ""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        core = [ln for ln in lines
                if ln not in ("武汉兴图新科电子股份有限公司", "招股意向书")
                and not re.fullmatch(r"1-1-\d+", ln)]
        if core:
            added_items.append({
                "type": "text", "text": "\n".join(core), "page_idx": p,
                "text_level": None, "caption": [], "from_gap_fill": True,
            })
    doc.close()

    stat = {"gap_pages": len(gap_pages), "filled": len(added_items),
            "pages": [p + 1 for p in gap_pages[:20]]}
    return items + added_items, stat


def parse_pdf(pdf_path: str, out_dir: str, reuse: bool = True,
              force_fallback: bool = False) -> dict:
    """解析入口：缓存复用 → MinerU VLM → PyMuPDF 回退。

    返回：{"source", "parser": "mineru|pymupdf", "items", "md_path", "page_count"}
    """
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cached_cl = os.path.join(out_dir, stem, "auto", f"{stem}_content_list.json")
    n = page_count(pdf_path)

    if reuse and not force_fallback and os.path.isfile(cached_cl):
        items = _with_gap_fill(load_content_list(cached_cl), pdf_path)
        return {"source": os.path.basename(pdf_path), "parser": "mineru",
                "items": items,
                "md_path": os.path.join(os.path.dirname(cached_cl), f"{stem}.md"),
                "page_count": n}

    if not force_fallback and config.MINERU_ENABLED:
        res = run_mineru(pdf_path, out_dir)
        if res["ok"]:
            items = _with_gap_fill(load_content_list(res["content_list_path"]), pdf_path)
            return {"source": os.path.basename(pdf_path), "parser": "mineru",
                    "items": items, "md_path": res["md_path"], "page_count": n}
        print(f"[pdf_parser] MinerU 失败：{(res['error'] or '')[-300:]}", flush=True)

    print("[pdf_parser] 回退 PyMuPDF 解析", flush=True)
    return {"source": os.path.basename(pdf_path), "parser": "pymupdf",
            "items": _parse_with_pymupdf(pdf_path), "md_path": None,
            "page_count": n}
