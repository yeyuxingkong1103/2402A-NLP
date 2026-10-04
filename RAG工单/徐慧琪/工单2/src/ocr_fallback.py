# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：PaddleOCR-VL 兜底（对 MinerU 解析质量不佳页面重识别）

定位（工单要求"MinerU 主 + PaddleOCR-VL 兜底"）：
  - 触发条件：quality_report 找出的"低文本页"（扫描件 / 图片页 / 复杂表格）。
  - 本机 PaddleOCR-VL 有独立 venv（D:\\model\\Paddle-OCR\\venv，paddleocr 3.7 +
    paddlepaddle-gpu 3.3.1），主环境未装 paddle → 通过**子进程**调用，零污染。
  - 强制 PADDLE_PDX_MODEL_SOURCE=local（模型目录在 venv 的官方模型缓存中），
    绝不触发下载（硬约束）。
  - 任何失败（venv 缺失 / 模型装载失败 / 超时）都不抛出：返回原样 items，
    主链路（MinerU 或 PyMuPDF）结果不受影响。

说明：本招股说明书 548 页均带文本层，实测 OCR 兜底通常**不会触发**；
本模块为"文档解析失败/扫描件"场景提供符合工单要求的容错路径。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import json
import os
import subprocess
import sys
import time

from src import config

# 子进程内联脚本：优先 PaddleOCRVL pipeline，回退 PaddleOCR(ocr)；输出 JSON 到 stdout
_CHILD_SCRIPT = r'''
import json, os, sys
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "local")
imgs = json.loads(sys.argv[1])
out = []
try:
    from paddleocr import PaddleOCRVL
    pipe = PaddleOCRVL()
    for img in imgs:
        try:
            for res in pipe.predict(img):
                md = getattr(res, "markdown", None)
                text = md.get("markdown_texts") if isinstance(md, dict) else None
                if not text:
                    text = getattr(res, "text", "") or ""
                out.append({"img": img, "text": str(text)})
        except Exception as e:
            out.append({"img": img, "text": "", "error": repr(e)})
except Exception as e1:
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False)
        for img in imgs:
            try:
                res_list = ocr.predict(img)
                texts = []
                for r in res_list:
                    js = getattr(r, "json", None) or {}
                    for item in (js.get("res", {}).get("rec_texts") or []):
                        texts.append(str(item))
                out.append({"img": img, "text": "\n".join(texts)})
            except Exception as e:
                out.append({"img": img, "text": "", "error": repr(e)})
    except Exception as e2:
        print(json.dumps({"fatal": repr(e1) + " | " + repr(e2)}))
        sys.exit(0)
print(json.dumps({"results": out}, ensure_ascii=False))
'''


def is_available() -> bool:
    """PaddleOCR-VL 独立 venv 是否可用。"""
    return os.path.isfile(config.PADDLEOCR_PYTHON)


def render_pages(pdf_path: str, pages: list[int], out_dir: str,
                 dpi: int = 150) -> list[tuple[int, str]]:
    """用 PyMuPDF 把指定页渲染为 PNG。返回 [(page_idx, png_path)]。"""
    import fitz

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    out: list[tuple[int, str]] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for p in pages:
        if p < 0 or p >= doc.page_count:
            continue
        pix = doc[p].get_pixmap(matrix=mat)
        path = os.path.join(out_dir, f"page_{p:04d}.png")
        pix.save(path)
        out.append((p, path))
    doc.close()
    return out


def run_paddleocr_vl(pages_and_images: list[tuple[int, str]],
                     timeout: int | None = None) -> list[dict]:
    """子进程调用 PaddleOCR-VL。返回 [{"page_idx", "text"}]（失败返回 []）。"""
    if not pages_and_images or not is_available():
        return []
    timeout = timeout or config.OCR_TIMEOUT_S
    imgs = [p for _idx, p in pages_and_images]
    idx_by_img = {p: idx for idx, p in pages_and_images}
    try:
        proc = subprocess.run(
            [config.PADDLEOCR_PYTHON, "-c", _CHILD_SCRIPT, json.dumps(imgs)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=config.ROOT,
        )
    except subprocess.TimeoutExpired:
        print(f"[ocr] PaddleOCR-VL 超时（>{timeout}s），本轮跳过", flush=True)
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"[ocr] PaddleOCR-VL 调用失败：{exc}", flush=True)
        return []

    payload = (proc.stdout or "").strip().splitlines()
    data = None
    for line in reversed(payload):
        try:
            data = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        print(f"[ocr] 无法解析输出：{(proc.stdout or '')[-300:]}", flush=True)
        return []
    if data.get("fatal"):
        print(f"[ocr] venv 内 OCR 初始化失败：{data['fatal'][:300]}", flush=True)
        return []

    out: list[dict] = []
    for r in data.get("results", []):
        text = (r.get("text") or "").strip()
        if text:
            out.append({"page_idx": idx_by_img.get(r.get("img"), -1), "text": text})
    return out


def augment_items(items: list[dict], pdf_path: str, out_dir: str,
                  max_pages: int | None = None) -> tuple[list[dict], dict]:
    """对低文本页做 OCR 并把结果作为 text 条目并入 items。

    返回 (新 items, 统计)。
    """
    from src import pdf_parser

    max_pages = max_pages or config.OCR_MAX_PAGES
    page_total = pdf_parser.page_count(pdf_path)
    rep = pdf_parser.quality_report(items, page_total)
    pages = rep["low_text_pages"][:max_pages]
    stat = {"triggered": bool(pages), "pages": pages, "added": 0,
            "available": is_available()}
    if not pages:
        return items, stat
    if not is_available():
        print("[ocr] 低文本页存在，但 PaddleOCR-VL venv 不可用 → 跳过兜底", flush=True)
        return items, stat

    t0 = time.time()
    rendered = render_pages(pdf_path, pages, os.path.join(out_dir, "_ocr_pages"))
    results = run_paddleocr_vl(rendered)
    stat["elapsed_s"] = round(time.time() - t0, 1)

    new_items = list(items)
    for r in results:
        if r["page_idx"] < 0 or not r["text"]:
            continue
        new_items.append({"type": "text", "text": r["text"],
                          "page_idx": int(r["page_idx"]), "text_level": None,
                          "caption": [], "from_ocr": True})
        stat["added"] += 1
    print(f"[ocr] 兜底完成：{stat}", flush=True)
    return new_items, stat
