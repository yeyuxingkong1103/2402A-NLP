# -*- coding: utf-8 -*-
"""
DocumentQualityAssessmentSkill · 检测器
五大功能：格式分布统计 / PDF 页面类型识别 / 文档长度分布 / 重复检测(MD5+SimHash) / 敏感信息检测
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import pymupdf  # PyMuPDF
except Exception:
    pymupdf = None

from .config import Config


# ---------------- 通用 ----------------
def iter_files(root: str, exts: List[str]):
    exts = [e.lower() for e in exts]
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                yield os.path.join(dirpath, fn)


def read_text(path: str) -> str:
    """尽量抽取纯文本（pdf 用 pymupdf；其它按 utf-8 读）。"""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf" and pymupdf is not None:
            doc = pymupdf.open(path)
            txt = "\n".join(p.get_text() for p in doc)
            doc.close()
            return txt
        if ext in (".txt", ".md", ".markdown"):
            with open(path, encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        if ext in (".docx", ".doc"):
            try:
                import docx  # python-docx 可选
                return "\n".join(p.text for p in docx.Document(path).paragraphs)
            except Exception:
                return ""
    except Exception:
        return ""
    return ""


# ---------------- 1) 格式分布 ----------------
def format_stats(files: List[str]) -> dict:
    c = Counter(os.path.splitext(f)[1].lower() or "(无扩展名)" for f in files)
    total = len(files) or 1
    return {
        "total_files": len(files),
        "by_ext": {k: {"count": v, "ratio": round(v / total, 4)} for k, v in c.most_common()},
    }


# ---------------- 2) PDF 页面类型识别 ----------------
def pdf_page_type(path: str, cfg: Config) -> Optional[dict]:
    if pymupdf is None:
        return None
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        return {"error": str(e), "type": "损坏/无法打开"}
    n = len(doc)
    scan_pages = text_pages = 0
    page_chars = []
    for p in doc:
        t = len((p.get_text() or "").strip())
        page_chars.append(t)
        if t < cfg.pdf_min_chars_per_page:
            scan_pages += 1
        else:
            text_pages += 1
    doc.close()
    if n == 0:
        dtype = "待确认(空文档)"
    elif scan_pages / n > cfg.scan_page_ratio:
        dtype = "扫描型"
    elif text_pages / n > cfg.text_page_ratio:
        dtype = "文字型"
    else:
        dtype = "混合型"
    return {
        "type": dtype,
        "pages": n,
        "scan_pages": scan_pages,
        "text_pages": text_pages,
        "scan_ratio": round(scan_pages / n, 4) if n else 0,
    }


# ---------------- 3) 文档长度分布 ----------------
def length_distribution(lengths: List[int], quantiles: List[float]) -> dict:
    if not lengths:
        return {"count": 0}
    xs = sorted(lengths)
    def q(p):
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]
    dist = {"count": len(xs), "min": xs[0], "max": xs[-1],
            "mean": round(sum(xs) / len(xs), 1)}
    for p in quantiles:
        dist[f"P{int(p*100)}"] = q(p)
    bins = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 100000), (100000, 10**9)]
    dist["bins"] = {f"{a}-{b}": sum(1 for x in xs if a <= x < b) for a, b in bins}
    return dist


# ---------------- 4) 重复检测 ----------------
def md5_of_file(path: str, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


def md5_duplicates(files: List[str]) -> List[dict]:
    seen = defaultdict(list)
    for f in files:
        m = md5_of_file(f)
        if m:
            seen[m].append(f)
    return [{"md5": m, "files": v, "count": len(v)} for m, v in seen.items() if len(v) > 1]


def _simhash(text: str, bits: int = 64, shingle: int = 3) -> int:
    text = "".join(text.split())
    if not text:
        return 0
    v = [0] * bits
    for i in range(max(1, len(text) - shingle + 1)):
        sh = text[i:i + shingle]
        h = int(hashlib.md5(sh.encode("utf-8")).hexdigest(), 16)
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if v[b] > 0:
            out |= (1 << b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicates(texts: Dict[str, str], cfg: Config) -> List[dict]:
    """SimHash 近似去重；仅对非空文本参与。返回「待确认的版本冲突列表」。"""
    sigs = {p: _simhash(t, cfg.simhash_bits, cfg.simhash_shingle)
            for p, t in texts.items() if t.strip()}
    items = list(sigs.items())
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = hamming(items[i][1], items[j][1])
            if d <= cfg.simhash_hamming_threshold:
                out.append({"file_a": items[i][0], "file_b": items[j][0],
                            "hamming": d, "status": "待确认(版本冲突?)"})
    return out


# ---------------- 5) 敏感信息检测 ----------------
PATTERNS = {
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "idcard": re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    "bankcard": re.compile(r"(?<!\d)(?:62|4\d|5[1-5])\d{13,17}(?!\d)"),
}


def sensitive_scan(texts: Dict[str, str], cfg: Config) -> dict:
    enabled = {"phone": cfg.enable_phone, "email": cfg.enable_email,
               "idcard": cfg.enable_idcard, "bankcard": cfg.enable_bankcard}
    hits = []
    counts = Counter()
    for path, text in texts.items():
        if not text:
            continue
        for kind, pat in PATTERNS.items():
            if not enabled.get(kind):
                continue
            for m in pat.finditer(text):
                s = m.group(0)
                ctx = text[max(0, m.start() - cfg.context_chars): m.end() + cfg.context_chars].replace("\n", " ")
                hits.append({"file": path, "kind": kind, "value": s, "context": ctx,
                             "status": "待审核"})
                counts[kind] += 1
    return {"enabled": enabled, "counts": dict(counts), "items": hits}


# ---------------- 6) 文档分类标签体系 ----------------
def classify_doc(name: str, pdf_type: Optional[str], text_len: int) -> str:
    """输出分类标签（供工作流路由）。"""
    tags = []
    if pdf_type == "扫描型":
        tags.append("Scan_PDF")
    elif pdf_type == "文字型":
        tags.append("Text_PDF")
    elif pdf_type == "混合型":
        tags.append("Mixed_PDF")
    if pdf_type is None:
        ext = os.path.splitext(name)[1].lower()
        tags.append({"md": "Markdown", "txt": "PlainText", "docx": "Word",
                     "xlsx": "Excel", "pptx": "Slides"}.get(ext.lstrip("."), "Other"))
    if re.search(r"(专利|CN\d+)", os.path.basename(name), re.IGNORECASE):
        tags.append("Patent")
    tags.append("ShortDoc" if text_len < 200 else "LongDoc")
    return "+".join(tags)
