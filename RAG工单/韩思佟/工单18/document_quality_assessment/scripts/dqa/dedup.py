# -*- coding: utf-8 -*-
"""
DQA · 重复检测（两层机制）

第一层 MD5：识别完全相同的文件（精确匹配，无需人工确认）
第二层 SimHash：识别高相似度文档（不同版本的同一份文件）——需要人工确认，
                因此输出"待确认的版本冲突列表"，附带相似片段供操作员判断。
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    import jieba
    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:                                            # noqa: BLE001
    _HAS_JIEBA = False


def _read_text(path: Path, cap: int = 400_000) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            doc = fitz.open(str(path))
            buf = []
            for page in doc:
                try:
                    buf.append(page.get_text("text") or "")
                except Exception:                                # noqa: BLE001
                    pass
                if sum(len(b) for b in buf) > cap:
                    break
            doc.close()
            return re.sub(r"\s+", " ", "".join(buf))[:cap]
        if ext in (".txt", ".md", ".markdown", ".csv", ".json", ".jsonl"):
            return re.sub(r"\s+", " ", path.read_text(encoding="utf-8", errors="ignore"))[:cap]
        if ext == ".docx":
            from docx import Document
            d = Document(str(path))
            return re.sub(r"\s+", " ", " ".join(p.text for p in d.paragraphs))[:cap]
    except Exception:                                        # noqa: BLE001
        return ""
    return ""


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return ""
    return h.hexdigest()


# ---------------------------------------------------------------- SimHash
def _tokens(text: str, n: int = 3) -> List[str]:
    if _HAS_JIEBA:
        toks = [t for t in jieba.lcut(text) if len(t.strip()) > 1]
        if len(toks) >= 20:
            return toks
    # 回退：字符 n-gram
    t = re.sub(r"\s+", "", text)
    return [t[i:i + n] for i in range(0, max(len(t) - n, 1), n)]


def simhash64(text: str, bits: int = 64) -> int:
    v = [0] * bits
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= (1 << i)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _sim(a: int, b: int, bits: int = 64) -> float:
    return round(1 - hamming(a, b) / bits, 4)


def _sample_snippet(text: str, n: int = 120) -> str:
    return text[:n]


def detect_duplicates(root: str | Path, exts=None,
                      simhash_threshold: float = 0.90,
                      max_files: int | None = None,
                      progress=None) -> dict:
    root = Path(root)
    exts = exts or {".pdf", ".docx", ".doc", ".md", ".txt"}
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix.lower() in exts]
    if max_files:
        files = files[:max_files]

    # ---- 第一层：MD5 精确去重
    by_md5: Dict[str, List[str]] = defaultdict(list)
    for p in files:
        h = md5_of(p)
        if h:
            by_md5[h].append(str(p))
    exact_groups = [{"md5": h, "数量": len(v), "文件": v}
                    for h, v in by_md5.items() if len(v) > 1]
    exact_dup_files = sum(len(g["文件"]) - 1 for g in exact_groups)

    # ---- 第二层：SimHash 近似去重
    # 先按 MD5 去重，同一 MD5 只保留一个代表，减少比较量
    reps = [v[0] for v in by_md5.values()]
    sh = {}
    for i, fp in enumerate(reps, 1):
        txt = _read_text(Path(fp))
        if len(txt) < 100:                     # 太短不做相似度比较（误报高）
            continue
        sh[fp] = (simhash64(txt), txt)
        if progress and i % 20 == 0:
            progress(i, len(reps))

    keys = list(sh.keys())
    low_conf: List[dict] = []
    seen_pairs = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            s = _sim(sh[a][0], sh[b][0])
            if s >= simhash_threshold:
                low_conf.append({
                    "文件A": a, "文件B": b, "相似度": s,
                    "A片段": _sample_snippet(sh[a][1]),
                    "B片段": _sample_snippet(sh[b][1]),
                    "建议": "人工确认是否为不同版本的同一份文档，是则只保留最新版",
                })
    low_conf.sort(key=lambda x: -x["相似度"])

    return {
        "参与检测文件数": len(files),
        "MD5精确重复组数": len(exact_groups),
        "MD5可清理文件数": exact_dup_files,
        "MD5重复组": exact_groups[:50],
        "待确认的版本冲突列表": low_conf[:100],
        "待确认数量": len(low_conf),
        "SimHash阈值": simhash_threshold,
    }
