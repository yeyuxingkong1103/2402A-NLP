# -*- coding: utf-8 -*-
"""工单7「测试评估台」——**纯标准库 Web 界面**（零第三方依赖）。

工单编号：人工智能NLP-RAG-功能测试及评估

为什么用标准库 ``http.server``
------------------------------
本机的工单1 venv（Python 3.11.15）里 **没有 gradio / streamlit / pandas**，
且工单纪律明令「禁止 pip install」。所以本界面用 ``http.server`` + 内嵌 HTML/JS
实现，只依赖解释器自带的模块；与 ``研发/app/ui/serve_fallback.py``（工单4 的同类兜底）
同一思路，但**本文件独立、不修改也不 import 它**。

三个标签页
----------
1. **问答测试**：输入问题 → 真实调用 ``QAEngine.answer(q, lang='auto', top_k=10,
   return_evidence=True)`` → 展示答案 / 是否拒答及原因 / 引用页 / Top-10 检索片段
   （chunk_id、文件、页码、类型、分数、前 200 字）/ 检索·生成·首字·总耗时 / 生成方式。
2. **评估看板**：读 ``测试/测试数据/评估结果.jsonl`` 与 ``评估汇总.json``（评测官产物）
   → 17 题表格（问题/答案/关键词覆盖率/是否判对/recall@5/@10/耗时/跨语料泄漏）
   + 汇总指标 + ``检索问题分析.md`` 摘要。产物缺失时显示「数据未就绪」并给出生成命令。
3. **测试流程演示**：一键逐步展开 4 步（问题提取 → 索引与检索 → 评估打分 → 结果输出），
   每步显示**真实产物路径**与**真实摘要统计**（全部从文件读，不写死）。

纪律
----
* 端口默认 **7861**，``--port`` 可覆盖；只监听 127.0.0.1。
* **禁止写死假数据**：每个数字都来自真实文件或真实 QAEngine 调用；文件缺失就如实说缺失。
* Ollama 是 CPU 单实例，故问答用全局锁**串行化**，且界面只在人工点击时单题调用。
* ``Answer`` 的 ``chunks`` 是检索层 top-k 命中，``evidence`` 是生成阶段用到的块，两者都展示。

用法
----
.. code-block:: powershell

    $env:PYTHONPATH='E:\\gao6gongdan\\工单7\\研发'
    & 'E:\\gao6gongdan\\工单1\\.venv\\Scripts\\python.exe' 部署/界面/测试评估台.py --port 7861

或（推荐，自动设好全部环境变量 + 日志重定向）::

    pwsh -NoProfile -File 部署/脚本/启动测试评估台.ps1 -Port 7861
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# 常量与路径
# --------------------------------------------------------------------------
WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
TITLE = "工单7 测试评估台"
SUBTITLE = "RAG 功能测试与评估 · 问答测试 / 评估看板 / 测试流程演示"
DEFAULT_PORT = 7861
DEFAULT_TOP_K = 10

REPO_ROOT = Path(__file__).resolve().parents[2]          # ...\工单7

DIR_TESTDATA = REPO_ROOT / "测试" / "测试数据"
DIR_EVALSRC = REPO_ROOT / "研发" / "评估"
DIR_INDEX = REPO_ROOT / "研发" / "data" / "index"
DIR_RAW = REPO_ROOT / "研发" / "data" / "raw"
DIR_PROCESSED = REPO_ROOT / "研发" / "data" / "processed"
DIR_INPUT = REPO_ROOT / "输入"
DIR_SHOTS = REPO_ROOT / "测试" / "测试截图"
DIR_LOG = REPO_ROOT / "部署" / "日志"

P_RESULTS = DIR_TESTDATA / "评估结果.jsonl"
P_CSV = DIR_TESTDATA / "评估结果.csv"
P_SUMMARY = DIR_TESTDATA / "评估汇总.json"
P_ANALYSIS = DIR_TESTDATA / "检索问题分析.md"
P_QUESTIONS = DIR_EVALSRC / "测试问题集.jsonl"
P_QUESTIONS_CSV = DIR_TESTDATA / "测试问题集.csv"
P_MANIFEST = DIR_INDEX / "index_manifest.json"
P_CHUNKS = DIR_PROCESSED / "chunks.jsonl"
P_SAMPLE_PDF = DIR_INPUT / "sample_questions.pdf"
P_CORPUS_ZIP = DIR_INPUT / "ccf_competition.zip"
P_THRESHOLDS = DIR_TESTDATA / "阈值标定.json"

#: 产物缺失时给出的生成命令（真实可跑，与 部署/脚本/一键跑测试评估.ps1 对齐）
GEN_CMD_RESULTS = "pwsh -NoProfile -File 部署/脚本/一键跑测试评估.ps1 -Only eval"
GEN_CMD_INDEX = "pwsh -NoProfile -File 部署/脚本/一键跑测试评估.ps1 -Only index"
GEN_CMD_QUESTIONS = "pwsh -NoProfile -File 部署/脚本/一键跑测试评估.ps1 -Only extract"


# --------------------------------------------------------------------------
# 通用读盘工具（带 mtime 缓存，避免每次轮询重复解析 598 KB 的 jsonl）
# --------------------------------------------------------------------------
_CACHE: dict[str, tuple[tuple[float, int], Any]] = {}
_CACHE_LOCK = threading.Lock()


def _stat_key(path: Path) -> tuple[float, int]:
    st = path.stat()
    return (st.st_mtime, st.st_size)


def _cached(path: Path, loader: Any) -> Any:
    key = str(path)
    stamp = _stat_key(path)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    value = loader(path)
    with _CACHE_LOCK:
        _CACHE[key] = (stamp, value)
    return value


def load_json(path: Path) -> Any:
    return _cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    def _load(p: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    return _cached(path, _load)


def load_text(path: Path) -> str:
    return _cached(path, lambda p: p.read_text(encoding="utf-8"))


def file_info(path: Path) -> dict[str, Any]:
    """真实文件元信息：存在性 / 字节数 / 修改时间 / 仓库相对路径。"""
    try:
        st = path.stat()
    except OSError:
        return {"path": str(path), "rel": _rel(path), "exists": False,
                "bytes": None, "mtime": None}
    return {
        "path": str(path),
        "rel": _rel(path),
        "exists": True,
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def human_bytes(n: Any) -> str:
    if not isinstance(n, (int, float)):
        return "-"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024
    return f"{v:.1f} GB"


def pct(x: Any, digits: int = 1) -> str:
    if not isinstance(x, (int, float)):
        return "-"
    return f"{x * 100:.{digits}f}%"


# --------------------------------------------------------------------------
# 迷你 Markdown → HTML（只覆盖 检索问题分析.md 实际用到的语法）
# --------------------------------------------------------------------------
_MD_TABLE_SEP = re.compile(r"^\|[\s:\-|]+\|$")


def _md_inline(text: str) -> str:
    s = html.escape(text, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def md_to_html(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        if not line.strip():
            i += 1
            continue
        # 表格
        if line.lstrip().startswith("|") and i + 1 < n and _MD_TABLE_SEP.match(lines[i + 1].strip()):
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            body: list[list[str]] = []
            while i < n and lines[i].lstrip().startswith("|"):
                body.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table class='md'><thead><tr>"
                       + "".join(f"<th>{_md_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>")
            for row in body:
                out.append("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lv = min(6, len(m.group(1)) + 2)          # 摘要里整体降两级
            out.append(f"<h{lv}>{_md_inline(m.group(2))}</h{lv}>")
            i += 1
            continue
        # 引用
        if line.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip())
                i += 1
            out.append("<blockquote>" + _md_inline(" ".join(buf)) + "</blockquote>")
            continue
        # 列表
        if re.match(r"^\s*[-*]\s+", line):
            out.append("<ul>")
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                out.append("<li>" + _md_inline(re.sub(r"^\s*[-*]\s+", "", lines[i])) + "</li>")
                i += 1
            out.append("</ul>")
            continue
        if re.match(r"^\s*\d+[.)]\s+", line):
            out.append("<ol>")
            while i < n and re.match(r"^\s*\d+[.)]\s+", lines[i]):
                out.append("<li>" + _md_inline(re.sub(r"^\s*\d+[.)]\s+", "", lines[i])) + "</li>")
                i += 1
            out.append("</ol>")
            continue
        # 普通段落（连续非空行合并）
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|\s*[-*]\s|\s*\d+[.)]\s|>|\|)", lines[i]):
            buf.append(lines[i].rstrip())
            i += 1
        out.append("<p>" + _md_inline(" ".join(buf)) + "</p>")
    return "\n".join(out)


def split_md_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """把 markdown 拆成 (前言, [(二级标题, 正文), ...])。"""
    parts = re.split(r"^##\s+(.+)$", text, flags=re.M)
    preamble = parts[0].strip()
    sections: list[tuple[str, str]] = []
    for idx in range(1, len(parts), 2):
        sections.append((parts[idx].strip(), parts[idx + 1].strip()))
    return preamble, sections


# --------------------------------------------------------------------------
# QAEngine 托管（懒加载 + 预热 + 串行化）
# --------------------------------------------------------------------------
class EngineBox:
    """持有唯一的 QAEngine；后台预热；问答走全局锁串行（Ollama 是 CPU 单实例）。"""

    def __init__(self) -> None:
        self._ask_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.state = "cold"            # cold | warming | ready | failed
        self.error: str | None = None
        self.warmup: dict[str, Any] | None = None
        self.warmup_ms: float | None = None
        self.health_cache: dict[str, Any] | None = None
        self.engine: Any = None
        self.started_at: str | None = None
        self.ask_count = 0

    # -- 状态 ---------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "state": self.state,
                "error": self.error,
                "warmup_ms": self.warmup_ms,
                "warmup": self.warmup,
                "started_at": self.started_at,
                "ask_count": self.ask_count,
            }

    def _set(self, **kw: Any) -> None:
        with self._state_lock:
            for k, v in kw.items():
                setattr(self, k, v)

    # -- 预热 ---------------------------------------------------------
    def start_warmup(self, *, with_generate: bool = True) -> None:
        with self._state_lock:
            if self.state != "cold":
                return
            self.state = "warming"
            self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        threading.Thread(target=self._warm, kwargs={"with_generate": with_generate},
                         name="qa-warmup", daemon=True).start()

    def _warm(self, *, with_generate: bool) -> None:
        t0 = time.perf_counter()
        try:
            sys.path.insert(0, str(REPO_ROOT / "研发"))
            from app.core.config import get_config
            from app.core.qa_engine import QAEngine

            cfg = get_config()
            engine = QAEngine(cfg=cfg)
            self.engine = engine
            try:
                warm = engine.warmup(with_generate=with_generate)
            except Exception as exc:                       # noqa: BLE001
                # 预热失败不致命：rule_composer / 检索链路仍可能可用
                warm = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._set(state="ready", warmup=warm,
                      warmup_ms=round((time.perf_counter() - t0) * 1000, 2))
        except Exception as exc:                            # noqa: BLE001
            self._set(state="failed", error=f"{type(exc).__name__}: {exc}",
                      warmup_ms=round((time.perf_counter() - t0) * 1000, 2))

    # -- 问答 ---------------------------------------------------------
    def ask(self, question: str, *, lang: str = "auto", top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """真实调用 QAEngine.answer(...)；返回可直接 JSON 化的载荷。"""
        snap = self.snapshot()
        if snap["state"] == "failed":
            raise RuntimeError(f"QAEngine 初始化失败：{snap['error']}")
        if snap["state"] != "ready":
            raise RuntimeError(f"QAEngine 尚未就绪（当前状态 {snap['state']}，请稍候重试）")

        with self._ask_lock:                                # 串行：CPU 单实例 Ollama
            t0 = time.perf_counter()
            result = self.engine.answer(question, lang=lang, top_k=top_k,
                                        return_evidence=True)
            wall_ms = round((time.perf_counter() - t0) * 1000, 2)
            with self._state_lock:
                self.ask_count += 1

        chunks = list(getattr(result, "chunks", None) or [])
        evidence = list(getattr(result, "evidence", None) or [])
        citations = list(getattr(result, "citations", None) or [])

        def _hits(items: list[dict[str, Any]], *, truncate: bool) -> list[dict[str, Any]]:
            out = []
            for i, h in enumerate(items):
                text = str(h.get("text") or "")
                out.append({
                    "rank": i + 1,
                    "chunk_id": h.get("chunk_id"),
                    "file_name": h.get("file_name"),
                    "page": h.get("page"),
                    "type": h.get("type") or h.get("modality"),
                    "modality_sub": h.get("modality_sub"),
                    "final_score": h.get("final_score", h.get("score")),
                    "vector_rank": h.get("vector_rank"),
                    "bm25_rank": h.get("bm25_rank"),
                    "text_len": len(text),
                    "text_head": text[:200] if truncate else text,
                })
            return out

        return {
            "question": question,
            "lang": lang,
            "top_k": top_k,
            "answer": result.answer,
            "answer_chars": len(result.answer or ""),
            "refused": bool(result.refused),
            "refusal_reason": result.refusal_reason,
            "refusal_detail": result.refusal_detail,
            "cited_pages": list(getattr(result, "cited_pages", None) or []),
            "citations": citations,
            "chunks": _hits(chunks, truncate=True),
            "evidence": _hits(evidence, truncate=True),
            "latency_ms": dict(result.latency_ms or {}),
            "wall_ms": wall_ms,
            "generation_method": result.generation_method,
            "backend": result.backend,
            "model": result.model,
            "language": result.language,
            "trace_id": result.trace_id,
            "cache": result.cache,
            "gate": (result.analysis or {}).get("gate") if isinstance(result.analysis, dict) else None,
            "analysis": result.analysis,
        }


ENGINE = EngineBox()


# --------------------------------------------------------------------------
# 评估看板载荷
# --------------------------------------------------------------------------
def build_eval_payload() -> dict[str, Any]:
    files = {
        "评估结果.jsonl": file_info(P_RESULTS),
        "评估结果.csv": file_info(P_CSV),
        "评估汇总.json": file_info(P_SUMMARY),
        "检索问题分析.md": file_info(P_ANALYSIS),
        "测试问题集.jsonl": file_info(P_QUESTIONS),
    }
    if not P_RESULTS.is_file():
        return {
            "ready": False,
            "reason": f"评测产物不存在：{_rel(P_RESULTS)}",
            "command": GEN_CMD_RESULTS,
            "files": files,
        }

    records = load_jsonl(P_RESULTS)
    rows: list[dict[str, Any]] = []
    for r in records:
        j = r.get("judge") or {}
        leak = j.get("cross_corpus_leak")
        if isinstance(leak, dict):
            leak_bool = bool(leak.get("leak"))
            leak_files = [f.get("file_name") for f in (leak.get("foreign") or [])]
        else:
            leak_bool = bool(leak)
            leak_files = []
        lat = r.get("latency_ms") or {}
        rows.append({
            "id": r.get("id"),
            "question": r.get("question"),
            "question_type": r.get("question_type"),
            "company": r.get("company"),
            "lang": r.get("lang"),
            "answer": r.get("answer"),
            "answer_chars": r.get("answer_chars"),
            "refused": r.get("refused"),
            "refusal_reason": r.get("refusal_reason"),
            "coverage": j.get("keyword_coverage"),
            "keyword_hit": j.get("keyword_hit_count"),
            "keyword_total": j.get("keyword_total"),
            "is_correct": j.get("is_correct"),
            "partial_correct": j.get("partial_correct"),
            "recall5": j.get("recall_at_5"),
            "recall10": j.get("recall_at_10"),
            "page_hit": j.get("page_hit"),
            "leak": leak_bool,
            "leak_files": leak_files,
            "citation_leak": j.get("citation_leak"),
            "total_ms": r.get("total_ms", lat.get("total")),
            "retrieve_ms": lat.get("retrieve"),
            "generate_ms": lat.get("generate"),
            "first_token_ms": lat.get("first_token"),
            "generation_method": r.get("generation_method"),
            "cited_pages": r.get("cited_pages"),
            "expected_pages_hint": r.get("expected_pages_hint"),
            "hits": [
                {
                    "rank": h.get("rank"),
                    "chunk_id": h.get("chunk_id"),
                    "page": h.get("page"),
                    "type": h.get("type"),
                    "final_score": h.get("final_score"),
                    "file_name": h.get("file_name"),
                    "vector_rank": h.get("vector_rank"),
                    "bm25_rank": h.get("bm25_rank"),
                    "text_head": str(h.get("text_head") or "")[:200],
                }
                for h in (r.get("hits") or [])
            ],
            "error": r.get("error"),
        })

    summary = None
    if P_SUMMARY.is_file():
        summary = load_json(P_SUMMARY)

    analysis = None
    if P_ANALYSIS.is_file():
        preamble, sections = split_md_sections(load_text(P_ANALYSIS))
        wanted = ("二、", "四、", "五、")
        picked = [(t, b) for (t, b) in sections if t.startswith(wanted)]
        analysis = {
            "path": _rel(P_ANALYSIS),
            "preamble_html": md_to_html(preamble),
            "sections": [{"title": t, "html": md_to_html(b)} for (t, b) in picked],
            "total_sections": len(sections),
        }

    return {
        "ready": True,
        "rows": rows,
        "count": len(rows),
        "summary": summary,
        "analysis": analysis,
        "files": files,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------
# 测试流程演示载荷（4 步，全部真实读盘）
# --------------------------------------------------------------------------
def _question_stats() -> dict[str, Any] | None:
    if not P_QUESTIONS.is_file():
        return None
    recs = load_jsonl(P_QUESTIONS)
    kw_total = sum(len(r.get("answer_keywords") or []) for r in recs)
    by_lang: dict[str, int] = {}
    by_pdf: dict[str, int] = {}
    pages = []
    for r in recs:
        by_lang[str(r.get("lang"))] = by_lang.get(str(r.get("lang")), 0) + 1
        src = str(r.get("source_pdf") or "?")
        by_pdf[src] = by_pdf.get(src, 0) + 1
        hint = str(r.get("expected_pages_hint") or "")
        if hint:
            pages.append(hint)
    return {
        "count": len(recs),
        "keywords": kw_total,
        "by_lang": by_lang,
        "by_source_pdf": by_pdf,
        "ids": [r.get("id") for r in recs],
        "page_hints": pages,
    }


def _pdf_page_count() -> int | None:
    try:
        import fitz                                          # pymupdf，已装
        with fitz.open(str(P_SAMPLE_PDF)) as doc:
            return doc.page_count
    except Exception:                                        # noqa: BLE001
        return None


def build_flow_payload() -> dict[str, Any]:
    manifest = load_json(P_MANIFEST) if P_MANIFEST.is_file() else None
    qstats = _question_stats()

    raw_pdfs = sorted(DIR_RAW.glob("*.pdf")) if DIR_RAW.is_dir() else []
    shots = sorted(DIR_SHOTS.glob("*.png")) if DIR_SHOTS.is_dir() else []

    steps: list[dict[str, Any]] = []

    # ---- 第 1 步：问题提取 ----
    steps.append({
        "no": 1,
        "key": "extract",
        "title": "问题提取",
        "subtitle": "sample_questions.pdf → 测试问题集",
        "artifacts": [
            {**file_info(P_SAMPLE_PDF), "label": "竞赛样例题 PDF"},
            {**file_info(P_QUESTIONS), "label": "测试问题集（JSONL，评测输入）"},
            {**file_info(P_QUESTIONS_CSV), "label": "测试问题集（CSV 副本）"},
        ],
        "stats": [
            {"k": "PDF 页数", "v": (_pdf_page_count() if P_SAMPLE_PDF.is_file() else None) or "—"},
            {"k": "提取题数", "v": f"{qstats['count']} 题" if qstats else "数据未就绪"},
            {"k": "判分关键词总数", "v": f"{qstats['keywords']} 个" if qstats else "—"},
            {"k": "题目来源标签数", "v": f"{len(qstats['by_source_pdf'])} 个（含跨文档题的多来源标签）" if qstats else "—"},
            {"k": "语料 PDF 数", "v": f"{len(raw_pdfs)} 份（研发/data/raw）"},
            {"k": "题型/语言分布", "v": json.dumps(qstats["by_lang"], ensure_ascii=False) if qstats else "—"},
        ],
        "note": "题目与答案要点由出题官从 输入/sample_questions.pdf 提取，逐题登记 answer_keywords 作为判分口径。",
        "ready": bool(qstats),
    })

    # ---- 第 2 步：索引与检索 ----
    modality = (manifest or {}).get("chunk_by_modality") or {}
    corpora = (manifest or {}).get("chunk_by_corpus") or {}
    bm25 = (manifest or {}).get("bm25") or {}
    vector = (manifest or {}).get("vector") or {}
    embedding = (manifest or {}).get("embedding") or {}
    steps.append({
        "no": 2,
        "key": "index",
        "title": "语料与索引",
        "subtitle": "CCF 语料 → 解析 → 分块 → 嵌入 → FAISS + BM25",
        "artifacts": [
            {**file_info(P_CORPUS_ZIP), "label": "CCF 竞赛语料压缩包"},
            {**file_info(P_MANIFEST), "label": "索引 manifest（权威口径）"},
            {**file_info(P_CHUNKS), "label": "全量分块 chunks.jsonl"},
            {**file_info(DIR_INDEX / str((manifest or {}).get("index_dir", "bge-m3_1024")).replace("/", "\\").split("\\")[-1] / "faiss.index"), "label": "FAISS 向量索引"},
        ],
        "stats": [
            {"k": "语料文件数", "v": f"{len(raw_pdfs)} 份 PDF"},
            {"k": "索引块总数", "v": f"{(manifest or {}).get('chunk_total', '—')} 块"},
            {"k": "按模态", "v": f"正文 {modality.get('text', '—')} / 表格 {modality.get('table', '—')} / 图像 {modality.get('image', '—')}"},
            {"k": "嵌入模型", "v": f"{(manifest or {}).get('embed_model', '—')}（{(manifest or {}).get('embed_dim', '—')} 维）"},
            {"k": "向量后端", "v": f"{(manifest or {}).get('vector_backend', '—')}"},
            {"k": "BM25 词表", "v": f"{bm25.get('vocab_size', '—')} 词（k1={bm25.get('k1', '—')}, b={bm25.get('b', '—')}）"},
            {"k": "语料覆盖", "v": f"{len(corpora)} 份年报"},
            {"k": "索引 run_id", "v": f"{(manifest or {}).get('run_id', '—')}"},
        ],
        "note": ("图像块数为 0 是本工单【已知降级】：本机缺 cv2/onnxruntime 且未缓存中文 CLIP 权重，"
                 "文本/表格索引照常构建（详见 manifest.known_limitations 与 部署/README.md 已知限制）。"),
        "ready": manifest is not None,
    })

    # ---- 第 3 步：评估打分 ----
    summary = load_json(P_SUMMARY) if P_SUMMARY.is_file() else None
    acc = (summary or {}).get("准确率") or {}
    rec = (summary or {}).get("检索召回") or {}
    sla = (summary or {}).get("SLA") or {}
    reach = (summary or {}).get("索引可达性") or {}
    leak = (summary or {}).get("跨语料泄漏") or {}
    n_rows = len(load_jsonl(P_RESULTS)) if P_RESULTS.is_file() else 0
    steps.append({
        "no": 3,
        "key": "judge",
        "title": "评估打分",
        "subtitle": "17 题逐题检索 + 生成 + 三层判分",
        "artifacts": [
            {**file_info(P_RESULTS), "label": "逐题评估结果（JSONL）"},
            {**file_info(P_SUMMARY), "label": "评估汇总（JSON）"},
            {**file_info(P_THRESHOLDS), "label": "可答性/判分阈值标定"},
        ],
        "stats": [
            {"k": "评估题数", "v": f"{n_rows} 题" if n_rows else "数据未就绪"},
            {"k": "判分关键词总数", "v": f"{reach.get('关键词总数', '—')} 个"},
            {"k": "严格判对", "v": f"{acc.get('严格判对题数', '—')}/{n_rows or '—'}"
                                    f"（严格准确率 {pct(acc.get('严格准确率'))}）"},
            {"k": "部分正确", "v": f"{acc.get('部分正确题数', '—')} 题"
                                  f"（{'、'.join(acc.get('部分正确题号') or []) or '—'}）"},
            {"k": "宽松准确率", "v": pct(acc.get("宽松准确率(严格+部分)"))},
            {"k": "平均关键词覆盖率", "v": pct(acc.get("平均关键词覆盖率"))},
            {"k": "平均 recall@5 / @10", "v": f"{rec.get('平均recall@5', '—')} / {rec.get('平均recall@10', '—')}"},
            {"k": "SLA 达标率", "v": f"{pct(sla.get('达标率'))}（阈值 {sla.get('阈值ms', '—')} ms）"},
            {"k": "跨语料泄漏", "v": f"{leak.get('题数', '—')} 题（{'、'.join(leak.get('题号') or []) or '—'}）"},
            {"k": "索引内单块可达率", "v": pct(reach.get("单块可达率"))},
        ],
        "note": ("判分口径：关键词覆盖率 ≥0.6 判对、≥0.3 记部分正确；recall@k 用 top-k 片段合并正文判定；"
                 "口径全文见 测试/测试数据/评估指标口径说明.md。"),
        "ready": summary is not None,
    })

    # ---- 第 4 步：结果输出 ----
    analysis_md = file_info(P_ANALYSIS)
    steps.append({
        "no": 4,
        "key": "output",
        "title": "结果输出",
        "subtitle": "表格 / 汇总 / 问题分析 / 截图",
        "artifacts": [
            {**file_info(P_CSV), "label": "逐题结果表（CSV，17 行）"},
            {**file_info(P_SUMMARY), "label": "评估汇总指标（JSON）"},
            {**analysis_md, "label": "检索问题分析报告（Markdown）"},
            {"label": "测试截图目录（PNG）", "path": str(DIR_SHOTS), "rel": _rel(DIR_SHOTS),
             "exists": DIR_SHOTS.is_dir(), "bytes": None,
             "mtime": f"{len(shots)} 张 PNG"},
        ],
        "stats": [
            {"k": "有效题率", "v": f"{(summary or {}).get('有效题数', '—')}/{(summary or {}).get('总数', '—')}"
                                  f"（{pct(((summary or {}).get('有效题数') or 0) / max(1, (summary or {}).get('总数') or 1))}）"},
            {"k": "SLA 通过率", "v": pct(sla.get("达标率"))},
            {"k": "首字达标率", "v": pct(sla.get("首字达标率"))},
            {"k": "分析报告字数", "v": f"{len(load_text(P_ANALYSIS))} 字符" if P_ANALYSIS.is_file() else "数据未就绪"},
            {"k": "归因章节数", "v": f"{len(split_md_sections(load_text(P_ANALYSIS))[1])} 节" if P_ANALYSIS.is_file() else "—"},
            {"k": "测试截图", "v": f"{len(shots)} 张"},
            {"k": "平均总耗时", "v": f"{((summary or {}).get('响应时间ms') or {}).get('平均', '—')} ms"},
        ],
        "note": "所有产物路径均为仓库内真实文件，点「查看」可直接用系统默认程序打开。",
        "ready": P_ANALYSIS.is_file(),
    })

    return {
        "work_order": WORK_ORDER,
        "repo_root": str(REPO_ROOT),
        "steps": steps,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_health_payload() -> dict[str, Any]:
    engine = ENGINE.snapshot()
    health = None
    if engine["state"] == "ready" and ENGINE.engine is not None:
        if ENGINE.health_cache is None:
            try:
                ENGINE.health_cache = ENGINE.engine.health()
            except Exception as exc:                          # noqa: BLE001
                ENGINE.health_cache = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        health = ENGINE.health_cache
    return {
        "work_order": WORK_ORDER,
        "title": TITLE,
        "repo_root": str(REPO_ROOT),
        "engine": engine,
        "health": health,
        "artifacts": {
            "评估结果.jsonl": file_info(P_RESULTS),
            "评估汇总.json": file_info(P_SUMMARY),
            "检索问题分析.md": file_info(P_ANALYSIS),
            "测试问题集.jsonl": file_info(P_QUESTIONS),
            "index_manifest.json": file_info(P_MANIFEST),
            "sample_questions.pdf": file_info(P_SAMPLE_PDF),
        },
        "examples": _example_questions(),
    }


def _example_questions() -> list[dict[str, str]]:
    if not P_QUESTIONS.is_file():
        return []
    try:
        recs = load_jsonl(P_QUESTIONS)
    except Exception:                                          # noqa: BLE001
        return []
    out = []
    for r in recs[:6]:
        out.append({"id": str(r.get("id")), "q": str(r.get("question"))[:70]})
    return out


# --------------------------------------------------------------------------
# HTML 页面
# --------------------------------------------------------------------------
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__ · __WORK_ORDER__</title>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --line:#e2e6f0; --ink:#1b2233; --muted:#6b7488;
          --accent:#2f6fed; --accent2:#5b8def; --ok:#1a7f4b; --warn:#b7791f; --bad:#c0392b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); font-size:14px; }
  header { background:linear-gradient(90deg,#2f6fed,#5b8def); color:#fff; padding:16px 24px 0; }
  header h1 { margin:0; font-size:20px; }
  header .sub { margin:6px 0 0; font-size:13px; opacity:.93; }
  .statusbar { display:flex; gap:14px; flex-wrap:wrap; margin:10px 0 0; font-size:12.5px; }
  .pill { background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
          border-radius:999px; padding:3px 11px; }
  .pill.ok::before { content:"● "; color:#9cf5c0; }
  .pill.warn::before { content:"● "; color:#ffe08a; }
  .pill.bad::before { content:"● "; color:#ffb3ab; }
  nav { display:flex; gap:4px; margin-top:14px; }
  nav button { border:0; background:rgba(255,255,255,.16); color:#fff; font-size:14px;
               padding:10px 20px; border-radius:8px 8px 0 0; cursor:pointer; font-family:inherit; }
  nav button.active { background:var(--bg); color:var(--accent); font-weight:700; }
  main { max-width:1420px; margin:0 auto 40px; padding:16px; }
  .panel { display:none; } .panel.active { display:block; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:14px; box-shadow:0 1px 3px rgba(20,30,60,.05); }
  .card h2 { margin:0 0 12px; font-size:15px; color:var(--muted); font-weight:700; }
  .card h3 { margin:16px 0 8px; font-size:14px; }
  textarea { width:100%; min-height:70px; border:1px solid var(--line); border-radius:8px;
             padding:10px; font-size:15px; font-family:inherit; resize:vertical; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:10px; }
  button.act { border:0; border-radius:8px; padding:10px 20px; font-size:15px; cursor:pointer;
               background:var(--accent); color:#fff; font-family:inherit; }
  button.act:disabled { opacity:.5; cursor:progress; }
  button.ghost { border:1px solid var(--line); background:#f0f3fb; color:var(--accent);
                 font-size:12.5px; padding:5px 10px; border-radius:7px; cursor:pointer;
                 font-family:inherit; }
  select,input[type=number] { border:1px solid var(--line); border-radius:7px; padding:6px 8px;
                              font-family:inherit; font-size:13px; }
  .muted { color:var(--muted); font-size:12.5px; }
  .answer { font-size:16px; line-height:1.8; white-space:pre-wrap; word-break:break-word;
            background:#fafbff; border:1px solid var(--line); border-radius:8px; padding:12px; }
  .refused { color:var(--bad); font-weight:700; }
  .cites { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .cite { background:#eef2fb; color:#1f4fd8; border-radius:6px; padding:4px 9px; font-size:12.5px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th,td { border-bottom:1px solid var(--line); text-align:left; padding:6px 8px; vertical-align:top; }
  th { background:#f0f3fb; position:sticky; top:0; z-index:1; }
  table.md th { background:#f0f3fb; }
  .scroll { max-height:560px; overflow:auto; border:1px solid var(--line); border-radius:8px; }
  .tag { display:inline-block; border-radius:4px; padding:1px 6px; font-size:11.5px; margin-right:5px;
         white-space:nowrap; }
  .tag.text { background:#e8eefc; color:#1f4fd8; }
  .tag.table { background:#fff8e1; color:#8d6e00; }
  .tag.image { background:#e8f5e9; color:#1b5e20; }
  .tag.ok { background:#e6f7ec; color:var(--ok); }
  .tag.no { background:#fdeceb; color:var(--bad); }
  .tag.mid { background:#fff5e0; color:var(--warn); }
  .tag.info { background:#eef2fb; color:var(--accent); }
  .pre { white-space:pre-wrap; word-break:break-word; font-family:Consolas,"Courier New",monospace;
         font-size:12px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:10px; }
  .kv { background:#fafbff; border:1px solid var(--line); border-radius:8px; padding:9px 11px; }
  .kv .k { color:var(--muted); font-size:12px; }
  .kv .v { font-size:15px; font-weight:700; margin-top:3px; word-break:break-word; }
  .step { border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:12px;
          background:#fff; }
  .step.dim { opacity:.35; }
  .step .head { display:flex; align-items:center; gap:10px; }
  .step .num { width:28px; height:28px; border-radius:50%; background:var(--accent); color:#fff;
               display:flex; align-items:center; justify-content:center; font-weight:700; flex:none; }
  .step .ttl { font-size:16px; font-weight:700; }
  .step .sub { color:var(--muted); font-size:12.5px; margin-left:auto; }
  .art { font-family:Consolas,"Courier New",monospace; font-size:12px; background:#f7f9ff;
         border:1px solid var(--line); border-radius:7px; padding:7px 9px; margin:6px 0; }
  .art.miss { background:#fdf1f0; border-color:#f3c9c4; }
  pre.json { background:#111827; color:#d7e2f5; border-radius:8px; padding:12px; overflow:auto;
             max-height:420px; font-size:12px; }
  footer { text-align:center; color:var(--muted); font-size:12.5px; padding:18px 16px 34px;
           border-top:1px solid var(--line); background:#fff; }
  blockquote { margin:8px 0; padding:6px 12px; border-left:3px solid var(--accent);
               background:#f7f9ff; color:#39415a; }
  code { background:#eef2fb; border-radius:4px; padding:1px 5px; font-size:12px; }
  details summary { cursor:pointer; color:var(--accent); font-size:12.5px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p class="sub">__WORK_ORDER__ · __SUBTITLE__</p>
  <div class="statusbar">
    <span class="pill" id="pillEngine">引擎：探测中…</span>
    <span class="pill" id="pillIndex">索引：探测中…</span>
    <span class="pill" id="pillEval">评估产物：探测中…</span>
    <span class="pill" id="pillRepo">仓库：__REPO_ROOT__</span>
  </div>
  <nav>
    <button data-tab="ask" class="active">问答测试</button>
    <button data-tab="eval">评估看板</button>
    <button data-tab="flow">测试流程演示</button>
  </nav>
</header>
<main>

  <!-- ================= 标签页 1：问答测试 ================= -->
  <section class="panel active" id="tab-ask">
    <div class="card">
      <h2>提问（真实调用 QAEngine.answer，单题串行）</h2>
      <textarea id="q" placeholder="请输入问题，例如：平安银行2019年年度报告中，报告期内实现营业收入是多少？"></textarea>
      <div class="row">
        <button class="act" id="askBtn">提问</button>
        <label class="muted">语言
          <select id="lang">
            <option value="auto">auto（自动）</option>
            <option value="zh">zh 中文</option>
            <option value="en">en English</option>
          </select>
        </label>
        <label class="muted">top_k <input id="topk" type="number" value="__TOPK__" min="1" max="20" style="width:64px" /></label>
        <span class="muted">return_evidence=True</span>
        <span class="muted" id="askStatus"></span>
      </div>
      <div class="row" id="examples"></div>
    </div>

    <div class="card">
      <h2>答案</h2>
      <div id="answer" class="muted">尚未提问。</div>
      <div id="refusal"></div>
      <div class="cites" id="cites"></div>
      <div class="grid" id="latency" style="margin-top:12px"></div>
      <div class="muted" id="meta" style="margin-top:10px"></div>
    </div>

    <div class="card">
      <h2>Top-10 检索片段（检索层 chunks）</h2>
      <div id="hits" class="muted">尚无检索结果。</div>
    </div>

    <div class="card">
      <h2>证据（生成阶段实际使用的块 evidence）</h2>
      <div id="evidence" class="muted">尚无证据。</div>
    </div>
  </section>

  <!-- ================= 标签页 2：评估看板 ================= -->
  <section class="panel" id="tab-eval">
    <div class="card">
      <h2>数据源</h2>
      <div class="grid" id="evalFiles"></div>
      <div class="row"><button class="ghost" id="reloadEval">重新读取</button>
        <span class="muted" id="evalStamp"></span></div>
    </div>
    <div id="evalBody"></div>
  </section>

  <!-- ================= 标签页 3：测试流程演示 ================= -->
  <section class="panel" id="tab-flow">
    <div class="card">
      <h2>一键演示：4 步走完「问题提取 → 索引与检索 → 评估打分 → 结果输出」</h2>
      <div class="row">
        <button class="act" id="playFlow">▶ 一键逐步演示</button>
        <button class="ghost" id="showAll">全部展开</button>
        <button class="ghost" id="resetFlow">重置</button>
        <span class="muted">数据全部从仓库真实文件读取，无写死数字</span>
      </div>
    </div>
    <div id="flowBody"></div>
  </section>

</main>
<footer>
  工单编号：__WORK_ORDER__　|　测试评估台 · 纯标准库 http.server 实现　|　端口 __PORT__<br />
  本机无 gradio / streamlit / pandas（且工单纪律禁止 pip install），故用标准库实现。
</footer>
<script>
const $ = id => document.getElementById(id);
const esc = s => (s === null || s === undefined) ? '' : String(s)
  .replace(/&/g,'&amp;').replace(/<[^>]*>/g, m => m.replace(/</g,'&lt;'));
const escAttr = s => String(s === null || s === undefined ? '' : s).replace(/"/g,'&quot;');
const num = (v, d) => (v === null || v === undefined || v === '' || isNaN(v)) ? '-' : Number(v).toFixed(d === undefined ? 0 : d);
const pctf = v => (v === null || v === undefined || isNaN(v)) ? '-' : (Number(v) * 100).toFixed(1) + '%';

/* ---------------- 标签切换 ---------------- */
document.querySelectorAll('nav button').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    $('tab-' + b.dataset.tab).classList.add('active');
    if (b.dataset.tab === 'eval') { loadEval(); }
    if (b.dataset.tab === 'flow') { loadFlow(); }
  };
});

/* ---------------- 健康状态 ---------------- */
let ENGINE_READY = false;
async function pollHealth(){
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    const e = d.engine || {};
    const pe = $('pillEngine');
    if (e.state === 'ready') {
      ENGINE_READY = true;
      pe.className = 'pill ok';
      pe.textContent = '引擎：就绪（预热 ' + num(e.warmup_ms, 0) + ' ms，已答 ' + (e.ask_count||0) + ' 题）';
      $('askBtn').disabled = false;
    } else if (e.state === 'failed') {
      ENGINE_READY = false;
      pe.className = 'pill bad';
      pe.textContent = '引擎：失败 — ' + (e.error || '');
    } else {
      ENGINE_READY = false;
      pe.className = 'pill warn';
      pe.textContent = '引擎：' + (e.state === 'warming' ? '索引与模型预热中…' : '尚未启动');
      $('askBtn').disabled = true;
      setTimeout(pollHealth, 2000);
    }
    const h = d.health;
    const pi = $('pillIndex');
    if (h && h.checks && h.checks.index) {
      const ix = h.checks.index;
      if (ix.ok) { pi.className='pill ok';
        pi.textContent = '索引：' + ix.chunks + ' 块 · ' + ix.dim + ' 维 · FAISS=' + ix.faiss +
                         ' · 图像块 ' + ix.image_chunks; }
      else { pi.className='pill bad'; pi.textContent = '索引：不可用 — ' + (ix.reason||''); }
    } else { pi.className='pill warn'; pi.textContent = '索引：等待引擎'; }
    const ev = (d.artifacts || {})['评估结果.jsonl'] || {};
    const pm = $('pillEval');
    if (ev.exists) { pm.className='pill ok'; pm.textContent = '评估产物：就绪（' + (ev.bytes/1024).toFixed(0) + ' KB）'; }
    else { pm.className='pill warn'; pm.textContent = '评估产物：未就绪'; }
    const box = $('examples');
    if (box && !box.dataset.done && (d.examples||[]).length) {
      box.dataset.done = '1';
      d.examples.forEach(x => {
        const b = document.createElement('button');
        b.className = 'ghost'; b.textContent = '[' + x.id + '] ' + x.q + '…';
        b.title = x.q;
        b.onclick = () => { $('q').value = x.q; };
        box.appendChild(b);
      });
    }
  } catch (err) { $('pillEngine').textContent = '引擎：无法连接后端 — ' + err; }
}
pollHealth();

/* ---------------- 问 答 ---------------- */
$('askBtn').onclick = doAsk;
$('q').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey||e.metaKey)) doAsk(); });

async function doAsk(){
  const question = $('q').value.trim();
  if (!question) { return; }
  $('askBtn').disabled = true;
  $('askStatus').textContent = '检索与生成中…（本机 Ollama 为 CPU 单实例，单题调用）';
  const t0 = performance.now();
  try {
    const r = await fetch('/api/ask', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ question: question, lang: $('lang').value,
                             top_k: parseInt($('topk').value || '__TOPK__', 10) }) });
    const d = await r.json();
    if (!r.ok) {
      $('answer').className = 'answer refused';
      $('answer').textContent = '请求失败：' + JSON.stringify(d.error || d);
      return;
    }
    renderAnswer(d, Math.round(performance.now() - t0));
  } catch (err) {
    $('answer').className = 'answer refused';
    $('answer').textContent = '请求异常：' + err;
  } finally {
    $('askBtn').disabled = !ENGINE_READY;
    $('askStatus').textContent = '';
    pollHealth();
  }
}

function renderAnswer(d, wall){
  const a = $('answer');
  a.className = 'answer';
  if (d.refused) {
    a.className = 'answer refused';
    a.textContent = '不清楚（拒答）';
    $('refusal').innerHTML = '<div class="muted" style="margin-top:6px">拒答原因：<b>' +
      esc(d.refusal_reason || '-') + '</b>' +
      (d.refusal_detail ? '<br />细节：' + esc(JSON.stringify(d.refusal_detail)) : '') + '</div>';
  } else {
    a.textContent = d.answer || '（空答案）';
    $('refusal').innerHTML = '<div class="muted" style="margin-top:6px">未拒答' +
      (d.gate ? '　|　可答性 gate：answerable=' + esc(d.gate.answerable) +
                '，reason=' + esc(d.gate.reason) + '，coverage=' + num(d.gate.coverage, 4) : '') + '</div>';
  }
  const cs = $('cites'); cs.innerHTML = '';
  (d.cited_pages || []).forEach(p => {
    const s = document.createElement('span'); s.className = 'cite'; s.textContent = '第 ' + p + ' 页';
    cs.appendChild(s);
  });
  if (!(d.cited_pages||[]).length) { cs.innerHTML = '<span class="muted">无引用页</span>'; }
  (d.citations || []).forEach(c => {
    const s = document.createElement('span'); s.className = 'cite';
    s.textContent = c.label || (c.corpus + '：第' + c.page + '页');
    s.title = c.quote || '';
    cs.appendChild(s);
  });
  const lat = d.latency_ms || {};
  $('latency').innerHTML = [
    ['检索耗时', num(lat.retrieve, 2) + ' ms'],
    ['生成耗时', num(lat.generate, 2) + ' ms'],
    ['首字耗时', num(lat.first_token, 2) + ' ms'],
    ['总耗时（引擎）', num(lat.total, 2) + ' ms'],
    ['总耗时（墙钟）', num(wall, 0) + ' ms'],
    ['生成方式', d.generation_method || '-'],
  ].map(([k,v]) => '<div class="kv"><div class="k">' + k + '</div><div class="v">' + esc(v) + '</div></div>').join('');
  $('meta').textContent =
    '后端 ' + (d.backend||'-') + ':' + (d.model||'-') +
    '　|　语言 ' + (d.language||'-') +
    '　|　缓存 ' + ((d.cache && d.cache.hit) ? '命中' : '未命中') +
    '　|　trace ' + (d.trace_id||'-') +
    '　|　答案 ' + (d.answer_chars||0) + ' 字' +
    '　|　top_k=' + (d.top_k||'-') + '，lang=' + (d.lang||'-') + '，return_evidence=True';
  renderHits('hits', d.chunks, '检索层未返回 chunk');
  renderHits('evidence', d.evidence, '生成层未使用证据块（可能为拒答或兜底）');
}

function typeTag(t){
  const k = (t === 'table' || t === 'image') ? t : 'text';
  const label = t === 'table' ? '表格' : (t === 'image' ? '图像' : '正文');
  return '<span class="tag ' + k + '">' + label + '</span>';
}

function renderHits(boxId, items, emptyMsg){
  const box = $(boxId);
  if (!items || !items.length) { box.className = 'muted'; box.textContent = emptyMsg; return; }
  box.className = '';
  let h = '<table><tr><th>#</th><th>chunk_id</th><th>文件</th><th>页码</th><th>类型</th>' +
          '<th>final_score</th><th>向量/BM25 名次</th><th>前 200 字</th></tr>';
  items.forEach(it => {
    h += '<tr><td>' + esc(it.rank) + '</td><td class="pre">' + esc(it.chunk_id) + '</td>' +
         '<td class="pre">' + esc(it.file_name) + '</td>' +
         '<td>' + esc(it.page) + '</td>' +
         '<td>' + typeTag(it.type) + (it.modality_sub ? esc(it.modality_sub) : '') + '</td>' +
         '<td>' + num(it.final_score, 6) + '</td>' +
         '<td>' + esc(it.vector_rank === null || it.vector_rank === undefined ? '-' : it.vector_rank) +
         ' / ' + esc(it.bm25_rank === null || it.bm25_rank === undefined ? '-' : it.bm25_rank) + '</td>' +
         '<td class="pre">' + esc((it.text_head||'').slice(0,200)) +
         (it.text_len > 200 ? ' …（共 ' + it.text_len + ' 字）' : '') + '</td></tr>';
  });
  box.innerHTML = h + '</table>';
}

/* ---------------- 评估看板 ---------------- */
let EVAL_LOADED = false;
$('reloadEval').onclick = () => { EVAL_LOADED = false; loadEval(); };

async function loadEval(){
  if (EVAL_LOADED) { return; }
  const body = $('evalBody');
  body.innerHTML = '<div class="card"><span class="muted">读取中…</span></div>';
  const r = await fetch('/api/eval');
  const d = await r.json();
  EVAL_LOADED = true;
  renderEvalFiles(d.files || {});
  if (!d.ready) {
    body.innerHTML = '<div class="card"><h2>数据未就绪</h2>' +
      '<p class="refused">' + esc(d.reason || '评估产物缺失') + '</p>' +
      '<p class="muted">评测官产出后本页自动可见。生成命令：</p>' +
      '<div class="art">' + esc(d.command || '') + '</div></div>';
    return;
  }
  $('evalStamp').textContent = '读取于 ' + d.generated_at + '　|　' + d.count + ' 题';
  let h = '';

  /* 汇总指标 */
  const s = d.summary || {};
  const acc = s['准确率'] || {}, rec = s['检索召回'] || {}, rt = s['响应时间ms'] || {};
  const sla = s['SLA'] || {}, ref = s['拒答'] || {}, leak = s['跨语料泄漏'] || {};
  const pg = s['页命中'] || {}, reach = s['索引可达性'] || {}, cfg = s['配置'] || {};
  h += '<div class="card"><h2>汇总指标（源：' + esc('测试/测试数据/评估汇总.json') +
       (s['生成时间'] ? '，生成于 ' + esc(s['生成时间']) : '') + '）</h2><div class="grid">';
  const cards = [
    ['评估题数', (s['有效题数'] !== undefined ? s['有效题数'] + '/' + s['总数'] + ' 有效' : '-')],
    ['严格判对', (acc['严格判对题数'] !== undefined ? acc['严格判对题数'] + ' 题（' + pctf(acc['严格准确率']) + '）' : '-')],
    ['部分正确', (acc['部分正确题数'] !== undefined ? acc['部分正确题数'] + ' 题' : '-')],
    ['宽松准确率', pctf(acc['宽松准确率(严格+部分)'])],
    ['平均关键词覆盖率', pctf(acc['平均关键词覆盖率'])],
    ['平均 recall@5', num(rec['平均recall@5'], 4)],
    ['平均 recall@10', num(rec['平均recall@10'], 4)],
    ['页命中率', pctf(pg['命中率'])],
    ['拒答', (ref['拒答题数'] !== undefined ? ref['拒答题数'] + ' 题（' + (ref['拒答题号']||[]).join('、') + '）' : '-')],
    ['跨语料泄漏', (leak['题数'] !== undefined ? leak['题数'] + ' 题（' + (leak['题号']||[]).join('、') + '）' : '-')],
    ['SLA 达标率', pctf(sla['达标率']) + '（阈值 ' + (sla['阈值ms']||'-') + ' ms）'],
    ['总耗时 平均/P90/P95', num(rt['平均'],1) + ' / ' + num(rt['P90'],1) + ' / ' + num(rt['P95'],1) + ' ms'],
    ['检索 平均 / 生成 平均', num(rt['检索_平均'],1) + ' / ' + num(rt['生成_平均'],1) + ' ms'],
    ['首字 平均 / 最大', num(rt['首字_平均'],1) + ' / ' + num(rt['首字_最大'],1) + ' ms'],
    ['索引规模', (cfg['索引chunk_total']||'-') + ' 块 · run ' + (cfg['索引run_id']||'-')],
    ['模型', (cfg['backend']||'-') + ' / ' + (cfg['模型']||'-') + ' / ' + (cfg['embed_model']||'-')],
    ['索引内单块可达率', pctf(reach['单块可达率']) + '（' + (reach['索引内可达']||'-') + '/' + (reach['关键词总数']||'-') + '）'],
    ['top_k / 阈值', (cfg['top_k']||'-') + ' / 覆盖 ' + (cfg['关键词阈值']||'-') + '，部分 ' + (cfg['部分正确阈值']||'-')],
  ];
  cards.forEach(([k,v]) => { h += '<div class="kv"><div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + '</div></div>'; });
  h += '</div></div>';

  /* 逐题表 */
  h += '<div class="card"><h2>逐题明细（' + d.count + ' 题）</h2><div class="scroll"><table>' +
       '<tr><th>题号</th><th>题型</th><th>公司</th><th>问题</th><th>答案</th><th>关键词覆盖率</th>' +
       '<th>是否判对</th><th>recall@5</th><th>recall@10</th><th>耗时 ms</th><th>跨语料泄漏</th><th>详情</th></tr>';
  d.rows.forEach((r2, i) => {
    const verdict = r2.is_correct ? '<span class="tag ok">判对</span>'
      : (r2.partial_correct ? '<span class="tag mid">部分正确</span>' : '<span class="tag no">未判对</span>');
    const leakTxt = r2.leak ? '<span class="tag no">泄漏</span>' : '<span class="tag ok">无</span>';
    h += '<tr><td><b>' + esc(r2.id) + '</b></td><td>' + esc(r2.question_type) + '</td>' +
         '<td>' + esc(r2.company) + '</td>' +
         '<td>' + esc((r2.question||'').slice(0,110)) + ((r2.question||'').length>110?'…':'') + '</td>' +
         '<td>' + esc((r2.answer||'').slice(0,150)) + ((r2.answer||'').length>150?'…':'') + '</td>' +
         '<td>' + num(r2.coverage, 3) + '<span class="muted"> (' + esc(r2.keyword_hit) + '/' + esc(r2.keyword_total) + ')</span></td>' +
         '<td>' + verdict + '</td><td>' + num(r2.recall5, 3) + '</td><td>' + num(r2.recall10, 3) + '</td>' +
         '<td>' + num(r2.total_ms, 0) + '</td><td>' + leakTxt + '</td>' +
         '<td><details><summary>展开</summary><div class="pre" style="max-width:560px">' +
         '<b>完整问题</b>：' + esc(r2.question) + '\n' +
         '<b>完整答案</b>：' + esc(r2.answer) + '\n' +
         '<b>引用页</b>：' + esc(JSON.stringify(r2.cited_pages)) + '　期望页提示 ' + esc(r2.expected_pages_hint||'-') + '\n' +
         '<b>拒答</b>：' + esc(r2.refused) + (r2.refusal_reason ? '（' + esc(r2.refusal_reason) + '）' : '') + '\n' +
         '<b>分段耗时</b>：检索 ' + num(r2.retrieve_ms,2) + ' / 生成 ' + num(r2.generate_ms,2) +
         ' / 首字 ' + num(r2.first_token_ms,2) + ' ms\n' +
         '<b>生成方式</b>：' + esc(r2.generation_method) + '\n' +
         (r2.leak_files && r2.leak_files.length ? '<b>泄漏文件</b>：' + esc(r2.leak_files.join('、')) + '\n' : '') +
         '<b>Top-' + (r2.hits||[]).length + ' 命中</b>：\n' +
         (r2.hits||[]).map(x => '  ' + x.rank + '. ' + x.chunk_id + ' p' + x.page + ' [' + x.type + '] ' +
            num(x.final_score,6) + '  ' + (x.text_head||'').slice(0,80) + '…').join('\n') +
         '</div></details></td></tr>';
  });
  h += '</table></div></div>';

  /* 分组统计 */
  const groups = [['按题型分组','按题型分组'], ['按公司分组','按公司分组'], ['按生成方式分组','按生成方式分组']];
  groups.forEach(([k, label]) => {
    const g = s[k]; if (!g || typeof g !== 'object') { return; }
    const keys = Object.keys(g); if (!keys.length) { return; }
    const cols = Object.keys(g[keys[0]] || {});
    h += '<div class="card"><h2>' + esc(label) + '</h2><table><tr><th>分组</th>' +
         cols.map(c => '<th>' + esc(c) + '</th>').join('') + '</tr>';
    keys.forEach(kk => {
      h += '<tr><td><b>' + esc(kk) + '</b></td>' + cols.map(c => {
        const v = (g[kk]||{})[c];
        return '<td>' + esc(typeof v === 'object' ? JSON.stringify(v) : v) + '</td>';
      }).join('') + '</tr>';
    });
    h += '</table></div>';
  });

  /* 检索问题分析摘要 */
  if (d.analysis) {
    h += '<div class="card"><h2>检索问题分析.md 摘要（源：' + esc(d.analysis.path) +
         '，共 ' + d.analysis.total_sections + ' 个二级章节，此处摘录 二/四/五）</h2>';
    h += d.analysis.preamble_html;
    d.analysis.sections.forEach(sec => {
      h += '<h3>' + esc(sec.title) + '</h3>' + sec.html;
    });
    h += '</div>';
  } else {
    h += '<div class="card"><h2>检索问题分析.md</h2><span class="muted">未找到该文件。</span></div>';
  }
  body.innerHTML = h;
}

function renderEvalFiles(files){
  const box = $('evalFiles');
  box.innerHTML = Object.keys(files).map(k => {
    const f = files[k];
    return '<div class="kv"><div class="k">' + esc(k) + '</div><div class="v" style="font-size:12.5px">' +
      (f.exists ? esc(f.rel) + '<br /><span class="muted">' + (f.bytes/1024).toFixed(1) + ' KB · ' + esc(f.mtime) + '</span>'
                : '<span class="refused">缺失：' + esc(f.rel) + '</span>') + '</div></div>';
  }).join('');
}

/* ---------------- 测试流程演示 ---------------- */
let FLOW = null;
$('playFlow').onclick = async () => { await ensureFlow(); playFlow(); };
$('showAll').onclick = async () => { await ensureFlow(); renderFlow(FLOW.steps.length); };
$('resetFlow').onclick = async () => { await ensureFlow(); renderFlow(0); };

async function ensureFlow(){
  if (FLOW) { return FLOW; }
  const r = await fetch('/api/flow');
  FLOW = await r.json();
  return FLOW;
}

async function loadFlow(){
  const body = $('flowBody');
  if (FLOW) { return; }
  body.innerHTML = '<div class="card"><span class="muted">读取中…</span></div>';
  await ensureFlow();
  renderFlow(0);
}

function renderFlow(visible){
  const body = $('flowBody');
  body.innerHTML = FLOW.steps.map((st, i) => {
    const dim = i < visible ? '' : ' dim';
    let h = '<div class="step' + dim + '" id="step' + i + '">';
    h += '<div class="head"><div class="num">' + st.no + '</div><div class="ttl">' + esc(st.title) + '</div>' +
         '<div class="sub">' + esc(st.subtitle) + '</div>' +
         (st.ready ? '<span class="tag ok">产物就绪</span>' : '<span class="tag no">产物缺失</span>') + '</div>';
    if (i < visible) {
      h += '<div class="grid" style="margin-top:10px">' +
        st.stats.map(x => '<div class="kv"><div class="k">' + esc(x.k) + '</div><div class="v">' + esc(x.v) + '</div></div>').join('') +
        '</div>';
      h += '<h3>真实产物路径</h3>';
      st.artifacts.forEach(a => {
        h += '<div class="art' + (a.exists ? '' : ' miss') + '">' +
             (a.exists ? '✅' : '❌') + ' <b>' + esc(a.label) + '</b>　' + esc(a.rel) +
             (a.exists ? '　<span class="muted">' + (a.bytes !== null && a.bytes !== undefined ? (a.bytes/1024).toFixed(1)+' KB' : '') +
                (a.mtime ? ' · ' + esc(a.mtime) : '') + '</span>'
              : '　<span class="refused">文件不存在</span>') + '</div>';
      });
      h += '<div class="muted" style="margin-top:8px">' + esc(st.note) + '</div>';
    }
    h += '</div>';
    return h;
  }).join('');
}

async function playFlow(){
  renderFlow(0);
  for (let i = 0; i <= FLOW.steps.length; i++) {
    renderFlow(i);
    if (i < FLOW.steps.length) {
      const el = $('step' + i);
      if (el) { el.scrollIntoView({behavior:'smooth', block:'center'}); }
      await new Promise(res => setTimeout(res, 1200));
    }
  }
}
</script>
</body>
</html>
"""


def render_page(port: int) -> str:
    return (PAGE_HTML
            .replace("__TITLE__", TITLE)
            .replace("__WORK_ORDER__", WORK_ORDER)
            .replace("__SUBTITLE__", SUBTITLE)
            .replace("__REPO_ROOT__", str(REPO_ROOT))
            .replace("__TOPK__", str(DEFAULT_TOP_K))
            .replace("__PORT__", str(port)))


# --------------------------------------------------------------------------
# HTTP 处理
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "Gongdan7TestBench/1.0"
    protocol_version = "HTTP/1.1"
    port = DEFAULT_PORT

    # -- 输出工具 ------------------------------------------------------
    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def _html(self, markup: str, status: int = 200) -> None:
        self._send(markup.encode("utf-8"), "text/html; charset=utf-8", status)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt: str, *args: Any) -> None:      # noqa: A003
        sys.stdout.write("[%s] %s %s\n" % (datetime.now().strftime("%H:%M:%S"),
                                           self.client_address[0], fmt % args))
        sys.stdout.flush()

    # -- 路由 ----------------------------------------------------------
    def do_GET(self) -> None:                                 # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._html(render_page(self.port))
                return
            if path in ("/health", "/api/health"):
                self._json(build_health_payload())
                return
            if path == "/api/eval":
                self._json(build_eval_payload())
                return
            if path == "/api/flow":
                self._json(build_flow_payload())
                return
            self._json({"error": {"code": "E_NOT_FOUND", "message": f"未知路径 {path}"}}, 404)
        except Exception as exc:                              # noqa: BLE001
            self._json({"error": {"code": "E_INTERNAL",
                                  "message": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()[-1500:]}}, 500)

    def do_POST(self) -> None:                                # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/api/ask", "/ask"):
                payload = self._body()
                question = str(payload.get("question") or "").strip()
                lang = str(payload.get("lang") or "auto")
                try:
                    top_k = int(payload.get("top_k") or DEFAULT_TOP_K)
                except (TypeError, ValueError):
                    top_k = DEFAULT_TOP_K
                top_k = max(1, min(20, top_k))
                if not question:
                    self._json({"error": {"code": "E_PARAM_QUESTION_EMPTY",
                                          "message": "question 不得为空"}}, 400)
                    return
                if len(question) > 500:
                    self._json({"error": {"code": "E_PARAM_QUESTION_TOOLONG",
                                          "message": "question 长度须 ≤500"}}, 400)
                    return
                sys.stdout.write("[%s] ask lang=%s top_k=%d q=%s\n"
                                 % (datetime.now().strftime("%H:%M:%S"), lang, top_k, question[:70]))
                sys.stdout.flush()
                result = ENGINE.ask(question, lang=lang, top_k=top_k)
                self._json(result)
                return
            self._json({"error": {"code": "E_NOT_FOUND", "message": f"未知路径 {path}"}}, 404)
        except Exception as exc:                              # noqa: BLE001
            self._json({"error": {"code": "E_INTERNAL",
                                  "message": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()[-1500:]}}, 500)


def build_server(port: int, *, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"port": int(port)})
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"[{WORK_ORDER}] 测试评估台（纯标准库 http.server，零第三方依赖）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--no-warmup", action="store_true",
                        help="不在后台预热 QAEngine（首个提问会明显变慢）")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")               # type: ignore[union-attr]
    except Exception:                                          # noqa: BLE001
        pass

    print("=" * 78)
    print(f"[{WORK_ORDER}] {TITLE}")
    print(f"仓库    ：{REPO_ROOT}")
    print(f"解释器  ：{sys.executable}")
    print(f"评估产物：{_rel(P_RESULTS)} → {'就绪' if P_RESULTS.is_file() else '缺失'}")
    print(f"索引    ：{_rel(P_MANIFEST)} → {'就绪' if P_MANIFEST.is_file() else '缺失'}")
    print("=" * 78)

    if not args.no_warmup:
        print("[预热] 后台加载索引 / 分词 / 嵌入 / 生成模型（约 20~30 s）…", flush=True)
        ENGINE.start_warmup(with_generate=True)

    try:
        server = build_server(args.port, host=args.host)
    except OSError as exc:
        print(f"[FATAL] 无法绑定 {args.host}:{args.port} —— {exc}", file=sys.stderr)
        print(f"        端口可能被占用；换端口：--port 7862", file=sys.stderr)
        return 2

    url = f"http://{args.host}:{args.port}/"
    print(f"[OK] 测试评估台已启动：{url}  （Ctrl+C 停止）", flush=True)
    print(f"     健康检查：{url}api/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止] 收到 Ctrl+C", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
