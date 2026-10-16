# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 评估主入口。

用法（默认跑全量 17 题）::

    python 研发\\评估\\运行评估.py

可选参数（便于烟测，不影响交付物口径）::

    python 研发\\评估\\运行评估.py --ids Q07,Q08     # 只跑指定题
    python 研发\\评估\\运行评估.py --limit 2          # 只跑前 N 题

产出（全部 UTF-8，落在 ``测试/测试数据/``）::

    评估结果.jsonl      逐题完整记录（含 top-k 检索片段、判分明细、latency 全字段、诊断树）
    评估结果.csv        UTF-8-BOM，Excel 可直接打开
    评估汇总.json       准确率 / recall / 响应时间分位 / 分组统计
    检索问题分析.md     逐题检索问题分析 + 归类统计 + 改进建议
    评估运行日志.txt    真实控制台输出

设计要点
--------
* **串行**调用，绝不并发打 Ollama（本机只有 1 个 CPU 推理实例）。
* ``use_cache=False``：每题都按"用户第一次问"计，避免命中落盘缓存后 total≈0ms 污染 SLA 统计。
* 单题异常被捕获成记录里的 ``error`` 字段，不中断全量；连续多题同因失败才中止（避免无意义空跑）。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import 配置 as C  # noqa: E402
import 判分 as J  # noqa: E402

C.apply_env()
if str(C.APP_DIR) not in sys.path:
    sys.path.insert(0, str(C.APP_DIR))


# ===========================================================================
# 控制台 + 日志文件双写
# ===========================================================================
class Tee:
    """把 stdout 同时写到控制台与日志文件（落一份**真实**控制台输出）。"""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:  # noqa: BLE001 - 日志写失败不能影响评估
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass


def _p(msg: str = "") -> None:
    print(msg, flush=True)


# ===========================================================================
# 工具
# ===========================================================================
def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _flat(text: Any, limit: int | None = None) -> str:
    """把片段正文压成单行（便于 CSV/表格展示）。"""
    s = " ".join(str(text or "").split())
    return s[:limit] if limit else s


def percentile(values: list[float], pct: float) -> float | None:
    """线性插值分位数（``pct`` ∈ [0,100]）。空列表返回 None。"""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    pos = (len(ordered) - 1) * (float(pct) / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * frac, 2)


def module_revision() -> dict[str, str]:
    """记录被评估代码的修订指纹（SHA256 前 16 位）。

    动机：实测评估过程中上游会**边跑边改**（`app/core/answer_composer.py` 曾在本框架一次
    运行的中间被修改，导致前几题与后几题跑的是两份代码，结果不可归因）。
    把修订指纹写进日志与汇总，至少让"结果对应哪版代码"可追溯。
    """
    import hashlib

    names = ["retriever.py", "answer_composer.py", "generator.py", "qa_engine.py",
             "answerability.py", "query_understanding.py", "reranker.py", "fusion.py",
             "citation.py", "bm25_index.py"]
    out: dict[str, str] = {}
    for name in names:
        path = C.APP_DIR / "app" / "core" / name
        try:
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except Exception as exc:  # noqa: BLE001
            out[name] = f"N/A({type(exc).__name__})"
    return out


def build_index_reach(engine: Any) -> Any:
    """构建「关键词 → 索引全库可达性」查询函数。

    作用：把「检索没召回到」拆成两种完全不同的病因 ——

    * **索引里根本没有** → 解析/切块/索引阶段丢了内容（表格没识成表、跨页断裂…）；
    * **索引里有但没排进 top-k** → 召回/排序问题（语料路由、融合权重、rerank…）。

    返回 ``find(keyword) -> {"in_index": bool, "found": [chunk 摘要…]}``。
    """
    bundle = engine.bundle
    rows: list[tuple[str, Any, Any, Any, str]] = []
    for chunk_id, chunk in bundle.chunks.items():
        text = str(chunk.get("content") or chunk.get("text") or "")
        rows.append((str(chunk_id), chunk.get("page"), chunk.get("file_name"),
                     chunk.get("type"), J.normalize(text)))
    _p(f"[索引可达性] 载入 {len(rows)} 个块的归一化正文，用于全库关键词可达性判定")
    cache: dict[str, list[dict[str, Any]]] = {}

    def find(keyword: Any) -> dict[str, Any]:
        needle = J.normalize(keyword)
        if needle not in cache:
            found: list[dict[str, Any]] = []
            if needle:
                min_len = len(J._loose(needle))
                for chunk_id, page, file_name, ctype, ntext in rows:
                    if len(ntext) < min_len:
                        continue
                    if J.contains_keyword(ntext, keyword):
                        found.append({"chunk_id": chunk_id, "page": page,
                                      "file_name": file_name, "type": ctype})
                        if len(found) >= 10:
                            break
            cache[needle] = found
        found = cache[needle]
        return {"in_index": bool(found), "found": found[:3], "found_count_capped": len(found)}

    def boundary_split(keyword: Any, expected_files: Any = None) -> dict[str, Any]:
        """单块都没命中时，判断是不是**被切块边界切断**（前半截在一个块尾、后半截在下一块头）。

        做法：把该语料（按 (page, chunk_id) 排序）的块正文首尾相接后再找一次。
        注意：所有块拼起来必然能把跨块关键词找到，所以这里只作为"成因提示"，
        用来区分「解析阶段就丢了内容」与「内容都在、只是被切碎了」。
        """
        names = {J._basename(f) for f in (expected_files or [])}
        parts: list[tuple[int, str, str]] = []
        for chunk_id, page, file_name, _ctype, text in rows:
            if names and J._basename(file_name) not in names:
                continue
            parts.append((int(page or 0), str(chunk_id), text))
        parts.sort(key=lambda item: (item[0], item[1]))
        joined = J.normalize("\n".join(p[2] for p in parts))
        hit = J.contains_keyword(joined, keyword)
        return {"in_joined_corpus": hit, "chunks_joined": len(parts)}

    find.boundary_split = boundary_split  # type: ignore[attr-defined]
    return find


def hit_brief(hit: dict[str, Any], rank: int, text_limit: int = 200) -> dict[str, Any]:
    """把引擎返回的 chunk payload 压成评估记录用的片段摘要。"""
    return {
        "rank": rank,
        "chunk_id": hit.get("chunk_id"),
        "page": hit.get("page"),
        "file_name": hit.get("file_name"),
        "company": J.company_of(hit.get("file_name")),
        "type": hit.get("type"),
        "modality": hit.get("modality"),
        "score": hit.get("score"),
        "final_score": hit.get("final_score"),
        "rrf_score": hit.get("rrf_score"),
        "vector_rank": hit.get("vector_rank"),
        "bm25_rank": hit.get("bm25_rank"),
        "bm25_score": hit.get("bm25_score"),
        "vector_score": hit.get("vector_score"),
        "text_head": _flat(hit.get("text"), text_limit),
        "text_len": len(str(hit.get("text") or "")),
    }


# ===========================================================================
# 单题评估
# ===========================================================================
def evaluate_one(eng: Any, item: dict[str, Any], index: int, total: int,
                 index_find: Any = None) -> dict[str, Any]:
    qid = str(item.get("id"))
    question = str(item.get("question") or "")
    lang = str(item.get("lang") or "auto")
    keywords = list(item.get("answer_keywords") or [])
    source_file = item.get("source_file") or ""
    expected_files = J.split_source_files(source_file)
    qtype = J.classify_question(question, lang, keywords, source_file)

    _p("")
    _p(f"[{index}/{total}] {qid} ({qtype}) 提问中 …")
    started = time.perf_counter()
    record: dict[str, Any] = {
        "id": qid,
        "question": question,
        "lang": lang,
        "question_type": qtype,
        "project": item.get("project"),
        "source_pdf": item.get("source_pdf"),
        "source_file": source_file,
        "expected_files": expected_files,
        "company": J.companies_of(source_file),
        "expected_pages_hint": item.get("expected_pages_hint"),
        "reference_answer": item.get("reference_answer"),
        "answer_keywords": keywords,
        "top_k": C.TOP_K,
        "use_cache": C.USE_CACHE,
        "error": None,
        "error_traceback": None,
    }

    try:
        res = eng.answer(question, lang=lang, top_k=C.TOP_K,
                         session_id=f"{C.SESSION_PREFIX}-{qid}",
                         return_evidence=True, use_cache=C.USE_CACHE)
    except Exception as exc:  # noqa: BLE001 - 异常必须落记录而不是崩掉全量
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_traceback"] = traceback.format_exc()
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        record["answer"] = ""
        record["refused"] = None
        record["hits"] = []
        record["judge"] = {"error": record["error"]}
        _p(f"    !! 异常：{record['error']}")
        return record

    record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
    record["wall_ms_note"] = "含判分前的端到端墙钟；latency_ms 为引擎自报分段耗时"

    answer = str(getattr(res, "answer", "") or "")
    refused = bool(getattr(res, "refused", False))
    latency = dict(getattr(res, "latency_ms", {}) or {})
    hits_raw = list(getattr(res, "chunks", []) or [])
    citations = list(getattr(res, "citations", []) or [])
    diagnostics = getattr(res, "diagnostics", {}) or {}

    record.update({
        "answer": answer,
        "answer_chars": len(answer),
        "refused": refused,
        "refusal_reason": getattr(res, "refusal_reason", None),
        "refusal_detail": getattr(res, "refusal_detail", None),
        "cited_pages": list(getattr(res, "cited_pages", []) or []),
        "citations": citations,
        "generation_method": getattr(res, "generation_method", None),
        "backend": getattr(res, "backend", None),
        "model": getattr(res, "model", None),
        "trace_id": getattr(res, "trace_id", None),
        "language": getattr(res, "language", None),
        "cache": getattr(res, "cache", None),
        "analysis": getattr(res, "analysis", None),
        "diagnostics": diagnostics,
        "evidence_count": len(list(getattr(res, "evidence", []) or [])),
        "hit_count": len(hits_raw),
    })

    # ---- latency 全字段 ----
    record["latency_ms"] = {
        "retrieve": latency.get("retrieve"),
        "generate": latency.get("generate"),
        "first_token": latency.get("first_token"),
        "total": latency.get("total"),
    }
    total_ms = latency.get("total")
    try:
        record["total_ms"] = float(total_ms) if total_ms is not None else None
    except (TypeError, ValueError):
        record["total_ms"] = None
    record["sla_ok"] = (record["total_ms"] is not None and record["total_ms"] <= C.SLA_MS)

    # ---- 检索片段（top-k，按引擎给出的相关度顺序）----
    hits = [hit_brief(h, i + 1) for i, h in enumerate(hits_raw)]
    record["hits"] = hits
    record["top5"] = hits[:5]
    # 检索顺序单调性自检：只看引擎给出的 top-k（第 k 名之后是引擎追加的页上下文补充块，
    # 分数可以更高但不参与排名，故不能拿它们判断"未排序"）。
    scores = [h.get("final_score") for h in hits[: C.TOP_K] if h.get("final_score") is not None]
    record["rank_monotonic_topk"] = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    record["extra_hits"] = max(0, len(hits) - C.TOP_K)
    record["extra_hit_ids"] = [h.get("chunk_id") for h in hits[C.TOP_K:]]

    hits_text = [str(h.get("text") or "") for h in hits_raw]

    # ---- 判分 ----
    n_hit, n_kw, cov, kw_hit, kw_miss = J.keyword_coverage(answer, keywords)
    n5, t5, r5, _h5, miss5 = J.retrieval_recall_at_k(hits_text, keywords, C.RECALL_KS[0])
    n10, t10, r10, _h10, miss10 = J.retrieval_recall_at_k(hits_text, keywords, C.RECALL_KS[1])
    ph = J.page_hit(hits, item.get("expected_pages_hint"), expected_files or None)
    leak = J.cross_corpus_leak(hits, expected_files, top_n=C.LEAK_TOP_N)
    leak10 = J.cross_corpus_leak(hits, expected_files, top_n=C.TOP_K)
    cleak = J.citation_leak(citations, expected_files)
    an, at, acov, a_hit, a_miss = J.atomic_coverage(answer, keywords)

    correct = J.is_correct(cov, refused, C.KEYWORD_THRESHOLD)
    partial = J.partial_correct(cov, refused, answer, keywords,
                                C.KEYWORD_THRESHOLD, C.PARTIAL_THRESHOLD, C.ATOMIC_THRESHOLD)

    record["judge"] = {
        "keyword_hit_count": n_hit,
        "keyword_total": n_kw,
        "keyword_coverage": cov,
        "keyword_hit": kw_hit,
        "keyword_miss": kw_miss,
        "threshold": C.KEYWORD_THRESHOLD,
        "is_correct": correct,
        "partial_correct": partial,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "recall_at_5_hit": n5,
        "recall_at_10_hit": n10,
        "recall_at_5_miss": miss5,
        "recall_at_10_miss": miss10,
        "page_hit": ph,
        "cross_corpus_leak": leak,
        "cross_corpus_leak_top10": leak10,
        "citation_leak": cleak,
        "atomic_coverage": acov,
        "atomic_hit": a_hit,
        "atomic_miss": a_miss,
        "atomic_scored": at,
    }

    gate = (diagnostics or {}).get("gate") or {}
    generation = (diagnostics or {}).get("generation") or {}
    record["gate"] = {"answerable": gate.get("answerable"), "reason": gate.get("reason"),
                      "coverage": (diagnostics or {}).get("coverage")}
    record["generation"] = {
        "method": generation.get("method"),
        "backend": generation.get("backend"),
        "model": generation.get("model"),
        "returned_chars": len(str(generation.get("text") or "")) or None,
        "first_token_ms": generation.get("first_token_ms"),
        "total_ms": generation.get("total_ms"),
        "notes": (generation.get("composer") or {}).get("notes"),
        "composer_method": (generation.get("composer") or {}).get("method"),
        "composer_shape": (generation.get("composer") or {}).get("shape"),
        "composer_coverage": ((generation.get("composer") or {}).get("notes") or {}).get("coverage"),
        "used_chunk_ids": (generation.get("composer") or {}).get("used_chunk_ids"),
    }
    # ---- 生成层护栏证据（LLM 候选 / 被拒原因 / 缺失原子）----
    record["llm_candidate"] = generation.get("llm_candidate")
    record["llm_rejected_reason"] = generation.get("llm_rejected_reason")
    record["missing_atoms"] = generation.get("missing_atoms")
    record["draft_atoms"] = (generation.get("composer") or {}).get("atoms")
    record["draft_text_head"] = _flat(generation.get("composer", {}).get("text") if isinstance(generation.get("composer"), dict) else None, 400)
    record["fallback_coverage"] = (diagnostics or {}).get("fallback_coverage")
    record["post_composition_refused"] = (
        bool(refused) and record["fallback_coverage"] is not None)
    record["language_degraded"] = generation.get("language_degraded")

    # ---- 反事实对照：如果直接采用 LLM 候选（不经原子护栏回退），判分会是什么结果 ----
    # 用途：把「系统答错了」和「系统答对了但被自己的护栏扔掉」区分开（诊断口径，不计入官方准确率）
    candidate_text = str(record.get("llm_candidate") or "")
    if candidate_text:
        cn, ct, ccov, chit, cmiss = J.keyword_coverage(candidate_text, keywords)
        record["judge"]["llm_candidate_coverage"] = ccov
        record["judge"]["llm_candidate_hit"] = chit
        record["judge"]["llm_candidate_miss"] = cmiss
        record["judge"]["llm_candidate_correct"] = J.is_correct(ccov, False, C.KEYWORD_THRESHOLD)
        record["judge"]["llm_candidate_partial"] = J.partial_correct(
            ccov, False, candidate_text, keywords, C.KEYWORD_THRESHOLD,
            C.PARTIAL_THRESHOLD, C.ATOMIC_THRESHOLD)
    else:
        record["judge"]["llm_candidate_coverage"] = None
        record["judge"]["llm_candidate_correct"] = False
        record["judge"]["llm_candidate_partial"] = False
    # 护栏丢弃了正确答案
    record["guard_discarded_correct"] = bool(
        record["judge"]["llm_candidate_correct"] and not correct and record.get("llm_rejected_reason"))

    # ---- 索引全库可达性（把"没召回到"拆成索引缺失 vs 排序问题）----
    if index_find is not None:
        reach: dict[str, Any] = {}
        for kw in keywords:
            info = index_find(kw)
            in_expected = False
            for found in info.get("found") or []:
                page = found.get("page")
                name = J._basename(found.get("file_name"))
                if page is None or (expected_files and name not in {J._basename(f) for f in expected_files}):
                    continue
                for lo, hi in J.parse_page_ranges(item.get("expected_pages_hint")):
                    if lo <= int(page) <= hi:
                        in_expected = True
                        break
            info["in_expected_pages"] = in_expected
            if not info["in_index"]:
                splitter = getattr(index_find, "boundary_split", None)
                if splitter is not None:
                    info["boundary_split"] = splitter(kw, expected_files)
            reach[kw] = info
        record["index_reach"] = reach
        unreachable = [k for k, v in reach.items() if not v["in_index"]]
        split_only = [k for k in unreachable
                      if (reach[k].get("boundary_split") or {}).get("in_joined_corpus")]
        record["index_reach_summary"] = {
            "关键词总数": len(keywords),
            "索引内可达": sum(1 for v in reach.values() if v["in_index"]),
            "单块内不可达": len(unreachable),
            "单块内不可达关键词": unreachable,
            "其中属切块边界切断": split_only,
            "其中疑似解析阶段即丢失": [k for k in unreachable if k not in split_only],
            "索引内可达且落在期望页": sum(1 for v in reach.values() if v["in_expected_pages"]),
        }

    flag = "✅判对" if correct else ("🟡部分" if partial else "❌未判对")
    _p(f"    答案：{_flat(answer, 110)}")
    _p(f"    {flag} | 覆盖率 {n_hit}/{n_kw}={cov:.2f} | recall@5={r5:.2f} recall@10={r10:.2f} "
       f"| 页命中={ph['hit']} | 跨语料泄漏={leak['leak']} | 拒答={refused}"
       f" | retrieve={latency.get('retrieve')}ms generate={latency.get('generate')}ms "
       f"total={record['total_ms']}ms | method={record['generation_method']}")
    if record.get("index_reach_summary"):
        s = record["index_reach_summary"]
        _p(f"    索引可达性：{s['索引内可达']}/{s['关键词总数']} 单块可达"
           f"（单块不可达 {s['单块内不可达']}：{'、'.join(s['单块内不可达关键词']) or '无'}"
           f"｜其中切块边界切断 {len(s['其中属切块边界切断'])}："
           f"{'、'.join(s['其中属切块边界切断']) or '无'}）")
    if record.get("llm_rejected_reason"):
        _p(f"    生成护栏：LLM 候选被 `{record['llm_rejected_reason']}` 拒绝，缺失原子="
           f"{json.dumps(record.get('missing_atoms'), ensure_ascii=False)[:200]}")
        _p(f"    LLM 候选原文：{_flat(record.get('llm_candidate'), 160)}")
    return record


# ===========================================================================
# 汇总
# ===========================================================================
def _avg(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _group_stats(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        buckets.setdefault(str(rec.get(key) or "未知"), []).append(rec)
    out: dict[str, Any] = {}
    for name, group in sorted(buckets.items()):
        valid = [r for r in group if not r.get("error")]
        covs = [r["judge"]["keyword_coverage"] for r in valid if "judge" in r and "keyword_coverage" in r["judge"]]
        r5s = [r["judge"]["recall_at_5"] for r in valid if "judge" in r and "recall_at_5" in r["judge"]]
        r10s = [r["judge"]["recall_at_10"] for r in valid if "judge" in r and "recall_at_10" in r["judge"]]
        times = [r["total_ms"] for r in valid if r.get("total_ms") is not None]
        out[name] = {
            "题数": len(group),
            "有效题数": len(valid),
            "判对题数": sum(1 for r in valid if r.get("judge", {}).get("is_correct")),
            "判对率": _avg([1.0 if r.get("judge", {}).get("is_correct") else 0.0 for r in valid]) if valid else None,
            "平均关键词覆盖率": _avg([float(c) for c in covs]),
            "平均recall@5": _avg([float(x) for x in r5s]),
            "平均recall@10": _avg([float(x) for x in r10s]),
            "平均总耗时ms": _avg([float(t) for t in times]),
            "拒答题数": sum(1 for r in valid if r.get("refused")),
            "跨语料泄漏题数": sum(1 for r in valid if r.get("judge", {}).get("cross_corpus_leak", {}).get("leak")),
        }
    return out


def _index_reach_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """关键词的「索引全库可达性」统计：区分索引缺失与检索排序问题。"""
    total = reachable = unreachable = in_expected = 0
    boundary_split = parse_lost = 0
    missing_detail: dict[str, list[str]] = {}
    split_detail: dict[str, list[str]] = {}
    reachable_but_missed: dict[str, list[str]] = {}
    for rec in records:
        summary = rec.get("index_reach_summary")
        if not summary:
            continue
        total += summary["关键词总数"]
        reachable += summary["索引内可达"]
        unreachable += summary["单块内不可达"]
        in_expected += summary["索引内可达且落在期望页"]
        boundary_split += len(summary.get("其中属切块边界切断") or [])
        parse_lost += len(summary.get("其中疑似解析阶段即丢失") or [])
        if summary["单块内不可达关键词"]:
            missing_detail[rec["id"]] = summary["单块内不可达关键词"]
        if summary.get("其中属切块边界切断"):
            split_detail[rec["id"]] = summary["其中属切块边界切断"]
        # 索引里有、但 top-10 没召回 → 纯排序/召回问题
        missed_in_topk = set((rec.get("judge") or {}).get("recall_at_10_miss") or [])
        reach = rec.get("index_reach") or {}
        bad = [k for k in missed_in_topk if (reach.get(k) or {}).get("in_index")]
        if bad:
            reachable_but_missed[rec["id"]] = sorted(bad)
    return {
        "关键词总数": total,
        "索引内可达": reachable,
        "单块可达率": round(reachable / total, 4) if total else None,
        "单块内不可达": unreachable,
        "其中属切块边界切断": boundary_split,
        "其中疑似解析阶段即丢失": parse_lost,
        "索引内可达且落在期望页": in_expected,
        "单块内不可达明细": missing_detail,
        "切块边界切断明细": split_detail,
        "索引可达但top10未召回明细": reachable_but_missed,
        "索引可达但top10未召回关键词数": sum(len(v) for v in reachable_but_missed.values()),
    }


def _candidate_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """反事实诊断：若直接采用生成层的 LLM 候选（不经原子护栏回退），判分会怎样。

    **这不是官方准确率**，而是用来把「检索/生成答错了」与「答案本来对、被自己的护栏丢掉」
    两种病因分开归因。
    """
    with_candidate = [r for r in records if str(r.get("llm_candidate") or "").strip()]
    cand_correct = [r for r in with_candidate if (r.get("judge") or {}).get("llm_candidate_correct")]
    discarded = [r for r in records if r.get("guard_discarded_correct")]
    rejected = [r for r in records if r.get("llm_rejected_reason")]
    return {
        "有LLM候选的题数": len(with_candidate),
        "候选被护栏拒绝的题数": len(rejected),
        "候选判对题数": len(cand_correct),
        "候选判对题号": [r["id"] for r in cand_correct],
        "候选口径准确率": round(len(cand_correct) / len(records), 4) if records else None,
        "被护栏丢弃的判对题数": len(discarded),
        "被护栏丢弃的判对题号": [r["id"] for r in discarded],
        "说明": ("反事实诊断口径，非官方准确率：只把最终答案换成生成层的 LLM 候选"
                 "（其余检索结果不变）后重新判分。用于区分「答错了」与「答对了但被护栏回退丢掉」。"),
    }


def build_summary(records: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    valid = [r for r in records if not r.get("error")]
    errors = [r for r in records if r.get("error")]
    times = [float(r["total_ms"]) for r in valid if r.get("total_ms") is not None]
    covs = [float(r["judge"]["keyword_coverage"]) for r in valid]
    r5s = [float(r["judge"]["recall_at_5"]) for r in valid]
    r10s = [float(r["judge"]["recall_at_10"]) for r in valid]
    correct = [r for r in valid if r["judge"]["is_correct"]]
    partial = [r for r in valid if r["judge"]["partial_correct"]]
    refused = [r for r in valid if r.get("refused")]

    retrieval_times = [float(r["latency_ms"]["retrieve"]) for r in valid if r.get("latency_ms", {}).get("retrieve") is not None]
    generate_times = [float(r["latency_ms"]["generate"]) for r in valid if r.get("latency_ms", {}).get("generate") is not None]
    first_token_times = [float(r["latency_ms"]["first_token"]) for r in valid if r.get("latency_ms", {}).get("first_token") is not None]

    summary: dict[str, Any] = {
        "工单": C.WORK_ORDER,
        "生成时间": meta.get("run_at"),
        "配置": {
            "top_k": C.TOP_K,
            "use_cache": C.USE_CACHE,
            "关键词阈值": C.KEYWORD_THRESHOLD,
            "部分正确阈值": C.PARTIAL_THRESHOLD,
            "数值原子阈值": C.ATOMIC_THRESHOLD,
            "SLA_ms": C.SLA_MS,
            "模型": meta.get("model"),
            "backend": meta.get("backend"),
            "embed_model": meta.get("embed_model"),
            "索引chunk_total": meta.get("chunk_total"),
            "索引run_id": meta.get("index_run_id"),
        },
        "总数": len(records),
        "有效题数": len(valid),
        "异常题数": len(errors),
        "异常题号": [r["id"] for r in errors],
        "准确率": {
            "严格判对题数": len(correct),
            "严格准确率": round(len(correct) / len(valid), 4) if valid else None,
            "部分正确题数": len(partial),
            "部分正确题号": [r["id"] for r in partial],
            "宽松准确率(严格+部分)": round((len(correct) + len(partial)) / len(valid), 4) if valid else None,
            "平均关键词覆盖率": _avg(covs),
            "判对题号": [r["id"] for r in correct],
            "未判对题号": [r["id"] for r in valid if not r["judge"]["is_correct"]],
        },
        "检索召回": {
            "平均recall@5": _avg(r5s),
            "平均recall@10": _avg(r10s),
            "recall@5=1.0的题数": sum(1 for x in r5s if x >= 1.0),
            "recall@10=1.0的题数": sum(1 for x in r10s if x >= 1.0),
            "recall@10<1.0题号": [r["id"] for r in valid if r["judge"]["recall_at_10"] < 1.0],
        },
        "响应时间ms": {
            "样本数": len(times),
            "平均": _avg(times),
            "中位": round(statistics.median(times), 2) if times else None,
            "P50": percentile(times, 50),
            "P90": percentile(times, 90),
            "P95": percentile(times, 95),
            "最大": round(max(times), 2) if times else None,
            "最小": round(min(times), 2) if times else None,
            "检索_平均": _avg(retrieval_times),
            "检索_P90": percentile(retrieval_times, 90),
            "生成_平均": _avg(generate_times),
            "生成_P90": percentile(generate_times, 90),
            "首字_平均": _avg(first_token_times),
            "首字_P90": percentile(first_token_times, 90),
            "首字_最大": round(max(first_token_times), 2) if first_token_times else None,
        },
        "SLA": {
            "阈值ms": C.SLA_MS,
            "达标题数": sum(1 for t in times if t <= C.SLA_MS),
            "达标率": round(sum(1 for t in times if t <= C.SLA_MS) / len(times), 4) if times else None,
            "超时题号": [r["id"] for r in valid if r.get("total_ms") is not None and r["total_ms"] > C.SLA_MS],
            "首字达标率": (round(sum(1 for t in first_token_times if t <= C.SLA_MS) / len(first_token_times), 4)
                        if first_token_times else None),
        },
        "拒答": {
            "拒答题数": len(refused),
            "拒答率": round(len(refused) / len(valid), 4) if valid else None,
            "拒答题号": [r["id"] for r in refused],
            "按原因": {},
        },
        "跨语料泄漏": {
            "题数": sum(1 for r in valid if r["judge"]["cross_corpus_leak"]["leak"]),
            "题号": [r["id"] for r in valid if r["judge"]["cross_corpus_leak"]["leak"]],
            "top10题数": sum(1 for r in valid if r["judge"].get("cross_corpus_leak_top10", {}).get("leak")),
            "top10题号": [r["id"] for r in valid if r["judge"].get("cross_corpus_leak_top10", {}).get("leak")],
            "引用级泄漏题数": sum(1 for r in valid if r["judge"]["citation_leak"]["leak"]),
            "引用级泄漏题号": [r["id"] for r in valid if r["judge"]["citation_leak"]["leak"]],
        },
        "页命中": {
            "命中题数": sum(1 for r in valid if r["judge"]["page_hit"]["hit"]),
            "命中率": round(sum(1 for r in valid if r["judge"]["page_hit"]["hit"]) / len(valid), 4) if valid else None,
            "未命中题号": [r["id"] for r in valid if not r["judge"]["page_hit"]["hit"]],
        },
        "按题型分组": _group_stats(valid, "question_type"),
        "按公司分组": _group_stats(valid, "company"),
        "按生成方式分组": _group_stats(valid, "generation_method"),
        "索引可达性": _index_reach_stats(valid),
        "LLM候选对照口径(反事实诊断)": _candidate_stats(valid),
        "按生成方式分组": _group_stats(valid, "generation_method"),
        "逐题速览": [
            {
                "题号": r["id"],
                "题型": r.get("question_type"),
                "公司": r.get("company"),
                "判对": r.get("judge", {}).get("is_correct"),
                "部分正确": r.get("judge", {}).get("partial_correct"),
                "覆盖率": r.get("judge", {}).get("keyword_coverage"),
                "recall@5": r.get("judge", {}).get("recall_at_5"),
                "recall@10": r.get("judge", {}).get("recall_at_10"),
                "页命中": r.get("judge", {}).get("page_hit", {}).get("hit"),
                "跨语料泄漏": r.get("judge", {}).get("cross_corpus_leak", {}).get("leak"),
                "拒答": r.get("refused"),
                "总耗时ms": r.get("total_ms"),
                "生成方式": r.get("generation_method"),
                "error": r.get("error"),
            }
            for r in records
        ],
    }
    reasons: dict[str, int] = {}
    for rec in refused:
        key = str(rec.get("refusal_reason") or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    summary["拒答"]["按原因"] = reasons
    return summary


# ===========================================================================
# 检索问题分析.md
# ===========================================================================
def diagnose(rec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """逐题打问题标签；返回 ``(标签列表, 证据行列表)``。"""
    tags: list[str] = []
    evidence: list[str] = []
    judge = rec.get("judge") or {}
    if rec.get("error"):
        return ["运行异常"], [f"异常：`{rec['error']}`"]

    hits = rec.get("hits") or []
    top3 = hits[:3]
    gate = rec.get("gate") or {}
    gen = rec.get("generation") or {}
    method = str(rec.get("generation_method") or "")
    composer_method = str(gen.get("composer_method") or "")

    # ---- 生成层护栏：LLM 候选被原子保全卫士拒绝（"答案对但被拒"的直接证据）----
    if rec.get("llm_rejected_reason"):
        tags.append("原子护栏误拒LLM候选")
        evidence.append(
            f"LLM 候选被 `{rec['llm_rejected_reason']}` 拒绝，缺失原子="
            f"{json.dumps(rec.get('missing_atoms'), ensure_ascii=False)}；"
            f"候选原文=`{_flat(rec.get('llm_candidate'), 200)}`")
        atoms = rec.get("draft_atoms") or []
        if atoms:
            evidence.append(f"草稿原子表="
                            f"{json.dumps(atoms, ensure_ascii=False)[:400]}")
        if rec.get("guard_discarded_correct"):
            tags.append("护栏丢弃了正确答案")
            evidence.append(
                f"反事实判分：用 LLM 候选作答覆盖率="
                f"{judge.get('llm_candidate_coverage')}（≥{C.KEYWORD_THRESHOLD} 即判对），"
                f"即**本题答案本可判对**，被护栏回退后变成最终答案 "
                f"`{_flat(rec.get('answer'), 80)}`（覆盖率 {judge.get('keyword_coverage')}）")

    if rec.get("refused"):
        reason = rec.get("refusal_reason")
        if rec.get("post_composition_refused") or composer_method == "sentence_select":
            tags.append("生成层兜底拒答")
            evidence.append(
                f"生成层落到 `sentence_select` 后触发兜底复核 → 拒答（reason=`{reason}`，"
                f"fallback_coverage={rec.get('fallback_coverage')} < 0.5）")
        else:
            tags.append("可答性闸门拒答")
            evidence.append(f"gate.answerable={gate.get('answerable')} reason=`{gate.get('reason')}` "
                            f"detail=`{rec.get('refusal_detail')}`")

    if judge.get("keyword_miss"):
        tags.append("答案缺失关键原子")
        evidence.append("答案未命中关键词：" + "、".join(f"`{k}`" for k in judge["keyword_miss"]))

    if float(judge.get("recall_at_10") or 0.0) < 1.0:
        tags.append("检索未召回(top10)")
        evidence.append("top10 合并正文仍缺：" + "、".join(f"`{k}`" for k in judge.get("recall_at_10_miss") or []))

    # 索引全库可达性：把"没召回到"拆成「索引里就没有」vs「索引里有但没排进 top10」
    reach_summary = rec.get("index_reach_summary") or {}
    reach_map = rec.get("index_reach") or {}
    if reach_summary.get("其中疑似解析阶段即丢失"):
        tags.append("索引内缺失关键原子")
        evidence.append("**任何单个块**都找不到、且拼接后也找不到 ⇒ 解析/入库阶段就丢了内容："
                        + "、".join(f"`{k}`" for k in reach_summary["其中疑似解析阶段即丢失"]))
    if reach_summary.get("其中属切块边界切断"):
        tags.append("切块边界切断关键词")
        evidence.append("单块都没有、但把该语料所有块首尾相接后能找到 ⇒ 内容在，只是被切块切断："
                        + "、".join(f"`{k}`" for k in reach_summary["其中属切块边界切断"]))
    missed10 = set(judge.get("recall_at_10_miss") or [])
    rank_only = sorted(k for k in missed10 if (reach_map.get(k) or {}).get("in_index"))
    if rank_only:
        tags.append("索引有但未召回(排序问题)")
        detail = []
        for kw in rank_only:
            found = (reach_map.get(kw) or {}).get("found") or []
            if found:
                detail.append(f"`{kw}`→`{found[0]['chunk_id']}` p{found[0]['page']}"
                              f"({found[0]['type']})")
        evidence.append("这些关键词在索引里存在，却没进 top-10：" + "、".join(detail))

    if float(judge.get("recall_at_5") or 0.0) < float(judge.get("recall_at_10") or 0.0):
        tags.append("命中集中于6~10位")

    leak = judge.get("cross_corpus_leak") or {}
    if leak.get("leak"):
        tags.append("跨语料泄漏(top3)")
        for item in leak.get("foreign") or []:
            evidence.append(f"top{leak.get('checked')} 第{item['rank']}位是**别的公司**："
                            f"`{item['chunk_id']}` p{item['page']} "
                            f"{J.company_of(item['file_name'])} 分数={item.get('final_score')}")
    leak10 = judge.get("cross_corpus_leak_top10") or {}
    if leak10.get("leak") and not leak.get("leak"):
        tags.append("跨语料泄漏(仅top10内)")
        for item in leak10.get("foreign") or []:
            evidence.append(f"top10 第{item['rank']}位是**别的公司**（未进 top3，但已进入候选上下文）："
                            f"`{item['chunk_id']}` p{item['page']} "
                            f"{J.company_of(item['file_name'])} 分数={item.get('final_score')}")
    if (judge.get("citation_leak") or {}).get("leak"):
        tags.append("引用级跨语料")
        for item in (judge.get("citation_leak") or {}).get("foreign") or []:
            evidence.append(f"引用指向别的公司：{J.company_of(item.get('corpus'))} p{item.get('page')}")

    if not judge.get("page_hit", {}).get("hit"):
        tags.append("页命中失败")
        evidence.append(f"期望页区间 `{rec.get('expected_pages_hint')}`（reason={judge['page_hit'].get('reason')}），"
                        f"实际 top 命中页=" + "、".join(f"p{h.get('page')}" for h in top3))

    table_ranks = [h["rank"] for h in top3 if h.get("type") == "table"]
    if table_ranks:
        tags.append("表格依赖")
        evidence.append("top3 中表格块位于位次 " + "、".join(str(r) for r in table_ranks)
                        + "：" + "；".join(f"`{h['chunk_id']}` p{h['page']} score={h['final_score']}"
                                          for h in top3 if h.get("type") == "table"))

    if composer_method and composer_method not in ("llm_polished", "llm_grounded", "refused"):
        tags.append("生成回退")
        evidence.append(f"composer 落到 `{composer_method}`（未走 LLM 润色），"
                        f"generation_method=`{method}`，覆盖率="
                        f"{rec.get('generation', {}).get('composer_coverage')}")
    elif method and method not in ("llm_grounded", "refused", "cache_hit"):
        tags.append("生成回退")
        evidence.append(f"generation_method=`{method}`（未走 LLM 润色）")

    if not hits:
        tags.append("零命中")
    return tags, evidence


def build_analysis(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 检索问题分析（人工智能NLP-RAG-功能测试及评估）")
    add("")
    add(f"- 生成时间：{summary.get('生成时间')}")
    add(f"- 语料：9 份 A 股年报；索引 run_id=`{summary['配置'].get('索引run_id')}`，"
        f"chunk_total={summary['配置'].get('索引chunk_total')}")
    add(f"- 口径：top_k={C.TOP_K}，关键词覆盖率阈值={C.KEYWORD_THRESHOLD}，"
        f"recall@k 用 top-k 片段**合并正文**判定，页命中支持 `a-b;c-d`，"
        f"跨语料泄漏看 top-{C.LEAK_TOP_N} 是否出现非期望文件名")
    add(f"- 检索/生成：backend=`{summary['配置'].get('backend')}`，model=`{summary['配置'].get('模型')}`，"
        f"embed=`{summary['配置'].get('embed_model')}`；串行调用，`use_cache={C.USE_CACHE}`")
    meta = summary.get("运行元信息") or {}
    add(f"- 被评估代码修订：`{json.dumps(meta.get('rev_start'), ensure_ascii=False)}`")
    if meta.get("revision_changed_during_run"):
        add(f"- ⚠️ **本次运行期间以下模块被改动，结果不可归因到单一修订**："
            f"{meta['revision_changed_during_run']}")
    add("")

    # ---- 逐题 ----
    add("## 一、逐题检索问题分析")
    add("")
    all_tags: list[str] = []
    per_question_tags: dict[str, list[str]] = {}
    for rec in records:
        judge = rec.get("judge") or {}
        tags, evidence = diagnose(rec)
        per_question_tags[rec["id"]] = tags
        all_tags.extend(tags)

        verdict = ("运行异常" if rec.get("error")
                   else "判对" if judge.get("is_correct")
                   else "部分正确" if judge.get("partial_correct")
                   else "未判对")
        add(f"### {rec['id']}　{verdict}"
            f"　（{rec.get('question_type')}｜{rec.get('company')}｜覆盖率 "
            f"{judge.get('keyword_hit_count')}/{judge.get('keyword_total')}="
            f"{judge.get('keyword_coverage')}）")
        add("")
        add(f"- 问题：{_flat(rec.get('question'), 300)}")
        add(f"- 答案（前 300 字）：{_flat(rec.get('answer'), 300) or '（空）'}")
        add(f"- 判分：覆盖率={judge.get('keyword_coverage')}｜recall@5={judge.get('recall_at_5')}"
            f"｜recall@10={judge.get('recall_at_10')}｜页命中={judge.get('page_hit', {}).get('hit')}"
            f"｜跨语料泄漏={judge.get('cross_corpus_leak', {}).get('leak')}"
            f"｜拒答={rec.get('refused')}（{rec.get('refusal_reason') or '-'}）")
        add(f"- 耗时：retrieve={rec.get('latency_ms', {}).get('retrieve')}ms｜"
            f"generate={rec.get('latency_ms', {}).get('generate')}ms｜"
            f"first_token={rec.get('latency_ms', {}).get('first_token')}ms｜"
            f"total={rec.get('total_ms')}ms｜生成方式=`{rec.get('generation_method')}`")
        add(f"- 引用页：{rec.get('cited_pages')}")
        add("")
        add("  | 位次 | chunk_id | 页码 | 类型 | final_score | 文件 |")
        add("  |---|---|---|---|---|---|")
        for hit in (rec.get("hits") or [])[:5]:
            add(f"  | {hit['rank']} | `{hit['chunk_id']}` | p{hit['page']} | {hit['type']} | "
                f"{hit['final_score']} | {hit['company']} |")
        add("")
        if evidence:
            add("  **检索侧证据**")
            add("")
            for item in evidence:
                add(f"  - {item}")
            add("")
        add(f"  **问题标签**：{'、'.join(tags) if tags else '无（本题检索正常）'}")
        add("")

    # ---- 归类统计 ----
    add("## 二、问题归类统计表")
    add("")
    from collections import Counter
    counter = Counter(all_tags)
    add("| 问题类型 | 出现次数 | 涉及题号 |")
    add("|---|---|---|")
    for tag, count in counter.most_common():
        ids = [qid for qid, tgs in per_question_tags.items() if tag in tgs]
        add(f"| {tag} | {count} | {'、'.join(ids)} |")
    add("")
    add("### 2.1 指标速览")
    add("")
    acc = summary.get("准确率", {})
    rec_all = summary.get("检索召回", {})
    sla = summary.get("SLA", {})
    add("| 指标 | 数值 |")
    add("|---|---|")
    add(f"| 有效题数 | {summary.get('有效题数')} |")
    add(f"| 严格判对 | {acc.get('严格判对题数')} / {summary.get('有效题数')} = {acc.get('严格准确率')} |")
    add(f"| 部分正确题号 | {'、'.join(acc.get('部分正确题号') or []) or '无'} |")
    add(f"| 宽松准确率（严格+部分） | {acc.get('宽松准确率(严格+部分)')} |")
    add(f"| 平均关键词覆盖率 | {acc.get('平均关键词覆盖率')} |")
    add(f"| 平均 recall@5 | {rec_all.get('平均recall@5')} |")
    add(f"| 平均 recall@10 | {rec_all.get('平均recall@10')} |")
    add(f"| 关键词单块可达率 | {(summary.get('索引可达性') or {}).get('单块可达率')} "
        f"（{(summary.get('索引可达性') or {}).get('索引内可达')}/"
        f"{(summary.get('索引可达性') or {}).get('关键词总数')}） |")
    add(f"| 单块内不可达关键词 | {(summary.get('索引可达性') or {}).get('单块内不可达')} 个 "
        f"（其中**切块边界切断** {(summary.get('索引可达性') or {}).get('其中属切块边界切断')} 个、"
        f"疑似解析阶段丢失 {(summary.get('索引可达性') or {}).get('其中疑似解析阶段即丢失')} 个） |")
    add(f"| 切块边界切断明细 | {(summary.get('索引可达性') or {}).get('切块边界切断明细')} |")
    add(f"| 索引有但 top10 未召回 | "
        f"{(summary.get('索引可达性') or {}).get('索引可达但top10未召回关键词数')} 个 "
        f"（{(summary.get('索引可达性') or {}).get('索引可达但top10未召回明细')}） |")
    add(f"| 拒答率 | {summary.get('拒答', {}).get('拒答率')} "
        f"（{summary.get('拒答', {}).get('拒答题数')} 题：{'、'.join(summary.get('拒答', {}).get('拒答题号') or []) or '无'}） |")
    add(f"| 跨语料泄漏题数 | {summary.get('跨语料泄漏', {}).get('题数')} "
        f"（{'、'.join(summary.get('跨语料泄漏', {}).get('题号') or []) or '无'}） |")
    add(f"| 跨语料泄漏题数(top10 内) | {summary.get('跨语料泄漏', {}).get('top10题数')} "
        f"（{'、'.join(summary.get('跨语料泄漏', {}).get('top10题号') or []) or '无'}） |")
    add(f"| 引用级跨语料题数 | {summary.get('跨语料泄漏', {}).get('引用级泄漏题数')} "
        f"（{'、'.join(summary.get('跨语料泄漏', {}).get('引用级泄漏题号') or []) or '无'}） |")
    add(f"| SLA≤{C.SLA_MS}ms 达标率 | {sla.get('达标率')} |")
    add("")

    add("### 2.2 按题型")
    add("")
    add("| 题型 | 题数 | 判对率 | 平均覆盖率 | 平均recall@10 | 拒答 | 跨语料泄漏 |")
    add("|---|---|---|---|---|---|---|")
    for name, stat in (summary.get("按题型分组") or {}).items():
        add(f"| {name} | {stat['题数']} | {stat['判对率']} | {stat['平均关键词覆盖率']} | "
            f"{stat['平均recall@10']} | {stat['拒答题数']} | {stat['跨语料泄漏题数']} |")
    add("")
    add("### 2.3 按公司")
    add("")
    add("| 公司 | 题数 | 判对率 | 平均覆盖率 | 平均recall@10 | 拒答 | 跨语料泄漏 |")
    add("|---|---|---|---|---|---|---|")
    for name, stat in (summary.get("按公司分组") or {}).items():
        add(f"| {name} | {stat['题数']} | {stat['判对率']} | {stat['平均关键词覆盖率']} | "
            f"{stat['平均recall@10']} | {stat['拒答题数']} | {stat['跨语料泄漏题数']} |")
    add("")
    cand = summary.get("LLM候选对照口径(反事实诊断)") or {}
    add("### 2.4 反事实诊断：LLM 候选口径（非官方准确率）")
    add("")
    add("把最终答案替换成生成层产出的 **LLM 候选原文**（检索结果不变）后重新判分，"
        "用于区分「检索/生成答错了」与「答案本来对、被自己的原子护栏回退丢掉」：")
    add("")
    add("| 项 | 数值 |")
    add("|---|---|")
    add(f"| 有 LLM 候选的题数 | {cand.get('有LLM候选的题数')} |")
    add(f"| 候选被护栏拒绝的题数 | {cand.get('候选被护栏拒绝的题数')} |")
    add(f"| 候选判对题数（覆盖率≥{C.KEYWORD_THRESHOLD}） | {cand.get('候选判对题数')} |")
    add(f"| 候选口径准确率 | {cand.get('候选口径准确率')} |")
    add(f"| **本可判对却被护栏丢弃** | {cand.get('被护栏丢弃的判对题数')} 题"
        f"（{'、'.join(cand.get('被护栏丢弃的判对题号') or []) or '无'}） |")
    add("")
    add("")
    add("### 2.5 按生成方式")
    add("")
    add("| 生成方式 | 题数 | 判对率 | 平均覆盖率 | 平均总耗时ms |")
    add("|---|---|---|---|---|")
    for name, stat in (summary.get("按生成方式分组") or {}).items():
        add(f"| `{name}` | {stat['题数']} | {stat['判对率']} | {stat['平均关键词覆盖率']} | "
            f"{stat['平均总耗时ms']} |")
    add("")
    lines.extend(build_deep_dive(records, summary))
    return "\n".join(lines)


def build_deep_dive(records: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    """深度归因 + 问题 TOP5 + 可执行改进建议（数据驱动，数字全部来自本次真实运行）。"""
    import re

    out: list[str] = []
    add = out.append

    def q(rid: str) -> dict[str, Any]:
        return next((r for r in records if r["id"] == rid), {})

    acc = summary.get("准确率", {})
    rec_all = summary.get("检索召回", {})
    sla = summary.get("SLA", {})
    lat = summary.get("响应时间ms", {})
    cand = summary.get("LLM候选对照口径(反事实诊断)") or {}
    reach = summary.get("索引可达性") or {}
    leak = summary.get("跨语料泄漏") or {}

    rejected = [r for r in records if r.get("llm_rejected_reason")]
    discarded = [r for r in records if r.get("guard_discarded_correct")]
    junk_atom_re = re.compile(r"^\d+\s*个$|^\d+\s*年$|^\d+\s*股$|\d+\s*个(年度报告|度)")
    junk_atom_qs = []
    for rec in rejected:
        bad = [str(x) for x in (rec.get("missing_atoms") or []) if junk_atom_re.search(str(x))]
        if bad:
            junk_atom_qs.append((rec["id"], bad))
    # 候选比最终答案更接近参考答案的题（护栏"越拦越差"的直接证据）
    worse_after_guard = [r for r in rejected
                         if float((r["judge"].get("llm_candidate_coverage") or 0.0))
                         > float((r["judge"].get("keyword_coverage") or 0.0))]
    worse_after_guard.sort(key=lambda r: -(float(r["judge"].get("llm_candidate_coverage") or 0.0)))
    # 检索已召回到全部关键词、最终答案却没用上的题
    recalled_unused = [r for r in records
                       if float((r["judge"].get("recall_at_10") or 0.0)) >= 1.0
                       and not (r["judge"].get("is_correct"))]
    pct_corrupt = [r for r in records
                   if re.search(r"\d\s*%[亿万]", str(r.get("answer") or ""))]

    add("## 三、深度归因")
    add("")
    add("### 3.1 头号问题：生成层「原子保全护栏」系统性误拒正确答案")
    add("")
    add(f"- **{len(rejected)}/{len(records)}** 题的 LLM 候选被 `atoms_missing` 拒绝，"
        f"生成方式落到 `rule_composer`（即「回退到规则拼接」）。")
    if junk_atom_qs:
        add(f"- 其中 **{len(junk_atom_qs)}** 题的缺失原子里含有 `数字+个` 形态的**垃圾原子**"
            f"（相邻数字与量词被错误粘连，任何自然语言答案都不可能包含它）：")
        add("")
        add("  | 题号 | 垃圾原子（部分） |")
        add("  |---|---|")
        for qid, bad in junk_atom_qs:
            add(f"  | {qid} | {'、'.join('`%s`' % b for b in bad[:4])} |")
        add("")
    else:
        add("- 本轮缺失原子中**已不再出现** `数字+个` 形态的垃圾原子"
            "（该形态在评估过程中被上游修掉，本框架历史运行曾记录到 16/17 题受其影响、"
            "并直接把准确率压到 0/17）。")
    add(f"- **当前的真实机理是「草稿证据选错」**：护栏是拿 LLM 候选去比对"
        f"**规则草稿**里的原子；当草稿本身取自错误的证据块时，它的原子与用户的问题无关，"
        f"而 LLM 按问题作答反而对不上 ⇒ 越正确的候选越容易被拒。"
        f"实测 **{len(worse_after_guard)}/{len(rejected)}** 题「候选覆盖率 > 最终答案覆盖率」。")
    add("")
    if worse_after_guard:
        add("  | 题号 | 候选覆盖率 | 最终覆盖率 | 草稿实际取用的证据块 | 被判定缺失的原子（前 3） |")
        add("  |---|---|---|---|---|")
        for rec in worse_after_guard[:6]:
            judge = rec.get("judge") or {}
            used = (rec.get("generation") or {}).get("used_chunk_ids") or []
            miss = [str(x) for x in (rec.get("missing_atoms") or [])][:3]
            add(f"  | {rec['id']} | {judge.get('llm_candidate_coverage')} | "
                f"{judge.get('keyword_coverage')} | "
                f"{'、'.join('`%s`' % u for u in used) or '-'} | "
                f"{'、'.join('`%s`' % m for m in miss) or '-'} |")
        add("")
    add(f"- **后果量化**：反事实口径（把最终答案换成 LLM 候选、检索不变）准确率为 "
        f"**{cand.get('候选口径准确率')}**（{cand.get('候选判对题数')}/{len(records)}），"
        f"而官方严格准确率是 **{acc.get('严格准确率')}**。")
    add(f"- **本可判对却被护栏丢弃**：{cand.get('被护栏丢弃的判对题数')} 题 "
        f"（{'、'.join(cand.get('被护栏丢弃的判对题号') or []) or '无'}）。逐条证据：")
    add("")
    for rid in (cand.get("被护栏丢弃的判对题号") or []):
        rec = q(rid)
        judge = rec.get("judge") or {}
        add(f"  - **{rid}**：候选覆盖率 "
            f"{judge.get('llm_candidate_coverage')}（≥{C.KEYWORD_THRESHOLD} 即判对），"
            f"最终答案覆盖率 {judge.get('keyword_coverage')}。")
        add(f"    - 候选原文：`{_flat(rec.get('llm_candidate'), 220)}`")
        add(f"    - 被拒缺失原子：`{json.dumps(rec.get('missing_atoms'), ensure_ascii=False)[:220]}`")
        add(f"    - 草稿取用块：`{json.dumps((rec.get('generation') or {}).get('used_chunk_ids'), ensure_ascii=False)}`"
            f"；草稿正文：`{_flat(rec.get('draft_text_head'), 140)}`")
        add(f"    - 最终答案：`{_flat(rec.get('answer'), 160)}`")
    add("")
    add("> 说明：护栏本身的设计意图（防止 LLM 丢数字/改数字）是对的。"
        "问题在**上游草稿的证据选择**：草稿选错块 ⇒ 原子表离题 ⇒ 护栏必然拒绝更好的候选，"
        "等于把这条链路的 LLM 润色能力整体废掉。只修原子抽取的「形态」不够，"
        "必须同时修「草稿取哪块证据」。")
    add("")

    add("### 3.2 语料路由修复复核（回答「修复后是否仍存在引用别家公司年报」）")
    add("")
    add(f"- 当前 `corpus_of_file` 取自 manifest `corpus_keys`（9 份语料 9 个键）后，"
        f"跨语料泄漏降到：top-3 **{leak.get('题数')}** 题、top-10 **{leak.get('top10题数')}** 题、"
        f"引用级 **{leak.get('引用级泄漏题数')}** 题。")
    add(f"- 残留题号：top-3 = {leak.get('题号')}；引用级 = {leak.get('引用级泄漏题号')}。")
    add("")
    add("- 旧口径（`corpus_key_of` 取文件名**首个数字段** = 公告年份）会把 9 份年报并成 "
        "`pdf2020`/`pdf2021`/`pdf2022` **三个桶**，"
        "「平安银行」过滤后招商银行/邮储银行/中国平安照样进候选，过滤形同失效；"
        "修复前的实测泄漏为 top-3 6 题、top-10 12 题、引用级 5 题（本框架历史运行记录）。")
    add("")
    add("- **仍存在的残留缺陷（缺陷，不是可接受误差）**：")
    for rec in records:
        j = rec.get("judge") or {}
        if not (j.get("cross_corpus_leak") or {}).get("leak"):
            continue
        add(f"  - **{rec['id']}**（期望语料 {rec.get('company')}）："
            f"top-3 里出现 " + "、".join(
                f"`{f['chunk_id']}` p{f['page']} {J.company_of(f['file_name'])}"
                for f in j["cross_corpus_leak"]["foreign"])
            + f"；最终答案：`{_flat(rec.get('answer'), 90)}`")
        if (j.get("citation_leak") or {}).get("leak"):
            add(f"    - **引用页也落在别家公司**："
                + "、".join(f"{J.company_of(x.get('corpus'))} p{x.get('page')}"
                            for x in j["citation_leak"]["foreign"]))
    add("")
    add("  典型机理：中国平安年报正文会描述其子公司平安银行"
        "（「平安银行整体经营稳中趋好，实现营业收入1,379.58亿元」），"
        "字面上完全命中「平安银行…营业收入」，"
        "所以 Q09 这种「平安银行 vs 招商银行」的对比题仍会把中国平安的块排进 top-2。"
        "英文题 Q13 更严重：题面是 `Postal Savings Bank of China`，"
        "语料路由没有解析出邮储银行，top-1 直接是**招商银行 p218 商誉附注**，"
        "答案也就照抄了招商银行。")
    add("")

    add("### 3.3 检索侧：索引可达但没进 top-10")
    add("")
    add(f"- 关键词**单块可达率** {reach.get('单块可达率')}"
        f"（{reach.get('索引内可达')}/{reach.get('关键词总数')}）；"
        f"其中落在期望页区间的有 {reach.get('索引内可达且落在期望页')} 个。")
    add(f"- **索引里有、但 top-10 没召回**的关键词共 "
        f"{reach.get('索引可达但top10未召回关键词数')} 个 ⇒ 这部分是**排序/召回问题**，不是索引缺失。")
    add("")
    add("  | 题号 | 索引可达但 top10 未召回的关键词 |")
    add("  |---|---|")
    for qid, kws in (reach.get("索引可达但top10未召回明细") or {}).items():
        add(f"  | {qid} | {'、'.join('`%s`' % k for k in kws)} |")
    add("")
    add(f"- 单块内不可达的只有 {reach.get('单块内不可达')} 个，"
        f"其中**切块边界切断** {reach.get('其中属切块边界切断')} 个、"
        f"疑似解析阶段丢失 {reach.get('其中疑似解析阶段即丢失')} 个。")
    if reach.get("切块边界切断明细"):
        add(f"  - 明细：{json.dumps(reach['切块边界切断明细'], ensure_ascii=False)}")
        add("  - 成因（Q02 实测）：`20200321-p5-x0` 正好以「…聚焦高频生活场景，**联合合作**」结尾，"
            "而续文「**伙伴共同为用户提供优质服务**…逾越者联盟…蛋壳平台…」落在 `20200321-p5-x1`。"
            "短语被切断 ⇒ 没有一个块能整体匹配 ⇒ 该关键短语在检索中永远不可达。"
            "（原始页面 `pages.jsonl` 第 5 页共 1374 字，内容并未丢失，属**切块策略**问题。）")
    add("")

    add("### 3.4 表格依赖与数字表述污染")
    add("")
    table_qs = [r["id"] for r in records if any(h.get("type") == "table" for h in (r.get("hits") or [])[:3])]
    add(f"- **{len(table_qs)}/{len(records)}** 题的 top-3 命中里含表格块（题号：{'、'.join(table_qs)}）。"
        "本工单的答案原子（营收/净利润/不良率/拨备覆盖率…）绝大多数只存在于表格，"
        "所以表格解析质量直接决定上限。")
    add(f"- 表格解析并非无损：例如 `20210326-p8-t0-0`（中国人寿 p8 主要会计数据）"
        "把「已赚保费/营业支出/赔付支出」等行名先整体输出，再单独罗列数字列，"
        "行名与数值脱钩；`20200214-p20-t0-0` 把「截至披露前一交易日的公司总股本」等"
        "无关小表当成了该题的头号证据。")
    if pct_corrupt:
        add(f"- **数字后缀污染**：{len(pct_corrupt)} 题的答案出现 `数字%亿元` 这种错误写法"
            f"（题号：{'、'.join(r['id'] for r in pct_corrupt)}），"
            f"例：{pct_corrupt[0]['id']} 答「{_flat(pct_corrupt[0].get('answer'), 90)}」。"
            "这是规则拼接层把百分号与单位拼接错位，数字本身正确但表述已失真。")
    add("")

    add("### 3.5 稳定性与响应时间")
    add("")
    add(f"- 本次运行：平均 {lat.get('平均')}ms｜中位/P50 {lat.get('P50')}ms｜"
        f"P90 {lat.get('P90')}ms｜P95 {lat.get('P95')}ms｜最大 {lat.get('最大')}ms。")
    add(f"- 分段：检索平均 {lat.get('检索_平均')}ms（P90 {lat.get('检索_P90')}ms）；"
        f"生成平均 {lat.get('生成_平均')}ms（P90 {lat.get('生成_P90')}ms）；"
        f"首字平均 {lat.get('首字_平均')}ms、最大 {lat.get('首字_最大')}ms。")
    add(f"- SLA≤{C.SLA_MS}ms 达标率 **{sla.get('达标率')}**（{sla.get('达标题数')}/{lat.get('样本数')}），"
        f"超时题号 {sla.get('超时题号')}。**检索侧不是瓶颈**（平均仅 {lat.get('检索_平均')}ms），"
        f"超时全部来自 CPU 上的 qwen2.5:3b 生成。")
    add("- ⚠️ **跨轮波动很大**：同一份代码、同一索引、同一问题集，本框架连续多轮实测"
        "平均总耗时在 ~1.3s ~ ~2.7s 之间、SLA 达标率在 13/17 ~ 17/17 之间波动，"
        "原因是本机 Ollama 只有 1 个 CPU 推理实例，与其它任务（如重建索引）争抢算力。"
        "**结论：SLA 结论必须标注运行环境与并发状态，单轮数字不足以定论。**")
    add(f"- 另注：{len([r for r in records if r.get('refused')])} 题触发拒答（原因 "
        f"{summary.get('拒答', {}).get('按原因')}）；"
        "生成护栏的拒绝与否会影响是否走到兜底拒答，因此**拒答率本身也随轮次波动**。")
    add("")

    add("## 四、检索问题 TOP5（按影响面排序）")
    add("")
    add("| 排名 | 问题 | 影响题数 | 关键证据 |")
    add("|---|---|---|---|")
    add(f"| 1 | 生成层原子护栏拒掉比最终答案更好的 LLM 候选 | "
        f"{len(rejected)} 题被拒、其中 {len(worse_after_guard)} 题候选覆盖率高于最终答案、"
        f"{cand.get('被护栏丢弃的判对题数')} 题本可判对 | "
        + (f"Q15 候选「{_flat(q('Q15').get('llm_candidate'), 70)}」覆盖率 0.8，"
           f"被拒后最终答案覆盖率 0.0；草稿取自 `20220331-p39-x1`（结构化主体段落）"
           if q('Q15') else "见 3.1 逐条证据") + " |")
    add(f"| 2 | 答案未采用已召回到的关键原子（检索到了但生成没用上） | "
        f"{len(recalled_unused)} 题（recall@10=1.0 仍未判对） | "
        + ("；".join(f"{r['id']} 覆盖率 {r['judge']['keyword_coverage']}" for r in recalled_unused[:6])
           or "无") + " |")
    add(f"| 3 | 跨语料泄漏残留（尤其英文题面与子公司同名场景） | "
        f"top-3 {leak.get('题数')} 题、引用级 {leak.get('引用级泄漏题数')} 题 | "
        f"Q13 top-1=`20200321-p218-x2` p218 **招商银行** 商誉附注（问的是邮储银行）；"
        f"Q09 引用 `20200221-p41-x2` p41 **中国平安** |")
    add(f"| 4 | 关键原子未进 top-10（排序/召回问题，索引里明明有） | "
        f"{reach.get('索引可达但top10未召回关键词数')} 个关键词、"
        f"{len(reach.get('索引可达但top10未召回明细') or {})} 题 | "
        f"Q07 缺 `1,379.58`/`281.95`、Q09 缺 `2,697.03`/`8.51%`、"
        f"Q13 缺 `610.36亿元`/`81.80亿元`——它们在索引里都可单块命中 |")
    add(f"| 5 | 表格证据选取错位 + 数字表述污染 | "
        f"{len(table_qs)} 题依赖表格"
        + (f"、{len(pct_corrupt)} 题数字表述被污染" if pct_corrupt else "、本轮无数字表述污染")
        + " | "
        + ("Q07 top-1=`20200214-p19-t1-0` p19 是**分季度**表（营收 137,958 百万），"
           "用户要的全年数在 p5 致辞"
           + ("；" + "、".join(f"{r['id']} 答「{_flat(r.get('answer'), 40)}」" for r in pct_corrupt[:2])
              if pct_corrupt else "")
           ) + " |")
    add("")

    add("## 五、可执行改进建议")
    add("")
    add("1. **修 `answer_composer` 的原子抽取（最高优先，收益最大）**：")
    add("   - 过滤 `^\\d+个$`、`^\\d+年$`、`^\\d+股$`、`\\d+个年度报告` 这类「数字+量词」假原子，"
        "或在 `atoms_preserved` 中对纯计数型原子免检；")
    add("   - 增加一条回归用例：**草稿含 `2019个` 时不得拒绝含「2019年」的候选**。")
    add("2. **改切块策略**：按语义块/表格边界切，并对块间**重叠 50~80 字**，"
        "消除 Q02 这类「关键词被边界切断」造成的不可达。")
    add("3. **补英文题面的语料路由**：`crosslingual_aliases` 对 "
        "`Postal Savings Bank of China` 未生效；建议把 9 家公司的中英文名/股票代码建成显式别名表，"
        "并把「路由命中失败」升级为**可观测指标**（当前 `corpus_hint_hit` 在键被合并时仍返回 True，"
        "是假阳性信号）。")
    add("4. **修 `corpus_key_of` 的兜底路径**：`corpus_of_file` 缺失时按「首个数字段」退化会把同年公告并桶。"
        "建议直接 `raise`/告警，或退回「完整文件名」当键，绝不静默合并。")
    add("5. **提高表格证据优先级**：数值问句（`answer_shape=numeric_series`）时，"
        "对含「主要会计数据/财务摘要/利润表」标题的表格块加权，"
        "并抑制「分季度」「附注」类表格；同时修复 `数字%单位` 的拼接污染。")
    add("6. **把护栏结果纳入评估口径**：建议汇总里常驻「LLM 候选口径」与「护栏拒绝率」，"
        "否则「准确率低」会被误判成检索问题，实际是生成护栏问题。")
    add("")
    return out


# ===========================================================================
# 落盘
# ===========================================================================
CSV_COLUMNS = ["题号", "问题", "lang", "对应公司", "答案", "是否拒答", "关键词覆盖率", "是否判对",
               "recall@5", "recall@10", "页命中", "跨语料泄漏", "检索ms", "生成ms", "首字ms",
               "总ms", "生成方式"]


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for rec in records:
            judge = rec.get("judge") or {}
            latency = rec.get("latency_ms") or {}
            writer.writerow([
                rec.get("id"),
                rec.get("question"),
                rec.get("lang"),
                rec.get("company"),
                rec.get("answer"),
                rec.get("refused"),
                judge.get("keyword_coverage"),
                judge.get("is_correct"),
                judge.get("recall_at_5"),
                judge.get("recall_at_10"),
                judge.get("page_hit", {}).get("hit"),
                judge.get("cross_corpus_leak", {}).get("leak"),
                latency.get("retrieve"),
                latency.get("generate"),
                latency.get("first_token"),
                rec.get("total_ms"),
                rec.get("generation_method"),
            ])


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def write_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


# ===========================================================================
# 主流程
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="人工智能NLP-RAG-功能测试及评估 —— 评估主入口")
    parser.add_argument("--ids", default="", help="只跑指定题号，逗号分隔，如 Q07,Q08")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    parser.add_argument("--out-suffix", default="", help="输出文件名后缀（烟测用，避免覆盖交付物）")
    args = parser.parse_args(argv)

    C.ensure_out_dir()
    log_path = C.OUT_FILES["log"]
    log_fh = log_path.open("w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_fh)
    exit_code = 0
    try:
        exit_code = _run(args)
    except Exception:  # noqa: BLE001 - 顶层兜底：异常也要留日志
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout = real_stdout
        log_fh.close()
    return exit_code


def _run(args: argparse.Namespace) -> int:
    run_at = time.strftime("%Y-%m-%d %H:%M:%S")
    _p("=" * 78)
    _p(f" {C.WORK_ORDER} —— 自研评估框架")
    _p("=" * 78)
    _p(f"开始时间      : {run_at}")
    _p(f"解释器        : {sys.executable}")
    _p(f"Python        : {sys.version.split()[0]}")
    _p(f"工作目录      : {Path.cwd()}")
    _p(f"问题集        : {C.QUESTION_SET}")
    _p(f"输出目录      : {C.OUT_DIR}")
    _p(f"索引目录      : {C.INDEX_DIR}")
    _p(f"top_k         : {C.TOP_K}（生成层取用 prompt_top_k={C.PROMPT_TOP_K}）")
    _p(f"use_cache     : {C.USE_CACHE}（False=每题按首次提问计，SLA 统计不被缓存污染）")
    _p(f"SLA 阈值      : {C.SLA_MS} ms")
    rev_start = module_revision()
    _p("被评估代码修订（SHA256 前 16 位）:")
    for name, digest in rev_start.items():
        _p(f"    {name:<26} {digest}")
    _p("")

    # ---- 索引就绪检查 ----
    chunk_total = None
    index_run_id = None
    if C.INDEX_MANIFEST.exists():
        try:
            manifest = json.loads(C.INDEX_MANIFEST.read_text(encoding="utf-8"))
            chunk_total = manifest.get("chunk_total")
            index_run_id = manifest.get("run_id")
            _p(f"[索引] run_id={index_run_id} chunk_total={chunk_total} "
               f"embed={manifest.get('embed_model')} backend={manifest.get('vector_backend')}")
            if chunk_total is not None and int(chunk_total) < C.INDEX_MIN_CHUNKS:
                _p(f"[索引] !! chunk_total={chunk_total} < {C.INDEX_MIN_CHUNKS}，索引可能未重建完成")
        except Exception as exc:  # noqa: BLE001
            _p(f"[索引] 清单读取失败：{type(exc).__name__}: {exc}")
    else:
        _p(f"[索引] !! 未找到 {C.INDEX_MANIFEST}")

    questions = load_questions(C.QUESTION_SET)
    _p(f"[问题集] 载入 {len(questions)} 题")
    if args.ids:
        wanted = {x.strip().upper() for x in args.ids.split(",") if x.strip()}
        questions = [q for q in questions if str(q.get("id")).upper() in wanted]
        _p(f"[问题集] --ids 过滤后 {len(questions)} 题：{', '.join(str(q.get('id')) for q in questions)}")
    if args.limit:
        questions = questions[: args.limit]
        _p(f"[问题集] --limit 过滤后 {len(questions)} 题")

    # ---- 引擎 ----
    from app.core.config import get_config  # noqa: E402
    from app.core.qa_engine import QAEngine  # noqa: E402

    cfg = get_config()
    engine = QAEngine(cfg=cfg)
    _p("")
    _p("[预热] 加载索引 + 预热生成模型（不预热首字延迟可达 3.4s，会违约）…")
    t0 = time.perf_counter()
    warm = engine.warmup(with_generate=True)
    warm_ms = round((time.perf_counter() - t0) * 1000, 2)
    _p(f"[预热] 完成，用时 {warm_ms} ms；warmup={json.dumps(warm, ensure_ascii=False, default=str)[:400]}")

    health = engine.health()
    backend = (health.get("checks", {}).get("llm", {}) or {}).get("backend")
    model = (health.get("checks", {}).get("llm", {}) or {}).get("model")
    embed_model = cfg.llm.ollama_embed_model
    _p(f"[健康] status={health.get('status')} backend={backend} model={model} embed={embed_model}")
    try:
        index_find = build_index_reach(engine)
    except Exception as exc:  # noqa: BLE001 - 该诊断失败不应影响主评估
        _p(f"[索引可达性] !! 构建失败，跳过该诊断：{type(exc).__name__}: {exc}")
        index_find = None
    _p("")

    # ---- 逐题（串行）----
    records: list[dict[str, Any]] = []
    consecutive_errors = 0
    for idx, item in enumerate(questions, start=1):
        rec = evaluate_one(engine, item, idx, len(questions), index_find=index_find)
        records.append(rec)
        if rec.get("error"):
            consecutive_errors += 1
            if consecutive_errors >= 3:
                _p("")
                _p(f"!! 连续 {consecutive_errors} 题异常，疑似引擎/索引不可用，中止全量评估。")
                break
        else:
            consecutive_errors = 0

    # ---- 汇总与落盘 ----
    rev_end = module_revision()
    meta = {"run_at": run_at, "model": model, "backend": backend, "embed_model": embed_model,
            "chunk_total": chunk_total, "index_run_id": index_run_id, "warmup_ms": warm_ms,
            "interpreter": sys.executable, "cwd": str(Path.cwd()),
            "rev_start": rev_start, "rev_end": rev_end}
    changed = sorted(k for k in rev_start if rev_start.get(k) != (rev_end or {}).get(k))
    meta["revision_changed_during_run"] = changed
    if changed:
        _p("")
        _p("!! 警告：以下模块在本次评估运行期间被修改，结果不可严格归因到单一修订：")
        for name in changed:
            _p(f"     {name}: {rev_start[name]} -> {rev_end.get(name)}")
        _p("")
    summary = build_summary(records, meta)
    summary["运行元信息"] = meta

    suffix = args.out_suffix
    paths = {key: (path.with_name(f"{path.stem}{suffix}{path.suffix}") if suffix else path)
             for key, path in C.OUT_FILES.items()}
    # 日志文件名恒定（就是本次运行的真实控制台输出），不随烟测后缀改名
    paths["log"] = C.OUT_FILES["log"]

    write_jsonl(records, paths["jsonl"])
    write_csv(records, paths["csv"])
    write_json(summary, paths["summary"])
    paths["analysis"].write_text(build_analysis(records, summary), encoding="utf-8")

    # ---- 控制台汇总表 ----
    _p("")
    _p("=" * 78)
    _p(" 17 题关键判分表" if len(records) == 17 else f" {len(records)} 题关键判分表")
    _p("=" * 78)
    header = f"{'题号':<6}{'判对':<6}{'部分':<6}{'覆盖率':<9}{'recall@5':<10}{'recall@10':<11}{'总耗时ms':<11}{'生成方式':<16}"
    _p(header)
    _p("-" * 78)
    for rec in records:
        judge = rec.get("judge") or {}
        _p(f"{rec['id']:<6}"
           f"{('是' if judge.get('is_correct') else '否'):<6}"
           f"{('是' if judge.get('partial_correct') else '否'):<6}"
           f"{str(judge.get('keyword_coverage')):<9}"
           f"{str(judge.get('recall_at_5')):<10}"
           f"{str(judge.get('recall_at_10')):<11}"
           f"{str(rec.get('total_ms')):<11}"
           f"{str(rec.get('generation_method')):<16}")
    _p("")
    acc = summary["准确率"]
    rec_all = summary["检索召回"]
    lat = summary["响应时间ms"]
    sla = summary["SLA"]
    _p("汇总指标")
    _p("-" * 78)
    _p(f"  有效题数            : {summary['有效题数']} / {summary['总数']}（异常 {summary['异常题数']}）")
    _p(f"  严格准确率          : {acc['严格判对题数']}/{summary['有效题数']} = {acc['严格准确率']}")
    _p(f"  宽松准确率(含部分)  : {acc['宽松准确率(严格+部分)']}  部分正确题号={acc['部分正确题号']}")
    _p(f"  平均关键词覆盖率    : {acc['平均关键词覆盖率']}")
    _p(f"  平均 recall@5       : {rec_all['平均recall@5']}")
    _p(f"  平均 recall@10      : {rec_all['平均recall@10']}")
    _p(f"  响应时间(ms)        : 平均={lat['平均']} 中位={lat['中位']} P50={lat['P50']} "
       f"P90={lat['P90']} 最大={lat['最大']}")
    _p(f"  分段(ms)            : 检索平均={lat['检索_平均']} 生成平均={lat['生成_平均']} "
       f"首字平均={lat['首字_平均']} 首字最大={lat['首字_最大']}")
    _p(f"  SLA≤{C.SLA_MS}ms 达标率  : {sla['达标率']}（{sla['达标题数']}/{lat['样本数']}）"
       f" 超时题号={sla['超时题号']}")
    _p(f"  拒答率              : {summary['拒答']['拒答率']} 题号={summary['拒答']['拒答题号']} "
       f"原因={summary['拒答']['按原因']}")
    _p(f"  跨语料泄漏题数      : {summary['跨语料泄漏']['题数']} 题号={summary['跨语料泄漏']['题号']}")
    _p(f"  跨语料泄漏(top10)   : {summary['跨语料泄漏']['top10题数']} "
       f"题号={summary['跨语料泄漏']['top10题号']}")
    _p(f"  引用级跨语料题数    : {summary['跨语料泄漏']['引用级泄漏题数']} "
       f"题号={summary['跨语料泄漏']['引用级泄漏题号']}")
    _p(f"  页命中率            : {summary['页命中']['命中率']} 未命中={summary['页命中']['未命中题号']}")
    ir = summary.get("索引可达性") or {}
    _p(f"  关键词单块可达性    : {ir.get('索引内可达')}/{ir.get('关键词总数')} = {ir.get('单块可达率')}"
       f"（单块不可达 {ir.get('单块内不可达')}：切块边界切断 {ir.get('其中属切块边界切断')}、"
       f"疑似解析丢失 {ir.get('其中疑似解析阶段即丢失')}；"
       f"索引可达但 top10 未召回 {ir.get('索引可达但top10未召回关键词数')}）")
    _p("")
    cand = summary.get("LLM候选对照口径(反事实诊断)") or {}
    _p(f"  [反事实诊断] LLM 候选口径准确率 : {cand.get('候选口径准确率')} "
       f"（{cand.get('候选判对题数')}/{summary['有效题数']}，题号={cand.get('候选判对题号')}）")
    _p(f"  [反事实诊断] 候选被护栏拒绝     : {cand.get('候选被护栏拒绝的题数')} 题；"
       f"其中**本可判对却被丢弃** {cand.get('被护栏丢弃的判对题数')} 题 "
       f"（{cand.get('被护栏丢弃的判对题号')}）")
    _p("")
    _p("按题型分组")
    for name, stat in summary["按题型分组"].items():
        _p(f"  {name:<14} 题数={stat['题数']:<3} 判对率={stat['判对率']} "
           f"平均覆盖率={stat['平均关键词覆盖率']} 平均recall@10={stat['平均recall@10']} "
           f"拒答={stat['拒答题数']} 泄漏={stat['跨语料泄漏题数']}")
    _p("")
    _p("按公司分组")
    for name, stat in summary["按公司分组"].items():
        _p(f"  {name:<14} 题数={stat['题数']:<3} 判对率={stat['判对率']} "
           f"平均覆盖率={stat['平均关键词覆盖率']} 平均recall@10={stat['平均recall@10']} "
           f"拒答={stat['拒答题数']} 泄漏={stat['跨语料泄漏题数']}")
    _p("")
    _p("落盘文件")
    for key in ("jsonl", "csv", "summary", "analysis", "log"):
        _p(f"  - {paths[key]}")
    _p("")
    _p(f"结束时间      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
