# 工单编号：人工智能NLP-RAG-混合检索任务
"""PDF 工单（人工智能NLP-RAG项目-06-混合检索任务工单V1.1）逐项验收实跑脚本。

把工单 PDF 里的每一条验收项，**真跑一遍**并留下证据：
  · 向量检索（召回 + 重排）      · 多种嵌入模型
  · ≥3 种重排算法                · 全文检索（布尔/短语/模糊/多字段）
  · 混合检索 + 权重 + 融合算法    · 多轮对话 / 用户反馈 / 中英双语
  · 响应时间 ≤3s                 · 高并发稳定性（冒烟）
  · 代码注释含工单编号

设计约束（本机）：
  · Qdrant 本地嵌入式单进程 —— 本脚本必须独占运行，不可与其他评估并发；
  · 终端为 GBK，控制台只打印 ASCII 进度；完整证据写 UTF-8 JSON 与文本日志；
  · 所有检查逐项 try/except，单项失败不影响其余项，**失败如实记录**。

用法：
    python scripts/acceptance_pdf.py
产出：
    data/eval/acceptance_pdf.json    机器可读结果
    logs/acceptance_pdf.log          UTF-8 文本日志（含逐项证据）
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import bootstrap  # noqa: E402  必须最先导入（离线开关 + 大栈线程）

WORKORDER = "人工智能NLP-RAG-混合检索任务"
OUT_JSON = os.path.join("data", "eval", "acceptance_pdf.json")
OUT_LOG = os.path.join("logs", "acceptance_pdf.log")

CHECKS: list[dict] = []
_LOGLINES: list[str] = []


def _log(line: str = "") -> None:
    _LOGLINES.append(line)


def _say(line: str) -> None:
    """控制台进度（纯 ASCII，避开 Windows GBK 终端）。"""
    print(line, flush=True)


def run_check(cid: str, item: str, fn, note: str = "") -> None:
    """跑一项检查；异常即该项 FAIL，不影响后续。"""
    t0 = time.time()
    _say(f"[{cid}] running ...")
    try:
        evidence = fn() or {}
        status = "PASS" if evidence.get("ok", True) else "FAIL"
        if status == "PASS":
            evidence.pop("ok", None)
    except Exception as exc:  # noqa: BLE001 —— 验收脚本要留下失败证据而不是崩掉
        status = "FAIL"
        evidence = {"error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:]}
    elapsed = round(time.time() - t0, 2)
    row = {"id": cid, "item": item, "status": status,
           "elapsed_s": elapsed, "note": note, "evidence": evidence}
    CHECKS.append(row)
    _say(f"[{cid}] {status} ({elapsed}s)")
    _log(f"\n### [{cid}] {item} —— {status}（{elapsed}s）")
    if note:
        _log(f"说明：{note}")
    _log("证据：" + json.dumps(evidence, ensure_ascii=False)[:2000])


def _result_brief(r, *, answer_len: int = 160) -> dict:
    """把一次 rag.ask 结果浓缩成验收证据。"""
    a = r.analysis or {}
    ri = a.get("rerank_info") or (a.get("retrieval_info") or {}).get("rerank") or {}
    return {
        "question": r.question,
        "answer": (r.answer or "")[:answer_len],
        "mode": r.mode,
        "n_citations": len(r.citations or []),
        "strategy": a.get("retrieval_strategy") or (a.get("retrieval_info") or {}).get("strategy"),
        "candidates": a.get("recall_candidates"),
        "rerank_algorithm": ri.get("algorithm"),
        "rerank_n_in": ri.get("n_in"),
        "rerank_elapsed_ms": ri.get("elapsed_ms"),
        "rerank_degraded": ri.get("degraded"),
        "n_rank_changes": len(ri.get("rank_changes") or []),
        "top_chunk_id": a.get("top_chunk_id"),
        "lang": a.get("lang"),
        "total_ms": (r.timings or {}).get("total_ms"),
        "timings": r.timings,
    }


# ---------------------------------------------------------------------------
# A. 功能详细需求
# ---------------------------------------------------------------------------
def check_vector():
    from src import rag, runtime
    cfg = runtime.update(strategy="vector", rerank_enabled=True,
                         rerank_algorithm="model", top_k_recall=20)
    r = rag.ask("武汉兴图新科电子股份有限公司注册资本是多少？", use_cache=False)
    ev = _result_brief(r)
    ri = r.analysis.get("rerank_info") or {}
    ev["ok"] = bool(r.answer) and (ri.get("n_in") or 0) > 0 and not ri.get("degraded")
    ev["config"] = {"strategy": cfg.strategy, "top_k_recall": cfg.top_k_recall,
                    "top_k_rerank": cfg.top_k_rerank, "score_threshold": cfg.score_threshold,
                    "rerank_enabled": cfg.rerank_enabled}
    return ev


def check_embedding_models():
    from src import config, vector_store
    models = {k: {"path": v.get("path"), "dim": v.get("dim")} for k, v in
              config.EMBEDDING_MODELS.items()}
    cols = {}
    for k, v in models.items():
        dim = v.get("dim") or config.EMBEDDING_DIM
        try:
            cols[k] = vector_store.collection_exists(config.collection_name(k))
        except Exception as exc:  # noqa: BLE001
            cols[k] = f"check failed: {exc}"
    return {"registered_models": models, "collection_built": cols,
            "n_registered": len(models),
            "ok": len(models) >= 2}


def check_rerank_algorithms():
    from src import config, rag, runtime
    out = {}
    for algo in config.RERANK_ALGORITHMS:
        runtime.update(strategy="vector", rerank_algorithm=algo, top_k_recall=8)
        r = rag.ask("武汉兴图新科电子股份有限公司法定代表人是谁？", use_cache=False)
        ri = r.analysis.get("rerank_info") or {}
        out[algo] = {"n_in": ri.get("n_in"), "n_out": ri.get("n_out"),
                     "elapsed_ms": ri.get("elapsed_ms"),
                     "degraded": ri.get("degraded"),
                     "degrade_reason": ri.get("degrade_reason"),
                     "top1": r.analysis.get("top_chunk_id")}
    runtime.update(rerank_algorithm="model")
    hard = [a for a in ("model", "llm", "tfidf", "adaptive") if a in out]
    return {"algorithms": out, "n": len(hard),
            "ok": len(hard) >= 3 and all(v.get("degraded") is not True for v in out.values())}


def check_fulltext_syntax():
    from src import fulltext
    cases = {
        "bool_AND": "军用 AND 收入",
        "bool_OR": "力源 OR 兴图",
        "bool_NOT": "军用 NOT 民用",
        "phrase": '"军用领域"',
        "fuzzy": "注册资本~",
        "field_title": "title:军用",
        "field_body": "body:军用",
        "field_summary": "summary:军用",
    }
    out = {}
    for name, q in cases.items():
        try:
            hits = fulltext.search(q, top_k=5)
            out[name] = {"query": q, "n_hits": len(hits),
                         "top1": (hits[0].get("chunk_id") if hits else None)}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"query": q, "error": f"{type(exc).__name__}: {exc}"}
    ok = all(v.get("n_hits", 0) > 0 for v in out.values())
    return {"cases": out, "ok": ok}


def check_fulltext_index():
    from src import fulltext
    s = fulltext.stats()
    return {"stats": s, "ok": s.get("docs", 0) > 0}


def check_hybrid_fusion():
    from src import config, rag, runtime
    out = {}
    q = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
    for fusion in config.FUSION_METHODS:
        runtime.update(strategy="hybrid", fusion=fusion, rerank_algorithm="model")
        r = rag.ask(q, use_cache=False)
        ri = r.analysis.get("rerank_info") or {}
        out[fusion] = {"answer": (r.answer or "")[:80], "top_chunk_id": r.analysis.get("top_chunk_id"),
                       "n_candidates": r.analysis.get("recall_candidates"),
                       "rerank_n_in": ri.get("n_in"), "degraded": ri.get("degraded")}
    runtime.update(fusion="rrf")
    return {"fusions": out, "n": len(out),
            "ok": len(out) >= 2 and all(v.get("answer") for v in out.values())}


def check_hybrid_weights():
    from src import rag, runtime
    q = "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"
    out = {}
    for vw, fw in ((0.5, 0.5), (0.8, 0.2), (0.2, 0.8)):
        cfg = runtime.update(strategy="hybrid", fusion="weighted",
                             vector_weight=vw, fulltext_weight=fw)
        r = rag.ask(q, use_cache=False)
        out[f"v{vw}_f{fw}"] = {"normalized": cfg.normalized_weights(cfg.fusion),
                               "top_chunk_id": r.analysis.get("top_chunk_id"),
                               "answer": (r.answer or "")[:60]}
    runtime.update(vector_weight=0.5, fulltext_weight=0.5, fusion="rrf")
    tops = {v["top_chunk_id"] for v in out.values()}
    return {"configs": out, "distinct_tops": len(tops), "ok": len(out) == 3}


# ---------------------------------------------------------------------------
# C. 功能验收（交互）
# ---------------------------------------------------------------------------
def check_multiturn():
    from src import dialogue, rag, runtime
    runtime.update(strategy="hybrid")
    state = dialogue.get_state("acceptance")
    dialogue.reset_state("acceptance")
    state = dialogue.get_state("acceptance")
    r1, _ = rag.ask_turn("武汉兴图新科电子股份有限公司注册资本是多少？", state, use_cache=False)
    r2, t2 = rag.ask_turn("它的法定代表人是谁？", state, use_cache=False)
    ev = {
        "turn1_answer": (r1.answer or "")[:80],
        "turn2_question": "它的法定代表人是谁？",
        "turn2_resolved": t2.resolved_question,
        "turn2_rewrite_mode": t2.rewrite_mode,
        "turn2_answer": (r2.answer or "")[:80],
        "turns_in_state": state.turn_count,
    }
    ev["ok"] = (t2.rewrite_mode != "passthrough"
                and "兴图" in (t2.resolved_question or "")
                and bool(r2.answer))
    dialogue.reset_state("acceptance")
    return ev


def check_feedback():
    from src import rag, rerankers, runtime
    runtime.update(strategy="hybrid")
    r = rag.ask("武汉兴图新科电子股份有限公司法定代表人是谁？", use_cache=False)
    hit = {"source": "招股说明书1.pdf", "chunk_id": "acceptance#probe",
           "page_idx": 0, "block_type": "text"}
    before = rerankers.feedback_stats()
    rerankers.record_feedback(r.question, hit, helpful=True, strategy="hybrid",
                              reranker="bge-reranker-v2-m3")
    rerankers.record_feedback(r.question, hit, helpful=False, strategy="hybrid",
                              reranker="bge-reranker-v2-m3")
    after = rerankers.feedback_stats()
    # 自适应重排器能读到反馈（不崩、有排序）
    runtime.update(rerank_algorithm="adaptive")
    r2 = rag.ask("公司注册资本是多少？", use_cache=False)
    runtime.update(rerank_algorithm="model")
    return {"stats_before": before, "stats_after": after,
            "adaptive_rerank_after_feedback": _result_brief(r2),
            "ok": after.get("n", 0) >= before.get("n", 0) + 2}


def check_bilingual():
    from src import rag, runtime
    runtime.update(strategy="hybrid")
    q_en = "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"
    r_en = rag.ask(q_en, use_cache=False)
    q_zh = "武汉兴图新科电子股份有限公司注册资本是多少？"
    r_zh = rag.ask(q_zh, use_cache=False)
    ans_en = r_en.answer or ""
    has_cjk = any("一" <= ch <= "鿿" for ch in ans_en)
    return {"en_question": q_en, "en_answer": ans_en[:160],
            "en_lang": (r_en.analysis or {}).get("lang"),
            "en_answer_has_cjk": has_cjk,
            "zh_answer": (r_zh.answer or "")[:80],
            "zh_lang": (r_zh.analysis or {}).get("lang"),
            "numbers_en": [w for w in ans_en.split() if any(c.isdigit() for c in w)][:6],
            "ok": (r_en.analysis or {}).get("lang") == "en" and bool(ans_en)}


# ---------------------------------------------------------------------------
# D. 性能验收
# ---------------------------------------------------------------------------
def check_latency():
    from src import rag, runtime
    runtime.update(strategy="hybrid", rerank_algorithm="model")
    qs = ["武汉兴图新科电子股份有限公司注册资本是多少？",
          "武汉兴图新科电子股份有限公司法定代表人是谁？",
          "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"]
    rows = []
    for q in qs:
        r = rag.ask(q, use_cache=False)
        rows.append({"answer": (r.answer or "")[:40],
                     "total_ms": (r.timings or {}).get("total_ms"),
                     "timings": r.timings})
    ts = [x["total_ms"] for x in rows if x["total_ms"]]
    return {"rows": rows, "max_ms": max(ts) if ts else None,
            "ok": bool(ts) and max(ts) <= 3000.0}


def check_concurrency_smoke():
    from src import rag, runtime
    runtime.update(strategy="hybrid", rerank_algorithm="model")
    qs = ["武汉兴图新科电子股份有限公司注册资本是多少？",
          "武汉兴图新科电子股份有限公司法定代表人是谁？",
          "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
          "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"]
    n_threads, per = 10, 4

    def one(i: int):
        try:
            rag.ask(qs[i % len(qs)], use_cache=False)
            return True
        except Exception:  # noqa: BLE001
            return False

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(one, range(n_threads * per)))
    elapsed = time.time() - t0
    ok_n = sum(1 for x in results if x)
    return {"concurrency": n_threads, "requests": n_threads * per, "ok": ok_n,
            "errors": n_threads * per - ok_n, "elapsed_s": round(elapsed, 2),
            "qps": round((n_threads * per) / elapsed, 2) if elapsed else None,
            "status_ok": ok_n == n_threads * per}


# ---------------------------------------------------------------------------
# E/F. 备注与产出物
# ---------------------------------------------------------------------------
def check_workorder_tag():
    miss, n = [], 0
    for r, d, f in os.walk("."):
        d[:] = [x for x in d if x not in
                ("__pycache__", ".git", "data", "docs", "logs", ".pytest_cache")]
        for name in f:
            if name.endswith(".py"):
                n += 1
                p = os.path.join(r, name)
                head = "".join(open(p, encoding="utf-8").readlines()[:8])
                if WORKORDER not in head:
                    miss.append(p)
    return {"checked": n, "missing": miss, "ok": not miss}


def check_deliverables():
    need = {
        "docs/01-本机模型扫描结果.md": "模型扫描报告",
        "docs/02-系统架构与技术选型.md": "技术文档",
        "docs/03-三种检索策略配置与应用说明.md": "三策略配置说明",
        "docs/04-检索精确度报告.md": "检索精确度报告",
        "docs/05-优化前后对比分析.md": "优化前后对比",
        "docs/06-RAGAS评估报告.md": "RAGAS 报告",
        "docs/07-容错机制与长时间运行方案.md": "容错与长跑",
        "docs/08-高并发压测报告.md": "压测报告",
        "docs/09-工单验收对照表.md": "验收对照表",
        "docs/10-用户手册.md": "用户手册",
        "docs/11-演示视频录制脚本.md": "演示视频脚本",
        "docs/12-PDF工单逐项验收实跑记录.md": "PDF 工单逐项验收记录",
        "app.py": "中英双语界面",
        "evaluate_ragas.py": "RAGAS 脚本",
        "benchmark_load.py": "压测脚本",
    }
    out = {k: os.path.exists(k) for k in need}
    return {"present": out, "missing": [k for k, v in out.items() if not v],
            "ok": all(out.values())}


# ---------------------------------------------------------------------------
def main() -> int:
    from src import rag, runtime
    _log(f"# PDF 工单逐项验收实跑记录\n运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    _say("== warmup ==")
    t0 = time.time()
    warm = rag.warmup()
    _log(f"\n预热：{json.dumps(warm, ensure_ascii=False)}（{time.time() - t0:.1f}s）")
    _say(f"warmup ok={warm} in {time.time() - t0:.1f}s")

    run_check("A1", "向量检索（召回+重排）", check_vector)
    run_check("A2", "支持多种嵌入模型", check_embedding_models,
              note="注册表 3 个；是否已建第二个库见 evidence")
    run_check("A3", "≥3 种重排算法", check_rerank_algorithms)
    run_check("A4", "全文检索：布尔/短语/模糊/多字段语法", check_fulltext_syntax)
    run_check("A5", "倒排索引规模", check_fulltext_index)
    run_check("A7", "混合检索：三种融合算法", check_hybrid_fusion)
    run_check("A8", "混合检索：权重可调", check_hybrid_weights)
    run_check("C4", "交互：多轮对话（指代消解）", check_multiturn)
    run_check("C5", "交互：用户反馈（👍/👎 落库 + 自适应）", check_feedback)
    run_check("C6", "多语言：中英双语问答", check_bilingual)
    run_check("D1", "响应时间 ≤3 秒", check_latency)
    run_check("D2", "高并发稳定性（10 并发冒烟）", check_concurrency_smoke)
    run_check("E1", "代码注释含工单编号", check_workorder_tag)
    run_check("F1", "产出物齐备", check_deliverables)

    runtime.reset()
    summary = {"pass": sum(1 for c in CHECKS if c["status"] == "PASS"),
               "fail": sum(1 for c in CHECKS if c["status"] == "FAIL"),
               "n": len(CHECKS)}
    payload = {"workorder": WORKORDER, "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "warmup": warm, "checks": CHECKS, "summary": summary}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    _log(f"\n\n## 汇总：{summary}")
    with open(OUT_LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LOGLINES))
    _say(f"SUMMARY pass={summary['pass']} fail={summary['fail']} n={summary['n']}")
    _say(f"JSON -> {OUT_JSON}")
    _say(f"LOG  -> {OUT_LOG}")
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(bootstrap.run_with_large_stack(main))
