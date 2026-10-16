# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 说明：本文件复用自工单8（工单9 未修改其业务逻辑），仅补齐工单9 的编号头注释。
"""工单8 Graph RAG 评估脚本：17 题实跑 + 与 工单7 冻结结果并排对比。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

口径声明（**必须与 工单7 一致，否则对比表无意义**）：
    * 问题集：``研发/data/测试问题集.jsonl``（来源 = 工单7 ``研发/评估/测试问题集.jsonl``，17 题）；
      工单原文提到的 ``eval_question.md`` **未随附件提供**（附件目录内不存在），故采用该 17 题作为
      检索问题集，并在 设计/需求分析.md §5 如实说明这一替代口径。
    * 判分：**直接复用 工单7 的 ``研发/评估/判分.py``**（见 ``研发/评估/判分口径.py``），
      阈值 关键词 0.6 / 部分 0.3 / 原子 0.6、recall@5、recall@10、页命中区间、跨语料泄漏口径全部一致。
    * top_k = 10；``use_cache=False``（每题都是"用户第一次问"）。
    * 生成模型：Ollama ``qwen2.5:3b``（与 工单7 相同）。
    * **消融口径**：每题额外跑一次「关闭图谱」的检索（纯向量+BM25），用于给出
      「图谱扩展带来的增量」这一验收项的可追溯数字。

产物（全部落在本工单，工单7 目录只读）：
    * ``测试/测试数据/GraphRAG评估结果.jsonl`` / ``.csv``
    * ``测试/测试数据/GraphRAG评估汇总.json``
    * ``测试/测试数据/与工单7对比.json`` / ``与工单7对比.csv``
    * ``部署/日志/run_eval.log``

用法：
    pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval.py
    pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval.py --limit 3 --no-llm   # 冒烟
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "研发"), str(_ROOT / "研发" / "评估")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 判分口径 as J  # noqa: E402  —— 工单7 同源判分口径（中文模块名）
from app.graphrag.config import (BASE_EVAL_JSONL, BASE_EVAL_SUMMARY, BASE_INDEX_DIR,  # noqa: E402
                                 TEST_DATA_DIR, WORK_ORDER, ensure_dirs, get_config)

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答


def _logger(path: Path) -> Any:
    """结构化日志：复用 工单7 的 ``logging_conf``，但**路径必须钉在本工单**。

    为什么不自己写轻量 logger（实测踩坑 1）：检索层（``app/core/retriever.py``）会调用
    ``log.enter(...)`` 进入 span、``log.bind(...)`` 派生上下文；自写的极简 logger 没有这些方法，
    会在**每一题**上抛 ``AttributeError: '_Log' object has no attribute 'enter'``
    （实测后果：17 题全部退化为「检索失败」，严格准确率 0%）。

    为什么必须显式设 ``RAG_LOG__DIR``（实测踩坑 2）：``config.py`` 的 ``REPO_ROOT`` 是按
    **文件位置**算的（``parents[3]``），而这份 ``app/core`` 是从 工单7 复制来的 —— 不覆盖路径时
    日志会写进 ``E:\gao6gongdan\工单7\部署\日志\``，**越界写入只读的工单7 目录**
    （实测发现 ``工单7`` 下 ``app.log`` 被本工单进程追加）。这里在 ``setup_logging`` 之前
    强制把日志目录指向本工单，保证「工单7 只读」这条纪律成立。
    """
    import os

    repo_root = Path(__file__).resolve().parents[2]
    os.environ["RAG_LOG__DIR"] = str(repo_root / "部署" / "日志")
    os.environ.setdefault("RAG_DATA__INDEX_DIR", str(BASE_INDEX_DIR))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from app.core.logging_conf import setup_logging

        return setup_logging()
    except Exception as exc:  # noqa: BLE001 —— 日志器构造失败不得阻断评估（退化为空操作）
        print(f"[WARN] 结构化日志初始化失败（{type(exc).__name__}: {exc}），改用空操作日志器")

        class _Null:
            def log_event(self, *args: Any, **kwargs: Any) -> None:
                return None

            def bind(self, **kwargs: Any) -> "_Null":
                return self

            def enter(self, *args: Any, **kwargs: Any) -> "_Null":
                return self

            def set_output(self, *args: Any, **kwargs: Any) -> None:
                return None

        return _Null()


def _pct(value: float) -> float:
    return round(float(value) * 100, 2)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[idx], 2)


def load_base_results() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """读 工单7 冻结的评估结果（汇总 + 逐题）。缺失时返回空并显式标注。"""
    summary: dict[str, Any] = {}
    per_q: dict[str, dict[str, Any]] = {}
    try:
        if BASE_EVAL_SUMMARY.is_file():
            summary = json.loads(BASE_EVAL_SUMMARY.read_text(encoding="utf-8"))
        if BASE_EVAL_JSONL.is_file():
            with BASE_EVAL_JSONL.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    per_q[str(row.get("id"))] = row
    except Exception as exc:  # noqa: BLE001 —— 基线读取失败必须留痕，不静默当 0
        summary["_read_error"] = f"{type(exc).__name__}: {exc}"
    return summary, per_q


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单8 Graph RAG 评估：{WORK_ORDER}")
    parser.add_argument("--questions", default=None, help="问题集 jsonl（默认 研发/data/测试问题集.jsonl）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部 17 题）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true", help="跳过 Ollama 生成（只测检索，抽取式兜底）")
    parser.add_argument("--no-ablation", action="store_true", help="跳过「关闭图谱」的消融检索")
    parser.add_argument("--out-tag", default="", help="产物文件名后缀（如 _烟测）")
    args = parser.parse_args(argv)

    cfg = get_config()
    ensure_dirs(cfg)
    log = _logger(Path(cfg.log_dir) / "run_eval.log")
    if not J.available():
        print(f"[ERROR] 判分口径不可用：{J.import_error()}（需要 工单7 研发/评估/判分.py 存在）")
        return 3
    questions_path = Path(args.questions) if args.questions else Path(cfg.question_set)
    questions = [json.loads(line) for line in
                 questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[: int(args.limit)]
    print(f"[1/4] 问题集：{questions_path}（{len(questions)} 题）")

    # 延迟导入引擎：避免 argparse --help 也去加载索引/图谱
    from app.graphrag.engine import GraphRAGEngine

    started = time.perf_counter()
    engine = GraphRAGEngine(cfg=cfg, logger=log)
    warm = engine.warmup(with_llm=not args.no_llm)
    print(f"[2/4] 预热完成：索引 {warm.get('index_chunks')} 块，图谱 {warm.get('graph', {}).get('nodes')} 节点 / "
          f"{warm.get('graph', {}).get('edges')} 边，耗时 {warm.get('elapsed_ms')} ms")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(questions, 1):
        qid = str(item.get("id"))
        question = str(item.get("question"))
        keywords = list(item.get("answer_keywords") or [])
        t0 = time.perf_counter()
        try:
            ans = engine.answer(question, top_k=int(args.top_k), use_graph=True,
                                language=str(item.get("lang") or "auto"), use_llm=not args.no_llm)
            err: str | None = None
        except Exception as exc:  # noqa: BLE001 —— 单题异常不得中断整轮评估（如实计数）
            err = f"{type(exc).__name__}: {exc}"
            log.log_event("eval.question_failed", level="ERROR", id=qid, error=err)
            rows.append({"id": qid, "error": err, "question": question})
            print(f"      [{idx}/{len(questions)}] {qid} 失败：{err}")
            continue
        wall_ms = round((time.perf_counter() - t0) * 1000, 2)
        hits = list(ans.all_chunks)
        judge = J.score_answer(ans.answer, keywords, ans.refused)
        rec5 = J.recall_at_k(hits, keywords, 5)
        rec10 = J.recall_at_k(hits, keywords, 10)
        page = J.page_hit(hits, item.get("expected_pages_hint"),
                          source_filter=item.get("source_file"))
        leak = J.cross_corpus_leak(hits, item.get("source_file"))
        qtype = J.classify_question(question, str(item.get("lang") or "zh"), keywords,
                                    source_file=item.get("source_file"))
        # ---- 消融：关闭图谱的纯向量+BM25 检索（用于「图谱扩展增量」） ----
        ablation: dict[str, Any] = {}
        if not args.no_ablation:
            try:
                base = engine.retrieve(question, top_k=int(args.top_k), use_graph=False)
                base_hits = list(base.get("hits") or [])
                b5 = J.recall_at_k(base_hits, keywords, 5)
                b10 = J.recall_at_k(base_hits, keywords, 10)
                graph_hits = [h for h in hits[:5] if h.get("retrieval_channel") == "graph_only"]
                ablation = {
                    "base_recall_at_5": b5["recall_at_5"], "base_recall_at_10": b10["recall_at_10"],
                    "graph_recall_at_5": rec5["recall_at_5"], "graph_recall_at_10": rec10["recall_at_10"],
                    "recall_at_5_delta": round(rec5["recall_at_5"] - b5["recall_at_5"], 4),
                    "recall_at_10_delta": round(rec10["recall_at_10"] - b10["recall_at_10"], 4),
                    "graph_only_in_top5": len(graph_hits),
                    "graph_only_ids": [h.get("chunk_id") for h in graph_hits],
                    "base_top5_ids": [h.get("chunk_id") for h in base_hits[:5]],
                    "graph_top5_ids": [h.get("chunk_id") for h in hits[:5]],
                }
            except Exception as exc:  # noqa: BLE001 —— 消融失败不影响主结果
                ablation = {"error": f"{type(exc).__name__}: {exc}"}
        row = {
            "id": qid, "question": question, "lang": item.get("lang"), "question_type": qtype,
            "source_pdf": item.get("source_pdf"), "source_file": item.get("source_file"),
            "expected_pages_hint": item.get("expected_pages_hint"), "top_k": int(args.top_k),
            "answer": ans.answer, "answer_chars": len(ans.answer), "refused": ans.refused,
            "generation_method": ans.generation_method, "backend": ans.backend, "model": ans.model,
            "language": ans.language, "trace_id": ans.trace_id,
            "latency_ms": ans.latency_ms, "wall_ms": wall_ms,
            "evidence_count": len(ans.evidence_chunks), "hit_count": len(hits),
            "judge": judge, "recall": rec5 | rec10, "page_hit": page, "leak": leak,
            "graph_used": ans.graph_used,
            "graph": {
                "seed_names": (ans.graph or {}).get("seed_names"),
                "node_total": (ans.graph or {}).get("nodes") and len((ans.graph or {}).get("nodes")),
                "edge_total": (ans.graph or {}).get("edge_total_used"),
                "distances": (ans.graph or {}).get("distances_named"),
                "nodes": (ans.graph or {}).get("nodes"),
                "edges": (ans.graph or {}).get("edges"),
            },
            "diagnostics": ans.diagnostics,
            "ablation": ablation,
            "evidence_chunks": ans.evidence_chunks,
            "top_chunks": [{k: h.get(k) for k in ("rank", "chunk_id", "page", "file_name", "type",
                                                  "score", "base_score", "graph_bonus",
                                                  "graph_distance", "graph_entities",
                                                  "retrieval_channel")} for h in hits],
            "error": err,
        }
        rows.append(row)
        flag = "✓" if judge["is_correct"] else ("~" if judge["partial_correct"] else "✗")
        print(f"      [{idx}/{len(questions)}] {qid} {flag} 覆盖率={judge['keyword_coverage']:.3f} "
              f"recall@5={rec5['recall_at_5']:.3f} 总耗时={ans.latency_ms.get('total', 0):.0f}ms "
              f"图谱块={ablation.get('graph_only_in_top5', '-')}")

    ok_rows = [r for r in rows if not r.get("error")]
    n = len(ok_rows)
    latency = [float(r["latency_ms"].get("total") or 0.0) for r in ok_rows]
    retrieve_ms = [float(r["latency_ms"].get("retrieve") or 0.0) for r in ok_rows]
    graph_ms = [float(r["latency_ms"].get("graph") or 0.0) for r in ok_rows]
    generate_ms = [float(r["latency_ms"].get("generate") or 0.0) for r in ok_rows]
    correct = [r for r in ok_rows if r["judge"]["is_correct"]]
    partial = [r for r in ok_rows if r["judge"]["partial_correct"]]
    refused = [r for r in ok_rows if r.get("refused")]
    recall5 = [float(r["recall"]["recall_at_5"]) for r in ok_rows]
    recall10 = [float(r["recall"]["recall_at_10"]) for r in ok_rows]
    ablation_rows = [r for r in ok_rows if r.get("ablation") and "base_recall_at_5" in r["ablation"]]
    base5 = [float(r["ablation"]["base_recall_at_5"]) for r in ablation_rows]
    graph5 = [float(r["ablation"]["graph_recall_at_5"]) for r in ablation_rows]
    base10 = [float(r["ablation"]["base_recall_at_10"]) for r in ablation_rows]
    graph10 = [float(r["ablation"]["graph_recall_at_10"]) for r in ablation_rows]
    graph_only_qs = [r for r in ablation_rows if int(r["ablation"].get("graph_only_in_top5") or 0) > 0]

    base_summary, base_per_q = load_base_results()
    b_ok = {k: v for k, v in base_per_q.items() if k in {r["id"] for r in ok_rows}}
    b_acc = _mean([1.0 if (v.get("judge") or {}).get("is_correct") else 0.0 for v in b_ok.values()])
    b_recall5 = _mean([float((v.get("judge") or {}).get("recall_at_5") or 0.0) for v in b_ok.values()])
    b_lat = _mean([float(v.get("total_ms") or 0.0) for v in b_ok.values()])

    summary = {
        "工单": WORK_ORDER,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "口径": {
            "问题集": str(questions_path),
            "问题集来源": "工单7 研发/评估/测试问题集.jsonl（17 题）",
            "eval_question.md": "附件目录内未提供该文件，故以 工单7 的 17 题测试集作为检索问题集（设计/需求分析.md §5）",
            "判分": "直接复用 工单7 研发/评估/判分.py（阈值 0.6/0.3/0.6，数字边界口径一致）",
            "top_k": int(args.top_k), "use_cache": bool(cfg.retrieval.get("use_cache", False)),
            "生成模型": "qwen2.5:3b (ollama)" if not args.no_llm else "未启用（抽取式兜底）",
            "embed_model": "bge-m3:latest（工单7 索引，1024 维）",
            "图谱": {"节点": engine.graph.node_count, "边": engine.graph.edge_count,
                     "schema": engine.graph.schema.get("schema_version"),
                     "hop": cfg.retrieval.get("hop"), "graph_slots": cfg.retrieval.get("graph_slots")},
            "配置指纹": cfg.fingerprint(),
        },
        "总数": len(rows), "有效题数": n, "异常题数": len(rows) - n,
        "异常题号": [r["id"] for r in rows if r.get("error")],
        "准确率": {
            "严格判对题数": len(correct), "严格准确率": round(len(correct) / n, 4) if n else 0.0,
            "判对题号": [r["id"] for r in correct],
            "部分正确题数": len(partial), "部分正确题号": [r["id"] for r in partial],
            "宽松准确率(严格+部分)": round((len(correct) + len(partial)) / n, 4) if n else 0.0,
            "平均关键词覆盖率": _mean([float(r["judge"]["keyword_coverage"]) for r in ok_rows]),
        },
        "检索召回": {
            "平均recall@5": _mean(recall5), "平均recall@10": _mean(recall10),
            "recall@5=1.0的题数": sum(1 for v in recall5 if v >= 0.999),
            "recall@10=1.0的题数": sum(1 for v in recall10 if v >= 0.999),
        },
        "响应时间ms": {
            "样本数": n, "平均": _mean(latency), "中位": _percentile(latency, 0.5),
            "P90": _percentile(latency, 0.9), "P95": _percentile(latency, 0.95),
            "最大": round(max(latency), 2) if latency else 0.0,
            "最小": round(min(latency), 2) if latency else 0.0,
            "检索_平均": _mean(retrieve_ms), "图谱扩展_平均": _mean(graph_ms),
            "生成_平均": _mean(generate_ms),
        },
        "SLA": {"阈值ms": J.SLA_MS, "达标题数": sum(1 for v in latency if v <= J.SLA_MS),
                "达标率": round(sum(1 for v in latency if v <= J.SLA_MS) / n, 4) if n else 0.0,
                "超时题号": [r["id"] for r in ok_rows if float(r["latency_ms"].get("total") or 0) > J.SLA_MS]},
        "拒答": {"拒答题数": len(refused), "拒答率": round(len(refused) / n, 4) if n else 0.0,
                 "拒答题号": [r["id"] for r in refused]},
        "页命中": {"命中题数": sum(1 for r in ok_rows if (r.get("page_hit") or {}).get("hit")),
                   "命中率": round(sum(1 for r in ok_rows if (r.get("page_hit") or {}).get("hit")) / n, 4)
                   if n else 0.0},
        "跨语料泄漏": {"题数": sum(1 for r in ok_rows if (r.get("leak") or {}).get("leak")),
                       "题号": [r["id"] for r in ok_rows if (r.get("leak") or {}).get("leak")]},
        # ---- 图谱扩展增量（验收项：图谱扩展带来的增量）----
        "图谱扩展增量": {
            "参与消融的题数": len(ablation_rows),
            "基础检索_recall@5": _mean(base5) if base5 else None,
            "图谱增强_recall@5": _mean(graph5) if graph5 else None,
            "recall@5_提升": round((_mean(graph5) - _mean(base5)), 4) if base5 else None,
            "基础检索_recall@10": _mean(base10) if base10 else None,
            "图谱增强_recall@10": _mean(graph10) if graph10 else None,
            "recall@10_提升": round((_mean(graph10) - _mean(base10)), 4) if base10 else None,
            "图谱独有块进入Top5的题数": len(graph_only_qs),
            "图谱独有块进入Top5的题号": [r["id"] for r in graph_only_qs],
            "图谱独有块进入Top5总数": sum(int(r["ablation"].get("graph_only_in_top5") or 0)
                                          for r in ablation_rows),
            "逐题明细": [{"id": r["id"], **r["ablation"]} for r in ablation_rows],
        },
        # ---- 与 工单7 并排对比 ----
        "与工单7对比": {
            "基线来源": str(BASE_EVAL_JSONL),
            "基线汇总来源": str(BASE_EVAL_SUMMARY),
            "可比题数": len(b_ok),
            "工单7_严格准确率": b_acc, "工单8_严格准确率": round(len(correct) / n, 4) if n else 0.0,
            "严格准确率_提升": round((len(correct) / n if n else 0.0) - b_acc, 4),
            "工单7_平均recall@5": b_recall5, "工单8_平均recall@5": _mean(recall5),
            "recall@5_提升": round(_mean(recall5) - b_recall5, 4),
            "工单7_平均响应时间ms": b_lat,
            "工单8_平均响应时间ms": _mean(latency),
            "响应时间_变化ms": round(_mean(latency) - b_lat, 2),
            "工单7_索引chunk_total": (base_summary.get("配置") or {}).get("索引chunk_total"),
            "基线生成时间": base_summary.get("生成时间"),
        },
        "按题型分组": _group(ok_rows, lambda r: str(r.get("question_type"))),
        "运行元信息": {"warmup": warm, "interpreter": sys.executable, "cwd": str(Path.cwd()),
                       "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                       "config": cfg.to_dict()},
    }

    # ---- 落盘 ----
    data_dir = Path(TEST_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag
    jsonl_path = data_dir / f"GraphRAG评估结果{tag}.jsonl"
    csv_path = data_dir / f"GraphRAG评估结果{tag}.csv"
    sum_path = data_dir / f"GraphRAG评估汇总{tag}.json"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    columns = ["id", "question_type", "lang", "refused", "generation_method", "keyword_coverage",
               "is_correct", "partial_correct", "recall_at_5", "recall_at_10", "page_hit_hit",
               "leak", "total_ms", "retrieve_ms", "graph_ms", "generate_ms", "graph_only_top5",
               "base_recall_at_5", "graph_recall_at_5", "recall_at_5_delta", "answer_chars"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            if row.get("error"):
                writer.writerow([row["id"]] + [""] * (len(columns) - 2) + [row["error"]])
                continue
            ablation = row.get("ablation") or {}
            writer.writerow([
                row["id"], row["question_type"], row["lang"], row["refused"], row["generation_method"],
                row["judge"]["keyword_coverage"], row["judge"]["is_correct"],
                row["judge"]["partial_correct"], row["recall"]["recall_at_5"],
                row["recall"]["recall_at_10"], (row.get("page_hit") or {}).get("hit"),
                (row.get("leak") or {}).get("leak"), row["latency_ms"].get("total"),
                row["latency_ms"].get("retrieve"), row["latency_ms"].get("graph"),
                row["latency_ms"].get("generate"), ablation.get("graph_only_in_top5"),
                ablation.get("base_recall_at_5"), ablation.get("graph_recall_at_5"),
                ablation.get("recall_at_5_delta"), row["answer_chars"],
            ])
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.log_event("eval.done", jsonl=str(jsonl_path), summary=str(sum_path),
                  accuracy=summary["准确率"]["严格准确率"], recall5=summary["检索召回"]["平均recall@5"])
    # 收尾：显式卸载生成模型（资源纪律：不把 3B 模型常驻留给后续任务）
    try:
        import requests

        requests.post(f"{engine.llm.base_url}/api/generate",
                      json={"model": engine.llm.model, "keep_alive": 0}, timeout=5)
        print(f"[收尾] 已请求 Ollama 卸载生成模型（{engine.llm.model}）")
    except Exception as exc:  # noqa: BLE001 —— 卸载失败不影响评估结论
        print(f"[收尾] 卸载生成模型失败（{type(exc).__name__}: {exc}），不影响结果")
    print(f"[3/4] 产物：{jsonl_path.name} / {csv_path.name} / {sum_path.name}")
    print(f"[4/4] 严格准确率 {_pct(summary['准确率']['严格准确率'])}% "
          f"（{summary['准确率']['严格判对题数']}/{n}），平均 recall@5 {summary['检索召回']['平均recall@5']:.4f}，"
          f"平均响应 {summary['响应时间ms']['平均']:.0f} ms")
    print(f"      图谱扩展增量：recall@5 {summary['图谱扩展增量']['基础检索_recall@5']} → "
          f"{summary['图谱扩展增量']['图谱增强_recall@5']}（提升 {summary['图谱扩展增量']['recall@5_提升']}）")
    print(f"      对比工单7：严格准确率 {_pct(summary['与工单7对比']['工单7_严格准确率'])}% → "
          f"{_pct(summary['与工单7对比']['工单8_严格准确率'])}%")
    return 0


def _group(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    out: dict[str, Any] = {}
    for name, members in groups.items():
        n = len(members)
        out[name] = {
            "题数": n,
            "判对题数": sum(1 for r in members if r["judge"]["is_correct"]),
            "判对率": round(sum(1 for r in members if r["judge"]["is_correct"]) / n, 4) if n else 0.0,
            "平均关键词覆盖率": _mean([float(r["judge"]["keyword_coverage"]) for r in members]),
            "平均recall@5": _mean([float(r["recall"]["recall_at_5"]) for r in members]),
            "平均recall@10": _mean([float(r["recall"]["recall_at_10"]) for r in members]),
            "平均总耗时ms": _mean([float(r["latency_ms"].get("total") or 0.0) for r in members]),
            "拒答题数": sum(1 for r in members if r.get("refused")),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
