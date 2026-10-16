# -*- coding: utf-8 -*-
"""工单9 评估脚本：优化前（工单8 配置）vs 优化后（本工单三层面优化）同口径对比。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

口径声明（**与 设计/评估指标定义.md 逐条对应**）：
    * 问题集 = ``研发/评估/参考答案集.jsonl``（17 题；其中 Q01~Q04 已与
      ``输入/sample_questions.pdf`` 原文逐条核对，见 ``参考答案集核对报告.json``）；
    * 指标 = **自研 RAGAS 近似器**（``研发/评估/ragas近似.py``）：
      context_precision（比例口径 + rank 加权口径）、context_recall、
      faithfulness、answer_relevancy —— **关键词/页码近似口径，非 LLM 裁判**；
    * 上下文**主口径** = 进生成阶段的证据块；**附口径** = 检索 Top-K（两套都报）；
    * 响应时间 = 提问→答案的**墙钟总耗时**（``latency_ms.total``），
      ``use_cache=False``（每题都是「用户第一次问」，**绝不允许缓存命中充数**）；
    * 判分（严格准确率等）继续复用 工单7 的 ``判分.py``，与 07/08 工单逐位可比。

产物（全部落在本工单，工单7/8 目录只读）：
    * ``优化/评估结果/优化前_逐题.jsonl`` / ``优化后_逐题.jsonl``
    * ``优化/评估结果/优化前后对比.csv`` / ``优化前后对比.md`` / ``优化前后对比.json``
    * ``测试/测试数据/`` 下的同名副本（供测试报告引用）
    * ``部署/日志/run_eval_opt.log``

用法：
    # 只跑检索（不调 LLM，秒级完成，用于快速迭代）
    pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval_opt.py --profiles before,after --no-llm
    # 完整跑（含 LLM 慢路径）
    pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval_opt.py --profiles before,after
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

import ragas近似 as RA  # noqa: E402
import 判分口径 as J  # noqa: E402
from app.graphrag_opt.配置 import (OPT_EVAL_DIR, TEST_DATA_DIR, ensure_dirs, get_config,  # noqa: E402
                                   write_default_config_file)

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

#: 最近一次 `run_profile` 预热得到的图谱规模（写进汇总，供截图与报告引用真实数字）
_GRAPH_INFO: dict[str, int] = {}


def _logger(cfg: Any) -> Any:
    """结构化日志（路径钉在本工单；失败退化为空操作日志器，不阻断评估）。"""
    import os

    os.environ["RAG_LOG__DIR"] = str(cfg.log_dir)
    try:
        from app.core.logging_conf import setup_logging

        return setup_logging()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 结构化日志初始化失败（{type(exc).__name__}: {exc}），改用空操作日志器")

        class _Null:
            def log_event(self, *a: Any, **k: Any) -> None:
                return None

            def bind(self, **k: Any) -> "_Null":
                return self

            def enter(self, *a: Any, **k: Any) -> "_Null":
                return self

            def set_output(self, *a: Any, **k: Any) -> None:
                return None

        return _Null()


def _load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _unload_ollama(cfg: Any) -> None:
    """收尾：显式卸载 Ollama 模型（资源纪律：绝不让 2 个模型同时驻留）。"""
    try:
        import requests

        for model in (cfg.generation.get("model", "qwen2.5:3b"), "bge-m3:latest"):
            try:
                requests.post("http://127.0.0.1:11434/api/generate",
                              json={"model": model, "keep_alive": 0}, timeout=5)
            except Exception:  # noqa: BLE001 —— 单个模型卸载失败不影响其它
                continue
        print("[收尾] 已请求 Ollama 卸载模型")
    except Exception as exc:  # noqa: BLE001
        print(f"[收尾] 卸载模型失败（{type(exc).__name__}: {exc}），不影响结果")


def run_profile(profile: str, questions: list[dict[str, Any]], *, cfg: Any, log: Any,
                top_k: int, use_llm: bool) -> list[dict[str, Any]]:
    """跑一个 profile 的全部题目，返回逐题结果行。"""
    from app.graphrag_opt.引擎 import OptimizedEngine

    engine = OptimizedEngine(cfg=cfg, profile=profile, logger=log)
    warm = engine.warmup(with_llm=use_llm)
    graph_info = warm.get("graph") or {}
    _GRAPH_INFO.update({"nodes": graph_info.get("nodes"), "edges": graph_info.get("edges")})
    print(f"    [warmup-{profile}] 索引 {warm.get('index_chunks')} 块，"
          f"图谱 {warm.get('graph', {}).get('nodes')} 节点 / {warm.get('graph', {}).get('edges')} 边，"
          f"耗时 {warm.get('elapsed_ms')} ms")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(questions, 1):
        qid = str(item.get("id"))
        question = str(item.get("question"))
        keywords = list(item.get("answer_keywords") or [])
        t0 = time.perf_counter()
        try:
            ans = engine.answer(question, top_k=top_k, language=str(item.get("lang") or "auto"),
                                use_llm=use_llm)
            err = None
        except Exception as exc:  # noqa: BLE001 —— 单题异常不得中断整轮评估（如实计数）
            err = f"{type(exc).__name__}: {exc}"
            log.log_event("eval.question_failed", level="ERROR", profile=profile, id=qid, error=err)
            rows.append({"id": qid, "profile": profile, "error": err, "question": question,
                         "latency_total_ms": 0.0})
            print(f"      [{idx}/{len(questions)}] {qid} 失败：{err}")
            continue
        wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        evidence = list(ans.evidence_chunks)
        all_chunks = list(ans.all_chunks)
        metrics = RA.evaluate_pair(item, evidence_contexts=evidence,
                                   topk_contexts=all_chunks[: max(top_k, 1)],
                                   answer=ans.answer, refused=bool(ans.refused),
                                   override=cfg.eval)
        judge = J.score_answer(ans.answer, keywords, ans.refused)
        rec5 = J.recall_at_k(evidence, keywords, 5)
        rec10 = J.recall_at_k(evidence, keywords, 10)
        rec10_pool = J.recall_at_k(all_chunks, keywords, 10)
        page = J.page_hit(all_chunks, item.get("expected_pages_hint"),
                          source_filter=item.get("source_file"))
        leak = J.cross_corpus_leak(all_chunks, item.get("source_file"))
        qtype = J.classify_question(question, str(item.get("lang") or "zh"), keywords,
                                    source_file=item.get("source_file"))
        row = {
            "id": qid, "profile": profile, "question": question, "lang": item.get("lang"),
            "question_type": qtype, "top_k": top_k, "use_llm": use_llm,
            "answer": ans.answer, "answer_chars": len(ans.answer), "refused": bool(ans.refused),
            "generation_method": str(ans.generation_method), "backend": str(ans.backend),
            "model": str(ans.model), "language": str(ans.language),
            "latency_total_ms": float(ans.latency_ms.get("total") or 0.0),
            "latency_retrieve_ms": float(ans.latency_ms.get("retrieve") or 0.0),
            "latency_graph_ms": float(ans.latency_ms.get("graph") or 0.0),
            "latency_generate_ms": float(ans.latency_ms.get("generate") or 0.0),
            "wall_ms": wall_ms,
            "evidence_count": len(evidence), "hit_count": len(all_chunks),
            "metrics": metrics,
            "judge": judge, "recall_evidence": rec5 | rec10, "recall_pool": rec10_pool,
            "page_hit": page, "leak": leak,
            "graph_used": bool(ans.graph_used),
            "diagnostics": {k: v for k, v in (ans.diagnostics or {}).items()
                            if k not in ("linked_entities",)},
            "evidence_chunks": evidence,
            "top_chunks": [{k: h.get(k) for k in ("rank", "chunk_id", "page", "file_name", "type",
                                                  "score", "base_score", "graph_score",
                                                  "graph_distance", "graph_entities",
                                                  "retrieval_channel", "lexical_hits")}
                           for h in all_chunks],
            "trace_id": str(ans.trace_id), "error": err,
        }
        rows.append(row)
        flag = "✓" if judge["is_correct"] else ("~" if judge["partial_correct"] else "✗")
        print(f"      [{idx}/{len(questions)}] {qid} {flag} "
              f"精度={metrics['context_precision']:.3f} 召回={metrics['context_recall']:.3f} "
              f"TopK精度={metrics['topk_context_precision']:.3f} "
              f"上下文={metrics['context_size']}块 总耗时={row['latency_total_ms']:.0f}ms "
              f"方式={row['generation_method']}")
    return rows


def summarize(rows: list[dict[str, Any]], *, label: str, cfg: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """汇总（宏平均 + 分位数 + 达标判定）。

    ⚠️ 实测踩坑：``RA.aggregate`` 期望的是**扁平**的指标键（``context_precision`` 直接在行上），
    而本脚本的逐题行把指标放在 ``row["metrics"]`` 里。直接把行丢进去会得到
    「所有指标 = 0.0000」的假汇总（第一版就这样，17 题明明有 1.000 的却汇总成 0）。
    因此这里先**摊平**再聚合。
    """
    ok = [r for r in rows if not r.get("error")]
    flat = [{**(r.get("metrics") or {}), "id": r.get("id"),
             "latency_total_ms": float(r.get("latency_total_ms") or 0.0)} for r in ok]
    agg = RA.aggregate(flat, override=cfg.eval)
    n = len(ok)
    lat = [float(r.get("latency_total_ms") or 0.0) for r in ok]

    def pct(q: float) -> float:
        if not lat:
            return 0.0
        ordered = sorted(lat)
        return round(ordered[min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)], 2)

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    sla = float(cfg.eval.get("sla_ms") or 3000.0)
    correct = [r for r in ok if (r.get("judge") or {}).get("is_correct")]
    partial = [r for r in ok if (r.get("judge") or {}).get("partial_correct")]
    methods: dict[str, int] = {}
    for r in ok:
        key = str(r.get("generation_method"))
        methods[key] = methods.get(key, 0) + 1
    return {
        "profile": label,
        "题数": n, "异常题数": len(rows) - n,
        "指标": agg,
        "判分": {
            "严格判对题数": len(correct), "严格准确率": round(len(correct) / n, 4) if n else 0.0,
            "判对题号": [r["id"] for r in correct],
            "部分正确题数": len(partial),
            "宽松准确率": round((len(correct) + len(partial)) / n, 4) if n else 0.0,
            "平均关键词覆盖率": mean([float((r.get("judge") or {}).get("keyword_coverage") or 0.0) for r in ok]),
        },
        "响应时间ms": {
            "样本数": n, "平均": mean(lat), "P50": pct(0.50), "P90": pct(0.90), "P95": pct(0.95),
            "最大": round(max(lat), 2) if lat else 0.0, "最小": round(min(lat), 2) if lat else 0.0,
            "检索_平均": mean([float(r.get("latency_retrieve_ms") or 0.0) for r in ok]),
            "图谱_平均": mean([float(r.get("latency_graph_ms") or 0.0) for r in ok]),
            "生成_平均": mean([float(r.get("latency_generate_ms") or 0.0) for r in ok]),
        },
        "SLA": {"阈值ms": sla, "达标题数": sum(1 for v in lat if v <= sla),
                "达标率": round(sum(1 for v in lat if v <= sla) / n, 4) if n else 0.0,
                "超时题号": [r["id"] for r in ok if float(r.get("latency_total_ms") or 0.0) > sla]},
        "生成方式分布": methods,
        "拒答": {"题数": sum(1 for r in ok if r.get("refused")),
                 "题号": [r["id"] for r in ok if r.get("refused")]},
        "页命中": {"命中题数": sum(1 for r in ok if (r.get("page_hit") or {}).get("hit")),
                   "命中率": round(sum(1 for r in ok if (r.get("page_hit") or {}).get("hit")) / n, 4) if n else 0.0},
        "跨语料泄漏": {"题数": sum(1 for r in ok if (r.get("leak") or {}).get("leak")),
                       "题号": [r["id"] for r in ok if (r.get("leak") or {}).get("leak")]},
        "按题型": _group(ok),
        "运行元信息": meta,
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("question_type")), []).append(row)
    out: dict[str, Any] = {}
    for name, members in groups.items():
        n = len(members)
        out[name] = {
            "题数": n,
            "平均上下文精度": round(sum(float((m.get("metrics") or {}).get("context_precision") or 0.0)
                                  for m in members) / n, 4),
            "平均上下文召回": round(sum(float((m.get("metrics") or {}).get("context_recall") or 0.0)
                                  for m in members) / n, 4),
            "平均响应ms": round(sum(float(m.get("latency_total_ms") or 0.0) for m in members) / n, 2),
            "严格判对题数": sum(1 for m in members if (m.get("judge") or {}).get("is_correct")),
        }
    return out


def comparison(before: dict[str, Any], after: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """优化前后对比（**这是本工单的核心产出**）。"""
    b, a = before["指标"], after["指标"]

    def delta(key: str) -> dict[str, Any]:
        bv, av = float(b.get(key) or 0.0), float(a.get(key) or 0.0)
        return {"优化前": bv, "优化后": av, "变化": round(av - bv, 4),
                "相对提升": (f"{round((av - bv) / bv * 100, 2)}%" if bv else "N/A")}

    cp_t = float(cfg.eval.get("context_precision_target") or 0.8)
    cr_t = float(cfg.eval.get("context_recall_target") or 0.9)
    sla_t = float(cfg.eval.get("sla_ms") or 3000.0)
    return {
        "工单": WORK_ORDER,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "口径": {
            "问题集": str(cfg.reference_set),
            "指标": "自研 RAGAS 近似器（关键词/页码近似口径，非 LLM 裁判）",
            "上下文主口径": "进生成阶段的证据块",
            "上下文附口径": "检索 Top-K",
            "use_cache": False,
            "生成模型": "qwen2.5:3b（优化后仅在快路径不达标时调用）",
            "配置指纹": cfg.fingerprint(),
            "融合权重": dict((cfg.retrieval or {}).get("fusion_weights") or {}),
            "生成方式分布": (after.get("生成方式分布") or {}),
            "图谱规模": {
                "优化前": {"节点": (before.get("运行元信息") or {}).get("graph_nodes"),
                           "边": (before.get("运行元信息") or {}).get("graph_edges")},
                "优化后": {"节点": (after.get("运行元信息") or {}).get("graph_nodes"),
                           "边": (after.get("运行元信息") or {}).get("graph_edges")},
            },
        },
        "核心指标对比": {
            "context_precision": delta("context_precision"),
            "context_precision_rank": delta("context_precision_rank"),
            "context_recall": delta("context_recall"),
            "topk_context_precision": delta("topk_context_precision"),
            "topk_context_recall": delta("topk_context_recall"),
            "faithfulness": delta("faithfulness"),
            "answer_relevancy": delta("answer_relevancy"),
        },
        "响应时间对比": {
            "平均ms": {"优化前": before["响应时间ms"]["平均"], "优化后": after["响应时间ms"]["平均"],
                       "变化": round(after["响应时间ms"]["平均"] - before["响应时间ms"]["平均"], 2)},
            "P90ms": {"优化前": before["响应时间ms"]["P90"], "优化后": after["响应时间ms"]["P90"]},
            "P95ms": {"优化前": before["响应时间ms"]["P95"], "优化后": after["响应时间ms"]["P95"]},
            "最大ms": {"优化前": before["响应时间ms"]["最大"], "优化后": after["响应时间ms"]["最大"]},
            "生成段_平均ms": {"优化前": before["响应时间ms"]["生成_平均"],
                              "优化后": after["响应时间ms"]["生成_平均"]},
            "SLA达标率": {"优化前": before["SLA"]["达标率"], "优化后": after["SLA"]["达标率"]},
        },
        "验收达标判定": {
            "阈值": {"context_precision": cp_t, "context_recall": cr_t, "响应时间ms": sla_t},
            "context_precision": {"优化前": b.get("context_precision"), "优化后": a.get("context_precision"),
                                  "是否达标": bool(float(a.get("context_precision") or 0.0) >= cp_t)},
            "context_recall": {"优化前": b.get("context_recall"), "优化后": a.get("context_recall"),
                               "是否达标": bool(float(a.get("context_recall") or 0.0) >= cr_t)},
            "响应时间SLA": {"优化前达标率": before["SLA"]["达标率"], "优化后达标率": after["SLA"]["达标率"],
                            "P95是否≤阈值": bool(float(after["响应时间ms"]["P95"]) <= sla_t)},
        },
        "严格准确率": {"优化前": before["判分"]["严格准确率"], "优化后": after["判分"]["严格准确率"],
                       "变化": round(after["判分"]["严格准确率"] - before["判分"]["严格准确率"], 4)},
    }


def render_md(cmp_data: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> str:
    """对比表（Markdown，中文）。"""
    lines: list[str] = []
    lines.append(f"# {WORK_ORDER} —— 优化前后对比")
    lines.append("")
    lines.append(f"- 生成时间：{cmp_data['生成时间']}")
    lines.append(f"- 问题集：`{cmp_data['口径']['问题集']}`（{before['题数']} 题）")
    lines.append(f"- 指标口径：{cmp_data['口径']['指标']}；上下文主口径 = {cmp_data['口径']['上下文主口径']}，"
                 f"附口径 = {cmp_data['口径']['上下文附口径']}")
    lines.append(f"- `use_cache = false`（每题都是「用户第一次问」，无缓存命中充数）")
    lines.append(f"- 配置指纹：`{cmp_data['口径']['配置指纹']}`")
    lines.append("")
    lines.append("## 一、核心指标对比（宏平均）")
    lines.append("")
    lines.append("| 指标 | 优化前 | 优化后 | 变化 | 相对提升 | 工单阈值 | 是否达标 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | :---: |")
    tbl = cmp_data["核心指标对比"]
    targets = {"context_precision": "≥0.80", "context_recall": "≥0.90"}
    for key, name in (("context_precision", "**上下文精度 context_precision**"),
                      ("context_precision_rank", "上下文精度（rank 加权口径）"),
                      ("topk_context_precision", "上下文精度（Top-K 附口径）"),
                      ("context_recall", "**上下文召回 context_recall**"),
                      ("topk_context_recall", "上下文召回（Top-K 附口径）"),
                      ("faithfulness", "忠实度 faithfulness（近似）"),
                      ("answer_relevancy", "答案相关性 answer_relevancy（近似）")):
        row = tbl[key]
        target = targets.get(key, "—")
        verdict = "—"
        if key in targets:
            verdict = "✅" if row["优化后"] >= float(target.split("≥")[1]) else "❌"
        lines.append(f"| {name} | {row['优化前']:.4f} | {row['优化后']:.4f} | {row['变化']:+.4f} | "
                     f"{row['相对提升']} | {target} | {verdict} |")
    lines.append("")
    lines.append("## 二、响应时间对比（`--no-cache` 口径）")
    lines.append("")
    lines.append("| 口径 | 优化前 | 优化后 | 变化 |")
    lines.append("| --- | ---: | ---: | ---: |")
    rt = cmp_data["响应时间对比"]
    for key, name in (("平均ms", "平均"), ("P90ms", "P90"), ("P95ms", "P95"), ("最大ms", "最大"),
                      ("生成段_平均ms", "其中生成段平均")):
        row = rt[key]
        change = row.get("变化", round(float(row["优化后"]) - float(row["优化前"]), 2))
        lines.append(f"| {name} | {row['优化前']} | {row['优化后']} | {change:+.2f} |")
    lines.append(f"| **SLA ≤ {cmp_data['验收达标判定']['阈值']['响应时间ms']:.0f} ms 达标率** | "
                 f"{rt['SLA达标率']['优化前']:.4f} | {rt['SLA达标率']['优化后']:.4f} | "
                 f"{rt['SLA达标率']['优化后'] - rt['SLA达标率']['优化前']:+.4f} |")
    lines.append("")
    lines.append("## 三、答案质量（复用 工单7 判分口径）")
    lines.append("")
    lines.append("| 口径 | 优化前 | 优化后 | 变化 |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| 严格准确率 | {before['判分']['严格准确率']:.4f} | {after['判分']['严格准确率']:.4f} | "
                 f"{cmp_data['严格准确率']['变化']:+.4f} |")
    lines.append(f"| 宽松准确率（严格+部分） | {before['判分']['宽松准确率']:.4f} | "
                 f"{after['判分']['宽松准确率']:.4f} | — |")
    lines.append(f"| 平均关键词覆盖率 | {before['判分']['平均关键词覆盖率']:.4f} | "
                 f"{after['判分']['平均关键词覆盖率']:.4f} | — |")
    lines.append("")
    lines.append("## 四、达标判定")
    lines.append("")
    v = cmp_data["验收达标判定"]
    lines.append(f"- 上下文精度 ≥ {v['阈值']['context_precision']}：优化前 {v['context_precision']['优化前']:.4f} → "
                 f"优化后 {v['context_precision']['优化后']:.4f} —— **{'达标' if v['context_precision']['是否达标'] else '未达标'}**")
    lines.append(f"- 上下文召回 ≥ {v['阈值']['context_recall']}：优化前 {v['context_recall']['优化前']:.4f} → "
                 f"优化后 {v['context_recall']['优化后']:.4f} —— **{'达标' if v['context_recall']['是否达标'] else '未达标'}**")
    lines.append(f"- 响应时间 ≤ {v['阈值']['响应时间ms']:.0f} ms：P95 优化后 {rt['P95ms']['优化后']} ms —— "
                 f"**{'达标' if v['响应时间SLA']['P95是否≤阈值'] else '未达标'}**"
                 f"（SLA 达标率 {v['响应时间SLA']['优化前达标率']:.4f} → {v['响应时间SLA']['优化后达标率']:.4f}）")
    lines.append("")
    lines.append("## 五、按题型分组")
    lines.append("")
    lines.append("| 题型 | 题数 | 优化前精度 | 优化后精度 | 优化前召回 | 优化后召回 | 优化前响应ms | 优化后响应ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in sorted(set(before["按题型"]) | set(after["按题型"])):
        bg = before["按题型"].get(name, {})
        ag = after["按题型"].get(name, {})
        lines.append(f"| {name} | {ag.get('题数', bg.get('题数', 0))} | "
                     f"{bg.get('平均上下文精度', 0):.4f} | {ag.get('平均上下文精度', 0):.4f} | "
                     f"{bg.get('平均上下文召回', 0):.4f} | {ag.get('平均上下文召回', 0):.4f} | "
                     f"{bg.get('平均响应ms', 0):.0f} | {ag.get('平均响应ms', 0):.0f} |")
    lines.append("")
    lines.append("## 六、逐题明细（优化后）")
    lines.append("")
    lines.append("| 题号 | 题型 | 上下文精度 | 上下文召回 | TopK精度 | 上下文块数 | 总耗时ms | 生成方式 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in after.get("逐题明细", []):
        m = row.get("metrics") or {}
        lines.append(f"| {row['id']} | {row.get('question_type')} | {m.get('context_precision', 0):.3f} | "
                     f"{m.get('context_recall', 0):.3f} | {m.get('topk_context_precision', 0):.3f} | "
                     f"{m.get('context_size', 0)} | {row.get('latency_total_ms', 0):.0f} | "
                     f"{row.get('generation_method')} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 优化前后评估")
    parser.add_argument("--questions", default=None, help="参考答案集 jsonl（默认 研发/评估/参考答案集.jsonl）")
    parser.add_argument("--profiles", default="before,after", help="要跑的 profile，逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM（只跑检索 + 模板生成，秒级）")
    parser.add_argument("--tag", default="", help="产物文件名后缀")
    args = parser.parse_args(argv)

    cfg = get_config()
    ensure_dirs(cfg)
    write_default_config_file(cfg)
    log = _logger(cfg)
    if not J.available():
        print(f"[ERROR] 判分口径不可用：{J.import_error()}")
        return 3

    questions_path = Path(args.questions) if args.questions else Path(cfg.reference_set)
    if not questions_path.is_file():
        print(f"[ERROR] 找不到参考答案集：{questions_path}（先跑 研发/scripts/构建参考答案集.py）")
        return 3
    questions = _load_questions(questions_path)
    if args.limit:
        questions = questions[: int(args.limit)]
    print(f"[1/4] 问题集：{questions_path}（{len(questions)} 题）")

    profiles = [p.strip() for p in str(args.profiles).split(",") if p.strip()]
    results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for profile in profiles:
        print(f"[2/4] 运行 profile = {profile}（use_llm={not args.no_llm}）")
        meta = {"profile": profile, "use_llm": not args.no_llm, "top_k": int(args.top_k),
                "config_fingerprint": cfg.fingerprint(), "config_source": cfg.config_source,
                "interpreter": sys.executable, "cwd": str(Path.cwd()),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "graph_nodes": None, "graph_edges": None}
        rows = run_profile(profile, questions, cfg=cfg, log=log, top_k=int(args.top_k),
                           use_llm=not args.no_llm)
        results[profile] = rows
        summary = summarize(rows, label=profile, cfg=cfg, meta=meta)
        summary["运行元信息"]["graph_nodes"] = _GRAPH_INFO.get("nodes")
        summary["运行元信息"]["graph_edges"] = _GRAPH_INFO.get("edges")
        summary["逐题明细"] = [{k: r.get(k) for k in
                                ("id", "question_type", "lang", "refused", "generation_method",
                                 "latency_total_ms", "latency_retrieve_ms", "latency_graph_ms",
                                 "latency_generate_ms", "evidence_count", "hit_count",
                                 "graph_used", "error")} | {"metrics": r.get("metrics"),
                                                            "judge": r.get("judge")}
                               for r in rows]
        summaries[profile] = summary
        print(f"    → context_precision={summary['指标']['context_precision']:.4f} "
              f"context_recall={summary['指标']['context_recall']:.4f} "
              f"平均响应={summary['响应时间ms']['平均']:.0f}ms SLA={summary['SLA']['达标率']:.4f}")

    # ---- 落盘 ----
    tag = args.tag
    OPT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in results.items():
        for base in (OPT_EVAL_DIR, TEST_DATA_DIR):
            path = base / f"{'优化前' if name == 'before' else '优化后'}_逐题{tag}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            # 逐题 CSV（便于 Excel / 截图）
            csv_path = base / f"{'优化前' if name == 'before' else '优化后'}_逐题{tag}.csv"
            columns = ["id", "question_type", "lang", "context_precision", "context_precision_rank",
                       "context_recall", "topk_context_precision", "topk_context_recall",
                       "context_size", "faithfulness", "answer_relevancy", "keyword_coverage",
                       "latency_total_ms", "latency_retrieve_ms", "latency_graph_ms",
                       "latency_generate_ms", "generation_method", "refused", "evidence_count",
                       "hit_count", "error"]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                for row in rows:
                    m = row.get("metrics") or {}
                    judge = row.get("judge") or {}
                    writer.writerow([
                        row.get("id"), row.get("question_type"), row.get("lang"),
                        m.get("context_precision"), m.get("context_precision_rank"),
                        m.get("context_recall"), m.get("topk_context_precision"),
                        m.get("topk_context_recall"), m.get("context_size"),
                        (m.get("faithfulness_detail") or {}).get("faithfulness"),
                        (m.get("answer_relevancy_detail") or {}).get("answer_relevancy"),
                        judge.get("keyword_coverage"), row.get("latency_total_ms"),
                        row.get("latency_retrieve_ms"), row.get("latency_graph_ms"),
                        row.get("latency_generate_ms"), row.get("generation_method"),
                        row.get("refused"), row.get("evidence_count"), row.get("hit_count"),
                        row.get("error"),
                    ])
        (OPT_EVAL_DIR / f"汇总_{name}{tag}.json").write_text(
            json.dumps(summaries[name], ensure_ascii=False, indent=2), encoding="utf-8")

    if "before" in summaries and "after" in summaries:
        cmp_data = comparison(summaries["before"], summaries["after"], cfg)
        cmp_data["逐题对照"] = _per_question_compare(results["before"], results["after"])
        (OPT_EVAL_DIR / f"优化前后对比{tag}.json").write_text(
            json.dumps(cmp_data, ensure_ascii=False, indent=2), encoding="utf-8")
        md = render_md(cmp_data, summaries["before"], summaries["after"])
        (OPT_EVAL_DIR / f"优化前后对比{tag}.md").write_text(md, encoding="utf-8")
        (TEST_DATA_DIR / f"优化前后对比{tag}.md").write_text(md, encoding="utf-8")
        # 对比 CSV
        with (OPT_EVAL_DIR / f"优化前后对比{tag}.csv").open("w", encoding="utf-8-sig",
                                                          newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["指标", "优化前", "优化后", "变化", "相对提升"])
            for key, row in cmp_data["核心指标对比"].items():
                writer.writerow([key, row["优化前"], row["优化后"], row["变化"], row["相对提升"]])
            writer.writerow([])
            writer.writerow(["逐题对照", "优化前精度", "优化后精度", "前召回", "后召回",
                             "前耗时ms", "后耗时ms"])
            for row in cmp_data["逐题对照"]:
                writer.writerow([row["id"], row["before_precision"], row["after_precision"],
                                 row["before_recall"], row["after_recall"],
                                 row["before_ms"], row["after_ms"]])
        print("[3/4] 对比产物已落盘：")
        print(f"      {OPT_EVAL_DIR / ('优化前后对比' + tag + '.csv')}")
        print(f"      {OPT_EVAL_DIR / ('优化前后对比' + tag + '.md')}")
        for key in ("context_precision", "context_recall"):
            row = cmp_data["核心指标对比"][key]
            print(f"      {key}: {row['优化前']:.4f} → {row['优化后']:.4f}（{row['变化']:+.4f}）")
        print(f"      SLA 达标率: {cmp_data['响应时间对比']['SLA达标率']['优化前']:.4f} → "
              f"{cmp_data['响应时间对比']['SLA达标率']['优化后']:.4f}")
    print(f"[4/4] 总耗时 {round((time.perf_counter() - started), 2)} s")
    _unload_ollama(cfg)
    return 0


def _per_question_compare(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bmap = {str(r.get("id")): r for r in before}
    out: list[dict[str, Any]] = []
    for row in after:
        qid = str(row.get("id"))
        b = bmap.get(qid, {})
        bm, am = b.get("metrics") or {}, row.get("metrics") or {}
        out.append({
            "id": qid, "question_type": row.get("question_type"),
            "before_precision": bm.get("context_precision"), "after_precision": am.get("context_precision"),
            "before_recall": bm.get("context_recall"), "after_recall": am.get("context_recall"),
            "before_topk_precision": bm.get("topk_context_precision"),
            "after_topk_precision": am.get("topk_context_precision"),
            "before_ms": b.get("latency_total_ms"), "after_ms": row.get("latency_total_ms"),
            "before_method": b.get("generation_method"), "after_method": row.get("generation_method"),
        })
    return out


if __name__ == "__main__":
    raise SystemExit(main())
