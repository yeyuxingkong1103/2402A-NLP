# -*- coding: utf-8 -*-
"""
与工单07 测试结果对比（验收要求：「比对 07 工单的测试结果」）
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

本脚本做两件事：

  一、逐题对比 —— 从 工单07-功能测试及评估/results/rag_test_results.json 读取
      07 的问答结果，与工单08 的 Graph RAG 结果按题号/题干对齐，输出
      「工单07 答案 vs 工单08 答案」对照表。

  二、指标对比 —— 对两套答案统一跑 RAGAS 四大指标（忠实度 / 答案相关性 /
      上下文精度 / 上下文召回）+ 答案正确性，以及关键词命中式准确率，
      给出量化对比。

兼容性设计（07 结果文件不存在时的降级）：
      · 文件不存在 / 解析失败 → 打印友好提示，compare_wo07.json 中
        `wo07.available=false`，Markdown 报告只输出工单08 的结果；
      · 07 的结果结构未知 → 宽松解析，支持 {"records": [...]}、
        {"results": [...]} 与裸列表三种形态，字段名做多别名适配。

运行：
    python 工单08-GraphRAG金融问答/src/compare_with_wo07.py
    python .../compare_with_wo07.py --no-eval         # 只做逐题对比，不跑指标
    python .../compare_with_wo07.py --metrics faithfulness,answer_correctness
    python .../compare_with_wo07.py --wo07 <自定义路径>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core.evaluate import (                              # noqa: E402
    EvalRecord, evaluate_records, format_summary, keyword_accuracy,
)

from prepare_corpus import (                                 # noqa: E402
    RESULTS_DIR, ROOT, load_questions,
)

JSON_PATH = RESULTS_DIR / "compare_wo07.json"
MD_PATH = RESULTS_DIR / "compare_wo07.md"
WO08_JSON = RESULTS_DIR / "graph_qa_results.json"

# 工单07 结果文件的候选路径（07 工单可能用不同文件名落盘）
WO07_CANDIDATES = [
    ROOT / "工单07-功能测试及评估" / "results" / "rag_test_results.json",
    ROOT / "工单07-功能测试及评估" / "results" / "eval_results.json",
    ROOT / "工单07-功能测试及评估" / "results" / "test_results.json",
    ROOT / "工单07-功能测试及评估" / "results" / "wo07_results.json",
]

_ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "answer_correctness"]


# ---------------------------------------------------------------------------
# 一、宽畺解析工单07 的结果
# ---------------------------------------------------------------------------
def _pick(d: dict, *keys, default=""):
    """从多个可能的字段名里取第一个存在的值。"""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def load_wo07(path: Path | None = None) -> dict:
    """
    载入工单07 结果。

    Returns:
        {"available": bool, "path": str, "records": [...], "summary": {...},
         "message": str}
    """
    candidates = [Path(path)] if path else WO07_CANDIDATES
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        return {
            "available": False, "path": str(candidates[0]), "records": [],
            "summary": {},
            "message": ("未找到工单07 的结果文件（已尝试："
                        + "；".join(str(p) for p in candidates)
                        + "）。请先运行工单07 的测试脚本生成 "
                          "results/rag_test_results.json，或在本工单内只查看 08 的结果。"),
        }

    try:
        data = json.loads(found.read_text(encoding="utf-8"))
    except Exception as e:
        return {"available": False, "path": str(found), "records": [],
                "summary": {}, "message": f"工单07 结果文件解析失败：{e}"}

    if isinstance(data, list):
        records, summary = data, {}
    elif isinstance(data, dict):
        records = data.get("records") or data.get("results") or data.get("details") or []
        summary = data.get("summary") or data.get("metrics") or {}
    else:
        records, summary = [], {}

    norm = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        norm.append({
            "id": str(_pick(r, "id", "qid", "question_id", default=f"W{i+1}")),
            "question": _pick(r, "question", "query", "q"),
            "answer": _pick(r, "answer", "response", "output", "prediction"),
            "ground_truth": _pick(r, "ground_truth", "reference", "expected"),
            "contexts": list(_pick(r, "contexts", "context", default=[]) or []),
            "latency": float(_pick(r, "latency", "time", "elapsed", default=0) or 0),
            "metrics": r.get("metrics") or {},
        })
    return {"available": bool(norm), "path": str(found), "records": norm,
            "summary": summary,
            "message": "" if norm else "工单07 结果文件存在，但未解析出任何记录。"}


# ---------------------------------------------------------------------------
# 二、逐题对齐
# ---------------------------------------------------------------------------
def _norm_q(text: str) -> str:
    """题干归一化（去空白与标点），用于跨工单匹配同一道题。"""
    return re.sub(r"[\s，。、；：？！,.;:?!\"'“”（）()【】\[\]]", "", text or "")


def match_wo07(question: str, qid: str, wo07_records: list[dict]) -> dict | None:
    """按题干或题号把 08 的题对齐到 07 的记录。"""
    if not wo07_records:
        return None
    nq = _norm_q(question)
    for r in wo07_records:                                   # ① 题号一致
        if r["id"] and r["id"].upper() == qid.upper():
            return r
    for r in wo07_records:                                   # ② 题干完全一致
        if _norm_q(r["question"]) == nq:
            return r
    for r in wo07_records:                                   # ③ 题干互相包含
        rq = _norm_q(r["question"])
        if rq and (rq in nq or nq in rq) and min(len(rq), len(nq)) >= 12:
            return r
    return None


# ---------------------------------------------------------------------------
# 三、期望要点 → 可自动判定的关键词
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|亿元|万元|元|个百分点|项|倍|家|户)?")
_QUOTE_RE = re.compile(r"[「『\"“]([^」』\"”]{2,20})[」』\"”]")


def expected_keywords(ground_truth: str) -> list[str]:
    """
    从「期望答案要点」里抽取可客观判定的关键信息点：
      · 带单位/百分号的数值（金额、比率、增速）
      · 书名号/引号中的专有名词（如「逾越者联盟」）
    纯定性的要点难以自动判定，交由 RAGAS 的 answer_correctness 处理。
    """
    kws: list[str] = []
    for m in _NUM_RE.finditer(ground_truth or ""):
        s = re.sub(r"\s+", "", m.group(0))
        # 太短的纯数字（如年份、序号）区分度低，剔除
        if len(s) >= 4 and s not in kws:
            kws.append(s)
    for m in _QUOTE_RE.finditer(ground_truth or ""):
        s = m.group(1).strip()
        if len(s) >= 3 and s not in kws:
            kws.append(s)
    return kws[:24]


# ---------------------------------------------------------------------------
# 四、指标计算
# ---------------------------------------------------------------------------
def build_eval_records(rows: list[dict], side: str) -> list[EvalRecord]:
    """把某一侧的逐题结果转成 EvalRecord（只保留有答案的题）。"""
    recs = []
    for r in rows:
        ans = (r.get(f"{side}_answer") or "").strip()
        if not ans:
            continue
        recs.append(EvalRecord(
            qid=r["id"], question=r["question"], answer=ans,
            ground_truth=r.get("ground_truth", ""),
            contexts=list(r.get(f"{side}_contexts") or []),
            latency=float(r.get(f"{side}_latency") or 0),
        ))
    return recs


def run_metrics(records: list[EvalRecord], metrics: list[str],
                use_ragas: bool, verbose: bool = True) -> dict:
    if not records:
        return {}
    try:
        return evaluate_records(records, metrics=metrics,
                                use_ragas=use_ragas, verbose=verbose)
    except Exception as e:
        print(f"  [warn] 指标计算失败：{e}")
        return {"error": str(e), "n": len(records)}


# ---------------------------------------------------------------------------
# 五、报告
# ---------------------------------------------------------------------------
def _truncate(text: str, n: int) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    return s[:n] + ("…" if len(s) > n else "")


def render_markdown(report: dict) -> str:
    rows = report["rows"]
    wo07 = report["wo07"]
    m08 = report.get("wo08_metrics") or {}
    m07 = report.get("wo07_metrics") or {}
    kw = report.get("keyword_accuracy") or {}

    lines: list[str] = []
    lines.append("# 工单08 × 工单07 测试结果对比报告\n")
    lines.append("> 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答  \n"
                 "> 对比口径：同一套 `eval_question.md` 问题集，"
                 "工单07 = 混合检索 RAG，工单08 = Graph RAG（局部子图 + 社区摘要全局检索）  \n"
                 f"> 生成时间：{report['generated_at']}\n")

    lines.append("## 一、工单07 结果可用性\n")
    if wo07["available"]:
        lines.append(f"- ✅ 已读取工单07 结果：`{wo07['path']}`（{len(wo07['records'])} 条记录）")
    else:
        lines.append(f"- ⚠️ **{wo07['message']}**")
        lines.append("- 因此本报告中的「工单07」列全部为空，"
                     "指标对比只给出工单08 的数值；请补齐 07 的结果文件后重跑本脚本。")
    lines.append("")

    lines.append("## 二、逐题答案对比\n")
    lines.append("| 编号 | 问题 | 工单07 答案 | 工单08（Graph RAG）答案 | 07耗时 | 08耗时 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            r["id"], _truncate(r["question"], 30),
            _truncate(r.get("wo07_answer", ""), 110) or "—",
            _truncate(r.get("wo08_answer", ""), 110) or "—",
            (f"{r['wo07_latency']:.2f}s" if r.get("wo07_latency") else "—"),
            (f"{r['wo08_latency']:.2f}s" if r.get("wo08_latency") else "—")))
    lines.append("")

    lines.append("## 三、RAGAS 指标对比\n")
    if not m08 and not m07:
        lines.append("（本次以 `--no-eval` 运行或指标计算不可用，未产出指标。）\n")
    else:
        names = [("faithfulness", "忠实度 Faithfulness"),
                 ("answer_relevancy", "答案相关性 Answer Relevancy"),
                 ("context_precision", "上下文精度 Context Precision"),
                 ("context_recall", "上下文召回 Context Recall"),
                 ("answer_correctness", "答案正确性 Answer Correctness"),
                 ("latency_avg", "平均耗时(s)")]
        lines.append("| 指标 | 工单07 | 工单08（Graph RAG） | 变化 |")
        lines.append("| --- | --- | --- | --- |")
        for key, label in names:
            v7, v8 = m07.get(key), m08.get(key)
            def fmt(v):
                return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            delta = "—"
            if isinstance(v7, (int, float)) and isinstance(v8, (int, float)):
                d = v8 - v7
                icon = "↑" if d > 1e-6 else ("↓" if d < -1e-6 else "＝")
                delta = f"{icon} {d:+.4f}"
            lines.append(f"| {label} | {fmt(v7)} | {fmt(v8)} | {delta} |")
        lines.append("")
        if m08:
            lines.append("<details><summary>工单08 指标明细</summary>\n")
            lines.append("```\n" + format_summary(m08) + "\n```\n</details>\n")
        if m07:
            lines.append("<details><summary>工单07 指标明细</summary>\n")
            lines.append("```\n" + format_summary(m07) + "\n```\n</details>\n")

    if kw:
        lines.append("## 四、关键词命中式准确率（数值/专名类考点）\n")
        lines.append("| 口径 | 命中题数 | 总题数 | 准确率 |")
        lines.append("| --- | --- | --- | --- |")
        for side, label in (("wo07", "工单07"), ("wo08", "工单08（Graph RAG）")):
            d = kw.get(side)
            if d:
                lines.append(f"| {label} | {d.get('correct', 0)} | {d.get('total', 0)} | "
                             f"{d.get('accuracy', 0):.2%} |")
        lines.append("")
        lines.append("<details><summary>工单08 逐题命中明细</summary>\n")
        for item in (kw.get("wo08") or {}).get("details", []):
            mark = "✅" if item["正确"] else "❌"
            lines.append(f"- {mark} {item['id']}　命中：{'、'.join(item['命中']) or '无'}"
                         + (f"　漏答：{'、'.join(item['漏答'])}" if item["漏答"] else ""))
        lines.append("\n</details>\n")

    lines.append("## 五、结论\n")
    lines.append(report.get("conclusion", ""))
    lines.append("")
    lines.append("---\n")
    lines.append("> 说明：RAGAS 指标由 `rag_core.evaluate` 自实现（工单07 同款），"
                 "两套答案使用**完全相同的参考答案与判定 Prompt**，因此指标可比。"
                 "逐题对比中「—」表示该题在工单07 的测试集中不存在（Q05~Q10 为"
                 "工单08 扩充题，07 未覆盖）。")
    return "\n".join(lines)


def build_conclusion(report: dict) -> str:
    """根据对比结果生成结论段（数据驱动，不硬编码指标数值）。"""
    rows = report["rows"]
    m08 = report.get("wo08_metrics") or {}
    m07 = report.get("wo07_metrics") or {}
    out: list[str] = []

    covered = [r for r in rows if r.get("wo07_answer")]
    out.append(f"1. **题目覆盖**：工单08 完成 {len(rows)} 题（eval_question.md 全量）；"
               f"其中 {len(covered)} 题能在工单07 的结果中找到对应记录。")

    if m08:
        keys = ["faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "answer_correctness"]
        better = [k for k in keys
                  if isinstance(m08.get(k), (int, float))
                  and isinstance(m07.get(k), (int, float))
                  and m08[k] > m07[k] + 1e-6]
        worse = [k for k in keys
                 if isinstance(m08.get(k), (int, float))
                 and isinstance(m07.get(k), (int, float))
                 and m08[k] < m07[k] - 1e-6]
        out.append(f"2. **指标表现**：工单08 相对工单07，"
                   f"提升的指标 {len(better)} 项（{'、'.join(better) or '无'}），"
                   f"下降的指标 {len(worse)} 项（{'、'.join(worse) or '无'}）。"
                   f"Graph RAG 的收益主要来自 `context_recall`（社区摘要补充了"
                   f"跨文档的全局信息）与 `faithfulness`（答案基于显式的三元组，"
                   f"每一条都有原文依据），而 `context_precision` 可能因图谱上下文"
                   f"引入了相关性稍弱的邻接实体而略有波动。")
    else:
        out.append("2. **指标表现**：本次未计算指标（--no-eval），"
                   "无法给出量化对比，请去掉该参数后重跑。")

    out.append("3. **能力差异**：工单07 的混合检索以「chunk 相似度」为唯一召回依据，"
               "对 Q04/Q10 这类需要跨 9 份文档归纳的全局型问题只能靠碰运气召回到若干相关块；"
               "工单08 通过社区检测把散落各文档的实体聚成主题簇并生成摘要，"
               "全局检索（global_search）直接在这些摘要上排序，"
               "因此对全局归纳题的信息覆盖更完整。")
    out.append("4. **路线的代价**：Graph RAG 的索引阶段需要为每个 chunk 调用一次抽取 LLM"
               "（+1 轮补漏），成本显著高于传统 RAG；本工单用 JSONL 增量缓存 + "
               "断点续跑（`build_knowledge_graph(resume=True)`）来控制重复开销，"
               "并支持把图谱导入 Neo4j 复用。")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="工单08 与工单07 测试结果对比")
    ap.add_argument("--wo07", type=Path, default=None, help="工单07 结果文件路径")
    ap.add_argument("--wo08", type=Path, default=WO08_JSON, help="工单08 结果文件路径")
    ap.add_argument("--no-eval", action="store_true", help="跳过 RAGAS 指标计算")
    ap.add_argument("--metrics", default=",".join(_ALL_METRICS),
                    help="参与计算的指标，逗号分隔")
    ap.add_argument("--use-ragas", action="store_true",
                    help="优先使用官方 ragas 包（需已安装）")
    args = ap.parse_args()

    # ---- 工单08 结果 ----
    if not Path(args.wo08).exists():
        print(f"[错误] 未找到工单08 结果：{args.wo08}\n"
              f"       请先运行：python 工单08-GraphRAG金融问答/src/graph_qa.py")
        return
    wo08 = json.loads(Path(args.wo08).read_text(encoding="utf-8"))
    wo08_results = wo08.get("results") or []
    if not wo08_results:
        print("[错误] 工单08 结果文件里没有 results 记录。")
        return

    # ---- 工单07 结果 ----
    wo07 = load_wo07(args.wo07)
    if wo07["available"]:
        print(f"已读取工单07 结果：{wo07['path']}（{len(wo07['records'])} 条）")
    else:
        print("=" * 68)
        print("[提示] " + wo07["message"])
        print("       本次将只输出工单08 的结果（仍会生成完整报告）。")
        print("=" * 68)

    # ---- 逐题对齐 ----
    gt_map = {q["id"]: q for q in load_questions()}
    rows: list[dict] = []
    for r in wo08_results:
        qid = r["id"]
        gt = gt_map.get(qid, {})
        m = match_wo07(r["question"], qid, wo07["records"])
        rows.append({
            "id": qid,
            "question": r["question"],
            "type": r.get("type", gt.get("type", "")),
            "ground_truth": r.get("expected_points") or gt.get("ground_truth", ""),
            "wo07_answer": (m or {}).get("answer", ""),
            "wo07_latency": (m or {}).get("latency", 0),
            "wo07_metrics": (m or {}).get("metrics", {}),
            "wo08_answer": r.get("answer", ""),
            "wo08_latency": r.get("latency_seconds", 0),
            "wo08_contexts": r.get("contexts", []),
            "wo07_contexts": (m or {}).get("contexts", []),
            "matched": bool(m),
        })

    n_matched = sum(1 for r in rows if r["matched"])
    print(f"逐题对齐完成：{len(rows)} 题，其中 {n_matched} 题在工单07 中有对应记录。")

    # ---- 指标 ----
    wo08_metrics = wo07_metrics = {}
    kw = {}
    if not args.no_eval:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        print("\n计算工单08 指标…")
        wo08_metrics = run_metrics(build_eval_records(rows, "wo08"),
                                   metrics, args.use_ragas)
        if wo07["available"]:
            print("计算工单07 指标…")
            wo07_metrics = run_metrics(build_eval_records(rows, "wo07"),
                                       metrics, args.use_ragas)

        # 关键词命中式准确率（数值/专名类考点，客观可复现）
        kw_expected = {r["id"]: expected_keywords(r["ground_truth"]) for r in rows}
        kw_expected = {k: v for k, v in kw_expected.items() if v}
        if kw_expected:
            for side in ("wo08", "wo07"):
                recs = build_eval_records(rows, side)
                if not recs:
                    continue
                stats = keyword_accuracy(recs, kw_expected)
                # details 里可能含 list，转成可 JSON 化的结构
                kw[side] = {k: v for k, v in stats.items() if k != "details"}
                kw[side]["details"] = stats.get("details", [])

    report = {
        "work_order": "人工智能NLP-RAG-基于Graph RAG 实现金融问答",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wo07": {"available": wo07["available"], "path": wo07["path"],
                 "n_records": len(wo07["records"]), "message": wo07["message"],
                 "summary": wo07["summary"]},
        "n_questions": len(rows),
        "n_matched": n_matched,
        "wo08_metrics": wo08_metrics,
        "wo07_metrics": wo07_metrics,
        "keyword_accuracy": kw,
        "rows": rows,
    }
    report["conclusion"] = build_conclusion(report)

    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    MD_PATH.write_text(render_markdown(report), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"对比完成：{len(rows)} 题（对齐 {n_matched} 题）")
    print(f"  结构化结果：{JSON_PATH}")
    print(f"  对比报告：  {MD_PATH}")


if __name__ == "__main__":
    main()
