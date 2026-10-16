# -*- coding: utf-8 -*-
"""工单8 对比脚本：工单7（基线）vs 工单8（Graph RAG）并排对比表。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

数据来源（**全部是实跑产物，不手写**）：
    * 基线：``E:\\gao6gongdan\\工单7\\测试\\测试数据\\评估汇总.json`` 与 ``评估结果.jsonl``（工单7 冻结结果）
    * 本次：``测试/测试数据/GraphRAG评估汇总.json`` 与 ``GraphRAG评估结果.jsonl``（本工单实跑）

产物：
    * ``测试/测试数据/与工单7对比.json``   —— 四组指标（准确率 / recall@5 / 响应时间 / 图谱扩展增量）
    * ``测试/测试数据/与工单7对比.csv``    —— 逐题并排表（便于人工核对）
    * ``测试/测试数据/对比结论.md``        —— 结论与说明（真实数字 + 未提升项的根因）

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_comparison.py [--tag ""]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(_ROOT / "研发"))

from app.graphrag.config import (BASE_EVAL_JSONL, BASE_EVAL_SUMMARY, TEST_DATA_DIR, WORK_ORDER,  # noqa: E402
                                 ensure_dirs, get_config)

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答

#: 判分阈值（与 工单7 一致，仅用于文档说明；实际判分在 run_eval 内部完成）
KEYWORD_THRESHOLD = 0.6


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_baseline() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary: dict[str, Any] = {}
    if BASE_EVAL_SUMMARY.is_file():
        summary = json.loads(BASE_EVAL_SUMMARY.read_text(encoding="utf-8"))
    per_q = {str(r.get("id")): r for r in _read_jsonl(BASE_EVAL_JSONL)}
    return summary, per_q


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单8 对比表：{WORK_ORDER}")
    parser.add_argument("--tag", default="", help="本次结果文件后缀（默认空）")
    args = parser.parse_args(argv)
    cfg = get_config()
    ensure_dirs(cfg)
    out_dir = Path(TEST_DATA_DIR)
    cur_summary_path = out_dir / f"GraphRAG评估汇总{args.tag}.json"
    cur_jsonl_path = out_dir / f"GraphRAG评估结果{args.tag}.jsonl"
    if not cur_summary_path.is_file() or not cur_jsonl_path.is_file():
        print(f"[ERROR] 缺少本次评估产物：{cur_summary_path} / {cur_jsonl_path}")
        print("        请先跑：pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval.py")
        return 2
    cur_summary = json.loads(cur_summary_path.read_text(encoding="utf-8"))
    cur_rows = {str(r.get("id")): r for r in _read_jsonl(cur_jsonl_path)}
    base_summary, base_rows = _load_baseline()
    if not base_rows:
        print(f"[ERROR] 读不到 工单7 冻结结果：{BASE_EVAL_JSONL}（本机需存在 工单7 目录）")
        return 3

    ids = [qid for qid in cur_rows if qid in base_rows]
    ids.sort()

    def _cur(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        judge = row.get("judge") or {}
        recall = row.get("recall") or {}
        latency = row.get("latency_ms") or {}
        ablation = row.get("ablation") or {}
        table = {"coverage": judge.get("keyword_coverage"), "correct": judge.get("is_correct"),
                 "partial": judge.get("partial_correct"), "recall5": recall.get("recall_at_5"),
                 "recall10": recall.get("recall_at_10"), "total_ms": latency.get("total"),
                 "retrieve_ms": latency.get("retrieve"), "graph_ms": latency.get("graph"),
                 "generate_ms": latency.get("generate"),
                 "base_recall5": ablation.get("base_recall_at_5"),
                 "graph_only5": ablation.get("graph_only_in_top5")}
        value = table.get(key)
        return default if value is None else float(value if not isinstance(value, bool) else int(value))

    def _base(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        judge = row.get("judge") or {}
        table = {"coverage": judge.get("keyword_coverage"), "correct": judge.get("is_correct"),
                 "partial": judge.get("partial_correct"), "recall5": judge.get("recall_at_5"),
                 "recall10": judge.get("recall_at_10"), "total_ms": row.get("total_ms")}
        value = table.get(key)
        return default if value is None else float(value if not isinstance(value, bool) else int(value))

    # ---- 逐题并排 ----
    per_question: list[dict[str, Any]] = []
    for qid in ids:
        c, b = cur_rows[qid], base_rows[qid]
        per_question.append({
            "id": qid,
            "question_type": c.get("question_type") or (b.get("judge") or {}).get("question_type"),
            "lang": c.get("lang"),
            "工单7_覆盖率": _base(b, "coverage"), "工单8_覆盖率": _cur(c, "coverage"),
            "工单7_判对": bool(_base(b, "correct")), "工单8_判对": bool(_cur(c, "correct")),
            "工单7_部分正确": bool(_base(b, "partial")), "工单8_部分正确": bool(_cur(c, "partial")),
            "工单7_recall@5": _base(b, "recall5"), "工单8_recall@5": _cur(c, "recall5"),
            "工单7_recall@10": _base(b, "recall10"), "工单8_recall@10": _cur(c, "recall10"),
            "工单7_总耗时ms": _base(b, "total_ms"), "工单8_总耗时ms": _cur(c, "total_ms"),
            "工单8_检索ms": _cur(c, "retrieve_ms"), "工单8_图谱扩展ms": _cur(c, "graph_ms"),
            "工单8_生成ms": _cur(c, "generate_ms"),
            "工单8_关闭图谱recall@5": _cur(c, "base_recall5"),
            "工单8_图谱独有块数": int(_cur(c, "graph_only5")),
            "工单7_生成方式": b.get("generation_method"), "工单8_生成方式": c.get("generation_method"),
            "覆盖率差": round(_cur(c, "coverage") - _base(b, "coverage"), 4),
            "recall@5差": round(_cur(c, "recall5") - _base(b, "recall5"), 4),
        })

    base_correct = [q for q in per_question if q["工单7_判对"]]
    cur_correct = [q for q in per_question if q["工单8_判对"]]
    base_partial = [q for q in per_question if q["工单7_部分正确"]]
    cur_partial = [q for q in per_question if q["工单8_部分正确"]]
    n = len(per_question)
    base_recall5 = _mean([q["工单7_recall@5"] for q in per_question])
    cur_recall5 = _mean([q["工单8_recall@5"] for q in per_question])
    base_recall10 = _mean([q["工单7_recall@10"] for q in per_question])
    cur_recall10 = _mean([q["工单8_recall@10"] for q in per_question])
    base_lat = _mean([q["工单7_总耗时ms"] for q in per_question])
    cur_lat = _mean([q["工单8_总耗时ms"] for q in per_question])
    base_cov = _mean([q["工单7_覆盖率"] for q in per_question])
    cur_cov = _mean([q["工单8_覆盖率"] for q in per_question])

    ablation = cur_summary.get("图谱扩展增量") or {}
    comparison = {
        "work_order": WORK_ORDER,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "口径": {
            "问题集": "工单7 研发/评估/测试问题集.jsonl（17 题）—— eval_question.md 未随附件提供，替代口径见 设计/需求分析.md §3.5",
            "判分": "直接复用 工单7 研发/评估/判分.py（关键词阈值 0.6 / 部分 0.3 / 原子 0.6）",
            "top_k": cur_summary.get("口径", {}).get("top_k"),
            "基线来源": str(BASE_EVAL_JSONL), "基线汇总": str(BASE_EVAL_SUMMARY),
            "基线生成时间": base_summary.get("生成时间"),
        },
        "可比题数": n,
        "准确率": {
            "工单7_严格判对题数": len(base_correct), "工单7_严格准确率": round(len(base_correct) / n, 4) if n else 0.0,
            "工单8_严格判对题数": len(cur_correct), "工单8_严格准确率": round(len(cur_correct) / n, 4) if n else 0.0,
            "严格准确率_提升": round((len(cur_correct) - len(base_correct)) / n, 4) if n else 0.0,
            "工单7_判对题号": [q["id"] for q in base_correct], "工单8_判对题号": [q["id"] for q in cur_correct],
            "工单7_宽松判对题数": len(base_correct) + len(base_partial),
            "工单8_宽松判对题数": len(cur_correct) + len(cur_partial),
            "工单7_平均关键词覆盖率": base_cov, "工单8_平均关键词覆盖率": cur_cov,
            "平均覆盖率_提升": round(cur_cov - base_cov, 4),
        },
        "检索召回": {
            "工单7_平均recall@5": base_recall5, "工单8_平均recall@5": cur_recall5,
            "recall@5_提升": round(cur_recall5 - base_recall5, 4),
            "工单7_平均recall@10": base_recall10, "工单8_平均recall@10": cur_recall10,
            "recall@10_提升": round(cur_recall10 - base_recall10, 4),
        },
        "响应时间": {
            "工单7_平均ms": base_lat, "工单8_平均ms": cur_lat,
            "变化ms": round(cur_lat - base_lat, 2),
            "工单8_检索平均ms": _mean([q["工单8_检索ms"] for q in per_question]),
            "工单8_图谱扩展平均ms": _mean([q["工单8_图谱扩展ms"] for q in per_question]),
            "工单8_生成平均ms": _mean([q["工单8_生成ms"] for q in per_question]),
            "说明": "两次评测均为 top_k=10、use_cache=False、Ollama qwen2.5:3b；耗时是同机同模型的端到端墙钟",
        },
        "图谱扩展增量": {
            "参与消融题数": ablation.get("参与消融的题数"),
            "基础检索_recall@5": ablation.get("基础检索_recall@5"),
            "图谱增强_recall@5": ablation.get("图谱增强_recall@5"),
            "recall@5_提升": ablation.get("recall@5_提升"),
            "基础检索_recall@10": ablation.get("基础检索_recall@10"),
            "图谱增强_recall@10": ablation.get("图谱增强_recall@10"),
            "recall@10_提升": ablation.get("recall@10_提升"),
            "图谱独有块进入Top5的题数": ablation.get("图谱独有块进入Top5的题数"),
            "图谱独有块进入Top5的题号": ablation.get("图谱独有块进入Top5的题号"),
            "图谱独有块进入Top5总数": ablation.get("图谱独有块进入Top5总数"),
        },
        "逐题": per_question,
    }

    json_path = out_dir / "与工单7对比.json"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "与工单7对比.csv"
    columns = ["id", "question_type", "lang",
               "工单7_覆盖率", "工单8_覆盖率", "覆盖率差",
               "工单7_判对", "工单8_判对",
               "工单7_recall@5", "工单8_recall@5", "recall@5差",
               "工单7_recall@10", "工单8_recall@10",
               "工单7_总耗时ms", "工单8_总耗时ms",
               "工单8_检索ms", "工单8_图谱扩展ms", "工单8_生成ms",
               "工单8_关闭图谱recall@5", "工单8_图谱独有块数",
               "工单7_生成方式", "工单8_生成方式"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in per_question:
            writer.writerow(row)

    # ---- 结论（真实数字 + 未提升项根因） ----
    acc_delta = comparison["准确率"]["严格准确率_提升"]
    r5_delta = comparison["检索召回"]["recall@5_提升"]
    lat_delta = comparison["响应时间"]["变化ms"]
    lines = [
        "# 工单8 vs 工单7 对比结论",
        "",
        f"> 生成时间：{comparison['generated_at']}；可比题数 **{n}**；判分口径与 工单7 完全一致（直接复用其 `判分.py`）。",
        f"> 数据源：`测试/测试数据/与工单7对比.json`（逐题明细同目录 CSV）。",
        "",
        "## 1. 并排对比",
        "",
        "| 指标 | 工单7（基线） | 工单8（Graph RAG） | 变化 |",
        "| --- | --- | --- | --- |",
        f"| 严格准确率 | {comparison['准确率']['工单7_严格准确率']:.2%}（{len(base_correct)}/{n}） | "
        f"{comparison['准确率']['工单8_严格准确率']:.2%}（{len(cur_correct)}/{n}） | {acc_delta:+.2%} |",
        f"| 宽松准确率（严格+部分） | {comparison['准确率']['工单7_宽松判对题数']}/{n} | "
        f"{comparison['准确率']['工单8_宽松判对题数']}/{n} | "
        f"{comparison['准确率']['工单8_宽松判对题数'] - comparison['准确率']['工单7_宽松判对题数']:+d} 题 |",
        f"| 平均关键词覆盖率 | {base_cov:.4f} | {cur_cov:.4f} | {cur_cov - base_cov:+.4f} |",
        f"| 平均 recall@5 | {base_recall5:.4f} | {cur_recall5:.4f} | {r5_delta:+.4f} |",
        f"| 平均 recall@10 | {base_recall10:.4f} | {cur_recall10:.4f} | {cur_recall10 - base_recall10:+.4f} |",
        f"| 平均响应时间 | {base_lat:.0f} ms | {cur_lat:.0f} ms | {lat_delta:+.0f} ms |",
        "",
        "## 2. 图谱扩展带来的增量（消融口径）",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 参与消融的题数 | {ablation.get('参与消融的题数')} |",
        f"| 关闭图谱 recall@5 → 开启图谱 recall@5 | {ablation.get('基础检索_recall@5')} → {ablation.get('图谱增强_recall@5')}"
        f"（{ablation.get('recall@5_提升')}） |",
        f"| 关闭图谱 recall@10 → 开启图谱 recall@10 | {ablation.get('基础检索_recall@10')} → {ablation.get('图谱增强_recall@10')}"
        f"（{ablation.get('recall@10_提升')}） |",
        f"| 图谱独有块进入 Top-5 的题数 | {ablation.get('图谱独有块进入Top5的题数')} 题"
        f"（{ablation.get('图谱独有块进入Top5总数')} 个块） |",
        f"| 题号 | {'、'.join(ablation.get('图谱独有块进入Top5的题号') or []) or '（无）'} |",
        "",
        "## 3. 结论",
        "",
    ]
    if r5_delta > 0:
        lines.append(f"* **图谱扩展在检索召回上是正增益**：平均 recall@5 提升 {r5_delta:+.4f}，"
                     f"{ablation.get('图谱独有块进入Top5的题数')} 道题因图谱独有块进入 Top-5 而获益。")
    else:
        lines.append(f"* **图谱扩展未带来 recall@5 提升（{r5_delta:+.4f}）**：如实登记。"
                     f"根因见 §4（图谱独有块与原 Top-5 高度重叠 / 相关性闸门挡掉了部分候选）。")
    if acc_delta > 0:
        lines.append(f"* **端到端严格准确率提升 {acc_delta:+.2%}**（{len(base_correct)}/{n} → {len(cur_correct)}/{n}）。")
    elif acc_delta == 0:
        lines.append(f"* **端到端严格准确率持平**（{len(base_correct)}/{n}）。"
                     f"回收的召回增量未转化为答案命中 —— 瓶颈在**生成层**（3B 模型对多值枚举题的取舍），"
                     f"属 工单9 优化范围。")
    else:
        lines.append(f"* **端到端严格准确率下降 {acc_delta:+.2%}**：如实登记，根因见 §4。")
    lines += [
        f"* 响应时间变化 {lat_delta:+.0f} ms：图谱扩展本身平均耗时 "
        f"{comparison['响应时间']['工单8_图谱扩展平均ms']:.1f} ms（图检索为内存 BFS，代价很小）；"
        f"总耗时主要仍由 Ollama 生成决定。",
        "",
        "## 4. 未提升项的根因（如实说明）",
        "",
    ]
    if r5_delta <= 0:
        lines += [
            "1. **图谱独有块与向量/BM25 的 Top-5 高度重叠**：本年报语料的答案块（董事长致辞、主要财务指标页）",
            "   本身就在基础检索的 Top-3 内，图谱再补一次仍是同一批块 ⇒ 召回上限已接近饱和；",
            "2. **相关性闸门**（图谱独有块必须含问句实词或命中实体名）挡掉了部分低重合候选，",
            "   这是为精确率做的取舍，代价是召回增量被压低；",
            "3. **关键词覆盖率的分子是「答案文本」**：召回增量若未被 3B 模型写进答案，就不体现在覆盖率上。",
        ]
    lines += [
        "",
        "## 5. 可追溯性",
        "",
        "| 数字 | 来源文件 |",
        "| --- | --- |",
        f"| 工单7 全部基线数字 | `{BASE_EVAL_SUMMARY}` / `{BASE_EVAL_JSONL}`（工单7 冻结产物，只读） |",
        f"| 工单8 逐题数字 | `{cur_jsonl_path}` |",
        f"| 工单8 汇总与消融 | `{cur_summary_path}` |",
        f"| 图谱规模 | `研发/data/graph/图谱构建统计.json` |",
        "",
    ]
    md_path = out_dir / "对比结论.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {json_path.name} / {csv_path.name} / {md_path.name}")
    print(f"     严格准确率 {comparison['准确率']['工单7_严格准确率']:.2%} → "
          f"{comparison['准确率']['工单8_严格准确率']:.2%}（{acc_delta:+.2%}）")
    print(f"     recall@5 {base_recall5:.4f} → {cur_recall5:.4f}（{r5_delta:+.4f}）")
    print(f"     平均响应 {base_lat:.0f} ms → {cur_lat:.0f} ms（{lat_delta:+.0f} ms）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
