# -*- coding: utf-8 -*-
"""
消融对比实验：不含图像解析 vs 含图像解析
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

这是工单04 的**核心证据**：同一套分块/检索/生成配置，只切换「图像语义块是否入库」，
在 16 个问题上对比准确率与检索行为，重点看两个图像问题（id 5、id 6）。

    对照组（A）wo04_text       —— 文本 + 表格（工单03 水平）
    实验组（B）wo04_multimodal —— 文本 + 表格 + 图像语义描述（工单04）

预期结论：
    · 两个图像问题在 A 组必然答错（信息不在文本层），B 组答对；
    · 其余 14 问准确率基本持平，说明图像块不引入明显干扰；
    · 检索耗时几乎不变（多出的图像块数量很少，见索引统计）。

用法：
    python src/compare_with_without_images.py               # 复用已有索引
    python src/compare_with_without_images.py --rebuild     # 强制重建两个索引
    python src/compare_with_without_images.py --ragas       # 追加 RAGAS 指标
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_extractor import RESULTS_DIR  # noqa: E402
from rag_core import config  # noqa: E402

ABLATION_JSON = RESULTS_DIR / "image_ablation.json"
ABLATION_MD = RESULTS_DIR / "image_ablation.md"
INDEX_STATS_JSON = RESULTS_DIR / "index_stats.json"

GROUP_A = ("text_only", "wo04_text", "不含图像解析（文本+表格）")
GROUP_B = ("with_images", "wo04_multimodal", "含图像解析（+图像语义块）")


def _index_stats() -> dict:
    if INDEX_STATS_JSON.exists():
        try:
            return json.loads(INDEX_STATS_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="图像解析消融对比实验")
    ap.add_argument("--rebuild", action="store_true", help="强制重建两个索引")
    ap.add_argument("--ragas", action="store_true", help="追加 RAGAS 指标（较慢）")
    ap.add_argument("--qids", default="", help="只对比指定问题 id（如 5,6）")
    args = ap.parse_args()

    from build_multimodal_index import ensure_index
    from run_evaluation import evaluate_questions

    qids = [int(x) for x in args.qids.split(",") if x.strip()] or None

    t0 = time.perf_counter()
    print("=" * 74)
    print("消融实验：准备两个索引")
    print("=" * 74)
    ensure_index(GROUP_A[0], rebuild=args.rebuild)
    ensure_index(GROUP_B[0], rebuild=args.rebuild)

    print("\n" + "=" * 74)
    print(f"实验组 A：{GROUP_A[2]}")
    print("=" * 74)
    ra = evaluate_questions(GROUP_A[1], qids=qids, ragas=args.ragas)
    print("\n" + "=" * 74)
    print(f"实验组 B：{GROUP_B[2]}")
    print("=" * 74)
    rb = evaluate_questions(GROUP_B[1], qids=qids, ragas=args.ragas)

    # ---- 逐题对比 ----
    da = {d["id"]: d for d in ra["details"]}
    db = {d["id"]: d for d in rb["details"]}
    rows = []
    for qid in sorted(set(da) | set(db)):
        a, b = da.get(qid), db.get(qid)
        rows.append({
            "id": qid,
            "question": (a or b)["question"],
            "不含图像_正确": a["是否答对"] if a else None,
            "含图像_正确": b["是否答对"] if b else None,
            "变化": _delta(a, b),
            "不含图像_漏答要点": a["漏答要点"] if a else [],
            "含图像_漏答要点": b["漏答要点"] if b else [],
            "不含图像_答案": a["answer"] if a else "",
            "含图像_答案": b["answer"] if b else "",
            "不含图像_检索页码": a["检索页码"] if a else [],
            "含图像_检索页码": b["检索页码"] if b else [],
            "含图像_命中图像块": b["命中图像块"] if b else [],
        })

    sa, sb = ra["summary"], rb["summary"]
    stats = _index_stats()
    n_img_chunks = stats.get("with_images", {}).get("chunk_types", {}).get("image", 0)
    summary = {
        "n_questions": sa["n_questions"],
        "A_不含图像": {
            "collection": GROUP_A[1], "accuracy": sa["accuracy"],
            "correct": sa["correct"], "n_judged": sa["n_judged"],
            "retrieval_latency_avg": sa["retrieval_latency_avg"],
            "total_latency_avg": sa["total_latency_avg"],
            "n_chunks": stats.get("text_only", {}).get("n_chunks"),
        },
        "B_含图像": {
            "collection": GROUP_B[1], "accuracy": sb["accuracy"],
            "correct": sb["correct"], "n_judged": sb["n_judged"],
            "retrieval_latency_avg": sb["retrieval_latency_avg"],
            "total_latency_avg": sb["total_latency_avg"],
            "n_chunks": stats.get("with_images", {}).get("n_chunks"),
            "n_image_chunks": n_img_chunks,
        },
        "accuracy_delta": (round((sb["accuracy"] or 0) - (sa["accuracy"] or 0), 4)
                           if sa["accuracy"] is not None and sb["accuracy"] is not None
                           else None),
        "retrieval_latency_delta": round(
            sb["retrieval_latency_avg"] - sa["retrieval_latency_avg"], 3),
        "image_questions": {
            "id5": {"不含图像": da.get(5, {}).get("是否答对"),
                    "含图像": db.get(5, {}).get("是否答对")},
            "id6": {"不含图像": da.get(6, {}).get("是否答对"),
                    "含图像": db.get(6, {}).get("是否答对")},
        },
        "对比耗时(s)": round(time.perf_counter() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    payload = {"workorder": "人工智能NLP-RAG-图像内容解析及检索优化",
               "summary": summary, "questions": rows,
               "group_A": ra["summary"], "group_B": rb["summary"]}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    _write_md(payload)

    # ---- 终端结论 ----
    print("\n" + "=" * 74)
    print("对比结论")
    print("=" * 74)
    print(f"  不含图像解析：准确率 {(sa['accuracy'] or 0) * 100:.1f}% "
          f"({sa['correct']}/{sa['n_judged']})，"
          f"chunks={summary['A_不含图像']['n_chunks']}")
    print(f"  含  图像解析：准确率 {(sb['accuracy'] or 0) * 100:.1f}% "
          f"({sb['correct']}/{sb['n_judged']})，"
          f"chunks={summary['B_含图像']['n_chunks']}"
          f"（图像块 {n_img_chunks}）")
    print(f"  准确率提升：{(summary['accuracy_delta'] or 0) * 100:+.1f} 个百分点")
    print(f"  图像问题 id5：{summary['image_questions']['id5']}")
    print(f"  图像问题 id6：{summary['image_questions']['id6']}")
    print(f"  检索耗时变化：{summary['retrieval_latency_delta']:+.3f}s")
    print(f"\n  报告：{ABLATION_JSON}\n        {ABLATION_MD}")


def _delta(a: dict | None, b: dict | None) -> str:
    if not a or not b or a["是否答对"] is None or b["是否答对"] is None:
        return "—"
    if a["是否答对"] == b["是否答对"]:
        return "持平"
    return "提升" if b["是否答对"] else "下降"


def _write_md(payload: dict) -> None:
    s = payload["summary"]
    a, b = s["A_不含图像"], s["B_含图像"]
    md = [
        "# 消融实验报告：不含图像解析 vs 含图像解析",
        "",
        "工单编号：人工智能NLP-RAG-图像内容解析及检索优化",
        "",
        f"- 对比时间：{s['generated_at']}（耗时 {s['对比耗时(s)']}s）",
        f"- A 组（不含图像解析）：`{a['collection']}`，{a['n_chunks']} chunks",
        f"- B 组（含图像解析）：`{b['collection']}`，{b['n_chunks']} chunks"
        f"（其中图像语义块 {b.get('n_image_chunks', 0)} 个）",
        "- 其余配置完全一致：structure 分块 + hybrid(rrf) 检索 + cascade 重排 + "
        "同一生成模型",
        "",
        "## 一、总体对比",
        "",
        "| 指标 | 不含图像解析 | 含图像解析 | 变化 |",
        "|------|--------------|------------|------|",
        f"| 准确率（要点全命中） | {(a['accuracy'] or 0) * 100:.1f}% "
        f"({a['correct']}/{a['n_judged']}) | {(b['accuracy'] or 0) * 100:.1f}% "
        f"({b['correct']}/{b['n_judged']}) | "
        f"**{(s['accuracy_delta'] or 0) * 100:+.1f} pp** |",
        f"| 检索耗时（均值） | {a['retrieval_latency_avg']}s | "
        f"{b['retrieval_latency_avg']}s | {s['retrieval_latency_delta']:+.3f}s |",
        f"| 端到端耗时（均值） | {a['total_latency_avg']}s | "
        f"{b['total_latency_avg']}s | — |",
        f"| 索引块数 | {a['n_chunks']} | {b['n_chunks']} | "
        f"+{(b['n_chunks'] or 0) - (a['n_chunks'] or 0)} |",
        "",
        "## 二、图像问题（核心证据）",
        "",
        f"- **id 5**（组织结构图销售部/大客户销售部）："
        f"不含图像 = {s['image_questions']['id5']['不含图像']}，"
        f"含图像 = {s['image_questions']['id5']['含图像']}",
        f"- **id 6**（2008 年 IC 市场增长率）："
        f"不含图像 = {s['image_questions']['id6']['不含图像']}，"
        f"含图像 = {s['image_questions']['id6']['含图像']}",
        "",
        "原因：这两个问题的答案分别位于《招股说明书2》第 39 页组织结构图"
        "（矢量绘制）与第 72 页 IC 市场结构图（内嵌位图）中，"
        "文本层只能取到零散节点名，取不到层级关系与增长率的正负号；"
        "图像语义块入库后，检索可以命中图像块，生成模型据此作答。",
        "",
        "## 三、逐题对比",
        "",
        "| id | 问题 | 不含图像 | 含图像 | 变化 | 漏答要点（含图像） |",
        "|----|------|----------|--------|------|--------------------|",
    ]
    for r in payload["questions"]:
        mark = lambda v: {True: "✓", False: "×", None: "—"}[v]
        md.append(
            f"| {r['id']} | {r['question'][:40].replace('|', '／')} | "
            f"{mark(r['不含图像_正确'])} | {mark(r['含图像_正确'])} | {r['变化']} | "
            f"{'、'.join(r['含图像_漏答要点'])[:30] or '—'} |")

    md += ["", "## 四、两个图像问题的答案对照", ""]
    for r in payload["questions"]:
        if r["id"] not in (5, 6):
            continue
        md += [f"### id {r['id']}：{r['question']}", "",
               "**不含图像解析（A 组）**", "", "```text",
               (r["不含图像_答案"] or "").strip()[:900], "```", "",
               f"检索页码：{r['不含图像_检索页码']}", "",
               "**含图像解析（B 组）**", "", "```text",
               (r["含图像_答案"] or "").strip()[:900], "```", "",
               f"检索页码：{r['含图像_检索页码']}", "",
               f"命中图像块：{[x['page'] for x in r['含图像_命中图像块']]}", "",
               "---", ""]
    ABLATION_MD.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
