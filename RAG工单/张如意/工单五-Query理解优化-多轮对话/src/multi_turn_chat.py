# -*- coding: utf-8 -*-
"""
工单05 多轮对话主程序
工单编号：人工智能NLP-RAG-Query理解优化任务

按 config.MULTI_TURN_SCRIPT 的 5 轮脚本逐轮执行，每轮完整打印：
    原始问题 → 识别意图 → 指代消解后的检索式 → 检索片段 Top3（含文档与页码）
    → 生成答案 → 本轮耗时（并标注是否满足 3 秒约束）

5 轮脚本考察的四种 Query 理解能力：
    第 1 轮  自包含问题，建立对话主体（兴图新科）
    第 2 轮  指代消解：「他」→ 兴图新科（跨文档/跨轮定位主体）
    第 3 轮  指代消解 + 话题继承：「这个公司」延续主体，问点换成「法定代表人」
    第 4 轮  省略句补全 + 主体切换：「那 X 呢？」切换到力源信息，谓语沿用上一轮
    第 5 轮  主体延续 + 跨文档检索：在《招股说明书2》的组织结构图里查销售处

输出：
    results/multi_turn_conversation.json   每轮完整链路（含 timings 明细）
    results/multi_turn_conversation.md     人读对话记录（含逐轮剖析）

运行：
    python multi_turn_chat.py                 # 默认 wo05 预设（cascade 重排，精度优先）
    python multi_turn_chat.py --fast          # 用 tfidf 重排，严格压 3 秒响应时间
    python multi_turn_chat.py --reranker llm  # 指定重排器
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import config                                       # noqa: E402

from wo05_common import (                                         # noqa: E402
    COLLECTION, RESULT_DIR, build_pipeline, index_ready, latency_summary,
    load_ground_truth, merged_spec, print_llm_usage, render_turn_md,
    run_script, write_json, write_md,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单05 多轮对话主程序（5 轮脚本）")
    ap.add_argument("--collection", default=COLLECTION, help="向量库集合名")
    ap.add_argument("--reranker", default=None,
                    help="重排器：none/tfidf/llm/adaptive/cascade（默认取 wo05 预设）")
    ap.add_argument("--fast", action="store_true",
                    help="等价于 --reranker tfidf：省一次 LLM 调用，压 3 秒响应时间")
    ap.add_argument("--top-k", type=int, default=None, help="送入生成的片段数")
    args = ap.parse_args()

    reranker = "tfidf" if args.fast else args.reranker

    if not index_ready(args.collection):
        print("[提示] 未检测到索引（data/index/bm25.pkl）。")
        print("       请先运行：python build_index.py  （需先准备两份招股说明书 PDF）")
        return 2

    pipeline = build_pipeline(use_query_understanding=True,
                              reranker=reranker, collection=args.collection,
                              top_k=args.top_k)

    questions = list(config.MULTI_TURN_SCRIPT)
    print("=" * 78)
    print("工单05 多轮对话演示（5 轮脚本）")
    print("=" * 78)
    print(f"分块策略：{pipeline.cfg.chunk_strategy}   检索：{pipeline.cfg.strategy}"
          f"/{pipeline.cfg.fusion}   重排：{pipeline.cfg.reranker}   "
          f"Query 理解：{'开' if pipeline.cfg.use_query_understanding else '关'}")
    if pipeline.cfg.reranker == "cascade":
        print("说明：cascade 重排会额外调用一次 LLM 精排（精度优先）；"
              "若需严格满足 3 秒约束，请加 --fast。")
    print("-" * 78)

    t0 = time.perf_counter()
    records = run_script(pipeline, questions, top_k_show=3, verbose=True)
    elapsed = time.perf_counter() - t0

    # ---- 汇总 ----
    lat = latency_summary(records)
    ground_truth = load_ground_truth()
    for rec in records:
        spec = merged_spec(rec["turn"], ground_truth)
        rec["考察能力"] = spec["exam"]
        rec["期望文档"] = spec["expected_doc"]

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"总耗时：{elapsed:.2f}s")
    print(f"平均每轮：{lat['平均耗时(s)']}s，最大：{lat['最大耗时(s)']}s，"
          f"3 秒达标率：{lat['3秒达标率'] * 100:.1f}%"
          f"（{lat['满足3秒约束轮次']}/{lat['轮数']} 轮）")
    errors = [r for r in records if r.get("error")]
    print(f"异常轮次：{len(errors)}（容错机制：单轮失败不中断整个对话）")
    usage = print_llm_usage()

    # ---- 落盘 ----
    report = {
        "meta": {
            "工单编号": "人工智能NLP-RAG-Query理解优化任务",
            "脚本": "config.MULTI_TURN_SCRIPT（5 轮）",
            "pipeline_config": pipeline.cfg.to_dict(),
            "总耗时(s)": round(elapsed, 3),
            "耗时汇总": lat,
            "LLM用量": usage,
        },
        "turns": records,
    }
    write_json(RESULT_DIR / "multi_turn_conversation.json", report)
    write_md(RESULT_DIR / "multi_turn_conversation.md", render_md(report, ground_truth))
    print(f"结果已写入：{RESULT_DIR / 'multi_turn_conversation.json'}")
    print(f"          {RESULT_DIR / 'multi_turn_conversation.md'}")
    return 0


def render_md(report: dict, ground_truth: dict | None = None) -> str:
    """渲染多轮对话 Markdown 记录（逐轮：问题 → 意图 → 检索式 → 片段 → 答案 → 耗时）。"""
    meta = report["meta"]
    lat = meta["耗时汇总"]
    lines = [
        "# 工单05 多轮对话记录（5 轮脚本）",
        "",
        "> 工单编号：人工智能NLP-RAG-Query理解优化任务  ",
        f"> 流水线配置：`chunk={meta['pipeline_config']['chunk_strategy']}`、"
        f"`strategy={meta['pipeline_config']['strategy']}/{meta['pipeline_config']['fusion']}`、"
        f"`reranker={meta['pipeline_config']['reranker']}`、"
        f"`use_query_understanding={meta['pipeline_config']['use_query_understanding']}`  ",
        f"> 总耗时 {meta['总耗时(s)']}s；平均每轮 {lat['平均耗时(s)']}s；"
        f"3 秒达标率 {lat['3秒达标率'] * 100:.1f}%（{lat['满足3秒约束轮次']}/{lat['轮数']}）",
        "",
        "## 一、逐轮记录",
        "",
    ]
    for rec in report["turns"]:
        spec = merged_spec(rec["turn"], ground_truth)
        lines.append(render_turn_md(rec, spec))

        # 耗时明细（证明「Query 理解不是延迟瓶颈」）
        timings = rec.get("timings") or {}
        if timings:
            parts = "；".join(f"{k} {v}s" for k, v in timings.items())
            lines.append(f"- **阶段耗时明细**：{parts}")
            lines.append("")

    lines += [
        "## 二、逐轮能力对照",
        "",
        "| 轮次 | 原始问题 | 考察能力 | 改写后的检索式 | 期望文档 | 本轮耗时(s) |",
        "|------|----------|----------|----------------|----------|-------------|",
    ]
    for rec in report["turns"]:
        q = rec["question"].replace("|", "\\|")
        rw = (rec.get("rewritten") or rec["question"]).replace("|", "\\|")
        lines.append(f"| {rec['turn']} | {q} | {rec.get('考察能力', '')} | "
                     f"`{rw}` | {rec.get('期望文档', '')} | {rec.get('latency', 0):.3f} |")
    lines += [
        "",
        "## 三、结论",
        "",
        "- 第 2/3 轮通过「他 / 这个公司」定位到武汉兴图新科（招股说明书1）；",
        "- 第 4 轮先做实体替换（力源信息），再从上一轮问题继承谓语「法定代表人是谁」，"
        "实现省略句补全；",
        "- 第 5 轮保持在力源信息主体上，跨文档检索其组织结构图（图像块）；",
        "- 每轮耗时构成见「阶段耗时明细」，Query 理解走规则通道时不产生 LLM 调用。",
        "",
        "> 详细剖析见 `docs/多轮对话设计.md`；消融实验（关闭 Query 理解的对比）"
        "见 `results/multiturn_ablation.md`。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
