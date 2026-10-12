# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""跑 5 轮验收序列，产出逐轮报告（改写/答案/检索上下文/引用/耗时）。

用法：
    python run_acceptance.py            # 标准口径：先 warmup 再跑（稳态单轮时延）
                                         # → data/eval/acceptance_result.json
    python run_acceptance.py --cold     # 不预热：首轮含建模/建索引成本（冷启动实测）
                                         # → data/eval/acceptance_result_cold.json
                                         # （不覆盖标准口径的落盘，两种口径各自留证）

本机约束（实测，非猜测）：Windows 主线程栈约 1MB，加载 bge-m3 / reranker 会
触发导入期栈溢出（进程 exit 139，Python 层捕获不到）。故真正的活儿全部跑在
`bootstrap.run_with_large_stack` 的大栈线程里 —— 少了这层包裹，本脚本会段错误。

计时口径：`timings["total_ms"]` **已含**多轮改写延迟（Task5 fix wave F4 修正），
故 `≤3s` 判定直接读它，**不**再叠加 `rewrite["latency_ms"]`（叠加会虚增 ~200ms/代词轮）。
"""
# bootstrap 必须最先导入（离线模式 + 大栈），不要与下面的导入合并或调换顺序
from src import bootstrap

import argparse
import json
import os
import sys

from src import config, context as ctx_mod, dialogue, rag


def contexts_of(result) -> tuple[str, list[dict]]:
    """还原本轮**真实喂给 LLM 的检索上下文**（逐字文本 + 片段定位）。

    `result.contexts` 就是 `rag.build_qa_context()` 产出的父块列表，
    `context.build_context()` 负责把它拼成提示词里的【文档片段】正文——
    两者同源，故这里重建的 `context_text` 与 LLM 实际看到的一致
    （含【片段N】/页码/标题路径/表格与组织结构图标记）。

    Task11 的 RAGAS 多轮指标（faithfulness / context recall）需要逐轮检索上下文，
    这里落盘以免评测时重跑一遍检索。
    """
    blocks = result.contexts or []
    text = ctx_mod.build_context(blocks) if blocks else ""
    rows: list[dict] = []
    for i, b in enumerate(blocks, start=1):
        rows.append({
            "index": i,
            "source": b.get("source", ""),
            "page_idx": int(b.get("page_idx", 0)),
            "page_display": int(b.get("page_idx", 0)) + 1,   # 0 基 → 1 基
            "heading_path": list(b.get("heading_path") or []),
            "text": (b.get("parent_text") or "").strip(),
            "is_table": bool(b.get("is_table")),
            "is_orgchart": bool(b.get("is_orgchart")),
            "table_caption": b.get("table_caption", ""),
            "org_path": b.get("org_path", ""),
        })
    return text, rows


def build_row(item: dict, result, turn) -> dict:
    """一轮 → 一行报告（纯函数，便于测试与复用）。"""
    hit = [f for f in item["expected_facts"] if f in result.answer]
    passed = len(hit) == len(item["expected_facts"])
    total_ms = result.timings.get("total_ms") or 0
    context_text, contexts = contexts_of(result)
    return {
        "turn": item["turn"], "question": item["question"],
        "resolved": turn.resolved_question, "rewrite_mode": turn.rewrite_mode,
        "coref_span": turn.coref_span, "answer": result.answer,
        "expected_facts": item["expected_facts"], "hit_facts": hit,
        "passed": passed, "total_ms": total_ms,
        "rewrite_ms": (result.rewrite or {}).get("latency_ms"),
        "timings": result.timings,
        "citations": result.citations, "mode": result.mode,
        # --- 检索上下文：逐字喂给 LLM 的文本 + 结构化片段（Task11 RAGAS 消费）---
        "context_text": context_text,
        "contexts": contexts,
    }


def run(spec: dict, warmup: bool = True) -> tuple[list[dict], bool, dict]:
    """跑完 5 轮，返回 (rows, all_ok, warmup_info)。"""
    warm_info: dict = {"skipped": True}
    if warmup:
        warm_info = rag.warmup()
        print(f"[warmup] {warm_info}", flush=True)

    dialogue.reset_all()
    st = dialogue.get_state("acceptance")
    rows: list[dict] = []
    ok = True

    for item in spec["turns"]:
        r, t = rag.ask_turn(item["question"], st)
        row = build_row(item, r, t)
        rt = row["total_ms"]
        ok &= row["passed"] and rt <= 3000
        rows.append(row)
        # 事实命中与超时是**两个**判据：这里显式标出超时，避免"每行都 OK 却 exit=1"
        over = f"  ⚠超时 {rt:.0f}ms>3000ms" if rt > 3000 else ""
        print(f"[{'OK ' if row['passed'] else 'FAIL'}] Q{item['turn']} {rt:.0f}ms "
              f"({t.rewrite_mode}) {t.resolved_question[:52]}{over}", flush=True)
        print(f"        {r.answer[:160]}", flush=True)

    return rows, bool(ok), warm_info


# ---------------------------------------------------------------------------
# 逐轮报告（docs/06-验收序列逐轮报告.md）
# ---------------------------------------------------------------------------
def _cite_line(c: dict) -> str:
    """一条引用 → 「哪份 PDF / 哪一页 / 表格或组织结构图节点路径」。"""
    src = c.get("source", "?")
    page = c.get("page_display") or (int(c.get("page_idx", 0)) + 1)
    kind, extra = "段落", ""
    if c.get("is_table"):
        kind = "表格"
        cap = c.get("table_caption") or c.get("table_id") or ""
        extra = f"｜表名：{cap}" if cap else ""
    elif c.get("is_orgchart") or c.get("org_path"):
        kind = "组织结构图"
        extra = f"｜节点路径：{c.get('org_path')}" if c.get("org_path") else ""
    head = " / ".join(c.get("heading_path") or [])
    return (f"- 📄 **{kind}**｜{src} 第 {page} 页"
            + (f"｜{head}" if head else "") + extra)


def to_markdown(payload: dict) -> str:
    """逐轮报告。每轮必须含：原问题 / 独立问题 / 改写方式 / 命中事实 /
    答案全文 / 引用溯源 / 总耗时与改写耗时（工单要求 4 与 6 的证据面）。"""
    rows = payload["rows"]
    mode = "预热后（warm）" if payload.get("mode") != "cold" else "冷启动（cold）"
    tmax = max((r["total_ms"] for r in rows), default=0)
    lines = [
        "# 验收序列逐轮报告（工单 05）",
        "",
        f"- 工单编号：{payload.get('workorder', config.WORKORDER_ID)}",
        f"- 序列：`{payload.get('dialogue_file', 'acceptance_dialogue.json')}`"
        f"（5 轮连续对话，带对话状态）",
        f"- 口径：{mode}；逐轮最大耗时 **{tmax:.0f}ms**；"
        f"未通过 **{payload.get('n_failed', 0)}** 轮；"
        f"超 3 秒 **{payload.get('n_over_3s', 0)}** 轮",
        "",
        "> 说明：本报告由 `python run_acceptance.py` 的落盘产物生成"
        "（`python run_acceptance.py --report-only` 可离线重生成），",
        "> 不是人工复述。每轮的「独立问题」是系统**真实改写结果**，"
        "「引用」是系统**真实返回的溯源**。",
        "",
        "## 逐轮结果一览",
        "",
        "| 轮次 | 改写方式 | 关键事实 | 总耗时 | 改写耗时 | 判定 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        facts = f"{len(r['hit_facts'])}/{len(r['expected_facts'])}"
        over = " ⚠>3s" if r["total_ms"] > 3000 else ""
        lines.append(f"| Q{r['turn']} | `{r['rewrite_mode']}` | {facts} | "
                     f"{r['total_ms']:.0f}ms{over} | "
                     f"{(r['rewrite_ms'] or 0):.0f}ms | "
                     f"{'✅' if r['passed'] else '❌'} |")
    lines += ["", "---", ""]

    for r in rows:
        span = r.get("coref_span") or {}
        lines += [
            f"## Q{r['turn']}：{r['question']}",
            "",
            f"- **改写方式**：`{r['rewrite_mode']}`"
            + (f"（指代实体：{span.get('entity')}）" if span.get("entity") else ""),
            f"- **独立问题**：{r['resolved']}",
            f"- **关键事实**：{'✅ 全部命中' if r['passed'] else '❌ 有缺失'}"
            f"　{len(r['hit_facts'])}/{len(r['expected_facts'])}",
            f"　　- 命中：{'、'.join(r['hit_facts']) or '（无）'}",
        ]
        miss = [f for f in r["expected_facts"] if f not in r["hit_facts"]]
        if miss:
            lines.append(f"　　- 缺失：{'、'.join(miss)}")
        t = r.get("timings") or {}
        stage = "　".join(
            f"{k.split('_')[0]} {t[k]:.0f}ms" for k in
            ("retrieve_ms", "rerank_ms", "gen_ms") if t.get(k) is not None)
        lines += [
            f"- **耗时**：总 {r['total_ms']:.0f}ms"
            f"（含改写 {(r['rewrite_ms'] or 0):.0f}ms）｜{stage}",
            "",
            "**答案全文：**",
            "",
            f"> {r['answer']}",
            "",
            "**引用溯源：**",
            "",
        ]
        cites = r.get("citations") or []
        lines += [_cite_line(c) for c in cites] or ["- （本轮无引用）"]
        lines += ["", "**本轮检索到的片段（喂给 LLM 的上下文）：**", ""]
        for b in (r.get("contexts") or []):
            badge = "📊 表格" if b.get("is_table") else (
                "🌳 组织结构图" if b.get("is_orgchart") else "📄 段落")
            head = " / ".join(b.get("heading_path") or [])
            lines.append(f"- [{b['index']}] {badge}｜{b.get('source')} "
                         f"第 {b.get('page_display')} 页"
                         + (f"｜{head}" if head else "")
                         + (f"｜{b.get('table_caption')}" if b.get("table_caption") else ""))
        lines += ["", "---", ""]
    return "\n".join(lines) + "\n"


def write_report(payload: dict) -> str:
    path = os.path.join(config.DOCS_DIR, "06-验收序列逐轮报告.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(payload))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工单5 5 轮验收序列")
    ap.add_argument("--cold", action="store_true",
                    help="跳过 warmup，测量冷启动（首轮含模型加载 ~25s）；"
                         "结果另存 acceptance_result_cold.json")
    ap.add_argument("--report-only", action="store_true",
                    help="不跑序列，用已有 acceptance_result.json 重新生成 docs/06 报告")
    args = ap.parse_args(argv)

    if args.report_only:
        src = os.path.join(config.EVAL_DIR, "acceptance_result.json")
        with open(src, encoding="utf-8") as fh:
            print(f"报告已重新生成：{write_report(json.load(fh))}（数据源 {src}）")
        return 0

    spec_path = os.path.join(config.ROOT, "data", "acceptance_dialogue.json")
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)

    rows, ok, warm_info = run(spec, warmup=not args.cold)

    os.makedirs(config.EVAL_DIR, exist_ok=True)
    out_name = "acceptance_result_cold.json" if args.cold else "acceptance_result.json"
    out = os.path.join(config.EVAL_DIR, out_name)
    payload = {
        "workorder": spec.get("workorder", config.WORKORDER_ID),
        "dialogue_file": os.path.basename(spec_path),
        "mode": "cold" if args.cold else "warm",
        "warmup": warm_info,
        "n_turns": len(rows),
        "n_failed": sum(1 for r in rows if not r["passed"]),
        "n_over_3s": sum(1 for r in rows if r["total_ms"] > 3000),
        "all_passed": bool(ok),
        "rows": rows,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    tmax = max((r["total_ms"] for r in rows), default=0)
    over = [(r["turn"], round(r["total_ms"])) for r in rows if r["total_ms"] > 3000]
    miss = [(r["turn"], f) for r in rows for f in r["expected_facts"]
            if f not in r["hit_facts"]]
    print(f"\n逐轮最大耗时 {tmax:.0f}ms（{'冷启动' if args.cold else '预热后'}）；"
          f"{'全部通过' if ok else '存在未通过项'}", flush=True)
    if over:
        print(f"  超 3 秒的轮次（轮, ms）：{over}", flush=True)
    if miss:
        print(f"  未命中的关键事实（轮, 事实）：{miss}", flush=True)
    print(f"  结果落盘 {out}", flush=True)

    # 标准口径（warm）才生成 docs/06 交付报告；冷启动那份是诊断留证，不进交付物
    if not args.cold:
        try:
            print(f"  逐轮报告 {write_report(payload)}", flush=True)
        except Exception as exc:                            # noqa: BLE001
            print(f"  ⚠ 报告生成失败（不影响验收结论）：{exc}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    # 必须在 main 之前包好大栈线程：main 里会加载 bge-m3 / reranker
    sys.exit(bootstrap.run_with_large_stack(main))
