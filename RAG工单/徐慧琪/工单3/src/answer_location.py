# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：答案定位报告（工单3 交付物之一）

工单要求：「答案定位报告：对 14 道题，说明每道题的答案在哪个 PDF、哪一页、
哪个表格/段落」。

做法（不靠人工翻页，而是**用系统自己的检索链路**产出证据）：
  1. 对每道题跑完整检索（查询理解 → 文档路由 → 混合召回 → 重排 → 上下文组装）；
  2. 在送进 LLM 的上下文里逐条核对 ground truth 的 key_facts 是否命中，
     得到「事实命中率」（比单纯看答案文本更严格，能定位到"哪一条没找回来"）；
  3. 输出每个命中片段所在 PDF、页码、标题路径；表格片段额外给出 table_id、
     表名、命中的行号，做到「哪个 PDF / 哪一页 / 哪个表 / 哪一行」四级定位。

输出：data/eval/answer_location.json + docs/06-答案定位报告.md
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import json
import os
import re
import time
from typing import Sequence

from src import config


def _norm(text: str) -> str:
    """去空白与全角空格，便于事实比对（"1,670 万股" 也能匹配 "1,670万股"）。"""
    return re.sub(r"[\s　]", "", text or "")


def match_facts(context_text: str, facts: Sequence[str]) -> tuple[list[str], list[str]]:
    """返回 (命中事实, 未命中事实)。"""
    ctx = _norm(context_text)
    hit, miss = [], []
    for f in facts:
        (hit if _norm(f) in ctx else miss).append(f)
    return hit, miss


def locate_question(question: str, facts: Sequence[str],
                    use_rerank: bool = True) -> dict:
    """单题定位：跑检索并把命中事实映射到具体片段与表格位置。"""
    from src import rag

    t0 = time.time()
    hits, analysis, timings = rag.retrieve(question, use_rerank=use_rerank)
    context_text, cmap, blocks, t_extra = rag.build_qa_context(hits)
    timings.update(t_extra)

    hit, miss = match_facts(context_text, facts)

    # 逐片段核对，标出"哪条事实出现在哪个片段"
    evidence = []
    for i, b in enumerate(blocks, start=1):
        btext = _norm(b.get("parent_text", ""))
        here = [f for f in facts if _norm(f) in btext]
        if not here:
            continue
        entry = {
            "fragment": i,
            "source": b.get("source", ""),
            "page_display": int(b.get("page_idx", 0)) + 1,
            "page_idx": int(b.get("page_idx", 0)),
            "heading_path": list(b.get("heading_path") or []),
            "is_table": bool(b.get("is_table")),
            "facts": here,
        }
        if b.get("is_table"):
            entry.update({
                "table_id": b.get("table_id", ""),
                "table_caption": b.get("table_caption", ""),
                "table_header": b.get("table_header", ""),
                "row_indices": sorted({int(h.get("row_index", -1))
                                       for h in b.get("child_hits", [])
                                       if int(h.get("row_index", -1)) >= 0}),
            })
        evidence.append(entry)

    return {
        "question": question,
        "facts": list(facts),
        "facts_hit": hit,
        "facts_missed": miss,
        "fact_recall": round(len(hit) / max(len(facts), 1), 4),
        "route": (analysis.get("doc_route") or {}).get("reason", ""),
        "evidence": evidence,
        "n_fragments": len(blocks),
        "elapsed_s": round(time.time() - t0, 2),
        "timings": timings,
    }


def build_report(gt_file: str | None = None, use_rerank: bool = True) -> dict:
    """对全部题目生成定位结果。"""
    gt_file = gt_file or config.GT_FILE
    with open(gt_file, encoding="utf-8") as fh:
        gt = json.load(fh)

    results = []
    for item in gt:
        r = locate_question(item["question"], item.get("key_facts") or [],
                            use_rerank=use_rerank)
        r.update({"id": item["id"], "source": item.get("source", ""),
                  "reference": item.get("reference", ""),
                  "answer_kind": item.get("answer_kind", ""),
                  "page_hint": item.get("page_hint", 0)})
        results.append(r)
        mark = "✓" if not r["facts_missed"] else "✗"
        print(f"  {mark} [{item['id']:>3}] 事实命中 "
              f"{len(r['facts_hit'])}/{len(r['facts'])} | "
              f"{r['source']} | {r['elapsed_s']}s", flush=True)

    n = len(results)
    # 第 1 题的耗时含模型冷启动（首次加载 bge-m3/reranker 要十几秒），
    # 会严重拉高均值；故同时给出"排除首题"的稳态均值，两个数都如实写进报告。
    steady = [r["elapsed_s"] for r in results[1:]] or [0.0]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_questions": n,
        "n_full_hit": sum(1 for r in results if not r["facts_missed"]),
        "fact_recall_avg": round(sum(r["fact_recall"] for r in results) / max(n, 1), 4),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / max(n, 1), 2),
        "avg_elapsed_steady_s": round(sum(steady) / len(steady), 2),
        "warmup_first_s": results[0]["elapsed_s"] if results else 0.0,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------
def _fmt_evidence(ev: dict) -> str:
    page = ev["page_display"]
    head = " / ".join(ev.get("heading_path") or []) or "（无标题）"
    src = ev.get("source", "")
    if ev.get("is_table"):
        rows = ev.get("row_indices") or []
        row_txt = ("第 " + "、".join(str(r + 1) for r in rows) + " 行") if rows else "表头行"
        return (f"📊 **表格**｜{src} 第 {page} 页｜表名：{ev.get('table_caption') or '（无标题）'}"
                f"｜table_id：`{ev.get('table_id')}`｜命中{row_txt}")
    return f"📄 **段落**｜{src} 第 {page} 页｜{head}"


def to_markdown(report: dict) -> str:
    """定位结果 → Markdown 报告（写入 docs/06-答案定位报告.md）。"""
    lines = [
        "# 答案定位报告（工单 03）",
        "",
        f"- 工单编号：{config.WORKORDER_ID}",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目数：{report['n_questions']}；全部事实命中的题目：{report['n_full_hit']}"
        f"；平均事实命中率：{report['fact_recall_avg']:.1%}",
        f"- 平均检索耗时：{report.get('avg_elapsed_steady_s', report['avg_elapsed_s'])}s"
        f"（不含 LLM 生成；已排除第 1 题的模型冷启动 {report.get('warmup_first_s', 0)}s）",
        "",
        "> 说明：本报告不是人工翻页结果，而是**用系统自身的检索链路**产出的证据——",
        "> 每题跑完整检索后，核对 ground truth 的 `key_facts` 是否出现在送进 LLM 的上下文里，",
        "> 并给出该事实所在的 PDF / 页码 / 表格或段落 / 表格行号。",
        "",
        "---",
        "",
    ]
    for r in report["results"]:
        status = "✅ 全部命中" if not r["facts_missed"] else f"⚠️ 缺 {len(r['facts_missed'])} 条"
        kind = {"table": "表格题", "text": "文本题"}.get(r.get("answer_kind"), "—")
        lines += [
            f"## [{r['id']}] {r['question']}",
            "",
            f"- 目标文档：**{r['source']}**（题型：{kind}）　检索结果：{status}"
            f"　事实命中 {len(r['facts_hit'])}/{len(r['facts'])}"
            f"　耗时 {r['elapsed_s']}s",
            f"- 路由：{r['route']}",
            f"- 参考答案：{r['reference']}",
        ]
        if r["facts_missed"]:
            lines.append(f"- ⚠️ 未命中事实：{r['facts_missed']}")
        lines.append("")
        if r["evidence"]:
            lines.append("**答案定位：**")
            lines.append("")
            for ev in r["evidence"]:
                lines.append(f"- {_fmt_evidence(ev)}　→ 命中事实：{'、'.join(ev['facts'])}")
        else:
            lines.append("**答案定位：**（未在检索上下文中找到任何 key fact —— 需要重点排查）")
        lines += ["", "---", ""]
    return "\n".join(lines)


def main() -> int:
    from src import bootstrap as _bs
    report = build_report()
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    json_path = os.path.join(config.EVAL_DIR, "answer_location.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    md_path = os.path.join(config.DOCS_DIR, "06-答案定位报告.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(report))
    print(f"\n已写入 {json_path}\n已写入 {md_path}")
    print(f"全部命中 {report['n_full_hit']}/{report['n_questions']}，"
          f"平均事实命中率 {report['fact_recall_avg']:.1%}")
    return 0


if __name__ == "__main__":
    from src import bootstrap
    bootstrap.run_with_large_stack(main)
