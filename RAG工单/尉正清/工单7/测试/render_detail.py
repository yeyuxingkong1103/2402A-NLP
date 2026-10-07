# 工单编号：人工智能NLP-RAG-功能测试及评估
"""把评估结果渲染成逐题明细 markdown

拆成独立文件的原因是行数：eval_ccf.py 同时放测量逻辑和排版逻辑会超过
300 行的单文件上限。这里只做排版，不碰任何检索或大模型调用。

明细一律由脚本从结果 JSON 生成而不是手抄 —— 手抄的表格和图里的数字迟早对不上，
而这份工单的核心交付物就是「10 个问题的检索结果和评估结果」。
"""
def _cell(text, width=70):
    """把一段正文塞进 markdown 表格单元格。

    表格块的正文本身就是 markdown（含大量 `|` 和换行），直接放进单元格会把
    整张表的列数冲乱 —— 实测预览里出现 `|项目|2021年/末|` 后表格就散了。
    所以 `|` 转义、换行压平。
    """
    return (text or "").replace("|", "\\|").replace("\n", " ")[:width]


def _gold_str(gold):
    return "；".join(f"{d} 第 {', '.join(str(p) for p in ps)} 页"
                     for d, ps in gold.items())


def render_markdown(out):
    """把结果 JSON 渲染成逐题明细，供测试报告引用。

    明细一律由脚本生成而不是手抄 —— 手抄的表格和图里的数字迟早对不上，
    而这份工单的核心交付物就是「10 个问题的检索结果和评估结果」。
    """
    s, rows = out["summary"], out["rows"]
    L = [
        "# 测试结果明细", "",
        "**工单编号**：人工智能NLP-RAG-功能测试及评估  ",
        f"**生成时间**：{out['generated_at']}  ",
        f"**知识库**：`{out['kb']}`，{out['chunks']} 块 / {len(out['docs_in_kb'])} 份年报，"
        f"Top-K = {out['top_k']}  ",
        f"**本轮**：{'含大模型评估' if out['with_llm'] else '仅检索指标（--no-llm）'}", "",
        "> 本文件由 `eval_ccf.py` 自动生成，请勿手工编辑。",
        "> 指标定义见 `设计/设计说明.md` 第四节，问题分析见 `测试报告.md`。", "",
        "---", "", "## 汇总", "",
        f"| 指标 | 数值 |", "|---|---|",
        f"| 文档命中率 | {s['doc_hit']:.1%} |",
        f"| 页召回率 | {s['page_recall']:.1%} |",
        f"| 页精确率 | {s['page_precision']:.1%} |",
        f"| 事实点召回 | {s['fact_recall']:.1%} |",
        f"| MRR | {s['mrr']:.3f} |",
        f"| 平均检索耗时 | {s['retrieval_seconds']:.2f}s |",
    ]
    if out["with_llm"]:
        L += [
            f"| 答案有据率（grounded） | RAG {s['grounded']:.1%} / 纯LLM {s['plain_grounded']:.1%} |",
            f"| 答案切题率（relevant） | RAG {s['relevant']:.1%} / 纯LLM {s['plain_relevant']:.1%} |",
            f"| 答案事实点召回 | RAG {s['answer_fact_recall']:.1%} / "
            f"纯LLM {s['plain_answer_fact_recall']:.1%} |",
            f"| 平均生成耗时 | RAG {s['answer_seconds']:.1f}s / "
            f"纯LLM {s['plain_seconds']:.1f}s |",
        ]
    L += ["", f"- 未完全命中目标文档的题：{s['full_doc_miss'] or '无'}",
          f"- 完全没召回标准页的题：{s['zero_page_hit'] or '无'}"]
    if s.get("industry_fail") is not None:
        L.append(f"- 跨文档题行业覆盖不全：{s['industry_fail'] or '无'}")
    if s.get("failed"):
        L.append(f"- **存在调用失败的题**：{s['failed']}（原因见各题 `*_error`）")
    L += ["", "---", ""]

    for r in rows:
        m = r.get("retrieval", {})
        L += [f"## Q{r['id']}｜{r['qtype']}｜{r['question']}", "",
              f"- **题目来源**：{r['origin']}",
              f"- **目标文档**：{'、'.join(r['docs'])}",
              f"- **标准答案页**：{_gold_str(r['gold_pages'])}",
              f"- **Query 改写**：{r.get('rewrite') or '（无）'}",
              f"- **子问题**：{len(r.get('sub_questions') or [])} 个"
              + ("｜".join(r["sub_questions"]) if r.get("sub_questions") else ""),
              ""]
        if r.get("error"):
            L += [f"> ⚠️ {r['error']}", ""]
            continue
        if m:
            L += [
                f"**检索指标**：文档命中 {m['doc_hit']:.0%}｜页召回 "
                f"{(m['page_recall'] or 0):.0%}｜页精确 {m['page_precision']:.0%}｜"
                f"事实点 {m['fact_recall']:.0%}｜MRR {m['mrr']:.2f}｜"
                f"行业覆盖 {'/'.join(m['industry_cover']) or '—'}｜"
                f"检索耗时 {r['retrieval_seconds']:.2f}s"
                + (f"｜**未命中文档 {m['doc_missing']}**" if m["doc_missing"] else ""),
                "",
                "**检索结果**（✅ = 该块落在标准答案页上）", "",
                "| # | 文档 | 页 | 类型 | 分数 | 正文摘要 |", "|---|---|---|---|---|---|",
            ]
            for i, h in enumerate(r.get("hits", []), 1):
                mark = "✅" if (h["doc"], h["page"]) in {
                    (d, p) for d, ps in r["gold_pages"].items() for p in ps} else ""
                L.append(f"| {i} | {h['doc']} | {h['page']} {mark} | {h['type']} | "
                         f"{h['score']:.4f} | {_cell(h['preview'])} |")
            L += ["",
                  "> 分数不是单调递减的：`_recall_all` 对多路查询做的是"
                  "**按名次轮流合并**，不是按分数全局排序。"
                  "这样重写和子问题只能补充候选，不会把原问题的正确结果挤掉。", ""]
        if out["with_llm"]:
            j = r.get("judge") or {}
            L += [f"**RAG 回答**（生成 {r.get('rag_seconds', 0):.1f}s｜裁判 有据="
                  f"{j.get('grounded', '?')} 切题={j.get('relevant', '?')}）", "",
                  f"> {r.get('rag_answer') or r.get('judge_error') or '（缺失）'}", "",
                  f"　裁判理由：{j.get('reason', '—')}", ""]
            pj = r.get("plain_judge") or {}
            L += [f"**纯 LLM 对照**（不检索，生成 {r.get('plain_seconds', 0):.1f}s｜裁判 有据="
                  f"{pj.get('grounded', '?')} 切题={pj.get('relevant', '?')}）", "",
                  f"> {r.get('plain_answer') or r.get('plain_error') or '（缺失）'}", ""]
            if pj.get("reason"):
                L.append(f"　裁判理由：{pj['reason']}")
            L.append("")
        L += ["---", ""]
    return "\n".join(L)


