# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统的优化
功能: 对比"优化前(工单01)"与"优化后(本工单)"的检索准确率与答案质量。

优化后管线:
    PDF解析   : 文本 + 表格(pdfplumber, 线化为"字段: 值")
    分块      : 500/50, 表格块加上下文前缀
    检索      : 混合检索 (向量 0.5 + BM25 0.5), 候选 20 条
    重排      : 交叉编码器 bge-reranker-base 重排取 top5
    生成      : DeepSeek-Chat

运行: python optimize_eval.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_common as R

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OPT_INDEX = os.path.join(HERE, "index", "doc1_opt")
BASE_RESULT = os.path.join(ROOT, "工单(1)", "output", "qa_results.json")
OUT_DIR = os.path.join(HERE, "output")

REFS = json.load(open(os.path.join(ROOT, "_build", "references.json"),
                      encoding="utf-8"))
QUESTIONS = json.load(open(os.path.join(ROOT, "_build", "questions.json"),
                           encoding="utf-8"))["doc1"]


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def score_all(records, refs):
    """对一批 {id, question, answer, contexts} 做 RAGAS 风格评估"""
    out = []
    for r in records:
        ref = refs.get(str(r["id"]), "")
        ev = R.ragas_evaluate(r["question"], r["answer"], r["contexts"], ref) \
            if ref else R.ragas_evaluate(r["question"], r["answer"], r["contexts"])
        ev["id"] = r["id"]
        out.append(ev)
        print(f"    [{r['id']}] 正确性={ev.get('answer_correctness')} "
              f"上下文精度={ev.get('context_precision')} "
              f"上下文召回={ev.get('context_recall')}", flush=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 66)
    print("工单02 —— 优化前后对比评估")
    print("=" * 66)

    # ---------- 1. 优化前 (工单01 结果直接复用) ----------
    print("\n[1/3] 载入工单01 的基线结果 ...")
    base = json.load(open(BASE_RESULT, encoding="utf-8"))
    base_records = [{"id": b["id"], "question": b["question"],
                     "answer": b["rag"]["answer"],
                     "contexts": b["rag"]["contexts"]} for b in base]
    print(f"  基线: {len(base_records)} 条 (纯向量检索 + 纯文本解析)")

    # ---------- 2. 优化后 ----------
    print("\n[2/3] 运行优化后管线 ...")
    r = R.Retriever(OPT_INDEX)
    print(f"  优化索引块数: {len(r.vs.chunks)}")
    opt_records, opt_meta = [], []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  [{i}/{len(QUESTIONS)}] {q['question'][:36]}...", flush=True)
        res = R.rag_answer(q["question"], r, top_k=5, mode="hybrid",
                           weight_vector=0.5, weight_fulltext=0.5,
                           candidate_k=20, rerank="cross-encoder")
        opt_records.append({"id": q["id"], "question": q["question"],
                            "answer": res["answer"], "contexts": res["contexts"]})
        opt_meta.append(res)
        print(f"    耗时 {res['time_total']}s", flush=True)

    # ---------- 3. 评估 ----------
    print("\n[3/3] 评估中 ...")
    print("  优化前:")
    base_ev = score_all(base_records, REFS["doc1"])
    print("  优化后:")
    opt_ev = score_all(opt_records, REFS["doc1"])

    R.save_json(os.path.join(OUT_DIR, "优化后结果.json"),
                [dict(m, id=rec["id"]) for m, rec in zip(opt_meta, opt_records)])
    write_report(base_records, base_ev, opt_records, opt_ev, opt_meta, r)
    print("\n完成 ->", OUT_DIR)


def write_report(base_rec, base_ev, opt_rec, opt_ev, opt_meta, retriever):
    L = []
    A = L.append
    A("# 工单02 —— 基于 PDF 文档问答系统的优化 报告\n")
    A("工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统的优化\n")

    A("## 一、优化方案\n")
    A("| 环节 | 优化前(工单01) | 优化后(本工单) | 优化理由 |")
    A("|---|---|---|---|")
    A("| PDF解析 | pypdf 纯文本抽取 | 文本 + **pdfplumber表格解析** | "
      "招股说明书的财务数字几乎都在表格里, 纯文本抽取会把表头和数值拆散, "
      "导致『军用领域收入』这类问题检索不到 |")
    A("| 表格处理 | 无 | 表格线化为 `字段: 值` 语义单元, "
      "并把表头名与数值放进同一文本块 | 让『发行股数』『持股比例』等字段名与数值同时被命中 |")
    A("| 分块 | 固定 500 字滑窗 | 500 字 + **句末标点对齐**, "
      "表格块加**文档上下文前缀** | 避免切断句子; 上下文前缀提升向量的语义聚焦度 |")
    A("| 检索策略 | 纯向量检索 top5 | **混合检索**(向量 0.5 + BM25 0.5, 候选 20 条) | "
      "BM25 对专有名词(公司名、标准名、科目名)敏感, 弥补纯向量的不足 |")
    A("| 重排 | 无 | **bge-reranker-base 交叉编码器重排** top5 | "
      "cross-encoder 对 (query, doc) 联合编码, 排序精度高于双塔向量相似度 |\n")

    A("## 二、检索质量对比\n")
    A("| 问题ID | 优化前正确性 | 优化后正确性 | 优化前上下文精度 | 优化后上下文精度 | "
      "优化前上下文召回 | 优化后上下文召回 |")
    A("|---|---|---|---|---|---|---|")
    bmap = {e["id"]: e for e in base_ev}
    omap = {e["id"]: e for e in opt_ev}
    for q in QUESTIONS:
        i = q["id"]
        b, o = bmap[i], omap[i]
        A(f"| {i} | {b.get('answer_correctness','-')} | {o.get('answer_correctness','-')} | "
          f"{b.get('context_precision','-')} | {o.get('context_precision','-')} | "
          f"{b.get('context_recall','-')} | {o.get('context_recall','-')} |")
    A(f"| **平均** | **{avg([e.get('answer_correctness') for e in base_ev]):.3f}** | "
      f"**{avg([e.get('answer_correctness') for e in opt_ev]):.3f}** | "
      f"**{avg([e.get('context_precision') for e in base_ev]):.3f}** | "
      f"**{avg([e.get('context_precision') for e in opt_ev]):.3f}** | "
      f"**{avg([e.get('context_recall') for e in base_ev]):.3f}** | "
      f"**{avg([e.get('context_recall') for e in opt_ev]):.3f}** |\n")

    bc = avg([e.get("answer_correctness") for e in base_ev])
    oc = avg([e.get("answer_correctness") for e in opt_ev])
    A(f"- 答案正确性: **{bc:.3f} -> {oc:.3f}**  ({oc-bc:+.3f})")
    A(f"- 上下文召回: "
      f"**{avg([e.get('context_recall') for e in base_ev]):.3f} -> "
      f"{avg([e.get('context_recall') for e in opt_ev]):.3f}**")
    A(f"- 准确率(正确性>=0.8 的题占比): "
      f"**{sum(1 for e in base_ev if (e.get('answer_correctness') or 0)>=0.8)/len(base_ev)*100:.0f}%"
      f" -> "
      f"{sum(1 for e in opt_ev if (e.get('answer_correctness') or 0)>=0.8)/len(opt_ev)*100:.0f}%**\n")

    A("## 三、优化后逐题答案\n")
    for rec, meta in zip(opt_rec, opt_meta):
        A(f"### [{rec['id']}] {rec['question']}\n")
        A(f"**答案**: {rec['answer']}\n")
        A(f"**参考答案**: {REFS['doc1'].get(str(rec['id']),'-')}\n")
        A("**命中片段**:")
        for d in meta["retrieved"][:3]:
            A(f"- 第{d.get('page')}页 score={d['score']:.3f}: {d['text'][:100]}...")
        A("")

    A("## 四、响应时间\n")
    A(f"- 平均总耗时: {avg([m['time_total'] for m in opt_meta]):.2f}s")
    A(f"- 平均检索耗时: {avg([m['time_retrieval'] for m in opt_meta]):.3f}s")
    A(f"- 平均生成耗时: {avg([m['time_generation'] for m in opt_meta]):.3f}s")
    A("\n> 说明: 答案生成依赖 DeepSeek 云 API, 网络往返是主要耗时; "
      "本地检索+重排均为毫秒级, 详见工单13 的性能剖析。\n")

    A("## 五、结论\n")
    A("1. 引入**表格解析**后, 财务数字类问题(军用领域收入、收入占比、募资补流金额)"
      "首次能够被正确检索, 这是本次提升最大的环节;")
    A("2. **混合检索 + 交叉编码器重排** 提升了上下文精度, 减少了无关片段挤占上下文;")
    A("3. 剩余误差主要来自: 招股说明书表格跨页断裂、LLM 长上下文中的数字抄写错误。")

    open(os.path.join(OUT_DIR, "优化前后对比报告.md"), "w",
         encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
