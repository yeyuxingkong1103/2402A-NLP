# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
功能:
    1. 对 10 个测试问题执行 RAG 问答 (检索 + LLM 生成)
    2. 对同样 10 个问题执行"仅 LLM"基线问答 (不检索)
    3. 对比两者的答案差异
    4. 使用 RAGAS 风格指标进行评估 (忠实度/答案相关性/上下文精度/上下文召回/答案正确性)

测试问题与参考答案统一从 ../_build/ 读取 (参考答案已人工逐条核对 PDF 原文):
    ../_build/questions.json    -> doc1 段: 测试问题
    ../_build/references.json   -> doc1 段: 参考答案; doc1_keywords: 关键词

运行: python qa_eval.py
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_common as R

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(os.path.dirname(HERE), "_build")
INDEX_DIR = os.path.join(HERE, "index", "doc1_text")
OUT_DIR = os.path.join(HERE, "output")

# 测试问题与参考答案 (从 _build 读取, 不再硬编码; JSON 键为字符串, 转成 int 便于按问题 id 查找)
QUESTIONS = R.load_json(os.path.join(BUILD_DIR, "questions.json"))["doc1"]
_REFS = R.load_json(os.path.join(BUILD_DIR, "references.json"))
REFERENCE = {int(k): v for k, v in _REFS["doc1"].items() if str(k).isdigit()}
KEYWORDS = {int(k): v for k, v in _REFS.get("doc1_keywords", {}).items()
            if str(k).isdigit()}


def run_rag(r, q):
    """RAG 检索增强问答"""
    return R.rag_answer(q["question"], r, top_k=5, mode="vector")


def run_baseline(q):
    """基线: 不检索, 直接让 LLM 回答"""
    ans = R.llm(q["question"], max_tokens=512, temperature=0.2)
    return {"question": q["question"], "answer": ans, "contexts": [],
            "retrieved": [], "time_total": 0.0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("加载索引:", INDEX_DIR)
    r = R.Retriever(INDEX_DIR)
    print(f"索引块数: {len(r.vs.chunks)}")

    qs = QUESTIONS
    results = []
    for i, q in enumerate(qs, 1):
        print(f"[{i}/{len(qs)}] {q['question'][:40]}...")
        rag_res = run_rag(r, q)
        base_res = run_baseline(q)
        ref = REFERENCE.get(q["id"], "")
        kws = KEYWORDS.get(q["id"], [])

        # 并行评估 RAG 结果
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_rag = ex.submit(R.ragas_evaluate, q["question"], rag_res["answer"],
                              rag_res["contexts"], ref)
            f_base = ex.submit(R.ragas_evaluate, q["question"],
                               base_res["answer"], [ref] if ref else [], ref)
            ev_rag, ev_base = f_rag.result(), f_base.result()

        # 客观指标: 参考答案关键词命中率 (不依赖 LLM)
        ev_rag["keyword_hit"] = round(R.keyword_hit(rag_res["answer"], kws), 3)
        ev_base["keyword_hit"] = round(R.keyword_hit(base_res["answer"], kws), 3)

        rag_res["eval"] = ev_rag
        base_res["eval"] = ev_base
        rag_res["id"] = q["id"]
        base_res["id"] = q["id"]
        rag_res["reference"] = ref
        base_res["reference"] = ref
        results.append({"id": q["id"], "question": q["question"],
                        "rag": rag_res, "baseline": base_res})
        print(f"    RAG 耗时 {rag_res['time_total']}s | "
              f"忠实度 {ev_rag.get('faithfulness')} | 正确性 "
              f"{ev_rag.get('answer_correctness')} || 纯LLM 正确性 "
              f"{ev_base.get('answer_correctness')}")

    R.save_json(os.path.join(OUT_DIR, "qa_results.json"), results)
    write_report(results, r)
    print("完成 -> ", os.path.join(OUT_DIR, "qa_results.json"))


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def write_report(results, r):
    L = []
    A = L.append
    A("# 工单01 —— 基于 PDF 文档的问答系统 结果报告\n")
    A("工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统\n")
    A(f"- 文档: 《招股说明书1.pdf》(武汉兴图新科电子股份有限公司)")
    A(f"- 索引块数: {len(r.vs.chunks)}")
    A(f"- 检索策略: 向量检索 (bge 系列嵌入模型 "
      f"{os.path.basename(R.EMBED_MODEL_PATH)}, 余弦相似度, top_k=5)")
    A(f"- 生成模型: DeepSeek-Chat")
    A(f"- 参考答案: 取自 ../_build/references.json (已人工逐条核对 PDF 原文)\n")

    A("## 一、检索问答结果\n")
    for r_ in results:
        A(f"### [{r_['id']}] {r_['question']}\n")
        A(f"**参考答案** (人工核对): {r_['rag'].get('reference', '-')}\n")
        A(f"**RAG 回答** (耗时 {r_['rag']['time_total']}s):\n\n{r_['rag']['answer']}\n")
        A(f"**仅 LLM 回答** (无检索):\n\n{r_['baseline']['answer']}\n")
        A("**检索到的片段(前3)**:")
        for d in r_["rag"]["retrieved"][:3]:
            A(f"- 第{d['page']}页 (score={d['score']:.3f}): {d['text'][:120]}...")
        A("")

    A("## 二、RAG 评估指标\n")
    A("| 问题ID | 忠实度 | 答案相关性 | 上下文精度 | 上下文召回 | 答案正确性 | 关键词命中率 | 耗时(s) |")
    A("|---|---|---|---|---|---|---|---|")
    for r_ in results:
        e = r_["rag"]["eval"]
        A(f"| {r_['id']} | {e.get('faithfulness','-')} | "
          f"{e.get('answer_relevancy','-')} | {e.get('context_precision','-')} | "
          f"{e.get('context_recall','-')} | {e.get('answer_correctness','-')} | "
          f"{e.get('keyword_hit','-')} | "
          f"{r_['rag']['time_total']} |")
    e = [r_["rag"]["eval"] for r_ in results]
    A(f"| **平均** | **{avg([x.get('faithfulness') for x in e]):.3f}** | "
      f"**{avg([x.get('answer_relevancy') for x in e]):.3f}** | "
      f"**{avg([x.get('context_precision') for x in e]):.3f}** | "
      f"**{avg([x.get('context_recall') for x in e]):.3f}** | "
      f"**{avg([x.get('answer_correctness') for x in e]):.3f}** | "
      f"**{avg([x.get('keyword_hit') for x in e]):.3f}** | "
      f"**{avg([r_['rag']['time_total'] for r_ in results]):.2f}** |\n")

    A("## 三、RAG vs 纯 LLM 对比分析\n")
    A("| 问题ID | RAG 正确性 | 纯LLM 正确性 | RAG 忠实度 | 纯LLM 忠实度 |")
    A("|---|---|---|---|---|")
    for r_ in results:
        A(f"| {r_['id']} | {r_['rag']['eval'].get('answer_correctness','-')} | "
          f"{r_['baseline']['eval'].get('answer_correctness','-')} | "
          f"{r_['rag']['eval'].get('faithfulness','-')} | "
          f"{r_['baseline']['eval'].get('faithfulness','-')} |")
    rc = avg([r_["rag"]["eval"].get("answer_correctness") for r_ in results])
    bc = avg([r_["baseline"]["eval"].get("answer_correctness") for r_ in results])
    rk = avg([r_["rag"]["eval"].get("keyword_hit") for r_ in results])
    bk = avg([r_["baseline"]["eval"].get("keyword_hit") for r_ in results])
    A(f"\n- RAG 平均答案正确性: **{rc:.3f}**")
    A(f"- 纯 LLM 平均答案正确性: **{bc:.3f}**")
    A(f"- 提升: **{rc-bc:+.3f}**")
    A(f"- RAG 平均关键词命中率: **{rk:.3f}**  |  纯 LLM 平均关键词命中率: "
      f"**{bk:.3f}**\n")
    A("### 结论")
    A("1. 纯 LLM 无法访问《招股说明书1.pdf》私有内容, 对具体财务数字、"
      "股本结构等问题通常给出笼统或错误回答, 忠实度与正确性显著偏低;")
    A("2. RAG 通过向量检索把原文片段送入上下文, 答案有明确出处, "
      "数字类问题正确性明显提升;")
    A("3. 存在的问题: 表格内数字(如军用领域收入明细)以纯文本方式解析时"
      "容易错行, 需在工单03中引入表格解析优化。")

    A("\n## 四、响应时间\n")
    A(f"- 平均检索耗时: {avg([r_['rag']['time_retrieval'] for r_ in results]):.3f}s")
    A(f"- 平均生成耗时: {avg([r_['rag']['time_generation'] for r_ in results]):.3f}s")
    A(f"- 平均总耗时: {avg([r_['rag']['time_total'] for r_ in results]):.3f}s")
    A("\n> 说明: 检索秒级达标; 生成阶段受 LLM 网络往返影响, "
      "耗时优化见工单13。")

    open(os.path.join(OUT_DIR, "结果报告.md"), "w", encoding="utf-8").write(
        "\n".join(L))


if __name__ == "__main__":
    main()
