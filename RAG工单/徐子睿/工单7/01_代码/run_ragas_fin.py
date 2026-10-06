# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务 | 人工智能NLP-RAG-功能测试及评估任务
# 模块：run_ragas_fin —— RAGAS 标准指标体系评估（金融年报 QA，工单 07）
# 说明：读取 fin_qa_result.json（问题/答案/检索上下文/参考答案），用 RAGAS 计算
#       faithfulness / answer_relevancy / context_precision / context_recall / answer_correctness；
#       评判 LLM 与嵌入走本机 Ollama 的 OpenAI 兼容端点（qwen2:7b + bge-m3）。
# 用法：<venv_ragas>\python.exe evaluation\run_ragas_fin.py
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")

from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import (Faithfulness, AnswerRelevancy, LLMContextRecall,  # noqa: E402
                           LLMContextPrecisionWithReference, AnswerCorrectness)
from ragas.run_config import RunConfig  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

METRIC_COLS = ["faithfulness", "answer_relevancy",
               "llm_context_precision_with_reference", "context_recall", "answer_correctness"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="fin", choices=["fin", "graph"],
                    help="fin=常规 RAG（07 baseline）；graph=Graph RAG（08/09）")
    ap.add_argument("--src-file", default="", help="直接指定输入 json（如 graph_qa_result_opt.json）")
    a = ap.parse_args()
    base = os.environ.get("OLLAMA_OPENAI_BASE", "http://localhost:11434/v1")
    if a.src_file:
        src = a.src_file
        out_name = "ragas_result_" + os.path.splitext(a.src_file)[0] + ".json"
    else:
        src = "fin_qa_result.json" if a.src == "fin" else "graph_qa_result.json"
        out_name = "ragas_result_fin.json" if a.src == "fin" else "ragas_result_graph.json"
    data = json.load(open(os.path.join(HERE, src), encoding="utf-8"))["detail"]
    rows = [r for r in data if r.get("contexts")]
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"],
                                retrieved_contexts=r["contexts"], reference=r["gold_answer"])
               for r in rows]
    ds = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(ChatOpenAI(base_url=base, api_key="ollama", model="qwen2:7b",
                                         temperature=0.0, timeout=600, max_retries=1))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(base_url=base, api_key="ollama", model="bge-m3",
                                                      check_embedding_ctx_length=False, timeout=600))
    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference(),
               LLMContextRecall(), AnswerCorrectness()]
    res = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb,
                   run_config=RunConfig(max_workers=1, timeout=600), raise_exceptions=False)
    df = res.to_pandas()
    out = {"n": len(rows), "scores": {}, "per_question": []}
    for col in METRIC_COLS:
        if col in df.columns:
            vals = [v for v in df[col].tolist() if v == v]
            out["scores"][col] = round(sum(vals) / len(vals), 4) if vals else None
            out["scores"][col + "_n"] = len(vals)
    for i, r in enumerate(rows):
        rec = {"id": r["id"], "question": r["question"]}
        for col in METRIC_COLS:
            if col in df.columns:
                v = df[col].tolist()[i]
                rec[col] = None if v != v else round(float(v), 4)
        out["per_question"].append(rec)
    print("\n==== RAGAS (%s, n=%d) ====" % (src, len(rows)))
    for col in METRIC_COLS:
        if col in out["scores"]:
            print("  %-42s %s  (有效 %d/%d)" % (col, out["scores"][col],
                                                out["scores"].get(col + "_n", 0), len(rows)))
    json.dump(out, open(os.path.join(HERE, out_name), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved -> evaluation/%s" % out_name)


if __name__ == "__main__":
    main()
