# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于Graph+RAG实现金融问答任务 / 人工智能NLP-RAG-Graph+RAG优化任务
# 关联工单：人工智能NLP-RAG-基于Graph+RAG实现金融问答任务 | 人工智能NLP-RAG-Graph+RAG优化任务
# 模块：run_ragas_graph —— Graph RAG 的 RAGAS 评估（与工单 07 基线同口径对比）
# 用法：<venv_ragas>\python.exe evaluation\run_ragas_graph.py
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
    base = os.environ.get("OLLAMA_OPENAI_BASE", "http://localhost:11434/v1")
    # 可通过环境变量指定输入/输出，便于工单 09 做「优化前/优化后」两次评估
    in_name = os.environ.get("RAGAS_IN", "graph_qa_result.json")
    out_name = os.environ.get("RAGAS_OUT", "ragas_result_graph.json")
    data = json.load(open(os.path.join(HERE, in_name), encoding="utf-8"))["detail"]
    print("[ragas] in=%s out=%s" % (in_name, out_name))
    rows = [r for r in data if r.get("contexts")]
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"],
                                retrieved_contexts=r["contexts"], reference=r["gold_answer"])
               for r in rows]
    ds = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(ChatOpenAI(base_url=base, api_key="***", model="qwen2:7b",
                                         temperature=0.0, timeout=600, max_retries=1))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(base_url=base, api_key="***", model="bge-m3",
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
    print("\n==== RAGAS (Graph RAG, n=%d) ====" % len(rows))
    for col in METRIC_COLS:
        if col in out["scores"]:
            print("  %-42s %s  (有效 %d/%d)" % (col, out["scores"][col],
                                                out["scores"].get(col + "_n", 0), len(rows)))
    json.dump(out, open(os.path.join(HERE, out_name), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved -> evaluation/%s" % out_name)


if __name__ == "__main__":
    main()
