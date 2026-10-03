# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：run_ragas —— RAGAS 标准指标体系评估（可选依赖）
# 说明：读取 qa_bilingual_result.json（RAG 链路产出的 问题/答案/检索上下文/参考答案），
#       用 RAGAS 计算 faithfulness、answer_relevancy、context_precision、context_recall、
#       answer_correctness。评判 LLM 与嵌入模型走本机 Ollama 的 OpenAI 兼容端点
#       （qwen2:7b + bge-m3），与系统本身同源，无需外部 API / 不产生费用。
#
# 环境准备（独立 venv，避免污染主项目依赖）：
#   python -m venv .venv_ragas
#   uv pip install --python .venv_ragas\Scripts\python.exe "ragas==0.2.15" "langchain-community<0.4" "langchain<0.4" "langchain-openai<0.4"
#
# 用法：
#   .venv_ragas\Scripts\python.exe evaluation\run_ragas.py --lang zh
import argparse
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
                           LLMContextPrecisionWithoutReference, AnswerCorrectness)
from ragas.run_config import RunConfig  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

METRIC_COLS = ["faithfulness", "answer_relevancy",
               "llm_context_precision_without_reference",
               "context_recall", "answer_correctness"]


def make_judge(base, gen_model, embed_model):
    llm = ChatOpenAI(base_url=base, api_key="ollama", model=gen_model,
                     temperature=0.0, timeout=600, max_retries=1)
    emb = OpenAIEmbeddings(base_url=base, api_key="ollama", model=embed_model,
                           check_embedding_ctx_length=False, timeout=600)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def run(lang, base, gen_model, embed_model, limit=0):
    data = json.load(open(os.path.join(HERE, "qa_bilingual_result.json"), encoding="utf-8"))["records"]
    rows = [r for r in data if r["lang"] == lang and r["contexts"]]
    if limit:
        rows = rows[:limit]
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"],
                                retrieved_contexts=r["contexts"], reference=r["gold_answer"])
               for r in rows]
    ds = EvaluationDataset(samples=samples)
    llm, emb = make_judge(base, gen_model, embed_model)
    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithoutReference(),
               LLMContextRecall(), AnswerCorrectness()]
    res = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb,
                   run_config=RunConfig(max_workers=1, timeout=600), raise_exceptions=False)
    df = res.to_pandas()
    out = {"lang": lang, "n": len(rows), "gen_model": gen_model, "embed_model": embed_model,
           "scores": {}, "per_question": []}
    for col in METRIC_COLS:
        if col in df.columns:
            vals = [v for v in df[col].tolist() if v == v]  # 去 NaN
            out["scores"][col] = round(sum(vals) / len(vals), 4) if vals else None
            out["scores"][col + "_n"] = len(vals)
    for i, r in enumerate(rows):
        rec = {"id": r["id"], "question": r["question"]}
        for col in METRIC_COLS:
            if col in df.columns:
                v = df[col].tolist()[i]
                rec[col] = None if v != v else round(float(v), 4)
        out["per_question"].append(rec)
    print("\n==== RAGAS (%s, n=%d) ====" % (lang, len(rows)))
    for col in METRIC_COLS:
        if col in out["scores"]:
            print("  %-45s %s  (有效 %d/%d)" % (col, out["scores"][col],
                                                out["scores"].get(col + "_n", 0), len(rows)))
    path = os.path.join(HERE, "ragas_result_%s.json" % lang)
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--base", default="http://localhost:11434/v1")
    ap.add_argument("--gen", default="qwen2:7b")
    ap.add_argument("--embed", default="bge-m3")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.lang, a.base, a.gen, a.embed, a.limit)
