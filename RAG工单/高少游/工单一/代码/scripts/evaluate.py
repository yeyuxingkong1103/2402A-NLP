# -*- coding: utf-8 -*-
"""RAG 评估脚本：对 10 道题分别用 RAG 和纯 LLM 回答，并用 ragas 评估对比
用法（项目根目录，激活 langchain2 环境，需已启动 Ollama）：
    python scripts/evaluate.py [--top-k 6] [--in-memory]
        --in-memory: 强制在内存中构建向量库（网络/临时目录无法落盘时使用）

产物（写到 output/ 目录）：
    evaluation_report.json   原始评估结果
    evaluation_report.md     RAG vs 纯LLM 对比报告
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from src import config
from src.knowledge_base import _kv_file, build_documents, get_all_documents, get_embeddings, load_kb
from src.qa_engine import QAEngine
from src.retriever import HybridRetriever

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 本地裁判模型（deepseek-r1）偏慢且对严格 JSON 偶有失败，采用较短超时 + 少量重试，
# 使不兼容的指标快速失败并跳过，避免整轮评估长时间卡死。
_RUN_CONFIG = RunConfig(timeout=120, max_retries=2, max_workers=1)


def load_questions() -> list:
    with open(config.QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def make_engine(in_memory: bool) -> QAEngine:
    """构建问答引擎：优先加载持久化向量库，否则在内存中构建。"""
    if in_memory or not _kv_file(config.DB_DIR).exists():
        logger.info("在内存中构建向量库 ...")
        docs = build_documents()
        store = FAISS.from_documents(docs, get_embeddings())
        retr = HybridRetriever(store=store, corpus=docs)
    else:
        store = load_kb()
        retr = HybridRetriever(store=store, corpus=get_all_documents())
    return QAEngine(retriever=retr)


def build_judge():
    from langchain_ollama import ChatOllama

    judge = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=config.LLM_MAX_TOKENS,
    )
    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(get_embeddings())


def run_generation(engine: QAEngine, questions: list) -> list:
    """为每题生成 RAG 答案与纯 LLM 答案。"""
    rows = []
    for item in questions:
        q = item["question"]
        rag = engine.answer_rag(q)
        only = engine.answer_llm_only(q)
        rows.append(
            {
                "id": item["id"],
                "question": q,
                "reference": item.get("reference", ""),
                "rag_answer": rag.answer,
                "rag_elapsed": round(rag.elapsed, 2),
                "rag_contexts": [s["text"] for s in rag.sources],
                "rag_pages": [s["page"] for s in rag.sources],
                "llm_only_answer": only.answer,
                "llm_only_elapsed": round(only.elapsed, 2),
            }
        )
        print(
            f"  [id={item['id']}] RAG {rag.elapsed:.1f}s / LLM {only.elapsed:.1f}s"
        )
        print(f"      RAG: {rag.answer[:70]}")
    return rows


def evaluate_rag(rows: list, judge_llm, judge_emb, run_config=None) -> dict:
    """对 RAG 答案用 ragas 计算 faithful / relevancy / context_precision / context_recall / correctness。"""
    samples = []
    for r in rows:
        if not r["rag_contexts"]:
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                reference=r["reference"] or None,
                response=r["rag_answer"],
                retrieved_contexts=r["rag_contexts"],
            )
        )
    if not samples:
        return {}
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config or _RUN_CONFIG,
    )
    return result


def evaluate_llm_only(rows: list, judge_llm, judge_emb, run_config=None) -> dict:
    """对纯 LLM 答案计算 answer_correctness（无检索上下文，故不适用上下文类指标）。"""
    samples = []
    for r in rows:
        if not r["reference"]:
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                reference=r["reference"],
                response=r["llm_only_answer"],
                retrieved_contexts=[],  # 纯 LLM 无上下文
            )
        )
    if not samples:
        return {}
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset, metrics=[answer_correctness], llm=judge_llm, embeddings=judge_emb,
                      run_config=run_config or _RUN_CONFIG)
    return result


def result_to_dict(result) -> dict:
    """把 ragas EvaluationResult 转成可序列化 dict。"""
    try:
        for m in result.metric_scores:
            return {
                "metric": m.metric.name,
                "score": float(m.score),
                "per_sample": [float(x) if x is not None else None for x in m.scores],
            }
    except Exception:
        pass
    pandas = result.to_pandas()
    return {
        "metric": list(pandas.keys()),
        "score": [float(v) if v is not None else None for v in pandas.iloc[0].tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 评估")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--in-memory", action="store_true", help="内存构建向量库")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="跳过 ragas 评分，仅生成回答与耗时（RAG vs 纯LLM 对比）")
    args = parser.parse_args()

    config.TOP_K = args.top_k
    questions = load_questions()
    print(f"加载 {len(questions)} 道评估问题")

    engine = make_engine(args.in_memory)
    rows = run_generation(engine, questions)

    # 先生成结果立即落盘，保证即使后续 ragas 失败也保留关键数据
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rag_metrics": {},
        "llm_only_metrics": {},
        "cases": rows,
    }
    json_path = config.OUTPUT_DIR / "evaluation_report.json"

    if args.skip_ragas:
        print("\n跳过 ragas 评分（--skip-ragas）。")
    else:
        judge_llm, judge_emb = build_judge()
        print("\n开始 ragas 评估（RAG 链路）...")
        try:
            rag_res = evaluate_rag(rows, judge_llm, judge_emb)
            report["rag_metrics"] = result_to_dict(rag_res) if rag_res else {}
        except Exception as exc:
            print(f"RAG ragas 评估失败（已跳过）：{exc}")
            report["rag_metrics"] = {"error": str(exc)}

        print("开始 ragas 评估（纯 LLM 链路）...")
        try:
            llm_res = evaluate_llm_only(rows, judge_llm, judge_emb)
            report["llm_only_metrics"] = result_to_dict(llm_res) if llm_res else {}
        except Exception as exc:
            print(f"纯LLM ragas 评估失败（已跳过）：{exc}")
            report["llm_only_metrics"] = {"error": str(exc)}

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSON 报告已写入: {json_path}")
    print("评估完成.")


if __name__ == "__main__":
    main()