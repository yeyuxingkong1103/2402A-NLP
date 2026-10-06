# -*- coding: utf-8 -*-
"""RAGAS 独立评分：读取 evaluation_report.json 中的既有答案，用本地裁判模型评分。
避免重复搭建知识库/重新生成，便于单独尝试与时间盒控制。
用法（项目根目录，激活 langchain2 环境）：
    python scripts/ragas_score.py
产物：output/ragas_metrics.json
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.run_config import RunConfig

from src import config

# 时间盒：超过该秒数则放弃（本地小模型作为裁判偏慢）
MAX_SECONDS = 540
_RUN_CONFIG = RunConfig(timeout=90, max_retries=2, max_workers=1)


def load_cases() -> list:
    p = Path(config.OUTPUT_DIR) / "evaluation_report.json"
    return json.loads(p.read_text(encoding="utf-8"))["cases"]


def main() -> None:
    from evaluate import result_to_dict  # 复用序列化（同目录具名导入）

    cases = load_cases()
    if not cases:
        print("no cases"); return
    _build_judge = None
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_ollama import ChatOllama
    from src.knowledge_base import get_embeddings

    judge = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL,
                       temperature=0.1, num_predict=config.LLM_MAX_TOKENS)
    judge_llm = LangchainLLMWrapper(judge)
    judge_emb = LangchainEmbeddingsWrapper(get_embeddings())

    out = {"rag_metrics": {}, "llm_only_metrics": {}, "timed_out": False}
    t0 = time.time()

    # RAG 链路：5 项指标
    rag_samples = [SingleTurnSample(user_input=c["question"], reference=c.get("reference") or None,
                    response=c["rag_answer"], retrieved_contexts=c["rag_contexts"]) for c in cases if c["rag_contexts"]]
    if rag_samples and time.time() - t0 < MAX_SECONDS:
        from ragas.metrics import (answer_correctness, answer_relevancy,
                                   context_precision, context_recall, faithfulness)
        try:
            res = evaluate(EvaluationDataset(samples=rag_samples),
                           metrics=[faithfulness, answer_relevancy, context_precision,
                                    context_recall, answer_correctness],
                           llm=judge_llm, embeddings=judge_emb, run_config=_RUN_CONFIG)
            out["rag_metrics"] = result_to_dict(res) if res else {}
            print("RAG 链路完成")
        except Exception as exc:
            print(f"RAG 链路失败：{exc}")

    # 纯 LLM 链路：answer_correctness
    llm_samples = [SingleTurnSample(user_input=c["question"], reference=c.get("reference") or "",
                   response=c["llm_only_answer"], retrieved_contexts=[]) for c in cases if c.get("reference")]
    if llm_samples and time.time() - t0 < MAX_SECONDS:
        from ragas.metrics import answer_correctness
        try:
            res = evaluate(EvaluationDataset(samples=llm_samples), metrics=[answer_correctness],
                           llm=judge_llm, embeddings=judge_emb, run_config=_RUN_CONFIG)
            out["llm_only_metrics"] = result_to_dict(res) if res else {}
            print("纯 LLM 链路完成")
        except Exception as exc:
            print(f"纯 LLM 链路失败：{exc}")

    out["elapsed"] = time.time() - t0
    if time.time() - t0 >= MAX_SECONDS:
        out["timed_out"] = True
    dst = Path(config.OUTPUT_DIR) / "ragas_metrics.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {dst}")


if __name__ == "__main__":
    main()