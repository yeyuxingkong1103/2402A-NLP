"""读 eval/eval_set.json，逐条调 rag.ask（或读缓存），用 RAGAS 打 4 指标，写 eval/report.md。

用法：python eval/run_ragas.py [--fresh]
依赖：真实 DeepSeek + Milvus + 本地 bge-m3/reranker（Redis 可选）。
--fresh 强制重新跑 rag.ask 并覆盖缓存；否则命中 eval/raw_results.json 时直接复用（干净 A/B）。
评估 LLM=DeepSeek、embedding=bge-m3，faithfulness 用自定义 prompt（忽略引用格式词）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

import rag  # noqa: E402
from ingest import load_embedder  # noqa: E402

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall  # noqa: E402

from faithfulness_prompts import build_faithfulness  # noqa: E402

logger = logging.getLogger(__name__)

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"
RAW_RESULTS = Path(__file__).resolve().parent / "raw_results.json"
REPORT = Path(__file__).resolve().parent / "report.md"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
TRUNC = 30  # 明细表长文本截断长度

# 优化前（旧 faithfulness prompt）基线，用于前后对比
BASELINE = {
    "faithfulness": 0.4251,
    "answer_relevancy": 0.9161,
    "context_precision": 0.8709,
    "context_recall": 0.8889,
}


class BgeM3Embeddings(Embeddings):
    """把 ingest.load_embedder() 的 SentenceTransformer 包成 langchain Embeddings。"""

    def __init__(self):
        self._model = load_embedder()  # lru_cache 单例，与检索共用

    def embed_documents(self, texts):
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def build_llm():
    """DeepSeek 经 langchain-openai 包装，供 RAGAS 评估判定使用。"""
    return LangchainLLMWrapper(ChatOpenAI(
        model=rag.DEEPSEEK_MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY1"),
        base_url=rag.DEEPSEEK_BASE_URL,
        temperature=0,
    ))


def _trunc(text: str) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= TRUNC else text[:TRUNC] + "…"


def _run_ragas_items(items) -> list[dict]:
    """逐条调 rag.ask（独立 session 防记忆串扰），失败跳过，返回 raw 记录。"""
    raw = []
    for i, item in enumerate(items, 1):
        try:
            r = rag.ask(item["question"], session_id=f"eval-{i}")
        except Exception as e:  # noqa: BLE001
            logger.error("[%d/%d] rag.ask 失败，跳过：%s", i, len(items), e)
            continue
        raw.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "source_chunk_id": item.get("source_chunk_id", ""),
            "answer": r["answer"],
            "retrieved_contexts": [s["content"] for s in r["sources"]],
        })
        logger.info("[%d/%d] %s", i, len(items), item["question"][:30])
    return raw


def load_raw(items, fresh: bool) -> list[dict]:
    """fresh=False 且缓存与评测集题目一致时读缓存；否则重跑 rag.ask 并写缓存。"""
    if not fresh and RAW_RESULTS.exists():
        cached = json.loads(RAW_RESULTS.read_text(encoding="utf-8"))
        if [c["question"] for c in cached] == [i["question"] for i in items]:
            logger.info("命中缓存 %s（%d 条），跳过 rag.ask", RAW_RESULTS, len(cached))
            return cached
        logger.warning("缓存与评测集题目不一致，忽略缓存")
    raw = _run_ragas_items(items)
    RAW_RESULTS.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写缓存 %s（%d 条）", RAW_RESULTS, len(raw))
    return raw


def run_eval(raw) -> list[dict]:
    """RAGAS 打 4 指标（faithfulness 用自定义 prompt），返回每样本指标分 dict。"""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["ground_truth"],
        )
        for r in raw
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset,
        metrics=[build_faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=build_llm(),
        embeddings=LangchainEmbeddingsWrapper(BgeM3Embeddings()),
    )
    return result.to_pandas().to_dict("records")


def _mean_scores(details: list[dict]) -> dict:
    """逐指标求平均，跳过 nan（RAGAS 判定失败项）；全 nan 记 None。"""
    avg = {}
    for m in METRICS:
        vals = [d[m] for d in details if not math.isnan(d[m])]
        avg[m] = round(sum(vals) / len(vals), 4) if vals else None
    return avg


def render_report(avg: dict, rows: list[dict], n: int) -> str:
    head = [
        "# RAGAS 评测报告", "",
        f"- 评测集：eval/eval_set.json（成功评测 {n} 条）",
        "- 评测集数据源：hypertension_guide（混合 guide.pdf 指南 + medical_dialogue.csv 问诊语料）",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 评估 LLM：{rag.DEEPSEEK_MODEL}（DeepSeek）",
        "- 评估 embedding：BGE-m3（1024 维）",
        "- RAGAS 版本：0.3.9", "",
        "## 指标平均分", "",
        "| 指标 | 平均分 | 有效样本 |", "|---|---|---|",
    ]
    for m in METRICS:
        valid = sum(1 for d in rows if not math.isnan(d[m]))
        head.append(f"| {m} | {avg[m] if avg[m] is not None else 'nan'} | {valid}/{n} |")
    head += ["", "## 前后对比", "",
             "| 指标 | 优化前 | 优化后 | 变化 |", "|---|---|---|---|"]
    for m in METRICS:
        after = avg[m]
        delta = round(after - BASELINE[m], 4) if after is not None else None
        head.append(
            f"| {m} | {BASELINE[m]} | {after if after is not None else 'nan'} | "
            f"{delta if delta is not None else 'nan'} |"
        )
    head += ["", "## 已知限制", "",
             "- faithfulness 曾偏低：答案中的「根据资料N（第X页）」「结论：」「您好」等引用格式词不在检索上下文里，被误判为幻觉——已在自定义 prompt 中缓解。",
             "- nan：deepseek-flash 对 RAGAS 结构化输出（JSON）解析不稳定，导致部分指标评分缺失。",
             "- 30 条里 2 条真实检索失败（#14、#19，context_precision/recall=0.0）。", "",
             "## 明细", "",
             "| # | source_chunk_id | 问题 | 标准答案 | RAG 答案 | faithfulness | answer_relevancy | context_precision | context_recall |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        head.append(
            f"| {i} | {r['chunk_id']} | {_trunc(r['question'])} | {_trunc(r['ground_truth'])} "
            f"| {_trunc(r['answer'])} | {r['faithfulness']} | {r['answer_relevancy']} "
            f"| {r['context_precision']} | {r['context_recall']} |"
        )
    return "\n".join(head) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略缓存，强制重新跑 rag.ask")
    args = ap.parse_args()

    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    raw = load_raw(items, args.fresh)
    if not raw:
        raise SystemExit("无成功样本，请先检查 Milvus / DeepSeek 是否可用")

    records = run_eval(raw)
    details = [{
        "chunk_id": r["source_chunk_id"],
        "question": r["question"],
        "ground_truth": r["ground_truth"],
        "answer": r["answer"],
    } for r in raw]
    for d, rec in zip(details, records):
        for m in METRICS:
            d[m] = round(float(rec.get(m, 0)), 4)
    avg = _mean_scores(details)

    REPORT.write_text(render_report(avg, details, len(details)), encoding="utf-8")
    logger.info("报告已写入 %s（平均分 %s）", REPORT, avg)


if __name__ == "__main__":
    main()
