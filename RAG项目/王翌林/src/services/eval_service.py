"""RAGAS 评测服务：Faithfulness / Answer Relevancy / Context Precision / Context Recall。

为什么是"内置 Judge + RAGAS 双引擎"：
- 首选 ragas 官方实现——指标定义与学术界对齐，结果可对外引用；
- 但 ragas 依赖 langchain 等较重三方包，环境未必装得上（或版本冲突），
  故内置一套 LLM-as-Judge 实现保底：同一套指标定义、同一个判定模型（DeepSeek），
  这样"评测在任何环境都能跑通"，不会因装不上依赖就让质量评估缺位。
run_eval 默认 engine="auto"：先用内置判分（顺带产出 answer/contexts），
再尝试用官方 ragas 覆盖指标值，成功则标记 engine="ragas"，失败则沿用内置结果。

被谁调用：api/eval 路由、离线评测脚本。依赖：rag.retriever（拿上下文）、llm_service（判分）。
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.logging import get_logger
from src.rag import prompt as prompt_utils
from src.rag.retriever import retrieve
from src.services import llm_service

logger = get_logger("eval.ragas")

JUDGE_PROMPTS = {
    "faithfulness": (
        "你是评测专家。请判断【回答】中的每个事实性陈述是否都能被【知识片段】支持。\n"
        "只输出一个 0 到 1 之间的小数（1 表示全部有依据，0 表示完全没有依据），不要解释。\n\n"
        "【知识片段】\n{context}\n\n【回答】\n{answer}\n\n分数："
    ),
    "answer_relevancy": (
        "你是评测专家。请判断【回答】是否直接、完整地回应了【问题】。\n"
        "只输出一个 0 到 1 之间的小数（1 表示高度相关，0 表示答非所问），不要解释。\n\n"
        "【问题】\n{question}\n\n【回答】\n{answer}\n\n分数："
    ),
    "context_precision": (
        "你是评测专家。下面列出多个【知识片段】，请判断其中与【问题】真正相关的片段占比。\n"
        "只输出一个 0 到 1 之间的小数（相关片段数 / 总片段数），不要解释。\n\n"
        "【问题】\n{question}\n\n【知识片段】\n{context}\n\n分数："
    ),
    "context_recall": (
        "你是评测专家。请判断【知识片段】是否覆盖了回答【参考答案】所需的全部关键信息。\n"
        "只输出一个 0 到 1 之间的小数（1 表示完全覆盖，0 表示完全没覆盖），不要解释。\n\n"
        "【参考答案】\n{ground_truth}\n\n【知识片段】\n{context}\n\n分数："
    ),
}


def _judge(prompt: str) -> float:
    """LLM 打分，解析失败返回 0.0。"""
    text = llm_service.simple_complete(prompt, temperature=0.0, max_tokens=16)
    for token in text.replace("：", " ").replace("。", " ").split():
        try:
            value = float(token)
            return max(0.0, min(1.0, value))
        except ValueError:
            continue
    logger.warning("评测打分解析失败：%s", text[:50])
    return 0.0


def load_dataset(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """读取评测数据集（jsonl：persona_code/persona_id, question, ground_truth）。"""
    path = path or os.path.join(settings.data_dir, "eval", "ragas_dataset.jsonl")
    if not os.path.exists(path):
        logger.warning("评测数据集不存在：%s", path)
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _builtin_evaluate(sample: Dict[str, Any], persona_id: int,
                      top_k: int = 5, use_llm_answer: bool = True) -> Dict[str, Any]:
    """内置 LLM-as-Judge 评测单条样本。"""
    question = sample["question"]
    hits = retrieve(persona_id, question, rerank_top_n=top_k)
    context = prompt_utils.build_context_block(hits, max_chars=3000)

    if use_llm_answer:
        persona_prompt = (
            "你是一名专业的心理陪伴咨询师。请基于知识片段，用温暖、专业、不超过 200 字的中文回答用户问题。"
        )
        answer = llm_service.simple_complete(
            f"知识片段：\n{context}\n\n用户问题：{question}\n\n请回答：",
            system=persona_prompt, temperature=0.3, max_tokens=400,
        ) or "（生成失败）"
    else:
        answer = sample.get("answer", "")

    ground_truth = sample.get("ground_truth", "")
    scores = {
        "faithfulness": _judge(JUDGE_PROMPTS["faithfulness"].format(context=context, answer=answer)),
        "answer_relevancy": _judge(
            JUDGE_PROMPTS["answer_relevancy"].format(question=question, answer=answer)
        ),
        "context_precision": _judge(
            JUDGE_PROMPTS["context_precision"].format(question=question, context=context)
        ),
    }
    # context_recall 需要参考答案才能算，数据集没给 ground_truth 时就跳过该指标
    if ground_truth:
        scores["context_recall"] = _judge(
            JUDGE_PROMPTS["context_recall"].format(ground_truth=ground_truth, context=context)
        )

    return {
        "question": question,
        "answer": answer,
        "contexts": [h.get("text", "") for h in hits],
        "ground_truth": ground_truth,
        "scores": scores,
    }


def run_builtin_eval(persona_id: int, limit: int = 10,
                     dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """用内置 Judge 跑完整评测并聚合平均分。

    只取属于该角色的样本（persona_id/persona_code 为空表示通用样本，也纳入）。
    单条样本失败不影响整体：只记日志并放入带 error 的占位结果，保证评测能跑完。
    """
    samples = [s for s in load_dataset(dataset_path)
               if s.get("persona_id") in (None, persona_id) or s.get("persona_code") in (None, "")]
    samples = samples[:limit]
    if not samples:
        return {"persona_id": persona_id, "samples": 0, "metrics": {}, "details": []}

    details = []
    for sample in samples:
        try:
            details.append(_builtin_evaluate(sample, persona_id))
        except Exception as exc:
            logger.error("评测样本失败：%s", exc)
            details.append({"question": sample.get("question"), "error": str(exc), "scores": {}})

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metrics = {}
    for name in metric_names:
        values = [d["scores"][name] for d in details if name in d.get("scores", {})]
        if values:
            metrics[name] = round(sum(values) / len(values), 4)
    return {"persona_id": persona_id, "samples": len(details), "metrics": metrics, "details": details}


def _try_ragas(persona_id: int, samples: List[Dict[str, Any]], answers: List[Dict]) -> Optional[Dict]:
    """尝试使用 ragas 官方实现（需要 ragas + langchain-openai 可用）。"""
    try:
        from ragas import EvaluationDataset, evaluate  # noqa: F401
        from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                                   faithfulness)
    except Exception as exc:
        logger.info("ragas 官方实现不可用，使用内置 Judge：%s", exc)
        return None

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        llm = LangchainLLMWrapper(ChatOpenAI(
            model=settings.llm_model, api_key=settings.llm_api_key,
            base_url=settings.llm_base_url, temperature=0.0,
        ))
        for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
            metric.llm = llm

        dataset = EvaluationDataset.from_list([
            {
                "user_input": a["question"],
                "response": a["answer"],
                "retrieved_contexts": a["contexts"],
                "reference": a["ground_truth"],
            }
            for a in answers if a["ground_truth"]
        ])
        if not len(dataset):
            return None
        result = evaluate(dataset=dataset, metrics=[
            faithfulness, answer_relevancy, context_precision, context_recall
        ])
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        return {k: round(float(v), 4) for k, v in scores.items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.warning("ragas 官方评测失败，回退内置 Judge：%s", exc)
        return None


def run_eval(persona_id: int, limit: int = 10, dataset_path: Optional[str] = None,
             engine: str = "auto") -> Dict[str, Any]:
    """执行评测并返回 {engine, samples, metrics, report_path}。"""
    start = time.time()
    builtin = run_builtin_eval(persona_id, limit, dataset_path)
    engine_used = "builtin_llm_judge"

    # auto：只要 ragas 可用就优先采信官方指标（对齐学术界口径），否则沿用内置 Judge 的结果
    if engine in ("auto", "ragas") and builtin["details"]:
        ragas_metrics = _try_ragas(persona_id, None, builtin["details"])
        if ragas_metrics:
            builtin["metrics"] = ragas_metrics
            engine_used = "ragas"

    report_dir = os.path.join(settings.data_dir, "eval", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"ragas_report_persona{persona_id}_{int(time.time())}.json")
    report = {
        "persona_id": persona_id,
        "engine": engine_used,
        "samples": builtin["samples"],
        "metrics": builtin["metrics"],
        "elapsed_seconds": round(time.time() - start, 2),
        "details": builtin["details"],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("评测完成 persona=%s engine=%s metrics=%s 报告=%s",
                persona_id, engine_used, builtin["metrics"], report_path)
    return {
        "persona_id": persona_id,
        "engine": engine_used,
        "samples": builtin["samples"],
        "metrics": builtin["metrics"],
        "report_path": report_path,
        "elapsed_seconds": report["elapsed_seconds"],
    }