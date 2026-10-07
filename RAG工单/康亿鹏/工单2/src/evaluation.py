# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：RAG 评估模块。实现与 RAGAS 对齐的四项核心指标
          （忠实度 Faithfulness、答案相关性 Answer Relevancy、
            上下文精确率 Context Precision、上下文召回率 Context Recall）。

说明：优先使用 RAGAS 官方实现；当环境中 RAGAS 不可用（依赖冲突）时，
      自动降级为内置的大模型评审（LLM-as-a-Judge）实现，保证评估流程可运行。
"""
import json
import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_cached_llm
from src.utils import logger

# ---------------------------------------------------------------------------
# 内置评审提示词（LLM-as-a-Judge）
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """请评估【回答】是否完全由【参考资料】支持（忠实度）。
评分标准：1.0 表示回答中的每个事实都能在参考资料中找到依据；0 表示大量内容为编造。
只输出 JSON：{{"score": 0~1 的小数, "reason": "简要理由"}}

【参考资料】
{context}

【回答】
{answer}

JSON："""

_RELEVANCY_PROMPT = """请评估【回答】与【问题】的相关程度（答案相关性）。
评分标准：1.0 表示回答直接、完整地回应了问题；0 表示答非所问。
只输出 JSON：{{"score": 0~1 的小数, "reason": "简要理由"}}

【问题】
{question}

【回答】
{answer}

JSON："""

_CONTEXT_PRECISION_PROMPT = """请评估检索到的【上下文片段】中，与【问题】真正相关的比例（上下文精确率）。
评分标准：1.0 表示所有片段都与问题相关；0 表示几乎没有相关片段。
只输出 JSON：{{"score": 0~1 的小数, "reason": "简要理由"}}

【问题】
{question}

【上下文片段】
{context}

JSON："""

_CONTEXT_RECALL_PROMPT = """请评估【上下文片段】是否覆盖了【参考答案】所需的全部信息（上下文召回率）。
评分标准：1.0 表示参考信息全部能在片段中找到；0 表示几乎找不到。
只输出 JSON：{{"score": 0~1 的小数, "reason": "简要理由"}}

【问题】
{question}

【参考答案】
{ground_truth}

【上下文片段】
{context}

JSON："""


@dataclass
class EvalRecord:
    """单条问答的评估记录。"""

    question_id: object = None
    question: str = ""
    rag_answer: str = ""
    pure_answer: str = ""
    contexts: list = field(default_factory=list)
    ground_truth: str = ""
    pages: list = field(default_factory=list)
    response_time: float = 0.0
    metrics: dict = field(default_factory=dict)


def _parse_score(text: str) -> float:
    """从模型输出中解析 0~1 的评分。"""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
        except Exception:
            pass
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        value = float(match.group(1))
        if value > 1:  # 兼容 0~100 打分
            value = value / 100.0
        return max(0.0, min(1.0, value))
    return 0.0


def _judge(prompt_template: str, **kwargs) -> float:
    """调用大模型进行打分，异常时返回 0 分并记录日志。"""
    try:
        llm = get_cached_llm(streaming=False)
        prompt = ChatPromptTemplate.from_template(prompt_template).format(**kwargs)
        content = llm.invoke(prompt).content
        return round(_parse_score(content), 4)
    except Exception as exc:
        logger.warning("评审调用失败：%s", exc)
        return 0.0


def score_faithfulness(question: str, answer: str, contexts: list) -> float:
    return _judge(_FAITHFULNESS_PROMPT, context="\n\n".join(contexts)[:6000], answer=answer)


def score_answer_relevancy(question: str, answer: str) -> float:
    return _judge(_RELEVANCY_PROMPT, question=question, answer=answer)


def score_context_precision(question: str, contexts: list) -> float:
    return _judge(_CONTEXT_PRECISION_PROMPT, question=question, context="\n\n".join(contexts)[:6000])


def score_context_recall(question: str, ground_truth: str, contexts: list) -> float:
    if not ground_truth:
        return None
    return _judge(_CONTEXT_RECALL_PROMPT, question=question, ground_truth=ground_truth,
                  context="\n\n".join(contexts)[:6000])


def evaluate_records_builtin(records: list, use_ragas: bool = True) -> dict:
    """对评估记录计算指标。

    优先尝试 RAGAS 官方实现；不可用时使用内置评审。
    返回 {"backend": ..., "scores": {指标: 均值}, "details": [...]}
    """
    if use_ragas:
        try:
            return _evaluate_with_ragas(records)
        except Exception as exc:
            logger.warning("RAGAS 不可用（%s），降级为内置 LLM 评审。", exc)

    details = []
    for rec in records:
        rec.metrics = {
            "faithfulness": score_faithfulness(rec.question, rec.rag_answer, rec.contexts),
            "answer_relevancy": score_answer_relevancy(rec.question, rec.rag_answer),
            "context_precision": score_context_precision(rec.question, rec.contexts),
            "context_recall": score_context_recall(rec.question, rec.ground_truth, rec.contexts),
        }
        details.append({"id": rec.question_id, "question": rec.question, **rec.metrics})

    valid = {k: [d[k] for d in details if d.get(k) is not None] for k in
             ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
    scores = {k: (round(sum(v) / len(v), 4) if v else None) for k, v in valid.items()}
    return {"backend": "builtin-llm-judge", "scores": scores, "details": details}


def _evaluate_with_ragas(records: list) -> dict:
    """使用 RAGAS 官方实现评估（new API: ragas 0.2+）。"""
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from src.embeddings import get_embedding_model

    # 兼容不同版本的 RAGAS 指标导入路径
    try:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except ImportError:
        from ragas.metrics.collections import (  # type: ignore
            faithfulness, answer_relevancy, context_precision, context_recall,
        )

    samples = [
        {
            "user_input": rec.question,
            "response": rec.rag_answer,
            "retrieved_contexts": rec.contexts or ["（未检索到上下文）"],
            "reference": rec.ground_truth or rec.rag_answer,
        }
        for rec in records
    ]
    dataset = EvaluationDataset.from_list(samples)
    judge_llm = LangchainLLMWrapper(get_cached_llm(streaming=False))
    judge_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    df = result.to_pandas()
    scores = {
        "faithfulness": round(float(df["faithfulness"].mean()), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "context_precision": round(float(df["context_precision"].mean()), 4),
        "context_recall": round(float(df["context_recall"].mean()), 4),
    }
    details = [
        {"id": rec.question_id, "question": rec.question,
         "faithfulness": float(df.loc[i, "faithfulness"]),
         "answer_relevancy": float(df.loc[i, "answer_relevancy"]),
         "context_precision": float(df.loc[i, "context_precision"]),
         "context_recall": float(df.loc[i, "context_recall"])}
        for i, rec in enumerate(records)
    ]
    return {"backend": "ragas", "scores": scores, "details": details}
