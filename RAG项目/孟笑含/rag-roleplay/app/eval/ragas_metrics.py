# -*- coding: utf-8 -*-
"""RAGAS 评测封装：组装样本 → LLM-as-judge 打分 → 返回总分与每题明细。"""
from langchain_core.embeddings import Embeddings
# 解析：langchain 嵌入接口（ragas 需要此接口的嵌入实现）
from langchain_openai import ChatOpenAI
# 解析：OpenAI 兼容聊天模型（judge 用）
from ragas import EvaluationDataset, SingleTurnSample, evaluate
# 解析：ragas 核心组件（数据集/单轮样本/评测入口）
from ragas.embeddings import LangchainEmbeddingsWrapper
# 解析：把 langchain 嵌入包装给 ragas 用
from ragas.llms import LangchainLLMWrapper
# 解析：把 langchain LLM 包装给 ragas 用
from ragas.metrics import (
    # 解析：导入 5 个评测指标
    AnswerCorrectness,
    # 解析：回答正确性
    AnswerRelevancy,
    # 解析：回答相关度
    ContextPrecision,
    # 解析：上下文精确率
    ContextRecall,
    # 解析：上下文召回率
    Faithfulness,
    # 解析：忠实度（幻觉检测）
)

METRIC_NAMES = [
    # 解析：指标名列表（报告输出顺序）
    "faithfulness",
    # 解析：忠实度
    "answer_relevancy",
    # 解析：相关度
    "context_precision",
    # 解析：精确率
    "context_recall",
    # 解析：召回率
    "answer_correctness",
    # 解析：正确性
]


class LocalEmbeddings(Embeddings):
    """把本地 BGE-m3 包装成 langchain Embeddings 接口（ragas 需要）。"""

    def __init__(self, embedder):
        # 解析：构造——包装本地向量化器
        self._embedder = embedder
        # 解析：保存 BGE-m3 实例

    def embed_documents(self, texts):
        # 解析：文档向量化（ragas 调）
        return self._embedder.embed_documents(texts)
        # 解析：透传本地模型

    def embed_query(self, text):
        # 解析：查询向量化
        return self._embedder.embed_query(text)
        # 解析：透传本地模型


def build_judge_llm(base_url: str, api_key: str, model: str):
    """OpenAI 兼容的 judge 大模型（DeepSeek/千问/豆包等均可）。"""
    return ChatOpenAI(
        # 解析：构造 judge LLM
        model=model,
        # 解析：模型名
        api_key=api_key,
        # 解析：密钥
        base_url=base_url,
        # 解析：API 地址
        temperature=0,
        # 解析：温度 0（判分要稳定可复现）
        timeout=120,
        # 解析：超时 120 秒
        max_retries=3,
        # 解析：失败重试 3 次
    )


def run_ragas(samples: list[dict], judge_llm, embeddings) -> tuple[dict[str, float], list[dict]]:
    """samples: EvalPipeline.run_all 的产物。返回 (指标总分, 每题得分行)。"""
    dataset = EvaluationDataset(
        # 解析：构建评测数据集
        samples=[
            # 解析：逐样本
            SingleTurnSample(
                # 解析：单轮样本
                user_input=s["user_input"],
                # 解析：问题
                response=s["response"],
                # 解析：生成回答
                retrieved_contexts=s["retrieved_contexts"],
                # 解析：检索上下文
                reference=s["reference"],
                # 解析：参考答案
                reference_contexts=s["reference_contexts"],
                # 解析：标准上下文
            )
            for s in samples
            # 解析：遍历全部样本
        ]
    )
    metrics = [
        # 解析：评测指标列表
        Faithfulness(),
        # 解析：忠实度
        AnswerRelevancy(),
        # 解析：相关度
        ContextPrecision(),
        # 解析：精确率
        ContextRecall(),
        # 解析：召回率
        AnswerCorrectness(),
        # 解析：正确性
    ]
    result = evaluate(
        # 解析：执行评测（LLM-as-judge）
        dataset=dataset,
        # 解析：数据集
        metrics=metrics,
        # 解析：指标
        llm=LangchainLLMWrapper(judge_llm),
        # 解析：judge 大模型
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        # 解析：本地 BGE 嵌入（context_recall/precision 需要）
    )

    scores: dict[str, float] = {}
    # 解析：总分容器
    try:
        # 解析：优先从 result.scores 取
        scores = {name: float(result.scores[name]) for name in METRIC_NAMES}
        # 解析：逐指标提取
    except Exception:
        # 解析：scores 接口异常（ragas 版本差异）
        table = result.to_pandas()
        # 解析：改用 DataFrame
        scores = {name: float(table[name].mean()) for name in METRIC_NAMES}
        # 解析：取均值

    rows = []
    # 解析：每题明细容器
    table = result.to_pandas()
    # 解析：完整结果表
    for i, sample in enumerate(samples):
        # 解析：逐样本
        row = {"question": sample["user_input"]}
        # 解析：行首是问题
        for name in METRIC_NAMES:
            # 解析：逐指标
            try:
                # 解析：尝试取该行该指标
                row[name] = float(table[name].iloc[i])
                # 解析：取值
            except Exception:
                # 解析：缺失/异常（judge 调用失败）
                row[name] = float("nan")
                # 解析：记 NaN（报告层显示 —）
        rows.append(row)
        # 解析：收集行
    return scores, rows
    # 解析：返回总分与明细
