# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""RAGAS 评估：faithfulness / answer_relevancy / context_precision / context_recall。

RAGAS 打分依赖两类外部服务：LLM（四项指标全都要）与 embedding（仅
answer_relevancy 需要）。任一不可用时，在结果里如实写「未执行 / 部分执行」及
确切原因，绝不伪造数值。

与 Task 18 brief 的实现差异（均在本环境实测确认，ragas 0.2.15）：
1. 0.2.x 的 `evaluate()` 需要 langchain 形态的 llm / embeddings——缺省会自建
   OpenAI 客户端去读 OPENAI_API_KEY。故显式传入 ChatOpenAI（取 key 口径同
   `rag04.generate.llm.LLMClient`：API → QWEN_API → OPENAI_API_KEY）与本地
   Ollama 的 bge-m3 embedding。
2. 样本用 0.2 原生的 `EvaluationDataset`/`SingleTurnSample`，不走 legacy 列名
   （question/answer/contexts/ground_truth）的弃用兼容分支。
3. 「需要 embedding 的指标」由 ragas 自身的 `MetricWithEmbeddings` 判定，而非
   写死三个：实测只有 answer_relevancy 需要 embedding，故本地 embedding 不可用
   时只降级这一项，其余三项照常执行并标注为「部分执行」。
4. 结果除 brief 的 `{available, metrics, reason}` 外附执行范围元数据
   （`partial` / `metrics_run` / `metrics_skipped` / `n_samples` / `n_missing`
   / `judge_model`），供报告如实标注哪些跑了、哪些没跑、为什么。
5. 导入 ragas 时局部屏蔽第三方 import 期告警（langchain-community 已 sunset），
   并默认关掉 RAGAS 的用量上报（离线/内网环境）。
"""
from __future__ import annotations

import logging
import os
import types
import warnings
from pathlib import Path

from rag04.config import Settings

logger = logging.getLogger("rag04.ragas")

_METRIC_KEYS = ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall")

_METRIC_DESC = {
    "faithfulness": "答案是否忠于检索上下文（抗幻觉）",
    "answer_relevancy": "答案与问题的相关度",
    "context_precision": "检索上下文的精确度",
    "context_recall": "检索上下文的召回率",
}

# LLM key 口径与 rag04.generate.llm.LLMClient._default_client 一致
_KEY_ENVS = ("API", "QWEN_API", "OPENAI_API_KEY")
_DEFAULT_BASE_URL = "https://api.deepseek.com"


def _unavailable(reason: str) -> dict:
    """构造「未执行」结果：不给任何数值，只写确切原因。"""
    return {
        "available": False,
        "metrics": {},
        "reason": reason,
        "partial": False,
        "metrics_run": [],
        "metrics_skipped": list(_METRIC_KEYS),
        "skipped_reason": reason,
        "n_samples": 0,
        "n_missing": {},
        "judge_model": "",
    }


def _import_ragas() -> types.SimpleNamespace:
    """延迟导入 ragas，并压掉第三方 import 期噪声。

    ragas 0.2.15 的 `ragas.llms.base` 会 import `langchain_community`（已宣布
    sunset），后者在 import 时抛 DeprecationWarning：与本次评估无关，且会污染
    调用方（含测试套件）的告警统计，故只在导入点局部屏蔽。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.metrics.base import MetricWithEmbeddings

    return types.SimpleNamespace(
        EvaluationDataset=EvaluationDataset, SingleTurnSample=SingleTurnSample,
        evaluate=evaluate, faithfulness=faithfulness,
        answer_relevancy=answer_relevancy, context_precision=context_precision,
        context_recall=context_recall, MetricWithEmbeddings=MetricWithEmbeddings)


def _api_key() -> str:
    """按项目既有口径取 LLM key（API → QWEN_API → OPENAI_API_KEY）。"""
    for var in _KEY_ENVS:
        value = os.environ.get(var)
        if value:
            return value
    return ""


def _build_judge(settings: Settings, key: str):
    """构造 RAGAS 0.2.x 要的 langchain LLM（评判模型＝生成模型同一后端）。"""
    from langchain_openai import ChatOpenAI

    base = os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
    # temperature=0 让打分可复现；max_retries=0 与项目 LLM 客户端同口径
    # （重试不叠加，避免一次评估长时间挂死）。
    return ChatOpenAI(model=settings.llm_model, api_key=key, base_url=base,
                      temperature=0, timeout=settings.llm_timeout_s,
                      max_retries=0)


def _build_embeddings(settings: Settings):
    """本地 Ollama 上的 bge-m3。未装 langchain_ollama 时抛错，由调用方降级。"""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embed_model,
                            base_url=settings.ollama_url)


def _needs_embeddings(metric, base_cls) -> bool:
    """指标是否依赖 embedding（以 ragas 自身的基类为准，不写死名单）。"""
    if base_cls is None:                   # 取不到基类时按 0.2.x 实测结论兜底
        return metric.name == "answer_relevancy"
    return isinstance(metric, base_cls)


def run_ragas(result: dict, settings: Settings) -> dict:
    """对评估结果跑 RAGAS。返回 {available, metrics, reason}（外加执行范围标注）。"""
    # RAGAS 默认向 t.explodinggradients.com 上报用量（后台线程 + atexit 各 1s
    # 超时）。本项目评估在离线/内网环境运行，默认关闭上报；调用方显式设置时尊重之。
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    try:
        r = _import_ragas()
    except Exception as e:
        return _unavailable(f"ragas 不可用：{type(e).__name__}: {e}")

    key = _api_key()
    if not key:
        return _unavailable(
            "未配置 LLM API key，RAGAS 需 LLM 打分"
            f"（环境变量 {' / '.join(_KEY_ENVS)} 均未设置）")

    rows = result.get("rows", [])
    if not rows:
        return _unavailable("评估结果无逐题数据（rows 为空），无可评样本")

    candidates = {
        "faithfulness": r.faithfulness,
        "answer_relevancy": r.answer_relevancy,
        "context_precision": r.context_precision,
        "context_recall": r.context_recall,
    }

    # embedding 不可用时只降级依赖它的指标（部分执行），其余照跑。
    embeddings = None
    skipped: list[str] = []
    skipped_reason = ""
    try:
        embeddings = _build_embeddings(settings)
    except Exception as e:
        skipped_reason = (
            f"本地 embedding（{settings.embed_model}@{settings.ollama_url}）"
            f"不可用：{type(e).__name__}: {e}")
        logger.warning("RAGAS 降级为部分执行：%s", skipped_reason)

    base_cls = getattr(r, "MetricWithEmbeddings", None)
    if embeddings is None:
        skipped = [k for k in _METRIC_KEYS
                   if _needs_embeddings(candidates[k], base_cls)]
    run_names = [k for k in _METRIC_KEYS if k not in skipped]
    metrics = [candidates[k] for k in run_names]

    try:
        samples = [
            r.SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=[
                    c["snippet"] for c in row.get("citations", [])
                ] or ["(无上下文)"],
                reference="、".join(row["answer_key"]),
            )
            for row in rows
        ]
        judge = _build_judge(settings, key)
        out = r.evaluate(
            r.EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
        )
        df = out.to_pandas()
        # 只认本次实际提交的指标列：未提交的指标不得从结果里"捡"数值回来。
        values = {k: round(float(df[k].mean()), 4)
                  for k in run_names if k in df.columns}
        if not values:
            raise RuntimeError("RAGAS 跑完但未产出任何指标列")
        n_missing = {k: int(df[k].isna().sum()) for k in values}
    except Exception as e:
        logger.error("RAGAS 评估失败：%s: %s", type(e).__name__, e)
        return _unavailable(f"评估执行失败：{type(e).__name__}: {e}")

    return {
        "available": True,
        "metrics": values,
        "reason": "",
        "partial": bool(skipped),
        "metrics_run": [k for k in _METRIC_KEYS if k in values],
        "metrics_skipped": skipped,
        "skipped_reason": skipped_reason,
        "n_samples": len(df),
        "n_missing": n_missing,
        "judge_model": settings.llm_model,
    }


def write_ragas_report(res: dict, path: Path, mode: str = "full_04") -> Path:
    """产出 RAGAS 评估报告。未执行/部分执行时如实说明范围与原因。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# RAGAS 评估报告", "",
             f"- 运行模式：`{mode}`", ""]

    if not res.get("available"):
        lines += [
            "## 执行状态：未执行", "",
            f"**原因**：{res.get('reason', '未知')}", "",
            "本报告不提供任何未经执行得出的数值。", "",
            "### 复现步骤", "",
            "1. 配置 LLM API key（环境变量 `API` 或 `OPENAI_API_KEY`）",
            "2. 运行 `python scripts/run_eval.py full_04 zh` 生成评估数据",
            "3. 重新运行本脚本", "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        return path

    m = res.get("metrics", {})
    run_keys = list(res.get("metrics_run")
                    or [k for k in _METRIC_KEYS if k in m])
    skipped = list(res.get("metrics_skipped")
                   or [k for k in _METRIC_KEYS if k not in m])
    partial = bool(res.get("partial"))

    lines += ["## 执行状态：部分执行" if partial else "## 执行状态：已执行", ""]
    if partial:
        lines += [
            f"> **部分执行**：已跑 {'、'.join(run_keys) or '无'}；"
            f"未跑 {'、'.join(skipped) or '无'}——"
            f"{res.get('skipped_reason') or res.get('reason') or '未说明原因'}。",
            "",
        ]

    lines += [
        "## 一、指标结果", "",
        "| 指标 | 数值 | 含义 |",
        "| --- | --- | --- |",
    ]
    for key in _METRIC_KEYS:
        value = m[key] if key in m else "未执行"
        lines.append(f"| {key} | {value} | {_METRIC_DESC[key]} |")

    missing = res.get("n_missing") or {}
    nan_keys = [k for k in _METRIC_KEYS if missing.get(k)]
    if nan_keys:
        detail = "、".join(
            f"{k} 缺 {missing[k]}/{res.get('n_samples', '?')} 条" for k in nan_keys)
        lines += [
            "",
            f"> ⚠️ 以下指标有样本未成功打分（`raise_exceptions=False` 下记 NaN，"
            f"均值已跳过这些样本）：{detail}。",
        ]

    lines += [
        "",
        "## 二、说明", "",
        "- 评估语料：16 道工单题目",
        f"- 样本数：{res.get('n_samples', '-')}",
        "- ground truth：`questions.py` 中的人工校核标准答案要点",
        f"- 评判模型：`{res.get('judge_model') or '与生成模型同一后端'}`"
        "（`temperature=0`，但仍存在 LLM 打分波动）",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
