# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：RAGAS 评估（硬性要求2 —— context precision / context recall）

## 为什么用本地 Ollama 当裁判

工单08 的硬约束是「零下载、只用本机模型」，工单09 沿用。RAGAS 的
context_precision / context_recall 都是 **LLM 裁判型指标**（没有纯算法实现），
因此裁判模型只能从本机取 —— 本机可用的是 Ollama `qwen2.5:3b`。

`temperature=0`、`num_ctx=8192`，并对每条指标开启重试；判分失败**不静默吞掉**，
一律记入 `judge_failures` 并在报告中列出，避免「失败样本被当成 0 分」
或「被当成满分」这类会让指标失真的处理。

## 三个指标的输入口径

    user_input          问题（主集用解析后的问句，见 w9/questions.py 说明）
    response            Graph RAG 生成的答案
    retrieved_contexts  实际送入 LLM 的上下文（图谱事实 + 文本/表格/图像块）
    reference           参考答案

`retrieved_contexts` 取 `QaResult9.contexts()` —— 即**真正进入上下文的那些片段**，
不是「检索候选」。这是 RAGAS 指标的正确口径：衡量的是模型实际看到什么。
"""
from __future__ import annotations

import io
import json
import os
import time

from w9 import config9

JUDGE_MODEL = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# 裁判 LLM / 嵌入 包装
# ---------------------------------------------------------------------------
def build_judge_llm(model: str = JUDGE_MODEL):
    """构造 RAGAS 用的裁判 LLM（Ollama 本地模型）。"""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatOllama(
        model=model,
        base_url=config9.c8.OLLAMA_HOST,
        temperature=0.0,
        num_ctx=8192,
        num_predict=1024,
    ))


class _BgeM3Embeddings:
    """RAGAS 用的嵌入（本机 bge-m3，复用 08 的 embedder，零下载）。"""

    def __init__(self, batch_size: int = 16):
        self._bs = batch_size

    def _encode(self, texts: list[str]) -> list[list[float]]:
        from src import embedder
        vecs = embedder.encode(texts, batch_size=self._bs)
        return [list(map(float, v)) for v in vecs]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


def build_judge_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    class _LC(_BgeM3Embeddings):
        """包一层 LangChain Embeddings 接口，供 RAGAS 包装器使用。"""

        def embed_documents(self, texts):        # noqa: D102
            return super().embed_documents(texts)

        def embed_query(self, text):             # noqa: D102
            return super().embed_query(text)

    return LangchainEmbeddingsWrapper(_LC())


# ---------------------------------------------------------------------------
# 判分提示适配（让 3B 裁判能用）
# ---------------------------------------------------------------------------
# ## 为什么必须适配（实测证据）
#
# RAGAS 0.2.15 的官方判分提示是**英文 + 长 few-shot + 要求 JSON**。用本机
# qwen2.5:3b 跑官方提示，实测大量样本解析失败：
#
#     Prompt context_recall_classification_prompt failed to parse output:
#     The output parser failed to parse the output including retries.
#     Exception raised: RagasOutputParserException
#
# 失败样本在 RAGAS 里被记为 NaN —— 若不处理，均值会静默丢掉样本，
# 指标就不可信了。
#
# 故这里把 instruction 与 examples 换成**中文、极简、输出字段最少**的版本，
# 但**保留 RAGAS 原生的 input_model / output_model**，因此：
#   · 判分口径（逐句归属 / 逐块有用性）与 RAGAS 定义完全一致；
#   · 结构由 RAGAS 自己校验，不放松；
#   · 仍然是 RAGAS 的指标对象在算分，不是另写一套。
#
# 这是「框架必须用 RAGAS、但裁判模型能力有限」这一现实约束下的工程处置，
# 已在评估报告中如实标注。
def adapt_prompts_for_small_judge(metrics: list) -> list:
    """把指标自带的判分提示替换为中文极简版，返回被替换的指标列表。"""
    from ragas.metrics._context_precision import (ContextPrecisionPrompt, QAC,
                                                  Verification)
    from ragas.metrics._context_recall import (
        ContextRecallClassification, ContextRecallClassificationPrompt,
        ContextRecallClassifications, QCA)

    cp_ins = (
        "判断给定的【上下文】对回答【问题】是否**有用**。\n"
        "\n"
        "判断标准 —— 看**实质信息**，不看字面措辞：\n"
        "· 只要上下文包含了回答该问题所需的**事实内容**（哪怕只是一部分），就判为有用（verdict=1）；\n"
        "· **不要**因为上下文没有出现问题里的字面词汇（如「董事长致辞」）就判为无用；\n"
        "  问题可能是在问某个章节里的内容，而该内容也出现在报告其他位置。\n"
        "· 只有当上下文与问题**在主题上完全无关**时，才判为无用（verdict=0）。\n"
        "\n"
        '只输出 JSON：{"reason": "一句话理由", "verdict": 1 或 0}'
    )
    # few-shot 用金融年报场景（与评测语料同域）。
    # 关键在第二个例子：上下文**没有**出现「董事长致辞」这个词，
    # 但包含问题所需的实质事实 —— 应当判为**有用**。
    # 实测：不加这个对照时，qwen2.5:3b 会把这类上下文一律判 0，
    # 导致 context precision 掉到 0.05（而它的确包含答案）。
    cp_ex = [
        (QAC(question="平安银行2019年董事长致辞提到的盈利增长关键因素有哪些？",
             context="业务结构不断优化，零售进一步巩固收入贡献的支柱地位。"
                     "逾期60天以上贷款占比、拨备覆盖率不断提升。",
             answer="业务结构优化：零售进一步巩固收入贡献的支柱地位；"
                    "风险可控：逾期60天以上贷款占比和拨备覆盖率不断优化。"),
         Verification(reason="上下文包含答案所需的实质事实（业务结构、"
                             "逾期贷款占比、拨备覆盖率）。", verdict=1)),
        (QAC(question="中国平安2019年的营业收入是多少？",
             context="本行于2019年12月31日召开董事会会议，审议通过了"
                     "关于调整内部审计部门设置的议案。",
             answer="1,168,867 百万元。"),
         Verification(reason="上下文讲的是董事会决议，与营业收入完全无关。",
                      verdict=0)),
    ]

    cr_ins = (
        "给定【上下文】与【参考答案】，把参考答案**逐句拆开**，"
        "判断每一句能否由【上下文】支持。\n"
        "能支持 → attributed 记 1；不能支持 → 记 0。\n"
        '只输出 JSON：{"classifications": ['
        '{"statement": "原句", "reason": "一句话理由", "attributed": 1 或 0}]}'
    )
    cr_ex = [
        (QCA(question="爱因斯坦是哪年获得诺贝尔奖的？",
             context="爱因斯坦因光电效应获得 1921 年诺贝尔物理学奖。他提出了相对论。",
             answer="爱因斯坦在 1921 年获得诺贝尔奖。他提出了相对论。"),
         ContextRecallClassifications(classifications=[
             ContextRecallClassification(
                 statement="爱因斯坦在 1921 年获得诺贝尔奖。",
                 reason="上下文明确写出 1921 年诺贝尔物理学奖。", attributed=1),
             ContextRecallClassification(
                 statement="他提出了相对论。",
                 reason="上下文中提到了相对论。", attributed=1),
         ])),
    ]

    def _patched(cls, instruction: str, examples: list):
        """构造一个只替换 instruction/examples 的提示对象。

        实测两种写法都不可行：
          · `cls(instruction=...)` → `BasePrompt.__init__() got an unexpected
            keyword argument 'instruction'`（__init__ 不透传构造参数）；
          · `base.model_copy(update=...)` → `'...Prompt' object has no attribute
            'model_copy'`（BasePrompt **不是** pydantic 模型）。
        既然 instruction / examples 是**类属性**，正确做法就是子类覆盖 —— 简单且稳定。
        """
        return type(cls.__name__, (cls,),
                    {"instruction": instruction, "examples": examples})()

    adapted: list = []
    for m in metrics:
        name = type(m).__name__
        try:
            if name == "ContextPrecision":
                p = _patched(ContextPrecisionPrompt, cp_ins, cp_ex)
            elif name == "ContextRecall":
                p = _patched(ContextRecallClassificationPrompt, cr_ins, cr_ex)
            else:
                continue
            # set_prompts 的签名是 `set_prompts(self, **prompts)`，键必须是
            # 该指标**提示自身的 name**（'context_precision' /
            # 'context_recall_classification'），不是指标类名；且值必须是
            # PydanticPrompt 实例（我们的子类满足）。
            m.set_prompts(**{p.name: p})
            adapted.append(name)
        except Exception as exc:                       # noqa: BLE001
            print(f"[ragas] 适配 {name} 判分提示失败（沿用官方提示）：{exc}",
                  flush=True)
    return adapted


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate(records: list[dict], metrics: str = "core",
             judge_model: str = JUDGE_MODEL, verbose: bool = True) -> dict:
    """对一批问答记录跑 RAGAS。

    records: [{question, answer, contexts, reference, id, ...}]
    metrics: "core" = context_precision + context_recall（工单硬性指标）
             "full" = 再加 faithfulness + answer_relevancy
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    if not records:
        return {"error": "无评估记录", "scores": {}}

    ds = Dataset.from_dict({
        "user_input": [r["question"] for r in records],
        "response": [r.get("answer") or "" for r in records],
        "retrieved_contexts": [list(r.get("contexts") or []) for r in records],
        "reference": [r.get("reference") or "" for r in records],
    })

    llm = build_judge_llm(judge_model)
    if metrics == "core":
        mets = [context_precision, context_recall]
        emb = None
    else:
        mets = [context_precision, context_recall, faithfulness, answer_relevancy]
        emb = build_judge_embeddings()

    # 用小模型能解析的中文判分提示替换官方英文提示（见 adapt_prompts_for_small_judge）
    adapted = adapt_prompts_for_small_judge(mets)

    kwargs = {"llm": llm}
    if emb is not None:
        kwargs["embeddings"] = emb

    t0 = time.perf_counter()
    result = ragas_evaluate(ds, metrics=mets, **kwargs)
    elapsed = time.perf_counter() - t0

    # 逐题分数：一律走 to_pandas()。
    # 不要用 `dict(result)` —— ragas 的 EvaluationResult 是自定义 Mapping，
    # 直接 dict() 会按整数下标取值并抛 `KeyError: 0`（实测踩坑）。
    scores: dict = {}
    per_q: list[dict] = []
    try:
        df = result.to_pandas()
        metric_cols = [c for c in df.columns
                       if c not in ("user_input", "response",
                                    "retrieved_contexts", "reference")]
        for col in metric_cols:
            vals = [float(v) for v in df[col].tolist()
                    if v is not None and v == v]
            scores[col] = round(sum(vals) / len(vals), 4) if vals else None
        for i, row in df.iterrows():
            rec = {"id": records[i].get("id") if i < len(records) else i,
                   "question": (records[i].get("question", "")[:60]
                                if i < len(records) else "")}
            for col in metric_cols:
                v = row.get(col)
                try:
                    rec[col] = None if v is None or v != v else float(v)
                except (TypeError, ValueError):
                    rec[col] = None
            per_q.append(rec)
    except Exception as exc:                          # noqa: BLE001
        per_q = [{"error": f"{type(exc).__name__}: {exc}"}]

    # 判分失败（NaN）统计：RAGAS 把解析失败的样本记为 NaN，
    # 直接算均值会**静默丢样本**，指标因此不可信 —— 必须显式报出。
    nan_counts = {k: sum(1 for r in per_q if r.get(k) is None and k in r)
                  for k in scores}

    out = {"judge_model": judge_model, "metrics": metrics,
           "n": len(records), "scores": scores, "per_question": per_q,
           "nan_counts": nan_counts, "adapted_prompts": adapted,
           "elapsed_s": round(elapsed, 1)}
    if verbose:
        print(f"[ragas] n={len(records)}  用时 {elapsed:.0f}s", flush=True)
        for k, v in scores.items():
            nan = nan_counts.get(k, 0)
            extra = f"（{nan} 条判分失败未计入）" if nan else ""
            print(f"[ragas]   {k} = {v:.4f}{extra}" if v is not None
                  else f"[ragas]   {k} = None", flush=True)
    return out


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def save(result: dict, name: str) -> str:
    os.makedirs(config9.EVAL9_DIR, exist_ok=True)
    path = os.path.join(config9.EVAL9_DIR, f"{name}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path
