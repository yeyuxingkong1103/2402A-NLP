#!/usr/bin/env python3
"""用 **RAGAS 0.4** 对本项目已有的评测结果做一次**第三方指标对照**。

为什么要它：自建的 4 个指标（来源命中/要点覆盖/引用保真/分流准确）是**我们自己定义的**，
口径再清楚也难免"自己出题自己判"。RAGAS 是公开的第三方实现，用它跑一遍同一批答案，
至少能回答一个问题：**我们的结论换个尺子量还成不成立**。

⚠️ 三条**必须先说清**的局限（否则读数会被读过头）：

1. **裁判是本地小模型**（默认 `qwen2.5:3b`，跑在 Ollama 上）——
   它本身会出错，尤其是中文长答案。RAGAS 的 faithfulness / context_precision
   都依赖裁判的抽取与判断，**弱裁判的绝对值不可当权威**，只能看**同一裁判下的相对差异**。
2. **"上下文"是引用里的截断片段**（每条 160 字，见 `citations_raw[].snippet`），
   **不是**真正喂给大模型的完整检索上下文 ⇒ faithfulness / context_precision 是**近似**。
3. **没有 ground truth**（我们没有人工参考答案，只有期望要点 `claims`）⇒
   **不能**跑 context_recall / context_precision_with_reference / answer_correctness。
   这里只跑**不需要参考答案**的三个：`ResponseRelevancy`、`Faithfulness`、
   `LLMContextPrecisionWithoutReference`。

本脚本**只用标准库 + ragas**（不 import 本项目的包），因此可以直接在 `ragas` 那个 conda 环境里跑。

用法（在 ragas 环境里）::

    E:\\Anaconda\\envs\\ragas\\python.exe scripts/eval_ragas.py \\
        --in eval/results/recall-off.jsonl --limit 20 \\
        --out eval/results/ragas-recall-off.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 用户可能写的名字 -> ragas 的规范指标名（`--metrics` 里允许用短名/别名）。
RAGAS_METRIC_ALIASES = {
    "relevancy": "response_relevancy",
    "answer_relevancy": "response_relevancy",
    "response_relevancy": "response_relevancy",
    "faithfulness": "faithfulness",
    "context_precision": "llm_context_precision_without_reference",
    "llm_context_precision_without_reference": "llm_context_precision_without_reference",
}

#: 每个规范指标**要不要嵌入向量**。
#: ⚠️ 这条决定了"本地 vLLM 能不能独立跑"：vLLM 只服务 chat，**没有 `/v1/embeddings`**，
#: 所以只想出一版可信读数时，就选**不需要嵌入**的那两个（只要裁判模型）。
RAGAS_METRIC_NEEDS_EMBEDDINGS = {
    "response_relevancy": True,
    "faithfulness": False,
    "llm_context_precision_without_reference": False,
}

#: 默认三件套（需要嵌入 —— 之前本机跑 pilot 就是这么跑的）。
DEFAULT_RAGAS_METRICS = "relevancy,faithfulness,context_precision"


def resolve_metrics(spec: str) -> tuple[list[str], bool]:
    """把 ``--metrics`` 解析成 ``(规范指标名列表, 是否需要嵌入)``；认不出的名字抛 ``ValueError``。

    抽成**纯函数**是为了能在**没装 ragas** 的机器上单测这条选择逻辑
    （装了 ragas 才能构造 Metric 对象，但"选哪些"这一步根本不需要它）。
    """
    names: list[str] = []
    for piece in str(spec or "").split(","):
        token = piece.strip().lower()
        if not token:
            continue
        if token not in RAGAS_METRIC_ALIASES:
            raise ValueError(f"不认识的指标 {token!r}；可选：{sorted(set(RAGAS_METRIC_ALIASES))}")
        canonical = RAGAS_METRIC_ALIASES[token]
        if canonical not in names:          # 去重但保持用户给的顺序
            names.append(canonical)
    if not names:
        raise ValueError("--metrics 是空的：至少要给一个指标")
    needs = any(RAGAS_METRIC_NEEDS_EMBEDDINGS[name] for name in names)
    return names, needs


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_dataset(records: list[dict], limit: int) -> tuple[list[dict], dict]:
    """把我们的评测记录变成 RAGAS 的样本；**跳过与原因都计数**（不静默丢数据）。"""
    skipped: dict[str, int] = {"error": 0, "empty_answer": 0, "no_context": 0}
    samples: list[dict] = []
    for record in records:
        if record.get("error"):
            skipped["error"] += 1
            continue
        answer = str(record.get("answer") or "").strip()
        if not answer:
            skipped["empty_answer"] += 1
            continue
        contexts = [str(cite.get("snippet") or "").strip()
                    for cite in (record.get("citations_raw") or []) if cite.get("snippet")]
        if not contexts:
            # 拒答/日常题本来就没有引用；这类题**语义上**也评不了 faithfulness
            skipped["no_context"] += 1
            continue
        samples.append({
            "user_input": str(record.get("question") or ""),
            "response": answer,
            "retrieved_contexts": contexts,
            "id": record.get("id"), "expect": record.get("expect"),
        })
        if limit and len(samples) >= limit:
            break
    return samples, skipped


def _legacy_embeddings(inner):
    """把 ragas 0.4 的**新**嵌入接口适配成 legacy 指标要的接口。

    为什么需要：0.4.3 里 legacy 指标（`ResponseRelevancy` 等）走的是
    `BaseRagasEmbeddings.embed_query/embed_documents`，而
    `ragas.embeddings.OpenAIEmbeddings` 是新体系（只有 `embed_text/embed_texts`）。
    直接用会在评估中途抛 `AttributeError: 'OpenAIEmbeddings' object has no attribute
    'embed_query'`，而 ragas 把它吞成 `nan` —— **看起来像"这个指标算不出来"**，
    其实是接线错了（第一次跑就被骗了三分钟）。
    """
    from ragas.embeddings.base import BaseRagasEmbeddings

    class _Adapter(BaseRagasEmbeddings):
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def embed_query(self, text: str) -> list[float]:
            return self._wrapped.embed_text(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._wrapped.embed_texts(list(texts))

        async def aembed_query(self, text: str) -> list[float]:
            return await self._wrapped.aembed_text(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await self._wrapped.aembed_texts(list(texts))

    return _Adapter(inner)


def _inject_no_think(completions, *, max_tokens: int | None = None):
    """把 `completions` 资源包一层：每次 `create` 都注入"关思考"的字段（**纯逻辑，不依赖 openai**）。

    单独抽出来是为了能在**没装 openai** 的机器上单测这段注入逻辑
    （项目 venv 里就没有 openai；而真正建客户端的部分必须在 ragas 环境里跑）。

    ``max_tokens`` 也很关键（2026-09-28 真机踩到）：**ragas/instructor 自己不设 max_tokens**，
    于是 vLLM 会一直生成到模型上限（本机 `--max-model-len 16384`）——
    实测**每个裁判调用要 ~640 秒**、还伴随 `IncompleteOutputException` 与自动重试，
    40 个任务跑 10 分钟才完成 1 个（照这速度要 7 小时）。给个上限就正常了；
    生产链路本来也是 `LLM_MAX_TOKENS=1024`，所以这里默认值跟生产一致。
    """
    class _Completions:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def create(self, **kwargs):
            extra = dict(kwargs.pop("extra_body", None) or {})
            extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
            if max_tokens:
                kwargs.setdefault("max_tokens", max_tokens)
            return self._wrapped.create(extra_body=extra, **kwargs)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    return _Completions(completions)


def _no_think_client(inner, *, max_tokens: int | None = 1024):
    """把客户端换成**关思考**的版本：每次 ``chat.completions.create`` 都带
    ``chat_template_kwargs={"enable_thinking": False}``。

    为什么要这样（2026-09-28 真机，两次踩坑）：
    * 本地 vLLM 的 27B 默认**会先输出一大段思考**，而生产是靠请求体里的这个字段关掉的
      （见 `legal_rag/generate/openai_compat.py`）。RAGAS 自己组装请求，插不进这个字段
      ⇒ 只能在"客户端"这一层注入，否则量出来的 faithfulness 是裁判在自说自话、还慢好几倍；
    * ⚠️ **第一版写成"包一层普通对象"，结果 ragas/instructor 直接崩**：
      `UserWarning: Client should be an instance of openai.OpenAI or openai.AsyncOpenAI`，
      随后每个任务都报 `AttributeError: 'NoneType' object has no attribute 'chat'`
      —— 它做了 isinstance 检查，非 OpenAI 的客户端会被当成空壳。
    * 所以现在反过来：**继承 `openai.OpenAI`**（isinstance 天然成立），只覆写 `chat`，
      让 `chat.completions.create` 走注入版本，其余行为完全不变。

    ``inner`` 是已经建好的 `OpenAI` 实例：**复用它的全部内部状态**（连 http 客户端/连接池都同一个），
    只是换成子类。
    """
    from openai import OpenAI

    class _NoThinkOpenAI(OpenAI):
        @property
        def chat(self):                      # type: ignore[override]
            base = super().chat
            wrapper = _NoThinkChat(base, getattr(self, "_judge_max_tokens", max_tokens))
            return wrapper

    class _NoThinkChat:
        def __init__(self, wrapped, max_tokens: int | None):
            self._wrapped = wrapped
            self.completions = _inject_no_think(wrapped.completions, max_tokens=max_tokens)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    replacement = _NoThinkOpenAI.__new__(_NoThinkOpenAI)
    replacement.__dict__.update(inner.__dict__)
    replacement._judge_max_tokens = max_tokens          # 供下面 chat 属性取用
    return replacement


#: 认定"裁判就是生产那个 27B"的模型名（命中就提示**自我评判**偏差）。
PRODUCTION_JUDGE_NAMES = {"qwen27b", "qwen3-27b", "qwen3.8-27b-fp8"}


def judge_limitations(judge_model: str, judge_max_tokens: int) -> list[str]:
    """按**实际配置**生成"这次读数有哪些限制"（纯函数，便于单测）。

    为什么要有它：这段文案以前是写死的"裁判是本地小模型（弱）"——
    换成 27B 之后就成了**假话**（2026-09-28 跑完才发现）。读数报告里的限制条款
    必须跟着配置走，否则后人会照着错的限制去理解数字。
    """
    limitations = [
        "上下文取自引用片段（每条截断 160 字），不是完整检索上下文",
        "无 ground truth ⇒ 不跑 context_recall / answer_correctness",
    ]
    if judge_model in PRODUCTION_JUDGE_NAMES:
        limitations.insert(0, "裁判=本机 27B（与**生产同模型**）：因此存在**自我评判**偏差，"
                              "绝对值仍不可当权威，只看相对趋势")
    else:
        limitations.insert(0, f"裁判={judge_model}（可能是弱模型）：绝对值不可当权威")
    if judge_max_tokens and judge_max_tokens < 4096:
        limitations.append(f"裁判 max_tokens={judge_max_tokens}：结构化输出可能被截断"
                           f"（表现为指标值缺失 nan，实测 faithfulness 中招）")
    return limitations


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 第三方指标对照")
    parser.add_argument("--in", dest="inputs", action="append", required=True,
                        help="已有评测结果 JSONL（可多次）")
    parser.add_argument("--limit", type=int, default=20, help="每个文件最多评几题（裁判很慢）")
    parser.add_argument("--judge-model", default="qwen2.5:3b")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--embed-model", default="bge-m3:latest")
    parser.add_argument("--embed-base-url", default="",
                        help="嵌入端点（默认与裁判同一个）。本地 vLLM **不提供** /v1/embeddings，"
                             "所以只用不需要嵌入的指标时可以留空")
    parser.add_argument("--metrics", default=DEFAULT_RAGAS_METRICS,
                        help="要跑哪些指标（逗号分隔）。response_relevancy 需要**嵌入**；"
                             "faithfulness / context_precision **只需要裁判模型**。"
                             "想在没有嵌入端点时跑，就写 --metrics faithfulness,context_precision")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="并发请求数。RAGAS 默认 **16** —— 走 SSH 隧道 + 单卡 vLLM 时太大："
                             "2026-09-28 实测 16 并发把隧道打断，24 个指标里有 11 个变成 nan")
    parser.add_argument("--judge-max-tokens", type=int, default=1024,
                        help="单次裁判调用的 max_tokens。[注意] **别设 0**：ragas 自己不设上限，"
                             "vLLM 会一直生成到模型上限 —— 实测每调用 ~640 秒、"
                             "40 个任务 10 分钟只完成 1 个（与生产 LLM_MAX_TOKENS=1024 对齐）")
    parser.add_argument("--judge-timeout", type=int, default=180, help="单次裁判调用的超时（秒）")
    parser.add_argument("--think", action="store_true",
                        help="让裁判**保留**思考过程（默认关，与生产一致："
                             "请求体带 chat_template_kwargs={'enable_thinking': False}）")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    try:
        from openai import OpenAI
        from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        # ⚠️ 用 `ragas.metrics`（legacy `Metric` 体系）而不是 `ragas.metrics.collections`：
        #    0.4.3 的 `evaluate()` 会校验 `isinstance(m, ragas.metrics.base.Metric)`，
        #    而 collections 里的新类继承的是 `BaseMetric` ⇒ 会被判为"没初始化"直接 TypeError。
        #    （collections 是给新的 @experiment 用的；这里求稳走 legacy，屏蔽弃用告警。）
        import warnings
        # ⚠️ 别加 `module="ragas"`：DeprecationWarning 是从**调用方**（本脚本）这一层抛出来的，
        #    按模块过滤会漏掉、日志里刷三行弃用告警（第一次在云端跑就是这样）。
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ragas.metrics import (Faithfulness,
                                   LLMContextPrecisionWithoutReference,
                                   ResponseRelevancy)
    except Exception as exc:  # noqa: BLE001
        print(f"!! 依赖缺失（要在 ragas 环境里跑）：{type(exc).__name__}: {exc}\n"
              f"   例：E:\\Anaconda\\envs\\ragas\\python.exe {Path(__file__).name}", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.judge_base_url, api_key="ollama", timeout=300.0)
    if not args.think:
        # 与生产同口径：关掉裁判的思考过程（否则它会先自说自话一大段，又慢又失真）
        client = _no_think_client(client, max_tokens=max(args.judge_max_tokens, 1))
    judge = llm_factory(args.judge_model, provider="openai", client=client)

    #: 指标名 -> (构造函数, **要不要嵌入**)。要嵌入的只有 response_relevancy。
    try:
        selected, needs_embeddings = resolve_metrics(args.metrics)
    except ValueError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 2
    builders = {
        "response_relevancy": lambda emb: ResponseRelevancy(llm=judge, embeddings=emb),
        "faithfulness": lambda emb: Faithfulness(llm=judge),
        "llm_context_precision_without_reference":
            lambda emb: LLMContextPrecisionWithoutReference(llm=judge),
    }
    embed_base = args.embed_base_url or args.judge_base_url
    embeddings = None
    if needs_embeddings:
        embed_client = client if embed_base == args.judge_base_url else OpenAI(
            base_url=embed_base, api_key="ollama", timeout=300.0)
        embeddings = _legacy_embeddings(OpenAIEmbeddings(client=embed_client, model=args.embed_model))
    metrics = [builders[name](embeddings) for name in selected]
    print(f"裁判模型 = {args.judge_model}（{args.judge_base_url}）")
    print(f"嵌入模型 = {args.embed_model if needs_embeddings else '（本次指标不需要嵌入）'}"
          + (f"（{embed_base}）" if needs_embeddings else ""))
    print(f"指标     = {[m.name for m in metrics]}")
    if not needs_embeddings:
        print("           [注意] 本地 vLLM 没有 /v1/embeddings，所以**跳过了需要嵌入的 "
              "response_relevancy**；这不是失败，是『用能用的指标先出一版可信读数』"
              "（faithfulness / context_precision 只要裁判模型）")

    # ⚠️ 限制文案要**跟着实际配置走**：以前写死"裁判是本地小模型（弱）"，
    #    换成本机 27B 之后就成了假话（2026-09-28 真机跑完才发现）。逻辑见 :func:`judge_limitations`。
    report: dict = {
        "judge_model": args.judge_model, "embed_model": args.embed_model,
        "metrics": [m.name for m in metrics],
        "needs_embeddings": needs_embeddings,
        "max_workers": args.max_workers, "judge_timeout": args.judge_timeout,
        "limitations": judge_limitations(args.judge_model, args.judge_max_tokens),
        "runs": {},
    }
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.is_absolute():
            path = ROOT / path
        records = load_records(path)
        samples, skipped = build_dataset(records, args.limit)
        print(f"\n=== {path.name}：读 {len(records)} 题，取 {len(samples)} 题"
              f"（跳过 {skipped}）")
        if not samples:
            print("  没有可评的样本（都缺引用或都出错），跳过")
            report["runs"][path.stem] = {"evaluated": 0, "skipped": skipped}
            continue

        dataset = EvaluationDataset(samples=[
            SingleTurnSample(user_input=s["user_input"], response=s["response"],
                             retrieved_contexts=s["retrieved_contexts"])
            for s in samples])
        started = time.perf_counter()
        # ⚠️ 并发必须可调：RAGAS 默认 16 个 worker，走 SSH 隧道 + 单卡 vLLM 时会把连接打挂
        #    （2026-09-28 实测：隧道直接断，24 个指标值里 11 个成了 nan —— 看起来像"指标算不出来"，
        #    其实是并发太高把唯一通路搞断了）。
        try:
            run_config = RunConfig(max_workers=max(1, args.max_workers),
                                   timeout=max(30, args.judge_timeout))
        except Exception as exc:  # noqa: BLE001 - 老版本 ragas 没有 RunConfig
            print(f"[注意] 这个 ragas 版本不支持 RunConfig（{type(exc).__name__}）："
                  f"并发只能用默认值，隧道可能被打断", file=sys.stderr)
            run_config = None
        result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False,
                          show_progress=True, run_config=run_config)
        elapsed = time.perf_counter() - started
        frame = result.to_pandas()

        per_item: list[dict] = []
        missing: dict[str, int] = {metric.name: 0 for metric in metrics}
        for index, sample in enumerate(samples):
            row = frame.iloc[index]
            entry = {"id": sample["id"], "expect": sample["expect"]}
            for metric in metrics:
                value = row.get(metric.name)
                try:
                    number = float(value) if value is not None else None
                except (TypeError, ValueError):
                    number = None
                # ⚠️ `nan` 必须当成**缺失**并计数：ragas 会把"裁判没给出可解析结果"
                #    吞成 nan，直接平均就成了"看起来有分"的假数字（本轮真踩过）。
                if number is None or number != number:
                    number = None
                    missing[metric.name] += 1
                entry[metric.name] = number
            per_item.append(entry)

        def _mean(name: str) -> float | None:
            values = [item[name] for item in per_item if item.get(name) is not None]
            return round(statistics.fmean(values), 4) if values else None

        summary = {metric.name: _mean(metric.name) for metric in metrics}
        print(f"  耗时 {elapsed:.0f}s；均分 = "
              + "  ".join(f"{k}={v}" for k, v in summary.items()))
        if any(missing.values()):
            print(f"  [注意] 有指标**算不出值**（ragas 吞成 nan，已按缺失计数，不计入均分）：{missing}")
            print("         常见原因：裁判模型输出被截断/不按结构化格式（换更强的裁判再试）")
        by_expect: dict[str, dict] = {}
        for expect in sorted({item["expect"] or "?" for item in per_item}):
            group = [item for item in per_item if (item["expect"] or "?") == expect]
            by_expect[expect] = {
                "n": len(group),
                **{metric.name: (round(statistics.fmean(
                    [g[metric.name] for g in group if g.get(metric.name) is not None]), 4)
                    if any(g.get(metric.name) is not None for g in group) else None)
                    for metric in metrics},
            }
            print(f"    {expect:<20} n={len(group):<4} "
                  + "  ".join(f"{k}={v}" for k, v in by_expect[expect].items() if k != "n"))
        report["runs"][path.stem] = {
            "input": str(path), "evaluated": len(samples), "skipped": skipped,
            "elapsed_seconds": round(elapsed, 1), "mean": summary,
            "missing_per_metric": missing,
            "by_expect": by_expect, "per_item": per_item,
        }

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\n证据已写：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
