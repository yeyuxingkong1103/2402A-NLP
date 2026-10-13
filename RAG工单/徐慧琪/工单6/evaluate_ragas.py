# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）
模块：RAGAS 评估（本地 Ollama 裁判 + 本地 bge-m3，全程离线）

用途与口径：
  对本工单的 RAG 问答链路（src/rag.py::ask）做 RAGAS 四项指标评估，产出
  data/eval/ragas_results.json，供验收与「向量 / 全文 / 混合」三种检索策略对比。

  · 待评样本：data/test_questions.json 的测试题，逐题 rag.ask(use_cache=False)
    收集 (question, answer, contexts)；金标取 data/eval/ground_truth.json 的
    reference —— 缺金标的题**如实跳过并在结果中标注**，绝不编造金标。
  · 评判 LLM：本机 Ollama qwen2.5:3b（langchain-ollama，temperature=0）；
  · 评判 Embedding：本机 Ollama bge-m3（仅 answer_relevancy 的反向生成需要）。
  · 技术路线与工单5 的 RAGAS 一致（同裁判模型、同 embedding、同指标口径），
    两轮数字可横向对比。

  离线保证：脚本开头 `from src import bootstrap` 已在进程级置好离线开关，
  评判模型全部取自本机已 pull 的 Ollama 模型，不触发任何下载。

  stage 中间产物：先把 rag.ask 的原始回答落 data/eval/ragas_answers.json
  （回答 + contexts），重跑评判可用 --reuse-answers 复用，不必重新问答。

用法：
  python evaluate_ragas.py                     # 全部测试题，hybrid 策略
  python evaluate_ragas.py --limit 2 --out data/eval/ragas_smoke.json   # 冒烟
  python evaluate_ragas.py --reuse-answers     # 复用回答，只重跑评判
  python evaluate_ragas.py --strategy vector   # 换检索策略后对比
  python evaluate_ragas.py --metrics faithfulness,context_recall
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402,F401 —— 必须最先导入（离线开关 + Windows 大栈）

# 终端可能是 GBK：个别符号打印不出来时替换而不是抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_ALL_METRICS = ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall")
_DEFAULT_METRICS = "faithfulness,answer_relevancy,context_precision,context_recall"
_DEFAULT_OUT = os.path.join("data", "eval", "ragas_results.json")
_STAGE_NAME = "ragas_answers.json"

_METRIC_DESC = {
    "faithfulness": "回答是否忠实于检索上下文（1=无编造）",
    "answer_relevancy": "回答与问题的相关性（由答案反向生成问题计算）",
    "context_precision": "检索上下文中与标准答案相关的比例（需金标）",
    "context_recall": "检索上下文对标准答案的覆盖度（需金标）",
}


def _banner(text: str) -> None:
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}", flush=True)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_path(p: str) -> str:
    """相对路径一律相对**项目根**（而不是当前工作目录）解析。"""
    return p if os.path.isabs(p) else os.path.join(bootstrap.project_root(), p)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def _parse_metrics(raw: str) -> list[str]:
    """解析 --metrics（逗号分隔），未知指标名直接报错，不静默忽略。"""
    names = [s.strip() for s in (raw or "").split(",") if s.strip()]
    unknown = [n for n in names if n not in _ALL_METRICS]
    if unknown:
        raise ValueError(f"未知指标 {unknown}（可选：{list(_ALL_METRICS)}）")
    if not names:
        raise ValueError("--metrics 不能为空")
    return names


# ---------------------------------------------------------------------------
# 评判模型（本机 Ollama，不下载）
# ---------------------------------------------------------------------------
def _check_judge() -> dict:
    """检查裁判 LLM / embedding 是否在本机 Ollama 里（只读 /api/tags）。

    裁判 LLM 缺失 = 无法评判 → 调用方明确报错并非 0 退出；
    embedding 缺失只影响 answer_relevancy，可由调用方降级为部分执行。
    """
    from src import config, llm

    try:
        models = llm.list_models(force=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"无法连接本机 Ollama（{config.OLLAMA_HOST}）：{exc}\n"
            f"请先启动 Ollama 服务（ollama serve）后重试；禁止下载模型。") from exc

    def _pick(target: str) -> str:
        for name in models:
            if name == target or name.split(":")[0] == target.split(":")[0]:
                return name
        return ""

    judge_llm = _pick(config.OLLAMA_MODEL)
    judge_emb = _pick(config.EMBEDDING_MODEL_NAME)
    if not judge_llm:
        raise RuntimeError(
            f"本机 Ollama 中找不到评判 LLM {config.OLLAMA_MODEL!r}。"
            f"已安装：{models}\n本工单禁止下载，请先 pull 或改 config.OLLAMA_MODEL。")
    return {"llm": judge_llm, "embeddings": judge_emb,
            "host": config.OLLAMA_HOST, "available": models}


def _build_judge_llm(model: str):
    """ragas 要 langchain 形态的 LLM；指向本机 Ollama（temperature=0 可复现）。"""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper
    from src import config

    lc = ChatOllama(model=model, base_url=config.OLLAMA_HOST,
                    temperature=0.0, num_ctx=config.OLLAMA_NUM_CTX)
    return LangchainLLMWrapper(lc)


def _build_judge_embeddings(model: str):
    """ragas embeddings 走本机 Ollama 的 bge-m3（独立进程，不再在本地重复加载）。"""
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src import config

    emb = OllamaEmbeddings(model=model, base_url=config.OLLAMA_HOST)
    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# 第一步：收集 rag.ask 的原始回答（stage 产物）
# ---------------------------------------------------------------------------
def _compact_contexts(result) -> tuple[list[str], list[dict]]:
    """RagResult.contexts（父块片段）→ (纯文本列表, 定位信息列表)。

    RAGAS 的 contexts 只吃字符串；定位信息（来源/页码/是否表格）另存一份，
    便于逐题回溯"这个片段到底来自哪"。
    """
    texts: list[str] = []
    meta: list[dict] = []
    for b in (result.contexts or []):
        text = (b.get("parent_text") or b.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        meta.append({
            "source": b.get("source", ""),
            "page_display": int(b.get("page_idx", 0)) + 1,
            "is_table": bool(b.get("is_table")),
            "table_caption": b.get("table_caption", ""),
        })
    return texts, meta


def _load_inputs(limit: int):
    """读测试题与金标；返回 (questions, gt_by_id)。"""
    from src import config

    with open(config.EVAL_QUESTIONS_FILE, encoding="utf-8") as fh:
        questions = json.load(fh)
    with open(config.GT_FILE, encoding="utf-8") as fh:
        gt = {g["id"]: g for g in json.load(fh)}
    if limit:
        questions = questions[:limit]
    return questions, gt


def _read_stage(stage_file: str) -> dict:
    with open(stage_file, encoding="utf-8") as fh:
        return json.load(fh)


def collect_answers(limit: int, strategy: str, reuse: bool,
                    errors: list, skipped: list) -> tuple[list[dict], str]:
    """逐题 rag.ask 收集回答；--reuse-answers 时直接读 stage 产物。

    · 缺金标（ground_truth.json 无该 id 或 reference 为空）的题**跳过**，
      记入 skipped，不进入 RAGAS 样本，也不去问答；
    · 单题问答异常不中断整体：记入 errors 后继续下一题。

    返回 (rows, used_strategy)：used_strategy 是**回答实际收集时**的策略
    （复用 stage 时以 stage 为准，防止用 --strategy 覆盖出错误口径）。
    """
    from src import config, rag, runtime

    questions, gt = _load_inputs(limit)
    stage_file = os.path.join(config.EVAL_DIR, _STAGE_NAME)

    if reuse:
        if not os.path.isfile(stage_file):
            raise RuntimeError(f"--reuse-answers 需要 {stage_file}，请先完整跑一次收集回答")
        stage = _read_stage(stage_file)
        rows = list(stage.get("rows", []))
        used_strategy = stage.get("strategy") or strategy
        if used_strategy != strategy:
            print(f"[警告] stage 产物的检索策略是 {used_strategy}，"
                  f"与本次 --strategy {strategy} 不同；结果按 {used_strategy} 记录"
                  f"（要换策略请不加 --reuse-answers 重跑问答）", flush=True)
        if limit:
            rows = rows[:limit]
        elif stage.get("limit"):
            print(f"[警告] stage 产物是 --limit {stage['limit']} 时生成的，"
                  f"本次复用只有 {len(rows)} 题", flush=True)
        # 金标以**当前** ground_truth.json 为准重新挂接（金标可能更新过）
        kept: list[dict] = []
        for r in rows:
            ref = (gt.get(r.get("id")) or {}).get("reference", "")
            if ref:
                r["reference"] = ref
            if not r.get("reference"):
                skipped.append({"id": r.get("id"), "question": r.get("question", ""),
                                "reason": "ground_truth.json 中无该题 reference（如实跳过，未编造金标）"})
                continue
            kept.append(r)
        print(f"[reuse] 从 {stage_file} 复用 {len(kept)} 题回答（未重新问答）", flush=True)
        return kept, used_strategy

    runtime.update(strategy=strategy)
    print(f"[runtime] 检索策略 = {strategy} | {json.dumps(runtime.describe(), ensure_ascii=False)}",
          flush=True)

    todo: list[dict] = []
    for q in questions:
        ref = (gt.get(q["id"]) or {}).get("reference", "")
        if not ref:
            skipped.append({"id": q["id"], "question": q["question"],
                            "reason": "ground_truth.json 中无该题 reference（如实跳过，未编造金标）"})
            print(f"[skip] id={q['id']} 无金标 reference，跳过", flush=True)
            continue
        todo.append(q)

    _banner(f"warmup（加载本地模型，首次约 1~2 分钟）")
    t0 = time.time()
    warm = rag.warmup()
    print(f"warmup: {json.dumps(warm, ensure_ascii=False)} | {time.time() - t0:.1f}s", flush=True)

    rows: list[dict] = []
    for i, q in enumerate(todo, start=1):
        qid, question = q["id"], q["question"]
        print(f"[{i}/{len(todo)}] id={qid} {question[:36]}...", flush=True)
        t1 = time.time()
        try:
            r = rag.ask(question, use_cache=False)
        except Exception as exc:  # noqa: BLE001 —— 单题失败不中断整体
            errors.append({"stage": "ask", "id": qid,
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [error] 问答失败，已跳过该题：{exc}", flush=True)
            continue
        contexts, ctx_meta = _compact_contexts(r)
        rows.append({
            "id": qid,
            "question": question,
            "answer": r.answer,
            "contexts": contexts,
            "context_meta": ctx_meta,
            "reference": (gt.get(qid) or {}).get("reference", ""),
            "mode": r.mode,
            "from_cache": r.from_cache,
            "timings": r.timings,
            "elapsed_s": round(time.time() - t1, 2),
        })
        print(f"  mode={r.mode} n_contexts={len(contexts)} "
              f"{time.time() - t1:.1f}s | {r.answer[:50]}", flush=True)

    # stage 产物先落盘：即使后续 RAGAS 评判失败，回答也已保存可复用
    _write_json(stage_file, {
        "generated_at": _now(),
        "strategy": strategy,
        "limit": limit,
        "warmup": warm,
        "n": len(rows),
        "rows": rows,
        "skipped_no_gt": skipped,
    })
    print(f"stage 产物已保存：{stage_file}（{len(rows)} 题）", flush=True)
    return rows, strategy


# ---------------------------------------------------------------------------
# 第二步：RAGAS 评判
# ---------------------------------------------------------------------------
_METRIC_POOL: dict = {}


def _metric_objects(names: list[str], llm_w, emb_w) -> list:
    """取 ragas 指标对象并绑定本地裁判（llm 全部需要，embedding 仅相关性需要）。"""
    if not _METRIC_POOL:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ragas.metrics import (answer_relevancy, context_precision,
                                       context_recall, faithfulness)
        _METRIC_POOL.update({
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        })
    metrics = []
    for name in names:
        m = _METRIC_POOL[name]
        m.llm = llm_w
        if name == "answer_relevancy":
            m.embeddings = emb_w
        metrics.append(m)
    return metrics


def _scores_from_df(df, rows: list[dict], metric_names: list[str],
                    errors: list) -> list[dict]:
    """结果 DataFrame → 逐题明细；NaN 记为 None 并作为该题该指标的失败记录。"""
    per_q: list[dict] = []
    for i, r in enumerate(rows):
        row = {"id": r["id"], "question": r["question"], "answer": r["answer"],
               "mode": r.get("mode", ""), "n_contexts": len(r.get("contexts") or [])}
        for m in metric_names:
            if m not in df.columns:
                row[m] = None
                errors.append({"stage": "ragas", "id": r["id"], "metric": m,
                               "error": "结果表中无该指标列（该指标整体未产出）"})
                continue
            v = df.iloc[i][m]
            if v != v:                       # NaN：裁判超时/失败，无有效分数
                row[m] = None
                errors.append({"stage": "ragas", "id": r["id"], "metric": m,
                               "error": "裁判未产出有效分数（NaN），汇总时已按缺失处理"})
            else:
                row[m] = round(float(v), 3)
        per_q.append(row)
    return per_q


def _evaluate_dataset(rows: list[dict], metrics: list, llm_w, emb_w, show: bool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from datasets import Dataset
        from ragas import evaluate
    ds = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [list(r.get("contexts") or []) for r in rows],
        "ground_truth": [r.get("reference") or "" for r in rows],
    })
    result = evaluate(ds, metrics=metrics, llm=llm_w, embeddings=emb_w,
                      show_progress=show)
    return result.to_pandas()


def run_ragas(rows: list[dict], metric_names: list[str], llm_w, emb_w,
              errors: list) -> dict:
    """整体评判；整体失败则退化为**逐题评判**，单题失败只记录不中断。"""
    metrics = _metric_objects(metric_names, llm_w, emb_w)

    try:
        df = _evaluate_dataset(rows, metrics, llm_w, emb_w, show=True)
        per_q = _scores_from_df(df, rows, metric_names, errors)
    except Exception as exc:  # noqa: BLE001 —— 整体失败：逐题隔离重试
        errors.append({"stage": "ragas", "scope": "batch",
                       "error": f"整体评判失败，已退化为逐题评判：{type(exc).__name__}: {exc}"})
        print(f"[warn] 整体评判失败（{type(exc).__name__}: {exc}），改为逐题评判", flush=True)
        per_q = []
        for r in rows:
            try:
                df = _evaluate_dataset([r], metrics, llm_w, emb_w, show=False)
                per_q.extend(_scores_from_df(df, [r], metric_names, errors))
            except Exception as exc2:  # noqa: BLE001 —— 单题失败：记录并继续
                errors.append({"stage": "ragas", "id": r["id"],
                               "error": f"单题评判失败：{type(exc2).__name__}: {exc2}"})
                per_q.append({"id": r["id"], "question": r["question"],
                              "answer": r["answer"], "mode": r.get("mode", ""),
                              "n_contexts": len(r.get("contexts") or []),
                              **{m: None for m in metric_names}})

    summary = {}
    n_valid = {}
    for m in metric_names:
        vals = [p[m] for p in per_q if p.get(m) is not None]
        summary[m] = round(sum(vals) / len(vals), 3) if vals else None
        n_valid[m] = len(vals)
    return {"summary": summary, "n_valid": n_valid, "per_question": per_q}


def _failed_questions(errors: list) -> list[dict]:
    """从 errors 汇总失败题列表（按 id 归并原因）。"""
    agg: dict = {}
    for e in errors:
        qid = e.get("id")
        if qid is None:
            continue
        key = str(qid)
        reason = e.get("error", "")
        if e.get("metric") and "metric" not in reason:
            reason = f"[{e['metric']}] {reason}"
        agg.setdefault(key, [])
        if reason and reason not in agg[key]:
            agg[key].append(reason)
    return [{"id": k, "reasons": v} for k, v in agg.items()]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    from src import config

    ap = argparse.ArgumentParser(
        description="RAGAS 评估（本地 Ollama qwen2.5:3b 裁判 + bge-m3 embedding，全程离线）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑测试题前 N 题（0=全部），便于冒烟")
    ap.add_argument("--strategy", choices=sorted(config.RETRIEVAL_STRATEGIES),
                    default=config.RETRIEVAL_STRATEGY_DEFAULT,
                    help="检索策略：vector / fulltext / hybrid（默认 hybrid）")
    ap.add_argument("--out", default=_DEFAULT_OUT,
                    help=f"评估结果输出 JSON（默认 {_DEFAULT_OUT}；相对路径相对项目根）")
    ap.add_argument("--metrics", default=_DEFAULT_METRICS,
                    help=f"逗号分隔的指标（可选：{','.join(_ALL_METRICS)}）")
    ap.add_argument("--reuse-answers", action="store_true",
                    help=f"复用 {_STAGE_NAME} 里的回答，不重新问答（只重跑评判）")
    args = ap.parse_args()

    # RAGAS 默认会向外部上报用量；离线环境关掉（须在 import ragas 前设置）
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    errors: list = []
    skipped: list = []

    try:
        metric_names = _parse_metrics(args.metrics)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr, flush=True)
        return 2

    out_file = _resolve_path(args.out)
    t_start = time.time()
    _banner(f"RAGAS 评估 | {config.WORKORDER_ID} | strategy={args.strategy}")

    # --- 评判模型可用性：裁判 LLM 缺失 → 明确报错并非 0 退出 ---
    try:
        judge = _check_judge()
    except RuntimeError as exc:
        print(f"\n[错误] {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"裁判 LLM = {judge['llm']} | 裁判 embedding = "
          f"{judge['embeddings'] or '(缺失)'} | host = {judge['host']}", flush=True)
    if not judge["embeddings"] and "answer_relevancy" in metric_names:
        reason = (f"本机 Ollama 未找到 bge-m3（{config.EMBEDDING_MODEL_NAME}），"
                  f"answer_relevancy 无法计算，已跳过该指标（其余指标照常）")
        print(f"[警告] {reason}", flush=True)
        errors.append({"stage": "setup", "metric": "answer_relevancy", "error": reason})
        metric_names = [m for m in metric_names if m != "answer_relevancy"]
        if not metric_names:
            print("[错误] 没有可执行的指标", file=sys.stderr, flush=True)
            return 1

    # --- 第一步：收集回答（或复用 stage 产物） ---
    try:
        rows, used_strategy = collect_answers(
            args.limit, args.strategy, args.reuse_answers, errors, skipped)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] 回答收集失败：{type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1
    if not rows:
        print("\n[错误] 没有可评估的样本（全部缺金标或问答失败）",
              file=sys.stderr, flush=True)
        return 1
    print(f"样本 {len(rows)} 题 | 跳过（缺金标）{len(skipped)} 题 | "
          f"累计用时 {time.time() - t_start:.1f}s", flush=True)

    # --- 第二步：本地 Ollama 评判 ---
    _banner(f"RAGAS 评判（{len(rows)} 题 × {len(metric_names)} 指标，本地裁判较慢，请耐心）")
    t_eval = time.time()
    try:
        llm_w = _build_judge_llm(judge["llm"])
        emb_w = _build_judge_embeddings(judge["embeddings"]) if judge["embeddings"] else None
    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] 裁判初始化失败：{type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1

    ragas = run_ragas(rows, metric_names, llm_w, emb_w, errors)
    eval_s = round(time.time() - t_eval, 1)

    out = {
        "workorder": config.WORKORDER_ID,
        "timestamp": _now(),
        "strategy": used_strategy,
        "n_samples": len(rows),
        "metrics": metric_names,
        "judge": {"llm": judge["llm"], "embeddings": judge["embeddings"],
                  "host": judge["host"]},
        "summary": ragas["summary"],
        "summary_n_valid": ragas["n_valid"],
        "per_question": ragas["per_question"],
        "skipped_no_gt": skipped,
        "failed_questions": _failed_questions(errors),
        "errors": errors,
        "stage_file": os.path.join(config.EVAL_DIR, _STAGE_NAME),
        "eval_elapsed_s": eval_s,
        "total_elapsed_s": round(time.time() - t_start, 1),
    }
    _write_json(out_file, out)

    _banner(f"完成：{out_file} | 评判 {eval_s}s | 总用时 {out['total_elapsed_s']}s")
    print("指标均值：", json.dumps(ragas["summary"], ensure_ascii=False, indent=1), flush=True)
    print(f"有效样本数：{json.dumps(ragas['n_valid'], ensure_ascii=False)}", flush=True)
    if skipped:
        print(f"跳过（缺金标）：{[s['id'] for s in skipped]}", flush=True)
    if errors:
        print(f"错误/缺失记录 {len(errors)} 条，详见结果 JSON 的 errors 字段", flush=True)

    # 全部指标都无有效值 = 评判整体失败 → 非 0 退出
    if all(v is None for v in ragas["summary"].values()):
        print("[错误] 所有指标均无有效分数，评判整体失败", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    # Windows 主线程栈只有约 1MB：把全部工作放进 16MB 大栈线程执行
    sys.exit(bootstrap.run_with_large_stack(main))
