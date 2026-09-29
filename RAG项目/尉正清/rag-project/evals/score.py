# -*- coding: utf-8 -*-
"""用 RAGAS 对采集结果打分。

必须用**评测专用 venv** 运行（ragas 依赖 langchain-core 0.3，与项目环境的 1.x 不兼容）：

    evals/.venv/bin/python evals/score.py --tag baseline
    evals/.venv/bin/python evals/score.py --tag baseline --limit 5      # 先小样本试跑

四个指标：
    faithfulness       回答是否忠于检索到的资料（查幻觉）
    answer_relevancy   回答是否切题
    context_precision  检索到的资料有多少是真正有用的
    context_recall     标准答案所需的信息，检索是否都覆盖到了
"""
import argparse
import json
import os
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型与判分配置
JUDGE_BASE_URL = "https://api.deepseek.com/v1"
JUDGE_MODEL = "deepseek-flash"
# deepseek-flash 是推理模型，判分要产 JSON，token 给小了正文会空
JUDGE_MAX_TOKENS = 4096
JUDGE_TIMEOUT = 240
EMBED_MODEL = "/root/models/bge-m3"


def load_api_key():
    """优先环境变量，其次项目 .env。"""
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(EVAL_DIR), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("未找到 DEEPSEEK_API_KEY")


def load_results(tag, limit=0):
    path = os.path.join(EVAL_DIR, "results_%s.jsonl" % tag)
    if not os.path.exists(path):
        raise FileNotFoundError("结果文件不存在: %s\n请先运行 collect.py" % path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                print("  跳过（采集失败）: %s" % r["id"])
                continue
            if not r.get("answer") or not r.get("contexts"):
                print("  跳过（无回答或无检索结果）: %s" % r["id"])
                continue
            rows.append(r)
    return rows[:limit] if limit else rows


def build_judge():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=load_api_key(),
        base_url=JUDGE_BASE_URL,
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
        timeout=JUDGE_TIMEOUT,
        max_retries=2,
    )


def build_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # 复用知识库同一个 BGE-M3，保证判分用的向量空间与检索一致
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="判分并发数")
    args = ap.parse_args()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)
    from ragas.run_config import RunConfig

    rows = load_results(args.tag, args.limit)
    if not rows:
        print("没有可评测的样本")
        return 1

    print("评测标记: %s" % args.tag)
    print("有效样本: %d" % len(rows))
    print("判分模型: %s（并发 %d）" % (JUDGE_MODEL, args.workers))
    print()

    ds = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config = RunConfig(max_workers=args.workers, timeout=JUDGE_TIMEOUT * 2)

    t0 = time.time()
    print("开始判分（这一步较慢，deepseek-flash 是推理模型）...")
    result = evaluate(ds, metrics=metrics,
                      llm=build_judge(), embeddings=build_embeddings(),
                      run_config=run_config, raise_exceptions=False)
    elapsed = time.time() - t0

    df = result.to_pandas()
    metric_cols = [c for c in df.columns
                   if c in ("faithfulness", "answer_relevancy",
                            "context_precision", "context_recall")]

    # 明细落盘，便于逐题回看
    detail_path = os.path.join(EVAL_DIR, "scores_%s_detail.csv" % args.tag)
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    # 汇总：整体 + 分角色
    summary = {
        "tag": args.tag,
        "samples": len(rows),
        "elapsed_sec": round(elapsed, 1),
        "judge_model": JUDGE_MODEL,
        "overall": {c: round(float(df[c].mean()), 4) for c in metric_cols},
        "per_role": {},
    }
    roles = [r["role_key"] for r in rows]
    df["_role"] = roles
    for role, sub in df.groupby("_role"):
        summary["per_role"][role] = {
            "samples": len(sub),
            **{c: round(float(sub[c].mean()), 4) for c in metric_cols},
        }

    summary_path = os.path.join(EVAL_DIR, "scores_%s.json" % args.tag)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RAGAS 评测结果 [%s]" % args.tag)
    print("=" * 60)
    for c in metric_cols:
        print("  %-20s %.4f" % (c, summary["overall"][c]))
    print("\n分角色:")
    for role, s in summary["per_role"].items():
        print("  %-20s n=%d  " % (role, s["samples"]) +
              "  ".join("%s=%.3f" % (c.split("_")[0][:5], s[c]) for c in metric_cols))
    print("\n耗时 %.1f 分钟" % (elapsed / 60))
    print("汇总: %s" % summary_path)
    print("明细: %s" % detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
