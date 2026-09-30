# -*- coding: utf-8 -*-
"""RAGAS 评测：faithfulness / answer_relevancy / context_precision / context_recall。

与自研评测（`run_eval.py`）的关系
---------------------------------
**并列呈现，不互相取代**：

    run_eval.py   —— 业务视角：来源命中率、答案覆盖率、引用标注率、拒答正确性
    run_ragas.py  —— 学术标准：RAGAS 四指标（基于 LLM 裁判打分）

两者用的是**同一套 43 题 QA 集**，结果可直接对照。

⚠️ 本脚本必须在**任何会拉起 torch 的导入之前**调用 ensure_ragas_compat()
-----------------------------------------------------------------------
ragas 0.4.3 与本环境有两处不兼容（详见 `app/core/ragas_compat.py`）：

1. ragas 模块顶层硬导入 `langchain_community.chat_models.vertexai`，
   而 `langchain-community 0.4.2` 已删除该模块；
2. **更凶险**：本环境 `torch` 与 `pyarrow` 有原生 DLL 加载顺序冲突 ——
   若 torch 先加载，随后 `import ragas` 会**直接段错误（SIGSEGV）**，
   **不是 Python 异常，try/except 兜不住**。

因此下面的导入顺序是**硬性约定**，改动前请先读 `ragas_compat.py` 的 docstring。

关于裁判模型与被测模型相同（自我偏好偏差）
------------------------------------------
本脚本用 `qwen2.5:7b` 既当**被测模型**（后端生成回答）又当**裁判模型**（RAGAS 打分）。
这会引入 **self-preference 偏差** —— 模型倾向给自己的输出打高分，分数可能系统性偏高。

**因此 RAGAS 分数不可当作绝对结论**，必须与 `run_eval.py` 的自研指标对照看。
报告中已明确标注该局限。

用法
----
    D:\\anaconda3\\envs\\rag_env\\python.exe eval\\run_ragas.py
    D:\\anaconda3\\envs\\rag_env\\python.exe eval\\run_ragas.py --limit 3   # 快速冒烟
"""
# ================================================================
# ⚠️ 顺序不能动：垫片必须在最前，早于任何会拉起 torch 的导入
# ================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ragas_compat import ensure_ragas_compat  # noqa: E402

ensure_ragas_compat()

# 此后才可以安全导入 ragas 与其它模块
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import secrets  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import httpx  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

EVAL_DIR = Path(__file__).resolve().parent

BASE = os.environ.get("RAGLORA_API_BASE", "http://127.0.0.1:8000/api")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# ChatOllama 要裸地址（不带 /v1），OllamaEmbeddings 同理
OLLAMA_HOST = OLLAMA_URL.removesuffix("/v1")
JUDGE_MODEL = os.environ.get("RAGLORA_JUDGE_MODEL", "qwen2.5:7b")
# 嵌入用 Ollama 上的 bge-m3（HTTP 调用）—— **刻意不在评测进程里加载 torch 版 bge-m3**：
# 那会多占约 2.3GB 内存，且让本进程也踩上 torch/pyarrow 的段错误风险。
EMBED_MODEL = os.environ.get("RAGLORA_JUDGE_EMBED", "bge-m3:latest")

ROLE_MAP = {"doctor": "medical", "lawyer": "legal"}

# 每题取多少条 context 参与评测（与后端精排条数一致）
CONTEXT_K = 5

# ⚠️ 并发必须先降下来（2026-09-17 实测踩坑）
# ------------------------------------------------------------
# ragas 的 `RunConfig.max_workers` **默认是 16**，而本机只有一张 8G 显卡跑单个 7B 模型
# （Ollama 对单模型默认串行）。16 路并发会把请求堆在队列里 → 撞 180s 超时 →
# 重试 10 次仍失败 → **该题该指标记为 NaN**，且 `raise_exceptions=False` 让它
# **静默消失**。
#
# 实测后果：首轮 40 题里 `context_precision` **只有 3 题算出值**（37 题 NaN），
# 而脚本把 3 个样本的均值报成了 **1.0000** —— 一个看起来满分、实则无意义的数字。
# 其余三个指标没暴露，是因为 `context_precision` 要对**每个上下文**单独裁判
# （5 条上下文 = 5 倍调用量），最先被并发压垮。
#
# 降到 2 路 + 放宽超时后，同一批失败样本重跑**全部成功**（实测 63s / 5 题）。
MAX_WORKERS = int(os.environ.get("RAGLORA_RAGAS_WORKERS", "2"))
JUDGE_TIMEOUT = 600

# 某指标有效样本占比低于此值时，报告里明确标记为「不可信」，不当作结论
MIN_VALID_RATIO = 0.8


def load_qa() -> dict:
    return json.loads((EVAL_DIR / "qa_set.json").read_text(encoding="utf-8"))


def collect_samples(client: httpx.Client, headers: dict, chars: dict,
                    qa: dict, limit: int) -> tuple[list[SingleTurnSample], list[dict]]:
    """跑一遍后端，收集 RAGAS 需要的 (问题, 回答, 检索上下文, 参考答案)。

    context 从 `/api/search` 取（**返回完整 text**），而不是从 chat 响应的
    `sources` 取 —— 后者只保留前 300 字（`persona.build_sources` 的截断），
    不足以支撑 context_precision / context_recall。
    """
    samples: list[SingleTurnSample] = []
    meta: list[dict] = []

    # ROLE_MAP 是「角色 slug -> 评测集键」（与 run_eval.py 保持一致）
    for slug, role in ROLE_MAP.items():
        if slug not in chars:
            print(f"  跳过 {role}：角色 {slug} 不存在")
            continue
        items = qa[role]
        if limit:
            items = items[:limit]

        print(f"\n▶ {role}（角色 {slug}，{len(items)} 题）")
        print("-" * 70)

        for i, item in enumerate(items, 1):
            q = item["q"]
            # ⚠️ 每题独立会话 —— 与 run_eval.py 同一约定。
            # 早期版本一个会话连问多题，上下文堆积会把模型风格带偏、不再标注引用。
            conv = client.post(f"{BASE}/conversations", headers=headers,
                               json={"character_id": chars[slug]["id"]}).json()
            t0 = time.time()
            try:
                res = client.post(f"{BASE}/chat/completions", headers=headers, json={
                    "conversation_id": conv["id"], "question": q,
                }).json()
            except Exception as e:
                print(f"  {i:>2}. !! 对话失败: {e}")
                continue
            answer = res.get("answer", "")

            # 取完整上下文（与对话同一问题、同一集合、同样精排）
            try:
                s = client.post(f"{BASE}/search", headers=headers, json={
                    "query": q, "collection": chars[slug]["kb_collection"],
                    "top_k": CONTEXT_K, "use_rerank": True,
                }).json()
                contexts = [h.get("text", "") for h in s.get("hits", []) if h.get("text")]
            except Exception as e:
                print(f"  {i:>2}. !! 检索失败: {e}")
                contexts = []

            # 参考答案：由 QA 集的期望关键词拼成。
            # ⚠️ 这不是人工撰写的标准答案，因此 context_precision / context_recall
            #    只能视为**指示性**结果，不能当精确值 —— 报告里已标注。
            kws = item.get("expect_keywords", [])
            reference = "回答应包含以下要点：" + "、".join(kws) if kws else ""

            samples.append(SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference,
            ))
            meta.append({
                "role": role, "q": q, "answer": answer,
                "n_contexts": len(contexts),
                "latency": time.time() - t0,
            })
            print(f"  {i:>2}. {len(contexts)} 条上下文  {time.time()-t0:>5.1f}s  {q[:36]}")

    return samples, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每个角色只跑前 N 条（0=全部）")
    ap.add_argument("--report", default=str(EVAL_DIR / "ragas_report.md"))
    ap.add_argument("--json", default=str(EVAL_DIR / "ragas_result.json"))
    args = ap.parse_args()

    qa = load_qa()
    client = httpx.Client(timeout=600)

    uname = f"ragas_{int(time.time())}"
    pw = secrets.token_hex(8)   # 临时用户密码每次随机生成，不写死
    r = client.post(f"{BASE}/auth/register",
                    json={"username": uname, "password": pw})
    if r.status_code != 200:
        r = client.post(f"{BASE}/auth/login",
                        json={"username": uname, "password": pw})
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    chars = {c["slug"]: c for c in client.get(f"{BASE}/characters", headers=headers).json()}

    print("=" * 78)
    print(f"RAGAS 评测  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"裁判模型：{JUDGE_MODEL}　嵌入模型：{EMBED_MODEL}")
    print("=" * 78)

    samples, meta = collect_samples(client, headers, chars, qa, args.limit)
    if not samples:
        print("\n没有收集到任何样本，退出。")
        return 1

    print()
    print("=" * 78)
    print(f"开始 RAGAS 打分（{len(samples)} 题 × 4 指标，需 LLM 裁判，较慢）...")
    print("=" * 78)

    judge_llm = LangchainLLMWrapper(ChatOllama(
        model=JUDGE_MODEL, base_url=OLLAMA_HOST, temperature=0.0))
    judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model=EMBED_MODEL, base_url=OLLAMA_HOST))

    dataset = EvaluationDataset(samples=samples)
    print(f"并发 max_workers={MAX_WORKERS}，超时 {JUDGE_TIMEOUT}s")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=JUDGE_TIMEOUT),
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    metric_cols = [c for c in
                   ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                   if c in df.columns]

    # ---- 汇总 ----
    # ⚠️ 必须同时统计**有效样本数**。踩过的坑：首轮 context_precision 只有 3/40
    #    题算出值，脚本却把 3 个样本的均值报成 1.0000 —— 看起来满分，实则无意义。
    #    因此这里对有效率不足的指标明确降级为「不可信」。
    summary = {}
    valid_ratio = {}
    for col in metric_cols:
        vals = [float(v) for v in df[col].tolist() if v == v]   # 过滤 NaN
        n_valid = len(vals)
        valid_ratio[col] = n_valid / len(df) if len(df) else 0.0
        summary[col] = statistics.mean(vals) if vals else float("nan")

    print()
    print("=" * 78)
    print("RAGAS 结果")
    print("=" * 78)
    unreliable = []
    for col in metric_cols:
        n = sum(1 for v in df[col].tolist() if v == v)
        flag = ""
        if valid_ratio[col] < MIN_VALID_RATIO:
            flag = "  ⚠️ 有效率过低，不可信"
            unreliable.append(col)
        print(f"  {col:<20} {summary[col]:.4f}   （有效 {n}/{len(df)} 题）{flag}")

    if unreliable:
        print()
        print("!" * 78)
        print("⚠️ 以下指标有效率不足 %.0f%%，其得分**不能当作结论**：" % (MIN_VALID_RATIO * 100))
        for col in unreliable:
            print(f"     {col}  —— 仅 {valid_ratio[col]*100:.0f}% 的题算出值")
        print("   常见原因：裁判模型并发过高压垮 Ollama 导致超时。")
        print("   处置：调低 RAGLORA_RAGAS_WORKERS 后重跑。")
        print("!" * 78)

    # ---- 落盘 ----
    Path(args.json).write_text(
        json.dumps({"summary": summary, "valid_ratio": valid_ratio,
                    "max_workers": MAX_WORKERS,
                    "rows": df.to_dict(orient="records")},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    lines = [
        "# RAGAS 评测报告",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"> 样本量：{len(samples)} 题　评测用户：{uname}",
        f"> 裁判模型：`{JUDGE_MODEL}`　嵌入模型：`{EMBED_MODEL}`",
        "",
        "## ⚠️ 必须先读：本报告的三个局限",
        "",
        "1. **自我偏好偏差**：被测模型与裁判模型**同为 `qwen2.5:7b`**。",
        "   模型倾向给自己的输出打高分，**分数可能系统性偏高**。",
        "   因此 RAGAS 分数**不可当作绝对结论**，必须与自研指标（`eval_report.md`）对照看。",
        "2. **参考答案是关键词拼成的，不是人工撰写的标准答案**。",
        "   因此 `context_precision` / `context_recall` 只能视为**指示性**结果。",
        "3. **拒答类题目未纳入**：RAGAS 四指标均以「有可检索上下文」为前提，",
        "   而拒答题的正确答案恰恰是不引用任何上下文，指标不适用。",
        "   拒答能力请看 `eval_report.md` 的「拒答正确性」。",
        "",
        f"> 裁判并发：max_workers={MAX_WORKERS}（默认 16 会压垮单卡 7B，见下）",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 得分 | 有效样本 | 含义 |",
        "|---|---:|---:|---|",
    ]
    meaning = {
        "faithfulness": "回答中的论断能否被检索到的上下文支撑（越低越可能编造）",
        "answer_relevancy": "回答与问题的相关程度",
        "context_precision": "检索到的上下文里，相关内容排得是否靠前",
        "context_recall": "参考答案的要点能否被检索到的上下文覆盖",
    }
    for col in metric_cols:
        n_valid = round(valid_ratio[col] * len(df))
        mark = "" if valid_ratio[col] >= MIN_VALID_RATIO else " ⚠️"
        lines.append(f"| `{col}` | {summary[col]:.4f}{mark} | {n_valid}/{len(df)} | "
                     f"{meaning.get(col, '')} |")

    if unreliable:
        lines += [
            "",
            f"> ⚠️ **`{'` / `'.join(unreliable)}` 的有效样本不足 "
            f"{MIN_VALID_RATIO*100:.0f}%，其得分不可当作结论。**",
            "> 常见原因是裁判模型并发过高压垮 Ollama 导致超时（已置为 NaN）。",
            f"> 本次运行 max_workers={MAX_WORKERS}。",
        ]

    lines += ["", "## 逐题明细", "",
              "| # | 角色 | 问题 | " + " | ".join(f"`{c}`" for c in metric_cols) + " | 上下文数 |",
              "|---:|---|---|" + "---:|" * len(metric_cols) + "---:|"]
    for i, (row, m) in enumerate(zip(df.to_dict(orient="records"), meta), 1):
        cells = []
        for c in metric_cols:
            v = row.get(c)
            cells.append("—" if v is None or v != v else f"{float(v):.2f}")
        lines.append(f"| {i} | {m['role']} | {m['q'][:40]} | " +
                     " | ".join(cells) + f" | {m['n_contexts']} |")

    lines += ["", "## 与自研指标的对照", "",
              "| 维度 | RAGAS | 自研（run_eval.py） |", "|---|---|---|",
              "| 回答是否忠于检索内容 | `faithfulness` | 引用标注率 |",
              "| 回答是否切题 | `answer_relevancy` | 答案覆盖率 |",
              "| 检索是否精准 | `context_precision` | 来源命中率 |",
              "| 检索是否全面 | `context_recall` | — |",
              "| 边界是否守住 | 不适用 | 拒答正确性 |",
              "",
              "> 两套指标角度不同：RAGAS 用 LLM 裁判做语义判断，自研用关键词做确定性判断。",
              "> **结论一致时可信度高；分歧处值得单独看题。**"]

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {args.report}")
    print(f"原始数据已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
