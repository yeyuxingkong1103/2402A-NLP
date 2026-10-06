# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
工单二 - 10道工单问题对比测试（优化前 vs 优化后）

作用：
    对工单要求的 10 个问题，在优化版服务上跑 RAG 问答，并与工单一基线
    results.json 做前后对比，生成：
      tests/results.json   机器可读（含 before/after）
      tests/results.md     人工可读的对照表（含耗时/关键词命中前后对比）

用法：
    cd tests
    python test_questions.py
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx

# ============================================================
# 配置
# ============================================================
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TOP_K = 3  # 优化版默认重排保留 3 条
TIMEOUT_RAG = 120.0

TESTS_DIR = Path(__file__).resolve().parent
RESULTS_JSON = TESTS_DIR / "results.json"
RESULTS_MD = TESTS_DIR / "results.md"
# 工单一基线结果（用于前后对比）
BASELINE_JSON = Path(r"D:\作业\6-专高NLP 作业\成品\工单一\tests\results.json")

# ============================================================
# 10 道工单测试问题（与工单一完全一致）
# ============================================================
QUESTIONS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
    "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
    "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
]

EXPECTED_KEYWORDS = {
    1: ["军用", "收入"],
    2: ["标准", "GB", "JT", "技术"],
    3: ["军用", "比重", "主营"],
    4: ["上游", "电子", "元器件", "芯片"],
    5: ["供应商", "重要"],
    6: ["下游", "行业", "应用"],
    7: ["工程", "国家科技进步", "一等奖"],
    8: ["注册资本", "万元"],
    9: ["法定代表人", "姓名"],
    10: ["募集资金", "补充流动资金", "%"],
}


def load_dotenv():
    env_path = TESTS_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def load_baseline():
    """加载工单一基线 results.json。"""
    if not BASELINE_JSON.exists():
        print(f"[警告] 基线文件不存在：{BASELINE_JSON}")
        return {}
    data = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline = {}
    for r in data.get("results", []):
        baseline[r["idx"]] = {
            "elapsed_sec": r["rag"]["elapsed_sec"],
            "keyword_hit": r["rag"]["keyword_hit"],
            "keyword_total": r["rag"]["keyword_total"],
            "error": r["rag"]["error"],
        }
    return baseline


def call_rag(client, question):
    """调用优化版 /api/chat，返回 (answer, sources, timing, elapsed, error)。"""
    t0 = time.time()
    try:
        resp = client.post(
            f"{API_BASE}/api/chat",
            json={"query": question, "top_k": TOP_K, "stream": False},
            timeout=TIMEOUT_RAG,
        )
    except Exception as e:
        return "", [], {}, time.time() - t0, f"请求失败：{e}"

    if resp.status_code != 200:
        return "", [], {}, time.time() - t0, f"HTTP {resp.status_code}: {resp.text[:300]}"

    try:
        data = resp.json()
        return (
            data.get("answer", ""),
            data.get("references", []) or [],
            data.get("timing", {}),
            round(time.time() - t0, 3),
            None,
        )
    except Exception as e:
        return "", [], {}, time.time() - t0, f"JSON解析失败：{e}"


def keyword_hit(answer, keywords):
    if not answer:
        return 0, []
    hit = [kw for kw in keywords if kw in answer]
    return len(hit), hit


def main():
    load_dotenv()
    baseline = load_baseline()

    print("=" * 70)
    print("  工单二 - 10道工单问题对比测试（优化前 vs 优化后）")
    print(f"  优化版后端：{API_BASE}  top_k={TOP_K}")
    print(f"  基线来源：{BASELINE_JSON}")
    print("=" * 70)

    results = []
    # 先跑一遍预热（让模型/索引加载完成）
    print("[预热] 正在加载模型与索引 ...")
    with httpx.Client() as client:
        try:
            client.get(f"{API_BASE}/api/health", timeout=30)
        except Exception:
            pass

    with httpx.Client() as client:
        for idx, q in enumerate(QUESTIONS, 1):
            print(f"\n[{idx}/10] {q}")
            ans, src, timing, elapsed, err = call_rag(client, q)
            kws = EXPECTED_KEYWORDS.get(idx, [])
            kw_n, kw_list = keyword_hit(ans, kws)

            bl = baseline.get(idx, {})
            bl_t = bl.get("elapsed_sec", "-")
            bl_kw = f"{bl.get('keyword_hit',0)}/{bl.get('keyword_total',0)}"

            status = "OK" if not err else "ERR"
            cached = timing.get("cached", False) if timing else False
            print(f"  [{status}] 优化后 {elapsed}s (cached={cached}) | 基线 {bl_t}s | 关键词 {kw_n}/{len(kws)} vs 基线 {bl_kw}")

            results.append({
                "idx": idx,
                "question": q,
                "expected_keywords": kws,
                "after": {
                    "answer": ans,
                    "sources": src,
                    "timing": timing,
                    "elapsed_sec": elapsed,
                    "error": err,
                    "keyword_hit": kw_n,
                    "keyword_hit_list": kw_list,
                    "keyword_total": len(kws),
                },
                "before": bl,
            })

    # ---- 写 results.json ----
    RESULTS_JSON.write_text(
        json.dumps({"api_base": API_BASE, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写出：{RESULTS_JSON}")

    # ---- 写 results.md ----
    md = render_md(results, baseline)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(f"已写出：{RESULTS_MD}")

    # ---- 汇总 ----
    after_times = [r["after"]["elapsed_sec"] for r in results if not r["after"]["error"]]
    before_times = [r["before"]["elapsed_sec"] for r in results if r["before"].get("elapsed_sec")]
    after_kw = sum(r["after"]["keyword_hit"] for r in results)
    after_kw_total = sum(r["after"]["keyword_total"] for r in results)
    before_kw = sum(r["before"].get("keyword_hit", 0) for r in results)
    before_kw_total = sum(r["before"].get("keyword_total", 0) for r in results)

    print("\n" + "=" * 70)
    if after_times:
        print(f"  优化后平均响应：{sum(after_times)/len(after_times):.2f}s（范围 {min(after_times):.2f}-{max(after_times):.2f}s）")
    if before_times:
        print(f"  优化前平均响应：{sum(before_times)/len(before_times):.2f}s（范围 {min(before_times):.2f}-{max(before_times):.2f}s）")
    print(f"  关键词命中：优化后 {after_kw}/{after_kw_total} vs 优化前 {before_kw}/{before_kw_total}")
    print("=" * 70)


def render_md(results, baseline):
    lines = []
    lines.append("# 工单二 - 10道工单问题对比测试（优化前 vs 优化后）\n")
    lines.append(f"- 优化版后端：`{API_BASE}`（top_k={TOP_K}）")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 汇总表
    after_times = [r["after"]["elapsed_sec"] for r in results if not r["after"]["error"]]
    before_times = [r["before"]["elapsed_sec"] for r in results if r["before"].get("elapsed_sec")]
    after_kw = sum(r["after"]["keyword_hit"] for r in results)
    after_kw_total = sum(r["after"]["keyword_total"] for r in results)
    before_kw = sum(r["before"].get("keyword_hit", 0) for r in results)
    before_kw_total = sum(r["before"].get("keyword_total", 0) for r in results)

    lines.append("## 汇总\n")
    lines.append("| 指标 | 优化前（工单一） | 优化后（工单二） | 提升 |")
    lines.append("|---|---|---|---|")
    if after_times and before_times:
        avg_b = sum(before_times) / len(before_times)
        avg_a = sum(after_times) / len(after_times)
        speedup = (1 - avg_a / avg_b) * 100 if avg_b > 0 else 0
        lines.append(f"| 平均响应时间 | {avg_b:.2f}s | {avg_a:.2f}s | 提升 {speedup:.1f}% |")
        lines.append(f"| 最快响应 | {min(before_times):.2f}s | {min(after_times):.2f}s | - |")
        lines.append(f"| 最慢响应 | {max(before_times):.2f}s | {max(after_times):.2f}s | - |")
    lines.append(f"| 关键词命中 | {before_kw}/{before_kw_total} | {after_kw}/{after_kw_total} | - |")
    lines.append("")

    # 逐题对比
    lines.append("## 逐题对比\n")
    lines.append("| # | 问题(前30字) | 优化前耗时 | 优化后耗时 | 优化前关键词 | 优化后关键词 | 缓存 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        q = r["question"][:30] + ("…" if len(r["question"]) > 30 else "")
        bl_t = r["before"].get("elapsed_sec", "-")
        af_t = r["after"]["elapsed_sec"] if not r["after"]["error"] else "ERR"
        bl_kw = f"{r['before'].get('keyword_hit',0)}/{r['before'].get('keyword_total',0)}"
        af_kw = f"{r['after']['keyword_hit']}/{r['after']['keyword_total']}" if not r["after"]["error"] else "-"
        cached = "是" if r["after"].get("timing", {}).get("cached") else "否"
        lines.append(f"| {r['idx']} | {q} | {bl_t}s | {af_t}s | {bl_kw} | {af_kw} | {cached} |")
    lines.append("")

    # 详细回答
    lines.append("## 详细回答（优化后）\n")
    for r in results:
        lines.append(f"### Q{r['idx']}：{r['question']}\n")
        lines.append(f"**期望关键词**：{', '.join(r['expected_keywords'])}\n")
        af = r["after"]
        if af["error"]:
            lines.append(f"- ❌ 错误：`{af['error']}`\n")
        else:
            lines.append(f"- ⏱ 耗时：{af['elapsed_sec']}s（检索 {af['timing'].get('search_sec','-')}s + LLM {af['timing'].get('llm_sec','-')}s）")
            lines.append(f"- 🔑 关键词命中：{af['keyword_hit']}/{af['keyword_total']}（{', '.join(af['keyword_hit_list']) or '无'}）")
            lines.append(f"- 📄 引用来源数：{len(af['sources'])}")
            lines.append(f"- 🚀 缓存命中：{'是' if af['timing'].get('cached') else '否'}\n")
            lines.append("```text")
            lines.append(af["answer"] or "（空）")
            lines.append("```\n")
        lines.append("---\n")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
