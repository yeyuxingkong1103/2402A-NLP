# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""
工单四 - 16道工单问题测试（6力源题〈含2图像题〉 + 10兴图题）

作用：
    对工单要求的 16 个问题测试 RAG 问答效果，
    生成 tests/results.json 和 tests/results.md。

用法：
    cd tests
    python test_questions.py
"""

import json
import os
import time
from pathlib import Path

import httpx

# ============================================================
# 配置
# ============================================================
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TOP_K = 3
TIMEOUT_RAG = 120.0

TESTS_DIR = Path(__file__).resolve().parent
RESULTS_JSON = TESTS_DIR / "results.json"
RESULTS_MD = TESTS_DIR / "results.md"

# ============================================================
# 16 道工单测试问题（6 力源题〈含 2 图像题〉 + 10 兴图题）
# ============================================================
QUESTIONS = [
    # --- 力源信息（招股说明书2.pdf）---
    "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
    "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
    "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
    "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
    # 图像题1：组织结构图（第39页，纯矢量图）
    "武汉力源信息技术股份有限公司意向书组织结构图中，销售部由几个部门构成，其中大客户销售部由几个销售处构成？",
    # 图像题2：IC市场应用结构与增长图（第72页，内嵌位图）
    "武汉力源信息技术股份有限公司意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
    # --- 兴图新科（招股说明书1.pdf）---
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
    1: ["1670", "万股", "25.04"],
    2: ["仓储", "物流中心", "研发中心", "电子商务", "募集资金"],
    3: ["赵马克", "42.35", "控股股东"],
    4: ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源"],
    5: ["4个", "6个", "电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部"],
    6: ["汽车", "14.0", "-2.0"],
    7: ["军用", "收入"],
    8: ["标准", "GB", "JT", "技术"],
    9: ["军用", "比重", "主营"],
    10: ["上游", "电子", "元器件", "芯片"],
    11: ["供应商", "重要"],
    12: ["下游", "行业", "应用"],
    13: ["工程", "国家科技进步", "一等奖"],
    14: ["注册资本", "万元"],
    15: ["法定代表人", "姓名"],
    16: ["募集资金", "补充流动资金", "%"],
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


def call_rag(client, question):
    """调用 /api/chat，返回 (answer, sources, timing, elapsed, error)。"""
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

    print("=" * 70)
    print("  工单四 - 16道工单问题测试（6力源题〈含2图像题〉 + 10兴图题）")
    print(f"  后端：{API_BASE}  top_k={TOP_K}")
    print("=" * 70)

    results = []
    # 预热
    print("[预热] 检查服务状态 ...")
    with httpx.Client() as client:
        try:
            resp = client.get(f"{API_BASE}/api/health", timeout=30)
            print(f"  服务状态：{resp.json()}")
        except Exception as e:
            print(f"  [错误] 服务未启动：{e}")
            return

    with httpx.Client() as client:
        for idx, q in enumerate(QUESTIONS, 1):
            print(f"\n[{idx}/16] {q}")
            ans, src, timing, elapsed, err = call_rag(client, q)
            kws = EXPECTED_KEYWORDS.get(idx, [])
            kw_n, kw_list = keyword_hit(ans, kws)

            status = "OK" if not err else "ERR"
            has_table = any(s.get("block_type") == "table" for s in src)
            has_image = any(s.get("block_type") == "image" for s in src)
            print(f"  [{status}] {elapsed}s | 关键词 {kw_n}/{len(kws)} | 表格:{has_table} 图像:{has_image}")

            results.append({
                "idx": idx,
                "question": q,
                "expected_keywords": kws,
                "answer": ans,
                "sources": src,
                "timing": timing,
                "elapsed_sec": elapsed,
                "error": err,
                "keyword_hit": kw_n,
                "keyword_hit_list": kw_list,
                "keyword_total": len(kws),
                "has_table_source": has_table,
                "has_image_source": has_image,
            })

    # ---- 写 results.json ----
    RESULTS_JSON.write_text(
        json.dumps({"api_base": API_BASE, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写出：{RESULTS_JSON}")

    # ---- 写 results.md ----
    md = render_md(results)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(f"已写出：{RESULTS_MD}")

    # ---- 汇总 ----
    times = [r["elapsed_sec"] for r in results if not r["error"]]
    kw = sum(r["keyword_hit"] for r in results)
    kw_total = sum(r["keyword_total"] for r in results)
    image_hits = sum(1 for r in results if r.get("has_image_source"))

    print("\n" + "=" * 70)
    if times:
        print(f"  平均响应：{sum(times)/len(times):.2f}s（范围 {min(times):.2f}-{max(times):.2f}s）")
    print(f"  关键词命中：{kw}/{kw_total}（{kw/kw_total*100:.1f}%）")
    print(f"  图像来源命中：{image_hits}/16")
    print("=" * 70)


def render_md(results):
    lines = []
    lines.append("# 工单四 - 16道工单问题测试（图像内容解析增强版）\n")
    lines.append(f"- 后端：`{API_BASE}`（top_k={TOP_K}）")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 汇总
    times = [r["elapsed_sec"] for r in results if not r["error"]]
    kw = sum(r["keyword_hit"] for r in results)
    kw_total = sum(r["keyword_total"] for r in results)
    image_hits = sum(1 for r in results if r.get("has_image_source"))

    lines.append("## 汇总\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    if times:
        lines.append(f"| 平均响应时间 | {sum(times)/len(times):.2f}s |")
        lines.append(f"| 最快响应 | {min(times):.2f}s |")
        lines.append(f"| 最慢响应 | {max(times):.2f}s |")
    lines.append(f"| 关键词命中 | {kw}/{kw_total}（{kw/kw_total*100:.1f}%） |")
    lines.append(f"| 图像来源命中 | {image_hits}/16 |")
    lines.append("")

    # 逐题对比
    lines.append("## 逐题结果\n")
    lines.append("| # | 问题(前25字) | 耗时 | 关键词 | 表格 | 图像 |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        q = r["question"][:25] + ("…" if len(r["question"]) > 25 else "")
        t = r["elapsed_sec"] if not r["error"] else "ERR"
        kw_s = f"{r['keyword_hit']}/{r['keyword_total']}"
        table = "是" if r.get("has_table_source") else "否"
        image = "是" if r.get("has_image_source") else "否"
        lines.append(f"| {r['idx']} | {q} | {t}s | {kw_s} | {table} | {image} |")
    lines.append("")

    # 详细回答
    lines.append("## 详细回答\n")
    for r in results:
        lines.append(f"### Q{r['idx']}：{r['question']}\n")
        lines.append(f"**期望关键词**：{', '.join(r['expected_keywords'])}\n")
        lines.append(f"**耗时**：{r['elapsed_sec']}s | **关键词命中**：{r['keyword_hit']}/{r['keyword_total']}\n")
        lines.append(f"**回答**：\n{r['answer'][:600]}\n")
        if r.get("sources"):
            lines.append("**引用来源**：")
            for s in r["sources"][:3]:
                bt = s.get("block_type", "text")
                lines.append(f"- [{bt}] {s.get('source','')} 第{s.get('page_number',0)}页 (score={s.get('rerank_score',0):.3f})")
            lines.append("")
        lines.append("---\n")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
