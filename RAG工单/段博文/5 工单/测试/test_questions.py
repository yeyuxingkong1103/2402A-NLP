# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""
多轮对话准确率测试脚本（5 题对话链 + 中英文测试）

测试目标：
    1. 验证多轮对话能力（指代消解、主语继承、主语切换）；
    2. 验证 Query 改写正确率；
    3. 验证整体答案关键词命中率；
    4. 验证响应时间 ≤3秒；
    5. 支持中英文问答。

测试用例（按工单要求）：
    Q1: 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
    Q2: 他参与的哪个工程荣获了国家科技进步一等奖？
    Q3: 这个公司的法定代表人是谁？
    Q4: 那武汉力源信息技术股份有限公司呢？
    Q5: 武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？
    Q6 (英文): What was the revenue of Wuhan Xingtu Xinke Electronics from military sector?
"""

import json
import os
import time
import uuid
import sys
from typing import Dict, List

import requests

BASE = "http://127.0.0.1:8000"

# 5 题多轮对话链 + 1 题英文测试
DIALOG_CHAIN = [
    {
        "q": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "expect_rewrite": False,  # 首轮无需改写
        "expect_entity": "武汉兴图新科电子股份有限公司",
        "keywords": ["军用", "收入", "万元"],
        "desc": "首轮：完整独立问题",
    },
    {
        "q": "他参与的哪个工程荣获了国家科技进步一等奖？",
        "expect_rewrite": True,
        "expect_rewrite_contains": ["武汉兴图新科", "工程", "国家科技进步一等奖"],
        "keywords": ["工程", "国家科技进步一等奖"],
        "desc": "第二论：指代消解（他→武汉兴图新科）",
    },
    {
        "q": "这个公司的法定代表人是谁？",
        "expect_rewrite": True,
        "expect_rewrite_contains": ["武汉兴图新科", "法定代表人"],
        "keywords": ["法定代表人"],
        "desc": "第三论：主语继承（这个公司→武汉兴图新科）",
    },
    {
        "q": "那武汉力源信息技术股份有限公司呢？",
        "expect_rewrite": True,
        "expect_rewrite_contains": ["武汉力源信息技术", "法定代表人"],
        "keywords": ["法定代表人"],
        "desc": "第四论：主语切换（继承上一轮谓语，替换主语为武汉力源）",
    },
    {
        "q": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
        "expect_rewrite": False,
        "keywords": ["销售部", "销售处"],
        "desc": "第五论：完整独立问题（图像语义）",
    },
]

ENGLISH_TEST = {
    "q": "Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
    "keywords": ["legal representative", "法定代表人"],
    "desc": "英文问题测试",
}


def sep(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_health():
    try:
        r = requests.get(f"{BASE}/api/health", timeout=5)
        if r.ok:
            d = r.json()
            print(f"服务状态：{d.get('status')}, 集合块数：{d.get('collection_count')}")
            print(f"功能列表：{d.get('features')}")
            return True
    except Exception as e:
        print(f"健康检查失败：{e}")
    return False


def run_dialog_chain() -> Dict:
    """跑 5 题多轮对话链。"""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    results = []
    total_time = 0.0
    rewrite_correct = 0
    rewrite_total = 0

    sep(f"多轮对话测试 (session_id={session_id})")

    for i, item in enumerate(DIALOG_CHAIN, 1):
        q = item["q"]
        print(f"\n--- Q{i}: {item['desc']} ---")
        print(f"原问题: {q}")

        t0 = time.time()
        try:
            r = requests.post(
                f"{BASE}/api/chat",
                json={"query": q, "session_id": session_id, "use_rewrite": True, "stream": False},
                timeout=60,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            if not r.ok:
                print(f"  [FAIL] HTTP {r.status_code}: {r.text[:200]}")
                results.append({
                    "idx": i, "query": q, "ok": False, "error": r.text[:200],
                    "elapsed": elapsed,
                })
                continue

            data = r.json()
            answer = data.get("answer", "")
            rewritten = data.get("rewritten_query", q)
            query_type = data.get("query_type", "")
            rewritten_by = data.get("rewritten_by", "")

            print(f"改写后: {rewritten}")
            print(f"改写方式: {rewritten_by} (类型: {query_type})")
            print(f"耗时: {elapsed:.2f}s (rewrite={data['timing']['rewrite']}s, "
                  f"search={data['timing']['search']}s, llm={data['timing']['llm']}s)")
            print(f"答案: {answer[:200]}{'...' if len(answer) > 200 else ''}")

            # 1. 验证改写正确性
            rewrite_ok = True
            if item.get("expect_rewrite"):
                rewrite_total += 1
                if rewritten == q:
                    print(f"  [WARN] 期望改写但未改写")
                    rewrite_ok = False
                else:
                    for kw in item.get("expect_rewrite_contains", []):
                        if kw not in rewritten:
                            print(f"  [FAIL] 改写结果缺少关键词 '{kw}': {rewritten}")
                            rewrite_ok = False
                    if rewrite_ok:
                        rewrite_correct += 1
                        print(f"  [OK] 改写正确")

            # 2. 验证答案关键词
            hit_kws = [kw for kw in item["keywords"] if kw in answer]
            kw_ok = len(hit_kws) == len(item["keywords"])
            print(f"  关键词命中: {len(hit_kws)}/{len(item['keywords'])} -> {hit_kws}")

            # 3. 验证响应时间
            time_ok = elapsed <= 3.0
            if not time_ok:
                print(f"  [WARN] 响应超时 {elapsed:.2f}s > 3s")

            results.append({
                "idx": i,
                "query": q,
                "rewritten": rewritten,
                "rewritten_by": rewritten_by,
                "query_type": query_type,
                "answer": answer,
                "rewrite_ok": rewrite_ok,
                "keywords_hit": hit_kws,
                "keywords_total": len(item["keywords"]),
                "elapsed": elapsed,
                "time_ok": time_ok,
                "ok": kw_ok and rewrite_ok,
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"idx": i, "query": q, "ok": False, "error": str(e)})

    # ===== 英文测试 =====
    sep("英文问答测试")
    en = ENGLISH_TEST
    print(f"Q: {en['q']}")
    try:
        t0 = time.time()
        r = requests.post(
            f"{BASE}/api/chat",
            json={"query": en["q"], "stream": False},
            timeout=60,
        )
        elapsed = time.time() - t0
        if r.ok:
            data = r.json()
            answer = data.get("answer", "")
            print(f"答案: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            hit = [kw for kw in en["keywords"] if kw.lower() in answer.lower()]
            print(f"关键词命中: {len(hit)}/{len(en['keywords'])} -> {hit}")
            results.append({
                "idx": "EN",
                "query": en["q"],
                "answer": answer,
                "keywords_hit": hit,
                "keywords_total": len(en["keywords"]),
                "elapsed": elapsed,
                "ok": len(hit) > 0,
            })
        else:
            print(f"  [FAIL] HTTP {r.status_code}")
            results.append({"idx": "EN", "query": en["q"], "ok": False, "error": r.text[:200]})
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append({"idx": "EN", "query": en["q"], "ok": False, "error": str(e)})

    # ===== 汇总 =====
    sep("测试汇总")
    main_results = [r for r in results if isinstance(r.get("idx"), int)]
    pass_count = sum(1 for r in main_results if r.get("ok"))
    total = len(main_results)
    accuracy = pass_count / total * 100 if total > 0 else 0

    rewrite_acc = rewrite_correct / rewrite_total * 100 if rewrite_total > 0 else 0

    total_kws_hit = sum(len(r.get("keywords_hit", [])) for r in main_results)
    total_kws = sum(r.get("keywords_total", 0) for r in main_results)
    kw_acc = total_kws_hit / total_kws * 100 if total_kws > 0 else 0

    avg_time = sum(r.get("elapsed", 0) for r in main_results) / total if total > 0 else 0
    p95_time = sorted([r.get("elapsed", 0) for r in main_results])
    p95_time = p95_time[int(len(p95_time) * 0.95)] if p95_time else 0

    print(f"\n对话链题数: {pass_count}/{total} 通过 ({accuracy:.1f}%)")
    print(f"改写正确率: {rewrite_correct}/{rewrite_total} ({rewrite_acc:.1f}%)")
    print(f"关键词命中率: {total_kws_hit}/{total_kws} ({kw_acc:.1f}%)")
    print(f"平均响应时间: {avg_time:.2f}s, P95: {p95_time:.2f}s")
    print(f"3秒内响应: {sum(1 for r in main_results if r.get('elapsed', 99) <= 3.0)}/{total}")

    return {
        "session_id": session_id,
        "total_questions": total,
        "passed": pass_count,
        "accuracy": round(accuracy, 2),
        "rewrite_accuracy": round(rewrite_acc, 2),
        "keyword_accuracy": round(kw_acc, 2),
        "avg_response_time": round(avg_time, 2),
        "p95_response_time": round(p95_time, 2),
        "results": results,
    }


def main():
    sep("工单五：Query理解优化任务 - 多轮对话测试")
    if not check_health():
        print("服务未就绪，退出")
        sys.exit(1)

    summary = run_dialog_chain()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n测试结果已保存：{out_path}")

    # markdown 报告
    md_path = os.path.join(out_dir, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 工单五：Query 理解优化任务 - 多轮对话测试报告\n\n")
        f.write(f"- 会话ID: `{summary['session_id']}`\n")
        f.write(f"- 通过题数: **{summary['passed']}/{summary['total_questions']}** ({summary['accuracy']}%)\n")
        f.write(f"- 改写正确率: **{summary['rewrite_accuracy']}%**\n")
        f.write(f"- 关键词命中率: **{summary['keyword_accuracy']}%**\n")
        f.write(f"- 平均响应时间: **{summary['avg_response_time']}s**\n")
        f.write(f"- P95 响应时间: **{summary['p95_response_time']}s**\n\n")
        f.write("## 详细结果\n\n")
        for r in summary["results"]:
            idx = r.get("idx")
            f.write(f"### Q{idx}: {r.get('query', '')[:50]}\n\n")
            if r.get("error"):
                f.write(f"**ERROR**: {r['error']}\n\n")
                continue
            f.write(f"- 改写后: `{r.get('rewritten', '')}`\n")
            f.write(f"- 改写方式: `{r.get('rewritten_by', '')}` / 类型 `{r.get('query_type', '')}`\n")
            f.write(f"- 耗时: {r.get('elapsed', 0):.2f}s\n")
            f.write(f"- 关键词命中: {len(r.get('keywords_hit', []))}/{r.get('keywords_total', 0)}\n")
            f.write(f"\n**答案**:\n\n```\n{r.get('answer', '')[:500]}\n```\n\n")
    print(f"测试报告已保存：{md_path}")


if __name__ == "__main__":
    main()
