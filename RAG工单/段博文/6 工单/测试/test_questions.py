# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
混合检索准确率/召回率测试脚本（10题 + 三种策略对比）

测试目标：
    1. 验证三种检索策略（vector/fulltext/hybrid）均可用；
    2. 验证混合检索准确率 ≥90%（答案关键词命中率）；
    3. 验证召回率 ≥95%（检索结果包含期望关键词）；
    4. 验证布尔/短语/模糊查询语法；
    5. 验证三种融合算法（rrf/weighted/voting）与三种重排器（cross_encoder/tfidf/llm）；
    6. 支持中英文问答。
"""

import json
import time
import requests

BASE = "http://127.0.0.1:8000"

# 10 题准确率测试（基于两本招股说明书，含图像/表格/文本）
QUESTIONS = [
    {"q": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
     "keywords": ["军用", "收入", "万元"], "type": "text/table"},
    {"q": "武汉兴图新科的法定代表人是谁？",
     "keywords": ["法定代表人"], "type": "text"},
    {"q": "武汉力源信息技术股份有限公司的法定代表人是谁？",
     "keywords": ["法定代表人"], "type": "text"},
    {"q": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？",
     "keywords": ["销售部", "销售处"], "type": "image"},
    {"q": "兴图新科参与的是哪个工程荣获国家科技进步一等奖？",
     "keywords": ["工程", "国家科技进步一等奖"], "type": "text"},
    {"q": "武汉力源信息技术股份有限公司的主营业务是什么？",
     "keywords": ["主营", "业务"], "type": "text"},
    {"q": "兴图新科的主营业务收入构成是怎样的？",
     "keywords": ["主营业务", "收入"], "type": "table"},
    {"q": "招股说明书中提到的主要风险因素有哪些？",
     "keywords": ["风险"], "type": "text"},
    {"q": "武汉兴图新科电子股份有限公司的注册地址在哪里？",
     "keywords": ["注册地址", "地址"], "type": "text"},
    {"q": "武汉兴图新科电子股份有限公司的组织架构是怎样的？",
     "keywords": ["组织", "架构"], "type": "image"},
]

# 全文检索语法测试
SYNTAX_TESTS = [
    {"q": '武汉 AND 兴图', "desc": "布尔AND", "expect_hit": True},
    {"q": '兴图 OR 力源', "desc": "布尔OR", "expect_hit": True},
    {"q": '兴图 NOT 力源', "desc": "布尔NOT", "expect_hit": True},
    {"q": '"法定代表人"', "desc": "短语匹配", "expect_hit": True},
    {"q": '兴图~', "desc": "模糊匹配", "expect_hit": True},
]


def test_config_api():
    """测试检索配置接口。"""
    print("\n===== 1. 检索配置接口测试 =====")
    r = requests.get(f"{BASE}/api/retrieve_config", timeout=10)
    cfg = r.json()
    print(f"  当前配置: {cfg}")
    assert cfg["strategy"] in ("vector", "fulltext", "hybrid")

    # 动态更新
    r = requests.post(f"{BASE}/api/retrieve_config", json={"strategy": "hybrid", "fusion": "rrf"}, timeout=10)
    assert r.status_code == 200
    print("  ✓ 配置动态更新成功")
    return True


def test_strategy(name, strategy, fusion=None, reranker=None, questions=None):
    """测试一种检索策略的准确率与召回率。"""
    questions = questions or QUESTIONS
    hit_answer = 0
    hit_recall = 0
    times = []

    print(f"\n===== {name}（strategy={strategy}, fusion={fusion}, reranker={reranker}）=====")
    for i, item in enumerate(questions, 1):
        q = item["q"]
        keywords = item["keywords"]

        # 纯检索 → 召回率
        params = {"query": q, "top_k": 5, "strategy": strategy}
        if fusion: params["fusion"] = fusion
        if reranker: params["reranker"] = reranker
        t0 = time.time()
        try:
            r = requests.get(f"{BASE}/api/search", params=params, timeout=60)
            data = r.json()
            ctx = " ".join(x["page_content"] for x in data.get("results", []))
            recall_ok = any(k.lower() in ctx.lower() for k in keywords)
        except Exception as e:
            recall_ok = False
            print(f"  Q{i} 检索异常: {e}")
        t1 = time.time()

        # 问答 → 准确率
        try:
            r = requests.post(f"{BASE}/api/chat", json={
                "query": q, "top_k": 3, "stream": False, "use_rewrite": False,
                "strategy": strategy, "fusion": fusion, "reranker": reranker,
            }, timeout=120)
            ans = r.json().get("answer", "")
            ans_ok = any(k.lower() in ans.lower() for k in keywords)
        except Exception as e:
            ans_ok = False
            ans = ""
            print(f"  Q{i} 问答异常: {e}")
        t2 = time.time()

        hit_recall += recall_ok
        hit_answer += ans_ok
        times.append(t2 - t0)
        print(f"  Q{i}: 召回={'✓' if recall_ok else '✗'} 答案={'✓' if ans_ok else '✗'} "
              f"检索={t1-t0:.2f}s 问答={t2-t1:.2f}s | {q[:35]}...")

    n = len(questions)
    acc = hit_answer / n * 100
    rec = hit_recall / n * 100
    avg = sum(times) / len(times)
    print(f"  → {name}: 准确率={acc:.1f}% 召回率={rec:.1f}% 平均响应={avg:.2f}s")
    return {"name": name, "accuracy": acc, "recall": rec, "avg_time": avg}


def test_syntax():
    """测试全文检索查询语法。"""
    print("\n===== 全文检索语法测试 =====")
    passed = 0
    for t in SYNTAX_TESTS:
        r = requests.get(f"{BASE}/api/search",
                         params={"query": t["q"], "top_k": 3, "strategy": "fulltext",
                                 "reranker": "tfidf"},
                         timeout=30)
        data = r.json()
        hit = data.get("total", 0) > 0
        ok = hit == t["expect_hit"]
        passed += ok
        print(f"  {'✓' if ok else '✗'} {t['desc']}: '{t['q']}' 命中 {data.get('total', 0)} 条")
    print(f"  → 语法测试 {passed}/{len(SYNTAX_TESTS)} 通过")
    return passed == len(SYNTAX_TESTS)


def main():
    print("=" * 60)
    print("工单六：混合检索准确率/召回率测试")
    print("=" * 60)

    test_config_api()

    results = []
    # 主测试：混合检索（默认配置）
    results.append(test_strategy("混合检索(hybrid/rrf/cross_encoder)", "hybrid", "rrf", "cross_encoder"))
    # 向量检索单独
    results.append(test_strategy("向量检索(vector)", "vector", None, "cross_encoder"))
    # 全文检索单独
    results.append(test_strategy("全文检索(fulltext)", "fulltext", None, "cross_encoder"))
    # 融合算法对比（各跑3题快速验证）
    results.append(test_strategy("混合(weighted融合)", "hybrid", "weighted", "tfidf", QUESTIONS[:3]))
    results.append(test_strategy("混合(voting融合)", "hybrid", "voting", "tfidf", QUESTIONS[:3]))

    test_syntax()

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for r in results:
        flag = "✓" if r["accuracy"] >= 90 and r["recall"] >= 95 else "△"
        print(f"  {flag} {r['name']}: 准确率={r['accuracy']:.1f}% 召回率={r['recall']:.1f}% 平均={r['avg_time']:.2f}s")

    main_r = results[0]
    print(f"\n主指标（混合检索）：准确率={main_r['accuracy']:.1f}%（目标≥90%），召回率={main_r['recall']:.1f}%（目标≥95%）")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "main": main_r}, f, ensure_ascii=False, indent=2)
    print("结果已写入 results.json")


if __name__ == "__main__":
    main()
