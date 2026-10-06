# -*- coding: utf-8 -*-
"""
Graph RAG vs RAG 对比测试
工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
"""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)

TEST_QUESTIONS = [
    {"q": "武汉力源信息技术股份有限公司的控股股东是谁,持股比例是多少?",
     "kws": ["武汉力源科技", "35", "控股股东"], "type": "relation"},
    {"q": "武汉兴图新科电子股份有限公司的注册资本是多少?",
     "kws": ["7360", "注册资本"], "type": "fact"},
    {"q": "销售部有几个下属部门,大客户销售部有几个销售处?",
     "kws": ["销售部", "大客户销售部", "销售处"], "type": "structure"},
    {"q": "武汉兴图新科参与制定了什么标准?",
     "kws": ["AVS", "标准"], "type": "fact"},
    {"q": "军用领域收入分别是多少?",
     "kws": ["收入", "军用"], "type": "financial"},
    {"q": "2008年IC市场增长最快和负增长的行业分别是哪些?",
     "kws": ["汽车电子", "消费电子", "增长"], "type": "image"},
]


def run_comparison():
    import qa_engine_v8

    print(f"{'='*60}\nGraph RAG vs RAG 对比测试 ({len(TEST_QUESTIONS)} 题)\n{'='*60}")

    results = []
    for item in TEST_QUESTIONS:
        q = item["q"]
        kws = item["kws"]
        print(f"\n[问题] {q[:40]}...")

        t0 = time.time()
        r = qa_engine_v8.answer_question(q)
        elapsed = time.time() - t0

        answer = r["rag_answer"]
        graph_score = r["graph_result"]["graph_score"]
        center_nodes = r["graph_result"]["center_nodes"]

        cov = sum(1 for k in kws if k in answer) / max(len(kws), 1)

        print(f"  图谱中心: {center_nodes}")
        print(f"  图谱分数: {graph_score}")
        print(f"  关键词覆盖: {cov:.0%} ({sum(1 for k in kws if k in answer)}/{len(kws)})")
        print(f"  响应: {elapsed:.2f}s")

        results.append({
            "question": q, "answer": answer, "type": item["type"],
            "graph_score": graph_score, "center_nodes": center_nodes,
            "coverage": cov, "time": round(elapsed, 3),
        })

    # 汇总
    avg_cov = sum(r["coverage"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"Graph RAG 平均关键词覆盖: {avg_cov:.2%}")
    print(f"Graph RAG 平均响应时间: {avg_time:.3f}s")

    # 按类型
    by_type = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"cov": [], "time": []}
        by_type[t]["cov"].append(r["coverage"])
        by_type[t]["time"].append(r["time"])

    print("\n按类型:")
    for t, v in by_type.items():
        print(f"  {t:10s}: 覆盖 {sum(v['cov'])/len(v['cov']):.0%} | 时间 {sum(v['time'])/len(v['time']):.3f}s")


if __name__ == "__main__":
    run_comparison()
