# -*- coding: utf-8 -*-
"""
多轮对话验收测试脚本
工单编号: 人工智能 NLP-RAG-Query 理解优化任务

测试验收标准的 5 轮多轮对话
"""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)


def test_dialogue():
    import qa_engine_v5
    sid = "acceptance_test"
    qa_engine_v5.new_session(sid)

    # 验收标准的 5 轮多轮对话
    dialogues = [
        {
            "turn": 1,
            "question": "报告期内,武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
            "expect_company": "武汉兴图新科电子股份有限公司",
            "expect_keywords": ["收入", "军用"],
        },
        {
            "turn": 2,
            "question": "他参与的哪个工程荣获了国家科技进步一等奖?",
            "expect_company": "武汉兴图新科电子股份有限公司",  # 指代消解: 他
            "expect_keywords": ["工程", "一等奖"],
        },
        {
            "turn": 3,
            "question": "这个公司的法定代表人是谁?",
            "expect_company": "武汉兴图新科电子股份有限公司",  # 指代消解: 这个公司
            "expect_keywords": ["法定代表人"],
        },
        {
            "turn": 4,
            "question": "那武汉力源信息技术股份有限公司呢?",
            "expect_company": "武汉力源信息技术股份有限公司",  # 公司切换
            "expect_keywords": ["法定代表人"],  # 继承上一轮话题
            "expect_switch": True,
        },
        {
            "turn": 5,
            "question": "武汉力源信息技术股份有限公司组织结构图中,哪个销售部的销售处最多?有哪些销售处?",
            "expect_company": "武汉力源信息技术股份有限公司",
            "expect_keywords": ["销售部", "销售处"],
        },
    ]

    print(f"{'='*60}\nV5 多轮对话验收测试 ({len(dialogues)} 轮)\n{'='*60}")
    results = []

    for d in dialogues:
        print(f"\n[轮次 {d['turn']}] Q: {d['question'][:50]}...")
        t0 = time.time()
        r = qa_engine_v5.answer_question(d["question"], sid)
        rt = time.time() - t0

        di = r["dialogue_info"]
        enriched = r["enriched"]
        company = di.get("current_company", "")
        switched = di.get("company_switched", False)
        answer = r["rag_answer"]

        # 验证
        checks = []
        ok_company = company == d["expect_company"]
        checks.append(("公司识别", ok_company, f"期望:{d['expect_company'][:20]} 实际:{company[:20]}"))

        if d.get("expect_switch"):
            checks.append(("公司切换", switched, f"期望切换=True 实际={switched}"))

        # 关键词覆盖率
        cov_kws = d.get("expect_keywords", [])
        cov_hit = sum(1 for k in cov_kws if k in answer or k in enriched)
        cov = cov_hit / max(len(cov_kws), 1)
        checks.append(("关键词覆盖", cov >= 0.5, f"{cov_hit}/{len(cov_kws)}={cov}"))

        # 指代消解验证
        if d["turn"] >= 2 and d["turn"] <= 3:
            # "他" / "这个公司" 应该被补全
            has_company_in_enriched = d["expect_company"][:6] in enriched
            checks.append(("指代消解", has_company_in_enriched, f"'{enriched[:40]}'"))

        all_ok = all(c[1] for c in checks)
        print(f"  补全: {enriched[:60]}")
        print(f"  公司: {company[:20]} | 切换: {switched} | 时间: {rt:.2f}s")
        print(f"  结果: {'✅ 通过' if all_ok else '❌ 未通过'}")
        for name, ok, detail in checks:
            print(f"    {'✅' if ok else '❌'} {name}: {detail}")
        print(f"  A: {answer[:80]}")

        results.append({
            "turn": d["turn"], "question": d["question"],
            "enriched": enriched, "company": company,
            "switched": switched, "cov": cov,
            "all_ok": all_ok, "response_time": round(rt, 3),
            "rag_answer": answer[:200],
        })

    # 汇总
    passed = sum(1 for r in results if r["all_ok"])
    total_cov = sum(r["cov"] for r in results) / len(results)
    total_time = sum(r["response_time"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"验收结果: {passed}/{len(results)} 轮通过")
    print(f"平均关键词覆盖: {total_cov:.2%}")
    print(f"平均响应时间: {total_time:.3f}s")
    print(f"整体达标: {'✅ 是' if passed >= len(dialogues)-1 and total_cov >= 0.9 else '❌ 否'}")

    # 保存
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "V5验收结果.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {path}")


if __name__ == "__main__":
    test_dialogue()
