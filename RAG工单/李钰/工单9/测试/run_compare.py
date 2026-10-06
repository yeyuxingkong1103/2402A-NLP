# -*- coding: utf-8 -*-
"""
V8 vs V9 对比测试脚本
工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
"""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)


TEST_SUITE = [
    {"question": "武汉力源信息技术股份有限公司的控股股东是谁,持股比例是多少?",
     "ref_keywords": ["武汉力源科技", "35", "控股股东"]},
    {"question": "武汉兴图新科电子股份有限公司的注册资本是多少?",
     "ref_keywords": ["7360", "注册资本"]},
    {"question": "销售部有几个下属部门,大客户销售部有几个销售处?",
     "ref_keywords": ["销售部", "大客户销售部", "销售处"]},
    {"question": "武汉兴图新科参与制定了什么标准?",
     "ref_keywords": ["AVS", "标准", "参与制定"]},
    {"question": "武汉兴图新科的法定代表人是谁?",
     "ref_keywords": ["法定代表人"]},
    {"question": "2008年IC市场增长最快和负增长的行业分别是哪些?",
     "ref_keywords": ["汽车电子", "消费电子", "增长"]},
    {"question": "军用领域收入分别是多少?",
     "ref_keywords": ["军用", "收入"]},
    {"question": "募集资金拟投资哪些项目?",
     "ref_keywords": ["募集", "ADC", "接口", "射频"]},
]


def main():
    import qa_engine_v9

    print(f"{'='*70}\nGraph RAG V8 vs V9 对比测试 ({len(TEST_SUITE)} 题)\n{'='*70}")

    result = qa_engine_v9.run_v8_v9_comparison(TEST_SUITE)

    v8 = result["v8"]["summary"]
    v9 = result["v9"]["summary"]

    print(f"\n--- 汇总对比 ---")
    print(f"{'指标':<20} {'V8':<12} {'V9':<12} {'Δ':<12} {'目标'}")
    print(f"-" * 70)
    for metric, target in [("context_precision", 0.80), ("context_recall", 0.90),
                            ("context_f1", None), ("avg_faithfulness", None)]:
        v = v8.get(metric, 0)
        n = v9.get(metric, 0)
        delta = n - v
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        tgt_str = f"≥ {target}" if target else "-"
        print(f"{metric:<20} {v:<12.4f} {n:<12.4f} {arrow} {delta:+.4f}    {tgt_str}")

    # 验收判断
    p_ok = v9["avg_context_precision"] >= 0.80
    r_ok = v9["avg_context_recall"] >= 0.90
    print(f"\n🎯 Context Precision ≥ 0.80: {'✅' if p_ok else '❌'} ({v9['avg_context_precision']})")
    print(f"🎯 Context Recall ≥ 0.90:   {'✅' if r_ok else '❌'} ({v9['avg_context_recall']})")

    # 逐题
    print(f"\n--- 逐题 ---")
    print(f"{'#':<3} {'问题(截断)':<35} {'V8 P':<8} {'V9 P':<8} {'V8 R':<8} {'V9 R':<8}")
    for i, (v8r, v9r) in enumerate(zip(result["v8"]["results"], result["v9"]["results"])):
        q = v9r["question"][:32]
        print(f"{i+1:<3} {q:<35} {v8r['context_precision']:<8.3f} {v9r['context_precision']:<8.3f} {v8r['context_recall']:<8.3f} {v9r['context_recall']:<8.3f}")

    # 保存
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "V8vsV9_对比结果.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
