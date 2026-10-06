# -*- coding: utf-8 -*-
"""
一键运行完整评估脚本
工单编号: 人工智能 NLP-RAG-功能测试及评估
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)


def main():
    import qa_engine_v7
    result = qa_engine_v7.run_full_evaluation()

    print(f"\n{'='*60}")
    print("📊 评估结果汇总")
    print(f"{'='*60}")

    rm = result["retrieval_metrics"]
    qm = result["qa_metrics"]
    pm = result["perf_metrics"]

    print(f"\n🔍 检索评估:")
    print(f"  Precision@5: {rm['precision@k']} (目标 ≥ 0.7) {'✅' if rm['precision@k'] >= 0.7 else '❌'}")
    print(f"  Recall@5:    {rm['recall@k']} (目标 ≥ 0.95) {'✅' if rm['recall@k'] >= 0.95 else '❌'}")
    print(f"  MRR:         {rm['mrr']} (目标 ≥ 0.8) {'✅' if rm['mrr'] >= 0.8 else '❌'}")
    print(f"  NDCG@5:      {rm['ndcg@k']} (目标 ≥ 0.85) {'✅' if rm['ndcg@k'] >= 0.85 else '❌'}")
    print(f"  HitRate@5:   {rm['hit_rate@k']} (目标 ≥ 0.95) {'✅' if rm['hit_rate@k'] >= 0.95 else '❌'}")

    print(f"\n💬 问答评估:")
    print(f"  关键词覆盖率: {qm['keyword_coverage']} (目标 ≥ 0.9) {'✅' if qm['keyword_coverage'] >= 0.9 else '❌'}")
    print(f"  BLEU-1:      {qm['bleu_1']}")
    print(f"  ROUGE-L:     {qm['rouge_l']}")
    print(f"  幻觉比例:    {qm['hallucination_ratio']} (目标 ≤ 0.05) {'✅' if qm['hallucination_ratio'] <= 0.05 else '❌'}")

    print(f"\n⚡ 性能评估:")
    print(f"  平均响应:   {pm['avg_time']}s")
    print(f"  P95 响应:   {pm['p95_time']}s")
    print(f"  3秒达标率:  {pm['within_3s_ratio']} ({pm['within_3s']}/{pm['count']}) {'✅' if pm['within_3s_ratio'] >= 0.95 else '❌'}")

    print(f"\n⚠️  问题分析: {len(result['problem_analysis'])} 个问题")
    for pa in result["problem_analysis"]:
        print(f"  Q{pa['id']}: {pa['issues']}")
        print(f"    原因: {pa['reason']}")
        print(f"    建议: {pa['suggestion']}")

    print(f"\n📄 完整报告已生成:")
    print(f"  Markdown: 测试报告.md")
    print(f"  JSON: 评估结果.json")

    # 总体达标判断
    all_ok = (rm['precision@k'] >= 0.7 and rm['recall@k'] >= 0.95 and
              qm['keyword_coverage'] >= 0.9 and pm['within_3s_ratio'] >= 0.95)
    print(f"\n{'='*60}")
    print(f"🎯 总体达标: {'✅ 是' if all_ok else '❌ 否'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
