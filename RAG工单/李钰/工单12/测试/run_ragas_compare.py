# -*- coding: utf-8 -*-
"""
15 题 LightRAG vs RAG 对比 + RAGAS 评估脚本
工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化

用法: python run_ragas_compare.py
"""
import os, sys, json, logging

_dev_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "研发")
sys.path.insert(0, _dev_dir)
logging.basicConfig(level=logging.INFO)

from qa_engine_v12 import RAGvsLightRAGEngine, GROUND_TRUTHS


def main():
    print("=" * 70)
    print("  LightRAG V12 vs 传统 RAG  -  RAGAS 风格评估")
    print("  工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化")
    print("=" * 70)

    engine = RAGvsLightRAGEngine()

    # 全量对比
    results = engine.compare_all()

    # 打印表格
    print(f"\n{'ID':<5} {'RAG 分数':>10} {'LightRAG':>10} {'提升':>8} {'备注':<30}")
    print("-" * 70)

    for r in results:
        rag_s = r["rag"]["score"]["score"] * 100
        light_s = r["light_rag"]["score"]["score"] * 100
        imp = (light_s - rag_s)
        icon = "✓" if imp >= 0 else "✗"

        # 判断 LightRAG 的双层检索是否生效
        local_kws = r["light_rag"]["local_kws"]
        global_kws = r["light_rag"]["global_kws"]
        mode = "局部" if local_kws and not global_kws else (
            "全局" if global_kws and not local_kws else "混合"
        )

        print(f"{r['id']:<5} {rag_s:>9.1f}% {light_s:>9.1f}% "
              f"{imp:>+7.1f}% {icon} {mode}")

    # 汇总
    rag_scores = [r["rag"]["score"]["score"] for r in results]
    light_scores = [r["light_rag"]["score"]["score"] for r in results]
    rag_avg = sum(rag_scores) / len(rag_scores)
    light_avg = sum(light_scores) / len(light_scores)
    improvement = (light_avg - rag_avg) / rag_avg * 100

    print("-" * 70)
    print(f"{'平均':<5} {rag_avg*100:>9.1f}% {light_avg*100:>9.1f}% "
          f"{(light_avg-rag_avg)*100:>+7.1f}%")
    print(f"\n  LightRAG 平均提升: +{improvement:.1f}%")
    print(f"  验收结论: {'✅ 通过' if improvement > 0 else '❌ LightRAG 未优于 RAG'}")

    # 详细指标
    print(f"\n--- RAGAS 分项指标 ---")
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevance"]
    print(f"{'指标':<20} {'RAG':>8} {'LightRAG':>8} {'提升':>8}")
    print("-" * 46)
    for m in metrics:
        rag_m = sum(r["rag"]["score"][m] for r in results) / len(results)
        light_m = sum(r["light_rag"]["score"][m] for r in results) / len(results)
        imp_m = (light_m - rag_m) * 100
        print(f"{m:<20} {rag_m*100:>7.1f}% {light_m*100:>7.1f}% {imp_m:>+6.1f}%")

    # 保存 JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_results.json")
    output = {
        "summary": {
            "num_questions": len(results),
            "rag_avg": round(rag_avg, 4),
            "lightrag_avg": round(light_avg, 4),
            "improvement_pct": round(improvement, 2),
        },
        "results": [
            {
                "id": r["id"],
                "question": r["question"],
                "rag_score": r["rag"]["score"],
                "lightrag_score": r["light_rag"]["score"],
                "improvement": r["improvement"],
                "light_rag_local_kws": r["light_rag"]["local_kws"],
                "light_rag_global_kws": r["light_rag"]["global_kws"],
            }
            for r in results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  详细数据已保存: {out_path}")

    return output


if __name__ == "__main__":
    main()
