# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化任务
模块：RAG vs LightRAG 对比报告
"""

import json
import os


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rag_results = load_json("rag_results.json")
    lr_results = load_json("lightrag_results.json")

    if not rag_results:
        print("❌ rag_results.json 不存在")
        return
    if not lr_results:
        print("⚠️ lightrag_results.json 不存在（LightRAG 还在跑）")

    # 用 id 索引
    rag_map = {r["id"]: r for r in rag_results}
    lr_map = {r["id"]: r for r in lr_results}

    all_ids = sorted(set(rag_map.keys()) | set(lr_map.keys()))

    print("=" * 100)
    print("RAG vs LightRAG 对比")
    print("=" * 100)

    print(f"\n{'ID':<6} {'RAG':<15} {'LightRAG':<15} {'胜出方':<10}")
    print("-" * 100)

    results = []
    for qid in all_ids:
        r = rag_map.get(qid, {})
        l = lr_map.get(qid, {})

        rag_time = r.get("rag_time", "-")
        lr_time = l.get("lightrag_time", "-")

        # 简单胜出判定（响应时间）
        if isinstance(rag_time, (int, float)) and isinstance(lr_time, (int, float)):
            winner = "RAG" if rag_time < lr_time else "LightRAG"
        else:
            winner = "-"

        print(f"{qid:<6} {str(rag_time):<15} {str(lr_time):<15} {winner:<10}")

        results.append({
            "id": qid,
            "question": r.get("question", l.get("question", "")),
            "rag_answer": r.get("rag_answer", ""),
            "rag_time": rag_time,
            "lightrag_answer": l.get("lightrag_answer", ""),
            "lightrag_time": lr_time,
            "winner": winner,
        })

    # 保存
    with open("rag_lightrag_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 对比结果已保存：rag_lightrag_compare.json")

    # 统计
    rag_wins = sum(1 for r in results if r["winner"] == "RAG")
    lr_wins = sum(1 for r in results if r["winner"] == "LightRAG")
    print(f"\nRAG 胜出：{rag_wins} 题")
    print(f"LightRAG 胜出：{lr_wins} 题")


if __name__ == "__main__":
    main()
