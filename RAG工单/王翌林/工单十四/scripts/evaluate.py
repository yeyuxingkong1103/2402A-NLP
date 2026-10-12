# -*- coding: utf-8 -*-
"""
工单十四：6题评估脚本（直接调用 RAGEngine，不依赖 API）
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_engine import RAGEngine

QUESTIONS = [
    {"id": 1, "question": "根据文本信息，该静电除尘器的发明人是？", "answer": "A",
     "keywords": ["吉特勒"]},
    {"id": 2, "question": "根据文本信息，以下哪个描述符合该静电除尘器的特征？", "answer": "C",
     "keywords": ["单个圆锥形", "80", "95", "台阶"]},
    {"id": 3, "question": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？", "answer": "A",
     "keywords": ["左侧", "左边", "左方", "左面"]},
    {"id": 4, "question": "在文件中第7页的图片中，尺寸X1，X2，X3分别代表什么部件的间隔距离？", "answer": "A",
     "keywords": ["配气带孔盘", "6"]},
    {"id": 5, "question": "根据文件中第7页图示，气流方向(7)首先经过哪个部件？紧接着会经过哪个部件？", "answer": "C",
     "keywords": ['6"', "6'"]},
    {"id": 6, "question": "根据文件中第7页图示，如果已知外壳直径D，那么h1和h2的尺寸可以用来计算什么？", "answer": "B",
     "keywords": ["配气带孔盘", "位置"]},
]


def evaluate():
    # 初始化引擎
    engine = RAGEngine(top_k=5)
    # 加载页面映射
    mapping_path = "./data/kb_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        engine.set_page_images({int(k): v for k, v in mapping.get("page_images", {}).items()})
        engine.set_page_texts({int(k): v for k, v in mapping.get("page_texts", {}).items()})
        print("知识库映射已加载")

    results = []
    for q in QUESTIONS:
        print(f"\n=== Q{q['id']}: {q['question'][:50]}... ===")
        t0 = time.time()
        try:
            data = engine.ask(q["question"])
            elapsed = time.time() - t0
            answer = data.get("answer", "")
            correct = (q["answer"] in answer) or any(kw in answer for kw in q["keywords"])
            results.append({
                "id": q["id"], "question": q["question"], "answer": answer,
                "correct_answer": q["answer"], "correct": correct,
                "latency_ms": data.get("latency_ms", round(elapsed * 1000, 1)),
                "mode": data.get("mode", ""),
            })
            print(f"  答案: {answer[:200]}")
            print(f"  正确: {'✅' if correct else '❌'} (标准答案: {q['answer']})")
            print(f"  耗时: {data.get('latency_ms', 0):.0f}ms")
        except Exception as e:
            results.append({"id": q["id"], "question": q["question"], "answer": f"ERROR: {e}",
                            "correct_answer": q["answer"], "correct": False, "latency_ms": 0})
            print(f"  错误: {e}")

    correct_count = sum(1 for r in results if r["correct"])
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    print(f"\n{'='*60}")
    print(f"评估结果: {correct_count}/{len(results)} = {correct_count/len(results)*100:.0f}%")
    print(f"平均响应时间: {avg_latency:.0f}ms")
    print(f"{'='*60}")

    out_path = "./docs/v14_evaluation_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "accuracy": correct_count / len(results),
                   "avg_latency_ms": avg_latency}, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")
    return results


if __name__ == "__main__":
    evaluate()
