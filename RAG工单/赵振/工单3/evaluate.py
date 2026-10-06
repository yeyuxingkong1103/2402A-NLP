"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化。"""

import json
import os
import time
from pathlib import Path

from rag import answer_table_question, build_index, load_index, model, retrieve


ROOT = Path(__file__).parent
PDF = Path(os.getenv("TASK03_PDF2", r"C:\Users\ZhuanZ\Downloads\RAG 工单\RAG 工单\附件\招股说明书2.pdf"))
QUESTIONS = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
LEGACY_RESULTS = json.loads((ROOT / "legacy_evaluation_results.json").read_text(encoding="utf-8"))


def main():
    encoder = model()
    folder = build_index(PDF, encoder)
    index = load_index(folder)
    results = []

    for question in QUESTIONS[:4]:
        started = time.perf_counter()
        info, retrieved = retrieve(question["question"], index, encoder)
        answer = answer_table_question(question["question"], retrieved, index) or "未找到足够的表格证据。"
        seconds = time.perf_counter() - started
        coverage = sum(keyword in answer for keyword in question["keywords"]) / len(question["keywords"])
        record = {
            "id": question["id"], "question": question["question"], "source": question["source"],
            "reference": question["reference"], "final_answer": answer,
            "retrieved_pages": list(dict.fromkeys(item["page"] for item in retrieved)),
            "retrieved_table_groups": list(dict.fromkeys(item.get("table_group") for item in retrieved if item.get("kind") == "table")),
            "keyword_coverage": round(coverage, 3), "request_seconds": round(seconds, 4),
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
        assert coverage == 1.0, f"题目{question['id']}关键词未覆盖完整"
        assert seconds <= 3, f"题目{question['id']}热请求超时：{seconds:.3f}s"

    # 这10题复用已完成的工单2真实模型结果作为回归，不重复伪称本次重新推理。
    for record in LEGACY_RESULTS:
        results.append({
            "id": record["id"], "question": record["question"], "source": "招股说明书1.pdf",
            "final_answer": record["final_answer"], "retrieved_pages": record["retrieved_pages"],
            "keyword_coverage": record["final_keyword_coverage"], "request_seconds": record["request_seconds"],
            "regression_source": "工单2 evaluation_results.json",
        })

    summary = {
        "new_table_questions": 4,
        "new_table_keyword_pass": sum(r["keyword_coverage"] == 1 for r in results[:4]),
        "legacy_regression_questions": 10,
        "legacy_keyword_pass": sum(r["keyword_coverage"] == 1 for r in results[4:]),
        "table_query_max_seconds": max(r["request_seconds"] for r in results[:4]),
        "table_query_under_3_seconds": all(r["request_seconds"] <= 3 for r in results[:4]),
        "legacy_results_reused": True,
        "scope_note": "固定题关键词覆盖是自动评测，不替代人工准确性审查；高并发压力未测。",
    }
    (ROOT / "evaluation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
