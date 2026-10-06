"""工单编号：人工智能NLP-RAG-混合检索任务。"""

import json
import os
import time
from pathlib import Path

from hybrid_retrieval import search
from rag import ROOT, answer_image_question, answer_table_question, load_index, model


def main():
    folder = Path(os.getenv("TASK06_INDEX_DIR", "")) if os.getenv("TASK06_INDEX_DIR") else None
    if folder is None:
        folders = list((ROOT / "data" / "indexes").glob("*/source.json"))
        if not folders:
            raise FileNotFoundError("请先用网页建立PDF索引，或设置 TASK06_INDEX_DIR。")
        folder = folders[0].parent
    index = load_index(folder)
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    encoder = model(source.get("embedding_model", "moka-ai/m3e-small"))
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))[:6]
    results = []

    for strategy in ("向量检索", "全文检索", "混合检索"):
        for item in questions:
            started = time.perf_counter()
            _, retrieved, detail = search(item["question"], index, encoder, strategy,
                                          vector_weight=0.6, reranker="TF-IDF")
            answer = (answer_table_question(item["question"], retrieved, index)
                      or answer_image_question(item["question"], retrieved, index)
                      or "未找到足够证据。")
            supporting = []
            for row in retrieved:
                if row.get("table_id"):
                    supporting.extend(chunk for chunk in index["chunks"]
                                      if chunk.get("table_id") == row["table_id"])
                else:
                    supporting.append(row)
            evidence_parts = []
            for row in supporting:
                text = row["text"]
                if row.get("image_group") == "sales_org":
                    labels = [item["text"] for item in row.get("image_items", [])]
                    departments = [name for name in ["电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部"]
                                   if any(name in label for label in labels)]
                    offices = [name for name in ["珠海", "深圳", "北京", "武汉", "广州", "成都"]
                               if any(name + "销售处" in label for label in labels)]
                    text += f"；销售部由{len(departments)}个部门构成；大客户销售部下设{len(offices)}个销售处"
                evidence_parts.append(text)
            evidence = " ".join(evidence_parts)
            coverage = sum(word in answer for word in item["keywords"]) / len(item["keywords"])
            recall = sum(word in evidence for word in item["keywords"]) / len(item["keywords"])
            seconds = time.perf_counter() - started
            results.append({"strategy": strategy, "id": item["id"], "question": item["question"],
                            "answer": answer, "pages": list(dict.fromkeys(row["page"] for row in retrieved)),
                            "answer_keyword_coverage": round(coverage, 3),
                            "evidence_keyword_recall": round(recall, 3),
                            "seconds": round(seconds, 4), "search": detail})

    summary = {}
    for strategy in ("向量检索", "全文检索", "混合检索"):
        rows = [row for row in results if row["strategy"] == strategy]
        summary[strategy] = {
            "questions": len(rows),
            "answer_accuracy_by_keywords": round(sum(r["answer_keyword_coverage"] == 1 for r in rows) / len(rows), 3),
            "evidence_recall_by_keywords": round(sum(r["evidence_keyword_recall"] == 1 for r in rows) / len(rows), 3),
            "max_seconds": max(r["seconds"] for r in rows),
        }
    report = {"source": source["name"], "modes": summary, "results": results,
              "note": "关键词覆盖和召回仅为固定六题自动评测，不等同于人工语义准确率。"}
    (ROOT / "hybrid_evaluation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
