"""工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化。"""

import json
import time

from rag import ROOT, answer_verified, build_index, load_index, model, retrieve


def main():
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    pdf_path = ROOT / "data" / "uploads" / "招股说明书1.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError("请先将工单附件招股说明书1.pdf放到 data/uploads/")

    embedding_model = model()
    index = load_index(build_index(pdf_path, embedding_model))
    rows = []
    for item in questions:
        started = time.perf_counter()
        _, evidence = retrieve(item["question"], index, embedding_model)
        answer, answer_seconds, _ = answer_verified(item["question"], evidence, "deepseek-r1:1.5b")
        clean = answer.replace(",", "").replace("，", "").replace(" ", "")
        score = sum(word.replace(",", "").replace(" ", "") in clean for word in item["keywords"]) / len(item["keywords"])
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "reference": item["reference"],
            "retrieved_pages": [chunk["page"] for chunk in evidence],
            "answer": answer,
            "keyword_coverage": score,
            "retrieval_seconds": round(time.perf_counter() - started - answer_seconds, 4),
            "request_seconds": round(time.perf_counter() - started, 4),
        })
        print(f"{item['id']}: 覆盖率 {score:.0%}，热请求 {rows[-1]['request_seconds']:.3f} 秒", flush=True)

    output = ROOT / "evaluation_results.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()
