"""人工智能NLP-RAG-基于PDF文档的问答系统：十题评估。"""

import argparse
import json
import os
import time
from pathlib import Path

from rag import ROOT, answer, answer_verified, build_index, load_index, model, retrieve


LOCAL_PDF = ROOT / "data" / "uploads" / "招股说明书1.pdf"
DEFAULT_PDF = LOCAL_PDF if LOCAL_PDF.exists() else Path(r"C:\Users\ZhuanZ\Downloads\RAG 工单\RAG 工单\附件\招股说明书1.pdf")


def score(text, keywords):
    clean = text.replace(",", "").replace("，", "").replace(" ", "")
    return sum(word.replace(",", "") in clean for word in keywords) / len(keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-r1:1.5b"))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--retrieval-only", action="store_true", help="只评估检索，不调用 LLM")
    args = parser.parse_args()

    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    embedding_model = model()
    folder = build_index(args.pdf, embedding_model)
    index = load_index(folder)
    rows = []
    for item in questions:
        start = time.perf_counter()
        _, results = retrieve(item["question"], index, embedding_model)
        retrieval_time = time.perf_counter() - start
        evidence = "\n".join(r["text"] for r in results)
        row = {
            "id": item["id"], "question": item["question"], "reference": item["reference"],
            "retrieved_pages": [r["page"] for r in results],
            "reference_page_recalled": item["page"] in [r["page"] for r in results],
            "evidence_coverage": score(evidence, item["keywords"]),
            "retrieval_seconds": round(retrieval_time, 3),
        }
        if not args.retrieval_only:
            try:
                final_answer, rag_seconds, rag_answer = answer_verified(item["question"], results, args.model, args.base_url)
                llm_answer, llm_seconds = answer(item["question"], [], args.model, args.base_url, use_context=False)
                row["rag_answer"] = rag_answer
                row["final_answer"] = final_answer
                row["llm_only_answer"] = llm_answer
                row["rag_generation_seconds"] = round(rag_seconds, 3)
                row["llm_only_seconds"] = round(llm_seconds, 3)
                row["rag_keyword_score"] = score(rag_answer, item["keywords"])
                row["final_keyword_score"] = score(final_answer, item["keywords"])
                row["llm_only_keyword_score"] = score(llm_answer, item["keywords"])
            except Exception as exc:
                row["api_error"] = str(exc)
        row["total_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)
        print(f"{row['id']}: 页码 {row['retrieved_pages']}，证据覆盖 {row['evidence_coverage']:.0%}，用时 {row['total_seconds']:.2f} 秒", flush=True)

    output = ROOT / "evaluation_results.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评估结果：{output}")


if __name__ == "__main__":
    main()
