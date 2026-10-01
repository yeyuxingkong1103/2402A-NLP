"""工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化。"""

import json
import re
import time
from pathlib import Path

import numpy as np

from rag import ROOT, answer_verified, build_index, extract_fact, load_index, model, retrieve


LOCAL_PDF = ROOT / "data" / "uploads" / "招股说明书1.pdf"
PDF = LOCAL_PDF if LOCAL_PDF.exists() else Path(r"C:\Users\ZhuanZ\Downloads\RAG 工单\RAG 工单\附件\招股说明书1.pdf")


def old_retrieve(question, index, embedding_model, top_k=5):
    """工单1基线：只按向量相似度排序，结果去重到不同 PDF 页。"""
    query = re.sub(r"武汉兴图新科电子股份有限公司|招股意向书|招股说明书|根据", "", question)
    vector = embedding_model.encode([query], normalize_embeddings=True)[0]
    scores = index["vectors"] @ vector
    results = []
    pages = set()
    for i in np.argsort(scores)[::-1]:
        chunk = index["chunks"][int(i)]
        if chunk["page"] in pages:
            continue
        results.append({**chunk, "score": float(scores[i])})
        pages.add(chunk["page"])
        if len(results) == top_k:
            break
    return results


def hit_rate(rows, field, k):
    return sum(any(page == row["reference_page"] for page in row[field][:k]) for row in rows)


def keyword_coverage(text, keywords):
    clean = text.replace(",", "").replace("，", "").replace(" ", "")
    return sum(word.replace(",", "").replace(" ", "") in clean for word in keywords) / len(keywords)


def main():
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    embedding_model = model()
    folder = build_index(PDF, embedding_model)
    index = load_index(folder)
    rows = []
    for item in questions:
        start = time.perf_counter()
        before = old_retrieve(item["question"], index, embedding_model)
        before_seconds = time.perf_counter() - start
        start = time.perf_counter()
        _, after = retrieve(item["question"], index, embedding_model)
        after_seconds = time.perf_counter() - start
        final_answer, answer_seconds, raw_answer = answer_verified(item["question"], after, "deepseek-r1:1.5b")
        baseline_answer = extract_fact(item["question"], before) or "未从基线前5个片段中提取到完整答案。"
        before_top1_coverage = keyword_coverage(before[0]["text"] if before else "", item["keywords"])
        before_top5_coverage = keyword_coverage("\n".join(r["text"] for r in before), item["keywords"])
        after_top1_coverage = keyword_coverage(after[0]["text"] if after else "", item["keywords"])
        after_top5_coverage = keyword_coverage("\n".join(r["text"] for r in after), item["keywords"])
        answer_coverage = keyword_coverage(final_answer, item["keywords"])
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "reference": item["reference"],
            "keywords": item["keywords"],
            "reference_page": item["page"],
            "before_pages": [r["page"] for r in before],
            "before_top1_text": before[0]["text"] if before else "",
            "before_top5_text": "\n".join(r["text"] for r in before),
            "before_top1_keyword_coverage": before_top1_coverage,
            "before_top5_keyword_coverage": before_top5_coverage,
            "baseline_answer": baseline_answer,
            "baseline_answer_keyword_coverage": keyword_coverage(baseline_answer, item["keywords"]),
            "before_seconds": round(before_seconds, 4),
            "after_pages": [r["page"] for r in after],
            "after_top1_text": after[0]["text"] if after else "",
            "after_top5_text": "\n".join(r["text"] for r in after),
            "after_top1_keyword_coverage": after_top1_coverage,
            "after_top5_keyword_coverage": after_top5_coverage,
            "after_seconds": round(after_seconds, 4),
            "optimized_answer": final_answer,
            "optimized_raw_answer": raw_answer,
            "answer_keyword_coverage": answer_coverage,
            "answer_seconds": round(answer_seconds, 4),
            "request_seconds": round(after_seconds + answer_seconds, 4),
        })
        print(f"{item['id']}: before={rows[-1]['before_pages']} after={rows[-1]['after_pages']}", flush=True)

    out = {
        "baseline": "工单1向量检索基线",
        "optimized": "工单2字符词面+语义混合检索和问题意图加权",
        "top1_page_hits_before": hit_rate(rows, "before_pages", 1),
        "top1_page_hits_after": hit_rate(rows, "after_pages", 1),
        "top5_page_hits_before": hit_rate(rows, "before_pages", 5),
        "top5_page_hits_after": hit_rate(rows, "after_pages", 5),
        "mean_before_top1_keyword_coverage": round(np.mean([r["before_top1_keyword_coverage"] for r in rows]), 4),
        "mean_after_top1_keyword_coverage": round(np.mean([r["after_top1_keyword_coverage"] for r in rows]), 4),
        "mean_before_top5_keyword_coverage": round(np.mean([r["before_top5_keyword_coverage"] for r in rows]), 4),
        "mean_after_top5_keyword_coverage": round(np.mean([r["after_top5_keyword_coverage"] for r in rows]), 4),
        "answer_keyword_coverage": round(np.mean([r["answer_keyword_coverage"] for r in rows]), 4),
        "baseline_answer_keyword_coverage": round(np.mean([r["baseline_answer_keyword_coverage"] for r in rows]), 4),
        "mean_retrieval_seconds_before": round(np.mean([r["before_seconds"] for r in rows]), 4),
        "mean_retrieval_seconds_after": round(np.mean([r["after_seconds"] for r in rows]), 4),
        "mean_answer_seconds": round(np.mean([r["answer_seconds"] for r in rows]), 4),
        "mean_request_seconds": round(np.mean([r["request_seconds"] for r in rows]), 4),
        "requests_under_3_seconds": sum(r["request_seconds"] <= 3 for r in rows),
        "questions": rows,
    }
    path = ROOT / "retrieval_comparison.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"comparison saved: {path}")
    print(f"top1 page recall: {out['top1_page_hits_before']}/10 -> {out['top1_page_hits_after']}/10")
    print(f"top5 page recall: {out['top5_page_hits_before']}/10 -> {out['top5_page_hits_after']}/10")


if __name__ == "__main__":
    main()
