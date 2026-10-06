"""工单编号：人工智能NLP-RAG-图像内容解析及检索优化。"""

import json
import os
import time
from pathlib import Path

import pymupdf

from rag import answer_image_question, image_chunks

ROOT = Path(__file__).parent
PDF = Path(os.getenv("TASK04_PDF2", r"C:\Users\ZhuanZ\Downloads\RAG 工单\RAG 工单\附件\招股说明书2.pdf"))
QUESTIONS = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))[4:6]


def main():
    started = time.perf_counter()
    images = image_chunks(PDF)
    index = {"source": {"name": PDF.name}}
    results = []
    for question in QUESTIONS:
        answer = answer_image_question(question["question"], images, index)
        coverage = sum(word in answer for word in question["keywords"]) / len(question["keywords"])
        record = {"id": question["id"], "question": question["question"], "answer": answer,
                  "page": question["page"], "keyword_coverage": coverage}
        results.append(record)
        assert coverage == 1, f"题目{question['id']}答案不完整：{answer}"

    out = ROOT / "PDF图像解析模块"
    out.mkdir(exist_ok=True)
    doc = pymupdf.open(PDF)
    crops = [(39, (105, 55, 515, 660), "组织架构图原图.png"),
             (72, (85, 360, 505, 540), "IC市场增长图原图.png")]
    for page_no, rect, name in crops:
        page = doc[page_no - 1]
        page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), clip=pymupdf.Rect(rect), alpha=False).save(out / name)

    report = {"pass": True, "ocr_pages": [item["page"] for item in images],
              "ocr_seconds": round(time.perf_counter() - started, 3), "questions": results}
    (ROOT / "image_parser_test_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

