"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化。"""

import json
import time
from pathlib import Path

import pymupdf

from rag import extract_fact, pdf_chunks, understand


ROOT = Path(__file__).parent


def main():
    started = time.perf_counter()
    fuzzy = understand("它怎么样？")
    assert "error" in fuzzy

    bad_pdf = ROOT / "data" / "invalid-test.pdf"
    bad_pdf.parent.mkdir(parents=True, exist_ok=True)
    bad_pdf.write_bytes(b"not a pdf")
    try:
        try:
            pdf_chunks(bad_pdf)
            raise AssertionError("无效PDF没有触发解析错误")
        except Exception as exc:
            assert isinstance(exc, (pymupdf.FileDataError, ValueError))
    finally:
        bad_pdf.unlink(missing_ok=True)

    english = extract_fact(
        "What is the registered capital?",
        [{"page": 52, "text": "法定代表人：程家明注册资本：5,520万元"}],
    )
    assert english and "5,520" in english

    result = {
        "fuzzy_question": {"pass": True, "message": fuzzy["error"]},
        "invalid_pdf": {"pass": True, "message": "显示可处理的解析错误"},
        "english_fact": {"pass": True, "answer": english},
        "seconds": round(time.perf_counter() - started, 3),
        "concurrency_stress": "未执行",
    }
    (ROOT / "stability_test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



