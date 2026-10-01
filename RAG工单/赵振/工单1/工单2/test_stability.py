"""工单2的输入容错、PDF解析异常和英文问答检查。"""

import json
import tempfile
from pathlib import Path

from rag import ROOT, load_index, model, pdf_chunks, retrieve, answer_verified, understand


def main():
    ambiguous = understand("它怎么样？")
    assert "error" in ambiguous, "模糊问题应提示补充信息"

    with tempfile.NamedTemporaryFile(suffix=".pdf") as bad_pdf:
        bad_pdf.write(b"not a pdf")
        bad_pdf.flush()
        try:
            pdf_chunks(bad_pdf.name)
        except Exception:
            parse_error_caught = True
        else:
            parse_error_caught = False
    assert parse_error_caught, "损坏PDF应触发解析错误，供界面显示"

    index_file = next((ROOT / "data" / "indexes").glob("*/source.json"))
    index = load_index(index_file.parent)
    embedding_model = model()
    question = "Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?"
    _, evidence = retrieve(question, index, embedding_model)
    answer, seconds, _ = answer_verified(question, evidence, "deepseek-r1:1.5b")
    assert "程家明" in answer, f"英文问题答案不正确：{answer}"

    result = {
        "ambiguous_input": "passed",
        "invalid_pdf_error_path": "passed",
        "english_question": question,
        "english_answer": answer,
        "query_seconds": round(seconds, 4),
    }
    (ROOT / "stability_test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
