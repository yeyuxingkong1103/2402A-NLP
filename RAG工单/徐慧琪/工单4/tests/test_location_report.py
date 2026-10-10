# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""答案定位报告生成器纯单测：题册骨架、证据出处、引用渲染与 CLI 落盘。

本文件**不读真实评估结果、不打开索引、不调用 LLM**：`build_report` 只接收
一个字典（测试自带切片夹具），`main` 的输入输出路径由参数指定。真实
`docs/reports/eval_full_04.json` 的生成属 phase 2（评估重跑完成后）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gen_location_report import (  # noqa: E402
    PLACEHOLDER_NO_RESULT,
    build_report,
    main,
)

ROOT = Path(__file__).resolve().parents[1]

# 16 题的固定顺序（与 rag04.eval.questions.QUESTIONS 一致）
QIDS = [1, 2, 3, 4, 5, 6, 260, 95, 33, 34, 957, 793, 795, 543, 531, 207]


def _row(qid: int) -> dict:
    """一条最小评估行：只保留生成器读取的字段。"""
    return {
        "qid": qid,
        "question": f"问题 {qid}",
        "answer": "答案正文测试串",
        "coverage": 1.0,
        "correct": True,
        "strict": qid == 5,
        "refused": False,
        "gold_pages": [39],
        "retrieved_pages": [39, 309],
        "citations": [
            {"doc_id": "招股说明书2", "page": 39, "block_type": "text",
             "source_id": "figuretext#39#fig4", "score": 6.638414,
             "image_path": "data/figures/招股说明书2_p39_7fb46462e43d.png"},
            {"doc_id": "招股说明书2", "page": 309, "block_type": "text",
             "source_id": "text#309#1493", "score": 6.187304, "image_path": ""},
        ],
    }


def _result(rows: list[dict] | None = None) -> dict:
    return {
        "mode": "full_04",
        "lang": "zh",
        "n_questions": 16,
        "metrics": {"answer_accuracy": 1.0},
        "rows": rows if rows is not None else [_row(5)],
    }


# ------------------------------------------------------------ 题册骨架

def test_report_lists_all_16_questions():
    txt = build_report(_result())
    for qid in QIDS:
        assert f"## id {qid}" in txt, f"缺 id {qid} 小节"
    assert txt.count("## id ") == 16


def test_report_keeps_standard_question_and_answer_key():
    txt = build_report(_result())
    # 标准问题与要点来自 QUESTIONS（不依赖评估结果）
    assert "组织结构图" in txt
    assert "渠道销售部" in txt and "珠海销售处" in txt
    assert "标准答案要点" in txt


def test_report_works_without_eval_result():
    """评估 JSON 缺失时仍产出题册，但必须显式标注待回填，不得编造数值。"""
    txt = build_report(None)
    assert PLACEHOLDER_NO_RESULT in txt
    for qid in QIDS:
        assert f"## id {qid}" in txt
    # 不得出现逐题成绩行（没有数据就不给数字）
    assert "**要点覆盖率**" not in txt
    assert "**实际召回页码**" not in txt
    assert "| 覆盖率 |" not in txt


# ------------------------------------------------------------ 证据与引用

def test_report_shows_gold_evidence_page_and_type():
    txt = build_report(_result())
    assert "招股说明书2" in txt
    assert "第 39 页" in txt
    assert "image" in txt


def test_report_prefers_matched_citation_source_id():
    """证据出处取命中 gold_pages 的真实引用块（含来源ID），便于人工复核。"""
    txt = build_report(_result())
    assert "figuretext#39#fig4" in txt
    assert "text#309#1493" in txt          # 引用来源表里也要有非金标引用


def test_report_includes_image_path_for_image_citation():
    txt = build_report(_result())
    assert "招股说明书2_p39_7fb46462e43d.png" in txt


def test_report_renders_coverage_and_verdict():
    txt = build_report(_result())
    assert "1.0" in txt
    assert "✅ 正确" in txt


def test_report_marks_refused_rows():
    row = _row(1)
    row.update({"correct": False, "coverage": 0.0, "refused": True})
    txt = build_report(_result([row]))
    assert "🚫" in txt


def test_report_renders_not_applicable_coverage():
    """英文模式 coverage 为 null（标准要点是中文），必须写「不适用」而非 0。"""
    row = _row(5)
    row.update({"coverage": None, "correct": None})
    txt = build_report(_result([row]))
    assert "不适用" in txt


def test_report_has_overview_table():
    txt = build_report(_result())
    assert "| id |" in txt
    for qid in QIDS:
        assert f"| {qid} |" in txt, f"总览表缺 id {qid}"


# ------------------------------------------------------------------ CLI

def test_main_writes_report_to_given_path(tmp_path):
    src = tmp_path / "eval_fixture.json"
    src.write_text(json.dumps(_result(), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "答案定位报告.md"

    rc = main(["--src", str(src), "--out", str(out)])

    assert rc == 0
    assert out.exists()
    txt = out.read_text(encoding="utf-8")
    assert "## id 5" in txt
    assert "答案正文测试串" in txt, "应读入夹具 JSON 的答案"
    assert src.as_posix() in txt, "报告应写明数据来源，便于溯源"
    assert PLACEHOLDER_NO_RESULT not in txt


def test_main_tolerates_missing_source(tmp_path):
    out = tmp_path / "答案定位报告.md"

    rc = main(["--src", str(tmp_path / "nope.json"), "--out", str(out)])

    assert rc == 0, "缺评估结果不应崩溃：题册骨架仍要产出"
    txt = out.read_text(encoding="utf-8")
    assert PLACEHOLDER_NO_RESULT in txt
    assert "## id 5" in txt
