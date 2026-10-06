# -*- coding: utf-8 -*-
"""评测与压测工具测试：题库 / 管线 / 报告 / 对比 / JTL 解析。"""

import json
from pathlib import Path
import pytest
from app.eval.dataset import load_eval_dataset, load_samples_from_report
from types import SimpleNamespace
from app.core.prompts import DEFAULT_PROMPT_TEMPLATE
from app.eval.runner import EvalPipeline
from app.eval.report import format_report
from app.eval.report import build_comparison
from app.stress.jtl import parse_jtl

# ---------- 题库加载（原 tests/test_eval_dataset.py） ----------

def write_dataset(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "eval_dataset.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def sample(question="高血压的诊断标准是什么？", **overrides):
    item = {
        "question": question,
        "reference_answer": "收缩压≥140mmHg或舒张压≥90mmHg。",
        "reference_contexts": ["诊断标准：收缩压≥140mmHg。"],
    }
    item.update(overrides)
    return item


def test_load_returns_name_role_and_samples(tmp_path):
    path = write_dataset(
        tmp_path,
        {"name": "基线", "role": "林医生", "samples": [sample()]},
    )
    data = load_eval_dataset(path)

    assert data.name == "基线"
    assert data.role == "林医生"
    assert len(data.samples) == 1
    assert data.samples[0]["question"] == "高血压的诊断标准是什么？"


def test_load_rejects_missing_samples(tmp_path):
    with pytest.raises(ValueError, match="samples"):
        load_eval_dataset(write_dataset(tmp_path, {"name": "x"}))


def test_load_rejects_sample_without_reference_answer(tmp_path):
    with pytest.raises(ValueError, match="第 1 题.*reference_answer"):
        load_eval_dataset(
            write_dataset(tmp_path, {"samples": [sample(reference_answer="")]})
        )


def test_load_rejects_sample_without_reference_contexts(tmp_path):
    with pytest.raises(ValueError, match="第 2 题.*reference_contexts"):
        load_eval_dataset(
            write_dataset(
                tmp_path,
                {"samples": [sample(), sample(reference_contexts=[])]},
            )
        )


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_dataset(tmp_path / "不存在.json")


REPORT_SAMPLE = {
    "user_input": "诊断标准？",
    "response": "收缩压≥140mmHg。",
    "retrieved_contexts": ["诊断标准：收缩压≥140mmHg。"],
    "reference": "收缩压≥140或舒张压≥90。",
    "reference_contexts": ["诊断标准：收缩压≥140mmHg。"],
}


def test_load_samples_from_report_maps_saved_fields(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"scores": {}, "rows": [], "samples": [REPORT_SAMPLE]}, ensure_ascii=False),
        encoding="utf-8",
    )
    samples = load_samples_from_report(path)

    assert len(samples) == 1
    assert samples[0] == REPORT_SAMPLE


def test_load_samples_from_report_rejects_missing_samples(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"scores": {}, "rows": []}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="samples"):
        load_samples_from_report(path)

# ---------- 评测管线（原 tests/test_eval_runner.py） ----------

ROLE = SimpleNamespace(
    id=2,
    name="林医生",
    persona="全科医生",
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
)


class FakeKnowledge:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    async def retrieve(self, role_id, query, top_k=4, recall_k=30):
        self.queries.append((role_id, query, top_k, recall_k))
        return self.chunks


class FakeLLM:
    def __init__(self, reply="参考答案"):
        self.reply = reply
        self.last_messages = None

    async def chat(self, messages):
        self.last_messages = messages
        return self.reply


@pytest.fixture()
def pipeline():
    knowledge = FakeKnowledge(["知识块一：低盐饮食。", "知识块二：运动150分钟。"])
    llm = FakeLLM()
    return EvalPipeline(knowledge_service=knowledge, llm=llm, role=ROLE), knowledge, llm


SAMPLE = {
    "question": "高血压患者每天吃多少盐？",
    "reference_answer": "不超过5克。",
    "reference_contexts": ["每日食盐摄入不超过5克。"],
}


async def test_run_sample_returns_ragas_sample_fields(pipeline):
    pipeline_obj, knowledge, _ = pipeline
    result = await pipeline_obj.run_sample(SAMPLE)

    assert result["user_input"] == SAMPLE["question"]
    assert result["response"] == "参考答案"
    assert result["retrieved_contexts"] == knowledge.chunks
    assert result["reference"] == "不超过5克。"
    assert result["reference_contexts"] == ["每日食盐摄入不超过5克。"]


async def test_run_sample_retrieves_for_role_and_question(pipeline):
    pipeline_obj, knowledge, _ = pipeline
    await pipeline_obj.run_sample(SAMPLE)

    role_id, query, top_k, recall_k = knowledge.queries[-1]
    assert role_id == ROLE.id
    assert query == SAMPLE["question"]
    assert top_k == pipeline_obj.top_k
    assert recall_k == pipeline_obj.recall_k


async def test_run_sample_injects_retrieved_contexts_into_prompt(pipeline):
    pipeline_obj, _, llm = pipeline
    await pipeline_obj.run_sample(SAMPLE)

    system = llm.last_messages[0]["content"]
    assert "知识块一：低盐饮食。" in system
    assert "知识块二：运动150分钟。" in system
    assert "林医生" in system  # 角色人设仍在


async def test_run_all_collects_every_sample(pipeline):
    pipeline_obj, _, _ = pipeline
    results = await pipeline_obj.run_all([SAMPLE, {**SAMPLE, "question": "第二题"}])

    assert len(results) == 2
    assert results[1]["user_input"] == "第二题"

# ---------- 报告格式化（原 tests/test_eval_report.py） ----------

SCORES = {
    "faithfulness": 0.92,
    "answer_relevancy": 0.87,
    "context_precision": 0.75,
    "context_recall": 0.60,
    "answer_correctness": 0.81,
}

ROWS = [
    {"question": "每天吃多少盐？", "faithfulness": 1.0, "answer_relevancy": 0.9},
    {"question": "运动多久？", "faithfulness": 0.8, "answer_relevancy": 0.7},
]


def test_report_contains_header_and_info():
    md = format_report("基线评测", "林医生", SCORES, ROWS, model="deepseek-chat")
    assert "# RAGAS 基线评测" in md
    assert "林医生" in md
    assert "deepseek-chat" in md


def test_report_contains_all_metric_scores():
    md = format_report("基线评测", "林医生", SCORES, ROWS)
    for name, score in SCORES.items():
        assert name in md
        assert f"{score:.3f}" in md


def test_report_contains_per_question_table():
    md = format_report("基线评测", "林医生", SCORES, ROWS)
    assert "每天吃多少盐？" in md
    assert "运动多久？" in md


def test_report_highlights_low_metrics_in_conclusion():
    md = format_report("基线评测", "林医生", SCORES, ROWS)
    # context_recall 最低，结论中应点名
    assert "context_recall" in md
    assert "0.600" in md


def test_report_omits_conclusion_section_when_no_scores():
    md = format_report("基线评测", "林医生", {}, [])
    assert "结论" not in md


def test_report_renders_nan_as_dash_with_footnote():
    rows = [{"question": "某题", "faithfulness": float("nan")}]
    md = format_report("基线评测", "林医生", {"faithfulness": 0.5}, rows)
    # 表格单元格里 nan 应显示为 —（脚注说明原因）
    assert "| 某题 | — |" in md
    assert "按有效样本聚合" in md

# ---------- 两轮对比（原 tests/test_eval_compare.py） ----------

BASELINE = {
    "faithfulness": 0.718,
    "answer_relevancy": 0.901,
    "answer_correctness": 0.597,
}
CURRENT = {
    "faithfulness": 0.931,
    "answer_relevancy": 0.884,
    "answer_correctness": 0.700,
}
BASE_ROWS = [
    {"question": "诊断标准？", "answer_correctness": 0.449, "faithfulness": 0.5},
]
CUR_ROWS = [
    {"question": "诊断标准？", "answer_correctness": 0.908, "faithfulness": 1.0},
]


def test_comparison_contains_delta_with_sign():
    md = build_comparison("基线", "第二轮", BASELINE, BASE_ROWS, CURRENT, CUR_ROWS)
    assert "+0.213" in md  # faithfulness 提升
    assert "-0.017" in md  # answer_relevancy 回归
    assert "+0.103" in md  # answer_correctness 提升


def test_comparison_contains_both_round_scores():
    md = build_comparison("基线", "第二轮", BASELINE, BASE_ROWS, CURRENT, CUR_ROWS)
    assert "0.718" in md
    assert "0.931" in md


def test_comparison_lists_improved_and_regressed_metrics():
    md = build_comparison("基线", "第二轮", BASELINE, BASE_ROWS, CURRENT, CUR_ROWS)
    assert "faithfulness" in md
    assert "answer_relevancy" in md
    # 结论里应分别列出提升与回归
    assert "提升" in md
    assert "回归" in md


def test_comparison_contains_per_question_deltas():
    md = build_comparison("基线", "第二轮", BASELINE, BASE_ROWS, CURRENT, CUR_ROWS)
    assert "诊断标准？" in md
    assert "+0.459" in md  # 第 1 题 correctness 0.449 -> 0.908

# ---------- Jmeter 结果解析（原 tests/test_jtl.py） ----------

SAMPLE_JTL = """timeStamp,elapsed,label,responseCode,responseMessage,threadName,dataType,success,failureMessage,bytes,sentBytes,grpThreads,allThreads,URL,Latency,IdleTime,Connect
1789473000000,100,POST /api/chat,200,OK,tg 1-1,text,true,,1200,400,1,1,http://127.0.0.1:8000/api/chat,98,0,1
1789473000100,200,POST /api/chat,200,OK,tg 1-1,text,true,,1200,400,1,1,http://127.0.0.1:8000/api/chat,195,0,1
1789473000300,300,POST /api/chat,200,OK,tg 1-1,text,true,,1200,400,1,1,http://127.0.0.1:8000/api/chat,290,0,1
1789473000600,400,POST /api/chat,200,OK,tg 1-1,text,true,,1200,400,1,1,http://127.0.0.1:8000/api/chat,390,0,1
1789473001000,500,POST /api/chat,500,err,tg 1-1,text,false,,100,400,1,1,http://127.0.0.1:8000/api/chat,490,0,1
"""


def test_parse_jtl_counts_samples_and_errors():
    stats = parse_jtl(SAMPLE_JTL)
    assert stats["total"] == 5
    assert stats["errors"] == 1
    assert stats["error_rate"] == pytest.approx(0.2)


def test_parse_jtl_computes_latency_percentiles():
    stats = parse_jtl(SAMPLE_JTL)
    # elapsed 升序：100,200,300,400,500
    assert stats["avg_ms"] == pytest.approx(300.0)
    assert stats["min_ms"] == 100
    assert stats["max_ms"] == 500
    assert stats["p95_ms"] == 500
    assert stats["median_ms"] == 300


def test_parse_jtl_computes_qps_from_span():
    stats = parse_jtl(SAMPLE_JTL)
    # 首尾时间戳差 1000ms（0→1000），QPS = 5/1.0 = 5.0
    assert stats["qps"] == pytest.approx(5.0, abs=0.5)
