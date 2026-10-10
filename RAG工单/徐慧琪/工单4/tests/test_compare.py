# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""Task 18：双模式对比渲染 + RAGAS 诚实执行路径的单元测试（无网络、无语料）。"""
from __future__ import annotations

from rag04.config import get_settings
from rag04.eval.compare import _delta, write_comparison
from rag04.eval.ragas_eval import _METRIC_KEYS, run_ragas, write_ragas_report


def _res(mode, acc, mrr_, lat):
    return {
        "mode": mode, "lang": "zh", "n_questions": 16,
        "metrics": {"answer_accuracy": acc, "mrr": mrr_, "hit_rate": 0.9,
                    "ndcg@5": 0.8, "latency_mean_ms": lat},
        "rows": [{"qid": 5, "question": "q", "answer": "a", "answer_key": ["k"],
                  "coverage": acc, "correct": True, "strict": True,
                  "gold_pages": [38], "block_type": "image",
                  "citations": [], "retrieved_pages": [38],
                  "latency_ms": lat, "llm_backend": "deepseek",
                  "refused": False, "error": ""}],
    }


def test_delta_computes_pp_change():
    assert _delta(0.95, 0.80) == "+15.0pp"
    assert _delta(0.70, 0.80) == "-10.0pp"


def test_write_comparison_contains_both_modes(tmp_path):
    p = write_comparison(_res("baseline_03", 0.55, 0.5, 1200),
                         _res("full_04", 0.94, 0.9, 800),
                         tmp_path / "cmp.md")
    txt = p.read_text(encoding="utf-8")
    assert "baseline_03" in txt
    assert "full_04" in txt
    assert "answer_accuracy" in txt
    assert "+39.0pp" in txt, "应给出提升幅度"


def test_write_comparison_creates_parent_dir(tmp_path):
    p = write_comparison(_res("baseline_03", 0.5, 0.5, 1),
                         _res("full_04", 0.9, 0.9, 1),
                         tmp_path / "deep" / "nested" / "cmp.md")
    assert p.exists()


# ---------- Fix 3：结论句必须由实测值推导，且缺值不得抛裸 TypeError ----------

def _res_en(mode, img_cov):
    """英文模式结果：answer_accuracy 按设计为 None，coverage 也是 None。"""
    r = _res(mode, None, 0.7, 900)
    r["lang"] = "en"
    r["metrics"]["answer_accuracy"] = None
    for row in r["rows"]:
        row["coverage"] = img_cov
    return r


def test_write_comparison_tolerates_missing_metric(tmp_path):
    """英文模式 answer_accuracy=None：不得抛裸 TypeError，须写明「不适用」。"""
    p = write_comparison(_res("baseline_03", 0.55, 0.5, 1200),
                         _res_en("full_04", None),
                         tmp_path / "cmp_en.md")
    txt = p.read_text(encoding="utf-8")

    assert "answer_accuracy" in txt
    assert "—" in txt and "不适用" in txt
    assert "不适用" in txt.split("## 四、结论")[1], "结论区不得给出准确率数值"


def test_write_comparison_image_note_is_derived_not_hardcoded(tmp_path):
    """实测覆盖率 0.1/0.2 时，不得再写「覆盖率为 0 属预期」。

    原实现把「baseline 无图像能力，覆盖率为 0 属预期」写死，与实测值矛盾。
    """
    p = write_comparison(_res("baseline_03", 0.55, 0.5, 1200),
                         _res("full_04", 0.94, 0.9, 800),
                         tmp_path / "cmp.md")
    txt = p.read_text(encoding="utf-8")

    assert "覆盖率为 0 属预期" not in txt
    assert "图像题从 0 覆盖提升至可用" not in txt
    assert "主要增益来源" not in txt, "不得保留未经消融实验支持的因果断言"
    assert "不作因果断言" in txt


def test_write_comparison_reports_measured_coverage_gain(tmp_path):
    """说明列必须写实测覆盖率与差值（baseline 0.1 → full 0.9）。"""
    base = _res("baseline_03", 0.55, 0.5, 1200)
    full = _res("full_04", 0.94, 0.9, 800)
    for r in base["rows"]:
        r["coverage"] = 0.1
    for r in full["rows"]:
        r["coverage"] = 0.9
    txt = write_comparison(base, full, tmp_path / "cmp.md").read_text(encoding="utf-8")

    assert "0.10 → 0.90" in txt
    assert "+80.0pp" in txt


def test_write_comparison_english_mode_notes_missing_coverage(tmp_path):
    """英文模式 coverage 为 None：说明列如实写「未上报」而不是 0.0。"""
    base = _res("baseline_03", 0.55, 0.5, 1200)
    full = _res_en("full_04", None)
    txt = write_comparison(base, full, tmp_path / "cmp.md").read_text(encoding="utf-8")

    assert "覆盖率未上报" in txt
    img_section = txt.split("## 三、图像题专项")[1].split("## 四、")[0]
    assert "| 0.0 |" not in img_section, "不得把缺失当作 0 上报"
    assert "| 不适用 |" in img_section


def test_comparison_documents_baseline_use_hybrid_caveat(tmp_path):
    """对比口径表下的脚注必须始终渲染：baseline 实测数据产出于 use_hybrid
    还是死开关的年代（实际跑了稠密+稀疏），不写明会让人误读优化幅度。"""
    txt = write_comparison(_res("baseline_03", 0.55, 0.5, 1200),
                           _res("full_04", 0.94, 0.9, 800),
                           tmp_path / "cmp.md").read_text(encoding="utf-8")

    assert "`use_hybrid` 尚未接线" in txt
    assert "已登记为后续项" in txt
    assert "只会低估本工单的优化幅度，不会夸大" in txt
    # 必须挂在对比口径表之后、指标表之前
    assert txt.index("图像题专项") > txt.index("`use_hybrid` 尚未接线")
    assert txt.index("## 二、指标对比") > txt.index("`use_hybrid` 尚未接线")
    assert txt.index("`use_hybrid` 尚未接线") > txt.index("| 图像 | 不解析 |")


def _eval_result() -> dict:
    """最小可用的评估结果（1 题、1 条引用），仅供 run_ragas 走通前置分支。"""
    return {
        "mode": "full_04", "lang": "zh", "n_questions": 1,
        "metrics": {"answer_accuracy": 1.0},
        "rows": [{
            "qid": 1, "question": "本次发行多少股？", "answer": "2000 万股",
            "answer_key": ["2000 万股"], "coverage": 1.0, "correct": True,
            "strict": True, "gold_pages": [38], "block_type": "text",
            "citations": [{"snippet": "本次发行 2000 万股"}],
            "retrieved_pages": [38], "latency_ms": 100.0,
            "llm_backend": "deepseek", "refused": False, "error": "",
        }],
    }


def test_run_ragas_without_key_reports_unavailable(monkeypatch):
    """无 LLM key 时必须如实返回「未执行」，不抛异常、不给数值。"""
    for var in ("API", "QWEN_API", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    res = run_ragas(_eval_result(), get_settings("full_04"))

    assert res["available"] is False
    assert res["metrics"] == {}
    assert res["reason"], "未执行必须写明原因"
    assert res["metrics_run"] == []


def test_write_ragas_report_unavailable_has_no_numbers(tmp_path):
    """未执行的报告只写状态 + 原因，不得出现任何数值指标行。"""
    reason = "未配置 LLM API key，RAGAS 需 LLM 打分"
    p = write_ragas_report({"available": False, "metrics": {}, "reason": reason},
                           tmp_path / "ragas.md", mode="full_04")
    txt = p.read_text(encoding="utf-8")

    assert "未执行" in txt
    assert reason in txt
    assert "| 指标 | 数值 | 含义 |" not in txt, "未执行不应渲染指标表"
    for key in _METRIC_KEYS:
        assert f"| {key} |" not in txt, f"不得出现 {key} 的伪造数值行"
