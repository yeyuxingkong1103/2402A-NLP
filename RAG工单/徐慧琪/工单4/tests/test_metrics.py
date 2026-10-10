# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from rag04.config import Settings
from rag04.eval.metrics import (
    precision_at_k, recall_at_k, hit_rate, mrr, ndcg_at_k, is_relevant, verdict,
)
from rag04.eval.questions import QUESTIONS
from rag04.eval.runner import _hits_in_doc, run_eval, write_report
from rag04.schema import Answer, Hit


def _h(page, cid=None, doc="d"):
    return Hit(chunk_id=cid or f"p{page}", doc_id=doc, page=page, block_type="text",
               source_id="s", text="t", score=1.0)


def test_is_relevant_by_page():
    assert is_relevant(_h(38), [38, 39])
    assert not is_relevant(_h(1), [38, 39])


def test_precision_at_k_basic():
    hits = [_h(38), _h(1), _h(38), _h(2)]
    assert precision_at_k(hits, [38], k=4) == 0.5


def test_precision_at_k_truncates_to_available():
    hits = [_h(38)]
    assert precision_at_k(hits, [38], k=5) == 1.0


def test_precision_at_k_empty():
    assert precision_at_k([], [38], k=5) == 0.0


def test_recall_at_k():
    hits = [_h(38), _h(71), _h(1)]
    assert recall_at_k(hits, [38, 71], k=3) == 1.0
    assert recall_at_k(hits, [38, 71], k=1) == 0.5


def test_recall_at_k_no_gold():
    assert recall_at_k([_h(1)], [], k=5) == 0.0


def test_hit_rate():
    assert hit_rate([True, False, True, True]) == 0.75
    assert hit_rate([]) == 0.0


def test_mrr_perfect_first():
    assert mrr([[_h(38)], [_h(71)]], [[38], [71]]) == 1.0


def test_mrr_second_position():
    got = mrr([[_h(1), _h(38)]], [[38]])
    assert abs(got - 0.5) < 1e-9


def test_mrr_no_hit_is_zero():
    assert mrr([[_h(1), _h(2)]], [[38]]) == 0.0


def test_mrr_averages_across_queries():
    got = mrr([[_h(38)], [_h(1), _h(71)]], [[38], [71]])
    assert abs(got - 0.75) < 1e-9


def test_ndcg_perfect_ranking_is_one():
    hits = [_h(38), _h(71)]
    assert abs(ndcg_at_k(hits, [38, 71], k=2) - 1.0) < 1e-9


def test_ndcg_penalizes_lower_position():
    good = ndcg_at_k([_h(38), _h(1)], [38], k=2)
    bad = ndcg_at_k([_h(1), _h(38)], [38], k=2)
    assert good > bad


def test_ndcg_zero_when_no_hit():
    assert ndcg_at_k([_h(1), _h(2)], [38], k=2) == 0.0


def test_ndcg_dedups_repeated_relevant():
    """同一页被召回多次不应刷高 NDCG。"""
    once = ndcg_at_k([_h(38, "a")], [38], k=2)
    twice = ndcg_at_k([_h(38, "a"), _h(38, "b")], [38], k=2)
    assert abs(once - twice) < 1e-9


# ---------- 正确性判定口径 ----------

def test_verdict_normal_uses_threshold():
    assert verdict(0.85, threshold=0.80, strict=False) is True
    assert verdict(0.75, threshold=0.80, strict=False) is False


def test_verdict_strict_requires_full_coverage():
    """id 5 严格口径：10 个要点全中才算对，9/10 也不行。"""
    assert verdict(1.0, threshold=0.80, strict=True) is True
    assert verdict(0.9, threshold=0.80, strict=True) is False
    assert verdict(0.8, threshold=0.80, strict=True) is False


def test_verdict_strict_ignores_threshold():
    """严格口径不受 threshold 影响，哪怕阈值调低也必须全中。"""
    assert verdict(0.9, threshold=0.10, strict=True) is False


def test_verdict_refused_is_never_correct():
    """RC6：拒答不得判为正确，即使字面覆盖率达标（id531 假阳性）。"""
    assert verdict(1.0, threshold=0.80, strict=False, refused=True) is False
    assert verdict(1.0, threshold=0.80, strict=True, refused=True) is False
    # 不拒答时口径不变，回归保护
    assert verdict(1.0, threshold=0.80, strict=False, refused=False) is True


def test_verdict_refused_defaults_false_for_backward_compat():
    assert verdict(0.9, threshold=0.80, strict=False) is True


# ---------- 执行器：离线桩（不调用外部 API / 不加载模型） ----------

def _answer_for(q, text_question, omit=(), refused=False):
    answer = " ".join(k for k in q.answer_key if k not in omit)
    hits = [_h(p, f"{q.doc_id}-p{p}", q.doc_id) for p in q.gold_pages]
    lang = "en" if text_question == q.question_en else "zh"
    return Answer(question=text_question, answer=answer, lang=lang, hits=hits,
                  refused=refused)


class _FakePipeline:
    """按题返回构造好的 Answer；hits_for 可整题替换召回结果，refuse 可整题拒答。"""

    def __init__(self, omit=None, hits_for=None, refuse=()):
        self.omit = omit or {}
        self.hits_for = hits_for or {}
        self.refuse = set(refuse)
        self.loaded = 0

    def _ensure_loaded(self):
        self.loaded += 1

    def ask(self, question):
        q = next(x for x in QUESTIONS if question in (x.question, x.question_en))
        ans = _answer_for(q, question, self.omit.get(q.qid, ()),
                          refused=q.qid in self.refuse)
        if q.qid in self.hits_for:
            ans.hits = list(self.hits_for[q.qid])
        return ans


def test_hits_in_doc_drops_other_document_same_page():
    """两本招股书页码重叠，异库同页块必须剔除。"""
    hits = [_h(39, "a", "招股说明书1"), _h(39, "b", "招股说明书2")]
    kept = _hits_in_doc(hits, "招股说明书2")
    assert [h.chunk_id for h in kept] == ["b"]


def test_run_eval_does_not_score_other_document_hits():
    """id 6 的金标页 310 只被「招股说明书1」召回时应算漏召，不得虚高。"""
    other_doc_310 = _h(310, "招股说明书1-p310", "招股说明书1")
    pipe = _FakePipeline(hits_for={6: [_h(72, "招股说明书2-p72", "招股说明书2"),
                                      other_doc_310]})
    result = run_eval(Settings(), pipeline=pipe)
    row6 = next(r for r in result["rows"] if r["qid"] == 6)

    assert row6["retrieved_pages"] == [72, 310]
    assert row6["scored_pages"] == [72], "异库 p310 不得进入计分"
    assert row6["hits_other_doc"] == 1
    assert result["metrics"]["recall@10"] == round((15 + 0.5) / 16, 4)
    assert pipe.loaded == 1


def test_run_eval_enforces_strict_verdict_for_id5():
    """id 5 是严格口径：9/10 要点也不判对（约束 10）。"""
    pipe = _FakePipeline(omit={5: ("武汉销售处",)})
    result = run_eval(Settings(), pipeline=pipe)
    row5 = next(r for r in result["rows"] if r["qid"] == 5)

    assert row5["coverage"] == 0.9
    assert row5["correct"] is False
    assert result["metrics"]["answer_accuracy"] == round(15 / 16, 4)


def test_run_eval_normal_verdict_uses_threshold():
    """普通题：覆盖率 0.8 恰好达标判对；0.67 判错。"""
    ok = run_eval(Settings(), pipeline=_FakePipeline(omit={2: ("其他与主营业务相关的营运资金",)}))
    row2 = next(r for r in ok["rows"] if r["qid"] == 2)
    assert abs(row2["coverage"] - 0.8) < 1e-9 and row2["correct"] is True

    bad = run_eval(Settings(), pipeline=_FakePipeline(omit={1: ("25.04%",)}))
    row1 = next(r for r in bad["rows"] if r["qid"] == 1)
    assert row1["coverage"] < 0.8 and row1["correct"] is False


def test_run_eval_refused_answer_is_not_correct():
    """RC6：id531 式假阳性——答案逐字复述了全部要点但明确拒答，不得判对。"""
    pipe = _FakePipeline(refuse={531})
    result = run_eval(Settings(), pipeline=pipe)
    row = next(r for r in result["rows"] if r["qid"] == 531)

    assert row["coverage"] == 1.0, "覆盖率仍如实上报"
    assert row["refused"] is True
    assert row["correct"] is False, "拒答不得计入 answer_accuracy"
    assert result["metrics"]["answer_accuracy"] == round(15 / 16, 4)


def test_run_eval_isolates_question_failure():
    class _Boom(_FakePipeline):
        def ask(self, question):
            if question == QUESTIONS[0].question:
                raise RuntimeError("boom")
            return super().ask(question)

    result = run_eval(Settings(), pipeline=_Boom())
    row0 = result["rows"][0]
    assert row0["error"].startswith("RuntimeError")
    assert row0["answer"] == "" and row0["correct"] is False
    assert result["metrics"]["answer_accuracy"] < 1.0


def test_run_eval_english_mode_marks_accuracy_unavailable():
    """answer_key 仅中文：英文模式不得把 ~0 的覆盖率当答案准确率上报。"""
    result = run_eval(Settings(), pipeline=_FakePipeline(), use_english=True)

    assert result["lang"] == "en"
    assert result["metrics"]["answer_accuracy"] is None
    assert "accuracy_note" in result["metrics"]
    assert all(r["coverage"] is None and r["correct"] is None for r in result["rows"])
    assert all(r["question"] == q.question_en
               for r, q in zip(result["rows"], QUESTIONS))
    assert result["metrics"]["recall@10"] == 1.0, "检索指标与语言无关，仍应计算"


# ---------- Fix 2：英文报告必须暴露拒答 + 逐行语言 ----------

def test_rows_record_answer_language(tmp_path):
    """逐题记录 ans.lang，英文报告可自证（不再靠整份报告语言推断）。"""
    zh = run_eval(Settings(), pipeline=_FakePipeline())
    en = run_eval(Settings(), pipeline=_FakePipeline(), use_english=True)
    assert {r["lang"] for r in zh["rows"]} == {"zh"}
    assert {r["lang"] for r in en["rows"]} == {"en"}


def test_write_report_english_mode_shows_refusal_not_not_applicable(tmp_path):
    """英文模式 correct 恒为 None，但拒答必须照实渲染，不得被「不适用」掩盖。"""
    result = run_eval(Settings(), pipeline=_FakePipeline(refuse={531}),
                      use_english=True)
    text = write_report(result, tmp_path / "en.md").read_text(encoding="utf-8")
    row531 = next(r for r in result["rows"] if r["qid"] == 531)

    assert row531["refused"] is True and row531["correct"] is None
    assert "id 531" in text and "🚫 拒答（不计正确）" in text
    assert "**语言**：en" in text
    # 非拒答题仍照旧显示「不适用」，避免与新文案混淆
    assert "不适用（本语言模式不判答案正确性）" in text


def test_retrieval_only_answer_is_never_scored_correct():
    """C5：三段降级全挂时返回的是检索原文兜底，不得计入 answer_accuracy。

    LLMClient 在源头把 retrieval_only 记为 refused（见 test_generate），
    这里验证评估侧据此判错：单要点题即使原文片段含要点也不算答对。
    """
    class _RetrievalOnly(_FakePipeline):
        def ask(self, question):
            ans = super().ask(question)
            ans.refused = True
            ans.llm_backend = "retrieval_only"
            ans.refusal_source = "retrieval_only"
            return ans

    result = run_eval(Settings(), pipeline=_RetrievalOnly())
    assert result["metrics"]["answer_accuracy"] == 0.0
    assert all(r["correct"] is False for r in result["rows"])


def test_write_report_creates_markdown(tmp_path):
    result = run_eval(Settings(), pipeline=_FakePipeline())
    out = write_report(result, tmp_path / "sub" / "report.md")

    assert out.exists() and out.parent.name == "sub"
    text = out.read_text(encoding="utf-8")
    assert "id 5" in text and "严格口径" in text
    assert "answer_accuracy" in text and "引用来源" in text

    en = write_report(run_eval(Settings(), pipeline=_FakePipeline(), use_english=True),
                      tmp_path / "en.md")
    assert "不适用" in en.read_text(encoding="utf-8")
