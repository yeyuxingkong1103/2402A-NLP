# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-混合检索任务（工单6）

工单编号：人工智能NLP-RAG-混合检索任务

``app/core/retrieval_strategy.py`` 的**离线单元测试**（纯内存假 bundle，
不联网、不调 Ollama、不需要真实向量索引）：

1. ``StrategyConfig`` 默认值 / ``to_dict`` ↔ ``from_dict`` 往返 / 容错解析 / ``replace(None)``；
2. ``validate()`` 的每一条非法配置（模式、融合、重排、top_k、权重、两个阈值、字段名）；
3. ``PRESETS`` / ``preset()`` 与三个 ``describe_*`` 清单；
4. ``apply_rerank`` 四种重排算法（含 CrossEncoder 不可用时**显式降级**为 rule，打桩不联网）；
5. ``search_by_strategy`` 的 fulltext 模式端到端（假 bundle + 假 BM25）。

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_retrieval_strategy.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import retrieval_strategy as RS  # noqa: E402
from app.core.fulltext_query import FIELD_LABELS  # noqa: E402
from app.core.text_utils import tokenize  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

CORPUS: dict[str, dict[str, Any]] = {
    "m1": {"content": "公司来自军用领域的收入为 100 万元", "section": "军用领域",
           "keywords": ["军用", "收入"], "evidence_weight": 1.0},
    "m2": {"content": "军用产品毛利率为 40%", "section": "毛利率",
           "keywords": ["军用"], "evidence_weight": 1.0},
    "m3": {"content": "民用市场收入为 200 万元", "section": "民用市场",
           "keywords": ["民用"], "evidence_weight": 1.0},
}
ORDER: tuple[str, ...] = ("m1", "m2", "m3")

_TERM_SCORES: dict[str, dict[str, float]] = {
    "军用": {"m1": 3.0, "m2": 2.0},
    "领域": {"m1": 2.0},
    "收入": {"m1": 1.0, "m3": 1.0},
    "民用": {"m3": 2.0},
}


class _FakeInnerBM25:
    idf: dict[str, float] = {"军用": 2.0, "领域": 1.5, "收入": 1.2}

    def get_scores(self, terms: list[str]) -> list[float]:
        scores = [0.0] * len(ORDER)
        for term in terms:
            for cid, value in _TERM_SCORES.get(str(term), {}).items():
                scores[ORDER.index(cid)] += value
        return scores


@pytest.fixture()
def bundle() -> SimpleNamespace:
    return SimpleNamespace(
        index_dir="",
        order=list(ORDER),
        chunks=CORPUS,
        bm25=SimpleNamespace(
            doc_tokens=[tokenize(CORPUS[cid]["content"]) for cid in ORDER],
            bm25=_FakeInnerBM25(),
        ),
    )


def _candidates() -> list[dict[str, Any]]:
    """每次返回全新的候选副本（``apply_rerank`` 会原地写字段）。"""
    return [
        {"chunk_id": "a", "content": "军用领域收入 100 万元", "rrf_score": 0.8, "evidence_weight": 1.0},
        {"chunk_id": "b", "content": "民用市场收入", "rrf_score": 0.4, "evidence_weight": 1.0},
    ]


# ---------------------------------------------------------------------------
# ① StrategyConfig 默认值与序列化
# ---------------------------------------------------------------------------
def test_default_config_values() -> None:
    cfg = RS.StrategyConfig()
    assert cfg.mode == RS.MODE_HYBRID
    assert cfg.embedding_model == "bge-m3"
    assert (cfg.vector_k, cfg.fulltext_k, cfg.top_k) == (20, 20, 5)
    assert cfg.fulltext_fields == ("content",) == RS.DEFAULT_FIELDS
    assert cfg.fulltext_op == "OR"
    assert cfg.enable_fuzzy is True
    assert cfg.fusion == "weighted_average"
    assert (cfg.vector_weight, cfg.fulltext_weight) == (0.5, 0.5)
    assert cfg.rrf_k == 60
    assert cfg.rerank_model == RS.RERANK_RULE
    assert (cfg.score_threshold, cfg.vector_min_similarity) == (0.0, 0.0)
    assert cfg.use_image_channel is True
    assert cfg.validate() == []


def test_to_dict_from_dict_roundtrip_ignores_derived_keys() -> None:
    cfg = RS.StrategyConfig(mode=RS.MODE_FULLTEXT, fulltext_fields=("content", "title"),
                            top_k=7, fulltext_op="AND", enable_fuzzy=False, fusion="rrf")
    payload = cfg.to_dict()
    # 派生展示键：只读，不参与构造
    assert payload["mode_label"] == RS.MODE_LABELS[cfg.mode]
    assert payload["fusion_label"] == RS.ALGORITHM_LABELS[cfg.fusion]
    assert payload["rerank_label"] == RS.RERANK_LABELS[cfg.rerank_model]
    assert payload["fulltext_field_labels"] == ["正文", "标题"]
    assert RS.StrategyConfig.from_dict(payload) == cfg


def test_from_dict_parses_string_fulltext_fields() -> None:
    assert RS.StrategyConfig.from_dict({"fulltext_fields": "content,title"}).fulltext_fields \
        == ("content", "title")
    assert RS.StrategyConfig.from_dict({"fulltext_fields": "content，title"}).fulltext_fields \
        == ("content", "title"), "中文逗号也必须能解析（界面直接回传中文字段名）"
    assert RS.StrategyConfig.from_dict({"fulltext_fields": ["content", "summary"]}).fulltext_fields \
        == ("content", "summary")


def test_from_dict_ignores_unknown_and_none_values() -> None:
    cfg = RS.StrategyConfig.from_dict({"nope": 1, "mode_label": "伪造", "top_k": None, "mode": None})
    assert cfg == RS.StrategyConfig()
    assert RS.StrategyConfig.from_dict(None) == RS.StrategyConfig()
    assert RS.StrategyConfig.from_dict({}) == RS.StrategyConfig()


def test_from_dict_coerces_scalar_types() -> None:
    cfg = RS.StrategyConfig.from_dict({
        "mode": "fulltext", "top_k": "7", "vector_k": "9", "fulltext_k": 11, "rrf_k": "3",
        "vector_weight": "0.25", "score_threshold": "0.5",
        "enable_fuzzy": 0, "use_image_channel": 1,
    })
    assert (cfg.top_k, cfg.vector_k, cfg.fulltext_k, cfg.rrf_k) == (7, 9, 11, 3)
    assert isinstance(cfg.top_k, int) and isinstance(cfg.rrf_k, int)
    assert cfg.vector_weight == pytest.approx(0.25)
    assert cfg.score_threshold == pytest.approx(0.5)
    assert cfg.enable_fuzzy is False and cfg.use_image_channel is True
    assert cfg.mode == "fulltext"


def test_replace_ignores_none_values_and_keeps_original() -> None:
    cfg = RS.StrategyConfig()
    assert cfg.replace(mode=None, top_k=None, fulltext_fields=None) == cfg
    changed = cfg.replace(mode=None, top_k=3, fulltext_op=None)
    assert changed.mode == cfg.mode, "None 不得覆盖原值"
    assert changed.top_k == 3
    assert cfg.top_k == 5, "原配置是冻结的，replace 必须返回新对象"
    assert RS.StrategyConfig().replace() == cfg


@pytest.mark.parametrize("mode,marker", [
    (RS.MODE_VECTOR, "向量检索"),
    (RS.MODE_FULLTEXT, "全文检索"),
    (RS.MODE_HYBRID, "混合检索"),
])
def test_label_mentions_mode_specific_facts(mode: str, marker: str) -> None:
    label = RS.StrategyConfig(mode=mode).label()
    assert label.startswith(marker)
    assert RS.StrategyConfig(mode=mode, rerank_model="tfidf").label().endswith("重排=tfidf")
    if mode == RS.MODE_HYBRID:
        assert "weighted_average" in label
    if mode == RS.MODE_FULLTEXT:
        assert "正文" in label


# ---------------------------------------------------------------------------
# ② validate()：每条非法配置都要有一个用例
# ---------------------------------------------------------------------------
def test_validate_accepts_default_and_every_preset() -> None:
    assert RS.StrategyConfig().validate() == []
    for name, cfg in RS.PRESETS.items():
        assert cfg.validate() == [], f"内置预设 {name} 自身不合法：{cfg.validate()}"


@pytest.mark.parametrize("kwargs,marker", [
    ({"mode": "keyword"}, "未知检索模式"),
    ({"fusion": "bm25"}, "未知融合算法"),
    ({"rerank_model": "bge-reranker"}, "未知重排模型"),
    ({"top_k": 0}, "top_k 必须 ≥1"),
    ({"vector_k": 0}, "vector_k 必须 ≥1"),
    ({"fulltext_k": 0}, "fulltext_k 必须 ≥1"),
    ({"vector_weight": 0.0, "fulltext_weight": 0.0}, "不能同时为 0"),
    ({"vector_weight": -0.1}, "权重不得为负"),
    ({"fulltext_weight": -1.0}, "权重不得为负"),
    ({"score_threshold": 1.5}, "score_threshold 必须落在 [0,1]"),
    ({"score_threshold": -0.5}, "score_threshold 必须落在 [0,1]"),
    ({"vector_min_similarity": 1.5}, "vector_min_similarity 必须落在 [0,1]"),
    ({"vector_min_similarity": -0.2}, "vector_min_similarity 必须落在 [0,1]"),
    ({"fulltext_fields": ("body",)}, "未知全文检索字段"),
    ({"fulltext_fields": ()}, "全文检索字段不能为空"),
], ids=["模式", "融合", "重排", "top_k=0", "vector_k=0", "fulltext_k=0", "权重同为0",
        "向量权重负", "全文权重负", "阈值1.5", "阈值负", "相似度1.5", "相似度负",
        "未知字段", "字段为空"])
def test_validate_rejects_illegal_config(kwargs: dict[str, Any], marker: str) -> None:
    problems = RS.StrategyConfig(**kwargs).validate()
    assert problems, f"{kwargs} 应被判为非法配置"
    assert any(marker in problem for problem in problems), problems


def test_validate_reports_every_problem_not_just_the_first() -> None:
    problems = RS.StrategyConfig(mode="x", fusion="y", rerank_model="z", top_k=0).validate()
    assert len(problems) == 4
    assert any("未知检索模式" in p for p in problems)
    assert any("未知融合算法" in p for p in problems)
    assert any("未知重排模型" in p for p in problems)
    assert any("top_k" in p for p in problems)


# ---------------------------------------------------------------------------
# ③ 预设与清单
# ---------------------------------------------------------------------------
def test_presets_cover_three_modes_and_three_fusions() -> None:
    assert {"vector", "fulltext", "hybrid"} <= set(RS.PRESETS)
    for mode in RS.MODES:
        assert any(cfg.mode == mode for cfg in RS.PRESETS.values()), f"缺少 {mode} 模式预设"
    assert RS.preset("hybrid").fusion == "weighted_average"
    assert RS.preset("hybrid_rrf").fusion == "rrf"
    assert RS.preset("hybrid_voting").fusion == "voting"
    assert RS.preset("vector").mode == RS.MODE_VECTOR
    assert RS.preset("FULLTEXT").mode == RS.MODE_FULLTEXT, "预设名应大小写不敏感"


@pytest.mark.parametrize("bad", ["", "  ", "向量", "hybrid2", None],
                         ids=["空串", "空白", "中文名", "hybrid2", "None"])
def test_preset_unknown_raises_value_error(bad: object) -> None:
    with pytest.raises(ValueError):
        RS.preset(bad)  # type: ignore[arg-type]


def test_describe_modes_returns_valid_ids() -> None:
    described = RS.describe_modes()
    assert described, "模式清单不得为空（界面下拉框依赖它）"
    assert [item["id"] for item in described] == list(RS.MODES)
    assert all(item["label"] for item in described)


def test_describe_rerank_models_returns_valid_ids() -> None:
    described = RS.describe_rerank_models()
    assert described
    assert [item["id"] for item in described] == list(RS.RERANK_MODELS)
    assert all(item["label"] for item in described)


def test_describe_fulltext_fields_returns_valid_ids() -> None:
    described = RS.describe_fulltext_fields()
    assert described
    assert {item["id"] for item in described} == set(FIELD_LABELS)
    assert all(item["label"] and item["default_boost"] > 0 for item in described)
    assert RS.fulltext_syntax_help().strip(), "语法提示不得为空"


# ---------------------------------------------------------------------------
# ④ apply_rerank
# ---------------------------------------------------------------------------
def test_apply_rerank_empty_candidates_returns_empty_with_note() -> None:
    ranked, note = RS.apply_rerank("军用", [], RS.StrategyConfig(rerank_model="rule"))
    assert ranked == []
    assert note["requested"] == "rule" and note["applied"] == "rule"
    assert note["degraded"] is False


def test_apply_rerank_none_orders_by_fusion_score() -> None:
    candidates = _candidates()
    candidates[0]["rrf_score"], candidates[1]["rrf_score"] = 0.1, 0.9
    ranked, note = RS.apply_rerank("军用", candidates, RS.StrategyConfig(rerank_model="none"))
    assert [item["chunk_id"] for item in ranked] == ["b", "a"]
    assert ranked[0]["final_score"] == 1.0
    assert ranked[1]["final_score"] == pytest.approx(0.1 / 0.9, abs=1e-6)
    assert all(item["rerank"] == {"method": "none"} for item in ranked)
    assert note["applied"] == "none" and note["degraded"] is False


def test_apply_rerank_tfidf_scores_and_reorders() -> None:
    ranked, note = RS.apply_rerank("军用领域收入", _candidates(),
                                   RS.StrategyConfig(rerank_model="tfidf"))
    assert note["applied"] == "tfidf" and note["degraded"] is False
    assert [item["chunk_id"] for item in ranked] == ["a", "b"]
    for item in ranked:
        assert 0.0 <= item["tfidf"] <= 1.0
        assert item["rerank"]["method"] == "tfidf"
        assert item["final_score"] == pytest.approx(0.5 * item["rrf_norm"] + 0.5 * item["tfidf"],
                                                    abs=1e-6)
    assert ranked[0]["final_score"] > ranked[1]["final_score"]


def test_apply_rerank_rule_marks_method_and_sorts_desc() -> None:
    ranked, note = RS.apply_rerank("军用领域收入", _candidates(),
                                   RS.StrategyConfig(rerank_model="rule"))
    assert note == {"requested": "rule", "applied": "rule", "degraded": False, "reason": None}
    assert all(item["rerank"]["method"] == "rule" for item in ranked)
    scores = [item["final_score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["chunk_id"] == "a", "与问句更相关的块应排在前面"


def test_apply_rerank_cross_encoder_degrades_to_rule_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """CrossEncoder 不可用时必须**显式降级**为规则重排并写明原因（打桩，不联网）。"""
    monkeypatch.setattr(RS, "_load_cross_encoder", lambda logger=None: (None, "打桩：模型不可用"))
    ranked, note = RS.apply_rerank("军用领域收入", _candidates(),
                                   RS.StrategyConfig(rerank_model="cross_encoder"))
    assert note["requested"] == "cross_encoder"
    assert note["applied"] == RS.RERANK_RULE
    assert note["degraded"] is True
    assert "打桩：模型不可用" in note["reason"]
    assert ranked and all(item["rerank"]["method"] == RS.RERANK_RULE for item in ranked)


# ---------------------------------------------------------------------------
# ⑤ search_by_strategy（fulltext 模式端到端）
# ---------------------------------------------------------------------------
def _fulltext_strategy(**overrides: Any) -> RS.StrategyConfig:
    kwargs: dict[str, Any] = {"mode": RS.MODE_FULLTEXT, "rerank_model": RS.RERANK_NONE,
                              "fulltext_fields": ("content",), "fulltext_k": 10, "top_k": 3}
    kwargs.update(overrides)
    return RS.StrategyConfig(**kwargs)


def test_search_by_strategy_fulltext_end_to_end(bundle: SimpleNamespace) -> None:
    result = RS.search_by_strategy(bundle, "军用", _fulltext_strategy())
    assert [hit["chunk_id"] for hit in result["hits"]] == ["m1", "m2"]
    assert [hit["rank"] for hit in result["hits"]] == [1, 2]
    assert result["hits"][0]["score"] == 1.0
    assert result["vector_hits"] == [], "fulltext 模式不得触发向量通道"
    assert result["fusion"]["algorithm"] == "single_channel"
    assert result["fulltext"]["boolean_hits"] == 2
    assert result["fulltext"]["query_ast"] == "军用"
    assert result["strategy"]["mode"] == RS.MODE_FULLTEXT
    assert result["strategy_label"].startswith("全文检索")
    assert result["candidate_count"] == 2
    assert {"fulltext_ms", "fusion_ms", "rerank_ms", "total_ms"} <= set(result["timings"])


def test_search_by_strategy_negative_query_returns_no_hits_with_note(bundle: SimpleNamespace) -> None:
    result = RS.search_by_strategy(bundle, "NOT 军用", _fulltext_strategy())
    assert result["hits"] == []
    assert "全文通道 0 召回" in result["notes"]


def test_search_by_strategy_rejects_illegal_config(bundle: SimpleNamespace) -> None:
    with pytest.raises(ValueError) as excinfo:
        RS.search_by_strategy(bundle, "军用", _fulltext_strategy(top_k=0))
    assert "检索策略配置非法" in str(excinfo.value)
    with pytest.raises(ValueError):
        RS.search_by_strategy(bundle, "军用", _fulltext_strategy(vector_min_similarity=1.5))


def test_search_by_strategy_respects_top_k_and_unknown_field(bundle: SimpleNamespace) -> None:
    assert len(RS.search_by_strategy(bundle, "军用", _fulltext_strategy(top_k=1))["hits"]) == 1
    with pytest.raises(ValueError):
        RS.search_by_strategy(bundle, "军用", _fulltext_strategy(fulltext_fields=("body",)))
