"""法律术语同义扩写测试（批次 13）。

约束（用户裁决，逐条对应）：
1. 术语表落盘 data/，代码只加载不硬编码
2. 只在关键词路做「口语 → 法条用语」单向扩展，原词保留
3. 配置开关默认关
4. 易混术语对禁止交叉扩展（经济补偿↔赔偿金、解除↔终止、N 与 2N 精确匹配）
5. 负向规则：非劳动争议词不参与扩展
"""

from pathlib import Path

import pytest

from app.retrieval.keyword_search import build_keyword_searcher_rows
from app.retrieval.synonym_expansion import (
    DEFAULT_TABLE_PATH,
    SynonymExpander,
    load_synonym_table,
)


@pytest.fixture(scope="module")
def expander() -> SynonymExpander:
    return SynonymExpander(DEFAULT_TABLE_PATH)


# --------------------------------------------------------------------------
# 术语表数据文件
# --------------------------------------------------------------------------


def test_table_file_is_versioned_and_complete() -> None:
    """术语表必须有版本号，每组带标准术语/替代表达/条文出处。"""
    table = load_synonym_table(DEFAULT_TABLE_PATH)

    assert table.version
    assert len(table.groups) >= 20
    for group in table.groups:
        assert group.standard, f"第 {group.id} 组缺标准术语"
        assert group.alternatives, f"第 {group.id} 组缺替代表达"
        assert group.source, f"第 {group.id} 组缺条文出处"


def test_table_file_lives_under_data_directory() -> None:
    """术语表放 data/，不进 backend/。"""
    path = Path(DEFAULT_TABLE_PATH)
    assert "data" in path.parts
    assert "backend" not in path.parts


def test_negative_rules_section_present(expander: SynonymExpander) -> None:
    """负向规则段必须存在，且列出非劳动争议词。"""
    assert expander.negative_keywords
    assert "月饼" in expander.negative_keywords


# --------------------------------------------------------------------------
# 触发与追加
# --------------------------------------------------------------------------


def test_expands_colloquial_phrase_to_statute_terms(expander: SynonymExpander) -> None:
    """口语「返聘」→ 法条用语「劳务关系」。"""
    phrases = expander.expand("退休以后被公司返聘，被辞退有补偿吗")

    assert "劳务关系" in phrases
    assert "依法享受养老保险待遇" in phrases


def test_expands_social_insurance_colloquial(expander: SynonymExpander) -> None:
    """口语「不缴社保」→ 法条用语「未依法为劳动者缴纳社会保险费」。"""
    phrases = expander.expand("公司让我签协议约定不缴社保合法吗")

    assert "未依法为劳动者缴纳社会保险费" in phrases


def test_no_expansion_when_query_already_uses_statute_term(
    expander: SynonymExpander,
) -> None:
    """原词保留：标准术语已在 query 里就不再追加同一词条。"""
    assert "二倍工资" not in expander.expand("二倍工资怎么算")
    assert "经济补偿" not in expander.expand("经济补偿金怎么算")


# --------------------------------------------------------------------------
# 易混术语对：禁止交叉扩展
# --------------------------------------------------------------------------


def test_confusable_pair_blocks_cross_expansion(expander: SynonymExpander) -> None:
    """同时提到赔偿金与补偿金时，两边都不扩展（避免混淆）。"""
    phrases = expander.expand("赔偿金和补偿金能同时要吗")

    assert "经济补偿" not in phrases
    assert "赔偿金" not in phrases


def test_confusable_pair_blocks_counterpart(expander: SynonymExpander) -> None:
    """query 里已有「赔偿金」时，不得因「补偿金」再扩出「经济补偿」。"""
    phrases = expander.expand("被辞退后公司只给了赔偿金，还能要补偿金吗")

    assert "经济补偿" not in phrases


def test_confusable_pair_blocks_termination_cross(expander: SynonymExpander) -> None:
    """解除 ↔ 终止 不交叉：提到「解除劳动合同」时不扩出「终止劳动合同」。"""
    phrases = expander.expand("合同到期不续签和解除劳动合同有什么区别")

    assert "终止劳动合同" not in phrases
    assert "解除劳动合同" not in phrases


def test_ascii_n_requires_word_boundary(expander: SynonymExpander) -> None:
    """N 与 2N 精确匹配：「2N」不触发经济补偿。"""
    assert "经济补偿" not in expander.expand("违法辞退能要2N吗")
    # 单独的 N 仍然触发
    assert "经济补偿" in expander.expand("协商解除拿 N 合理吗")


# --------------------------------------------------------------------------
# 负向规则
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "公司食堂的菜太难吃了，法律管吗",
        "中秋节不发月饼算违法吗",
        "邻居装修太吵能找劳动法吗",
    ],
)
def test_negative_rule_skips_non_labor_query(
    expander: SynonymExpander, question: str
) -> None:
    """非劳动争议词不参与扩展，保护拒答题。"""
    assert expander.expand(question) == []


# --------------------------------------------------------------------------
# 接入关键词检索
# --------------------------------------------------------------------------


def _build_searcher(*, with_expander: bool):
    rows = [
        ("law-a48", "用人单位违反本法规定解除劳动合同，应当支付赔偿金", "第四十八条"),
        ("law-a47", "经济补偿按劳动者在本单位工作的年限计算", "第四十七条"),
        ("noise-1", "劳动合同订立应当遵循合法公平原则", None),
        ("noise-2", "社会保险费征缴管理", None),
        ("noise-3", "工伤认定申请程序", None),
    ]
    expander = SynonymExpander(DEFAULT_TABLE_PATH) if with_expander else None
    return build_keyword_searcher_rows(rows, query_expander=expander)


def test_keyword_searcher_appends_expansion_tokens() -> None:
    """口语题借助扩展命中法条用语分块；原词仍在查询里。"""
    hits = _build_searcher(with_expander=True).search("被老板炒鱿鱼了怎么办")

    assert "law-a48" in [hit.chunk_key for hit in hits]


def test_keyword_searcher_without_expander_keeps_original_behavior() -> None:
    """不传扩展器时行为不变：口语题召不到法条用语分块。"""
    hits = _build_searcher(with_expander=False).search("被老板炒鱿鱼了怎么办")

    assert hits == []


def test_keyword_searcher_still_matches_original_terms_with_expander() -> None:
    """扩展不影响原词命中：原词该中的还中。"""
    hits = _build_searcher(with_expander=True).search("经济补偿按年限计算")

    assert "law-a47" in [hit.chunk_key for hit in hits]


# --------------------------------------------------------------------------
# 配置开关（默认关，可回退）
# --------------------------------------------------------------------------


def test_switch_defaults_to_disabled() -> None:
    """代码默认值必须是关：不配置就不改变现有行为。"""
    from app.core.config import Settings

    assert Settings.synonym_expansion_enabled is False


def test_switch_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量可开关，便于灰度与一键回退。"""
    from app.core.config import Settings

    monkeypatch.setenv("SYNONYM_EXPANSION_ENABLED", "true")
    assert Settings.from_environment().synonym_expansion_enabled is True
    monkeypatch.setenv("SYNONYM_EXPANSION_ENABLED", "false")
    assert Settings.from_environment().synonym_expansion_enabled is False


def test_assembly_builds_expander_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """装配层：开关关 → None（关键词路行为不变）；开关开 → 加载版本化术语表。"""
    from dataclasses import replace

    from app.core.config import settings as app_settings
    from app.retrieval import assembly

    monkeypatch.setattr(
        assembly, "settings", replace(app_settings, synonym_expansion_enabled=False)
    )
    assert assembly.build_query_expander_from_settings() is None

    monkeypatch.setattr(
        assembly, "settings", replace(app_settings, synonym_expansion_enabled=True)
    )
    expander = assembly.build_query_expander_from_settings()
    assert expander is not None
    assert expander.version == "v1.1"  # 批次 13-收尾：泛词收窄后版本升级


def test_assembly_falls_back_when_table_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """开关开着但表读不到时退回不扩展，不让检索整体失败。"""
    from dataclasses import replace

    from app.core.config import settings as app_settings
    from app.retrieval import assembly

    monkeypatch.setattr(
        assembly,
        "settings",
        replace(
            app_settings,
            synonym_expansion_enabled=True,
            synonym_table_path=str(tmp_path / "not_exist.json"),
        ),
    )
    assert assembly.build_query_expander_from_settings() is None


def test_expansion_tokens_scored_at_reduced_weight() -> None:
    """批次 13-收尾：扩展词在 BM25 降权（0.5），只命中扩展词的分块分数低于原词命中的分块，
    避免泛化扩展词在召回里挤占原词名额。"""
    # 附带噪声文档：rank_bm25 的 IDF 在 2 文档语料下（n=1, N=2）恒为 0，需 N>=3
    rows = [
        ("chunk-layoff", "试用期辞退有专门规定", None),
        ("chunk-dismiss", "用人单位与劳动者解除劳动合同的程序", None),
        ("noise-x", "劳动报酬应当按照合同约定足额支付", None),
    ]
    searcher = build_keyword_searcher_rows(
        rows, query_expander=SynonymExpander(DEFAULT_TABLE_PATH)
    )

    # 「辞退」是原词，"解除劳动合同"是扩展词：两块都命中，但扩展词-only 块拿 0.5 折分数
    hits = {hit.chunk_key: hit.score for hit in searcher.search("被辞退了怎么办")}

    assert "chunk-layoff" in hits
    assert "chunk-dismiss" in hits
    assert hits["chunk-dismiss"] < hits["chunk-layoff"]


def test_expansion_weight_zero_disables_expansion_scoring() -> None:
    """扩展词权重为 0 时，只命中扩展词的分块不得分（等效于关闭扩展）。"""
    rows = [("chunk-dismiss", "用人单位与劳动者解除劳动合同的程序", None)]
    searcher = build_keyword_searcher_rows(
        rows,
        query_expander=SynonymExpander(DEFAULT_TABLE_PATH),
        expansion_weight=0.0,
    )

    assert searcher.search("被辞退了怎么办") == []
