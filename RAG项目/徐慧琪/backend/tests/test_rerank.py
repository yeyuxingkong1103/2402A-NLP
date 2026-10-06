# 精排的单测用替身 scorer 与替身模块验排序与构造参数（不加载 2.2GB 模型）；
# 真模型只在最后一条集成测里出场，且必须有明确的高低于阈值断言——
# 只断言"能返回结果"的测试永远不会失败（项目已有八次前车之鉴）。
import sys
import types

import pytest

from app.retrieval.rerank import (
    RERANK_INPUT_TOPK, RERANK_OUTPUT_TOPK, RERANK_MODEL_PATH,
    load_reranker, pairs_for, top_blocks,
)

BLOCKS = [
    {"article_no": 1, "text": "第一条 为了保护民事主体的合法权益……"},
    {"article_no": 2, "text": "第二条 民法调整平等主体的自然人……"},
    {"article_no": 3, "text": "第三条 民事主体的人身权利……"},
]


def test_params_match_spec():
    # 技术方案 5.2 定的起始值，改它等于改行为且没有评估依据
    assert RERANK_INPUT_TOPK == 30
    assert RERANK_OUTPUT_TOPK == 5


def test_pairs_for_pairs_query_with_each_text():
    assert pairs_for("问题", BLOCKS) == [("问题", b["text"]) for b in BLOCKS]


def test_top_blocks_sorts_by_score_descending():
    # 替身 scorer 给"条号越大分越高"，断言排序而不是断言具体数值。
    # 按正文查表而不是解析正文——正文里的条号是中文数字，解析它等于在测 cn_num
    scores = {block["text"]: float(block["article_no"]) for block in BLOCKS}

    def scorer(pairs):
        return [scores[text] for _, text in pairs]

    ranked = top_blocks("问题", BLOCKS, scorer)
    assert [b["article_no"] for b in ranked] == [3, 2, 1]
    assert ranked[0]["rerank_score"] == 3.0


def test_top_blocks_truncates_to_output_topk():
    def scorer(pairs):
        return [1.0] * len(pairs)

    # 原文是 dict(b, ...)，但 b 在此作用域未定义（brief 笔误），按语义补成独立块
    many = [{"article_no": i, "text": f"第{i}条 x"} for i in range(20)]
    assert len(top_blocks("问题", many, scorer, output_topk=5)) == 5


def test_top_blocks_limits_scorer_input_to_input_topk():
    seen = {}

    def scorer(pairs):
        seen["n"] = len(pairs)
        return [0.5] * len(pairs)

    many = [{"article_no": i, "text": f"第{i}条 x"} for i in range(50)]
    top_blocks("问题", many, scorer, input_topk=30, output_topk=5)
    assert seen["n"] == 30, "精排输入必须截到 30，否则耗时随召回数线性上涨"


def test_top_blocks_does_not_mutate_input_blocks():
    # 上游可能还要用原顺序的 blocks 做别的（如精确块合并），原地改会埋雷
    def scorer(pairs):
        return [1.0] * len(pairs)

    before = [dict(b) for b in BLOCKS]
    top_blocks("问题", BLOCKS, scorer)
    assert BLOCKS == before


def test_top_blocks_on_empty_input_returns_empty():
    assert top_blocks("问题", [], lambda pairs: []) == []


def test_top_blocks_raises_when_scorer_returns_wrong_count():
    # zip 按短的一侧静默截断：少给分 = 块被悄悄丢掉且不报错，坏得无声无息；
    # 多给分则分数与块错位被丢，同样掩盖 scorer 的 bug，故两个方向都钉死
    def short_scorer(pairs):
        return [1.0] * (len(pairs) - 1)

    def long_scorer(pairs):
        return [1.0] * (len(pairs) + 1)

    with pytest.raises(ValueError, match="分数条数不符"):
        top_blocks("问题", BLOCKS, short_scorer)
    with pytest.raises(ValueError, match="分数条数不符"):
        top_blocks("问题", BLOCKS, long_scorer)


def test_model_path_points_at_deployed_dir():
    assert RERANK_MODEL_PATH == r"D:\Model\reranker"


def test_load_reranker_defaults_to_deployed_model_on_cuda(monkeypatch):
    # 全局约束"精排必须跑在 GPU"是硬要求（CPU 实测 30 条 3.43s，跑不赢 AC-12），
    # 而真模型集成测显式传了 device="cpu"——默认值被改成 cpu 原本没有任何测试会红
    captured = {}
    fake_module = types.ModuleType("sentence_transformers")

    def fake_cross_encoder(model_name_or_path, max_length=None, device=None):
        # 捕获构造参数就返回替身，真身会拖起 torch 与 2.2GB 权重
        captured.update(model_path=model_name_or_path,
                        max_length=max_length, device=device)
        return "替身精排模型"

    fake_module.CrossEncoder = fake_cross_encoder
    # `from sentence_transformers import CrossEncoder` 先查 sys.modules，
    # 塞进假模块即可拦截；monkeypatch 保证测后就地复原，不污染后续测试
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    assert load_reranker() == "替身精排模型"
    assert captured["model_path"] == RERANK_MODEL_PATH
    assert captured["device"] == "cuda"


@pytest.mark.skipif(not __import__("pathlib").Path(RERANK_MODEL_PATH).exists(),
                    reason="精排模型未部署")
def test_real_model_ranks_relevant_article_above_irrelevant():
    """真模型集成测：相关条必须显著高于无关条。

    阈值取 0.5：实测相关 0.809/0.771、无关 0.000，取中间值既容得下不同
    问法的波动，又能在模型加载错（如随机权重）时立刻变红。
    """
    model = load_reranker(device="cpu")
    query = "借款到期不还，我能要求对方支付逾期利息吗"
    blocks = [
        {"article_no": 676, "text": "第六百七十六条 借款人未按照约定的期限返还借款的，"
                                    "应当按照约定或者国家有关规定支付逾期利息。"},
        {"article_no": 1042, "text": "第一千零四十二条 禁止包办、买卖婚姻和其他干涉婚姻自由的行为。"},
    ]
    ranked = top_blocks(query, blocks, model.predict, output_topk=2)
    assert ranked[0]["article_no"] == 676
    assert ranked[0]["rerank_score"] > 0.5
    assert ranked[1]["rerank_score"] < 0.5
