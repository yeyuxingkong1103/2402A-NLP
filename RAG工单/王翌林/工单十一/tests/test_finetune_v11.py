# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Embedding模型微调任务
tests/test_finetune_v11.py —— 工单十一 微调数据集/评估器纯函数单测（不依赖 GPU/LLM/Milvus）
"""
import json

import pytest

from src.finetune_v11.qa_dataset import (build_qa_prompt, iter_chunks,
                                         make_pair, parse_questions,
                                         sample_chunks, split_train_dev)
from src.finetune_v11.ir_eval import (build_ir_inputs, compare_metrics,
                                      extract_metrics, pick_distractor_chunks)

WORK_ORDER = "人工智能NLP-RAG-Embedding模型微调任务"


# ---------- 问答对生成 ----------
def test_build_qa_prompt_truncates():
    msgs = build_qa_prompt("营收" * 2000, n_questions=3, max_chars=100)
    assert msgs[0]["role"] == "system"
    assert "3 个问题" in msgs[1]["content"]
    assert ("营收" * 40) in msgs[1]["content"] and ("营收" * 200) not in msgs[1]["content"]


@pytest.mark.parametrize("content,expected", [
    ('["平安银行2019年营业收入是多少？","净利润同比增长多少？"]', 2),
    ('```json\n["问题一：保费收入是多少？"]\n```', 1),
    ('1. 营业收入是多少亿元？\n2. 股息每股多少元？', 2),
    ('前导文字 ["问题：资产总额是多少？"] 后导', 1),
    ('', 0),
])
def test_parse_questions(content, expected):
    qs = parse_questions(content)
    assert len(qs) == expected
    assert all(len(q) >= 6 for q in qs)


def _mk_chunks():
    return [{"chunk_id": f"{d}_p{i}_c0", "doc_id": d, "page": i,
             "text": f"{d} 第{i}页 " + "内容" * 120}
            for d in ("A银行2019年报", "B保险2020年报") for i in range(1, 8)]


def test_iter_chunks_filters_short(tmp_path):
    data = {"doc_id": "测试年报", "chunks": [
        {"chunk_id": "c1", "page": 1, "text": "长文本" * 100},
        {"chunk_id": "c2", "page": 2, "text": "短"}]}
    (tmp_path / "测试年报_chunks.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")
    chunks = list(iter_chunks(tmp_path))
    assert len(chunks) == 1 and chunks[0]["chunk_id"] == "c1"


def test_sample_chunks_stratified_and_seeded():
    chunks = _mk_chunks()
    s1 = sample_chunks(chunks, per_doc=3, seed=7)
    s2 = sample_chunks(chunks, per_doc=3, seed=7)
    assert s1 == s2, "同种子应可复现"
    by_doc = {}
    for c in s1:
        by_doc.setdefault(c["doc_id"], 0)
        by_doc[c["doc_id"]] += 1
    assert all(v == 3 for v in by_doc.values()), "应按文档均匀分层"


def test_split_train_dev_no_chunk_leak():
    chunks = _mk_chunks()
    pairs = [make_pair(f"问题{j}：营收是多少？", c)
             for c in chunks for j in range(2)]
    train, dev = split_train_dev(pairs, dev_ratio=0.2, seed=42)
    train_ids = {p["chunk_id"] for p in train}
    dev_ids = {p["chunk_id"] for p in dev}
    assert not (train_ids & dev_ids), "同一 chunk 不得跨 train/dev"
    assert len(train) + len(dev) == len(pairs)


def test_make_pair_fields():
    p = make_pair("营收是多少？", _mk_chunks()[0])
    assert p["query"] and p["positive"] and p["work_order"] == WORK_ORDER


# ---------- 评估器构造 ----------
def test_build_ir_inputs_gen_and_gold():
    corpus = [{"chunk_id": "c1", "doc_id": "平安银行2019年报", "text": "t1"},
              {"chunk_id": "c2", "doc_id": "平安银行2019年报", "text": "t2"},
              {"chunk_id": "c3", "doc_id": "招商银行2019年报", "text": "t3"}]
    dev = [{"query": "营收？", "chunk_id": "c1"}]
    gold = [{"id": 1, "question": "董事长致辞提到哪些因素？",
             "gold_docs": ["平安银行2019年报"]}]
    queries, corpus_map, relevant = build_ir_inputs(dev, corpus, gold)
    assert len(corpus_map) == 3
    assert relevant["gen_0"] == {"c1"}
    assert relevant["v7_1"] == {"c1", "c2"}, "真实题为 doc 级相关"
    assert len(queries) == 2


def test_build_ir_inputs_drops_unrelated_query():
    dev = [{"query": "无相关？", "chunk_id": "不存在"}]
    queries, _, relevant = build_ir_inputs(dev, [], [])
    assert queries == {} and relevant == {}


def test_pick_distractors_excludes():
    chunks = _mk_chunks()
    picked = pick_distractor_chunks(chunks, exclude_ids={"A银行2019年报_p1_c0"},
                                    n=3, seed=1)
    assert len(picked) == 3
    assert all(c["chunk_id"] != "A银行2019年报_p1_c0" for c in picked)


def test_extract_and_compare_metrics():
    scores = {"fin_v11_mrr@10": 0.5, "fin_v11_ndcg@10": 0.6,
              "fin_v11_accuracy@1": 0.4, "other": "x"}
    m = extract_metrics(scores)
    assert m == {"mrr@10": 0.5, "ndcg@10": 0.6, "accuracy@1": 0.4}
    rows = compare_metrics({"mrr@10": 0.5, "ndcg@10": 0.6},
                           {"mrr@10": 0.55, "ndcg@10": 0.58})
    assert rows[0]["delta"] == pytest.approx(0.05)
    assert rows[1]["delta"] == pytest.approx(-0.02)
