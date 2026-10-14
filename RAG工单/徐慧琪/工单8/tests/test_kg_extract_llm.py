# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""第2层 LLM 抽取的单元测试（不依赖 Ollama：call_llm 全注入）。

两段式调用适配说明：qwen2.5:3b 单次同时抽实体+关系必退化截断（关系恒为 0），
故 extract_one 拆成「实体调用 + 关系调用」两次；stub 由单一 JSON 改为
按请求内容分发（关系调用返回 RELS_JSON、实体调用返回 ENTS_JSON），
注入点契约仍是 `messages -> str`，不依赖具体模型。
"""
import json

import pytest

from src import kg_extract_llm as kel

CHUNK = {"chunk_id": "c1",
         "text": "本公司控股股东为中国平安保险（集团）股份有限公司。",
         "source": "a.pdf", "page_idx": 42, "heading_path": ["公司治理"]}

ENTS_JSON = json.dumps({
    "entities": [
        {"name": "中国平安保险（集团）股份有限公司", "type": "Company",
         "aliases": ["中国平安"], "attrs": {}},
        {"name": "本公司", "type": "Company", "aliases": [], "attrs": {}},
    ]}, ensure_ascii=False)

RELS_JSON = json.dumps({
    "relations": [
        {"head": "中国平安保险（集团）股份有限公司", "type": "CONTROLS",
         "tail": "本公司", "confidence": 0.9,
         "evidence": "控股股东为中国平安保险（集团）股份有限公司"},
    ]}, ensure_ascii=False)


def _is_relations_call(msgs: list[dict]) -> bool:
    """关系调用的 system 消息含 "relations" 输出格式；实体调用只含 "entities"。"""
    return '"relations"' in msgs[0]["content"]


def _stub(ents: str = ENTS_JSON, rels: str = RELS_JSON):
    """两段式 stub：按请求分发，模拟模型对两类调用的响应。"""
    def call(msgs: list[dict]) -> str:
        return rels if _is_relations_call(msgs) else ents
    return call


@pytest.fixture(autouse=True)
def _tmp_ckpt(tmp_path, monkeypatch):
    monkeypatch.setattr(kel.config, "KG_LLM_CKPT",
                        str(tmp_path / "ckpt.jsonl"))
    monkeypatch.setattr(kel.config, "KG_LLM_RESULTS",
                        str(tmp_path / "results.jsonl"))
    kel._ckpt_cache.clear()
    yield


# ---------------------------------------------------------------------------
# extract_one：两段式抽取 + 闸门 + 重试
# ---------------------------------------------------------------------------

def test_extract_one_success():
    got = kel.extract_one(CHUNK, call_llm=_stub())
    assert got["error"] is None
    assert len(got["entities"]) == 2
    assert len(got["relations"]) == 1
    assert got["relations"][0].rel_type == "CONTROLS"


def test_extract_one_captures_error_without_raising():
    def boom(_msgs):
        raise RuntimeError("Ollama 挂了")
    got = kel.extract_one(CHUNK, call_llm=boom)
    assert got["error"] is not None
    assert got["entities"] == []


def test_extract_one_retries_then_succeeds():
    """首次实体调用抛错 → 整块重试；两段式下重试轮共 2+1=3 次调用。"""
    calls = {"n": 0}
    base = _stub()

    def flaky(msgs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("第一次失败")
        return base(msgs)

    got = kel.extract_one(CHUNK, call_llm=flaky)
    assert got["error"] is None
    assert calls["n"] == 3          # 第1轮仅到实体调用即失败；第2轮 实体+关系


def test_extract_one_handles_bad_json():
    got = kel.extract_one(CHUNK, call_llm=lambda _m: "抱歉，我无法回答")
    assert got["error"] is not None


def test_relations_call_receives_accepted_entity_names():
    """关系调用必须拿到①已接受的实体名，端点检查才有东西可匹配。"""
    seen = {"msgs": None}
    base = _stub()

    def spy(msgs):
        if _is_relations_call(msgs):
            seen["msgs"] = msgs
        return base(msgs)

    kel.extract_one(CHUNK, call_llm=spy)
    assert seen["msgs"] is not None
    content = seen["msgs"][-1]["content"]
    assert "中国平安保险（集团）股份有限公司" in content
    assert "本公司" in content


def test_relations_gate_drops_fabricated_evidence_and_unknown_endpoint():
    """闸门3（证据落地）与端点检查在关系调用上照常生效。"""
    bad = json.dumps({"relations": [
        {"head": "中国平安保险（集团）股份有限公司", "type": "CONTROLS",
         "tail": "本公司", "confidence": 0.9, "evidence": "编造的原文证据"},
        {"head": "不存在的公司", "type": "CONTROLS", "tail": "本公司",
         "confidence": 0.9, "evidence": "控股股东为中国平安保险（集团）股份有限公司"},
    ]}, ensure_ascii=False)
    got = kel.extract_one(CHUNK, call_llm=_stub(rels=bad))
    assert got["error"] is None
    assert got["relations"] == []
    assert len(got["entities"]) == 2          # 实体不受关系闸门影响


def test_relations_call_entities_are_ignored():
    """关系调用若又输出实体，一律忽略，防止重复计入。"""
    rels_with_ents = json.dumps({
        "entities": [{"name": "多出来的公司", "type": "Company",
                      "aliases": [], "attrs": {}}],
        "relations": [{"head": "中国平安保险（集团）股份有限公司",
                       "type": "CONTROLS", "tail": "本公司", "confidence": 0.9,
                       "evidence": "控股股东为中国平安保险（集团）股份有限公司"}]},
        ensure_ascii=False)
    got = kel.extract_one(CHUNK, call_llm=_stub(rels=rels_with_ents))
    assert [e.name for e in got["entities"]] == [
        "中国平安保险（集团）股份有限公司", "本公司"]
    assert len(got["relations"]) == 1


def test_no_entities_skips_relations_call():
    """实体调用无有效实体 → 跳过关系调用（端点必匹配不上，省一次 LLM）。"""
    calls = {"n": 0}

    def spy(msgs):
        calls["n"] += 1
        return json.dumps({"entities": []}, ensure_ascii=False)

    got = kel.extract_one(CHUNK, call_llm=spy)
    assert calls["n"] == 1
    assert got["entities"] == [] and got["relations"] == []


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip():
    kel.append_checkpoint({"chunk_id": "c1", "n_entities": 2, "n_relations": 1})
    kel.append_checkpoint({"chunk_id": "c2", "n_entities": 0, "n_relations": 0})
    done = kel.load_checkpoint()
    assert done == {"c1", "c2"}


# ---------------------------------------------------------------------------
# 结果持久化：load_results 往返 + resume 返回累积结果（工单08修正）
# ---------------------------------------------------------------------------

def test_load_results_missing_file_returns_empty():
    assert kel.load_results() == []


def test_results_roundtrip_extract_to_load_results():
    """extract_llm 成功块落盘 → load_results 重建的对象与原件逐字段一致。"""
    kel.extract_llm([{"chunk": CHUNK, "score": 10, "reasons": []}],
                    resume=False, call_llm=_stub(), verbose=False)
    recs = kel.load_results()
    assert len(recs) == 1
    assert recs[0]["chunk_id"] == "c1"

    ents = recs[0]["entities"]
    assert [e.entity_id for e in ents] == [
        "Company:中国平安保险（集团）股份有限公司", "Company:本公司"]
    assert [e.name for e in ents] == [
        "中国平安保险（集团）股份有限公司", "本公司"]
    assert [e.type for e in ents] == ["Company", "Company"]
    assert ents[0].aliases == ["中国平安"]
    assert ents[0].attrs["page"] == 42 and ents[0].attrs["source_doc"] == "a.pdf"

    rels = recs[0]["relations"]
    assert len(rels) == 1
    r = rels[0]
    assert r.head_id == "Company:中国平安保险（集团）股份有限公司"
    assert r.rel_type == "CONTROLS"
    assert r.tail_id == "Company:本公司"
    assert r.evidence == "控股股东为中国平安保险（集团）股份有限公司"
    assert r.source_doc == "a.pdf" and r.page == 42
    assert r.confidence == 0.9


def test_failed_block_writes_no_results_record():
    """容错隔离：失败块只进计数 checkpoint，不写结果文件。"""
    def boom(_msgs):
        raise RuntimeError("Ollama 挂了")
    out = kel.extract_llm([{"chunk": CHUNK, "score": 10, "reasons": []}],
                          resume=False, call_llm=boom, verbose=False)
    assert out["n_fail"] == 1
    assert kel.load_results() == []
    assert kel.load_checkpoint() == {"c1"}       # 失败也记完成索引（沿用旧语义）


def test_resume_returns_persisted_results_not_just_new_ones():
    """核心修正：resume 跳过已完成块时，其产物从结果文件读回并计入返回值。"""
    sel = [{"chunk": dict(CHUNK, chunk_id=f"c{i}"), "score": 10, "reasons": []}
           for i in range(3)]
    first = kel.extract_llm(sel, resume=False, call_llm=_stub(), verbose=False)
    assert first["n_ok"] == 3

    def must_not_call(_msgs):
        raise AssertionError("已完成块不应再调 LLM")

    second = kel.extract_llm(sel, resume=True, call_llm=must_not_call,
                             verbose=False)
    assert second["n_skipped"] == 3 and second["n_ok"] == 0
    assert second["n_skipped_no_results"] == 0
    # 返回的是全量累积结果（跨进程可并图），而非空集
    assert [(e.entity_id, e.type) for e in second["entities"]] == \
        [(e.entity_id, e.type) for e in first["entities"]]
    assert [r.to_dict() for r in second["relations"]] == \
        [r.to_dict() for r in first["relations"]]
    assert len(second["entities"]) == 6 and len(second["relations"]) == 3


def test_resume_with_legacy_checkpoint_without_results_skips_without_crash():
    """向后兼容：旧 checkpoint 有行、无结果文件 → 跳过且返回空，不抛异常。"""
    kel.append_checkpoint({"chunk_id": "c1", "n_entities": 2, "n_relations": 1})

    def must_not_call(_msgs):
        raise AssertionError("已完成块不应再调 LLM")

    out = kel.extract_llm([{"chunk": CHUNK, "score": 10, "reasons": []}],
                          resume=True, call_llm=must_not_call, verbose=False)
    assert out["n_skipped"] == 1
    assert out["n_skipped_no_results"] == 1
    assert out["entities"] == [] and out["relations"] == []


def test_load_results_last_record_wins_and_survives_bad_lines():
    """结果先写、checkpoint 后写：两步间崩溃 → 重抽再追加，后写者胜。"""
    kel.append_results({"chunk_id": "c1", "entities": [], "relations": []})
    with open(kel.config.KG_LLM_RESULTS, "a", encoding="utf-8") as f:
        f.write("这不是 JSON\n")
    kel.append_results(
        kel._results_record("c1", *kel._extract_two_calls(CHUNK, _stub())))
    recs = kel.load_results()
    assert len(recs) == 1
    assert len(recs[0]["entities"]) == 2 and len(recs[0]["relations"]) == 1


def test_load_results_nonexistent_path_returns_empty(tmp_path):
    assert kel.load_results(path=str(tmp_path / "没有这个文件.jsonl")) == []


# ---------------------------------------------------------------------------
# extract_llm：批量、断点续抽、限额、容错
# ---------------------------------------------------------------------------

def test_extract_llm_skips_done_chunks():
    kel.append_checkpoint({"chunk_id": "c1", "n_entities": 0, "n_relations": 0})
    called = {"n": 0}

    def spy(_m):
        called["n"] += 1
        return ENTS_JSON

    out = kel.extract_llm([{"chunk": CHUNK, "score": 10, "reasons": []}],
                          resume=True, call_llm=spy, verbose=False)
    assert called["n"] == 0
    assert out["n_skipped"] == 1


def test_extract_llm_accumulates():
    out = kel.extract_llm([{"chunk": CHUNK, "score": 10, "reasons": []}],
                          resume=False, call_llm=_stub(), verbose=False)
    assert out["n_ok"] == 1
    assert len(out["entities"]) == 2
    assert len(out["relations"]) == 1


def test_extract_llm_respects_limit():
    sel = [{"chunk": dict(CHUNK, chunk_id=f"c{i}"), "score": 10, "reasons": []}
           for i in range(10)]
    out = kel.extract_llm(sel, resume=False, limit=3,
                          call_llm=_stub(), verbose=False)
    assert out["n_ok"] == 3


def test_extract_llm_continues_after_failure():
    """单块失败不应中断整批（容错要求）：c1 持续失败，其余块照常。"""
    base = _stub()

    def sometimes(msgs):
        if "H1" in msgs[-1]["content"]:
            raise RuntimeError("c1 持续失败")
        return base(msgs)

    sel = [{"chunk": dict(CHUNK, chunk_id=f"c{i}", heading_path=[f"H{i}"]),
            "score": 10, "reasons": []} for i in range(3)]
    out = kel.extract_llm(sel, resume=False, call_llm=sometimes, verbose=False)
    assert out["n_ok"] == 2
    assert out["n_fail"] == 1
