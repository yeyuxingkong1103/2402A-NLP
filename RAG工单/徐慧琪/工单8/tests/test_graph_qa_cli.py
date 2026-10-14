# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import json

import graph_qa_cli as cli

GT = [{"id": 1, "question": "问题一？", "reference": "答案一",
       "key_facts": ["a"], "qtype": "事实型", "origin": "wo07"},
      {"id": 2, "question": "问题二？", "reference": "答案二",
       "key_facts": ["b"], "qtype": "数值型", "origin": "wo08"}]


class _FakeResult:
    def __init__(self, q):
        self._d = {"question": q, "answer": f"答：{q}", "mode": "khop",
                   "lang": "zh", "strategy": "hybrid",
                   "graph_facts": ["事实"], "subgraph": {"nodes": [], "edges": []},
                   "text_hits": [], "timings": {"total": 1.2},
                   "degraded": False, "degraded_reason": ""}

    def to_dict(self):
        return dict(self._d)


def test_run_question_shape():
    got = cli.run_question(GT[0], ask_fn=lambda q, **kw: _FakeResult(q))
    assert got["id"] == 1
    assert got["question"] == "问题一？"
    assert got["answer"] == "答：问题一？"
    assert got["qtype"] == "事实型"
    assert got["origin"] == "wo07"
    assert "timings" in got


def test_run_all_writes_json(tmp_path):
    out = tmp_path / "res.json"
    res = cli.run_all(GT, out_path=str(out),
                      ask_fn=lambda q, **kw: _FakeResult(q), verbose=False)
    assert len(res) == 2
    back = json.loads(out.read_text(encoding="utf-8"))
    assert len(back) == 2
    assert back[0]["id"] == 1


def test_run_all_respects_limit(tmp_path):
    res = cli.run_all(GT, out_path=str(tmp_path / "r.json"), limit=1,
                      ask_fn=lambda q, **kw: _FakeResult(q), verbose=False)
    assert len(res) == 1


def test_run_all_continues_on_failure(tmp_path):
    """单题失败不中断整批（容错要求）。"""
    n = {"i": 0}

    def flaky(q, **kw):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("检索炸了")
        return _FakeResult(q)

    res = cli.run_all(GT, out_path=str(tmp_path / "r.json"),
                      ask_fn=flaky, verbose=False)
    assert len(res) == 2
    assert res[0]["error"] is not None
    assert res[1]["answer"]


def test_load_ground_truth():
    items = cli.load_ground_truth()
    assert len(items) >= 14
    assert items[0]["question"]
