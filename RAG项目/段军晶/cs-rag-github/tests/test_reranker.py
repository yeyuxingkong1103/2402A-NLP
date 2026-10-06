# -*- coding: utf-8 -*-
"""BGE-reranker 封装测试

纯逻辑测试用假的打分函数，不加载 2.3GB 模型；真实加载测试标记为 slow。
"""

from typing import Any, Dict, List

import pytest

from backend.reranker import BGEReranker, RerankerNotAvailableError


def _cand(cid: str, content: str, page: int = 1) -> Dict[str, Any]:
    return {
        "chunk_id": cid, "content": content, "page_no": page,
        "file_name": "x.pdf", "page_nums": [page], "content_type": "text",
        "section_title": "", "score": 0.5,
    }


class _FakeModel:
    """按预设分数表打分的假模型"""

    def __init__(self, table: Dict[str, float]) -> None:
        self._table = table

    def predict(self, pairs, **kwargs):
        return [self._table.get(text, 0.0) for _, text in pairs]


def test_按重排分数降序排列():
    r = BGEReranker()
    r._model = _FakeModel({"低": 0.1, "高": 0.9, "中": 0.5})
    out = r.rerank("q", [_cand("a", "低"), _cand("b", "高"), _cand("c", "中")])
    assert [c["chunk_id"] for c in out] == ["b", "c", "a"]


def test_保留原有字段并新增_rerank_score():
    r = BGEReranker()
    r._model = _FakeModel({"文本": 0.7})
    out = r.rerank("q", [_cand("a", "文本", page=9)])
    assert out[0]["page_no"] == 9
    assert out[0]["file_name"] == "x.pdf"
    assert out[0]["score"] == 0.5                      # 原 RRF 分数保留
    assert out[0]["rerank_score"] == pytest.approx(0.7)


def test_top_k_截断():
    r = BGEReranker()
    r._model = _FakeModel({"a": 0.9, "b": 0.8, "c": 0.7})
    out = r.rerank("q", [_cand("a", "a"), _cand("b", "b"), _cand("c", "c")], top_k=2)
    assert len(out) == 2


def test_空候选返回空列表():
    r = BGEReranker()
    r._model = _FakeModel({})
    assert r.rerank("q", []) == []


def test_候选内容为空时仍可打分不报错():
    r = BGEReranker()
    r._model = _FakeModel({"": 0.3})
    out = r.rerank("q", [_cand("a", "")])
    assert len(out) == 1


def test_模型不可用时报明确异常():
    r = BGEReranker(model_path="D:/不存在的路径")
    with pytest.raises(RerankerNotAvailableError):
        r.rerank("q", [_cand("a", "文本")])


def test_不修改入参且返回新对象():
    """契约：不修改调用方传入的字典，返回项必须是新对象"""
    import copy

    r = BGEReranker()
    r._model = _FakeModel({"文本": 0.7})
    cands = [_cand("a", "文本"), _cand("b", "文本")]
    snapshot = copy.deepcopy(cands)

    out = r.rerank("q", cands)

    assert cands == snapshot                 # 入参逐字段未被修改
    assert "rerank_score" not in cands[0]    # 未把新字段写回入参
    assert out[0] is not cands[0]            # 返回的是新对象而非入参本身


class _BoomModel:
    """推理时直接抛错的假模型"""

    def predict(self, pairs, **kwargs):
        raise RuntimeError("模拟推理崩溃")


def test_推理抛错时归一化为重排不可用异常():
    r = BGEReranker()
    r._model = _BoomModel()
    with pytest.raises(RerankerNotAvailableError):
        r.rerank("q", [_cand("a", "文本")])


class _SlowModel:
    """推理慢于超时阈值的假模型"""

    def predict(self, pairs, **kwargs):
        import time as _time

        _time.sleep(3)
        return [0.5] * len(pairs)


def test_推理超时抛重排不可用异常且不挂起():
    import time as _time

    r = BGEReranker(timeout=1)
    r._model = _SlowModel()

    t0 = _time.time()
    with pytest.raises(RerankerNotAvailableError):
        r.rerank("q", [_cand("a", "文本")])
    elapsed = _time.time() - t0

    # 关键：必须是「超时后立刻返回」而不是「等推理跑完再报错」。
    # 若用 with ThreadPoolExecutor(...) 的写法（退出时 shutdown(wait=True)），
    # 这里会等满 3 秒，elapsed 断言即失败。
    assert elapsed < 2.5, f"超时未生效，实际耗时 {elapsed:.2f}s"


def test_残留的推理线程必须是_daemon():
    """
    超时后遗留的推理线程必须是 daemon。

    非 daemon 线程会被 concurrent.futures.thread 在 atexit 注册的 _python_exit
    join，导致一次性进程（如 Task 9 的对照实验脚本）在超时后被拖住到推理自然
    结束，污染墙钟统计。pytest 报的秒数**不包含**进程退出时间，故必须在此断言
    线程属性本身。
    """
    import threading
    import time as _time

    before = set(threading.enumerate())

    r = BGEReranker(timeout=1)
    r._model = _SlowModel()
    t0 = _time.time()
    with pytest.raises(RerankerNotAvailableError):
        r.rerank("q", [_cand("a", "文本")])
    assert _time.time() - t0 < 2.5

    leaked = [t for t in set(threading.enumerate()) - before if t.is_alive()]
    assert leaked, "预期能看到尚未跑完的遗留推理线程"
    assert all(t.daemon for t in leaked), (
        f"遗留线程必须为 daemon，否则会阻塞进程退出："
        f"{[(t.name, t.daemon) for t in leaked]}"
    )


def test_线程启动失败时归一化为重排不可用异常(monkeypatch):
    """
    线程创建失败（如线程/FD 耗尽）时 RuntimeError 不得逃逸。

    它必须被归一化为 RerankerNotAvailableError —— 否则本模块「对外只抛一种
    异常」的契约在资源耗尽时被击穿。故 start() 必须留在 try 之内。
    """

    def _boom_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr("threading.Thread.start", _boom_start)

    r = BGEReranker()
    r._model = _FakeModel({"文本": 0.7})
    with pytest.raises(RerankerNotAvailableError):
        r.rerank("q", [_cand("a", "文本")])


def test_模型加载失败归一化为重排不可用异常(tmp_path):
    """目录存在但内容不是有效模型 —— 原始异常是 ValueError，必须被归一化"""
    broken = tmp_path / "broken_model"
    broken.mkdir()
    (broken / "config.json").write_text("{}", encoding="utf-8")

    r = BGEReranker(model_path=str(broken))
    with pytest.raises(RerankerNotAvailableError):
        r.rerank("q", [_cand("a", "文本")])


def test_模型路径为空串时归一化为重排不可用异常():
    """空串路径：Path("").exists() 为真，会径直落入 CrossEncoder("")"""
    r = BGEReranker()
    r.model_path = ""
    with pytest.raises(RerankerNotAvailableError):
        r.rerank("q", [_cand("a", "文本")])


@pytest.mark.slow
def test_端到端_真实模型能加载并给出合理排序():
    """真实加载 2.3GB 模型，验证封装可用"""
    from backend.reranker import get_reranker

    cands = [
        _cand("irrelevant", "今天天气不错，适合出门散步。"),
        _cand("relevant", "软件质量特性划分为功能性、可靠性、易用性、效率、维护性和可移植性。"),
    ]
    out = get_reranker().rerank("软件质量量化评价包含哪些质量特性？", cands)
    assert out[0]["chunk_id"] == "relevant"
    assert out[0]["rerank_score"] > out[1]["rerank_score"]
