# 附加区块的**真依赖装配**（app/recommend/extras.py）。这两条用例原先住在
# tools/tests/test_ask.py 里 —— 装配搬进生产包时跟着搬过来：它们钉的不是 CLI 的
# 参数或渲染，而是「装配把哪个对象接到了哪条链路上」，归属由被测代码决定。
# 本模块的三条存疑都来自真跑（Task 9 报告存疑 9：search_fn 的空串回退、
# 精排有没有接上、客户端是不是每问一次现建）。
from unittest.mock import MagicMock

from app.generation.profiles import SIDE_PUBLIC
from app.recommend.extras import build_extras_fn
from app.recommend.fee_search import FEE_COLLECTION


class _FakeEncoder:
    """替身 encoder：固定形状的假向量，并记下被编码的文本。"""

    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, batch_size=None, return_dense=True,
               return_sparse=True, return_colbert_vecs=False):
        self.seen += list(texts)
        return {"dense_vecs": [[0.0] * 1023 + [1.0]] * len(texts),
                "lexical_weights": [{1: 0.5}] * len(texts)}


class _FakeMilvus:
    """替身 Milvus：记下检索请求，返回预置命中行（默认空 → 走到 no_corpus 即止）。"""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls: list[tuple] = []

    def hybrid_search(self, collection, reqs, ranker=None, limit=None,
                      output_fields=None):
        self.calls.append((collection, [req.anns_field for req in reqs]))
        return [[{"entity": dict(row)} for row in self.rows]]


class _FakeLLM:
    """替身 LLM：invoke 返回固定 JSON（generate_fee 只读 content / response_metadata）。"""

    model_name = "fake"

    def invoke(self, messages):
        from types import SimpleNamespace
        return SimpleNamespace(content='{"low": 1, "high": 10}',
                               response_metadata={})


class _FakeReranker:
    """替身精排：给「乙所」那段高分（CrossEncoder.predict 的形状：list[float]）。"""

    def predict(self, pairs):
        return [0.1 if "甲所" in text else 0.9 for _, text in pairs]


def test_build_extras_fn_searches_with_the_question_when_cause_is_unknown(monkeypatch):
    """案由认不出时，编码器必须看到**用户问句**，而不是空串。

    只断言「没抛异常」或「fee 是 no_corpus」都拦不住这条回归：回退去掉后
    search 收到空串会提前返回 []，结果同样是 no_corpus —— 假故障被答成
    「确实没有」。故断言点放在旁路（编码器看到了什么、检索有没有发出去）。
    """
    # get_llm 在 build_extras_fn 里被调用一次（客户端构造一次、复用）：
    # 本用例只走到检索（无命中 → no_corpus → 不问模型），故塞个占位对象，
    # 顺带把「构造时不读密钥」这件事钉在用例之外（真密钥由 CLI 冒烟覆盖）
    monkeypatch.setattr("app.generation.llm_router.get_llm", lambda side: object())
    conn, encoder, client = MagicMock(), _FakeEncoder(), _FakeMilvus()
    extras_fn = build_extras_fn(conn, client, encoder)
    block = extras_fn("今天天气怎么样")
    assert block["cause"] is None
    assert encoder.seen == ["今天天气怎么样"]
    assert client.calls == [(FEE_COLLECTION, ["dense", "sparse"])]
    assert block["fee"]["status"] == "no_corpus"


def test_build_extras_fn_hands_the_reranker_to_the_fee_search(monkeypatch):
    """技术方案 6.4 要求费用检索也是「混合检索 **+ 精排**」：build_extras_fn 必须把
    reranker.predict 交给 fee_search.search —— 不接的话 top1 由 RRF 位置融合决定，
    正是终审点名的「取错片段」上游。

    断言点放在**重排后的结果**（basis 指回哪一段）而不是「参数传没传」：只断言调用
    形态时，search 内部把 scorer 丢掉、或精排后仍返回原顺序，都照样绿。两条命中行
    只有片段文本不同，故 basis 直接指出 top1 是哪一段。
    """
    rows = [{"text": "甲所口径：每件 1 元至 5 元", "source_doc": "甲所",
             "source_no": "一、", "status": "现行有效"},
            {"text": "乙所口径：每件 1 元至 10 元", "source_doc": "乙所",
             "source_no": "二、", "status": "现行有效"}]
    monkeypatch.setattr("app.generation.llm_router.get_llm", lambda side: _FakeLLM())
    extras_fn = build_extras_fn(MagicMock(), _FakeMilvus(rows), _FakeEncoder(),
                                _FakeReranker())
    block = extras_fn("建房纠纷")
    assert block["fee"]["status"] == "ok"
    assert block["fee"]["basis"] == rows[1]["text"], "精排后的 top1 才是生成与依据的那一段"
    assert block["fee"]["source_no"] == "二、"


def test_build_extras_fn_constructs_the_llm_client_once(monkeypatch):
    """客户端在**装配时**构造一次、被所有问答复用（与 chain.build_chain 的「链只
    构造一次」同口径，CLI 与 HTTP 都靠这条不浪费）。

    把 get_llm 挪进返回的闭包不会报错，只会让每次问答多一次客户端初始化；而
    客户端构造本身还是「缺密钥当场抛」的位置（llm_router），movement 会把那条
    配置错误从启动期挪到每次提问。故用计数钉住，且让两次问答都真的走到生成
    （命中行给足），否则计数恒为 1 就成了一条不可能失败的断言。
    """
    calls: list[str] = []

    def fake_get_llm(side):
        calls.append(side)
        return _FakeLLM()

    monkeypatch.setattr("app.generation.llm_router.get_llm", fake_get_llm)
    rows = [{"text": "每件 1 元至 10 元", "source_doc": "甲所",
             "source_no": "一、", "status": "现行有效"}]
    extras_fn = build_extras_fn(MagicMock(), _FakeMilvus(rows), _FakeEncoder())
    for question in ("建房纠纷", "押金不退"):
        assert extras_fn(question)["fee"]["status"] == "ok"
    assert calls == [SIDE_PUBLIC], "两次问答共用一个客户端，不是每问一次现建"
