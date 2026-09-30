from app.core.config import Settings
from app.core.retrieve.query_rewrite import QueryRewriter


class _FakeLLM:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return self.resp


def test_rewrite_disabled_returns_original():
    # 关掉改写也要给出「原问题」这一路，返回空列表等于整条检索没有查询。
    r = QueryRewriter(Settings(query_rewrite_enabled=False))
    assert r.rewrite("高血压怎么吃盐？") == ["高血压怎么吃盐？"]


def test_rewrite_expands_and_dedupes():
    llm = _FakeLLM("高血压患者每日食盐摄入量\n- 限盐 5 克\n高血压患者每日食盐摄入量\n")
    r = QueryRewriter(Settings(query_rewrite_enabled=True), llm=llm)
    qs = r.rewrite("高血压怎么吃盐？")
    assert qs[0] == "高血压怎么吃盐？"  # 原问题恒在首位，改写出问题也不能丢
    assert "高血压患者每日食盐摄入量" in qs
    assert "限盐 5 克" in qs
    assert len(qs) == len(set(qs))  # 去重


def test_rewrite_strips_list_markers_but_keeps_leading_digits():
    """去编号只能去**编号**，不能把内容开头的数字一起吃掉。

    以前是 `lstrip("-·*0123456789.、 ")`——按字符集剥离，于是「2 型糖尿病的诊断标准」
    变成「型糖尿病的诊断标准」、「120/80mmHg 是多少」变成「/80mmHg 是多少」（实测）。
    被削过的查询会当作独立一路参与 RRF 融合，等于拿错查询去召回。
    """
    llm = _FakeLLM("1. 2 型糖尿病的诊断标准\n2、120/80mmHg 算高血压吗\n- 限盐每天 5 克\n")
    r = QueryRewriter(Settings(query_rewrite_enabled=True), llm=llm)

    assert r.rewrite("原始问题") == [
        "原始问题",
        "2 型糖尿病的诊断标准",
        "120/80mmHg 算高血压吗",
        "限盐每天 5 克",
    ]

    # 其余编号样式（1) / 1．）同样只吃编号本身
    r2 = QueryRewriter(Settings(query_rewrite_enabled=True), llm=_FakeLLM("3) 高血压分级标准\n4．24 小时尿钠怎么测\n"))
    assert r2.rewrite("原始问题") == [
        "原始问题",
        "高血压分级标准",
        "24 小时尿钠怎么测",
    ]


def test_rewrite_llm_failure_returns_original():
    # 改写只是可选增强：LLM 挂了要退回原问题，不能把整条问答链路一起拖垮。
    class _Boom:
        def chat(self, messages):
            raise RuntimeError("boom")

    r = QueryRewriter(Settings(query_rewrite_enabled=True), llm=_Boom())
    assert r.rewrite("问题") == ["问题"]
