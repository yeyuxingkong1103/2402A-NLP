# 公众侧两条端点的判据（设计 §四 的契约、§七 的错误映射、§六 的匿名可读）。
#
# 问答端点用**假 Answerer**：真问答已由 CLI 冒烟覆盖，HTTP 层测的是接线与映射
# （本任务不花 API 钱）。判据一律走真 create_app —— 中间件、异常映射、路由都
# 来自真装配，重资源换替身。
#
# 契约层最要紧的两条（都是交接来的）：
#   ①响应必须**含** lawyers 与 fee_range 两个键（with_extras=False 是合法装配组合，
#     漏传就静默丢键 —— 任务 1 复审的交接）；
#   ②问句按**字符数**再限一次（体上限中间件只认 Content-Length，分块传输绕得过 ——
#     任务 3 复审的交接）。
import logging

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import schemas
from app.core import errors, request_id
from app.generation.answer import QAResult
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC
from app.generation.schema import Citation
from app.recommend import cause as cause_mod
from tests._fakes_http import FakeConn, FakeServices, fake_factory

# 响应体里绝不许出现的标记：断言「不在」时越独特越可靠
LEAK = "milvus down：SECRET-原文"
QUESTION = "房东不退押金怎么办"
FEE_OK = {"status": "ok", "low": 800, "high": 1000, "unit": "元",
          "charge_basis": "小时", "basis": "……每有效工作小时 800 元……",
          "source_doc": "北京市律师服务收费管理办法", "source_no": "bjzs-3",
          "reason": None}
EXTRAS = {"cause": "房屋租赁合同纠纷", "field": "房产与建设工程",
          "lawyers": [{"name": "赵明", "org": "明理德正律师事务所",
                       "field": "房产与建设工程", "contact_hint": "通过平台留言获取",
                       "demo": True, "note": "示例数据"}],
          "fee": FEE_OK, "disclaimer": "参考区间，不构成报价或委托"}


class _Answerer:
    """假 Answerer：记下 (question, side) 并把预置的 QAResult 原样返回。

    记录调用参数是为了钉两件事：问句真的透传下去了、侧别**显式**传了 public
    （Answerer 的默认侧是 internal，漏传会让合规提示词静默失效）。
    """

    def __init__(self, result: QAResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def answer(self, question: str, side: str) -> QAResult:
        self.calls.append((question, side))
        return self.result


def _result(**overrides) -> QAResult:
    """造一个正常终态的 QAResult（字段可覆盖）。用真 QAResult 而不是替身对象：
    接口层读的就是它的字段名，替身会让字段改名漂过去。"""
    fields = dict(
        status="ok", answer="依据《民法典》第七百三十三条……", disclaimer="仅供参考",
        citations=[Citation(law="中华人民共和国民法典", article="733",
                            paragraph=None, item=None, quote="承租人……返还租赁物")],
        # text/status/law_id 是**内部**字段（块的原貌），契约里没有它们 ——
        # 用例拿它们来断言「投影真的把内部字段挡住了」
        sources=[{"article_no": 733, "path": "第十四章 租赁合同",
                  "source": "vector", "rerank_score": 0.91,
                  "text": "内部法条原文", "status": "现行有效", "law_id": "民法典"}],
        failures=[], extras=dict(EXTRAS))
    fields.update(overrides)
    return QAResult(**fields)


def _app(result: QAResult, **services_kwargs):
    """真 app + 假 Answerer。返回 (app, 假 answerer)。"""
    answerer = _Answerer(result)
    factory_fn, _ = fake_factory(FakeServices(answerer=answerer, **services_kwargs))
    return main.create_app(services_factory=factory_fn), answerer


def _ask(client: TestClient, question: str = QUESTION, **kwargs):
    return client.post("/api/v1/public/qa", json={"question": question}, **kwargs)


# ---- 契约：字段逐字段 ----


def test_the_lawyer_side_model_cannot_carry_the_two_block_keys():
    """律师侧基类 `QAAnswer` **不许**有 `lawyers`/`fee_range` 两个字段（设计 §四 原文：
    「不是置空，是根本不出现」）。

    这条钉在**基类**上，因为红线是在基类上成立的：律师侧响应模型（任务 5）继承它，
    而 pydantic 的字段定义是「不许出现」的唯一机制。实测（2026-09-30 任务 4 审查 M7）：
    在这条用例存在之前，往 `QAAnswer` 里加回这两个字段 → 五个测试文件 **88 passed
    全绿** —— 机制今天对，但改坏了没有任何红灯（本项目「不可能失败的测试」家族的
    「判据根本不存在」形态）。任务 5 还要在**响应层**再钉一次（§八 那行断言的
    对象是律师侧响应本身），两条不重复：这条守基类，那条守接线。

    判据用字面量集合而不是 `== set(QAAnswer.model_fields)` 的自洽写法：后者在
    「多加了字段」时恒绿，而「多了什么」正是要有人看一眼的东西。
    """
    assert set(schemas.QAAnswer.model_fields) == {
        "request_id", "status", "answer", "citations", "sources", "disclaimer",
        "failures"}
    assert "lawyers" not in schemas.QAAnswer.model_fields
    assert "fee_range" not in schemas.QAAnswer.model_fields


def test_the_response_carries_exactly_the_designed_fields():
    """设计 §四 的字段表，一次列全当锚点（多一个键少一个键都要有人看一眼）。

    其中 lawyers 与 fee_range 是**必须存在**的两个键（不是「非空」）—— 它们由
    extras 展开而来，而 extras 在 Answerer 的某些出口是 None、装配也可以不带
    extras（CLI 的 --no-extras）。只断言值的话，「键整个消失」的形态照样绿，
    而前端会在读第一个卡片时炸掉。
    """
    app, _ = _app(_result())
    with TestClient(app) as client:
        response = _ask(client)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"request_id", "status", "answer", "citations", "sources",
                         "cause", "field", "lawyers", "fee_range", "disclaimer",
                         "failures"}
    assert body["status"] == "ok"
    assert body["answer"] == _result().answer
    assert body["cause"] == EXTRAS["cause"]
    assert body["field"] == EXTRAS["field"]
    assert body["lawyers"] == EXTRAS["lawyers"]
    assert body["disclaimer"] == _result().disclaimer
    assert body["failures"] == []
    # fee_range 来自 extras["fee"]，且**不含 reason**（见下面那条投影用例）
    assert body["fee_range"] == {k: v for k, v in FEE_OK.items() if k != "reason"}


def test_the_question_and_the_side_reach_the_answerer():
    """问句要原样走到 Answerer，侧别要**显式**是 public。

    漏传侧别的形态很隐蔽：Answerer 的默认值是 internal，于是公众侧会拿律师侧
    的提示词（少三条合规约束）、并且永远没有附加区块 —— 两条都不报错。
    """
    app, answerer = _app(_result())
    with TestClient(app) as client:
        assert _ask(client).status_code == 200
    assert answerer.calls == [(QUESTION, SIDE_PUBLIC)]
    assert SIDE_INTERNAL != SIDE_PUBLIC  # 常量本身也得是两个不同的值


def test_when_there_is_no_block_the_two_keys_are_still_present():
    """extras 为 None 的出口（拒答且不问价、或装配不带 extras）里，两个键仍在。

    值给 [] 与 null：键的有无是契约，值的有无是本次回答的事实。把键省掉等于
    让客户端去判一个服务端内部判据（Answerer._extras_for 的三重裁决）。
    """
    app, _ = _app(_result(status="out_of_scope", answer="本平台只答法律问题。",
                          citations=[], sources=[], extras=None))
    with TestClient(app) as client:
        body = _ask(client).json()
    assert body["lawyers"] == []
    assert body["fee_range"] is None
    assert body["cause"] is None and body["field"] is None


def test_citations_are_projected_to_the_contract_shape():
    """引用投影成 §四 的四键：模型给的 `law`（自由文本、未经校验）不进对外契约。

    投影而不是透传的理由见 schemas 的模块 docstring；这条用例钉住「投影真的发生了」
    —— 透传（model_dump）时 `law` 会出现在响应里，而它不在契约中。
    """
    app, _ = _app(_result())
    with TestClient(app) as client:
        citation = _ask(client).json()["citations"][0]
    assert set(citation) == {"article", "paragraph", "item", "quote"}
    assert citation["article"] == "733"
    assert citation["quote"].startswith("承租人")


def test_sources_are_projected_and_do_not_leak_the_internal_fields():
    """检索块投影成 §四 的四键：内部字段（整条法条 `text`、`status`、`law_id`、
    `chunk_id`）一个都不出去。

    泄漏的形态是响应体随召回条数膨胀（5 条整条法条 ≈ 数十 KB），且对外把未经
    校验的库内字段（如效力状态）当成了接口契约的一部分 —— 前者是性能，后者是
    契约稳定性，两个都值得一条用例。
    """
    app, _ = _app(_result())
    with TestClient(app) as client:
        response = _ask(client)
    source = response.json()["sources"][0]
    assert set(source) == {"article_no", "path", "source", "rerank_score"}
    assert source == {"article_no": 733, "path": "第十四章 租赁合同",
                      "source": "vector", "rerank_score": 0.91}
    assert "内部法条原文" not in response.text
    assert "law_id" not in response.text


def test_the_fee_block_never_leaks_its_reason_text():
    """费用区块的 `reason` 不带出去：故障那一支里它装的是异常原文（"milvus down"）。

    设计 §七 明写异常原文只进日志（③b-1 的真跑抓到过外泄），而 attach._unavailable
    正是把原文写进了这个字段。故这里是**投影**（8 键白名单）而不是透传 ——
    CLI 的渲染层同样刻意不印它。
    """
    fee = {"status": "unavailable", "low": None, "high": None, "unit": None,
           "charge_basis": None, "basis": None, "source_doc": None,
           "source_no": None, "reason": f"费用信息暂时不可用：{LEAK}"}
    extras = dict(EXTRAS, fee=fee)
    app, _ = _app(_result(extras=extras))
    with TestClient(app) as client:
        response = _ask(client)
    assert response.status_code == 200
    assert response.json()["fee_range"]["status"] == "unavailable"
    assert "reason" not in response.json()["fee_range"]
    assert LEAK not in response.text


# ---- 状态码映射：故障 503 / 无依据 200 ----


@pytest.mark.parametrize("status", ["ok", "abstain", "need_more_info", "out_of_scope"])
def test_a_completed_answer_is_200_whatever_its_status(status):
    """§七 第 6 行：问答**正常完成**但没依据 → 200，由 status 字段表达。

    四个终态一起参数化（写成 `status != "ok"` 就抛的实现在这里必红）：追问与
    拒答不是错误，压成 4xx/5xx 会让前端把「模型在追问」显示成故障。
    """
    app, _ = _app(_result(status=status))
    with TestClient(app) as client:
        response = _ask(client)
    assert response.status_code == 200
    assert response.json()["status"] == status


def test_a_chain_failure_is_503_with_a_generic_message(caplog):
    """§七 第 5 行：故障 → 503，通用文案；异常原文只进日志（③b-1 的教训）。

    两个方向都断言：响应里没有原文，日志里有 —— 只说「没泄露」时，把日志一起
    删掉也照样绿，而那样排查故障就没有现场了。
    """
    app, _ = _app(_result(status=errors.FAILURE_STATUS,
                          answer=f"生成服务不可用：{LEAK}"))
    with caplog.at_level(logging.ERROR):
        with TestClient(app) as client:
            response = _ask(client)
    assert response.status_code == 503
    assert response.json()["error"] == {"code": "service_unavailable",
                                        "message": "服务暂时不可用，请稍后再试"}
    assert LEAK not in response.text
    assert any(LEAK in record.getMessage() for record in caplog.records), \
        "故障原文没有进日志"


# ---- 输入校验：字符数上限（分块传输也绕不过的那一层）----


def test_a_question_at_the_limit_passes_and_one_over_it_is_400():
    """边界两侧各一次：恰好等于上限要放行，多一个字要 400。

    上限值本身也当字面量锚点 —— 边界用例拿常量当输入时，把上限改成 100 万
    它照样绿，而那等于这层保护不存在（体上限是 64KiB，两层挡的东西不同）。
    """
    assert schemas.MAX_QUESTION_CHARS == 500
    app, answerer = _app(_result())
    with TestClient(app) as client:
        ok = _ask(client, "问" * schemas.MAX_QUESTION_CHARS)
        too_long = _ask(client, "问" * (schemas.MAX_QUESTION_CHARS + 1))
    assert ok.status_code == 200
    assert too_long.status_code == 400
    assert too_long.json()["error"]["code"] == "bad_request"
    # 超长问句**不**透传下去：拒绝而不是截断（截断会让引用回查对着半句话工作）
    assert [call[0] for call in answerer.calls] == ["问" * schemas.MAX_QUESTION_CHARS]


@pytest.mark.parametrize("body", [
    {},                                   # 没有 question
    {"question": ""},                     # 空串
    {"question": "   \n\t "},             # 只有空白：strip 后为空
    {"question": 42},                     # 不是字符串
    {"question": ["a", "b"]},             # 类型不对
])
def test_a_malformed_question_is_400_and_never_reaches_the_answerer(body):
    """畸形问句一律 400（校验层），且**不调到 Answerer**。

    空白那一条是本任务特意加的：`min_length=1` 挡不住「只含空格的问句」，它会一路
    走到编码器，在那儿以「sparse 转换后为空」炸成**服务故障** —— 把「输入不合法」
    说成「我们坏了」（③a 的红线）。strip 必须在长度判定之前生效。
    """
    app, answerer = _app(_result())
    with TestClient(app) as client:
        response = client.post("/api/v1/public/qa", json=body)
    assert response.status_code == 400
    assert answerer.calls == []


def test_a_question_is_stripped_before_it_is_answered():
    """两端的空白被剥掉后再走链路：编码器对空串会抛，而这层不该让「多打了空格」
    变成 400 或 503。链条上唯一的真相是校验后的值 —— 故断言透传的就是它的 strip。"""
    app, answerer = _app(_result())
    with TestClient(app) as client:
        assert _ask(client, f"  {QUESTION}  ").status_code == 200
    assert answerer.calls == [(QUESTION, SIDE_PUBLIC)]


# ---- 其他：request_id、装配缺失 ----


def test_every_response_carries_the_same_request_id_in_header_and_body():
    """§四：每个响应带 request_id，头与体同值（审计与日志靠它对齐）。"""
    app, _ = _app(_result())
    with TestClient(app) as client:
        response = _ask(client)
    assert response.headers[request_id.HEADER] == response.json()["request_id"]


def test_without_assembled_services_the_qa_endpoint_is_503():
    """装配没起来 → 503（不是 500，更不是 200 的空答案）：运维看状态码要能得出
    「这个实例没起来」的结论。"""
    factory_fn, _ = fake_factory()
    app = main.create_app(services_factory=factory_fn)
    probe = TestClient(app, raise_server_exceptions=False)
    response = probe.post("/api/v1/public/qa", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


# ---- GET /lawyers/recommend ----


def test_recommend_is_anonymous_and_returns_cards_and_a_null_fee_range():
    """匿名可读（§六：它与 /public/qa 一样本就该公开）+ 契约键位。

    fee_range 恒为 null 是本任务的判断（理由见 api/public.recommend_lawyers 的
    docstring）：这条端点没有问句、也就没有案由，费用链路的检索查询串无从谈起，
    编一个查询串只会产出一个与任何案由都不相干的价格。键位保留，故将来设计层
    回答了「cause 从哪来」之后填充它不需要改契约。
    """
    app, _ = _app(_result())
    with TestClient(app) as client:
        response = client.get("/api/v1/lawyers/recommend")  # 不带任何 token
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"request_id", "field", "lawyers", "fee_range"}
    assert body["field"] == cause_mod.GENERIC_FIELD
    assert body["fee_range"] is None
    assert len(body["lawyers"]) == 3
    for card in body["lawyers"]:
        # 领域匹配（FR-9.3）+ 演示标注（风险 11：示例数据必须标出来，
        # 否则对外构成虚假宣传）—— 卡片随响应一起给出这两样，前端不必再判断
        assert card["field"] == cause_mod.GENERIC_FIELD
        assert card["demo"] is True and card["note"]


def test_recommend_does_not_touch_the_database_or_the_answerer():
    """这条端点不查库、不调问答：

    「不跑费用链路」不只是省一次 DeepSeek 调用 —— 它意味着这条路径**没有任何
    外部依赖**（无检索、无生成、无留痕），所以哪怕 Milvus/MySQL 全挂了它也该
    照常返回。用例用假连接记账把它钉住，免得将来有人顺手在卡片前加一次检索。
    """
    business = FakeConn()
    app, answerer = _app(_result(), conn=business)
    with TestClient(app) as client:
        assert client.get("/api/v1/lawyers/recommend").status_code == 200
    assert business.statements == []
    assert answerer.calls == []
