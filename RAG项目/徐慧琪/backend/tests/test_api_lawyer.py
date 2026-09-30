# 律师侧三条端点的判据：设计 §四 的契约、§六 的三层隔离（路由表层面）、
# §七 的错误映射、§二 第 6/9 条（501 与 404 化）。
#
# 判据一律走真 create_app（中间件、异常映射、路由都来自真装配），重资源换替身；
# 只有最后一条真链路用例加载真模型、真连 Milvus，其余用例测接线与映射（不加载 2GB 模型）。
import gc
import logging

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import schemas
from app.core import errors, factory, request_id, security
from app.core.security import Role
from app.generation.answer import QAResult
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC
from app.retrieval.pipeline import RERANK_OUTPUT_TOPK, RECALL_TOPK, RetrievalResult
from tests._fakes_http import JWT_SECRET, FakeConn, FakeServices, bearer, fake_factory
from tests._fakes_http import jwt_token, without_request_id

QUESTION = "租赁合同解除后押金怎么退"
LEAK = "milvus down：SECRET-原文"
# 检索块的原貌（含**内部**字段 text/status/law_id）：用例拿它们断言投影真的挡住了内部字段
BLOCK = {"article_no": 584, "article_no_cn": "五百八十四",
         "path": "第三编 合同 > 第一分编 通 则 > 第八章 违约责任",
         "source": "vector", "rerank_score": 0.91, "text": f"内部原文-{LEAK}",
         "status": "现行有效", "law_id": "minfadian"}


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    """签名密钥走环境变量（security.load_secret 的唯一来源）：漏设的用例会以
    MissingSecretError 的形态变成 500，而那不是它们要测的东西（与另两份同形）。"""
    monkeypatch.setenv(security.JWT_SECRET_ENV, JWT_SECRET)


class _Answerer:
    """假 Answerer：记下 (question, side) 并把预置的 QAResult 原样返回。

    记录调用参数是律师侧最要紧的一条判据：Answerer.answer 的**默认**侧别恰好
    就是 internal，所以「漏传侧别」在响应内容上完全看不出来（拿到的仍是律师侧
    提示词）—— 只有参数才分得出「显式传了」与「撞上了默认值」。
    """

    def __init__(self, result: QAResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def answer(self, question: str, side: str) -> QAResult:
        self.calls.append((question, side))
        return self.result


class _Retriever:
    """假 retrieve：记下 (question, kwargs) 并返回预置的 RetrievalResult。

    真 retrieve 要真模型 + 真 Milvus（2GB 显存、秒级），接线用例不该为此付费；
    而「参数有没有递对」只有参数本身能证明 —— 判据因此落在调用记录上，
    服务对象用 `is` 比对（递错一条连接/模型都不会红在响应内容上）。
    """

    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def __call__(self, question: str, **kwargs) -> RetrievalResult:
        self.calls.append((question, kwargs))
        return self.result


class _AccountConn(FakeConn):
    """假账号连接，**多一个 close()**：真 Services 进 lifespan 后会被 release()
    逐个 close，而基类没有这个方法 —— 缺了它，用例会在日志里留下一条「回收
    account_conn 失败」的 warning（不抛错，但噪声会掩盖真故障信号）。"""

    def close(self) -> None:
        pass


def _result(**overrides) -> QAResult:
    """造一个正常终态的 QAResult（字段可覆盖）；用真 QAResult 而不是替身对象，
    接口层读的就是它的字段名，替身会让改名漂过去。"""
    fields = dict(status="ok", answer="依《民法典》第五百八十四条……",
                  disclaimer="", citations=[], sources=[], failures=[])
    fields.update(overrides)
    return QAResult(**fields)


def _app(*, answerer=None, retriever=None, monkeypatch=None,
         account_row=(1,)) -> tuple[TestClient, FakeServices]:
    """真 app + 假 Services。返回 (客户端, 那一套假 Services)。

    account_row 默认 (1,)（账号启用）：鉴权依赖逐请求查 is_active，没有这一行时
    任何 token 都会被判「账号不可用」→ 404（那会红在任何一条用例上，且指错方向）。"""
    services = FakeServices(answerer=answerer, account_conn=_AccountConn(row=account_row))
    factory_fn, _ = fake_factory(services)
    if retriever is not None and monkeypatch is not None:
        import app.api.lawyer as lawyer_mod
        monkeypatch.setattr(lawyer_mod, "retrieve", retriever)
    return TestClient(main.create_app(services_factory=factory_fn)), services


def _post(client: TestClient, path: str, *, token: str | None = None,
          body: dict | None = None):
    """打一条受保护路径；token 省略时就是「未认证」那一档。"""
    headers = {} if token is None else bearer(token)
    return client.post(path, json=body, headers=headers)


# ---- POST /api/v1/qa（FR-7.1）----


def test_the_lawyer_answer_carries_exactly_seven_keys_and_no_block_keys():
    """设计 §四 的红线：律师侧响应里**没有** `lawyers` 与 `fee_range`。

    两个方向都断言（键集字面量 + 逐键 not in）：只断言「不存在」时，把响应换成
    任意一个别的模型（少掉 answer）也照样绿；只断言键集时，错误信息里看不出
    意图。这条钉的是**响应本身**（任务 4 报告 §五 2 的带转移交项：基类那条用例
    守 QAAnswer 的定义，这条守接线有没有用上它）。

    `lawyers`/`fee_range` 随 extras 展开：一旦本端点误用 PublicQAAnswer（或
    public_qa_payload），两个键就会带着律师卡片与费用区间出现在律师侧响应里。"""
    answerer = _Answerer(_result())
    client, _ = _app(answerer=answerer)
    with client:
        response = _post(client, "/api/v1/qa", token=jwt_token(),
                         body={"question": QUESTION})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"request_id", "status", "answer", "citations", "sources",
                         "disclaimer", "failures"}
    assert "lawyers" not in body and "fee_range" not in body
    assert body["answer"] == _result().answer


def test_the_lawyer_side_is_passed_explicitly_and_not_left_to_the_default():
    """侧别必须**显式**传 SIDE_INTERNAL：Answerer 的签名默认值恰好也是 internal，
    漏传今天不红（这是最危险的一种「对」）。判据只能是调用参数 —— 内容分不出侧别。"""
    answerer = _Answerer(_result())
    client, _ = _app(answerer=answerer)
    with client:
        assert _post(client, "/api/v1/qa", token=jwt_token(),
                     body={"question": QUESTION}).status_code == 200
    assert answerer.calls == [(QUESTION, SIDE_INTERNAL)]
    assert SIDE_INTERNAL != SIDE_PUBLIC  # 常量本身也得是两个不同的值


@pytest.mark.parametrize("role", [Role.LAWYER, Role.ASSISTANT, Role.PARTNER])
def test_any_logged_in_role_may_ask(role):
    """任意已登录角色都可问（设计 §四 的「JWT」那一格）。三个角色一起参数化：
    将来有人给这条路径加上 require_roles(PARTNER) 之类的收窄，必须有人看一眼 ——
    而只测 lawyer 一个角色的用例对它全绿。"""
    answerer = _Answerer(_result())
    client, _ = _app(answerer=answerer)
    with client:
        response = _post(client, "/api/v1/qa", token=jwt_token(role=role),
                         body={"question": QUESTION})
    assert response.status_code == 200, response.text


def test_without_a_token_the_same_path_is_404_and_with_one_it_is_200():
    """200 与 404 的**对照**（设计 §六 + §二 第 9 条）：只测一半的用例挡不住
    「把鉴权依赖整个删掉」这种改动 —— 那时 200 那半照样绿。404 还必须与
    「路径不存在」逐字段同形：律师侧这三条路径的存在本身说明「这里有所内功能」。"""
    answerer = _Answerer(_result())
    client, _ = _app(answerer=answerer)
    with client:
        missing = client.get("/api/v1/no-such-path")
        anonymous = _post(client, "/api/v1/qa", body={"question": QUESTION})
        authorized = _post(client, "/api/v1/qa", token=jwt_token(),
                           body={"question": QUESTION})
        assert missing.status_code == anonymous.status_code == 404, "无 token 必须 404"
        assert without_request_id(anonymous) == without_request_id(missing)
        assert authorized.status_code == 200
        # 未认证的请求**不许**走到 Answerer（否则一次探测就白烧一次生成）
        assert answerer.calls == [(QUESTION, SIDE_INTERNAL)]


def test_a_chain_failure_on_the_lawyer_side_is_503_with_a_generic_message(caplog):
    """§七 第 5 行同样适用于律师侧：故障 → 503、通用文案，原文只进日志 ——
    律师侧不是「内部人就不怕泄露」，异常原文里可能有下游的主机名与库名。"""
    answerer = _Answerer(_result(status=errors.FAILURE_STATUS,
                                 answer=f"生成服务不可用：{LEAK}"))
    client, _ = _app(answerer=answerer)
    with caplog.at_level(logging.ERROR):
        with client:
            response = _post(client, "/api/v1/qa", token=jwt_token(),
                             body={"question": QUESTION})
    assert response.status_code == 503
    assert response.json()["error"] == {"code": "service_unavailable",
                                        "message": "服务暂时不可用，请稍后再试"}
    assert LEAK not in response.text
    assert any(LEAK in record.getMessage() for record in caplog.records), \
        "故障原文没有进日志"


# ---- POST /api/v1/search（只检索不生成）----


def _search_result() -> RetrievalResult:
    """一份预置的检索结果：块是**真块形状**（含 text/status/law_id 等内部字段），
    另塞了 chunks/recalled_blocks（检索层内部产物），据此断言它们没被带出去。"""
    return RetrievalResult(question=QUESTION, blocks=[dict(BLOCK)],
                           chunks=[{"chunk_id": "c1", "text": f"子块-{LEAK}"}],
                           exact_nos=[584], recalled_blocks=[dict(BLOCK)])


def test_search_passes_the_question_and_the_shared_services_to_the_retriever(monkeypatch):
    """接线判据：问句、四个重资源、top_k 全部按值/按身份递到 retrieve。

    两个方向：①递下去的东西对（四个对象用 `is` 比对 —— 递错一条连接不会红在
    响应上，真跑到线上却是「检索用的是另一套连接/模型」）；②回来之后只投影
    契约键（内部的 text/status/law_id 与 chunks/recalled_blocks 一个都不出去）。"""
    retriever = _Retriever(_search_result())
    client, services = _app(retriever=retriever, monkeypatch=monkeypatch)
    with client:
        response = _post(client, "/api/v1/search", token=jwt_token(),
                         body={"question": QUESTION})
    assert response.status_code == 200, response.text
    question, kwargs = retriever.calls[0]
    assert question == QUESTION
    assert kwargs["conn"] is services.conn
    assert kwargs["client"] is services.client
    assert kwargs["encoder"] is services.encoder
    assert kwargs["reranker"] is services.reranker
    # 缺省 top_k 只能有一份字面量：默认值在路由层取 pipeline.RERANK_OUTPUT_TOPK，
    # 这里同时钉住「取到了它」与「它的值是 5」（后者是防两侧一起漂的字面量锚）
    assert kwargs["top_k"] == RERANK_OUTPUT_TOPK == 5
    body = response.json()
    assert set(body) == {"request_id", "question", "blocks", "exact_nos"}
    assert body["question"] == QUESTION
    assert body["exact_nos"] == [584]
    assert set(body["blocks"][0]) == {"article_no", "article_no_cn", "path",
                                      "source", "rerank_score"}
    assert body["blocks"][0]["article_no"] == 584
    assert "内部原文" not in response.text and "子块" not in response.text
    assert "recalled_blocks" not in response.text


def test_search_honours_an_explicit_top_k_and_rejects_values_over_the_cap(monkeypatch):
    """top_k 的上限卡在 schema 层（超限 400，且**不**调到检索）。

    上限值当字面量锚（改成 100000 时用例必须红 —— 那样这道闸门就名存实亡），
    并断言它小于召回池：merge_blocks 吐不出比池子更多的块，上限若 ≥ 池子，
    这个判据连「闸门存在」都证明不了。"""
    assert schemas.MAX_TOP_K == 20  # 字面量锚：常量与用例一起漂时这条才拦得住
    assert schemas.MAX_TOP_K < RECALL_TOPK
    retriever = _Retriever(_search_result())
    client, _ = _app(retriever=retriever, monkeypatch=monkeypatch)
    with client:
        at_cap = _post(client, "/api/v1/search", token=jwt_token(),
                       body={"question": QUESTION, "top_k": schemas.MAX_TOP_K})
        over = _post(client, "/api/v1/search", token=jwt_token(),
                     body={"question": QUESTION, "top_k": schemas.MAX_TOP_K + 1})
        zero = _post(client, "/api/v1/search", token=jwt_token(),
                     body={"question": QUESTION, "top_k": 0})
    assert at_cap.status_code == 200
    assert [call[1]["top_k"] for call in retriever.calls] == [schemas.MAX_TOP_K]
    for response in (over, zero):
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"
    assert len(retriever.calls) == 1, "超限的请求不该走到检索"


def test_search_requires_a_token_like_the_other_two_lawyer_paths():
    """与 /qa 同一档：无 token 404、且与未知路径同形。"""
    retriever = _Retriever(_search_result())
    client, _ = _app(retriever=retriever, monkeypatch=None)
    with client:
        missing = client.get("/api/v1/no-such-path")
        anonymous = _post(client, "/api/v1/search", body={"question": QUESTION})
    assert anonymous.status_code == 404
    assert without_request_id(anonymous) == without_request_id(missing)
    assert retriever.calls == []


# ---- POST /api/v1/cases/search（设计 §二 第 6 条：注册但 501）----


def test_cases_search_is_501_when_logged_in_and_404_when_anonymous():
    """两个方向都要钉：这条路径**挂了鉴权**（§六 把它列在律师侧专用路径里），
    所以未认证时必须是 404 而不是 501 —— 501 会告诉未认证的探测者
    「这里有一个尚未开放的功能」，那比「路径存在」泄露得更多。

    501 的文案要能说明「未实现」（§七 第 4 行「reason 写明」）：前端据此显示
    「即将开放」而不是「服务器出错」。断言取两个语义锚（"未实现" 与 "FR-5.3"）
    而不是整句 —— 整句断言会把文案的每次润色都变成红灯，而文案本身不是契约。
    """
    answerer = _Answerer(_result())
    client, _ = _app(answerer=answerer)
    with client:
        missing = client.get("/api/v1/no-such-path")
        anonymous = _post(client, "/api/v1/cases/search")
        expired = _post(client, "/api/v1/cases/search", token=jwt_token(ttl_s=-10))
        authorized = _post(client, "/api/v1/cases/search", token=jwt_token(),
                           body={"question": QUESTION})
        assert authorized.status_code == 501, authorized.text
        error = authorized.json()["error"]
        assert error["code"] == "not_implemented"
        assert "未实现" in error["message"] and "FR-5.3" in error["message"]
        # request_id 在（§四：每个响应带 request_id）：501 也要能按 id 捞日志
        assert authorized.headers[request_id.HEADER] == authorized.json()["request_id"]
    assert answerer.calls == [], "501 的端点不许碰问答链路"
    for response in (anonymous, expired):
        assert response.status_code == 404
        assert without_request_id(response) == without_request_id(missing)


# ---- 真链路：只读的端到端证据（真 Milvus + 真模型 + 真 MySQL）----


def _e2e_available() -> bool:
    """真链路用例的前置：Milvus 与 MySQL 都在线才跑（离线时跳过，不假装通过）。"""
    try:
        from app.db.milvus import get_client
        get_client().list_collections()
        from app.db.mysql import connect
        connect().close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _e2e_available(), reason="MySQL/Milvus 未在线")
def test_a_real_search_returns_parent_blocks_through_the_http_layer():
    """本任务的端到端证据（不许省）：真问句 → 真检索链路 → HTTP 响应。

    判据压在「真结果」上而不是接线：blocks 非空、每块有 article_no 与 path。
    为什么这句问句：**不含条号**（"租赁合同解除"里没有「第…条」），于是它走的是
    纯向量路 —— 精确通路那半边由条文端点与 test_article_lookup 覆盖。

    账号连接用假体（只回答 is_active=1）：本用例要证的是检索链路能跑，
    真账号库那一列已由 test_api_deps 的真连库用例覆盖 —— 两边都不重复付费。

    模型**用完即放**（finally 里删引用 + gc）：bge-m3 与精排常驻会占几 GB 内存，
    而本机的崩溃史（账本「环境插曲」）正是内存压力下加载模型时的 access violation；
    这个文件只用它们一次，没有理由让它们陪跑到会话结束、把后面的用例挤到墙边。
    """
    from app.db.milvus import get_client
    from app.db.mysql import connect
    from app.ingest.embed import load_model
    from app.retrieval.rerank import load_reranker
    conn, client = connect(), get_client()
    encoder, reranker = load_model(), load_reranker()
    try:
        def services_fn():
            return factory.Services(conn=conn, client=client, encoder=encoder,
                                    reranker=reranker, answerer=None,
                                    account_conn=_AccountConn(row=(1,)))

        app = main.create_app(services_factory=services_fn)
        with TestClient(app) as http:
            response = _post(http, "/api/v1/search", token=jwt_token(),
                             body={"question": "租赁合同解除"})
    finally:
        # 连接与 Milvus 客户端已由 lifespan 的 close 释放（Services.close），
        # 这里只管模型（torch 对象没有 close，回收靠引用计数 + GC）
        del encoder, reranker
        gc.collect()
    assert response.status_code == 200, response.text
    blocks = response.json()["blocks"]
    assert blocks, "真链路检索没有召回任何父块"
    assert len(blocks) <= RERANK_OUTPUT_TOPK
    for block in blocks:
        assert block["article_no"] and block["path"], block
