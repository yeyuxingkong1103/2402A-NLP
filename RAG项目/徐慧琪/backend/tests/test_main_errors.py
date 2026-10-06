# 设计 §七 那张表的逐行判据：一行一条用例，名字里写明是哪一行。
#
# 为什么单独成文件：这张表是硬规格，混进 test_main.py 会让「哪一行被覆盖了」看不
# 出来；而漏一行的形态是**没有红灯**（少一个处理器，那个异常就悄悄变成 500）。
#
# 所有用例都走真 create_app 出的 app（中间件与处理器来自真装配）：另起一个
# FastAPI 再手工挂处理器，测的就是测试自己写的那份映射了。
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app import main
from app.core import errors, request_id
from app.core.security import AuthError
from app.generation.answer import Answerer
from app.generation.profiles import SIDE_PUBLIC
from tests._fakes_http import ResultStub, fake_factory

# 响应体里绝不许出现的标记。用显眼的字面量而不是真异常文本（如 "milvus down"）：
# 断言「不在」时，标记越独特越不会被无关文本碰巧命中
SECRET = "SECRET-原文"


class _ProbeBody(BaseModel):
    """校验用例的请求体模型：count 要 int，给字符串就会触发 400。"""

    count: int


class _Boom(Exception):
    """兜底用例抛的**自定义**异常：不能用 ValueError 这类框架/库也会抛的类型。

    换成 ValueError 的话，「处理器注册在 Exception 上」与「注册在 ValueError 上」
    用这条用例分辨不出来（变异实测确认），而后者放行 TypeError 等一切别的异常 ——
    那时 500 会退回框架的纯文本响应，统一形状与 request_id 全丢。
    """


def _raising(exc: Exception):
    """造一个抛指定异常的探针端点（接口层只看异常类型，不看它从哪来）。"""
    def endpoint() -> None:
        raise exc

    return endpoint


def _app(**raises: Exception) -> FastAPI:
    """起一个真装配的 app，并挂上若干「抛指定异常」的探针路由（键名即路径末段）。"""
    factory_fn, _ = fake_factory()
    app = main.create_app(services_factory=factory_fn)
    for name, exc in raises.items():
        app.add_api_route(f"/_probe/{name}", _raising(exc), methods=["GET"])
    return app


def test_row_1_a_validation_failure_is_400_not_the_framework_default_422():
    """§七 第 1 行。FastAPI 默认给 422，本表要 400 —— 不覆盖这条默认行为，
    「问句超长」会拿到 422，前端按表写的分支全部落空。"""
    app = _app()
    app.add_api_route("/_probe/needs-count", _needs_count, methods=["POST"])
    with TestClient(app) as client:
        response = client.post("/_probe/needs-count", json={"count": SECRET})
    assert response.status_code == 400
    assert response.json()["error"] == {"code": "bad_request",
                                        "message": "请求参数不合法"}
    # 校验详情里含**输入值**（pydantic 的 errors() 带 input），而公众侧的输入值
    # 就是提问原文 —— 设计 §五 连审计都不记公众侧提问，响应里更不能回显
    assert SECRET not in response.text
    assert response.headers[request_id.HEADER] == response.json()["request_id"]


def _needs_count(body: _ProbeBody) -> dict:
    """校验用例的端点：请求体不合法时 FastAPI 抛 RequestValidationError。"""
    return {"count": body.count}


def test_row_1_a_validation_failure_does_not_put_the_input_value_in_the_logs(caplog):
    """同一个决定在日志那一侧也要成立：日志是旁路，公众侧提问同样不该落在那里
    （③b-1 的教训是静默丢信号，不是「什么都要记下来」）。"""
    app = _app()
    app.add_api_route("/_probe/needs-count", _needs_count, methods=["POST"])
    with TestClient(app) as client:
        with caplog.at_level(logging.WARNING):
            client.post("/_probe/needs-count", json={"count": SECRET})
    assert SECRET not in caplog.text
    # 但字段位置在（只有「不记值」与「什么都不记」都成立，这条才算对）
    assert any("count" in record.getMessage() for record in caplog.records)


def test_row_1_a_too_long_question_is_400():
    """§七 第 1 行的另一半（问句超长）。路由（任务 4）抛 errors.BadRequest，
    这里只钉「它确实映射成 400，且原文不外泄」。"""
    app = _app(long=errors.BadRequest(f"问句超过上限：{SECRET}"))
    with TestClient(app) as client:
        response = client.get("/_probe/long")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"
    assert SECRET not in response.text


def test_row_2_an_auth_failure_is_404_with_a_generic_message():
    """§七 第 2 行（设计 §二 第 9 条）：未认证 / 越权一律 404，且不解释原因 ——
    401/403 等于对未认证的探测者承认「这个路径存在」。"""
    app = _app(auth=AuthError(f"token 校验失败：{SECRET}"))
    with TestClient(app) as client:
        response = client.get("/_probe/auth")
    assert response.status_code == 404
    assert response.json()["error"] == {"code": "not_found", "message": "未找到"}
    assert SECRET not in response.text


def test_row_2_an_auth_failure_is_indistinguishable_from_a_missing_path():
    """同一条判据的兑现方式：拿 404 与「路径本来就不存在」逐字段比对。

    只断言「两者都是 404」不够 —— 一个多带一个 error code、或 message 措辞不同，
    探测者照样能枚举出哪些路径存在（这才是这条要求要防的东西）。
    """
    app = _app(auth=AuthError("token 过期"))
    with TestClient(app) as client:
        refused = client.get("/_probe/auth")
        missing = client.get("/api/v1/lawyers/secret-no-such-route")
    assert refused.status_code == missing.status_code == 404
    assert refused.json()["error"] == missing.json()["error"]


def test_row_3_rate_limiting_is_429_with_retry_after():
    """§七 第 3 行 + §六：429 必须带 Retry-After，客户端据此退避而不是立刻重试
    （没有它，被限流的一方会当场再打一次，把限流变成放大器）。"""
    app = _app(limited=errors.RateLimited(f"IP 限流：{SECRET}", retry_after=37))
    with TestClient(app) as client:
        response = client.get("/_probe/limited")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "37"
    assert response.json()["error"]["code"] == "rate_limited"
    assert SECRET not in response.text


def test_row_4_not_implemented_carries_the_reason():
    """§七 第 4 行要求「reason 写明」（前端据此提示用户），所以这一行的 detail 是
    有意公开的 —— 与 400/429/500/503 的通用文案形成对照，那几行的原文只进日志。"""
    app = _app(stub=errors.FeatureNotImplemented("历史案件检索（FR-5.3）尚未实现"))
    with TestClient(app) as client:
        response = client.get("/_probe/stub")
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "not_implemented"
    assert "FR-5.3" in response.json()["error"]["message"]


def test_row_5_a_system_failure_is_503_with_a_generic_message(caplog):
    """§七 第 5 行：系统故障 → 503，**通用文案**。

    ③b-1 真跑抓到过异常原文（"milvus down"）外泄给公众，故这里两个方向都断言：
    响应里没有原文，而日志里有 —— 只说「没泄露」的话，把日志一起删掉也照样绿。
    """
    app = _app(down=errors.ServiceUnavailable(f"milvus down：{SECRET}"))
    with caplog.at_level(logging.ERROR):
        with TestClient(app) as client:
            response = client.get("/_probe/down")
    assert response.status_code == 503
    assert response.json()["error"] == {"code": "service_unavailable",
                                        "message": "服务暂时不可用，请稍后再试"}
    assert SECRET not in response.text
    assert response.json()["request_id"], "503 也要带 request_id（要能拿它捞日志）"
    assert any(SECRET in record.getMessage() for record in caplog.records)
    # 「错误响应自己带头 + 中间件没有才补」唯一的见证就在这里：503 由 ExceptionMiddleware
    # 下发（在用户中间件**里面**），它的响应同时经过 _error_response 与请求 ID 中间件 ——
    # 中间件若改成无条件 append，两个同名头会被 httpx 拼成 "abc, abc"，与体里的 id 不等。
    # （413 与 500 各只经过一个写着：413 的响应自己不带这个头、500 根本不走用户中间件，
    # 在那两条上做同名头断言是**变不出红**的——实测。）
    assert response.headers.get_list(request_id.HEADER) == [response.json()["request_id"]]


def test_row_6_a_completed_answer_without_basis_is_200():
    """§七 第 6 行：问答**正常完成**但没有依据 → 200，由 status 字段表达。

    四个非故障终态一起参数化：写成 `status != "ok"` 就抛的实现在这里必红 ——
    那正是「把追问与拒答也当故障」的形态。
    """
    for status in ("ok", "abstain", "need_more_info", "out_of_scope"):
        app = _app()
        _add_qa_probe(app, ResultStub(status, "未找到相关依据。"))
        with TestClient(app) as client:
            response = client.post("/_probe/qa")
        assert response.status_code == 200, status
        assert response.json()["status"] == status


def test_row_6_a_chain_failure_is_503_instead_of_200():
    """同一条红线的另一半：故障走 503，且正文里不能出现「没查到」那类措辞 ——
    把故障答成「查无此条」正是 ③a / ③b-1 花力气分清的两件事。"""
    app = _app()
    _add_qa_probe(app, ResultStub(errors.FAILURE_STATUS, f"生成服务不可用：{SECRET}"))
    with TestClient(app) as client:
        response = client.post("/_probe/qa")
    assert response.status_code == 503
    assert response.json()["error"]["message"] == "服务暂时不可用，请稍后再试"
    assert SECRET not in response.text


def _add_qa_probe(app: FastAPI, result: ResultStub) -> None:
    """挂一个模拟问答路由的端点：过 errors.raise_if_answer_failed 后正常返回。

    用真的 raise_if_answer_failed（不是测试里另写一份判断）：任务 4 的路由要走的
    就是它，这里换掉等于测了别的东西。
    """
    async def qa() -> dict:
        errors.raise_if_answer_failed(result)
        return {"status": result.status, "answer": result.answer}

    app.add_api_route("/_probe/qa", qa, methods=["POST"])


def test_row_6_the_failure_status_copies_the_real_answerers_terminal_state():
    """errors.FAILURE_STATUS 是 answer.py 里那个字面量的**副本**（core/ 不能 import
    generation/，反向会成环），副本必须有护栏：真跑一次会失败的 Answerer，
    看它到底吐什么状态，再喂给 raise_if_answer_failed。

    没有这条，answer.py 把 "error" 改成别的写法时，接口层会把系统故障当正常回答
    返回 200 —— 而所有用例都会继续绿（它们都在自己的替身上用字符串比较）。
    """
    def no_llm(_side):
        raise RuntimeError(f"生成服务不可用：{SECRET}")

    # corpus 给一个不是 None 的对象：Answerer 会跳过 LawCorpus 构造（本用例不需要库）
    answerer = Answerer(corpus=object(), llm_factory=no_llm)
    result = answerer.answer("押金不退怎么办", side=SIDE_PUBLIC)
    assert result.status == errors.FAILURE_STATUS
    with pytest.raises(errors.ServiceUnavailable):
        errors.raise_if_answer_failed(result)


def test_row_7_an_unexpected_exception_is_500_with_a_generic_message(caplog):
    """§七 第 7 行。细节只进日志：堆栈是排查的唯一线索，而响应只给通用文案。

    raise_server_exceptions=False 是必须的：由 ServerErrorMiddleware 调用的兜底
    处理器**出完响应后仍会把异常重抛**（Starlette 的行为），测试客户端默认会把
    它再抛给用例 —— 那样断言不到响应体，而这条要测的正是响应体。
    """
    app = _app(boom=_Boom(SECRET))
    with caplog.at_level(logging.ERROR):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/_probe/boom")
    assert response.status_code == 500
    assert response.json()["error"] == {"code": "internal_error",
                                        "message": "服务内部错误"}
    assert SECRET not in response.text
    assert response.headers[request_id.HEADER] == response.json()["request_id"]
    # 500 不过用户中间件（见下条用例），这个头只能由 _error_response 挂 ——
    # 钉「只挂一次」；它挂两次时 httpx 拼成 "abc, abc"，与体里的 id 不相等
    assert response.headers.get_list(request_id.HEADER) == [response.json()["request_id"]]
    # 另一半：细节真的进了日志（只断言「没泄露」时，把日志一起删掉也不会红）
    logged = [record for record in caplog.records if record.exc_info]
    assert logged, "未处理的异常没有把堆栈写进日志"
    assert SECRET in str(logged[0].exc_info[1])


def test_row_7_the_500_log_carries_the_request_id(caplog):
    """500 那条日志必须能按 request_id 捞到（复审修的 Important，与上一条配对）。

    这条路径的 id **只能显式带上**：兜底处理器由 ServerErrorMiddleware 调用，它在所有
    用户中间件外面，RequestIdMiddleware 的 finally: reset 早已跑过 —— 实测同一请求的
    响应头/体都有 id，这条日志记的却是 "-"。而 500 是唯一带堆栈的一类，用户报一个 id
    却 grep 不到，等于白记。
    """
    app = _app(boom=_Boom(SECRET))
    with caplog.at_level(logging.ERROR):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/_probe/boom")
    rid = response.headers[request_id.HEADER]
    # 断言的是**记录上的 request_id 属性**（格式串 %(request_id)s 显示的就是它），
    # 不是「消息文本里出现过 id」：后者只要有人在文案里抄一遍 id 就绿，而日志行
    # 开头的 [request_id] 槽位仍然是 "-"，运维按格式去 grep 还是空手
    assert [r for r in caplog.records if getattr(r, "request_id", None) == rid], \
        "500 的日志没带本次请求的 request_id（格式串第一列会是 -）"


def test_only_the_not_implemented_row_may_publish_its_detail():
    """通用文案是**默认**（fail-closed），公开 detail 要显式打开，且只有 §七 第 4 行
    那一行该打开。加新异常类型时顺手打开这个开关，等于把「异常原文只进日志」这条
    悄悄改掉 —— 故把开关的归属钉死，而不是靠下一双眼睛。"""
    published = {cls.__name__ for cls in vars(errors).values()
                 if isinstance(cls, type) and issubclass(cls, errors.ApiError)
                 and cls.public_detail}
    assert published == {"FeatureNotImplemented"}


def test_the_status_to_code_table_covers_every_row_of_the_design_table():
    """§七 七行里六行有状态码（200 那行不是错误），一次列全当锚点：删掉任何一行
    都会让 public_error 退回兜底文案（"请求未能完成"），而那在只断言状态码的用例
    里完全看不出来。

    表外多出来的两行都在这里登记：413（任务 3 的体上限，按 RFC 9110 选）与
    401（任务 4 的登录失败 —— §七 只有「未认证访问受保护路径 → 404」那一行，
    而 /auth/login 是公开入口，把它 404 化会让前端分不清「登录失败」与
    「没这个接口」）。两行都是**有意**的表外新增，故写进锚点让改动可见。
    """
    assert {status: errors.public_error(status)[0] for status in sorted(errors.PUBLIC_ERRORS)} == {
        400: "bad_request", 401: "unauthorized", 404: "not_found",
        413: "payload_too_large", 429: "rate_limited", 500: "internal_error",
        501: "not_implemented", 503: "service_unavailable"}
