# 应用装配（main.py）的判据：lifespan 装配几次、失败时回不回收、中间件、healthz。
#
# 边界：这里**不测**业务路由、鉴权、限流、审计（任务 4~7）。异常映射单独放在
# test_main_errors.py：那份文件的每条用例都对着设计 §七 的一行，混在这里会让
# 「哪一行被覆盖了」看不出来。
#
# 判据一律走**真 create_app**（不是另起一个 FastAPI 再手工挂中间件）：手工挂的
# 那份是测试自己写的接线，接线错了照样绿。
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app import main
from app.core import config, factory, request_id
from app.db.milvus import COLLECTION as LAW_COLLECTION
from app.db.milvus import MILVUS_URI, get_client
from app.db.mysql import DEFAULT_CONFIG, connect
from app.generation.profiles import SIDE_PUBLIC
from tests._fakes_http import FakeServices, fake_factory


def _probe_app(services: FakeServices | None = None):
    """造一个真装配的应用 + 一个假 Services。返回 (app, 装配流水)。"""
    factory_fn, built = fake_factory(services)
    return main.create_app(services_factory=factory_fn), built


def _tmp_config(tmp_path) -> config.Config:
    """临时目录里的假模型路径：失败回收的用例因此不依赖本机模型文件在不在。

    两个占位文件名写在这里而不引用 config.REQUIRED_MODEL_FILES：后者是**被测
    代码**的常量，引用它的话，把所需文件改成别的名字时用例会跟着一起改而不会红。
    """
    paths = []
    for name, filename in (("embed", "sparse_linear.pt"),
                           ("rerank", "model.safetensors")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / filename).write_text("placeholder", encoding="utf-8")
        paths.append(str(directory))
    return config.Config(milvus_uri=MILVUS_URI, mysql=dict(DEFAULT_CONFIG),
                         embed_model_path=paths[0], reranker_model_path=paths[1])


def test_the_heavy_resources_are_built_once_and_not_per_request():
    """装配恰好一次，且**请求不会触发重装**（设计 §三 定调第 2 条）。

    只断言「装配过」是不够的：把 build_services 放进路由里，这句断言照样绿。
    故先记下 1，再打三个请求，再断言仍是 1 —— 模型加载是秒级的，每请求一次
    不会报错，只会让服务慢到不可用。
    """
    app, built = _probe_app()
    with TestClient(app) as client:
        assert len(built) == 1
        for _ in range(3):
            assert client.get("/healthz").status_code == 200
        assert len(built) == 1
    # 退出 with 就是关闭：重资源必须被释放一次（进程退出前把连接还回去）
    assert built[0].closed == 1


def test_a_failing_model_loader_closes_what_was_already_built(monkeypatch, tmp_path):
    """装配中途失败时，已建成的连接与客户端必须被关掉（任务 1 的交接项）。

    这条只能**注入会抛的假加载器**来测：真加载器在本机不会失败，而失败路径的
    形态是「连接泄漏」—— 进程内看不见、测试外看不见，靠读代码相信等于没测。
    判据收在 close() 被调到上：conn 与 client 各自是一个会记账的替身。
    """
    closed: list[str] = []

    class _Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    def boom(path):
        raise RuntimeError("精排模型加载失败")

    monkeypatch.setattr("app.core.factory.load_config", lambda: _tmp_config(tmp_path))
    monkeypatch.setattr("app.core.factory.connect", lambda **kw: _Resource("conn"))
    # 账号连接（任务 4）也要打桩：工厂里它是**另一个**构造点，漏掉它会真去连
    # 本机 MySQL —— 那样这条用例测的就不是回收，而是「库在不在」
    monkeypatch.setattr("app.core.factory.connect_autocommit",
                        lambda **kw: _Resource("account_conn"))
    monkeypatch.setattr("app.core.factory.get_client", lambda uri: _Resource("client"))
    monkeypatch.setattr("app.core.factory.load_model", lambda path: object())
    monkeypatch.setattr("app.core.factory.load_reranker", boom)
    # 用默认装配器（真 build_services）+ 打桩的加载器：这条测的正是工厂自己
    # 的回收，若改用替身工厂就等于把被测对象换掉了
    app = main.create_app()
    with pytest.raises(RuntimeError, match="精排模型加载失败"):
        with TestClient(app):
            pass
    assert closed == ["conn", "account_conn", "client"]


def test_a_failed_startup_leaves_no_services_on_the_app(monkeypatch, tmp_path):
    """启动失败后 healthz 必须如实报 503，而不是拿着一套半成品资源去探针。

    这条与上一条分工不同：上一条管「回收了没」，这条管「app.state 干净不干净」。
    两者可以同时出错也可以只错一个 —— 比如把装配结果先挂上 app.state 再校验，
    回收照做，但 app.state 上留了个已关闭的 Services。
    """
    def boom(path):
        raise RuntimeError("精排模型加载失败")

    monkeypatch.setattr("app.core.factory.load_config", lambda: _tmp_config(tmp_path))
    monkeypatch.setattr("app.core.factory.connect", lambda **kw: object())
    monkeypatch.setattr("app.core.factory.connect_autocommit", lambda **kw: object())
    monkeypatch.setattr("app.core.factory.get_client", lambda uri: object())
    monkeypatch.setattr("app.core.factory.load_model", lambda path: object())
    monkeypatch.setattr("app.core.factory.load_reranker", boom)
    app = main.create_app()
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass
    # 不跑 lifespan 直接问：没有 services 时 healthz 必须报 down（失败方向朝下），
    # 若这里报 500，说明它在用某个残缺对象 —— 那比「没装成」更难排查
    probe = TestClient(app, raise_server_exceptions=False)
    response = probe.get("/healthz")
    assert response.status_code == 503
    assert response.json()["checks"] == {"mysql": "down", "milvus": "down"}


def test_after_a_normal_shutdown_healthz_reports_down():
    """正常关闭后同样不许留 services：留着就会拿**已关闭**的连接去探针 ——
    真依赖下的表现是 500 或时对时错，而「服务已经停了」这件事在响应上看不出来。"""
    app, built = _probe_app()
    with TestClient(app):
        pass
    assert built[0].closed == 1
    probe = TestClient(app, raise_server_exceptions=False)
    assert probe.get("/healthz").status_code == 503


def test_the_module_level_app_serves_healthz():
    """uvicorn 的入口是模块级 app（`app.main:app`）。没有这条，把最后一行删掉
    不会有任何红灯，而部署命令会直接起不来 —— 那种失败落在运维现场，不落在测试里。"""
    assert isinstance(main.app, FastAPI)
    assert any(route.path == "/healthz" for route in main.app.routes)


def test_every_response_carries_a_request_id_of_the_same_value_in_header_and_body():
    """设计 §四：每个响应带 request_id。头与体同值 —— 审计（任务 7）用体里的，
    日志用头里的，两处不同值的话对齐就断了。"""
    app, _ = _probe_app()
    with TestClient(app) as client:
        response = client.get("/healthz")
    rid = response.headers[request_id.HEADER]
    assert rid == response.json()["request_id"]
    # 断言形状而不是「非空」：给个固定字符串也算非空
    assert len(rid) == 32 and all(ch in "0123456789abcdef" for ch in rid)


def test_a_wellformed_client_request_id_is_reused():
    """透传：调用方（前端 / 网关 / 压测脚本）自带的 ID 才有对齐价值，一律重发
    等于把关联信息丢掉。"""
    app, _ = _probe_app()
    with TestClient(app) as client:
        response = client.get("/healthz",
                              headers={request_id.HEADER: "trace-abc.123_X"})
    assert response.headers[request_id.HEADER] == "trace-abc.123_X"
    assert response.json()["request_id"] == "trace-abc.123_X"


@pytest.mark.parametrize("bogus", ["x" * (request_id.MAX_LEN + 1), "id with space"])
def test_a_bogus_client_request_id_is_replaced_instead_of_rejected(bogus):
    """非法 ID 换一个、**不拒绝请求**（ID 是旁路信息，为它挡下业务请求是拿主路
    换旁路）。两种情况各测一条：超长（灌日志）与不在允许集里的字符（响应头与
    日志里的伪造面）。非 ASCII 不在请求级参数里 —— httpx 自己就发不出去，
    那种值留给下面的 sanitize 用例（红得对不上因的用例会被「删断言」修掉）。"""
    app, _ = _probe_app()
    with TestClient(app) as client:
        response = client.get("/healthz", headers={request_id.HEADER: bogus})
    assert response.status_code == 200
    rid = response.headers[request_id.HEADER]
    assert rid != bogus and len(rid) == 32


@pytest.mark.parametrize("bogus", ["bad\nvalue", "bad\rvalue", "", "x" * 65,
                                   "带中文的-id"])
def test_sanitize_rejects_values_that_could_forge_or_flood_logs(bogus):
    """换行这类值只能在**函数层**测：httpx 自己就拒绝发含换行的头，走请求的用例
    会因为它而失败（红得对不上因，维护时容易被「删断言」修掉）。"""
    assert request_id.sanitize(bogus) == ""


def test_the_request_id_reaches_the_logs(caplog):
    """「后续日志」不是口号：请求内的日志必须带这次请求的 ID。

    用 503（探针故障）触发日志，而不是在路由里手写一句 logger —— 那才是真实
    形态：出错时才最需要按 id 捞日志。
    """
    app, _ = _probe_app(FakeServices(mysql_ok=False))
    with TestClient(app) as client:
        with caplog.at_level(logging.ERROR):
            rid = client.get("/healthz").headers[request_id.HEADER]
    assert [r for r in caplog.records
            if getattr(r, "request_id", None) == rid], "请求内的日志没带 request_id"


def test_logs_outside_a_request_carry_the_placeholder(caplog):
    """请求外的日志给 '-'：让「没进请求」与「进了但 id 空」在输出上可区分。
    顺带钉住「记录工厂真的被装上了」—— 没装上时这里会 AttributeError 而不是静默绿。"""
    with caplog.at_level(logging.ERROR):
        logging.getLogger("app.probe").error("不在任何请求里的一条日志")
    # 与字面量比而不是与 NO_REQUEST 常量比：和常量比是恒真的（改常量一起漂），
    # 而占位符是对外可观测的 —— 换一个字面量得有人再看一眼
    assert caplog.records[-1].request_id == "-" == request_id.NO_REQUEST


def _add_echo(app: FastAPI, seen: list[int]) -> None:
    """往 app 上挂一个把请求体长度记下来的探针路由（体上限用例的观察点）。"""
    async def echo(request: Request) -> dict:
        body = await request.body()
        seen.append(len(body))
        return {"length": len(body)}

    app.add_api_route("/_probe/echo", echo, methods=["POST"])


def test_a_body_over_the_limit_is_rejected_before_the_route_runs():
    """超限 → 413（选它而不是 400 的理由见 BodySizeLimitMiddleware 的 docstring）。"""
    seen: list[int] = []
    app, _ = _probe_app()
    _add_echo(app, seen)
    with TestClient(app) as client:
        response = client.post("/_probe/echo",
                               content=b"x" * (main.MAX_BODY_BYTES + 1))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    # 关键的一条：路由**没跑**。只断言状态码的话，「先跑完路由再回 413」也照样绿，
    # 而那时上限形同虚设（体已经读进内存了）
    assert seen == []
    # 413 由中间件出，不走异常映射 —— 它的 request_id 只能来自更外层的请求 ID
    # 中间件，这条断言同时把两者的顺序钉住了
    assert response.headers[request_id.HEADER] == response.json()["request_id"]
    # 413 的响应体里只有 id、**没有**头（这一个头是中间件挂的，见 BodySizeLimitMiddleware），
    # 故这里钉的是「中间件恰好挂一次」；判据用同名头的**份数**而不是取值 ——
    # 挂两次时 httpx 拼成 "abc, abc"，与体里的 id 不等（头体对齐断掉）
    assert response.headers.get_list(request_id.HEADER) == [response.json()["request_id"]]


def test_a_body_exactly_at_the_limit_passes():
    """边界：恰好等于上限要放行。没有这条，把上限改成 `>=` 的变异不会红 ——
    而它会拒掉合法请求，且只在「刚好那么长」时发生。"""
    seen: list[int] = []
    app, _ = _probe_app()
    _add_echo(app, seen)
    # 上限本身也要有字面量锚：边界用例拿常量当输入，把上限改成 100MB 它照样绿 ——
    # 而那时这层保护等于没有（体上限拦的是整包上传，不是问句长度）
    assert main.MAX_BODY_BYTES == 64 * 1024
    with TestClient(app) as client:
        response = client.post("/_probe/echo", content=b"a" * main.MAX_BODY_BYTES)
    assert response.status_code == 200
    assert seen == [main.MAX_BODY_BYTES]


def test_the_probes_read_the_app_connections_without_writing():
    """两条一个都少不得：判据是**探针真发了 SQL 且全是 SELECT**。

    只断言状态码的话，「healthz 自己新开一个连接」也照样绿 —— 而它证明的是
    「库还在」，不是「这个实例手里的连接还能用」（MySQL wait_timeout 到期后正是
    这种形态：库活着、实例的读路径已断）。只查「发了 SQL」则可以放行一条 INSERT：
    监控本身就成了能改生产数据的定时任务。
    """
    services = FakeServices()
    app, _ = _probe_app(services)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert services.conn.statements, "healthz 没有用 app.state 里的连接"
    assert all(sql.lstrip().upper().startswith("SELECT")
               for sql in services.conn.statements)
    # 表名是这条探针的全部意义所在：退化成 "SELECT 1" 时上面两条全绿，
    # 而它指向空库/错库照样报 ok —— 正是 _probe_mysql 的 docstring 论证要避免的形态
    assert any("FROM article" in sql for sql in services.conn.statements)


def test_the_milvus_probe_asks_the_production_collection():
    """探针问的必须是**生产集合名**：写成别的名字（或不存在的名字）时，健康检查
    在真机上照样报 ok，而检索链路是全空的 —— 那是「健康但无用」的最坏形态。"""
    services = FakeServices()
    app, _ = _probe_app(services)
    with TestClient(app) as client:
        client.get("/healthz")
    assert services.client.asked == [LAW_COLLECTION]


def test_healthz_reports_each_dependency_and_leaks_no_reason():
    """依赖挂了 → 503、逐项报出哪个挂了，而异常原文只进日志（§七 的通用文案；
    ③b-1 真跑抓到过原文外泄给公众）。一个依赖挂不代表另一个也挂 —— 逐项报出来，
    运维才知道该修哪个、监控才不至于把整台机器摘掉。"""
    app, _ = _probe_app(FakeServices(mysql_ok=False))
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "down"
    assert response.json()["checks"] == {"mysql": "down", "milvus": "ok"}
    assert "gone away" not in response.text


def _dependencies_online() -> bool:
    """MySQL 与 Milvus 都在线才跑真依赖用例（离线时跳过，而不是假装通过）。"""
    try:
        connect().close()
        return get_client(MILVUS_URI).has_collection(LAW_COLLECTION)
    except Exception:
        return False


@pytest.mark.skipif(not _dependencies_online(), reason="MySQL/Milvus 未在线")
def test_healthz_connects_to_the_real_mysql_and_milvus():
    """项目惯例：能用真依赖就不用替身。注入的工厂只省掉三个模型字段（healthz
    用不到 encoder/reranker/answerer，加载它们要几十秒），连的是**真** MySQL 与
    真 Milvus —— 这条要回答的正是「app 起得来、依赖真接得上」，替身答不了它。"""
    def factory_fn():
        # account_conn 给 None：这两条用例只打 /healthz，不碰账号路径
        return factory.Services(conn=connect(), client=get_client(MILVUS_URI),
                                encoder=None, reranker=None, answerer=None,
                                account_conn=None)

    app = main.create_app(services_factory=factory_fn)
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["checks"] == {"mysql": "ok", "milvus": "ok"}
    assert response.json()["request_id"] == response.headers[request_id.HEADER]


@pytest.mark.skipif(not _dependencies_online(), reason="MySQL/Milvus 未在线")
def test_healthz_survives_being_hit_concurrently():
    """健康检查必须扛得住并发打（复审 Critical 的端到端见证）。

    编排层会周期并发地打它（liveness + readiness + 人工 curl 重叠），而它查的是
    **app.state 里那条共享连接**。修复前并发打这条端点会把 MySQL 协议搅坏：响应
    变成随机的 503，且连接停在损坏状态、后续业务请求一起挂。
    判据是「每一个响应都 200 且 mysql ok」—— 只说「有 200 的」时，改前那种
    大部分 503、偶尔 200 的形态照样绿。真 MySQL 真 Milvus，8 线程 × 4 次。
    """
    def factory_fn():
        # account_conn 给 None：这两条用例只打 /healthz，不碰账号路径
        return factory.Services(conn=connect(), client=get_client(MILVUS_URI),
                                encoder=None, reranker=None, answerer=None,
                                account_conn=None)

    app = main.create_app(services_factory=factory_fn)
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: client.get("/healthz"), range(32)))
    assert [r.status_code for r in responses] == [200] * 32
    assert all(r.json()["checks"]["mysql"] == "ok" for r in responses)
