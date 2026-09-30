# 接口**表面**的三条判据（不属于任何单条端点）：路由清单、自省文档、错方法探测。
#
# 为什么单独一个文件：这三条判据的对象是「这个应用对外露出什么形状」，而不是
# 「某条端点答什么」—— 路由清单要一次列全，自省文档与 405 都是**跨端点**的
# 暴露面；放进 test_api_lawyer.py 会让那个文件越过本项目 300 行的单文件闸门，
# 而按端点拆散它们（比如把 405 那条塞进 qa 的用例里）会让「表面」这件事
# 重新变得看不见，正是这三条要防的形态。
#
# 判据一律走真 create_app（路由注册、异常映射都来自真装配），重资源换替身。
import pytest
from fastapi.testclient import TestClient

from app import main
from tests._fakes_http import fake_factory, without_request_id


def _probe() -> TestClient:
    """真 app + 假 Services 的探针客户端。

    不进 with（lifespan 不跑）：本文件判的是路由表与错误映射，都不依赖重资源；
    不跑 lifespan 顺带省掉一次装配。raise_server_exceptions=False 让 500 以
    响应的形态出现而不是把异常抛进用例（这里的用例不该产生 500，但真出现时
    要能看到状态码）。
    """
    app = main.create_app(services_factory=fake_factory()[0])
    return TestClient(app, raise_server_exceptions=False)


# ---- Step 4.5：自省文档关掉（任务 4 审查挖到的跨任务缺口）----


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
def test_the_self_describing_docs_are_closed(path):
    """/openapi.json、/docs、/redoc 一律 404。

    这条缺口的形态：schema 会列出**全部已注册路径**（任务 5 一注册 /qa、/search、
    /cases/search，这些受保护路径就整表出现在公开 schema 里）—— 404 挡得住直接
    探测，挡不住来读文档的，§二 第 9 条辛苦维持的「不暴露路径存在性」当场只剩
    一半。故本任务把三份自省文档全关（main.create_app 的 FastAPI(...) 参数）。

    判据除状态码外还看**内容**：响应体里不许出现 "/api/v1/qa" 与
    "/api/v1/cases/search" —— 只断言 404 时，把文档换成一个同样返回 404 的
    定制页（或缓存了旧 schema 的反向代理）照样绿。
    """
    response = _probe().get(path)
    assert response.status_code == 404, f"{path} 仍可访问：{response.status_code}"
    assert "/api/v1/qa" not in response.text
    assert "/api/v1/cases/search" not in response.text


def test_the_registered_route_table_is_exactly_this_batchs_interface():
    """路由清单当字面量锚：设计 §四 的九条路径（任务 5 之后的状态）一次列全。

    「公众侧根本没有律师侧路由」这条隔离（§六 原文）在路由表层面成立的方式，
    就是这个清单：谁注册了什么，一眼可数，而不是靠读各模块的代码去拼。
    **任务 7/8 加 /admin/audit/export 与 /metrics 时要更新这份清单** —— 那时它
    变红是提醒（有人动了对外形状），不是故障。
    """
    app = main.create_app(services_factory=fake_factory()[0])
    assert {route.path for route in app.routes} == {
        "/healthz", "/api/v1/auth/login", "/api/v1/public/qa",
        "/api/v1/lawyers/recommend", "/api/v1/qa", "/api/v1/search",
        "/api/v1/cases/search", "/api/v1/law/nav",
        "/api/v1/law/{law_id}/articles/{article_no}"}


def test_a_wrong_method_is_indistinguishable_from_an_unknown_path():
    """错方法与未知路径必须**完全同形**（任务 6 修的缺口，原用例是「记录现状」）。

    修前：GET /api/v1/qa（这条路径只收 POST）返回 405 + `Allow: POST`，未知路径
    返回 404 —— 未认证的探测者换个方法打一发就能把「这里有这条路径」从 404 里
    分出来，`Allow` 还把正确方法一并奉上，与 §二 第 9 条正面冲突（今天就能被利用，
    不需要任何凭证）。

    修法在**错误映射**层：main._on_http_error 把状态码过一遍 errors.public_status
    （405 → 404），并连带丢掉 `Allow` 头。只把状态码改成 404 而留着 Allow 是
    **不够**的（等于没修），故判据三条一起：状态码 404、无 Allow 头、且与未知路径
    的响应体逐字节相同（挖掉 request_id）—— 第三条挡住的是「码一样但文案/字段不同」
    这种更细的缝。
    """
    probe = _probe()
    wrong_method = probe.get("/api/v1/qa")
    unknown_path = probe.get("/api/v1/no-such-path")
    assert wrong_method.status_code == 404
    assert "allow" not in wrong_method.headers
    assert without_request_id(wrong_method) == without_request_id(unknown_path)
