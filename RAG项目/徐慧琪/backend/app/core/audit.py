"""审计的写入时机与 HTTP 侧胶水（设计 §五、§六审计段、§九第 7 步）。

分工：`db/audit.py` 是那张表的家（DDL + 写入 + 查询），本模块回答**什么时候写、
写什么值** —— 一条 ASGI 中间件，一行 = 一次请求的收尾（FR-6.3）。落成中间件而
不是各路由各调一次：逐条加调用会漏，漏掉的端点静默不入账（AC-11 的完整率是假的）。

**旁路口径**：写入失败**不阻断请求**，但必须 `logging.warning`（③b-1 的教训原话：
静默吞掉的故障信号会让证据链连续几小时写不进去而全绿）。**写入本身丢线程池**：
中间件跑在事件循环线程上，一次阻塞的 INSERT+commit 会按住整个服务（终审实测并发
/healthz 被拖到 0.853s）——「失败不阻断」之外还要「慢不阻断」。

**AC-19 的硬边界落在代码路径上**（设计 §五 的两列注释）：问句与身份各有一道路径
gate，公众可读前缀（/public/qa、/law/*、/lawyers/recommend）上**根本不取** —— 不是
「取到了再传 None」，而是连请求体与 Authorization 头都不看。用例双向断言落库的行：
公众侧为 NULL、律师侧在，只测一向挡不住「两边都不写了」这种退化。

**连接选择**（任务 7 的裁决）：用 `Services.conn`（共享的业务连接）—— 沿用它得
**每请求多开一条 TCP + 认证**，而 `db/mysql.py` 明写写路径都用它。代价写在明面上：
业务连接那把 RLock 会把审计写排在同批读的后面，线程池只保证它不按住事件循环，
不消除这句排队。写入失败**不 rollback**（同连接上可能压着业务未提交的写；单行
INSERT 失败是原子的），异常由 `_write` 记 warning。

**注入缝**：`app.state.audit_writer` 在时优先用它（签名与 `db.audit.record` 同形），
不在时用默认写入 —— 给用例的替身口（「写入抛异常 → 请求仍 200 且日志有 warning」），
不是生产开关。
"""
from __future__ import annotations

import anyio
import json
import logging
import threading
import time
import weakref

from app.core import ratelimit, request_id, security
from app.db import audit as audit_db

logger = logging.getLogger(__name__)

# 路径 → action（设计 §五 的七种）。**精确匹配**（方法 + 路径）而不是前缀：
# /api/v1/law/ 下既有 nav 又有动态的条文路径，前缀匹配会把两者混成一个 action。
# 未列入的路径（/healthz、未知路径、/cases/search）不审计 —— action 列没有它们
# 的取值，「先编一个」等于掺假数据；test_audit 有一条用例与真实路由表双向对照。
_ACTION_PATHS = {
    ("POST", "/api/v1/auth/login"): "login",
    ("POST", "/api/v1/qa"): "qa",
    ("POST", "/api/v1/search"): "search",
    ("POST", "/api/v1/public/qa"): "qa",
    ("GET", "/api/v1/law/nav"): "nav",
    ("GET", "/api/v1/lawyers/recommend"): "recommend",
    ("GET", "/api/v1/admin/audit/export"): "export",
}

# 取材路径（律师侧，**唯一会去看请求体的两个**）：公众侧不在这里 —— 这就是
# AC-19 的判据落在代码路径上的形态（public 分支没有取问句的代码可走）。
_QUESTION_PATHS = frozenset(("/api/v1/qa", "/api/v1/search"))
# 需要看**响应体**的 action：qa/search 取召回路径，login 取刚签发 token 里的身份
_BODY_ACTIONS = frozenset(("qa", "search", "login"))

# query_text 的列宽（VARCHAR(500)）：截断只可能发生在「输入层已判 400 的超长问句」
# 那一类。截断是刻意的：写不进去 = 那一行整个丢失，比留一段截断的现场糟得多。
_QUERY_TEXT_MAX = 500
# recall_path 的列宽（VARCHAR(255)）：按逗号边界截，不切半个条号
_RECALL_PATH_MAX = 255
_VERIFY_MAX = 64

# 表已建好的连接（WeakSet，**按连接**而不是全局一次）：测试会拿假连接当
# services.conn，标记它不该让真连接跳过建表；而 id() 集合会被回收重用（重用的
# 形态是「新连接被当成已建表的旧连接」，表没建成而失败只是 warning）。
_ensured: "weakref.WeakSet" = weakref.WeakSet()
_ensure_lock = threading.Lock()


def takes_question(scope: dict) -> bool:
    """这一条请求要不要**取问句原文** —— 全流程唯一的判断点（采集与取值共用）。

    取（抄请求体）与写（`_fields` 里取值）各写一个判据时，「两边不一致」是静默的：
    取而不写 = 一份没用的内存拷贝，写而不取 = 恒 NULL（看起来像「律师从不提问」）。
    收成一处之后，公众侧那条边界只需要在这一个地方成立 —— 改坏它（比如
    `return True`）会让 AC-19 的落库用例当场变红，而这是本模块最不能出事的一处。
    """
    return scope.get("path") in _QUESTION_PATHS


def action_for(method: str, path: str) -> str | None:
    """这条请求算哪一种 action（设计 §五 的七种之一）；不认识给 None。

    条文的路径是动态段（`/api/v1/law/{law_id}/articles/{article_no}`），故它在
    精确表之后单独判：形状必须恰好是「前缀 + 一段 + /articles/ + 一段」，多一段
    少一段都不算 —— 否则 `/api/v1/law/minfadian/articles/1/extra` 会被记成一次
    条文查询（它其实只会拿到 404）。
    """
    action = _ACTION_PATHS.get((method, path))
    if action is not None:
        return action
    if method == "GET" and path.startswith("/api/v1/law/"):
        parts = path[len("/api/v1/law/"):].split("/")
        if len(parts) == 3 and parts[1] == "articles" and parts[0] and parts[2]:
            return "article"
    return None


class AuditMiddleware:
    """请求收尾写一行 audit_log（设计 §六「audit.py 在请求收尾写一行」）。

    位置（main.create_app 的装配顺序）：在请求 ID 之内、限流之外。在 ID 之内：
    写失败那条 warning 才带得上 request_id（记录工厂从 contextvar 取，值由
    RequestIdMiddleware 设好）——「审计写不进去」必须能与某次请求对上号。在限流
    之外：被 429 拦下的请求也要入账 —— 429 那些行是「拦截率」的唯一来源，放在
    限流里面则被拦的请求一行都不落，指标恒为 0，而 0 看起来最正常。代价：灌流量
    时每条被拒请求也各写一行（写入量被限流器本身约束住，可接受）。

    体采集是**旁路 tap**：receive/send 原样透传，只在旁边记一份，不消费、不缓存
    下行流 —— 下游拿到的消息一字不差（与「读了 body 再重放」的写法不同，那种写法
    要自己处理 more_body、断连与重放，多一种出错方式）。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            # lifespan / websocket 原样放行（与 RequestIdMiddleware 同口径：
            # 不认识的协议一律不碰）
            await self.app(scope, receive, send)
            return
        action = action_for(scope.get("method", ""), scope.get("path", ""))
        if action is None:
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        capture = _Capture()
        # tap 的判据是路径（takes_question）而不是 action：公众侧与律师侧同为
        # action="qa"，按 action 判会把公众提问也抄进内存（AC-19 的边界就浑了）
        try:
            await self.app(scope, _tap_receive(receive, capture,
                                               takes_question(scope)),
                           _tap_send(send, capture, action))
        finally:
            # finally 而不是 except：取消（客户端断连）与异常都要留一行 ——
            # 「这次请求发生过」本身就是审计要记的东西。写入丢线程池（见 _write），
            # 绝不往外抛（审计是旁路，不许阻断主路）
            await _write(scope, action, capture, started)


class _Capture:
    """一次请求的现场。请求体只对律师侧问句路径采集，响应体只对 _BODY_ACTIONS。"""

    def __init__(self) -> None:
        self.request_body = b""
        self.response_body = b""
        self.status_code: int | None = None


def _tap_receive(receive, capture: _Capture, takes_question: bool):
    """包一层 receive：透传消息，只在律师侧问句路径上顺手抄一份请求体。"""
    if not takes_question:
        return receive

    async def tapped():
        message = await receive()
        if message["type"] == "http.request":
            capture.request_body += message.get("body", b"")
        return message
    return tapped


def _tap_send(send, capture: _Capture, action: str):
    """包一层 send：透传响应，抄下状态码（所有 action），响应体按需抄。"""
    keep_body = action in _BODY_ACTIONS
    async def tapped(message):
        if message["type"] == "http.response.start":
            capture.status_code = message["status"]
        elif keep_body and message["type"] == "http.response.body":
            capture.response_body += message.get("body", b"")
        await send(message)
    return tapped


async def _write(scope: dict, action: str, capture: _Capture, started: float) -> None:
    """组装并落库；**任何失败只留 warning**（审计是旁路，不许阻断请求）。

    **落库丢线程池**：中间件跑在事件循环线程上，同步的 INSERT+commit 会把整个
    服务按住（终审实测：慢写 1s 时并发 /healthz 被拖到 0.853s，控制组 0.001s）。
    取材（验签、JSON）留在循环里 —— 纯 CPU、微秒级。取消语义：run_sync 默认
    cancellable=False，断连取消会等这次写完成再传播（宁可丢一发响应也不丢行）。
    """
    try:
        conn, writer = _sink(scope)
        fields = _fields(scope, action, capture, started)
        await anyio.to_thread.run_sync(writer, conn, fields)
    except Exception as exc:  # noqa: BLE001 —— 见 docstring：这里只有 warning
        logger.warning("审计写入失败（不阻断请求）：action=%s 异常=%r", action, exc)


def _sink(scope: dict):
    """取（连接, 写入函数）。写入函数可被 app.state.audit_writer 替换（用例的替身口）。"""
    app = scope.get("app")
    if app is None:
        raise RuntimeError("scope 里没有 app，拿不到装配好的连接")
    state = app.state
    services = getattr(state, "services", None)
    if services is None or getattr(services, "conn", None) is None:
        # 没装配（启动失败 / 尚未启动）不算请求失败：记 warning 走人
        raise RuntimeError("重资源未装配，审计无处可写")
    return services.conn, getattr(state, "audit_writer", None) or _default_write


def _fields(scope: dict, action: str, capture: _Capture, started: float) -> dict:
    """一次请求 → 一行 audit_log 的十个字段（列与设计 §五 逐列对齐）。

    时间只有一处：ts 不在这里（库侧 DEFAULT CURRENT_TIMESTAMP）；latency_ms 在应用侧
    算 —— 两个时钟来源早晚分叉，而「差一点」在按时间窗串证据链时就是漏行。
    """
    user_id, role, team_id = _identity(scope, action, capture)
    question = None
    if takes_question(scope):  # 律师侧才走这一支（AC-19，判据见 takes_question）
        question = _question_text(capture)
    recall_path = None
    verify_result = None
    if action in ("qa", "search"):
        recall_path, verify_result = _answer_facts(action, capture)
    return {"request_id": request_id.scope_request_id(scope),
            "user_id": user_id, "role": role, "team_id": team_id,
            "action": action, "query_text": question,
            "recall_path": recall_path, "verify_result": verify_result,
            "status_code": capture.status_code or 500,
            "latency_ms": max(0, int((time.monotonic() - started) * 1000))}


def _identity(scope: dict, action: str, capture: _Capture):
    """(user_id, role, team_id)；取不到就三个 None（**绝不猜**，也绝不抛）。

    **公众可读路径恒匿名**（设计 §五「公众侧为 NULL」、AC-19）：即便带了有效
    token 也返回三个 None —— 身份只记在律师侧与登录路径上。前缀表与限流器同源，
    且它排在取 token 之前：这几条路径连 Authorization 头都不看。

    其余路径两条来源，都要求**验签**（代价：多一次微秒级 HMAC，与「身份必须
    可信」相比不值一提）：
      - 请求的 Authorization 头。不验签直接解 base64 也能拿到 id，但那样审计里
        的身份是**任何人都能伪造**的；验签失败（过期/伪造/未认证）一律不给身份 ——
        那种请求本来就会拿到 404，记 NULL 是诚实的。
      - login：请求上没有 token（正是来换 token 的），但响应体里有刚签发的
        access_token —— 拿它验签还原出登录的人；只认 200 的那次（401 没有 token）。
    """
    if scope.get("path", "").startswith(ratelimit.PUBLIC_READ_PREFIXES):
        return None, None, None
    token = _bearer_token(scope)
    if token is None and action == "login" and capture.status_code == 200:
        token = _token_from_login_body(capture.response_body)
    if token is None:
        return None, None, None
    try:
        user = security.verify_token(token, secret=security.load_secret())
    except Exception:  # noqa: BLE001 —— 身份取不到不是请求失败（见 docstring）
        return None, None, None
    return user.id, user.role.value, user.team_id


def _bearer_token(scope: dict) -> str | None:
    """从 Authorization 头取 token；形状不对给 None（这条路径不负责鉴权）。"""
    raw = dict(scope.get("headers") or []).get(b"authorization", b"")
    scheme, _, token = raw.decode("latin-1").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _token_from_login_body(body: bytes) -> str | None:
    """从登录响应体里取 access_token（只认形状，不做别的判断）。"""
    payload = _json(body)
    token = payload.get("access_token") if isinstance(payload, dict) else None
    return token if isinstance(token, str) and token else None


def _question_text(capture: _Capture) -> str | None:
    """律师侧问句原文（截到列宽）。非字符串/空串给 None —— 记 NULL 而不是空串：
    「没取到」与「问了个空」在事后追溯里是两回事。"""
    payload = _json(capture.request_body)
    if not isinstance(payload, dict):
        return None
    question = payload.get("question")
    if not isinstance(question, str) or not question:
        return None
    return question[:_QUERY_TEXT_MAX]


def _answer_facts(action: str, capture: _Capture):
    """(recall_path, verify_result)，都取自 QAResult 经响应契约带出的字段。

    - **recall_path**（FR-6.3 的「召回路径」= 这条答案由哪几个片段拼出）：qa 读
      `sources`（= `QAResult.sources`，即 `RetrievalResult.blocks` —— 送进提示词、
      被引用校验逐条查过的那些父块）；search 读 `blocks`（= `RetrievalResult.blocks`
      本身）。取 `article_no` 而不是 chunk_id：条号是人能核对的落点（与 schemas
      的 SOURCE_FIELDS 同一口径），chunk_id 在精确置顶那条路上还是 None。
    - **verify_result**：qa 读响应里的 `status`（= `QAResult.status` 原值）——
      `answer.py` 的生成→校验→重生成循环里，`ok` 就是「引用全过」，`abstain` 就是
      「两次都没过而降级」，`error` 是链路故障。设计 §五 只给了列名没给口径，
      本任务的选择是**存原值、不自己编词表**（编辑词表等于把编排层的一种新状态
      静默吞掉）；search 不生成也不校验，恒 NULL。

    解析失败（响应不是 JSON、键不在）一律给 NULL 而不是抛：这两格是补充信息，
    它们取不到不该让整行审计丢掉。
    """
    payload = _json(capture.response_body)
    if not isinstance(payload, dict):
        return None, None
    blocks = payload.get("sources" if action == "qa" else "blocks")
    numbers = []
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and block.get("article_no") is not None:
                numbers.append(str(block["article_no"]))
    recall_path = _join_capped(numbers) if numbers else None
    status = payload.get("status")
    verify_result = (status[:_VERIFY_MAX]
                     if action == "qa" and isinstance(status, str) and status
                     else None)
    return recall_path, verify_result


def _join_capped(numbers: list[str]) -> str:
    """逗号连接，超列宽时**按逗号边界**截（不切半个条号）。"""
    joined = ",".join(numbers)
    if len(joined) <= _RECALL_PATH_MAX:
        return joined
    cut = joined[:_RECALL_PATH_MAX]
    return cut[:cut.rfind(",")] if "," in cut else cut


def _json(raw: bytes):
    """尽力解析 JSON；失败给 None（旁路取材不该因畸形体而抛）。"""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None


def _default_write(conn, fields: dict) -> None:
    """默认写入：建表（每连接一次）+ 落一行。两处都可能抛，由 _write 接住记 warning。"""
    _ensure_once(conn)
    audit_db.record(conn, **fields)


def _ensure_once(conn) -> None:
    """本连接上建一次表（弱集合标记，理由见 _ensured 的注释）。

    时机与 users/fee_log 的「启动时无条件建」同义，只是落点挪到首次写入前：
    main.py 的非空行已顶在闸门边缘，lifespan 里加不进这一行；而「表要先存在」
    这件事必须由**会写它的那条路径**自己保证 —— 依赖运维先跑某个脚本的形态是
    部署时静默漏掉，此后每条审计都失败，而失败只是 warning（正是最该防的形态）。
    **拿 not in 先判再进锁**：热路径上（表已建好）不进锁，省一次线程同步。
    表已建好时它只多花一次集合查询；没建好时多一次 CREATE TABLE IF NOT EXISTS。
    """
    if conn in _ensured:
        return
    with _ensure_lock:
        if conn not in _ensured:
            audit_db.ensure_table(conn)
            _ensured.add(conn)
