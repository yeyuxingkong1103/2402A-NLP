"""请求 ID：生成 / 透传、写进 scope、并注入每一条日志记录。

存在的理由（设计 §四）：每个响应带 request_id，与审计（任务 7）、日志对齐 ——
排查时拿公众报的一个 id 要能一路捞到那一次的检索与生成。ID 本身在中间件里定，
但它还有两个使用者，故独立成模块：

  - **日志**：用 setLogRecordFactory 把当前请求 ID 塞进每条 LogRecord。为什么
    不用 logging.Filter：过滤器挂在**处理器**上才生效（日志器的过滤器只管
    直接打在它身上的记录，传播过来的记录不看祖先日志器的 filters），要给
    uvicorn、本应用、将来 audit 的各路处理器逐个挂一遍，漏一个就静默丢 id。
    记录工厂是唯一一处能覆盖「任何日志器、任何处理器」的接缝。
  - **响应头**与错误体：主键名与取值规则在这里，中间件与异常处理器共用。

透传与否的裁决（本模块的本任务决策）：**透传但校验**。理由是排查场景里调用方
（前端、网关、压测脚本）自带的 ID 才有对齐价值，一律重发等于把关联信息丢掉；
而「原样透传」不能收 —— 值会进日志与响应头两处，不校验就能塞换行（日志伪造）
或超长串（灌爆日志）。故只接受长度 ≤ MAX_LEN 的 [A-Za-z0-9._-]，其余一律重新生成：
非法值不报错、不当场拒绝 —— 请求 ID 是旁路信息，为它挡下一个业务请求是拿主路
去换旁路。
"""
from __future__ import annotations

import contextvars
import logging
import uuid

# 对外响应头与入站请求头同名同义（X-Request-ID 是事实标准）
HEADER = "X-Request-ID"
# scope["state"] 与本模块 contextvar 用同一个键名，避免两处名字漂移
STATE_KEY = "request_id"
# 日志格式：部署侧（uvicorn --log-config 或本模块的 install 默认）用它才能显出 ID。
# 单独暴露成常量而不是埋进 install：启动命令里也要引用同一个格式串
LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
# 未在请求内时日志里的占位：显式给一个值（而不是漏键），让「没进请求」与
# 「进了但 ID 空」在输出上可分
NO_REQUEST = "-"

MAX_LEN = 64
# 允许的字符集刻意很窄：够 UUID/base62/带连字符的追踪号用，又不含换行与控制符
_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")

_current: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fl_request_id", default=NO_REQUEST)
_installed = False


def new_id() -> str:
    """生成一个新的请求 ID。uuid4 的十六进制串：32 字符、够短、不泄露时间与主机。"""
    return uuid.uuid4().hex


def sanitize(value: str) -> str:
    """校验外部传入的 ID：合法则原样返回，否则返回空串（调用方据此改用新 ID）。"""
    if 0 < len(value) <= MAX_LEN and all(ch in _ALLOWED for ch in value):
        return value
    return ""


def bind(scope: dict) -> str:
    """从入站头取（或生成）请求 ID，写进 scope["state"]，返回它。

    scope 而不是 Request：中间件在**调用下游之前**就要把 state 放好，而那时
    还没有 Request 对象；异常处理器拿到的 Request 读的是同一个 scope["state"]，
    两边天然共享，不必再把 ID 往下传一层。

    头的解码用 latin-1：HTTP 头是任意字节，utf-8 解码会在非 ASCII 字节上抛错，
    而为一个畸形头把请求打成 500 是最不划算的失败方向（latin-1 永不抛错，
    畸形字节随后被 sanitize 拒掉）。
    """
    raw = dict(scope.get("headers") or []).get(HEADER.lower().encode("ascii"), b"")
    rid = sanitize(raw.decode("latin-1"))
    rid = rid or new_id()
    # setdefault：uvicorn 在 scope 里已放了 "state"（ASGI lifespan state），
    # 直接赋值会把它的内容覆盖掉
    scope.setdefault("state", {})[STATE_KEY] = rid
    return rid


def current() -> str:
    """当前请求的 ID；不在请求内时返回 NO_REQUEST。"""
    return _current.get()


def scope_request_id(scope: dict) -> str:
    """从 scope 取请求 ID（中间件已放好）；没有则给占位符。

    任务 6 从 main.py 挪来（那边顶在行数闸门上）：它是 STATE_KEY 的读取口，与
    bind 是一对，放在同一模块里才不会出现「键名改了、读的地方没跟上」。
    中间件按理总是先跑过，但异常处理器也会被**没有经过中间件**的调用触发
    （单测直接调处理器、或将来换 ASGI 服务器时的启动期请求），所以这里不假设，
    缺了就给 NO_REQUEST —— 缺 id 的响应仍然是合法响应，缺 id 的日志仍然可读。
    """
    return (scope.get("state") or {}).get(STATE_KEY) or NO_REQUEST


def set_current(rid: str):
    """设置当前请求 ID，返回 contextvar 的 token（调用方负责在 finally 里 reset）。"""
    return _current.set(rid)


def reset(token) -> None:
    """撤销 set_current。收成函数是为了让中间件里只有一处碰 contextvar 内部结构。"""
    _current.reset(token)


def install() -> None:
    """把请求 ID 注入每条日志记录；**幂等**（create_app 可能被多次调用）。

    注入的是「记录上多了 request_id 属性」，要显出 ID 还得格式串里有
    %(request_id)s（LOG_FORMAT）。basicConfig 在根日志器已有处理器时是空操作，
    所以 uvicorn 或测试自己的日志配置不会被这里覆盖。
    """
    global _installed
    if _installed:
        return
    _installed = True
    previous = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.request_id = current()
        return record

    logging.setLogRecordFactory(record_factory)
    logging.basicConfig(format=LOG_FORMAT)
