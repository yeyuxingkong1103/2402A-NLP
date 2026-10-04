"""应用层防滥用：IP 加盐哈希维度的滑动窗口限流 + 公众侧并发上限（设计 §六）。

存在的理由（设计 §一 的范围裁决，用户 2026-09-29「应用层自足」）：单机部署下进程内
计数够用，Redis（多实例共享计数）与 Nginx（连接级限流、client_max_body_size）留给
部署阶段。**这条边界的代价写在明面上**：本模块的计数**每实例独立** —— 横向扩到 N 个
实例后每个实例各有一份账，全站实际上限是 N 倍，且同一客户端可能被不同实例轮流放行。
要在多实例下成立，得把计数挪到 Redis（设计 §一 已裁决不在本轮）。

为什么是 ASGI 中间件而不是路由上的 Depends：受不受限由**路径**决定，写在路由上的话
「哪几条挂了限流」要逐个路由去数，漏挂一处的表现是那条端点静默敞开（本项目最怕的
静默失效形态）；而 `api/*.py` 的路由是任务 4/5 的交付，本任务一个字都不改。收成一个
路径谓词后，「哪些路径受限」是一份可测的清单（test_main_ratelimit 的矩阵用例逐个钉）。

三条判定与它们的边界（都是设计 §六 表的原文，取舍见各自注释与任务 6 报告）：
  ①限流窗口：匿名可读的三条（/public/qa、/lawyers/recommend、/law/*）**加 /auth/login**
    （终审 I-3：登录匿名可达、每次 ~40ms scrypt、与律师侧共用 40 worker，不限流时既能
    无上限撞库又能打满线程池）；
  ②并发上限：只覆盖走检索/生成的那两条（/law/* 不计 —— 另一个成本模型；登录也不占）；
  ③/healthz 与律师侧不受限：前者被 429 会与「实例不健康」在监控上不可分；后者是
    内网自己人，卡它等于让一个人的长问句把同事挡在门外。
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable

from starlette.responses import JSONResponse

from app.core import config, errors, request_id

logger = logging.getLogger(__name__)

# 匿名可读的三条（设计 §四、§六）。用**前缀**匹配而不是逐条列举：/law/{law_id}/articles
# /{article_no} 是动态段，抄一份清单等于把路由表存第二份（漂了的表现是某条新路径不设防）。
# 前缀不会造成绕过：Starlette 的路由是精确形状匹配，没有别的前缀能打到同一个处理函数。
PUBLIC_READ_PREFIXES = ("/api/v1/public/", "/api/v1/lawyers/recommend",
                        "/api/v1/law/")
# 计入并发上限的一档**比上面窄**：只有走检索/生成/LLM 的那两条。/law/*（nav 实测单次
# 87,580 字节、条文查询）成本模型完全不同 —— 一次索引查询 vs 秒级推理+GPU。把它算进
# 20 个槽，等于让「浏览器加载一次导航树」占掉一份推理配额（拿贵资源给便宜端点买单）；
# 它仍留在限流窗口里，故匿名爬取仍按每 IP 计数。这是任务 6 的裁决点，报告里单列一结。
CAP_PREFIXES = ("/api/v1/public/", "/api/v1/lawyers/recommend")

# 登录端点（终审 I-3）：它匿名可达、每次 scrypt ~40ms/16MiB（DUMMY_HASH 让查无此人
# 也照付），且与律师侧共用 AnyIO 的 40 个 worker —— 不限流时脚本可无上限撞库，还能
# 用它打满线程池把律师侧所有 def 端点饿住（AC-14/AC-20 的缺口）。它是**精确路径**而
# 不是前缀：/auth/ 下将来若加别的端点，默认不受限比默认受限容易发现（误伤的形态更响）。
LOGIN_PATH = "/api/v1/auth/login"
# 全量清扫的最小间隔（秒）：清扫是 O(键数)，而「键数到顶」正是有人在灌键的时刻 ——
# 每个新键都全扫等于把一条内存通道换成一条 CPU 通道（见 RateLimiter._make_room）
SWEEP_INTERVAL_S = 1.0
# 键里保留的哈希长度（hex 字符数）：128 bit 截断，碰撞在任何现实键数下都可忽略
# （生日界 2^64 量级），而键只是进程内内存里的一串字符，不落库、不外发
KEY_HEX_CHARS = 32


def is_limited_path(path: str) -> bool:
    """这条路径要不要过限流窗口（**清单受限**：不认识的一律放行）。"""
    return path == LOGIN_PATH or path.startswith(PUBLIC_READ_PREFIXES)


def is_capped_path(path: str) -> bool:
    """这条路径要不要占公众侧并发槽（比限流清单更窄，理由见 CAP_PREFIXES）。"""
    return path.startswith(CAP_PREFIXES)


def client_key(scope: dict, salt: str) -> str:
    """源 IP 的加盐哈希（设计 §六 原文：不存 IP 原文）。

    HMAC 而不是 sha256(salt + ip)：拼接式哈希在盐长度可变时有长度扩展那类边角问题，
    HMAC 是标准库给的、意图明确的那种「带密钥的哈希」。
    只取 scope["client"]，**不看 X-Forwarded-For**：那是一串客户端可任意伪造的头，
    信它等于给攻击者一个「每次换一个假 IP 就重置配额」的开关（fail-open 的经典形态）。
    代价写进报告：挂到反向代理后面时所有请求的 client 都是代理地址 → 全站共享一份
    配额，上公网前必须与 Nginx 一起把「可信代理」口径接上（设计 §一 把 Nginx 层留给
    部署阶段）。缺 client 时给一个固定的 "unknown" 键：它们共享一份配额，比共享一个
    会碰撞成别人的键安全，也比放行安全。
    """
    host = (scope.get("client") or ("", 0))[0] or "unknown"
    digest = hmac.new(salt.encode("utf-8"), host.encode("utf-8"),
                      hashlib.sha256).hexdigest()
    return digest[:KEY_HEX_CHARS]


class RateLimiter:
    """按 key 计数的滑动窗口：`allow(key, now=None) -> bool`。

    **时钟可注入**（构造参数 clock，或每次调用传 now）：否则「窗口滑过后恢复」只能
    用 sleep 写，而 sleep 出来的用例在慢机器上必红（本项目惯例）。
    窗口是 `[now - window_s, now]`：**滑动**而不是固定分段 —— 固定分段在整点前后能
    打出 2 倍配额（两段各一次），滑动窗口的持续速率恒为 limit/window。
    数据结构 `dict[key] -> deque[命中时刻]`：时间递增，过期从左侧弹出。

    并发：allow 会在**线程池的多个线程**里被调用（def 端点进线程池），全部状态改动
    收在一把锁里。**锁里不碰任何 I/O**（与 db/mysql.py 的串行化同一教训：锁住一段
    对话可以，锁住一次网络调用不行）：这里只有内存操作，纳秒量级。
    """

    def __init__(self, limit: int, window_s: float, *,
                 max_keys: int = config.DEFAULT_RATE_LIMIT_MAX_KEYS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window_s = window_s
        self.max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_sweep = -math.inf

    def allow(self, key: str, now: float | None = None) -> bool:
        """这一次请求放不放行；放行就把它记进窗口（拒绝的不记）。"""
        moment = self._now(now)
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                # 只在**造新键**时做容量治理：老键的过期由自己的 _expire 顺手做，
                # 每次请求都全表扫一遍就成了一条免费的 CPU 攻击面
                self._make_room(moment)
                hits = self._hits[key] = deque()
            self._expire(hits, moment)
            if len(hits) >= self.limit:
                return False
            hits.append(moment)
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        """还要几秒才有额度（≥1）；没满时 0。

        只在 allow 刚返回 False 时有意义 —— 两次调用之间窗口会滑动，读到的可能是个
        更大方的值，不影响正确性（Retry-After 是给客户端的退避建议，不是承诺）。
        最老那次命中滑出窗口后 len < limit，故它减去当下就是答案。
        """
        moment = self._now(now)
        with self._lock:
            hits = self._hits.get(key)
            if not hits or len(hits) < self.limit:
                return 0
            return max(1, math.ceil(hits[0] + self.window_s - moment))

    def key_count(self) -> int:
        """当前键数。淘汰/清扫有没有真的发生，只有它能看见（用例的观察点）。"""
        with self._lock:
            return len(self._hits)

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _expire(self, hits: deque, now: float) -> None:
        """把滑出窗口的命中从左侧弹掉（时间递增，第一个没出窗的就是边界）。"""
        deadline = now - self.window_s
        while hits and hits[0] <= deadline:
            hits.popleft()

    def _make_room(self, now: float) -> None:
        """键数到顶时给新键腾地方：先按间隔全量清扫，还满就淘汰一个键。

        全量清扫按**间隔**限流（SWEEP_INTERVAL_S）：它 O(键数)，若每个新键都全扫，
        灌键的人就拿我们的 CPU 打我们；一秒一次把这一步的代价压回常数级。
        清扫后还满（键都是活的）＝真的到了 max_keys 个活跃源，此时**淘汰**而不是
        拒绝新键：拒绝等于让「把表填满」成为一种拒绝服务，而填满它只需要造键、
        不需要请求成功。淘汰取插入序最老的键 —— 它最接近过期，丢掉的只是计数精度
        （那个键从一个空窗口重新开始），可用性不受影响。代价：分布式灌键时精度
        让位，换来的是内存有界（键数 ≤ max_keys）。
        """
        if len(self._hits) < self.max_keys:
            return
        if now - self._last_sweep >= SWEEP_INTERVAL_S:
            self._last_sweep = now
            deadline = now - self.window_s
            # 重建字典（而不是逐个 del）：一次遍历，且顺带清掉空 deque 的键
            self._hits = {k: d for k, d in self._hits.items()
                          if d and d[-1] > deadline}
        while len(self._hits) >= self.max_keys:
            self._hits.pop(next(iter(self._hits)))


class ConcurrencyGate:
    """进程内并发闸门：`try_acquire()` / `release()`，**不排队**。

    不排队（超出即拒）是设计 §六 的原文：公众侧同时处理数 ≤ 20，超了 429。排队看起来
    更友好，实际是把 20 个槽变成一条没有上限的队列 —— 用户等的还是同一份资源，而
    「等多久」我们答不出来（取决于 LLM 与 GPU 排队），429 + Retry-After 至少是诚实的。

    计数是**公众侧全局**的，不是每 IP 一个：AC-14 要的是「公众侧不得挤占律师侧资源」，
    即公众侧整体一条 20 的池子。代价（报告里记档）：一个人开 20 个未完成的请求就能把
    别的公众用户挡在门外 —— 律师侧不受影响，这正是这条池子存在的意义。
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._in_flight = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self.limit:
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        """还一个槽。**必须在 finally 里调**：异常路径漏释放 = 配额泄漏，几十次之后
        公众侧永久 429 —— 正是本项目最怕的静默失效形态（没有异常、没有日志，只有
        越来越频繁的 429）。没有可还的槽时不动：把计数减成负数等于闸门凭空多放行。
        """
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight


class RateLimitMiddleware:
    """公众侧的限流窗口 + 并发闸门（设计 §六）。装配点与顺序见 main.create_app。

    位置（与 main 里的注释是同一份口径）：在请求 ID **之内**、体上限**之外**。
      - 请求 ID 在外：429 与 413 一样要带 request_id（响应体里的 id 读的是 scope，
        由更外层的中间件放好）。
      - 体上限在内：那是「读一个头就拒」的极廉价判定，而限流要计的是**所有**打到
        公开端点的尝试（含体超限的那些），否则灌大包那条路不计数。
      - 业务与鉴权在内（更里）：**限流不先于鉴权**——这是刻意的，见 _reject 的注释：
        受限清单全是公开路径，律师侧与未知路径根本不在清单里，所以「未认证 + 高频」
        永远拿到 404 而不是 429（429 会告诉探测者这条路径存在，违反 §二 第 9 条）。

    阈值与限流器**懒建**：app 在 import 期就建好（main.py 末尾的模块级 app），配置读
    在 import 期会把「缺盐」变成 import 失败 —— 那是另一个失败面（测试收集、CLI 都会
    撞上），而这条配置只被限流路径需要。懒建仍是「每实例一次」，不是每请求一次。
    """

    def __init__(self, app, *, rate: RateLimiter | None = None,
                 gate: ConcurrencyGate | None = None,
                 salt_loader: Callable[[], str] = config.load_rate_salt) -> None:
        self.app = app
        self._salt_loader = salt_loader
        self._lock = threading.Lock()
        # 一对限流器存**一个属性**里：读的一侧不加锁（GIL 下读一个名字是原子的），
        # 而分开存两个属性时可能读到「窗口建好了、闸门还是 None」的中间态
        self._built: tuple[RateLimiter, ConcurrencyGate] | None = (
            (rate, gate) if rate is not None and gate is not None else None)

    def _limiters(self) -> tuple[RateLimiter, ConcurrencyGate]:
        built = self._built
        if built is None:
            with self._lock:
                if self._built is None:
                    settings = config.load_rate_limit()
                    self._built = (
                        RateLimiter(settings.limit, settings.window_s,
                                    max_keys=settings.max_keys),
                        ConcurrencyGate(settings.concurrency))
                built = self._built
        return built

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not is_limited_path(scope["path"]):
            # 非 HTTP（lifespan/websocket）与不受限的路径一律原样放行。**清单受限**
            # 而不是「默认受限、白名单豁免」：漏一条路径时前者的表现是它敞开、后者
            # 是它被 429 —— 两种都危险，但清单被矩阵用例逐条钉住，漏了会红
            await self.app(scope, receive, send)
            return
        rate, gate = self._limiters()
        key = client_key(scope, self._salt_loader())
        if not rate.allow(key):
            await self._reject(scope, receive, send, "窗口限流",
                               rate.retry_after(key))
            return
        capped = is_capped_path(scope["path"])
        if capped and not gate.try_acquire():
            await self._reject(scope, receive, send, "公众侧并发超限", 1)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            # 走到这里说明 try_acquire 成功过（失败在上面就 return 了），故用 capped
            # 当释放条件；释放放 finally 是硬要求：异常路径不释放 = 配额泄漏
            if capped:
                gate.release()

    async def _reject(self, scope, receive, send, detail: str,
                      retry_after: int) -> None:
        """出 429：统一错误形状 + Retry-After，并记一条 warning。

        日志不能省：限流记录是 AC-20 的证据链，也是「谁在被挡」的唯一线索（本项目
        的教训：吞掉的故障信号会让证据链连续几小时写不进去而全绿）。等级取 warning
        与 main 对 4xx 的口径一致。

        形状由 _limited_body 拼（与 main._error_response 同一形状的**第二处**实现）：
        core 不能 import main（反向依赖且成环），故这里不强行共用函数，改用一条用例
        把两条 429 路径的响应体逐字节比一遍 —— 形状漂了它当场红。
        """
        exc = errors.RateLimited(detail, retry_after=retry_after)
        logger.warning("限流拒绝 %s %s：%s（Retry-After=%d）",
                       scope.get("method"), scope.get("path"), detail, retry_after)
        response = JSONResponse(status_code=exc.status_code,
                                headers={"Retry-After": str(exc.retry_after)},
                                content=_limited_body(scope, exc))
        await response(scope, receive, send)


def _limited_body(scope: dict, exc: errors.RateLimited) -> dict:
    """429 的统一错误形状（设计 §七）：`{"request_id", "error": {code, message}}`。

    code/message 走 errors.PUBLIC_ERRORS 那张表（429 行：rate_limited / 通用文案），
    与 main._error_response 查的是同一张表 —— 文案只有一处定义，这里只拼信封。
    """
    code, message = errors.public_error(exc.status_code)
    return {"request_id": request_id.scope_request_id(scope),
            "error": {"code": code, "message": message}}
