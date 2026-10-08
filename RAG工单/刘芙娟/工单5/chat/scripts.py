"""Redis 的 Key 构造、Lua 脚本、以及存储层的错误构造。

**本模块没有类，只有常量和纯函数** —— 它是 `store.py` 与 `messages.py`
共用的地基，拆出来的直接理由是 300 行上限（FR-037 / SC-009），
实质理由是这三样东西**不属于任何一个存储类**：
Key 的拼法、脚本的文本、错误的形状，都是跨类不变的约定。

---

## 为什么 Key 与脚本必须各只有一份定义

写死 `"medical:chat:session:..."` 字面量、或者把脚本复制一份到另一个模块，
后果都是**静默的**：

    Key 写错 → 读和写落在两条不同的 Key 上，表现为"写进去了但读不出来"
    脚本两份 → 改了一处没改另一处，表现为"某条路径的 TTL 不刷新"

两者都不会报错。因此 Key 由模板常量统一生成，脚本由本模块唯一持有。
"""

from __future__ import annotations

import secrets
import time

from . import (
    GUEST_USER_PREFIX,
    GUEST_USER_RANDOM_CHARS,
    META_KEY_TMPL,
    SESSION_KEY_TMPL,
    ChatError,
)

__all__ = [
    "guest_user_id",
    "now",
    "session_key",
    "meta_key",
    "atomic_append_script",
    "atomic_rename_script",
    "session_missing",
    "dep_error",
]

# 元数据 Hash 的字段名。
#
# 与 `Message` 的 JSON 字段是两套命名，刻意分开：前者是**存储布局**
# （改它要迁移数据），后者是**消息格式**（改它要改序列化）。
# 混用会让"给元数据加一个字段"和"给消息加一个字段"看起来是同一件事。
FIELD_USER_ID = "user_id"
FIELD_CREATED_AT = "created_at"
FIELD_LAST_ACTIVE_AT = "last_active_at"

META_FIELDS: tuple[str, ...] = (FIELD_USER_ID, FIELD_CREATED_AT, FIELD_LAST_ACTIVE_AT)


def now() -> int:
    """当前 Unix 秒。

    独立成函数而不是散落 `int(time.time())`：测试若要冻结时间，
    只需替换这一个点（`monkeypatch.setattr(scripts, "now", lambda: 0)`），
    而不是在每个调用处打补丁。`store.py` 与 `messages.py` 共用它 ——
    两份实现会让"创建时间"与"消息时间"在同一个请求里差出一秒。
    """

    return int(time.time())


def guest_user_id() -> str:
    """自动生成一个发起者标识：`guest-` + 8 位随机字符（Q5 裁决）。

    ⚠️ 熵远低于 `session_id`（那里是 256 bit 的凭证），这是**刻意的**：
    `user_id` 不参与任何校验、不保护任何东西，它只是一个显示名。
    给它高熵只会换来一串没人愿意看的字符。

    用 `token_urlsafe` 而不是 `uuid4`：后者带连字符与固定长度，
    看起来像内部 ID；而 `guest-` 前缀让人一眼看出"这是系统给的默认值，
    你可以改"。
    """

    return GUEST_USER_PREFIX + secrets.token_urlsafe(16)[:GUEST_USER_RANDOM_CHARS]


def session_key(session_id: str) -> str:
    return SESSION_KEY_TMPL.format(session_id=session_id)


def meta_key(session_id: str) -> str:
    return META_KEY_TMPL.format(session_id=session_id)


# ⚠️ 原子写入脚本。**`EXISTS` 判定被放在脚本内，而不是调用方先查一次。**
#
# 先查后写会留下一个竞态窗口：两次请求之间会话恰好过期，第二次写入就给一个
# 已过期的会话"顺手"补上了元数据 —— 而那是一个残缺的会话（有 `last_active_at`，
# 没有 `created_at`），之后 `get_meta` 会以一个莫名其妙的方式失败。
# 把判定做进同一个原子单元，这个窗口就不存在。
#
# 返回 `LLEN`（而不是 `OK`）：调用方需要知道写入后的条数，
# 而"写完再查一次"就不原子了。RPUSH 后 LLEN 恒 ≥ 1，因此 **0 是哨兵值**，
# 唯一来源是开头那个 `return 0` —— 即"会话不存在"。
#
# ⚠️ `LTRIM` 用负下标（`-N -1`）保留**最后** N 条。
#    写成 `LTRIM 0 N-1` 会保留**最早**的 N 条 —— 同样返回 N 条，
#    但用户刚说的话看不见了，倒记得半小时前的，且没有任何报错。
atomic_append_script = """
if redis.call('EXISTS', KEYS[2]) == 0 then
  return 0
end
redis.call('RPUSH', KEYS[1], ARGV[1])
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[3]), -1)
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('HSET', KEYS[2], 'last_active_at', ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[2])
return redis.call('LLEN', KEYS[1])
"""

# ⚠️ 改名同样要走 Lua，**不能用 `pipeline(transaction=True)` 里的 `EXISTS` + `HSET`**。
#
# 这是实现期实测到的一个真实缺陷：`pipeline` 虽然把命令包在 MULTI/EXEC 里原子执行，
# 但 `EXISTS` 的结果**只是返回值**，拦不住后面的 `HSET` ——
# 对一个不存在的 Key 执行 `HSET` 会把它创建出来。
# 于是"给不存在的会话改名"会留下一条只有 `user_id`、没有 `created_at` 的残缺元数据，
# 而之后每次 `get_meta` 都会以"缺少字段"失败，失败的时机与原因隔了三层。
atomic_rename_script = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
return 1
"""


def session_missing(session_id: str) -> ChatError:
    """会话不存在。**是 404 而不是 503。**

    用 `session_missing=True` 标志而不是让路由匹配文案 ——
    文案会被改写，而标志位不会被静默改掉（见 `ChatError` 的 docstring）。
    """

    return ChatError(
        "会话不存在或已过期（session_id=%s）" % session_id, session_missing=True
    )


def dep_error(exc: BaseException, action: str) -> ChatError:
    """把 Redis 客户端异常转成 `ChatError`。

    ⚠️ **不回显连接串** —— `REDIS_URL` 可能含密码
    （`redis://:password@host:6379/0`）。只报动作与异常类型。
    与 `backend/retrieve/store.py` 中"错误信息里 MUST NOT 回显 token"同一处置。
    """

    return ChatError(
        "%s失败：%s（Redis 不可达或返回错误）" % (action, type(exc).__name__)
    )
