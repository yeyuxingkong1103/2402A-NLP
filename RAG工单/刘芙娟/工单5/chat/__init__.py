"""S11 多轮对话（Redis 会话）的实现模块。

对外的入口有两个：

    · `backend/chat/dialogue.py` —— 单轮问答编排（服务端 `chat_routes.py` 调用）
    · `backend/chat/service.py`  —— 会话 CRUD 与上下文窗口（六个方法）

模块划分（按「什么时机可能出错」切分，与 `backend/query/`、`backend/retrieve/`、
`backend/generate/` 同一取向）：

    models    内部数据模型            —— 纯数据，不依赖 redis
    codec     序列化（可替换）         —— 纯计算；加密扩展点在此
    redact    隐私脱敏                —— 纯计算
    tokens    token 估算              —— 纯计算
    context   上下文窗口裁剪           —— 纯计算
    prompt    角色设定装载             —— 启动期读盘
    audit     审计日志                —— 纯记录
    store     会话存储                —— **唯一碰 Redis 的地方**
    service   会话 CRUD + 窗口         —— 请求期；失败是"会话不存在"
    dialogue  单轮编排                —— 请求期；失败是"检索/生成失败"

⚠️ **本包只放常量，不放逻辑** —— 与 `backend/api/__init__.py` 同一取向，
让「唯一处」这类约束在结构上可见。

---

## 与既有模块的边界

本包 MUST NOT 自己实现检索或生成 —— 它调用 `backend/retrieve/service.py`（I-05）
与 `backend/generate/service.py`（I-06），与 `/ask` 走**同一条**链路。
若 chat 另起一套，CLI 验证的链路与用户实际跑的链路就不再是同一条
（`backend/retrieve/service.py` 的模块文档写明「只有一条编排实现」）。

逐字话术（紧急前置、免责声明）的拼接 **不在本包内** —— 它由
`backend/api/pipeline.py:assemble_answer_text()` 唯一提供（D3 / FR-017）。
本包只是调用它，MUST NOT 复制那两句文本。
"""

from __future__ import annotations

# ---- Redis Key 前缀 ----
#
# ⚠️ 本常量是全仓**唯一**的 Key 前缀定义处。`store.py` 用它拼 Key，
# 测试与运维脚本也引用它 —— 任何地方写死 `"medical:chat:..."` 字面量
# 都会让"改前缀"变成一次全仓搜索，而漏掉一处就是静默的读写不同 Key。

KEY_PREFIX = "medical:chat:"
SESSION_KEY_TMPL = KEY_PREFIX + "session:{session_id}"
META_KEY_TMPL = KEY_PREFIX + "meta:{session_id}"

# ---- 默认值 ----
#
# ⚠️ 这些**只是 .env 没写时用什么**。真实取值一律走 `backend/api/config.py`
# （那里是配置项的唯一声明处），由 `serve.py` 在启动期读入并注入
# `ChatStore` —— 请求路径 MUST NOT 读环境变量（FR-035）。
#
# 放在本文件而不是 `store.py`，是因为 `service.py` 与测试都要用它们；
# 两处各写一份必然会漂。

# 会话存活时长（秒）。每次 append_message 刷新（FR-003）。
#
# ⚠️ **2026-10-08 从 1800（30 分钟）改为一年的产品裁决。**
#
# 原因：30 分钟的会话对使用者没有意义 —— 他关掉页面、第二天回来，
# 对话全没了，而**本机还记着那些标识**（localStorage 不过期），
# 界面显示的是一列点开就报"已过期"的条目。那不是"数据最小化"，
# 是"记着一堆打不开的东西"，比干脆不记更糟。
#
# ⚠️ **代价要写明**：这实质上等同于"不过期"。一年之后回来看，
# 这些对话同样会消失。若确需永久留存，MUST 另起规格讨论
# （constitution 的「知识与数据边界」与数据最小化原则都建立在"会过期"上）。
#
# 之所以仍保留一个 TTL 而不是彻底去掉：**它是唯一的兜底**。
# 去掉 `EXPIRE` 之后 Redis 里的会话只会增不会减，没有任何机制回收。
DEFAULT_SESSION_TTL = 365 * 24 * 3600   # 31536000 秒

# 单次送入模型的 token 预算。**这是估算的上限**，不是精确值（FR-011）。
DEFAULT_MAX_CONTEXT_TOKENS = 4000

# Redis List 的**存储**条数上限。与新消息写入在同一原子单元内截断（R2/R4）。
#
# ⚠️ 它与 `DEFAULT_MAX_CONTEXT_TOKENS` 是**两个独立的截断机制**（R4）：
#     TTL/LTRIM 管"存多少"，token 预算管"送多少给模型"。
#     把它们混为一谈的后果是"以为 TTL 会清理，所以不必限长" ——
#     而一个在 TTL 内被高频使用的会话永远不会触发 TTL，List 会无界增长。
DEFAULT_MAX_HISTORY_MESSAGES = 50

# 自动生成的发起者标识的前缀。
#
# ⚠️ 用**可读前缀**而非 uuid4：这个值要显示给使用者看，并且允许他改（Q5 裁决）。
#     `session_id` 用 token_urlsafe 是因为它是**凭证**，形态上就该与普通 ID
#     区分开；`user_id` 不是凭证，没有这个需求，可读性优先。二者的差别见 R8。
GUEST_USER_PREFIX = "guest-"
GUEST_USER_RANDOM_CHARS = 8

# ---- 消息角色 ----
#
# 闭集（FR-007）。非法取值 MUST 被拒绝 —— 一个拼错的 role 会让模型收到
# 一条它无法归类的消息，而 Redis 里"看起来"完全正常。
#
# 取值与 OpenAI messages 协议逐字对齐（R10 的改造依赖这一点）。
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLES: tuple[str, ...] = (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM)

# ---- 审计动作 ----
#
# 闭集。审计日志的 action 字段只取这些值（FR-026）。

ACTION_CREATE = "create"
ACTION_APPEND = "append"
ACTION_RENAME = "rename"
ACTION_CLEAR = "clear"
ACTION_DELETE = "delete"
ACTION_REPLY = "reply"

# ---- 退出码 ----
#
# 沿用 `backend/retrieve/__init__.py` 与 `backend/index/__init__.py` 的 0–3 约定，
# 与 `docs/05` §5 一致。

EXIT_OK = 0
EXIT_ARGS = 1
EXIT_DATA = 2
EXIT_DEP = 3


class ChatError(Exception):
    """会话存储或编排失败。

    覆盖三类：会话不存在、Redis 不可达、历史记录无法反序列化。

    ⚠️ 刻意**不继承** `ValueError` —— 与 `RetrievalError` / `LLMError` /
    `PromptError` 同一取向：继承内建异常会让 `except ValueError` 意外捕获到它，
    而调用链上游（Pydantic 校验）正在大量抛 `ValueError`。

    ⚠️ `message` MUST NOT 含消息内容或患者隐私字段（FR-025）。
    构造时只允许带 session_id / message_id / 长度这类元信息。
    """

    def __init__(self, message: str, *, session_missing: bool = False) -> None:
        super().__init__(message)
        self.message = message

        # 是否属于「会话不存在」这一档。
        #
        # 用标志位而不是让调用方匹配错误文案：路由层要据此返回 404 而不是 503
        # —— 而这两者对客户端的含义完全不同（"重新开一个会话" vs "稍后重试"）。
        # 靠匹配文案来区分，会在文案被改写时静默失效。
        self.session_missing = session_missing


__all__ = [
    "KEY_PREFIX",
    "SESSION_KEY_TMPL",
    "META_KEY_TMPL",
    "DEFAULT_SESSION_TTL",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_MAX_HISTORY_MESSAGES",
    "GUEST_USER_PREFIX",
    "GUEST_USER_RANDOM_CHARS",
    "ROLE_USER",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLES",
    "ACTION_CREATE",
    "ACTION_APPEND",
    "ACTION_RENAME",
    "ACTION_CLEAR",
    "ACTION_DELETE",
    "ACTION_REPLY",
    "EXIT_OK",
    "EXIT_ARGS",
    "EXIT_DATA",
    "EXIT_DEP",
    "ChatError",
]
