"""法律 RAG 的 HTTP 接口层。

浏览器不会直接调用检索器或大模型，而是先请求这里定义的接口。本文件负责：

- 接收和校验前端传来的问题；
- 识别当前登录用户并进行请求限流；
- 把问题交给 RagWorkflow；
- 用普通 JSON 或 SSE 流式事件把回答返回前端；
- 生成法律解决方案、PDF，以及查询或删除历史会话。

真正的“理解问题、检索、重排、生成答案”在 workflow.py 中完成。
"""

# hashlib 用 SHA-256 为解决方案缓存生成稳定且不暴露原文的键。
import hashlib
# json 用于序列化缓存指纹和流式响应数据。
import json
# logging 记录接口异常，便于排查问题，但不把内部错误直接暴露给用户。
import logging
# time.monotonic 提供不受系统时钟调整影响的限流计时。
import time
# defaultdict 自动创建桶；deque 可以高效移除过期请求时间。
from collections import defaultdict, deque
# quote 把中文文件名编码为 HTTP 响应头可以安全传输的形式。
from urllib.parse import quote

# APIRouter 组织接口；HTTPException 返回标准错误；Request 提供用户请求和应用服务。
from fastapi import APIRouter, HTTPException, Request
# Response 返回 PDF 二进制；StreamingResponse 返回逐步生成的 SSE 数据。
from fastapi.responses import Response, StreamingResponse

# current_user 根据请求中的登录凭证读取当前用户，凭证无效时抛出 401。
from ..auth import current_user
# render_solution_pdf 把 Markdown 法律方案排版为 PDF 字节。
from .answer import render_solution_pdf
# 下面三个请求模型负责校验问答、方案生成和 PDF 下载参数。
from .understand import AskRequest, SolutionGenerateRequest, SolutionPdfRequest


# 所有本文件路由都会带 /api/v1/legal 前缀，并在接口文档中归为 rag 标签。
router = APIRouter(prefix="/api/v1/legal", tags=["rag"])
# 创建本模块日志器，日志配置可单独识别 rag_api 来源。
logger = logging.getLogger("law_rag.rag_api")


class SimpleRateLimiter:
    """基于滑动窗口的内存速率限制器。

    按 key（user_id 或 IP）统计请求数，超过上限返回 429。
    匿名用户限制更严格，登录用户宽松一些。
    """

    def __init__(self, anonymous_limit: int = 5, user_limit: int = 20, window_seconds: int = 60):
        """保存匿名/登录用户上限和统计窗口，不会访问数据库。"""

        # 匿名用户每个时间窗口最多允许的请求数。
        self.anonymous_limit = anonymous_limit
        # 登录用户每个时间窗口最多允许的请求数。
        self.user_limit = user_limit
        # 滑动窗口长度，单位为秒。
        self.window = window_seconds
        # 每个 key 对应一个双端队列，队列中保存该用户近期请求时间。
        self._buckets: dict[str, deque] = defaultdict(deque)

    def check(self, key: str, is_anonymous: bool) -> None:
        """记录本次请求；超过当前身份的频率上限时抛出 HTTP 429。"""

        # 匿名用户和登录用户使用不同上限。
        limit = self.anonymous_limit if is_anonymous else self.user_limit
        # monotonic 只会向前走，适合计算时间间隔。
        now = time.monotonic()
        # 取得这个用户/IP 的请求时间队列；不存在时 defaultdict 自动新建。
        bucket = self._buckets[key]
        # 从最早记录开始删除已经落在时间窗口之外的请求。
        while bucket and bucket[0] <= now - self.window:
            bucket.popleft()
        # 清理后仍达到上限，拒绝本次请求且不加入队列。
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，请 {self.window} 秒后重试",
            )
        # 请求被允许，将当前时间放到队尾供后续统计。
        bucket.append(now)


# 应用进程共用一个限流器；重启服务会清空这份内存统计。
_rate_limiter = SimpleRateLimiter()


def _client_key(request: Request, user: dict | None) -> str:
    """为限流生成用户键：登录用户用 user_id，匿名用户用来源 IP。"""

    # 登录用户优先用账户 ID，同一用户换 IP 也共享限额。
    if user:
        return f"user:{user['user_id']}"
    # 经过反向代理时，X-Forwarded-For 可能包含多个 IP，第一个通常是原客户端。
    forwarded = request.headers.get("X-Forwarded-For", "")
    # 没有代理头时读取直连地址；测试环境可能没有 request.client，因此兜底 unknown。
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    # anon 前缀避免匿名 IP 与某个 user_id 文本意外冲突。
    return f"anon:{ip}"


def optional_user(request: Request) -> dict | None:
    """尝试识别登录用户；未登录或凭证失效时按匿名访问处理。"""

    # current_user 成功时返回用户字典。
    try:
        return current_user(request)
    except HTTPException:
        # 问答允许游客使用，所以认证异常在这里转成 None，而不是直接返回 401。
        return None


def _solution_cache_key(user: dict | None, session_id: str | None, payload: SolutionGenerateRequest) -> str:
    """根据用户、会话、问题、回答和来源生成唯一的方案缓存键。"""

    # 登录用户之间必须隔离；游客统一使用 anonymous 标识。
    user_part = user["user_id"] if user else "anonymous"
    # 没有会话时也放入明确占位文字，避免拼接结果含义不清。
    session_part = session_id or "no-session"
    # answer 可能是字符串或结构化对象；对象使用固定排序转换成稳定 JSON。
    answer_text = payload.answer if isinstance(payload.answer, str) else json.dumps(payload.answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # 来源列表同样固定排序和格式，相同输入才能得到相同缓存键。
    source_fingerprint = json.dumps(payload.sources, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # 用竖线拼接所有会影响方案内容的部分。
    raw = "|".join([user_part, session_part, payload.question.strip(), answer_text, source_fingerprint])
    # SHA-256 将可能很长的原文变成固定长度指纹，不把隐私文本直接放入 Redis key。
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # v2 是缓存版本；将来格式再升级时改版本即可与旧缓存隔离。
    return f"rag:solution:v2:{digest}"


@router.post("/ask")
def ask(payload: AskRequest, request: Request):
    """普通问答接口：等完整 RAG 流程结束后一次返回 JSON。"""

    # 登录成功得到用户字典；未登录则为 None，并继续以游客身份回答。
    user = optional_user(request)
    # 按 user_id 或 IP 检查最近一分钟请求次数，超限会直接抛出 429。
    _rate_limiter.check(_client_key(request, user), user is None)
    # 游客不保存会话；只有登录用户的 session_id 才向后传递。
    session_id = payload.session_id if user else None
    # 从应用启动时注册的服务中取 RagWorkflow，并执行完整问答流程。
    # payload.text() 得到规范化问题；其余参数控制联网、思考、检索和回答详细度。
    return {"success": True, "data": request.app.state.services["rag"].run(
        payload.text(), user, session_id, payload.include_web, payload.thinking_enabled,
        retrieval_mode=payload.retrieval_mode, answer_detail=payload.answer_detail,
    )}


@router.post("/ask/stream")
def ask_stream(payload: AskRequest, request: Request):
    """流式问答接口：通过 SSE 持续发送状态、思考步骤、文字片段和最终结果。"""

    # 识别登录身份；该接口同样允许游客使用。
    user = optional_user(request)
    # 流式请求也必须限流，避免长连接大量占用服务资源。
    _rate_limiter.check(_client_key(request, user), user is None)
    # 只有登录用户可以使用和保存会话上下文。
    session_id = payload.session_id if user else None

    def events():
        """把工作流产生的 Python 事件转换成浏览器可识别的 SSE 文本。"""

        # 流式生成过程中发生异常时，仍然发送 error 事件而不是突然断开且无说明。
        try:
            # workflow.stream 每次返回 (事件名称, 事件数据)。
            for event, data in request.app.state.services["rag"].stream(
                payload.text(), user, session_id, payload.include_web, payload.thinking_enabled,
                retrieval_mode=payload.retrieval_mode, answer_detail=payload.answer_detail,
            ):
                # SSE 格式要求 event 和 data 各占一行，并用空行表示一个事件结束。
                # ensure_ascii=False 让浏览器直接收到可读中文。
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as exc:
            # 将异常文字放进统一 JSON；前端据此显示本轮回答失败。
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"

    # StreamingResponse 不会等待 events 全部完成，而是边生成边发送。
    return StreamingResponse(
        events(),
        # text/event-stream 是浏览器 EventSource/SSE 约定的类型。
        media_type="text/event-stream",
        # 禁止代理缓存，并关闭 Nginx 缓冲，确保文字及时到达前端。
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/solution")
def solution(payload: SolutionGenerateRequest, request: Request):
    """根据本轮问题、答案和引用来源生成一份结构化法律解决方案。"""

    # 方案接口允许游客，但登录用户可以利用自己的会话历史。
    user = optional_user(request)
    # 生成方案通常会再次调用大模型，所以也需要限流。
    _rate_limiter.check(_client_key(request, user), user is None)
    # services 是应用统一注册的服务容器。
    services = request.app.state.services
    # rag 提供方案生成器。
    rag = services["rag"]
    # Redis 用于缓存相同输入的方案，减少重复模型调用和等待时间。
    redis_store = services["redis"]
    # 游客不关联会话，防止伪造 session_id 读取他人历史。
    session_id = payload.session_id if user else None

    # force_refresh=True 表示用户明确要求重新生成，不采用已有缓存。
    # getattr 提供兼容性：旧请求模型没有该字段时按 False 处理。
    force_refresh = getattr(payload, "force_refresh", False)

    # 根据完整输入生成用户隔离的缓存键。
    cache_key = _solution_cache_key(user, session_id, payload)

    # 普通请求优先读取 Redis；强制刷新时直接跳过这里。
    if not force_refresh:
        # 缓存值已经是可以直接返回前端的字典。
        cached_solution = redis_store.get_json_value(cache_key)
        # 找到缓存就立即返回，不再读取历史和调用大模型。
        if cached_solution:
            return {"success": True, "data": cached_solution}

    # history 保存本会话近期问答，帮助方案保持上下文一致。
    history = []
    # 只有登录且带会话 ID 时，才允许读取对应记忆。
    if user and session_id:
        try:
            # 新版项目优先使用统一 MemoryOrchestrator。
            memory = services.get("memory")
            if memory:
                # load_context 会同时处理短期历史和必要的上下文信息。
                context = memory.load_context(user["user_id"], session_id, query=payload.question)
                # getattr 防止旧上下文对象缺少 short_term；没有历史时使用空列表。
                history = getattr(context, "short_term", []) or []
            else:
                # 没有记忆服务时兼容旧逻辑，直接从 Redis 读取短期历史。
                history = redis_store.get_history(user["user_id"], session_id)
        except Exception:
            # 历史读取失败不会阻止方案生成，只记录日志并基于本轮内容继续。
            logger.warning(
                "法律解决方案读取会话历史失败，继续基于本轮回答生成",
                extra={"event": "solution_history_load_failed", "fields": {"session_id": session_id or ""}},
                exc_info=True,
            )

    # 把问题、已有回答、引用来源和会话历史交给大模型，生成可下载方案。
    generated = rag.generator.generate_solution(
        payload.question,
        payload.answer,
        payload.sources,
        history=history,
        # 方案生成不展示思考过程，避免把内部草稿输出给用户。
        thinking_enabled=False
    )
    # 把缓存键带入结果，方便后续追踪本次生成对应的输入版本。
    generated["cache_key"] = cache_key

    # 新生成结果写入 Redis；强制刷新时会覆盖相同键的旧内容。
    redis_store.set_json_value(
        cache_key,
        generated,
        # 使用历史记录 TTL 作为缓存有效期；配置不存在时默认保留七天。
        getattr(services["settings"], "history_ttl", 604800)
    )

    # 用统一接口结构返回方案。
    return {"success": True, "data": generated}


@router.post("/solution/pdf")
def solution_pdf(payload: SolutionPdfRequest, request: Request):
    """把已经生成的 Markdown 法律方案转换成可下载 PDF。"""

    # 下载方案属于账户功能，必须登录；失败时 current_user 直接返回 401。
    user = current_user(request)
    # 登录用户使用较宽松的用户限额，但仍需防止反复生成 PDF。
    _rate_limiter.check(_client_key(request, user), False)
    # 去掉首尾空白，避免只提交空格也通过检查。
    markdown = (payload.markdown or "").strip()
    # 内容为空或过短时无法形成有意义的法律方案，返回 400 提醒前端。
    if not markdown or len(markdown) < 50:
        raise HTTPException(status_code=400, detail="解决方案内容为空或过短，无法生成 PDF")
    # 将 Markdown 排版并渲染为 PDF 二进制数据。
    pdf = render_solution_pdf(markdown)
    # 中文下载名需要进行 URL 编码后才能安全放进 Content-Disposition。
    filename = quote("法律解决方案.pdf")
    # 直接返回 PDF 字节，不再套 success/data JSON。
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            # 同时提供 ASCII 备用文件名和 UTF-8 中文文件名。
            "Content-Disposition": f"attachment; filename=legal-solution.pdf; filename*=UTF-8''{filename}",
            # 法律方案可能包含个人事实，禁止浏览器或中间代理缓存。
            "Cache-Control": "no-store",
        },
    )


@router.get("/sessions")
def list_sessions(request: Request, limit: int = 20):
    """读取当前登录用户的最近咨询列表，供前端恢复历史会话。"""

    # 历史咨询属于私有数据，因此必须登录。
    user = current_user(request)
    # 从服务容器获取记忆服务。
    memory = request.app.state.services.get("memory")
    # 项目未启用记忆服务时返回空列表，而不是让页面报错。
    if not memory:
        return {"success": True, "data": []}
    # limit 最小为 1、最大为 20；异常的 0 或超大值都会被限制到安全范围。
    safe_limit = min(max(int(limit or 20), 1), 20)
    # 只按当前 user_id 查询，保证不同账户的历史相互隔离。
    summaries = memory.history.latest_messages(user["user_id"], limit=safe_limit)
    # 返回前端侧边栏需要的会话摘要。
    return {"success": True, "data": summaries}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    """删除当前用户指定会话的记忆，以及该会话上传的资料。"""

    # 删除操作必须登录；否则无法确定资料所有者。
    user = current_user(request)
    # 后续所有删除都同时使用 user_id 和 session_id，不能只按会话号删除。
    user_id = user["user_id"]
    # memory 负责短期/长期会话数据。
    memory = request.app.state.services.get("memory")
    # workspace 负责用户上传文件、解析结果和私有向量。
    workspace = request.app.state.services.get("workspace")

    # 删除涉及多个存储组件，统一捕获异常并给用户稳定的错误提示。
    try:
        # 配置了记忆服务时，删除该用户的当前会话记忆。
        if memory:
            memory.delete_session(user_id, session_id)
        # 删除会话文件；没有 workspace 时返回一个结构一致的空结果。
        workspace_result = workspace.delete_session_files(user, session_id) if workspace else {"file_count": 0, "warnings": []}
    except Exception as exc:
        # 详细异常写入服务器日志，便于管理员排查。
        logger.warning(
            "会话服务器数据清理失败",
            extra={"event": "rag_session_delete_failed", "fields": {"session_id": session_id, "error": str(exc)}},
            exc_info=True,
        )
        # 对外只返回简洁 500 信息，不泄露数据库或文件系统细节。
        raise HTTPException(status_code=500, detail="会话清理失败，请稍后重试") from exc

    # 告诉前端删除是否完成、删除了多少文件，以及是否存在非致命警告。
    return {
        "success": True,
        "data": {
            # 执行到这里代表主删除流程没有抛出异常。
            "deleted": True,
            # 回传会话 ID，方便前端从列表中精确移除。
            "session_id": session_id,
            # 统一转换为整数，兼容存储层返回 None 或数字字符串。
            "file_count": int(workspace_result.get("file_count") or 0),
            # 统一转换为列表，保证前端遍历时类型稳定。
            "warnings": list(workspace_result.get("warnings") or []),
        },
    }
