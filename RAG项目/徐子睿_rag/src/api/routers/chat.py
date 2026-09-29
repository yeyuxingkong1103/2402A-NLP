"""src/api/routers/chat.py —— 对话接口（非流式 + SSE 流式）。

在链路中的位置：
    HTTP → 【本文件】 → src/online/chain.py（run_chain / stream_chain）
                      → src/api/routers/role.py（取角色配置）
                      → src/models/database.py（校验会话归属）

一个接口两种形态（由请求体的 stream 字段决定），路由前缀 /api/v1/chat：
    stream=false  POST /completions  一次返回完整结果（JSON）
    stream=true   POST /completions  以 SSE 推流，事件序列 delta… → references → done

本文件只做"协议转换 + 权限校验"，不含任何 RAG 逻辑 ——
检索、精排、记忆、提示词全在 src/online/chain.py 里，
这样换协议（HTTP/WebSocket/gRPC）不必碰业务代码。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from configs.settings import get_settings
from src.api.deps import current_user
from src.api.routers.role import get_role_config
from src.models.database import ChatSession, User, db_session
from src.online.chain import run_chain, stream_chain
from src.schemas.roleplay import ChatRequest

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _session(sid: int, user: User) -> ChatSession:
    """取会话并校验它属于当前用户。

    参数：
        sid: 会话 id
        user: 当前登录用户
    返回：
        ChatSession 对象。
    异常：
        会话不存在或不属于当前用户 -> 404。

    越权校验不可省：
        会话 id 是自增整数，可以从 1 开始逐个试。
        不校验归属的话，任何人只要猜一个 id 就能往别人的会话里发消息、
        并读到该会话的历史（检索用的短期记忆正是从会话历史里取的）。

    为什么"不属于我"也返回 404 而不是 403：
        doc/会话 id 是可枚举的小整数，403 相当于确认"这个 id 存在但不是你的"，
        等于给出了一个可枚举的存在性探针。
        统一 404 让攻击者无法区分"不存在"和"无权访问"。

    为什么要新建一个 ChatSession 对象返回（而不是直接返回 session）：
        session 绑定在即将关闭的 db 会话上（这里是 `with db_session()`）。
        直接把它传出去，调用方访问属性时可能触发已关闭会话上的懒加载而报错。
        复制成一个脱离会话的普通对象，用起来才安全。
    """
    with db_session() as db:
        session = db.get(ChatSession, sid)
        if not session or session.user_id != user.id:
            raise HTTPException(status_code=404, detail="会话不存在")
        return ChatSession(id=session.id, user_id=session.user_id, role_id=session.role_id, tenant_id=session.tenant_id, title=session.title)


@router.post("/completions")
def completions(request: ChatRequest, user: User = Depends(current_user)):
    """对话主接口：按 stream 字段决定返回完整 JSON 还是 SSE 流。

    参数：
        request: ChatRequest（session_id / message / stream）
        user: 当前用户，由 Depends(current_user) 注入
    返回：
        stream=false 时返回 run_chain 的结果字典；
        stream=true 时返回 StreamingResponse（text/event-stream）。
    异常：
        消息为空或超长 -> 400；会话不属于自己 -> 404（由 _session 抛出）。

    校验顺序是刻意的（从便宜到昂贵）：
        1. 校验消息本身（纯内存判断，最便宜）
        2. 再查会话归属（要访问数据库）
        3. 最后取角色配置并执行链路（最贵，要调向量库和大模型）
        把便宜检查放前面，非法请求就不用付出后续的成本。

    len(request.message.strip()) == 0 这个写法：
        光判 `not request.message` 不够 —— 传一串空格或换行也能绕过。
        去空白后判断长度，才算真正"消息为空"。

    SSE 响应的两行格式 `event: xxx\\n` + `data: {...}\\n\\n`：
        这是 SSE 协议要求 —— 事件类型和载荷分两行，且以空行结束一个事件帧。
        每个 yield 一段而不是拼接成一整条，是为了让每个事件立刻 flush 出去；
        合并成一条的话中间会产生缓冲，前端看到的就是"卡一下然后一次出来一片"。

    ensure_ascii=False：
        中文以原文推送，体积更小，前端也不必额外解码。

    响应头的 X-Accel-Buffering: no 与 backend/server.py 同理：
        关掉 Nginx 的响应缓冲，否则流式效果会被缓冲吃掉。
    """
    if len(request.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="消息不能为空")
    if len(request.message) > get_settings().max_message_chars:
        raise HTTPException(status_code=400, detail=f"消息不能超过 {get_settings().max_message_chars} 个字符")
    session = _session(request.session_id, user)
    role = get_role_config(session.role_id, user.tenant_id)
    if not request.stream:
        return run_chain(role, user.id, session.id, request.message, user.tenant_id)

    def events():
        """把 chain 产出的事件字典翻译成 SSE 帧。

        把生成器再包一层，是因为 chain 层产出的是业务事件字典
        （{"event", "data"}），而 HTTP 层需要的是 SSE 文本协议。
        两者的职责分开：chain 不关心传输格式，本函数不关心业务含义。
        """
        for item in stream_chain(role, user.id, session.id, request.message, user.tenant_id):
            yield f"event: {item['event']}\n"
            yield "data: " + json.dumps(item["data"], ensure_ascii=False) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
