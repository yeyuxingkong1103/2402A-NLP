"""HTTP 路由。

本模块只做 **HTTP ↔ 模型** 的协议转换，MUST NOT 含业务判定 ——
风险分级、紧急判定、拒答决策、话术拼接全部在服务层（`stream.py` 及后续模块）。
这条与 `docs/02` §2.1 对前端的约束是同一条原则的服务端侧：业务判定必须收敛在
一处，否则 N7 的"可断言"无从测试。
"""

import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from . import ASK_PATH
from .capture import capture_question
from .errors import request_id_of
from .schemas import AskRequest
from .sse_headers import SSE_HEADERS
from .stream import stream_answer
from .validate import validate_question

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(ASK_PATH)
async def ask(request: Request, payload: AskRequest) -> StreamingResponse:
    """提问入口。

    ⚠️ 校验必须发生在**任何流式输出之前**。若先发出 `status` 帧再报错，客户端
    会拿到一半的流却等不到 `done`，无法区分"服务端拒绝"与"网络断了"。
    因此校验（`validate.py`，Phase 4 接入）在返回 `StreamingResponse` 之前完成。

    注意 `media_type` 只写 `text/event-stream`，`charset=utf-8` 由 Starlette
    自动补上。缺了它浏览器会按 latin-1 解析，中文变乱码 —— 而这类问题在
    开发机上用 curl 往往看不出来。
    """

    config = request.app.state.config
    request_id = request_id_of(request)

    # ⚠️ 校验 MUST 在构造 StreamingResponse **之前**完成。
    #
    # 若先返回流、再在生成器里报错，客户端会拿到 status 帧却等不到 done，
    # 无法区分"服务端拒绝了这个提问"与"网络断了" —— 而这两件事该做的处置
    # 完全不同（改问题 vs 重试）。抛在这里，异常处理器能正常返回 422 JSON。
    question = validate_question(payload.question)

    answer_id = str(uuid.uuid4())

    # FR-026：记录请求标识与问题原文。
    # 记的是**规范化后**的问题，与后续检索用的是同一串文本（见 validate.py）。
    logger.info(
        "收到提问 answer_id=%s request_id=%s question=%r",
        answer_id,
        request_id,
        question,
    )

    # 留存这次提问（specs/007），并取回查询向量（specs/008）。
    #
    # 位置：在 answer_id 生成之后（记录里要有它），且在构造响应之前
    # —— 响应一旦开始流出，请求就进入"已经在回答"的状态，
    # 那时再落盘会让"记录属于哪次提问"依赖于流的时序。
    #
    # ⚠️ 它 MUST NOT 影响下面返回的响应对象：capture_question 自己吞掉异常
    #    并留下日志（见 capture.py 的职责说明），这里不做任何错误处理 ——
    #    在调用点写 try/except 会让"失败不影响应答"变成两处约定，
    #    而其中一处迟早会被漏掉。
    #
    # 一行做两件事是刻意的：**留存与向量本来就是同一次编码的产物**。
    # 分成两次调用（一次留存、一次编码）会多花 100–300 ms 的推理，并引入
    # 第二个编码入口 —— 而后者是 S8 整篇规格在防的东西（FR-003）。
    query_vector = capture_question(question, answer_id)

    return StreamingResponse(
        stream_answer(
            question=question,
            answer_id=answer_id,
            config=config,
            query_vector=query_vector,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
