# -*- coding: utf-8 -*-
"""
问答接口

对外提供两个接口：
    POST /api/chat/session  —— 新建会话
    POST /api/chat/query    —— 提问（返回答案 + 溯源来源）

接口边界：
    响应体只包含**用户需要看到的内容**：答案正文与来源（文件名 + 页码）。
    检索分数虽然随来源一并返回（用于前端可选展示），但检索链路细节
    （命中过程、召回列表、中间分数分布）一律不对外暴露，
    它们只写在后端日志里（对齐 N5 约束）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.db import mysql, redis_client
from backend.logging_config import get_logger, new_request_id, set_session_id
from backend.rag_pipeline import get_pipeline_by_name

logger = get_logger(__name__)

# 本模块的路由前缀：下面所有 @router.post(...) 都挂到 /api/chat 之下，
# 最终对外地址就是 /api/chat/session、/api/chat/query 等。
# tags 只影响接口文档（/docs）里的分组显示。
router = APIRouter(prefix="/api/chat", tags=["问答"])


# ===========================================================================
# 请求 / 响应模型
# ===========================================================================

class SessionResponse(BaseModel):
    """新建会话响应"""

    session_id: str = Field(..., description="新建的会话 ID，后续提问携带该值以保持上下文")


class QueryRequest(BaseModel):
    """提问请求"""

    # 问题正文，第一位的 ... 表示必填。
    # 上限卡在 500 字有两个目的：挡住异常超长输入，
    # 同时避免用户把整段文档粘进来当问题 —— 那样检索会完全失焦。
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户问题，1-500 字",
        examples=["数据中心业务连续性分为几个等级？"],
    )
    session_id: Optional[str] = Field(
        None,
        description="会话 ID。不传则自动新建会话；传入已有 ID 可保持多轮上下文",
    )
    # 每次问答取几段原文拼进提示词。不传就用服务端配置的默认值。
    # 上限卡在 20：上下文越长，大模型越容易被无关片段带偏，生成也越慢。
    top_k: Optional[int] = Field(
        None, ge=1, le=20,
        description="检索片段数量，默认取服务端配置（RETRIEVE_TOP_K）",
    )
    role: Optional[str] = Field(
        "student",
        description=(
            "回答身份：student 学生（轻松易读）/ workplace 职场（严谨权威）。"
            "该参数只影响生成答案时使用的 system 提示词，"
            "检索、分块、重排、溯源链路完全不受影响；非法值自动回退 student"
        ),
        examples=["student"],
    )
    user_id: Optional[str] = Field(
        None,
        description=(
            "登录用户 ID。传入则本次问答会写入该用户的历史记录（MySQL 持久化）；"
            "不传则不写历史，行为与改造前一致"
        ),
    )

    @field_validator("question", mode="before")
    @classmethod
    def _validate_question(cls, value: Any) -> Any:
        """
        输入校验（V3 新增，对应任务书「输入校验防护」）。

        除了长度（长度由 Field 约束保证），还要拦住两类「长度为合法但语义为空」
        的输入：
            1. 纯空白 —— 用户误触回车
            2. 纯符号/标点 —— 无法构成检索线索，只会让模型产出无意义答案

        同时统一去除首尾空白。这样做的价值在于**会话历史与提示词的一致性**：
        同一问题的不同空白写法，在会话历史里不会再变成两条不同记录，进入模型的
        也不会是两段不同文本。（查询缓存 Key 另有 `redis_client._digest` 做
        strip / 小写 / 空白折叠，本就不受首尾空白影响。）

        采用 `mode="before"`，使长度约束作用于 strip **之后**的文本，
        从而符合「可见字数不超过 500」的直觉：首尾空白不计入上限。
        """
        # 非字符串输入（如数字、null）交给 pydantic 的类型校验去报错，这里不拦截
        if not isinstance(value, str):
            return value

        # 统一去掉首尾空白：用户复制粘贴时经常带上多余的空格与换行。
        text = value.strip()
        # 情形一：剥完空白后什么都不剩，例如误触回车直接提交。
        if not text:
            raise ValueError("问题不能为空或全为空白字符")
        # 去掉空白与常见标点后若什么都不剩，则视为无实质内容
        # 情形二：整句都是标点符号（例如 "。。。"）。这种输入检索不出任何线索，
        # 只会白消耗一次大模型调用，不如在入口就挡掉。
        meaningful = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
        if not meaningful:
            raise ValueError("问题需包含实质内容，不能只有标点或符号")
        # 返回清洗后的文本：后面的会话历史、检索、提示词用的都是这一份，
        # 保证同一个问题换个空白写法不会被当成两个不同的问题。
        return text


# 单条溯源来源 —— 这是「页码溯源」这条硬性要求在对外接口上的落地形态：
# 用户拿到的答案旁边必须能看到它出自哪份文件、哪一页，便于回去核对原文。
# 这里只放文件名、页码这类可以展示的信息，检索过程的中间细节不在这里。
class SourceItem(BaseModel):
    """单条溯源来源"""

    file_name: str = Field(..., description="来源文档名称")
    page_no: int = Field(..., description="来源页码（1 基），前端展示用")
    page_nums: List[int] = Field(..., description="完整页码列表，跨页片段为多值")
    chunk_id: str = Field(..., description="命中的知识块 ID")
    score: float = Field(..., description="检索相似度分数")
    summary: str = Field(..., description="命中片段摘要，便于用户快速核对")
    content_type: str = Field("text", description="片段类型：text / table / image")
    section_title: str = Field("", description="所属章节标题")


class QueryResponse(BaseModel):
    """提问响应"""

    session_id: str = Field(..., description="会话 ID，后续追问请原样回传")
    answer: str = Field(..., description="生成的答案正文")
    sources: List[SourceItem] = Field(
        default_factory=list, description="来源列表（文档名称 + 页码）"
    )
    # 这次答案是 V1 / V2 / V3 哪条链路产出的，前端可据此显示，方便版本对比
    pipeline: str = Field(..., description="处理该问题的链路版本")
    # 端到端耗时，用来验证"首次提问 ≤8 秒"这个性能指标是否达标
    latency_ms: int = Field(..., description="端到端耗时（毫秒）")
    # 是否命中查询缓存：同一个问题重复提问时直接返回上次的答案，不再重跑检索与生成
    cached: bool = Field(False, description="是否命中查询缓存")
    role: str = Field(
        "student", description="本次回答使用的身份（实际生效值，非法入参已回退）"
    )
    message_id: Optional[int] = Field(
        None,
        description="本次问答在历史记录中的消息 ID；未登录时不返回。收藏时回传该值",
    )


# ===========================================================================
# 历史记录落库（增量功能，旁路执行）
# ===========================================================================
# 为什么放在接口层而不是 pipeline 内部：
#     需求明确「不得改动已稳定的 RAG 核心」。把 chat_sessions / chat_messages
#     的写入放在 API 层，pipeline 完全不需要知道「历史记录」这件事，
#     PDF 解析、分块、向量化、检索、重排、生成、溯源逻辑一行未动。

# 溯源面板只允许展示「文档 / 来源段落 / 页码」三类信息，
# 因此落库时就把检索分数、chunk_id 剔除，数据库里不残留内部细节。
SOURCE_DISPLAY_KEYS = (
    "file_name", "page_no", "page_nums", "summary", "section_title", "content_type",
)


def _slim_sources(sources: Any) -> List[Dict[str, Any]]:
    # 裁剪溯源数据：只保留面板允许展示的字段（去检索分数、去 chunk_id）
    slim: List[Dict[str, Any]] = []
    # sources or [] 同时兼顾了 None 和空列表两种"没有来源"的情况。
    for item in sources or []:
        # 跳过结构不正常的元素，避免一条脏数据把整次落库带崩。
        if not isinstance(item, dict):
            continue
        # 只挑白名单里的键，其余字段（检索分数、chunk_id 等内部细节）不带进库。
        slim.append({k: item.get(k) for k in SOURCE_DISPLAY_KEYS if k in item})
    return slim


def _safe_session_owner(session_id: str) -> Optional[str]:
    # 查询会话归属用户；查询失败按「无归属」处理，不阻断问答
    try:
        return mysql.get_session_owner(session_id)
    except Exception as exc:
        logger.warning("查询会话归属失败（按无归属处理）：%s", exc)
        return None


def _persist_history(
    *, result: Dict[str, Any], question: str, user_id: str
) -> Dict[str, Any]:
    # 把本次问答写入 MySQL 历史记录（登录用户才写）。
    #
    # 失败只告警不抛异常：历史记录写不进去，不应影响用户拿到答案。
    # 返回带 message_id 的结果，供前端收藏时回传。
    # 补齐默认身份，保证后面读 result["role"] 时不会 KeyError。
    result.setdefault("role", "student")
    if not user_id:
        # 未登录调用方（curl / JMeter 压测 / 评测脚本）：不写历史，行为同改造前
        return result

    session_id = result.get("session_id") or ""
    try:
        # 用户 ID 必须真实存在才写历史，否则历史表里会多出指向不存在用户的记录。
        if not mysql.get_user_by_id(user_id):
            logger.warning("提问携带的 user_id 不存在，跳过历史记录 | 用户=%s", user_id)
            return result

        sources = _slim_sources(result.get("sources"))
        # 先把会话本身登记一遍（已存在则更新标题与身份），后面才能往它下面挂消息。
        mysql.upsert_chat_session(
            session_id=session_id,
            user_id=user_id,
            role=result["role"],
            title=question,
        )
        # 再把这一问一答作为一条消息写进历史；返回的消息 ID 前端收藏时要用。
        message_id = mysql.insert_chat_message(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=result.get("answer") or "",
            sources=sources,
            role=result["role"],
            latency_ms=int(result.get("latency_ms") or 0),
        )
        # 在 Redis 里记下"这个会话属于哪个用户"，
        # 下次带同一个 session_id 提问时才能做归属校验（见 query 里的校验逻辑）。
        redis_client.set_session_user(session_id, user_id)
        # 把消息 ID 挂回结果上，随响应一起返回给前端。
        result["message_id"] = message_id
    except Exception as exc:
        # 历史记录写失败只是少存了一条，答案本身没问题 —— 所以只告警，不抛异常。
        logger.warning("写入历史记录失败（不影响本次问答）：%s", exc)
    return result


# ===========================================================================
# 接口实现
# ===========================================================================

@router.post("/session", response_model=SessionResponse, summary="新建会话")
def create_session() -> SessionResponse:
    """创建一个新的对话会话，返回 session_id"""
    # 多轮对话的上下文存在 Redis 里，这里只是申请一个会话 ID 并把上下文初始化。
    session_id = redis_client.create_session()
    logger.info("新建会话：%s", session_id)
    return SessionResponse(session_id=session_id)


@router.post("/query", response_model=QueryResponse, summary="提问")
def query(request: QueryRequest) -> QueryResponse:
    """
    基于知识库回答问题，并返回来源文档与页码。

    - 答案严格依据检索到的原文片段生成，知识库中无相关内容时明确说明未找到
    - 每条答案都附带来源文档名称与页码，便于核对原文
    """
    # 给这次请求分配一个 request_id 并绑定会话号。之后整条链路
    # （检索、重排、生成）打的日志都会带上这两个标识，
    # 排查某一次具体提问时可以直接按 request_id 过滤日志。
    new_request_id()
    set_session_id(request.session_id)

    # 入口处再 strip 一次并兜底判空（校验器已经处理过，这里是双保险）。
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # ---- 增量改造：身份与登录用户（不改变检索/分块/重排/生成任何环节）----
    user_id = (request.user_id or "").strip()
    session_id = request.session_id

    # 会话归属校验：防止拿着别人的 session_id 继续提问、把内容写进自己历史
    if session_id and user_id:
        owner = _safe_session_owner(session_id)
        if owner and owner != user_id:
            # 归属对不上就不复用这个会话，改为新建 —— 宁可从零开始一段对话，
            # 也不能把 A 用户的上下文接到 B 用户的提问上。
            logger.warning("会话归属不一致，改为新建会话 | 会话=%s", session_id)
            session_id = None

    logger.info(
        "收到提问 | 问题=%r | 会话=%s | 身份=%s | 用户=%s",
        question, session_id or "(新建)", request.role or "student", user_id or "(未登录)",
    )

    # 走到这里才真正调用 RAG 链路：
    # 检索 → 重排 → 拼提示词 → 大模型生成 → 构造溯源。
    try:
        # 链路版本由配置驱动。当前 .env 配置为 v3（查询改写 + 重排）；
        # 代码内默认值为 v2，脱离 .env 运行时以 v2 为准。
        # 出问题改 .env 的 PIPELINE_VERSION 回退即可（可选 v1/v2/v3），接口契约不变。
        #
        # role 只透传到生成环节用于替换 system 提示词，检索部分原封不动；
        # 查询缓存 Key 也带上了 role，切换身份问同一句话不会命中同一份缓存。
        result: Dict[str, Any] = get_pipeline_by_name(
            settings.pipeline_version
        ).answer(
            question,
            session_id,
            top_k=request.top_k,
            role=request.role,
        )
    except Exception as exc:
        # 异常细节只进后端日志，对外返回可理解的中文提示
        # （不把堆栈、Milvus/MySQL 报错原文暴露给用户，这是接口边界的要求）
        logger.exception("问答处理失败 | 问题=%r", question)
        raise HTTPException(
            status_code=500,
            detail="问答服务处理失败，请稍后重试（详细信息见服务端日志）",
        ) from exc

    # 旁路写历史记录（登录用户才有；失败不影响本次问答）
    result = _persist_history(result=result, question=question, user_id=user_id)

    # 用户没传 session_id 时链路会新建一个会话，这里把最终生效的会话号
    # 写进日志上下文，并随响应返回给前端，供后续追问复用。
    set_session_id(result.get("session_id"))
    # 把链路返回的字典装配成响应模型。模型里没定义的字段会被自动忽略，
    # 所以检索链路细节想从这里漏出去也漏不出去。
    return QueryResponse(**result)


# ===========================================================================
# 流式提问（增量功能：答案边生成边推送，用户不用再干等）
# ===========================================================================
# 与 /api/chat/query 的关系：
#     原接口保持原样，一行未改 —— 评测脚本、JMeter 压测、外部调用方完全不受影响。
#     本接口是并行的流式版本，检索、溯源、缓存、落库规则与它完全一致。


def _sse(event: Dict[str, Any]) -> str:
    """把一条事件编码成 SSE 报文（data: <json> + 空行分隔）"""
    # SSE 协议规定：每条消息以 "data: " 开头、以两个换行结束，前端才认。
    # ensure_ascii=False 是为了让中文答案按原文推送（否则会被转义成 \uXXXX）。
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


@router.post("/stream", summary="提问（流式输出）")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    流式提问：先把溯源推给前端，再逐段推送答案正文，最后收尾。

    SSE 事件（每条 data 为一个 JSON 对象）：
        {"type": "meta",    "session_id", "sources", "role", "pipeline", "cached"}
            检索完成，前端可立即展示「参考来源」与身份标签
        {"type": "delta",   "text": "..."}     答案增量
        {"type": "done",    "session_id", "answer", "latency_ms", "cached"}
            生成结束（此时会话历史与查询缓存已写好）
        {"type": "saved",   "message_id": 123} 历史记录已落库，前端收藏要用它
        {"type": "error",   "detail": "..."}   出错

    说明：模型内部的推理过程（reasoning_content）不会出现在任何事件里。
    """
    # 入参处理与 /query 完全一致：清洗问题、取用户 ID、取会话 ID。
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    user_id = (request.user_id or "").strip()
    session_id = request.session_id

    # 会话归属校验：与 /query 同一套规则
    if session_id and user_id:
        owner = _safe_session_owner(session_id)
        if owner and owner != user_id:
            logger.warning("会话归属不一致，改为新建会话（流式）| 会话=%s", session_id)
            session_id = None

    # 同样打上 request_id 与会话号，方便和普通接口用同一套方式排查日志。
    new_request_id()
    set_session_id(session_id)
    logger.info(
        "收到提问（流式）| 问题=%r | 会话=%s | 身份=%s | 用户=%s",
        question, session_id or "(新建)", request.role or "student", user_id or "(未登录)",
    )

    def event_generator() -> Iterator[str]:
        """把 pipeline 的事件流原样转发给前端，并在最后补一条落库结果"""
        # final / meta 用来顺手记住链路推出过的两个关键事件，
        # 因为最后落库要用到它们里面的答案与溯源信息。
        final: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        try:
            pipeline = get_pipeline_by_name(settings.pipeline_version)
            # 链路一边生成一边把事件交出来，这里来一条就往前端推一条 ——
            # 用户不用等答案全部写完才看到内容。
            for event in pipeline.answer_stream(
                question,
                session_id,
                top_k=request.top_k,
                role=request.role,
            ):
                etype = event.get("type")
                # meta 事件里带着溯源来源，done 事件里带着最终答案与耗时，
                # 各抄一份留着后面落库用。
                if etype == "meta":
                    meta = event
                elif etype == "done":
                    final = event
                # 原样转发给前端（检索细节本来就只在后端日志里，不在这条流上）。
                yield _sse(event)
        except Exception:
            # 异常细节只进服务端日志，对外给可理解的中文提示
            logger.exception("流式问答处理失败 | 问题=%r", question)
            # 注意：数据已经开始往外推了，HTTP 状态码改不动了，
            # 只能把错误当成一条事件推给前端，由前端显示提示语。
            yield _sse({
                "type": "error",
                "detail": "问答服务处理失败，请稍后重试（详细信息见服务端日志）",
            })
            return

        # 旁路写历史记录（与 /query 共用同一个函数；失败不影响已经推给前端的答案）
        # 注意顺序：答案是边生成边推出去的，这里是生成结束之后才落库，
        # 所以落库就算失败也影响不到用户已经看到的答案。
        try:
            saved = _persist_history(
                result={
                    "session_id": final.get("session_id") or session_id or "",
                    "answer": final.get("answer") or "",
                    "sources": meta.get("sources") or [],
                    "role": meta.get("role") or request.role or "student",
                    "latency_ms": final.get("latency_ms") or 0,
                },
                question=question,
                user_id=user_id,
            )
            # 单独补一条 saved 事件把消息 ID 告诉前端（收藏功能要用它）。
            # 只有真的写进库了才有这个 ID，未登录用户不会收到这条事件。
            if saved.get("message_id"):
                yield _sse({"type": "saved", "message_id": saved["message_id"]})
        except Exception as exc:
            logger.warning("流式问答落库失败（不影响答案展示）：%s", exc)


    # 这里返回的是还没执行的生成器：等 FastAPI 收到它之后才开始逐条往外推，
    # 期间连接一直不关闭 —— 这就是"流式"的实现方式。
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # 禁止浏览器和中间层缓存这份响应，否则流式内容会被攒齐了再给。
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 关掉反向代理缓冲，保证增量能实时到达
        },
    )


@router.post("/clear", summary="清空会话上下文")
def clear_session(session_id: str) -> Dict[str, Any]:
    """
    清空指定会话的上下文，用于开启全新话题（对应「新建对话」交互）。
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    # 清掉这个会话在 Redis 里的多轮上下文，相当于"开一个新话题"，
    # 但会话 ID 本身保留不变，前端可以继续用同一个 ID 提问。
    redis_client.clear_session(session_id)
    return {"session_id": session_id, "cleared": True}
