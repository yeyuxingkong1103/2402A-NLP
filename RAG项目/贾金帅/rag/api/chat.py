"""RAG chat and short-term memory routes."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src import config
from .auth import get_current_user
from .common import _determine_evidence_level, _encode_stream_event, _extract_references
from .dependencies import (
    PIPELINE_AVAILABLE,
    PIPELINE_IMPORT_ERROR,
    _conversation_memory,
    get_contextualizer,
    get_llm,
    get_pipeline,
)
from .history import _append_message, _resolve_conversation
from .schemas import ChatRequest

router = APIRouter(tags=["chat"])


def chat(req: ChatRequest, _user: dict | None = Depends(get_current_user)):  # 依赖注入校验登录
    question = req.question.strip()  # 去掉问题首尾空白
    if not question:  # 问题为空
        raise HTTPException(status_code=400, detail="问题不能为空")  # 返回 400
    user_id = _user["id"] if _user else None  # 取当前用户 ID（游客为 None）
    conversation_id = _resolve_conversation(user_id, req.conversation_id, question, "qa")  # 找到/新建历史对话
    _append_message(conversation_id, "user", question)  # 先把用户问题写入历史
    session_id = _conversation_memory.ensure_session(req.session_id)  # 获取/创建记忆会话

    def event_stream():  # 流式生成器：逐事件产出
        try:  # 整体异常兜底
            if not PIPELINE_AVAILABLE:  # 管线不可用
                yield _encode_stream_event({  # 输出错误事件
                    "type": "error",
                    "message": "知识图谱检索当前不可用：" + PIPELINE_IMPORT_ERROR,
                })
                yield _encode_stream_event({"type": "done"})  # 输出结束事件
                return  # 提前结束

            pipeline = get_pipeline()  # 获取检索管线
            history = _conversation_memory.get_history(session_id)  # 取该会话的历史
            yield _encode_stream_event({"type": "status", "message": "正在结合上下文并检索相关信息..."})  # 通知前端状态
            contextualized = get_contextualizer().contextualize(question, history)  # 追问改写为独立问题
            output = pipeline.search(
                contextualized.resolved,
                web_search=req.web_search,
            )  # 执行本地检索；按请求决定是否追加 Tavily 联网搜索
            engine_result_counts = {  # 统计每个引擎命中数
                name: len(results) for name, results in output.engine_results.items()
            }
            engine_status: dict = {}  # 各引擎可用状态
            for _name, _engine in (pipeline.router.engines or {}).items():  # 遍历引擎
                _available = False  # 默认不可用
                _error = ""  # 错误信息
                try:  # 探测引擎
                    _available = bool(_engine.is_available())  # 是否可用
                except Exception as _exc:  # 探测失败
                    _error = f"{type(_exc).__name__}: {_exc}"  # 记录
                engine_status[_name] = {"available": _available, "error": _error}  # 保存状态

            if output.web_search_requested:
                engine_status["web"] = {
                    "available": not bool(output.web_search_error),
                    "error": output.web_search_error,
                }

            # 无检索结果时不调用 LLM，直接拒答
            no_results = len(output.fused_results) == 0  # 无融合结果
            if no_results:  # 触发拒答
                refusal_text = "这个问题我目前没有找到相关的医学资料，建议咨询医生获取专业建议。"  # 拒答文案
                yield _encode_stream_event({  # 输出元信息
                    "type": "meta",
                    "result_count": 0,
                    "entities": [],
                    "used_engines": output.used_engines,
                    "engine_result_counts": {name: 0 for name in output.engine_results},
                    "engine_status": engine_status,
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                })
                yield _encode_stream_event({"type": "token", "text": refusal_text})  # 拒答文字
                yield _encode_stream_event({  # 完成事件（带空引用）
                    "type": "done",
                    "answer": {
                        "answer": refusal_text,
                        "items": [],
                        "confidence": 0.0,
                        "references": [],
                    },
                    "reference_details": [],
                    "used_engines": output.used_engines,
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "complete": True,
                })
                return  # 直接结束，不走 LLM

            yield _encode_stream_event({  # 输出元信息事件（前端展示命中/引擎状态）
                "type": "meta",
                "result_count": len(output.fused_results),
                "intent": output.query_plan.intent if output.query_plan else "general_medical",
                "entities": output.query_plan.entities if output.query_plan else [],
                "planner_fallback": bool(
                    output.query_plan and output.query_plan.fallback
                ),
                "used_engines": output.used_engines,
                "engine_result_counts": engine_result_counts,
                "engine_status": engine_status,
                "session_id": session_id,
                "memory_used": contextualized.used_memory,
                "resolved_query": (
                    contextualized.resolved if contextualized.used_memory else ""
                ),
                "retrieval_id": output.retrieval_id,
                "conversation_id": conversation_id,
            })
            answer: dict = {  # 初始化答案对象
                "answer": "",
                "items": [],
                "confidence": 0.0,
                "references": [],
            }
            generation_complete = False  # 是否完整生成
            for generation_event in get_llm().generate_answer_stream(  # 流式调用大模型生成答案
                query=question,
                context=output.to_context_text(),
                intent=(
                    output.query_plan.intent
                    if output.query_plan else "general_medical"
                ),
                history=_conversation_memory.format_history(history),
            ):
                event_type = generation_event.get("type")  # 事件类型
                if event_type == "token":  # 文字片段
                    yield _encode_stream_event({  # 转发给前端
                        "type": "token",
                        "text": generation_event.get("text", ""),
                    })
                elif event_type == "error":  # 生成出错
                    yield _encode_stream_event({  # 转发错误
                        "type": "error",
                        "message": generation_event.get("message", "回答生成中断"),
                    })
                elif event_type == "result":  # 最终结果
                    candidate = generation_event.get("answer")  # 候选答案
                    if isinstance(candidate, dict):  # 是字典结构
                        answer = candidate  # 采用
                    generation_complete = bool(generation_event.get("complete"))  # 是否完成

            answer_text = str(answer.get("answer", ""))  # 取答案文本
            if answer_text and generation_complete:  # 有答案且生成完整
                _conversation_memory.append(  # 写入记忆
                    session_id=session_id,
                    user=question,
                    resolved_query=contextualized.resolved,
                    assistant=answer_text,
                )
                _append_message(conversation_id, "assistant", answer_text)  # 同时写入历史记录
            # 填充引用：来源名 / 溯源链接 / 证据等级
            refs = _extract_references(output.fused_results)  # 提取引用
            if refs:  # 有引用
                answer["references"] = refs  # 写入 done 事件
            # 组装引用详情（前端引用面板）
            reference_details = []  # 引用详情列表
            for _idx, _r in enumerate(output.fused_results[:10], start=1):  # 前 10 条
                _content = _r.content or {}  # 结果内容
                _source_path = str(_content.get("source_path") or "")  # 文件路径
                _source_link = ""  # 原文链接
                if _source_path.startswith(("http://", "https://")):  # 已是外链
                    _source_link = _source_path
                reference_details.append({  # 组装详情
                    "index": _idx,  # 引用编号
                    "title": _content.get("title") or _content.get("name") or "未知",  # 标题
                    "source_path": _source_path,  # 文件路径
                    "source_link": _source_link,  # 外链（可能为空）
                    "source_type": getattr(_r, "source", "") or "",  # 来源引擎
                    "evidence_level": _determine_evidence_level(_r),  # 证据等级
                    "preview": str(_content.get("text") or _content.get("document") or "")[:200],  # 正文预览
                })
            # 按证据等级排序：高 → 中 → 低（编号保持原样，与回答引用对应）
            _level_weight = {"高": 0, "中": 1, "低": 2}  # 等级权重
            reference_details.sort(  # 排序
                key=lambda _d: _level_weight.get(_d.get("evidence_level", "中"), 1)
            )
            # 回答本身表明“没有相关资料”时，不展示引用来源
            _refusal_markers = ("没有找到相关", "未找到相关", "没有收录", "暂未收录", "无法回答")
            if any(_m in answer_text for _m in _refusal_markers):  # 命中拒答表述
                reference_details = []  # 清空引用详情
                answer["references"] = []  # 清空引用
            yield _encode_stream_event({  # 输出完成事件
                "type": "done",
                "answer": answer,
                "reference_details": reference_details,  # 引用详情
                "used_engines": output.used_engines,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "complete": generation_complete,
            })
        except Exception as exc:  # 任何异常
            yield _encode_stream_event({  # 输出错误事件
                "type": "error",
                "message": f"知识图谱检索失败：{type(exc).__name__}: {exc}",
            })
            yield _encode_stream_event({"type": "done"})  # 输出结束

    return StreamingResponse(  # 返回流式响应
        event_stream(),  # 生成器
        media_type="application/x-ndjson",  # NDJSON 媒体类型
        headers={  # 响应头
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/chat/memory/status")
def chat_memory_status():
    """Return non-sensitive memory backend diagnostics."""
    configured = config.MEMORY_BACKEND in {"redis", "hybrid"}
    available = bool(getattr(_conversation_memory, "backend_available", False)) if configured else True
    return {
        "backend": config.MEMORY_BACKEND if configured and available else "local",
        "configured_backend": config.MEMORY_BACKEND,
        "available": available,
        "ttl_seconds": config.MEMORY_TTL_SECONDS,
        "max_turns": config.MEMORY_MAX_TURNS,
    }


@router.delete("/api/chat/memory/{session_id}")  # 清空某个会话记忆接口
def clear_chat_memory(session_id: str):  # 接收会话 ID
    """只清空该浏览器的会话记忆，不影响其他会话。"""
    return {"cleared": _conversation_memory.clear(session_id)}  # 调用记忆存储清除
router.add_api_route("/api/chat", chat, methods=["POST"])
