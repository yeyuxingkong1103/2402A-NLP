import json  # 导入 json，用于序列化 sources 等字段

from fastapi import APIRouter, Depends, HTTPException  # 导入 FastAPI 路由、依赖注入、异常
from fastapi.responses import StreamingResponse  # 导入流式响应类
from pydantic import BaseModel  # 导入 pydantic 的 BaseModel，用于定义请求体
from sqlalchemy.ext.asyncio import AsyncSession  # 导入异步数据库会话类型

from ...db.base import get_db  # 导入数据库会话依赖
from ...db.models import Character, Message, Session, User  # 导入 ORM 模型
from ...deps import get_embedding, get_llm, get_milvus, get_redis, get_rerank  # 导入各依赖提供者
from ...services.chat_service import ChatService  # 导入聊天服务
from ...services.rag_pipeline import RAGPipeline  # 导入 RAG 流水线
from ..deps import get_current_user  # 导入当前用户依赖
from ..schemas import ok  # 导入统一响应包装函数

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])  # 创建路由，前缀和标签


class ChatIn(BaseModel):  # 定义聊天请求体
    session_id: int | None = None  # 会话 ID，可选
    character_id: int | None = None  # 角色 ID，可选
    content: str  # 用户输入内容


def _chat_service() -> ChatService:  # 构造 ChatService 的工厂函数
    return ChatService(  # 返回 ChatService 实例
        get_llm(),  # LLM 依赖
        RAGPipeline(get_embedding(), get_rerank(), get_milvus()),  # RAG 流水线依赖
        get_redis(),  # redis 依赖
    )


def _character_dict(c: Character) -> dict:  # 把 Character ORM 对象转成 dict
    return {
        "id": c.id, "name": c.name, "persona": c.persona,  # 基础字段
        "worldview": c.worldview, "relationship": c.relationship,  # 世界观、关系
        "hidden_setting": c.hidden_setting, "sample_dialogue": c.sample_dialogue,  # 隐藏设定、示例对话
    }


async def _resolve(session_id, user, db):  # 校验会话归属并取出会话和角色
    s = await db.get(Session, session_id)  # 按 ID 取会话
    if s is None or s.user_id != user.id:  # 会话不存在或不属于当前用户
        raise HTTPException(status_code=404, detail="会话不存在")  # 返回 404
    c = await db.get(Character, s.character_id)  # 取出会话对应角色
    return s, c  # 返回会话和角色


@router.post("")  # 注册 POST /api/v1/chat
async def chat(body: ChatIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):  # 非流式聊天接口
    if body.session_id is None:  # 如果缺少 session_id
        raise HTTPException(status_code=400, detail="缺少 session_id")  # 返回 400
    s, c = await _resolve(body.session_id, user, db)  # 校验会话并取角色
    svc = _chat_service()  # 构造聊天服务
    turn = await svc.prepare(s.id, user.id, _character_dict(c), body.content)  # 准备本轮消息和来源
    reply = await svc.llm.chat(turn.messages)  # 调用 LLM 获取完整回复
    db.add(Message(session_id=s.id, role="user", content=body.content))  # 写入用户消息
    db.add(Message(session_id=s.id, role="assistant", content=reply, sources=json.dumps(turn.sources, ensure_ascii=False)))  # 写入助手消息及来源
    await db.commit()  # 提交事务
    redis = get_redis()  # 获取 redis 客户端
    await redis.push_message(s.id, "user", body.content)  # 推送用户消息到 redis 短期上下文
    await redis.push_message(s.id, "assistant", reply)  # 推送助手回复到 redis
    return ok({"reply": reply, "sources": turn.sources})  # 返回统一格式响应


@router.post("/stream")  # 注册 POST /api/v1/chat/stream
async def chat_stream(body: ChatIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):  # 流式聊天接口
    if body.session_id is None:  # 如果缺少 session_id
        raise HTTPException(status_code=400, detail="缺少 session_id")  # 返回 400
    s, c = await _resolve(body.session_id, user, db)  # 校验会话并取角色
    svc = _chat_service()  # 构造聊天服务
    character = _character_dict(c)  # 转成 dict 备用

    async def gen():  # 定义 SSE 生成器
        buf = []  # 缓存流式返回的 token
        try:  # 整体异常捕获
            turn = await svc.prepare(s.id, user.id, character, body.content)  # 准备本轮消息和来源
            yield f"event: sources\ndata: {json.dumps({'sources': turn.sources}, ensure_ascii=False)}\n\n"  # 先发 sources 事件
            async for token in svc.llm.chat_stream(turn.messages):  # 流式迭代 LLM 输出
                buf.append(token)  # 累积 token
                yield f"event: delta\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"  # 每个 token 发 delta 事件
            reply = "".join(buf)  # 拼接完整回复
            db.add(Message(session_id=s.id, role="user", content=body.content))  # 写入用户消息
            assistant_msg = Message(session_id=s.id, role="assistant", content=reply, sources=json.dumps(turn.sources, ensure_ascii=False))  # 构造助手消息
            db.add(assistant_msg)  # 写入助手消息
            await db.commit()  # 提交事务
            await db.refresh(assistant_msg)  # 刷新以获取自增 ID
            yield f"event: done\ndata: {json.dumps({'message_id': assistant_msg.id}, ensure_ascii=False)}\n\n"  # 发 done 事件
        except Exception as e:  # 捕获异常
            yield f"event: error\ndata: {json.dumps({'code': 3001, 'message': str(e)}, ensure_ascii=False)}\n\n"  # 发 error 事件

    return StreamingResponse(gen(), media_type="text/event-stream")  # 返回 SSE 流式响应