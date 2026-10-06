"""FastAPI 路由：用户注册/登录/问答/历史管理。"""  # 模块说明
import logging  # 运行日志
import time  # 计时（统计问答耗时）
from logging.handlers import RotatingFileHandler  # 按大小滚动的文件日志

from fastapi import Depends, FastAPI  # FastAPI 核心组件
from fastapi.responses import StreamingResponse  # 流式响应
from fastapi.staticfiles import StaticFiles  # 静态文件挂载（前端页面）
from pydantic import BaseModel  # 请求体模型
from sqlalchemy.orm import Session  # SQLAlchemy 会话

import config  # 全局配置
from auth import (  # 认证相关
    get_current_user,  # 从 token 提取用户
    login_user,  # 登录
    register_user,  # 注册
)
from chat_store import add_message, clear_history, get_history  # Redis 历史操作
from database import Role, init_db, get_db, get_role_persona  # 数据库初始化、依赖与角色查询
from rag_chain import get_rag_chain  # RAG 链

if not logging.getLogger().handlers:  # 只在根 logger 未配置时配置，避免重复挂 handler、重复打开日志文件
    log_dir = config.base_dir / "logs"  # 日志目录（基于项目根目录，不受运行时 cwd 影响）
    log_dir.mkdir(parents=True, exist_ok=True)  # 目录必须先存在，否则文件 handler 构造会直接报错
    logging.basicConfig(  # 同时输出到文件和控制台
        level=logging.INFO,  # 默认 INFO：请求入口、耗时、业务失败都能看到
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # 时间 + 级别 + 模块 + 内容
        handlers=[
            RotatingFileHandler(  # 文件：写满 5MB 自动滚动，最多保留 5 个备份
                log_dir / "rag.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
            ),
            logging.StreamHandler(),  # 控制台：保持终端实时可见
        ],
    )
logging.getLogger("jieba").setLevel(logging.WARNING)  # 抑制第三方库的 DEBUG 噪音（BM25 分词会刷屏）
logger = logging.getLogger("rag.api")  # 本模块日志器

app = FastAPI(title="RAG 知识库问答 API", version="1.0")  # FastAPI 应用实例


# ---------- 请求体模型 ----------
class RegisterRequest(BaseModel):  # 注册请求
    username: str
    password: str


class LoginRequest(BaseModel):  # 登录请求
    username: str
    password: str


class AskRequest(BaseModel):  # 提问请求
    question: str
    domain: str  # 领域（必填）
    role: str  # 提问对象角色名（doctor/teacher/admin，决定 AI 的回答口吻）


class ClearHistoryRequest(BaseModel):  # 清空历史请求
    domain: str  # 领域（必填）


# ---------- 启动时初始化数据库 ----------
@app.on_event("startup")  # FastAPI 启动事件
def startup():  # 启动时自动建表
    init_db()
    logger.info("数据库初始化完成，服务已就绪")  # 启动日志


# ---------- 路由 ----------
@app.post("/register")  # 注册新用户
def api_register(req: RegisterRequest, db: Session = Depends(get_db)):
    """注册普通用户（无角色绑定）。"""
    user = register_user(db, req.username, req.password)
    return {"message": "注册成功", "user": user.to_dict()}


@app.post("/login")  # 登录获取 JWT
def api_login(req: LoginRequest, db: Session = Depends(get_db)):
    """验证用户名密码，返回 JWT 令牌。"""
    token = login_user(db, req.username, req.password)  # 登录并拿 JWT
    return {"token": token}  # 返回令牌


@app.get("/roles")  # 返回可供提问的角色列表（前端下拉数据源）
def api_roles(db: Session = Depends(get_db)):
    """返回全部提问对象角色（name + 人设描述）。"""
    roles = db.query(Role).all()  # 查全部角色
    return {"roles": [{"name": r.name, "persona": r.persona} for r in roles]}


@app.post("/ask")  # 问答（流式返回）
async def api_ask(  # 用 async + astream：同步路由会被丢到线程池线程，langchain 的 dict Runnable 内部用 asyncio.gather 需要 event loop，线程池线程没有 loop 会报 RuntimeError
    req: AskRequest,
    user: dict = Depends(get_current_user),  # 鉴权：从 token 提取用户
):
    """提问接口；按所选角色人设流式返回答案。"""
    persona = get_role_persona(req.role)  # 按本次选择的提问对象取人设文本
    chain = get_rag_chain(req.domain, user["user_id"], persona)  # 构建该用户+领域+角色的 RAG 链

    async def stream_answer():  # 异步生成器：逐段输出答案
        answer = ""  # 收集完整答案用于存历史
        started = time.perf_counter()  # 计时起点（用于统计回答耗时）
        logger.info(  # 请求入口日志（只记问题长度，不落问题原文）
            "问答开始 user_id=%s domain=%s role=%s 问题长度=%d",
            user["user_id"], req.domain, req.role, len(req.question),
        )
        try:  # 生成过程异常时记录堆栈，便于排查
            async for chunk in chain.astream(req.question):  # 异步流式，在 event loop 线程跑
                answer += chunk
                yield chunk  # 流式返回当前片段
        except Exception:  # 生成失败：先记日志再原样抛出（前端会显示错误）
            logger.exception("问答失败 user_id=%s domain=%s", user["user_id"], req.domain)
            raise
        # 流结束后把本轮 Q&A 存入 Redis 历史
        if req.domain:
            add_message(user["user_id"], req.domain, "user", req.question)
            add_message(user["user_id"], req.domain, "assistant", answer)
        logger.info(  # 完成日志：耗时 + 答案长度
            "问答完成 user_id=%s domain=%s 耗时=%.2fs 答案长度=%d",
            user["user_id"], req.domain, time.perf_counter() - started, len(answer),
        )

    return StreamingResponse(stream_answer(), media_type="text/plain")


@app.get("/history")  # 查询历史对话
def api_history(
    domain: str,  # 领域（必填）
    user: dict = Depends(get_current_user),
):
    """返回当前用户指定领域的历史对话。"""
    history = get_history(user["user_id"], domain)
    return {"history": history}


@app.post("/clear-history")  # 清空历史对话
def api_clear_history(
    req: ClearHistoryRequest,
    user: dict = Depends(get_current_user),
):
    """清空当前用户指定领域的历史对话。"""
    clear_history(user["user_id"], req.domain)
    return {"message": "历史已清空"}


@app.get("/domains")  # 返回全部领域列表
def api_domains(user: dict = Depends(get_current_user)):
    """返回全部可选领域及对应集合名（登录用户均可访问）。"""
    domains = []  # 组装领域信息
    for d, collection in config.domain_collections.items():
        domains.append({
            "domain": d,
            "collection": collection,
        })
    return {"domains": domains}


# ---------- 静态前端页面（放最后，避免拦截 /register /login 等显式路由） ----------
class NoCacheStaticFiles(StaticFiles):  # 静态资源禁用缓存：避免浏览器继续跑旧版前端
    def file_response(self, *args, **kwargs):  # 覆盖文件响应，统一补上缓存头
        response = super().file_response(*args, **kwargs)  # 先取默认响应
        response.headers["Cache-Control"] = "no-cache"  # 每次请求都向服务器校验（未改动时返回 304）
        return response  # 返回带缓存头的响应


app.mount("/", NoCacheStaticFiles(directory="static", html=True), name="static")  # 挂载前端页面
