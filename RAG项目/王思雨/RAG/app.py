# -*- coding: utf-8 -*-
"""HTTP 服务模块：FastAPI 提供认证、用户、会话、问答、知识库、评测、缓存共 16 个接口。"""

import hashlib                                # 导入 hashlib，用于生成教学版 token
import json                                   # 导入 json，用于流式结束帧的序列化
import time                                   # 导入 time，用于读取文件更新时间
from contextlib import asynccontextmanager    # 导入异步上下文管理器，用于声明 lifespan
from pathlib import Path                      # 导入 Path，用于路径处理

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile   # 导入 FastAPI 组件
from fastapi.middleware.cors import CORSMiddleware   # 导入跨域中间件
from fastapi.responses import StreamingResponse      # 导入流式响应类

import config                                 # 导入配置模块，端口与白名单从这里读
import db                                     # 导入数据库模块，取 Redis 客户端做缓存统计
import db_user                                # 导入用户数据模块，处理账号与角色
import enrich                                 # 导入增强模块，供知识库重建调用
import ingest                                 # 导入解析模块，供上传接口调用
import memory                                 # 导入记忆模块，供缓存清理与记忆集合初始化
import rag                                    # 导入问答模块，供问答接口调用
import schemas                                # 导入数据模型模块，定义请求与响应结构
import vector_store                           # 导入向量库模块，用于查询集合状态
from logger import get_logger                 # 导入日志工具，用于记录接口调用

logger = get_logger("app")                    # 创建本模块的 logger 实例

EVAL_JSON = Path(__file__).resolve().parent / "data" / "ragas_result.json"   # 评测结果文件（第 10 步产出）

def _check(fn) -> bool:                       # 内部函数：探测依赖是否可用
    """执行一个探测函数，成功返回 True，失败返回 False 并记录原因。"""
    try:                                      # 探测失败不应让接口报错
        fn()                                  # 执行探测
        return True                           # 可用
    except Exception as exc:                  # 不可用
        logger.warning("依赖探测失败：%s", exc)   # 记录原因
        return False                          # 不可用


def _fail(action: str, exc: Exception) -> HTTPException:   # 内部函数：统一异常转 HTTP 500
    """把内部异常转成 HTTP 500，完整堆栈只写日志，不返回给前端。"""
    logger.error("%s失败：%s", action, exc)    # 记录日志（含异常信息）
    return HTTPException(status_code=500, detail=str(exc))   # 构造响应，只带异常信息


_PROBES = (                                   # 三项依赖的探测动作，供启动检查与健康检查复用
    lambda: vector_store.get_milvus_client().list_collections(),   # 向量库探测
    lambda: db_user.list_roles(),             # 关系库探测
    lambda: memory.get_short_memory(0))       # 缓存探测


@asynccontextmanager                           # 声明为异步上下文管理器
async def lifespan(_app: FastAPI):             # 应用生命周期处理函数
    """启动时初始化角色与记忆集合并打印依赖状态；本次没有需要清理的资源。"""
    for label, action in (("预置角色", db_user.init_default_roles),      # 初始化两个预置角色
                          ("记忆集合", memory.create_memory_collection)):   # 建长期记忆集合
        try: action()                         # 执行初始化
        except Exception as exc: logger.warning("%s 初始化失败：%s", label, exc)   # 失败只告警
    logger.info("启动检查：Milvus=%s MySQL=%s Redis=%s", *[_check(f) for f in _PROBES])   # 打印依赖状态
    yield                                      # 分隔启动与关闭；yield 之后可写清理逻辑
    logger.info("服务已停止")                    # 关闭时记录一条日志


app = FastAPI(title="RAG 电力维修问答助手", version="1.0.0", lifespan=lifespan)   # 创建应用并挂生命周期

app.add_middleware(CORSMiddleware,             # 注册跨域中间件，供前端调用
    allow_origins=config.API_CORS_ORIGINS,    # 白名单来自 .env
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])   # 允许凭证、任意方法与请求头


@app.get("/health", response_model=schemas.HealthResp)        # 健康检查
def health() -> dict:                         # 处理函数
    """健康检查：返回服务状态与三项依赖的连通情况。"""
    milvus, mysql, redis = (_check(f) for f in _PROBES)   # 依次探测三项依赖
    return {"status": "ok", "milvus": milvus, "mysql": mysql, "redis": redis}   # 返回状态


@app.post("/auth/register", response_model=schemas.RegisterResp)   # 注册
def register(body: schemas.RegisterReq) -> dict:      # 处理函数
    """用户注册：用户名已存在返回 400。"""
    try:                                      # 只包住数据库调用，业务判断放在外面
        user_id = db_user.register_user(body.username, body.password)   # 调用注册
    except Exception as exc:                  # 数据库异常
        raise _fail("注册", exc)               # 统一转 500
    if not user_id: raise HTTPException(status_code=400, detail="用户名已存在")   # 已存在返回 400
    return {"user_id": user_id, "username": body.username, "msg": "注册成功"}   # 返回结果


@app.post("/auth/login", response_model=schemas.LoginResp)   # 登录
def login(body: schemas.LoginReq) -> dict:    # 处理函数
    """用户登录：失败返回 401，成功返回教学版 token。"""
    try:                                      # 只包住数据库调用
        user = db_user.login_user(body.username, body.password)   # 调用登录
    except Exception as exc:                  # 数据库异常
        raise _fail("登录", exc)               # 统一转 500
    if not user: raise HTTPException(status_code=401, detail="用户名或密码错误")   # 账号或密码错返 401
    token = hashlib.sha256(f"{user['user_id']}{config.PASSWORD_SALT}".encode()).hexdigest()   # 简单令牌
    return {"user_id": user["user_id"], "username": user["username"], "token": token}   # 返回结果


@app.get("/user/{user_id}", response_model=schemas.UserInfoResp)   # 查用户
def get_user(user_id: int) -> dict:           # 处理函数
    """按编号查询用户信息，不存在返回 404。"""
    try:                                      # 只包住数据库调用
        user = db_user.get_user(user_id)      # 查询用户
    except Exception as exc:                  # 数据库异常
        raise _fail("查询用户", exc)           # 统一转 500
    if not user: raise HTTPException(status_code=404, detail="用户不存在")   # 不存在返回 404
    return user                               # 返回用户信息


@app.get("/roles", response_model=schemas.RoleListResp)   # 角色列表
def list_roles() -> dict:                     # 处理函数
    """返回全部角色，供前端做角色选择。"""
    try:                                      # 统一捕获异常
        return {"roles": db_user.list_roles()}   # 返回角色列表
    except Exception as exc:                  # 查询失败
        raise _fail("查询角色", exc)               # 统一转 500


@app.post("/conversation", response_model=schemas.CreateConversationResp)   # 创建会话
def create_conversation(body: schemas.CreateConversationReq) -> dict:   # 处理函数
    """创建会话，返回新会话编号。"""
    try:                                      # 统一捕获异常
        cid = db_user.create_conversation(body.user_id, body.role_id, body.title)   # 创建会话
        return {"conversation_id": cid}       # 返回会话编号
    except Exception as exc:                  # 创建失败
        raise _fail("创建会话", exc)               # 统一转 500


@app.get("/conversation/{user_id}", response_model=schemas.ConversationListResp)   # 会话列表
def list_conversations(user_id: int) -> dict:   # 处理函数
    """列出某个用户的全部会话，附带角色名。"""
    try:                                      # 统一捕获异常
        roles = {r["role_id"]: r["role_name"] for r in db_user.list_roles()}   # 角色编号到名称
        return {"conversations": [            # 逐个会话补上角色名后返回
            {"conversation_id": c["conversation_id"], "role_name": roles.get(c["role_id"], ""),
             "title": c.get("title", ""), "created_at": c.get("created_at", "")}
            for c in db_user.list_conversations(user_id)]}
    except Exception as exc:                  # 查询失败
        raise _fail("查询会话", exc)               # 统一转 500


@app.delete("/conversation/{conversation_id}", response_model=schemas.MsgResp)   # 删除会话
def delete_conversation(conversation_id: int, user_id: int = Query(..., description="用户编号")) -> dict:
    """删除会话：只能删自己的，不属于自己则返回 404。"""
    try:                                      # 只包住数据库调用
        affected = db_user.delete_conversation(conversation_id, user_id)   # 双条件删除
    except Exception as exc:                  # 数据库异常
        raise _fail("删除会话", exc)           # 统一转 500
    if not affected: raise HTTPException(status_code=404, detail="会话不存在或不属于该用户")   # 404
    return {"msg": "deleted"}                 # 返回成功


@app.get("/message/{conversation_id}", response_model=schemas.MessageListResp)   # 消息列表
def list_messages(conversation_id: int, limit: int = Query(50, description="最多返回条数")) -> dict:
    """按编号升序返回某个会话的消息。"""
    try:                                      # 统一捕获异常
        return {"messages": db_user.list_messages(conversation_id, limit)}   # 返回消息列表
    except Exception as exc:                  # 查询失败
        raise _fail("查询消息", exc)               # 统一转 500


def _stream_chat(body: schemas.ChatReq):      # 内部函数：流式问答生成器
    """流式产出问答内容：逐段推正文，最后推来源帧与结束标记。"""
    try:                                      # 生成过程中断不应让连接挂死
        for piece in rag.ask(body.query, body.user_id, body.role_id, body.conversation_id, stream=True):   # 流式问答
            yield f"data: {piece}\n\n"        # SSE 帧：以两个换行结尾
    except Exception as exc:                  # 生成失败
        logger.error("流式问答失败：%s", exc)  # 记录日志
        yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"   # 推送错误帧
    yield "data: [DONE]\n\n"                  # 推送结束标记


@app.post("/chat", response_model=schemas.ChatResp)   # 问答
def chat(body: schemas.ChatReq):              # 处理函数
    """问答接口：非流式返回完整结果，流式用 SSE 逐段推送。"""
    if body.stream:                           # 流式分支
        return StreamingResponse(_stream_chat(body), media_type="text/event-stream")   # SSE 响应
    try:                                      # 非流式统一捕获异常
        result = rag.ask(body.query, body.user_id, body.role_id, body.conversation_id, stream=False)   # 调用
        return {"answer": result["answer"], "sources": result["sources"],       # 答案与来源
                "rewritten_query": result["rewritten_query"], "cache_hit": result["cache_hit"],
                "conversation_id": result["conversation_id"], "usage": result["usage"],
                "long_memory_used": result["long_memory_used"]}   # 其余五项，含长期记忆条数
    except Exception as exc:                  # 问答失败
        raise _fail("问答", exc)               # 统一转 500


@app.post("/kb/upload", response_model=schemas.UploadResp)   # 上传资料
async def kb_upload(file: UploadFile = File(..., description="PDF 文件")) -> dict:   # 处理函数
    """上传 PDF 并解析：本接口只解析不入库，入库请调用 /kb/rebuild。"""
    try:                                      # 统一捕获异常
        target_dir = Path(config.PDF_DIR)     # 资料目录来自配置
        target_dir.mkdir(parents=True, exist_ok=True)   # 目录不存在就创建
        target = target_dir / file.filename   # 目标文件路径
        target.write_bytes(await file.read())  # 写入上传内容
        result = ingest.parse_pdf(str(target))  # 解析该 PDF
        return {"file_name": file.filename, "status": "parsed",   # 文件名与状态
                "pages": result["pages"], "chunks": len(result["chunks"])}   # 页数与块数
    except Exception as exc:                  # 上传或解析失败
        raise _fail("上传解析", exc)               # 统一转 500


def _rebuild_task() -> None:                  # 内部函数：知识库重建后台任务
    """后台执行完整重建链路（解析 → 增强 → 入库），供 /kb/rebuild 调用。"""
    import reindex                            # 延迟导入，避免启动时就背重依赖
    result = reindex.run_reindex()            # 三步编排统一在 reindex.py 里
    if result["ok"]: logger.info("重建任务完成：%s", result["detail"])   # 三步全过记明细
    else: logger.error("重建任务失败于 %s：%s", result["failed"], result["error"])   # 记失败点


# 本路由不挂 response_model：schemas.KbRebuildResp 只有 status/msg 两个字段，
# 挂上之后 FastAPI 会按模型序列化，把要返回的 steps 过滤掉
# （第 8.5 步在 /cache/clear 上踩过同一个坑，见设计文档 16.9.3）；schemas.py 本步禁改。
@app.post("/kb/rebuild")                      # 重建知识库（不挂 response_model）
def kb_rebuild(tasks: BackgroundTasks) -> dict:   # 处理函数
    """触发知识库重建：后台依次跑解析、增强、入库，立即返回。"""
    tasks.add_task(_rebuild_task)             # 把任务交给后台执行
    logger.info("已提交知识库重建任务：解析 → 增强 → 入库")   # 记录日志
    return {"status": "started", "steps": ["ingest", "enrich", "index"],   # 状态与三步
            "msg": "重建任务已在后台执行，请稍后用 /kb/status 查询"}       # 提示


@app.get("/kb/status", response_model=schemas.KbStatusResp)   # 知识库状态
def kb_status() -> dict:                      # 处理函数
    """返回知识库状态：集合名、向量条数与增强文件更新时间。"""
    try:                                      # 统一捕获异常
        name, count = config.MILVUS_COLLECTION, 0   # 集合名；向量条数（向量库不可用时为 0）
        try:                                  # 向量库可能不可用，不可用时降级为 0
            client = vector_store.get_milvus_client()   # 取客户端
            client.load_collection(name)      # 加载集合
            counted = client.query(collection_name=name, filter="", output_fields=["count(*)"])   # 统计
            count = int(counted[0].get("count(*)", 0)) if counted else 0   # 取条数
        except Exception as exc:              # 向量库不可用
            logger.warning("知识库状态查询降级为 0：%s", exc)   # 记录原因
        updated = ""                          # 增强文件更新时间
        if enrich.OUT_JSON.exists():          # 文件存在才取时间
            updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(enrich.OUT_JSON.stat().st_mtime))
        return {"collection": name, "row_count": count, "last_update": updated}   # 返回状态
    except Exception as exc:                  # 其他异常
        raise _fail("查询知识库状态", exc)               # 统一转 500


@app.get("/eval/result", response_model=schemas.EvalResp)   # 评测结果
def eval_result() -> dict:                    # 处理函数
    """读取第 10 步产出的评测结果文件，文件不存在时返回空汇总与空明细。"""
    try:                                      # 统一捕获异常
        if not EVAL_JSON.exists():            # 还没跑过评测
            return {"summary": {}, "details": []}   # 返回空汇总与空明细，前端显示空表
        return json.loads(EVAL_JSON.read_text(encoding="utf-8"))   # 直接返回文件里的完整结构
    except Exception as exc:                  # 读取失败
        raise _fail("读取评测结果", exc)               # 统一转 500


def _scan_keys(client, pattern: str) -> list:   # 内部函数：按前缀扫描键
    """用 SCAN 游标分批扫描匹配的键，避免 KEYS 阻塞 Redis。"""
    found, cursor = [], 0                     # 保存匹配到的键；游标从 0 开始
    while True:                               # 分批扫描直到游标归零
        cursor, batch = client.scan(cursor=cursor, match=pattern, count=200)   # 扫一批
        found.extend(batch)                   # 收进结果
        if cursor == 0: break                 # 游标归零表示扫完
    return found                              # 返回全部匹配键


@app.get("/cache/stats", response_model=schemas.CacheStatsResp)   # 缓存统计
def cache_stats() -> dict:                    # 处理函数
    """统计答案缓存与短期记忆：键总数、按用户分组、有短期记忆的用户数。"""
    try:                                      # 统一捕获异常
        redis_client = db.get_redis_client()  # 取 Redis 客户端
        keys = _scan_keys(redis_client, "cache:rag:*")   # 扫描全部答案缓存键
        by_user = {}                          # user_id 字符串 → 该用户的缓存键数量
        for key in keys:                      # 键形如 cache:rag:answer:{uid}:{role}:{问题}
            parts = key.split(":")            # 按冒号切分
            uid = parts[3] if len(parts) > 3 else ""   # 第 4 段是用户编号，越界则为空串
            if uid.isdigit(): by_user[uid] = by_user.get(uid, 0) + 1   # 只统计数字编号并累加
        short_users = len(_scan_keys(redis_client, "mem:short:*"))   # 有短期记忆的用户数
        logger.info("缓存统计：答案缓存 %d 个、%d 个用户；短期记忆 %d 个用户", len(keys), len(by_user), short_users)   # 日志
        return {"total_keys": len(keys), "by_user": by_user,   # 答案缓存总数与按用户分组
                "short_memory_users": short_users}             # 有短期记忆的用户数
    except Exception as exc:                  # 统计失败
        raise _fail("统计缓存", exc)           # 统一转 500


@app.get("/cache/clear", response_model=schemas.ClearCacheResp)   # 清空缓存
def cache_clear(user_id: int = Query(..., description="用户编号")) -> dict:   # 处理函数
    """只清空指定用户的短期记忆与答案缓存。"""
    try:                                      # 统一捕获异常
        cleared = memory.clear_short_memory(user_id)   # 清空该用户的短期记忆
        redis_client = db.get_redis_client()  # 取 Redis 客户端
        # 键结构为 cache:rag:answer:{user_id}:{role}:{归一化问题}，模式必须带 answer: 段
        keys = _scan_keys(redis_client, f"cache:rag:answer:{int(user_id)}:*")   # 只扫该用户的缓存键
        if keys: redis_client.delete(*keys)   # 有缓存才批量删除
        logger.info("用户 %s 缓存已清理：缓存键 %d 个，短期记忆 %s", user_id, len(keys), bool(cleared))   # 日志
        return {"msg": "cleared", "cache_keys": len(keys), "short_memory": bool(cleared)}   # 返回明细
    except Exception as exc:                  # 清理失败
        raise _fail("清理缓存", exc)           # 统一转 500


# 启动方式：uvicorn app:app --reload --port 8000（端口可在 .env 的 API_PORT 调整）
