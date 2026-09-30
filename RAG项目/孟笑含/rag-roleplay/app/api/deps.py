# -*- coding: utf-8 -*-
"""FastAPI 依赖：数据库会话、Redis、短期记忆、大模型、当前用户。"""
from fastapi import Depends, Header, HTTPException, Request
# 解析：FastAPI 依赖注入组件与异常

from app.config import settings
# 解析：全局配置
from app.core.llm import OpenAILLM
# 解析：大模型兼容层
from app.models.db import get_db
# 解析：数据库会话依赖
from app.models.tables import User
# 解析：用户表模型
from app.store.memory import RedisMemoryStore
# 解析：短期记忆（Redis 实现）
'''
Depends：依赖注入核心，路由里 x = Depends(func) 会自动调用 func。
Header：读取请求头，用于 X-Token。
Request：拿到 request.app.state，这是跨请求共享状态的关键。
get_db 直接从 models.db 导入，不是本文件定义的——本文件只是"转发"。
RedisMemoryStore：短期记忆的 Redis 实现类
'''

# 依赖注入：全局 Redis 客户端（进程内单例）
def get_redis(request: Request):
    # 解析：返回全局 Redis 客户端
    return request.app.state.redis
    # 解析：从 app.state 取启动时创建的连接（所有请求共享）
'''
app.state 是 FastAPI 应用级命名空间，整个进程只有一份。
Redis 客户端在应用启动时（lifespan/startup）创建并挂到 app.state.redis。
这里只做"取出来"，保证所有请求复用同一个连接池，避免每次新建连接。
注意：没有 async，因为是纯内存取值，不涉及 IO。
'''

# 依赖注入：短期记忆（Redis List，最近10轮）
def get_memory(request: Request, redis=Depends(get_redis)) -> RedisMemoryStore:
    # 解析：构造短期记忆对象（redis 由 get_redis 依赖注入）
    return RedisMemoryStore(
        # 解析：创建 Redis 短期记忆
        redis,
        # 解析：传入 Redis 客户端
        max_rounds=settings.history_max_rounds,
        # 解析：保留轮数（默认 10）
        ttl_seconds=settings.history_ttl_seconds,
        # 解析：过期时间（默认 7 天）
    )


# 依赖注入：大模型客户端（LLM_MOCK=true 时返回预设回复的 MockLLM）
async def get_llm():
    # 解析：大模型客户端工厂
    if settings.llm_mock:
        # 解析：压测模式
        from app.core.llm import MockLLM
        # 解析：延迟导入 MockLLM

        return MockLLM()
        # 解析：返回假 LLM（不发外部请求）
    return OpenAILLM(
        # 解析：真实 LLM 客户端
        base_url=settings.llm_base_url,
        # 解析：API 地址
        api_key=settings.llm_api_key,
        # 解析：密钥
        model=settings.llm_model,
        # 解析：模型名
        timeout=settings.llm_timeout,
        # 解析：超时
    )
'''
llm_mock=true 时返回 MockLLM：压测/本地开发用，不发外部请求，直接返回预设回复。
真实模式返回 OpenAILLM：兼容 OpenAI 协议的客户端（可接 vLLM、Ollama、DeepSeek 等）。
延迟导入 MockLLM：只有压测时才加载，减少正常启动开销。
为什么 async？因为可能未来要异步初始化（如预热连接），这里保持接口一致。
'''


def _shared_rag_models(request: Request):
    """知识库与长期记忆共享的 BGE 模型实例（懒加载单例，避免重复占内存）。"""
    state = request.app.state
    # 解析：应用状态（跨请求共享）
    if not hasattr(state, "shared_embedder"):
        # 解析：未创建共享模型
        from app.rag.models import BGEM3Embedder, BGEReranker, LazyModelProxy
        # 解析：延迟导入模型类

        state.shared_embedder = LazyModelProxy(lambda: BGEM3Embedder(settings.embedding_model_path))
        # 解析：向量化模型懒加载代理（首次使用才加载约 2.5GB）
        state.shared_reranker = LazyModelProxy(lambda: BGEReranker(settings.rerank_model_path))
        # 解析：重排模型懒加载代理
    return state.shared_embedder, state.shared_reranker
    # 解析：返回共享实例（知识库与记忆两服务共用，省一半内存）
'''
hasattr 判断：第一次调用才创建，之后直接复用——懒汉式单例。
LazyModelProxy：包装一个 lambda，真正的 BGEM3Embedder 在第一次被调用方法时才加载。
BGE-M3 模型约 2.5GB，若启动就加载会拖慢启动、占内存。
代理对象本身几乎不占内存。
共享：知识库和长期记忆两个服务都调用这个函数，拿到同一个 embedder/reranker 实例，省一半内存。
线程安全：FastAPI 单进程 + GIL，hasattr 后赋值这段极短，实际并发下几乎不会重复创建；即便重复，也只是浪费一次加载。
'''

def _shared_milvus(request: Request):
    # 解析：共享 Milvus 连接
    state = request.app.state
    # 解析：应用状态
    if not hasattr(state, "shared_milvus"):
        # 解析：未创建
        from app.rag.milvus_store import MilvusStore
        # 解析：延迟导入

        state.shared_milvus = MilvusStore(settings.milvus_host, settings.milvus_port)
        # 解析：创建 Milvus 封装（惰性连接）
    return state.shared_milvus
    # 解析：返回共享实例
'''
同样的懒加载单例模式。
MilvusStore 封装了 Milvus 客户端，惰性连接（构造时不真正连，首次操作才连）。
知识库 Collection 和记忆 Collection 共用一个 MilvusStore 实例。
'''

def get_knowledge_service(request: Request):
    """知识库服务：构造零成本；BGE 模型首次真正检索时才加载（线程安全单例）。"""
    state = request.app.state
    # 解析：应用状态
    if not hasattr(state, "knowledge_service"):
        # 解析：未创建知识库服务
        from app.core.llm import OpenAILLM, QueryRewriter
        # 解析：延迟导入 LLM 与改写器
        from app.services.knowledge_service import KnowledgeService
        # 解析：延迟导入知识库服务

        embedder, reranker = _shared_rag_models(request)
        # 解析：取共享 BGE 模型
        rewriter = None
        # 解析：改写器默认关闭
        if settings.query_rewrite:
            # 解析：开启查询改写
            rewriter = QueryRewriter(
                # 解析：构造改写器
                OpenAILLM(
                    # 解析：独立 LLM 客户端
                    base_url=settings.llm_base_url,
                    # 解析：API 地址
                    api_key=settings.llm_api_key,
                    # 解析：密钥
                    model=settings.llm_model,
                    # 解析：模型
                    timeout=settings.llm_timeout,
                    # 解析：超时
                )
            )
        state.knowledge_service = KnowledgeService(
            # 解析：构造知识库服务（构造零成本，模型懒加载）
            embedder=embedder,
            # 解析：共享向量化模型
            reranker=reranker,
            # 解析：共享重排模型
            milvus=_shared_milvus(request),
            # 解析：共享 Milvus
            chunk_size=settings.chunk_size,
            # 解析：分块大小
            overlap=settings.chunk_overlap,
            # 解析：重叠量
            remove_watermark=settings.pdf_remove_watermark,
            # 解析：去水印开关
            query_rewriter=rewriter,
            # 解析：改写器（可能为 None）
            chunking_mode=settings.chunking_mode,
            # 解析：分块模式
            semantic_threshold=settings.semantic_threshold,
            # 解析：语义阈值
            parent_chunk_size=settings.parent_chunk_size,
            # 解析：父块大小
        )
    return state.knowledge_service
    # 解析：返回知识库服务单例
'''
hasattr 单例：全进程只有一个 KnowledgeService。
延迟导入：KnowledgeService、QueryRewriter 用到才导入，加快启动。
QueryRewriter（可选）：开启 query_rewrite 时，用一个独立 LLM 客户端把用户问题改写成更适合检索的形式（如补全指代、扩展同义词）。
传入的参数：
chunk_size / overlap：分块大小和重叠。
remove_watermark：PDF 去水印。
chunking_mode：分块策略（固定长度 / 语义 / 父子块）。
semantic_threshold：语义分块阈值。
parent_chunk_size：父子块模式下父块大小。
构造零成本：因为 embedder/reranker 是懒加载代理，这里 KnowledgeService(...) 只是保存引用。
'''

def get_memory_service(request: Request):
    """长期记忆服务：与知识库共享 BGE 模型与 Milvus 连接。"""
    state = request.app.state
    # 解析：应用状态
    if not hasattr(state, "memory_service"):
        # 解析：未创建记忆服务
        from app.services.memory_service import MemoryService
        # 解析：延迟导入

        embedder, reranker = _shared_rag_models(request)
        # 解析：取共享 BGE 模型
        state.memory_service = MemoryService(
            # 解析：构造长期记忆服务
            embedder=embedder,
            # 解析：共享向量化模型
            reranker=reranker,
            # 解析：共享重排模型
            milvus=_shared_milvus(request),
            # 解析：共享 Milvus
            similarity_threshold=settings.memory_similarity,
            # 解析：记忆相似度过滤阈值
        )
    return state.memory_service
    # 解析：返回记忆服务单例
'''
同样单例模式。
复用 _shared_rag_models 和 _shared_milvus：与知识库共享 BGE 和 Milvus。
similarity_threshold：记忆检索时的相似度过滤阈值，低于此值不注入。
'''


# 依赖注入：从 X-Token 解析当前用户（无效返回 401）
async def get_current_user(
    # 解析：鉴权依赖
    x_token: str | None = Header(default=None),
    # 解析：请求头 X-Token（可选）
    db=Depends(get_db),
    # 解析：数据库会话
    redis=Depends(get_redis),
    # 解析：Redis 客户端
) -> User:
    if not x_token:
        # 解析：无凭证
        raise HTTPException(status_code=401, detail="缺少登录凭证 X-Token")
        # 解析：401
    uid = redis.get(f"session:token:{x_token}")
    # 解析：Redis 查 token 对应的用户 ID
    if uid is None:
        # 解析：token 无效或过期
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
        # 解析：401
    user = db.get(User, int(uid))
    # 解析：MySQL 查用户
    if user is None:
        # 解析：用户不存在
        raise HTTPException(status_code=401, detail="用户不存在")
        # 解析：401
    return user
    # 解析：返回当前用户（路由层直接使用）

'''
Header(default=None)：从请求头读 X-Token，没有则为 None。
第一道校验：无 token → 401。
redis.get(f"session:token:{x_token}")：Redis 里查这个 token 对应的 user_id。
这是无状态鉴权：服务端不存 session 对象，只存 token→uid 映射。
Redis 自带过期（ex=7天），过期后自动返回 None。
第二道校验：Redis 没有 → token 无效/过期 → 401。
db.get(User, int(uid))：MySQL 查用户。
为什么还要查库？因为 token 有效不代表用户还在（可能被删）。
第三道校验：用户不存在 → 401。
返回 User 对象，路由层直接拿到当前用户。
'''
