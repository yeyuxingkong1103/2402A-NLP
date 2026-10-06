"""Shared runtime settings, lazy services, and conversation memory."""
import os
import socket

from src import config
from src.core.memory import (
    ConversationContextualizer,
    ConversationMemoryStore,
    HybridConversationMemoryStore,
    MySQLConversationMemoryStore,
    RedisConversationMemoryStore,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目根目录（app.py 所在目录）
STATIC_DIR = os.path.join(BASE_DIR, "static")  # 前端静态文件目录
SERVER_HOST = "0.0.0.0"  # 监听所有网卡，便于局域网访问
SERVER_PORT = int(os.environ.get("MEDRAG_PORT", "5000"))  # 服务端口，默认 5000
_AUTH_COOKIE = "medrag_token"  # 登录 Cookie 名（HttpOnly，服务端强制登录用）


def _get_lan_ip() -> str:
    """获取访问当前主机时应使用的局域网 IPv4 地址。"""
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建 UDP 套接字（不真正发包）
    try:
        connection.connect(("8.8.8.8", 80))  # 建立到外网的假连接，让系统选出本机网卡
        return connection.getsockname()[0]  # 返回本机局域网 IPv4 地址
    except OSError:  # 外网不可达时降级处理
        try:
            return socket.gethostbyname(socket.gethostname())  # 用主机名反查 IP
        except OSError:
            return "127.0.0.1"  # 实在查不到就返回本机回环地址
    finally:
        connection.close()  # 关闭套接字，释放资源

try:  # 尝试导入知识图谱检索器
    from src.knowledge_graph.graph_retrieval import DrugGraphRetriever  # noqa: E402

    GRAPH_AVAILABLE = True  # 标记图谱模块可用
    GRAPH_IMPORT_ERROR = ""  # 清空导入错误信息
except Exception as exc:  # 导入失败（例如缺少 neo4j 包）
    DrugGraphRetriever = None  # 置空检索器类
    GRAPH_AVAILABLE = False  # 标记图谱模块不可用
    GRAPH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"  # 记录失败原因

_retriever = None  # 全局缓存图谱检索器实例


def get_retriever():
    """懒加载：第一次调用时才创建 DrugGraphRetriever。"""
    global _retriever  # 声明使用全局缓存变量
    if _retriever is None:  # 还没创建过
        _retriever = DrugGraphRetriever()  # 创建实例并缓存
    return _retriever  # 返回缓存的检索器


# ---------------------------------------------------------------------------
# 接入 RAG 生成服务；底层模型统一由 src/model/llm.py 提供
# ---------------------------------------------------------------------------
try:  # 尝试导入 RAG 答案生成服务
    from src.core.generator.service import RAGGenerator  # noqa: E402

    LLM_CLIENT_AVAILABLE = True  # 标记大模型模块可用
    LLM_CLIENT_ERROR = ""  # 清空错误信息
except Exception as exc:  # 导入失败
    RAGGenerator = None  # 置空生成器类
    LLM_CLIENT_AVAILABLE = False  # 标记不可用
    LLM_CLIENT_ERROR = f"{type(exc).__name__}: {exc}"  # 记录失败原因

_llm = None  # 全局缓存大模型生成服务
_contextualizer = None  # 全局缓存追问改写器

_memory_kwargs = dict(
    max_sessions=config.MEMORY_MAX_SESSIONS,  # 最多保留的会话数
    max_turns=config.MEMORY_MAX_TURNS,  # 每个会话记忆轮数
    ttl_seconds=config.MEMORY_TTL_SECONDS,  # 记忆过期时间
    max_history_chars=config.MEMORY_MAX_HISTORY_CHARS,  # 记忆文本最大字符数
)
if config.MEMORY_BACKEND == "hybrid":
    _conversation_memory = HybridConversationMemoryStore(
        config.MEMORY_REDIS_URL, redis_key_prefix=config.MEMORY_REDIS_KEY_PREFIX, **_memory_kwargs
    )
elif config.MEMORY_BACKEND == "mysql":
    _conversation_memory = MySQLConversationMemoryStore(**_memory_kwargs)
elif config.MEMORY_BACKEND == "redis":
    _conversation_memory = RedisConversationMemoryStore(
        config.MEMORY_REDIS_URL, key_prefix=config.MEMORY_REDIS_KEY_PREFIX, **_memory_kwargs
    )
else:
    _conversation_memory = ConversationMemoryStore(**_memory_kwargs)


def get_llm():
    """懒加载：第一次调用时创建 RAG 生成服务。"""
    global _llm  # 使用全局缓存
    if not LLM_CLIENT_AVAILABLE:  # 模块不可用时直接报错
        raise RuntimeError("项目 LLM 模块不可用：" + LLM_CLIENT_ERROR)
    if _llm is None:  # 还没创建
        _llm = RAGGenerator()  # 创建并缓存
    return _llm  # 返回生成器


def get_contextualizer():
    """复用共享模型，把追问改写成独立完整问题。"""
    global _contextualizer  # 使用全局缓存
    if _contextualizer is None:  # 还没创建
        _contextualizer = ConversationContextualizer(get_llm().model_client)  # 用同一模型客户端创建
    return _contextualizer  # 返回改写器


def _llm_supports_images() -> bool:
    """项目 LLM 是否支持图片输入（openai / anthropic 协议均支持）。"""
    return LLM_CLIENT_AVAILABLE  # 模块可用即认为支持图片


# ---------------------------------------------------------------------------
# 接入完整 RAG 检索管线：src/core/retrieval/pipeline.py
# （AI 语义规划 -> vector/graph 召回 -> 融合 -> 重排，供问答使用）
# ---------------------------------------------------------------------------
try:  # 尝试导入完整检索管线
    from src.core.retrieval.pipeline import RetrievalPipeline  # noqa: E402

    PIPELINE_AVAILABLE = True  # 标记管线可用
    PIPELINE_IMPORT_ERROR = ""  # 清空错误
except Exception as exc:  # 导入失败
    RetrievalPipeline = None  # 置空管线类
    PIPELINE_AVAILABLE = False  # 标记不可用
    PIPELINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"  # 记录失败原因

_pipeline = None  # 全局缓存检索管线实例


def get_pipeline():
    """懒加载：第一次调用时才创建 RetrievalPipeline（连接 Milvus/Neo4j）。"""
    global _pipeline  # 使用全局缓存
    if _pipeline is None:  # 还没创建
        _pipeline = RetrievalPipeline()  # 创建并缓存（此时才会连接向量库/图谱）
    return _pipeline  # 返回管线


# Resolve static assets from the project root, not from this package directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "static")
