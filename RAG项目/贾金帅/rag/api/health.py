"""Application and retrieval dependency health routes."""
from fastapi import APIRouter

from .dependencies import (
    GRAPH_AVAILABLE,
    GRAPH_IMPORT_ERROR,
    LLM_CLIENT_AVAILABLE,
    LLM_CLIENT_ERROR,
    _llm_supports_images,
    get_llm,
    get_retriever,
)

router = APIRouter(tags=["health"])


def _vector_status() -> dict:  # 检测向量库状态
    """真实检测 Milvus 向量库连接与集合是否存在。"""
    info: dict = {"available": False, "error": ""}  # 默认不可用
    try:  # 尝试连接
        from src import config as _cfg  # 读配置
        from pymilvus import connections, utility  # 导入 pymilvus

        # 复用向量引擎使用的连接别名；检测后保持连接，
        # 避免断开导致后续 /api/chat 向量检索报“should create connection first”
        connections.connect(  # 建立 Milvus 连接
            alias=_cfg.MILVUS_ALIAS,
            host=_cfg.MILVUS_HOST,
            port=str(_cfg.MILVUS_PORT),
        )
        info["collection"] = utility.has_collection(_cfg.MILVUS_COLLECTION)  # 检查集合是否存在
        info["available"] = bool(info["collection"])  # 集合存在即可用
        if not info["collection"]:  # 集合不存在
            info["error"] = f"集合不存在：{_cfg.MILVUS_COLLECTION}"  # 记录原因
    except ImportError:  # 没装 pymilvus
        info["error"] = "pymilvus 未安装"  # 记录
    except Exception as exc:  # 其他连接异常
        info["error"] = f"{type(exc).__name__}: {exc}"  # 记录异常
    return info  # 返回状态


@router.get("/api/health")  # 健康检查接口（前端页面加载时调用）
def health():
    llm_info: dict = {"available": LLM_CLIENT_AVAILABLE}  # 大模型模块是否可用
    if LLM_CLIENT_AVAILABLE:  # 可用时补充详情
        try:
            _instance = get_llm()  # 创建/获取生成器
            llm_info["protocol"] = _instance.protocol  # 协议（openai/anthropic）
            llm_info["model"] = _instance.model  # 模型名
            llm_info["supports_images"] = _llm_supports_images()  # 是否支持图片
        except Exception as exc:  # 获取失败
            llm_info["error"] = f"{type(exc).__name__}: {exc}"  # 记录错误
    else:  # 模块不可用
        llm_info["error"] = LLM_CLIENT_ERROR  # 记录导入错误
    return {  # 汇总返回各组件状态
        "status": "ok",  # 服务本身正常
        "service": "med-rag-api",  # 服务标识
        "graph": GRAPH_AVAILABLE,  # 图谱模块是否可用
        "graph_error": GRAPH_IMPORT_ERROR,  # 图谱导入错误
        "llm": llm_info,  # 大模型状态
        "vector": _vector_status(),  # 向量库状态（真实检测）
    }


def graph_status():  # 查看图谱检索是否可用
    """查看图谱检索是否可用、Neo4j 是否连得上。"""
    info: dict = {"available": GRAPH_AVAILABLE}  # 模块是否可用
    if GRAPH_AVAILABLE:  # 可用时实测连接
        try:
            get_retriever().verify_connectivity()  # 验证 Neo4j 连通性
            info["neo4j_connected"] = True  # 连接成功
        except Exception as exc:  # 连接失败
            info["neo4j_connected"] = False  # 标记未连接
            info["neo4j_error"] = f"{type(exc).__name__}: {exc}"  # 记录原因
    else:  # 模块不可用
        info["import_error"] = GRAPH_IMPORT_ERROR  # 返回导入错误
    return info  # 返回状态
router.add_api_route("/api/graph/status", graph_status, methods=["GET"])
