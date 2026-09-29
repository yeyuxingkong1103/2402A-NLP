# -*- coding: utf-8 -*-
"""集中管理所有配置，从 .env 文件加载。"""
import logging  # 日志框架
import logging.handlers  # 按天滚动的文件处理器在这个子模块里
import os  # 读环境变量
from dataclasses import dataclass  # 用 dataclass 声明配置结构
from pathlib import Path  # 定位项目根目录

from dotenv import load_dotenv  # 第三方库：把 .env 内容注入环境变量

load_dotenv()  # 自动加载项目根目录的 .env

_ROOT = Path(__file__).resolve().parent.parent  # 项目根目录（本文件在 src/ 下，向上两级）


def setup_logging(level: str = "INFO") -> None:  # 全局日志初始化入口
    """全局日志：控制台实时看 + logs/app.log 按天滚动落盘（WSL/Windows 通用）。

    在 Streamlit 入口最早调用一次；各模块用 logging.getLogger(__name__)。
    """
    log_dir = _ROOT / "logs"  # 日志目录放项目根下
    log_dir.mkdir(exist_ok=True)  # 已存在不报错
    logging.basicConfig(  # 全局只配一次，其他模块 getLogger 直接用
        level=getattr(logging, level.upper(), logging.INFO),  # 字符串级别转枚举，非法值回退 INFO
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # 格式：时间 + 级别 + 模块名 + 内容
        datefmt="%H:%M:%S",  # 时间只显示到秒，控制台更清爽
        handlers=[  # 双通道输出
            logging.StreamHandler(),  # 控制台实时看
            logging.handlers.TimedRotatingFileHandler(  # 文件按天滚动，长期留存
                log_dir / "app.log", when="midnight", encoding="utf-8"),  # 午夜切分；utf-8 防中文乱码
        ],
    )


def _flag(name: str, default: str = "true") -> bool:  # 开关解析小工具
    """把 .env 里的开关字符串转成布尔值。"""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")  # 宽松解析：1/true/yes/on 都算开


@dataclass  # 自动生成 __init__，字段声明即配置项
class Settings:  # 全局配置类
    """全局配置（dataclass 保证类型安全）。"""

    # ====== DeepSeek 大模型 ======
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")  # 必填项，缺失由 validate() 拦截
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # 默认官方地址，可换成兼容代理
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # 默认 deepseek-chat

    # ====== Milvus 向量库 ======
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")  # Milvus 服务地址
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "medical_knowledge")  # 知识库集合名
    milvus_memory_collection: str = os.getenv("MILVUS_MEMORY_COLLECTION", "user_memory")  # 用户长期记忆集合名

    # ====== Redis 短期记忆 ======
    redis_host: str = os.getenv("REDIS_HOST", "localhost")  # Redis 地址
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))  # getenv 返回字符串，端口要转 int
    redis_db: int = int(os.getenv("REDIS_DB", "0"))  # 逻辑库编号
    session_ttl: int = int(os.getenv("SESSION_TTL", "1800"))  # 秒

    # ====== 模型路径 ======
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 向量模型：bge-m3 对中英混合检索效果好
    reranker_model: str = os.getenv("RERANKER_MODEL", "models/bge-reranker-v2-m3")  # 重排模型本地路径
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # 推理设备：cpu/cuda

    # ====== 检索与相关性 ======
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))  # 引用准入阈值，低于此不展示不引用
    vec_threshold: float = float(os.getenv("VEC_RELEVANCE_THRESHOLD", "0.45"))  # 向量召回粗筛阈值
    reranker_enabled: bool = _flag("RERANKER_ENABLED")  # 重排总开关：关掉则回退用向量分数排序
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "12"))  # 重排候选数：先粗召回再多取一些精排
    answer_top_k: int = int(os.getenv("ANSWER_TOP_K", "5"))  # 普通问答取 Top-5
    list_top_k: int = int(os.getenv("LIST_TOP_K", "8"))  # 清单型问题取更多条，保证找全

    # ====== 知识库路径 ======
    guideline_path: str = os.getenv(  # 知识库 PDF 路径，可换成其他指南
        "GUIDELINE_PATH", "data/guidelines/国家基层高血压防治管理指南2025版.pdf"  # 默认内置高血压指南
    )


# 全局单例，导入即加载
settings = Settings()  # 导入即实例化，全项目共享同一份配置


def validate() -> None:  # 启动自检入口
    """启动时校验关键配置是否齐全，缺失则给出明确提示。"""
    missing = []  # 收集缺失项
    if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-请"):  # 空值或占位符都算未配置
        missing.append("DEEPSEEK_API_KEY（请在 .env 里填入真实 Key）")  # 记入缺失清单
    if missing:  # 有缺失就终止启动
        raise SystemExit(  # 退出并打印修复指引
            f"❌ 配置缺失：{', '.join(missing)}\n"  # 列出缺哪些配置
            f"   请复制 .env.example 为 .env 并填写正确值"  # 告诉用户怎么修
        )