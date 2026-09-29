# -*- coding: utf-8 -*-
"""Streamlit 界面入口。"""
import logging  # 标准日志：记录启动各阶段耗时，性能排查的第一手资料
import sys  # 改模块搜索路径用
import threading  # 后台线程跑模型预热，不阻塞 UI 首屏
from pathlib import Path  # 跨平台拼路径

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把项目根目录插到搜索路径最前：streamlit run src/main.py 直跑也能 import src 包

import streamlit as st  # Streamlit 每次交互整脚本重跑，重对象全靠 cache_resource 常驻内存

from src.chat_session import SessionStore  # 会话存储：多会话管理，数据落在 st.session_state
from src.config import setup_logging, validate  # 日志初始化 + 启动配置校验
from src import ui_chat, ui_sidebar, ui_style  # 三个 UI 模块分工：样式/侧边栏/聊天区，入口只做组装

APP_TITLE = "🏥 医小助 · 你的 AI 健康顾问"  # 页首标题常量

# 轻量 import 先跑：set_page_config + 标题 立刻出来，用户看到的是 Streamlit spinner
# 而不是浏览器白屏转圈
st.set_page_config(page_title="医小助 · 你的 AI 健康顾问", page_icon="🏥", layout="wide")  # 必须是最先执行的 st 调用：宽屏布局 + 浏览器标签页标题图标
st.title(APP_TITLE)  # 先渲染标题：重模型加载期间用户看到页面骨架而不是白屏

setup_logging()  # 控制台 + logs/app.log 双通道
validate()  # 缺 API Key 直接退出并提示，避免跑到一半才报错
logger = logging.getLogger("main")  # 入口模块日志器


def _warmup_reranker_async():  # reranker 预热函数：在后台线程里跑
    """后台线程预热 reranker CrossEncoder（27s 冷启动，不等它）。"""
    try:  # 预热失败不致命，捕获后仅记日志
        from src import reranker  # noqa: F401 — 触发模块级 _model=None
        reranker.is_ready()  # 内部调 _get_model() 真正加载
        reranker.rerank("测试", [{"text": "高血压患者应低盐饮食", "vec": 0.5}], 1)  # 真跑一次推理：把计算图和线程池烧热，首次提问不再冷启动
        logger.info("reranker 预热完成")  # 成功日志：确认后台预热生效
    except Exception as exc:  # 加载失败只告警
        logger.warning("reranker 预热失败：%s", exc)  # 查询时还有懒加载兜底，预热失败不影响功能


def _warmup_milvus_async():  # Milvus 连接预热函数
    """后台线程预热 Milvus 连接。"""
    try:  # 失败不阻塞启动
        from src.retrieval import _get_client  # 复用检索模块的连接工厂，全局同一客户端
        _get_client().list_collections(timeout=5)  # 一次轻量 RPC 完成建连握手，首次查询省去连接建立延迟
        logger.info("Milvus 连接预热完成")  # 成功日志
    except Exception as exc:  # 连接失败仅告警
        logger.warning("Milvus 预热失败（不阻塞启动）：%s", exc)  # 不阻塞启动：向量检索异常时主链仍可降级回答


def _warmup_ocr_async():  # OCR 预热函数：用户首次上传图片不等模型加载
    """后台线程预热 PaddleOCR（冷启动 ~22s，用户首次上传不等模型加载）。"""
    try:  # 失败不阻塞启动
        from src import image_ocr  # 模块内单例懒加载，_get_ocr 才真正初始化
        if image_ocr.ocr_available():  # 只探测是否安装 paddleocr，本身不加载模型
            image_ocr._get_ocr()  # 触发真正的模型加载（冷启动约 22s），提前搬到后台
            logger.info("OCR 预热完成")  # 成功日志
    except Exception as exc:  # 加载失败仅告警
        logger.warning("OCR 预热失败（不阻塞启动）：%s", exc)  # 上传图片时会再尝试懒加载


@st.cache_resource(show_spinner="⏳ 加载中…")  # 进程内只执行一次：脚本重跑/页面刷新直接拿缓存，重对象不重复加载
def _init():  # 重资源统一初始化入口
    """进程内只加载一次。重 import 全在这里，st.title 已经先跑完了。"""
    import time  # 计时各阶段耗时
    t0 = time.time()  # 总计时起点

    from src.bm25_index import BM25Index  # BM25 关键词倒排索引
    from src.ingestion import get_embedder  # 向量嵌入模型加载器（bge-m3）
    from src.memory import LongTermMemory, ShortTermMemory  # Redis 短期记忆 + Milvus 长期记忆
    from src.rag_chain import RAGChain  # RAG 主链
    from src.retrieval import load_records  # 从 Milvus 读取知识库切片
    from src.role import RoleManager  # 角色人设管理
    logger.info("[%.2fs] import 完成", time.time() - t0)  # 延迟导入耗时打点

    t1 = time.time()  # 重置阶段计时
    embedder = get_embedder()  # 加载向量模型（首次需下载权重，最耗时环节之一）
    logger.info("[%.2fs] embedder 完成 (+%.2fs)", time.time() - t0, time.time() - t1)  # 耗时日志

    t1 = time.time()  # 重置阶段计时
    records = load_records()  # 读全量知识切片，供 BM25 建索引
    logger.info("[%.2fs] load_records 完成 (+%.2fs), %d 条",  # 条数与耗时写日志
                time.time() - t0, time.time() - t1, len(records))

    t1 = time.time()  # 重置阶段计时
    bm25 = BM25Index()  # 建倒排索引对象
    bm25.build(records)  # 全量分词入索引，O(n) 一次性成本
    logger.info("[%.2fs] bm25.build 完成 (+%.2fs)", time.time() - t0, time.time() - t1)  # 耗时日志

    t1 = time.time()  # 重置阶段计时
    role_mgr = RoleManager()  # 加载角色配置
    logger.info("[%.2fs] RoleManager 完成 (+%.2fs)", time.time() - t0, time.time() - t1)  # 耗时日志

    t1 = time.time()  # 重置阶段计时
    chain = RAGChain(role_mgr)  # 组装 RAG 主链（内部创建 LLM 客户端）
    logger.info("[%.2fs] RAGChain 完成 (+%.2fs)", time.time() - t0, time.time() - t1)  # 耗时日志

    try:  # Redis 可能不可用，单独兜底
        short_mem = ShortTermMemory()  # 连接 Redis 短期记忆
    except Exception as exc:  # 连接失败则降级
        logger.warning("Redis 不可用，短期记忆降级：%s", exc)  # 告警但不中断：短期记忆是可选项
        short_mem = None  # 置 None 表示降级，调用方判空跳过

    t1 = time.time()  # 重置阶段计时
    try:  # Milvus 记忆库也可能不可用
        long_mem = LongTermMemory(embedder)  # 建长期记忆（用户病史/用药的向量库）
        logger.info("[%.2fs] LongTermMemory 完成 (+%.2fs)", time.time() - t0, time.time() - t1)  # 耗时日志
    except Exception as exc:  # 连接失败则降级
        logger.warning("Milvus 长期记忆不可用，已降级：%s", exc)  # 告警但不中断
        long_mem = None  # 置 None：无长期记忆也能正常问答

    # embedder 加载完再启动后台预热，避免 CUDA 并发加载死锁
    threading.Thread(target=_warmup_reranker_async, daemon=True).start()  # 后台预热 reranker；daemon 线程随主进程退出，不阻碍关闭
    threading.Thread(target=_warmup_milvus_async, daemon=True).start()  # 后台预热 Milvus 连接
    threading.Thread(target=_warmup_ocr_async, daemon=True).start()  # 后台预热 OCR

    logger.info("[%.2fs] _init 全部完成", time.time() - t0)  # 总耗时打点
    return embedder, role_mgr, chain, short_mem, long_mem, bm25  # 六个重对象作为缓存结果返回


def _rerank_ready():  # 不阻塞地查 reranker 状态的小工具
    """不阻塞地查 reranker 状态 — 只看内存里有没有 model，不触发加载。"""
    from src import reranker  # 局部导入拿模块单例
    return reranker.model_loaded()  # 只看标志位，绝不在 UI 线程触发模型加载


embedder, role_mgr, chain, short_mem, long_mem, bm25 = _init()  # 首次运行真正初始化，之后脚本重跑秒回缓存

ui_style.apply(role_mgr.list_roles())  # 注入全局 CSS 与各角色气泡配色
store = SessionStore(st.session_state)  # 会话存 session_state：Streamlit 重跑时数据不丢
role_id = ui_sidebar.render(role_mgr, store, bool(bm25.records), _rerank_ready())  # 渲染侧边栏并取当前角色；bm25.records 非空即索引就绪
session = store.current(role_id)  # 取该角色的当前会话
ui_chat.render(role_mgr, session, role_id, chain, embedder, bm25, short_mem, long_mem)  # 渲染聊天区并处理本轮输入