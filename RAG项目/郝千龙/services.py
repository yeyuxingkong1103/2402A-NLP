# -*- coding: utf-8 -*-  # 声明文件编码为 utf-8，保证中文字符正常解析
"""服务层：会话管理、RAG 端到端编排、知识库动态更新。"""  # 模块级文档字符串，说明本文件职责
from __future__ import annotations  # 开启 PEP 563 延迟注解求值，让 int | None 等类型注解在旧版本也能用

from functools import lru_cache  # 导入 lru_cache，用于把检索器做成进程内单例缓存

from sqlalchemy import func  # 导入 sqlalchemy 的 func，用于在 ORM 查询里调用 count 等聚合函数

from database import ChatMessage, ChatSession, KnowledgeChunk, SessionLocal, init_db  # 导入数据库表模型、会话工厂和初始化函数
from ingest import ingest_file  # 导入文件入库函数，用于知识库动态更新
from logger import log  # 导入全局日志器，统一输出日志
from memory_store import get_messages, push_messages  # 导入短期记忆读写函数（Redis / 进程内）
from rag import TranslationRetriever  # 导入 RAG 检索器类，封装 BM25 + 向量混合检索
from generation import chat, chat_stream, format_retrieved  # 导入大模型生成与检索结果格式化函数
from pathlib import Path  # 导入 Path 类型，仅用于类型注解（add_knowledge 形参）


@lru_cache(maxsize=1)  # 装饰器：进程内最多缓存 1 个实例，避免重复构建 BM25 索引
def get_retriever() -> TranslationRetriever:
    """检索器单例：BM25 索引构建昂贵，进程内只建一次；更新后需 cache_clear。"""
    init_db()  # 首次调用时确保数据库表已创建（幂等）
    return TranslationRetriever.build()  # 构建并返回检索器（含 BM25 索引与 Milvus 连接）


def list_roles_with_data() -> list[dict]:
    """角色列表：只保留知识库中有数据的角色（前端选择框用）。

    数据来源优先级：
    1. Milvus 分区统计（与"按角色分区检索"的真实可用数据一致）；
    2. Milvus 不可用时回退 SQL knowledge_chunks 按角色计数；
    3. 两者都拿不到、或所有角色都无数据（首次空库）时返回全部角色，
       避免选择框为空导致系统不可用。
    """
    from roles import list_roles  # 延迟导入：避免循环依赖与启动期开销
    from vector_store import role_data_stats  # 延迟导入 Milvus 分区统计函数

    all_roles = list_roles()  # 先拿到全部角色定义，作为后续过滤和兜底的基准
    stats = role_data_stats()  # Milvus 按分区统计 {role_code: 条数}，与检索时真实可用数据一致
    if stats is None:  # Milvus 不可用或查询失败时回退到 SQL 统计
        try:
            with SessionLocal() as db:  # 开启数据库会话上下文，自动关闭
                rows = (  # 查询每个 role_code 的分块数量
                    db.query(KnowledgeChunk.role_code, func.count(KnowledgeChunk.id))  # 选出角色码与对应分块计数
                    .group_by(KnowledgeChunk.role_code)  # 按角色码分组聚合
                    .all()  # 立即执行并取全部结果
                )
            stats = {role: n for role, n in rows}  # 把查询结果转成 {role_code: 条数} 字典
        except Exception as exc:  # 捕获 SQL 异常，避免角色接口整体崩掉
            log.warning("sql role stats failed: %s", exc)  # 记录警告日志便于排查
            return all_roles  # 兜底：直接返回全部角色，保证选择框不空
    log.info("role data stats: %s", stats)  # 记录各角色数据量，便于运维核对
    return [r for r in all_roles if stats.get(r["code"], 0) > 0] or all_roles  # 只保留有数据的角色；若全为空则兜底返回全部角色


def ensure_session(user_id: int, role_code: str, session_id: int | None) -> int:
    """确保会话存在：传入 session_id 且属于该用户则复用，否则新建会话。"""
    init_db()  # 确保表结构已初始化（幂等，多次调用无副作用）
    with SessionLocal() as db:  # 打开数据库会话上下文，退出时自动 close
        if session_id:  # 前端传入了会话 id，尝试复用已有会话
            row = db.get(ChatSession, session_id)  # 按主键查询会话记录
            if row and row.user_id == user_id:  # 校验归属，防止越权复用他人会话
                return row.id  # 归属正确，直接复用该会话 id
        row = ChatSession(user_id=user_id, role_code=role_code, title="新对话")  # 否则新建会话，默认标题"新对话"
        db.add(row)  # 把新会话加入会话
        db.commit()  # 提交事务，写入数据库并生成自增 id
        db.refresh(row)  # 刷新对象，拿到数据库分配的 id
        return row.id  # 返回新会话 id


def persist_turn(session_id: int, user_text: str, assistant_text: str) -> None:
    """把本轮 user/assistant 两条消息全量落库（MySQL，长期冷数据）。"""
    with SessionLocal() as db:  # 打开数据库会话上下文
        db.add(ChatMessage(session_id=session_id, role="user", content=user_text))  # 写入用户消息
        db.add(ChatMessage(session_id=session_id, role="assistant", content=assistant_text))  # 写入助手消息
        db.commit()  # 提交事务，两条消息一起落库，保证一致性


def answer(
    user_id: int,  # 当前提问用户 id，用于会话归属与短期记忆隔离
    role_code: str,  # 当前角色编码，用于按角色分区检索与记忆
    query: str,  # 用户本轮提问原文
    session_id: int | None = None,  # 可选会话 id，传入则复用，None 则新建
    top_k: int | None = None,  # 可选检索条数，None 表示用默认值
) -> dict:  # 返回字典含 session_id / answer / retrieved / hit_count
    """端到端 RAG 问答：会话 → 记忆 → 检索 → 生成 → 写回记忆与落库。"""
    sid = ensure_session(user_id, role_code, session_id)  # ① 会话管理：复用或新建会话，拿到 sid
    history = get_messages(user_id, role_code, sid)       # ② 读短期记忆：取最近 N 轮对话作为上下文（rewrite_query 用）
    retriever = get_retriever()  # 获取检索器单例（含 BM25 + Milvus）；chat 内部会执行 rewrite_query → hybrid_search → rerank
    text, hits = chat(query, retriever, history=history, top_k=top_k, role_code=role_code)  # ③④⑤ 检索+生成：chat 内部完成 query 改写、混合召回、重排、最终生成
    push_messages(  # ⑥ 写回短期记忆（Redis / 进程内），供下一轮作为 history 读取
        user_id,  # 用户维度
        role_code,  # 角色维度
        sid,  # 会话维度，三级 key 共同定位记忆队列
        [{"role": "user", "content": query}, {"role": "assistant", "content": text}],  # 本轮 user+assistant 两条消息
    )
    persist_turn(sid, query, text)  # ⑦ 全量消息持久化到 MySQL，便于回溯/审计
    log.info("chat user=%s role=%s session=%s", user_id, role_code, sid)  # 记录本轮调用信息，便于追踪
    return {  # 组装返回给前端的结果
        "session_id": sid,  # 会话 id（前端下次提问需回传以保持多轮）
        "answer": text,  # 模型生成的最终回答
        "retrieved": format_retrieved(hits),  # 检索资料（调试/展示用）
        "hit_count": len(hits),  # 命中条数，前端可展示"参考了 N 条资料"
    }


def add_knowledge(path: Path) -> int:
    """知识库动态更新入口：解析文件 → 入 BM25/Milvus/MySQL，返回分块数。"""
    return ingest_file(path, get_retriever())  # 复用单例检索器，把新文件解析、分块并写入三处索引


# =====================================================================
# 知识点说明（RAG：端到端编排）
# ---------------------------------------------------------------------
# 1. answer() 串起完整 RAG 链路：
#    会话管理 → 读短期记忆（memory_store）→ Query 改写（query_rewrite）
#    → 混合召回（rag.hybrid_search：BM25 + 向量）→ 重排（rerank）
#    → 生成（generation.chat：LangChain 优先、OpenAI 兜底）
#    → 写回短期记忆（Redis）+ 全量消息持久化（MySQL）。
# 2. 双层记忆设计：Redis 存"最近 N 轮"供本轮生成用（热数据，O(1) 读写）；
#    MySQL chat_messages 全量落库（冷数据，可回溯/审计）。大模型本身
#    无状态，"多轮对话能力"完全靠应用层外部记忆拼装出来。
# 3. get_retriever 用 lru_cache 单例：BM25 索引构建昂贵，进程内只建一次；
#    add_knowledge 更新后由调用方 cache_clear 重建索引，保证增量可见。
# =====================================================================
