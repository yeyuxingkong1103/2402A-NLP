"""RAG 主链：混合召回（稠密向量 + BM25 稀疏 -> RRF 融合）-> BGE 重排 -> DeepSeek 生成（LCEL 编排）。"""  # 模块说明
import logging  # 运行日志
import time  # 计时（统计各检索阶段耗时）
from functools import lru_cache  # 缓存装饰器，用于链的单例缓存
from typing import List  # 类型注解

from langchain_core.documents import Document  # LangChain 文档对象
from langchain_core.output_parsers import StrOutputParser  # 把模型输出解析为纯字符串
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示词模板
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # LCEL 组合组件

import config  # 全局配置
from chat_store import format_history  # 从 Redis 取历史并格式化为文本
from hybrid_search import bm25_search, rrf_fuse  # BM25 稀疏召回 + RRF 多路融合
from llm import get_llm  # LLM 工厂函数
from reranker import rerank_documents  # BGE 重排普通函数
from vector_store import get_vector_store  # Milvus 向量库

logger = logging.getLogger("rag.chain")  # 本模块日志器


def build_prompt(domain: str = None, persona: str = None):  # 构建提示词：角色人设 + RAG 硬约束
    """用人设描述拼到 RAG 约束前；persona 为空时回退默认「知识库问答助手」。"""
    persona = persona or "你是一个严谨的中文知识库问答助手"  # 角色人设；未传入时回退默认
    system_content = (  # system 消息：人设 + RAG 约束 + 历史与参考内容占位符
        f"{persona}。请仅根据下面提供的【参考内容】回答用户问题，"
        "不要编造参考内容之外的信息；如果参考内容不足以回答，请直接说明没有足够信息。\n"
        "回答末尾用 [编号] 标注引用了哪些片段。\n\n"
        "【历史对话】\n{history}\n\n"  # {history} 占位符：之前的多轮对话
        "【参考内容】\n{context}"  # {context} 占位符：检索到的片段文本
    )
    return ChatPromptTemplate.from_messages(  # 聊天提示词模板
        [("system", system_content), ("human", "{question}")]  # system + human 结构同原 PROMPT
    )


def collection_name_for(domain: str) -> str:  # 领域 -> Milvus 集合名
    """返回 domain 对应的 Milvus 集合名（domain 必传，必须为 domain_collections 的合法键）。"""
    return config.domain_collections[domain]


def retrieve(question: str, domain: str) -> List[Document]:  # 普通检索函数：两路召回 + RRF + 重排
    """稠密向量召回 k 条 + BM25 关键词召回 sparse_k 条，RRF 融合后 BGE 重排取 top_n。"""
    started = time.perf_counter()  # 总耗时起点
    vectorstore = get_vector_store(domain)  # 该领域对应的 Milvus 向量库（内部有缓存）
    t0 = time.perf_counter()  # 稠密召回起点
    dense_docs = vectorstore.similarity_search(question, k=config.retrieve_k)  # 路 1：稠密向量语义召回
    t1 = time.perf_counter()  # 稀疏召回起点
    sparse_docs = bm25_search(  # 路 2：BM25 关键词召回
        collection_name_for(domain), question, config.sparse_k
    )
    t2 = time.perf_counter()  # 融合起点
    candidates = rrf_fuse([dense_docs, sparse_docs], k=config.rrf_k)  # RRF 按排名融合、按 pk 去重
    t3 = time.perf_counter()  # 重排起点
    result = rerank_documents(candidates, question, config.rerank_top_n)  # 融合候选统一交 BGE 重排
    t4 = time.perf_counter()  # 全流程结束
    logger.info(  # 各阶段条数与耗时（排查召回质量/延迟时最有用的一条）
        "检索完成 domain=%s 稠密=%d(%.2fs) 稀疏=%d(%.2fs) 融合=%d(%.2fs) 重排=%d(%.2fs) 合计=%.2fs",
        domain,
        len(dense_docs), t1 - t0,
        len(sparse_docs), t2 - t1,
        len(candidates), t3 - t2,
        len(result), t4 - t3,
        t4 - started,
    )
    if not result:  # 一条都没召回：通常是集合没入库或领域选错
        logger.warning("检索结果为空 domain=%s（请确认该领域集合已 ingest）", domain)
    return result


def format_docs(docs: List[Document]) -> str:  # 把检索结果格式化成提示词上下文
    """把重排后的片段拼成带编号和来源的上下文文本。"""
    parts = []  # 各片段文本容器
    for i, doc in enumerate(docs, start=1):  # 从 1 开始逐条编号
        source = doc.metadata.get("source", "unknown")  # 片段来源文件
        score = doc.metadata.get("rerank_score")  # 重排相关性得分
        head = f"[{i}] 来源: {source}"  # 片段头部：编号 + 来源
        if score is not None:  # 有得分则附上相关度
            head += f"（相关度 {float(score):.3f}）"
        parts.append(f"{head}\n{doc.page_content}")  # 头部 + 片段正文
    return "\n\n".join(parts)  # 用空行拼接所有片段


@lru_cache(maxsize=32)  # 按 (domain, user_id, persona) 缓存多条链（每用户+领域+角色独立）
def get_rag_chain(domain: str, user_id: int = None, persona: str = None):  # 工厂函数：构建 RAG 链
    """返回 LCEL 链：question(str) -> answer(str)，支持 invoke / stream。

    domain 指定该领域的集合检索，并把该用户该领域的历史对话注入 prompt（必传）。
    user_id 用于隔离不同用户的对话历史（None 时走默认键，适合命令行模式）。
    persona 为该用户的角色人设文本，注入 system prompt 决定回答风格（None 时用默认人设）。
    """

    def load_history(_):  # 链内动态取历史（每次提问时执行，不是构建链时）
        return format_history(user_id, domain)  # 从 Redis 取该用户该领域最近 N 轮历史

    return (  # LCEL 管道：数据从左向右流动
        {
            "context": RunnableLambda(lambda q: retrieve(q, domain))  # 普通检索函数包装成链组件
            | RunnableLambda(format_docs),  # 检索结果 -> 格式化为上下文文本
            "question": RunnablePassthrough(),  # 问题原样透传
            "history": RunnableLambda(load_history),  # 动态取该用户的对话历史
        }
        | build_prompt(domain, persona)  # 填充提示词模板（角色人设 + RAG 约束 + {history}/{context}/{question}）
        | get_llm()  # 调用 DeepSeek 生成答案
        | StrOutputParser()  # 把输出解析为字符串
    )
