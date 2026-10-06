# -*- coding: utf-8 -*-
"""
检索模块：向量检索 + BM25 关键词检索 -> RRF 融合 -> CrossEncoder 重排 -> 分数过滤
支持多集合（按角色路由）和多路召回（知识库 + 长期记忆）

在系统中的位置（本模块是 RAG 链路的核心）：
    上游是对话接口——它拿到用户提问后调用 multi_route_recall/hybrid_search 取资料；
    下游依赖 db_milvus——向量检索、集合存在性检查、长期记忆查询、BM25 语料全量拉取都走它；
    产出的 Document 列表交给 prompt_templates.build_context 拼成提示词。

职责：
    「召回 + 精排 + 过滤」三件事。先宽召回保证不漏（向量 + 关键词两路各 TOP_K_RECALL 条），
    再用 CrossEncoder 精排保证准（只留 TOP_K_RERANK 条），最后用分数阈值滤掉明显不相关的。

关键设计取舍：
    1. 为什么要向量 + BM25 两路？向量擅长语义相近（"撞人赔偿"能命中"交通事故责任"），
       但容易漏掉精确的法律术语/案号；BM25 正好相反。两路互补再用 RRF 融合排名。
    2. 为什么用 RRF 而不是加权求和分数？因为两路的分数不可比（余弦相似度 vs BM25 分值），
       RRF 只用"排名"不用"分数"，天然不需要归一化，也更鲁棒。
    3. 重排模型和 BM25 索引都做模块级缓存：加载一次反复用，否则每次提问都要重新加载几百 MB 模型。
"""

import os  # 路径操作（校验重排模型目录是否存在）
from typing import List, Optional, Dict  # 类型标注

import jieba  # 中文分词
from langchain_core.documents import Document  # 统一文档结构
from langchain_community.retrievers import BM25Retriever  # 关键词检索器
from langchain_classic.retrievers import EnsembleRetriever  # RRF 融合
from langchain_classic.retrievers import ContextualCompressionRetriever  # 检索+重排包装器
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker  # 重排压缩器
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # 本地 CrossEncoder

from config import (  # 配置
    RERANK_MODEL, RERANK_DEVICE,
    TOP_K_RECALL, TOP_K_RERANK, SCORE_THRESHOLD,
    MILVUS_COLLECTION, MILVUS_COLLECTIONS,
    get_collection_for_role,
)
from db_milvus import get_vectorstore, get_milvus_client, fetch_all_text
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# 模块级单例（进程内缓存，随服务生命周期存活）
# 之所以放在模块级而不是函数里：FastAPI 的每次请求可能落在不同线程上，
# 模块级的变量天然是"整个进程共享一份"，避免重复加载大模型把内存吃爆
_reranker: Optional[HuggingFaceCrossEncoder] = None  # 重排模型只加载一次（bge-reranker-large 几百 MB，加载很慢）
_bm25_cache: Dict[str, BM25Retriever] = {}  # 集合名 -> BM25 索引（多集合）：每个角色一份，互不干扰


def _chinese_tokenize(text: str) -> List[str]:
    """
    BM25 预分词：jieba 分词 -> 去空白 -> 转小写

    Args:
        text: 一条入库文本（或用户查询），由 BM25Retriever 在内部对语料和查询分别调用
    Returns:
        词列表（已去空白并转小写，便于"AI"/"ai"这类大小写混排也能命中）
    为什么必须传这个函数:
        BM25Retriever 默认按空格切词，中文一整句会被当成一个"词"，导致 BM25 完全失效；
        用 jieba 切成词后，关键词路召回才有意义
    注意:
        建索引和查查询必须用**同一个**分词函数，否则两边的"词"对不上，分数会全为 0
    """
    return [w.strip().lower() for w in jieba.lcut(text) if w.strip()]  # 顺带丢掉纯空白词元（换行/空格）


def get_reranker() -> HuggingFaceCrossEncoder:
    """
    获取重排模型单例（首次调用时加载，之后复用）

    Returns:
        HuggingFaceCrossEncoder 实例（bge-reranker-large，运行在 RERANK_DEVICE 指定的设备上）
    异常:
        配置的模型目录不存在时抛 RuntimeError —— 属于启动配置错误，
        尽早报错比等到第一次提问时才失败更容易定位
    为什么要单例:
        CrossEncoder 加载要读几百 MB 权重，且首次前向还要预热；
        每次检索都重新 new 一个会让首字延迟从毫秒级涨到几十秒
    注意:
        这里是懒加载（第一次真正用到才加载），不是模块导入时就加载，
        这样"只聊天不用知识库"的场景启动会更快
    """
    global _reranker  # 要改写模块级变量，必须声明 global，否则会创建同名局部变量、缓存失效
    if _reranker is None:  # 双检：只有第一次进来才真正加载
        if not os.path.isdir(RERANK_MODEL):
            raise RuntimeError(f"重排模型路径不存在：{RERANK_MODEL}")
        _reranker = HuggingFaceCrossEncoder(
            model_name=RERANK_MODEL,  # 本地权重目录（无需联网下载）
            model_kwargs={"device": RERANK_DEVICE},  # 走 config 配置的设备，项目里是 CPU，避免和向量模型抢显存
        )
        logger.info(f"重排模型已加载：{RERANK_MODEL}")
    return _reranker


def refresh_bm25(collection_name: str = None):
    """
    从指定 Milvus 集合全量拉文本重建 BM25 索引
    入库后调用使新文档立刻可被关键词检索

    Args:
        collection_name: 目标集合名（如 rag_legal / rag_psychology / rag_companion）；
                         传 None 时用默认集合 MILVUS_COLLECTION
    Returns:
        None（结果写进模块级缓存 _bm25_cache，键是集合名）
    异常:
        内部不吞异常：Milvus 连不上、集合不存在等会直接抛给调用方；
        启动期的批量重建请走 refresh_all_bm25（那里逐个 try 住不会互相影响）
    性能取舍:
        BM25 是"全量统计"算法（需要整库文档算词频与平均长度），没有增量更新接口，
        所以每次入库后都要全量重拉一次；文档量大时这步会明显变慢，属于当前实现的固有代价
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    global _bm25_cache  # 要写模块级字典，声明 global（这里其实只做读+改元素，但声明后语义更明确）
    docs = fetch_all_text(collection_name)  # 全量拉取该集合的所有 chunk（只取文本和 metadata）
    if docs:
        _bm25_cache[collection_name] = BM25Retriever.from_documents(
            documents=docs,  # 语料就是 Milvus 里的全部 chunk，保证关键词路和向量路看到的是同一批数据
            k=TOP_K_RECALL,  # 检索时默认返回条数，与向量路保持一致，后续 RRF 融合才公平
            preprocess_func=_chinese_tokenize,  # 中文必须自定义分词，否则 BM25 等于失效（见该函数说明）
        )
        logger.info(f"BM25 索引已重建：集合 {collection_name}，语料 {len(docs)} 条")
    else:
        _bm25_cache.pop(collection_name, None)  # 移除空索引：留着旧索引会检索到已被删除的文档
        logger.warning(f"BM25 索引为空：集合 {collection_name}（无数据）")


def refresh_all_bm25():
    """
    启动时重建所有角色的 BM25 索引

    Returns:
        None；逐个集合重建，任何单个集合失败都不会中断整体流程
    说明:
        遍历 MILVUS_COLLECTIONS（role_key -> 集合名）而不是只处理默认集合，
        因为每个角色都有自己的集合；某个角色还没入库时只打 warning，不影响其它角色可用
    """
    for role_key, collection_name in MILVUS_COLLECTIONS.items():
        try:
            refresh_bm25(collection_name)
        except Exception as e:
            logger.warning(f"刷新 BM25 失败 {collection_name}：{e}")  # 用 warning 而非 raise：启动阶段"能用多少算多少"


def hybrid_search(query: str, top_k: int = TOP_K_RERANK,
                  score_threshold: float = SCORE_THRESHOLD,
                  collection_name: str = None) -> List[Document]:
    """
    混合检索 + 重排 + 分数过滤（指定集合）

    流程：
        1. Milvus 向量检索召回 TOP_K_RECALL 条（Dense）
        2. BM25 关键词检索召回 TOP_K_RECALL 条（Sparse）
        3. EnsembleRetriever 用 RRF 融合两路排名
        4. CrossEncoder 对候选集精排，只留 top_k 条
        5. 余弦相似度低于 score_threshold 的过滤掉

    Args:
        query:           用户提问（原始文本，不做改写；两路检索各自内部处理）
        top_k:           重排后保留条数，默认 TOP_K_RERANK（最终喂给大模型的资料篇数，太多会稀释重点）
        score_threshold: 分数阈值（0~1，配合 Milvus 的 COSINE 度量），低于它的候选直接丢弃
        collection_name: 目标集合名；传 None 用默认集合（多角色场景请由多路召回传角色对应集合）
    Returns:
        List[Document]，按相关度从高到低排序，每条 metadata 里会多出重排模型给的 "score"；
        全部被过滤时返回空列表（不抛异常），由上层决定怎么回答
    异常:
        集合不存在、BM25 索引未初始化时抛 RuntimeError —— 属于"知识库没准备好"，
        与"检索结果不相关（返回空列表）"是两类不同问题，所以这里的处理方式也不一样
    参数取舍:
        top_k 太小会漏掉有用的参考资料，太大则会把无关内容也塞进提示词，
        既浪费 token 又容易让模型被带偏；阈值同理，调高更"宁缺毋滥"（可能直接答不上），调低更"有问必答"（可能答错）
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    # 0. 前置检查：拿到向量存储与原生客户端，先确认集合真的存在
    vectorstore = get_vectorstore(collection_name)  # 指定集合的向量存储（内部复用单例，不会重复连库）
    client = get_milvus_client()  # 原生客户端（用于 has_collection 这类元数据操作）
    if not client.has_collection(collection_name):
        raise RuntimeError(f"知识库集合 {collection_name} 不存在，请先入库")

    # 1. Dense 路：向量检索
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RECALL})  # 粗召回故意放宽，把精排交给重排模型

    # 2. Sparse 路：BM25 关键词检索
    if collection_name not in _bm25_cache:
        raise RuntimeError(f"BM25 索引未初始化：集合 {collection_name}，请先入库或重启服务")  # 不用静默降级成单路：两路都缺一路会让召回质量悄悄变差

    bm25 = _bm25_cache[collection_name]

    # 3. RRF 融合
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],  # 两个检索器，顺序对应下面的权重
        weights=[0.5, 0.5],  # 权重含义：向量路和关键词路各占一半话语权（相加为 1；想更偏语义就调成 [0.7,0.3]）
        c=60,  # RRF 平滑常数：得分 = Σ 1/(c + 排名)，c 越大越"平权"、越小越偏向各路的头部结果；60 是学界经验值
    )

    # 4. CrossEncoder 重排
    # 与向量检索的区别：向量是"查询和文档分别编码再算距离"（快但粗），
    # CrossEncoder 把"查询+文档"拼在一起过一遍模型（慢但准），所以只用它给少量候选排序
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=top_k)  # top_n 就是最终保留条数
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,  # 压缩器 = 重排器：负责给候选打分并截断
        base_retriever=ensemble,  # 基础检索器 = 上面的融合检索，负责产出候选集
    )
    results = final_retriever.invoke(query)  # invoke 一次跑完"融合检索 -> 重排截断"，结果已是排序后的前 top_k

    # 5. 分数过滤
    filtered = []
    for doc in results:
        score = doc.metadata.get("score", 1.0)  # 没拿到分数的（理论上不会）默认按满分处理，宁放行不误杀
        if score >= score_threshold:
            filtered.append(doc)
        else:
            logger.debug(f"过滤低分文档：score={score:.4f} text={doc.page_content[:30]}")  # 用 debug：批次里被过滤的通常很多，info 会刷屏

    if not filtered:
        logger.warning(f"查询 '{query[:30]}' 的结果全被分数过滤掉（集合 {collection_name}）")  # 这条日志是排查"答非所问/答不知道"的关键线索
    return filtered


def multi_route_recall(query: str, user_id: int = 0, top_k: int = TOP_K_RERANK,
                       role_key: str = None) -> List[Document]:
    """
    多路召回：从不同数据源查询数据合并

    召回路：
        1. Milvus 知识库（根据角色路由到对应集合）
        2. Milvus 长期记忆（用户之前的对话要点）

    Args:
        query:    用户提问
        user_id:  用户 ID（业务库自增主键）；大于 0 才会去查长期记忆，
                  传 0 或负数表示"没有具体用户"（如匿名体验），跳过这条召回
        top_k:    知识库那一路最终保留条数，同时决定总返回上限为 top_k*2
        role_key: 角色标识；传了才路由到角色专属集合，不传则用默认集合
    Returns:
        List[Document]，两路结果合并去重后截断到 top_k*2 条
        （截断上限放宽到 2 倍，是因为长期记忆里可能有话题延续的关键信息，
         不该被知识库的结果挤掉；真正的排序仍由各路的分数/顺序决定）
    降级行为:
        任何一路失败都只记日志、不抛异常（知识库失败用 warning、长期记忆失败用 debug），
        保证"记忆服务挂了也能照常基于知识库回答"
    """
    # 根据角色确定要检索的集合
    if role_key:
        collection_name = get_collection_for_role(role_key)  # 每个角色一个集合：法律/心理/陪伴互不污染
    else:
        collection_name = MILVUS_COLLECTION  # 没指定角色时用默认集合

    all_docs: List[Document] = []

    # 路 1：知识库混合检索
    try:
        kb_docs = hybrid_search(query, top_k=top_k, collection_name=collection_name)  # 内部已做融合+重排+阈值过滤
        all_docs.extend(kb_docs)
        logger.info(f"知识库召回 {len(kb_docs)} 条（集合 {collection_name}）")
    except RuntimeError as e:
        logger.warning(f"知识库召回失败：{e}")  # 只捕 RuntimeError：那是"集合不存在/索引没建"这类已知情况

    # 路 2：长期记忆
    if user_id > 0:  # 有具体用户才查记忆（匿名场景没有 user_id，查了也没数据）
        try:
            from db_milvus import search_long_term_memory  # 函数内导入：避免模块级循环依赖，也省一次启动开销
            mem_docs = search_long_term_memory(user_id, query, top_k=3)  # 记忆只取 3 条：它是补充信息，不能喧宾夺主
            all_docs.extend(mem_docs)
            logger.info(f"长期记忆召回 {len(mem_docs)} 条")
        except Exception as e:
            logger.debug(f"长期记忆召回失败：{e}")  # 用 debug：记忆不可用属可接受降级，不该刷 warning 干扰排查

    # 去重
    # 两路可能召回同一段文本（例如用户上次问过同样的问题，答案被存进了长期记忆），
    # 重复资料既浪费 token，也会让模型误以为"这条特别重要"
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc.page_content[:100]  # 用前 100 字做指纹：比整段比对快，且足够区分不同 chunk
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs[:top_k * 2]  # 截断上限（去重后仍可能超量，最终总条数要受控）
