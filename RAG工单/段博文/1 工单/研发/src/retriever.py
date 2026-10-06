# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
检索模块（简化版，单集合）：向量检索 + BM25 关键词检索 -> RRF 融合 ->
CrossEncoder 重排 -> 分数过滤。

在系统中的位置（本模块是 RAG 链路的核心）：
    上游是问答接口——它拿到用户提问后调用 hybrid_search 取资料；
    下游依赖 db_milvus——向量检索、集合存在性检查、BM25 语料全量拉取
    都走它；产出的 Document 列表交给 llm_client 拼成提示词。

职责：
    「召回 + 精排 + 过滤」三件事。先宽召回保证不漏（向量 + 关键词两路
    各 TOP_K_RECALL 条），再用 CrossEncoder 精排保证准（只留 TOP_K_RERANK
    条），最后用分数阈值滤掉明显不相关的。

关键设计取舍：
    1. 为什么要向量 + BM25 两路？向量擅长语义相近（"撞人赔偿"能命中
       "交通事故责任"），但容易漏掉精确的法律术语/案号；BM25 正好相反。
       两路互补再用 RRF 融合排名。
    2. 为什么用 RRF 而不是加权求和分数？因为两路的分数不可比（余弦
       相似度 vs BM25 分值），RRF 只用「排名」不用「分数」，天然不需要
       归一化，也更鲁棒。
    3. 重排模型和 BM25 索引都做模块级缓存：加载一次反复用，否则每次
       提问都要重新加载几百 MB 模型。
"""

import os  # 路径操作（校验重排模型目录是否存在）
from typing import List, Optional  # 类型标注

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
)
from db_milvus import get_vectorstore, get_milvus_client, fetch_all_text
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# 模块级单例（进程内缓存，随服务生命周期存活）
# 之所以放在模块级而不是函数里：FastAPI 的每次请求可能落在不同线程上，
# 模块级的变量天然是「整个进程共享一份」，避免重复加载大模型把内存吃爆
_reranker: Optional[HuggingFaceCrossEncoder] = None  # 重排模型只加载一次（bge-reranker-large 几百 MB，加载很慢）
_bm25: Optional[BM25Retriever] = None  # BM25 索引单例（单集合，直接用全局变量）


def _chinese_tokenize(text: str) -> List[str]:
    """
    BM25 预分词：jieba 分词 -> 去空白 -> 转小写

    参数：
        text：一条入库文本（或用户查询），由 BM25Retriever 在内部对语料
              和查询分别调用。
    返回：
        词列表（已去空白并转小写，便于 "AI"/"ai" 这类大小写混排也能命中）。
    为什么必须传这个函数：
        BM25Retriever 默认按空格切词，中文一整句会被当成一个「词」，导致
        BM25 完全失效；用 jieba 切成词后，关键词路召回才有意义。
    注意：
        建索引和查查询必须用**同一个**分词函数，否则两边的「词」对不上，
        分数会全为 0。
    """
    return [w.strip().lower() for w in jieba.lcut(text) if w.strip()]  # 顺带丢掉纯空白词元


def get_reranker() -> HuggingFaceCrossEncoder:
    """
    获取重排模型单例（首次调用时加载，之后复用）

    返回：
        HuggingFaceCrossEncoder 实例（bge-reranker-large，运行在
        RERANK_DEVICE 指定的设备上）。
    异常：
        配置的模型目录不存在时抛 RuntimeError —— 属于启动配置错误，
        尽早报错比等到第一次提问时才失败更容易定位。
    为什么要单例：
        CrossEncoder 加载要读几百 MB 权重，且首次前向还要预热；每次检索
        都重新 new 一个会让首字延迟从毫秒级涨到几十秒。
    注意：
        这里是懒加载（第一次真正用到才加载），不是模块导入时就加载，
        这样「只做检索不调 LLM」的场景启动会更快。
    """
    global _reranker  # 要改写模块级变量，必须声明 global
    if _reranker is None:  # 双检：只有第一次进来才真正加载
        if not os.path.isdir(RERANK_MODEL):
            raise RuntimeError(f"重排模型路径不存在：{RERANK_MODEL}")
        _reranker = HuggingFaceCrossEncoder(
            model_name=RERANK_MODEL,  # 本地权重目录（无需联网下载）
            model_kwargs={"device": RERANK_DEVICE},  # 走 config 配置的设备，项目里是 CPU，避免和向量模型抢显存
        )
        logger.info(f"重排模型已加载：{RERANK_MODEL}")
    return _reranker


def refresh_bm25() -> None:
    """
    从 Milvus 全量拉文本重建 BM25 索引；入库后调用使新文档立刻可被关键词检索

    返回：
        None（结果写进模块级变量 _bm25）。
    异常：
        内部不吞异常：Milvus 连不上等会直接抛给调用方；启动期调用时
        由调用方自行 try 住。
    性能取舍：
        BM25 是「全量统计」算法（需要整库文档算词频与平均长度），没有
        增量更新接口，所以每次入库后都要全量重拉一次；文档量大时这步
        会明显变慢，属于当前实现的固有代价。
    """
    global _bm25
    texts = fetch_all_text()  # 全量拉取集合所有 chunk 的纯文本
    if texts:
        # BM25Retriever 接收 Document 列表；这里把每条文本包成一个 Document
        docs = [Document(page_content=t) for t in texts if t and t.strip()]
        _bm25 = BM25Retriever.from_documents(
            documents=docs,  # 语料就是 Milvus 里的全部 chunk
            k=TOP_K_RECALL,  # 检索时默认返回条数，与向量路保持一致，后续 RRF 融合才公平
            preprocess_func=_chinese_tokenize,  # 中文必须自定义分词，否则 BM25 等于失效
        )
        logger.info(f"BM25 索引已重建：语料 {len(docs)} 条")
    else:
        _bm25 = None  # 移除空索引：留着旧索引会检索到已被删除的文档
        logger.warning("BM25 索引为空（集合无数据）")


def hybrid_search(query: str, top_k: int = TOP_K_RERANK,
                  score_threshold: float = SCORE_THRESHOLD) -> List[Document]:
    """
    混合检索 + 重排 + 分数过滤

    流程：
        1. Milvus 向量检索召回 TOP_K_RECALL 条（Dense）
        2. BM25 关键词检索召回 TOP_K_RECALL 条（Sparse）
        3. EnsembleRetriever 用 RRF 融合两路排名
        4. CrossEncoder 对候选集精排，只留 top_k 条
        5. 余弦相似度低于 score_threshold 的过滤掉

    参数：
        query：           用户提问（原始文本，不做改写；两路检索各自内部处理）。
        top_k：           重排后保留条数，默认 TOP_K_RERANK（最终喂给大模型
                          的资料篇数，太多会稀释重点）。
        score_threshold： 分数阈值（0~1，配合 Milvus 的 COSINE 度量），低于
                          它的候选直接丢弃。
    返回：
        List[Document]，按相关度从高到低排序，每条 metadata 里会多出重排
        模型给的 "score"；全部被过滤时返回空列表（不抛异常），由上层决定
        怎么回答。
    异常：
        集合不存在、BM25 索引未初始化时抛 RuntimeError —— 属于「知识库
        没准备好」，与「检索结果不相关（返回空列表）」是两类不同问题，
        所以这里的处理方式也不一样。
    参数取舍：
        top_k 太小会漏掉有用的参考资料，太大则会把无关内容也塞进提示词，
        既浪费 token 又容易让模型被带偏；阈值同理，调高更「宁缺毋滥」
        （可能直接答不上），调低更「有问必答」（可能答错）。
    """
    # 0. 前置检查：确认集合真的存在
    client = get_milvus_client()
    from config import MILVUS_COLLECTION  # 函数内导入避免循环依赖
    if not client.has_collection(MILVUS_COLLECTION):
        raise RuntimeError(f"知识库集合 {MILVUS_COLLECTION} 不存在，请先入库")

    # 1. Dense 路：向量检索
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RECALL})  # 粗召回故意放宽，把精排交给重排模型

    # 2. Sparse 路：BM25 关键词检索
    if _bm25 is None:
        raise RuntimeError("BM25 索引未初始化，请先入库或重启服务")  # 不用静默降级成单路：两路都缺一路会让召回质量悄悄变差
    bm25 = _bm25

    # 3. RRF 融合
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],  # 两个检索器，顺序对应下面的权重
        weights=[0.5, 0.5],  # 权重含义：向量路和关键词路各占一半话语权（相加为 1；想更偏语义就调成 [0.7,0.3]）
        c=60,  # RRF 平滑常数：得分 = Σ 1/(c + 排名)，c 越大越「平权」、越小越偏向各路的头部结果；60 是学界经验值
    )

    # 4. CrossEncoder 重排
    # 与向量检索的区别：向量是「查询和文档分别编码再算距离」（快但粗），
    # CrossEncoder 把「查询+文档」拼在一起过一遍模型（慢但准），所以只用它给少量候选排序
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=top_k)  # top_n 就是最终保留条数
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,  # 压缩器 = 重排器：负责给候选打分并截断
        base_retriever=ensemble,  # 基础检索器 = 上面的融合检索，负责产出候选集
    )
    results = final_retriever.invoke(query)  # invoke 一次跑完「融合检索 -> 重排截断」

    # 5. 分数过滤
    filtered = []
    for doc in results:
        score = doc.metadata.get("score", 1.0)  # 没拿到分数的（理论上不会）默认按满分处理，宁放行不误杀
        if score >= score_threshold:
            filtered.append(doc)
        else:
            logger.debug(f"过滤低分文档：score={score:.4f} text={doc.page_content[:30]}")  # 用 debug：批次里被过滤的通常很多，info 会刷屏

    if not filtered:
        logger.warning(f"查询 '{query[:30]}' 的结果全被分数过滤掉")
    return filtered
