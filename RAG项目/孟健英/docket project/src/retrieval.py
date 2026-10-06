# -*- coding: utf-8 -*-
"""混合检索：BM25 + Milvus 双路召回 → RRF 融合 → 返回带元数据的候选。"""
import logging  # 日志：记录角色过滤等检索环节，便于复盘召回质量
from typing import List  # 类型注解：声明本模块函数返回的都是字典列表

from pymilvus import MilvusClient  # Milvus 官方客户端：连接向量数据库做 ANN 相似度检索
from sentence_transformers import SentenceTransformer  # Embedding 模型类：查询向量化必须与入库用同一模型，否则向量空间不匹配

from src import intent, query_tools  # 意图拆路与查询工具：多路召回的口径都在这里定义
from src.bm25_index import BM25Index  # BM25 字面匹配索引：与向量语义检索两路互补，合成混合检索
from src.config import settings  # 统一配置中心：Milvus 地址、集合名、rerank_top_n 都从 .env 读

logger = logging.getLogger(__name__)  # 本模块日志器
_client = None  # 懒加载，避免导入就连 Milvus

def _get_client() -> MilvusClient:  # 获取全局唯一 Milvus 连接（单例：检索高频调用，不能每次重连）
    global _client  # 声明要改的是模块级变量
    if _client is None:  # 首次调用才真正建立连接
        _client = MilvusClient(uri=settings.milvus_uri, timeout=10)  # 按 .env 配置的地址连接；timeout 10s：Milvus 挂了就让用户快速看到错误，别等 30s
    return _client  # 之后的调用都复用同一条连接

def load_records(limit: int = 6000) -> List[dict]:  # 全量拉取知识片段：供 BM25 建索引与引用展示
    """拉取全部知识片段（供 BM25 索引与引用展示使用）。"""
    client = _get_client()  # 复用全局连接
    if not client.has_collection(settings.milvus_collection):  # 集合还没建（未入库）就直接返回空，避免 query 抛错
        return []  # 空知识库 = 无记录
    rows = client.query(  # 标量全量查询：不带向量，纯按主键扫描取数
        settings.milvus_collection, filter="", limit=limit,  # filter="" 不过滤；limit 是内存保护上限
        output_fields=["text", "source", "page"],  # 只取文本与元数据，不取向量本体（省流量）
    )  # query 是标量取数，search 才是向量检索，别混淆
    return [  # 统一字段结构
        {"text": r["text"], "source": r.get("source", "未知来源"), "page": r.get("page", 0)}  # 缺字段给默认值兜底，防脏数据
        for r in rows  # 逐行转换
    ]  # 列表推导一次成型

def _vector_search(query: str, embedder: SentenceTransformer, top_k: int) -> List[dict]:  # 向量召回这一路：查询编码成向量去 Milvus 检索
    """Milvus 向量检索，返回 [{text, source, page, vec}]。"""
    import numpy as np  # 函数内导入：numpy 只在向量化时用，减少模块加载开销
    client = _get_client()  # 复用全局连接

    raw = embedder.encode(query, normalize_embeddings=True)  # 查询向量化：必须与入库同一 Embedding 模型；归一化后内积≈余弦相似度
    qv = np.asarray(raw, dtype=np.float32).reshape(-1).astype(float).tolist()  # 展平成一维并转 Python float 列表，满足类型要求；pymilvus 3.x 的 struct.pack 必须收 Python float，numpy.float32/64 一律不认
    res = client.search(  # ANN 近似最近邻检索：毫秒级返回 Top-K
        settings.milvus_collection,  # 指定知识库集合
        data=[qv], limit=top_k, output_fields=["text", "source", "page"],  # 单查询向量；limit 控制召回条数；顺带取回文本与来源
    )  # 返回按相似度（距离）排序的命中
    hits = []  # 收集解析后的候选
    for hit in res[0]:  # res[0] 是第一个（也是唯一一个）查询向量的命中列表
        entity = hit.get("entity", {})  # entity 里装着 output_fields 指定的字段
        hits.append({  # 组装成统一结构的候选
            "text": entity.get("text", ""),  # 片段原文
            "source": entity.get("source", "未知来源"),  # 来源文档名（引用溯源用）
            "page": int(entity.get("page", 0)),  # 页码（引用定位用）
            "vec": float(hit.get("distance", 0.0)),  # 向量相似度分数：后续预截断与阈值兜底都靠它
        })  # 一条命中组装完毕
    return hits  # 向量路结果

def hybrid_search(  # 混合检索核心：BM25 + 向量双路召回，RRF 融合
    query: str,  # 检索词（可能已被改写）
    bm25: BM25Index,  # 字面匹配索引
    embedder: SentenceTransformer,  # 语义向量模型
    top_k: int | None = None,  # 融合后返回条数
) -> List[dict]:  # 返回带元数据的候选列表
    """双路召回 + RRF 融合，返回候选（含来源、页码、向量分数）。"""
    top_k = top_k or settings.rerank_top_n  # 默认只召回重排所需的条数：粗排够用就行，别让精排打全库
    vec_hits = _vector_search(query, embedder, top_k)  # 路一：向量语义召回（概念相关但未必共享关键词）
    records = bm25.records  # BM25 底层全量记录（融合时的 id 对齐基准）
    if not records:  # BM25 索引为空（知识库未建）时退化为单向量召回
        return vec_hits  # 直接返回向量路结果

    pos = {r["text"]: i for i, r in enumerate(records)}  # 文本→行号映射：把两路命中对齐到同一 id 空间才能融合
    K, rrf, vec_score = 60, {}, {}  # K=60 是 RRF 经典平滑常数；rrf 存融合分；vec_score 顺带保留向量分
    for rank, (idx, _) in enumerate(bm25.search(query, top_k=top_k)):  # 路二：BM25 字面召回（药名/术语精确命中），逐条按排名累加
        rrf[idx] = rrf.get(idx, 0) + 1 / (K + rank + 1)  # RRF 公式：1/(k+rank)，只看排名位置不看原始分数，天然免疫两路分数尺度差异
    for rank, hit in enumerate(vec_hits):  # 向量路同样按排名位置累加 RRF 分
        idx = pos.get(hit["text"])  # 按文本找回行号，与 BM25 路对齐
        if idx is None:  # 向量命中的文本不在 BM25 记录里（理论上不该发生），跳过
            continue  # 无法对齐的命中不参与融合
        rrf[idx] = rrf.get(idx, 0) + 2 / (K + rank + 1)  # 向量路权重更高
        vec_score[idx] = hit["vec"]  # 记录该条向量分，后面预截断和兜底要用

    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]  # 按融合分降序取 Top-K：RRF 不依赖分数尺度、只看排名，工程上最稳
    return [{**records[i], **{"vec": vec_score.get(i, 0.0), "rrf": score}} for i, score in fused]  # 原文/来源/页码与向量分、RRF 分合并成完整候选

def multi_search(  # 多路召回入口：基准问句 + 角色偏向 + 意图拆路，合并去重
    role: dict, query: str, bm25: BM25Index, embedder: SentenceTransformer  # 角色配置决定偏向词与禁用术语
) -> List[dict]:  # 返回去重过滤后的候选
    """基准 + 角色偏向 + 多路口径召回，按文本去重合并。

    用药类问题按五大类降压药（CCB/ACEI/ARB/利尿剂/β受体阻滞剂）拆路检索，
    清单型问题按盐/脂/酒/糖/甘草拆路，避免只召回一两条；最后按角色禁用术语后置。
    """
    routes = [query] + query_tools.role_queries(role, query) + intent.multi_queries(query)  # 拼装检索路线：原问句保底 + 角色增强 + 清单/用药拆路，提高召回率
    merged = []  # 各路结果按文本去重后合并到这里
    for route in routes:  # 逐路执行检索
        for item in hybrid_search(route, bm25, embedder):  # 每路内部仍是 BM25+向量 RRF 融合
            text = item.get("text", "")  # 取片段文本做去重键
            if not text:  # 空文本是脏数据
                continue  # 直接跳过
            match = next((m for m in merged if m["text"] == text), None)  # 同一片段可能被多路重复召回，按文本查重
            if match:  # 已收录过
                match["vec"] = max(match.get("vec", 0.0), item.get("vec", 0.0))  # 保留各路中最高的向量分，利于后续预截断
            else:  # 新片段
                merged.append(item)  # 直接收录
    filtered = query_tools.role_filter(role, merged)  # 角色后置过滤：剔除含禁用术语的片段（如西医角色不展示中医辨证）
    banned = role.get("banned_terms") or []  # 该角色配置的禁用术语表
    if banned and len(filtered) < len(merged):  # 配了禁用词且确实删了片段才记日志
        removed = len(merged) - len(filtered)  # 被移除的条数
        logger.debug("role_filter: %d → %d 条（移除 %d 条含禁用术语）",  # 调试日志：过滤前后条数对比
                     len(merged), len(filtered), removed)  # 日志占位参数
    return filtered  # 过滤后的候选送重排
