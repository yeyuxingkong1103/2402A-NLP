# -*- coding: utf-8 -*-
"""检索模块：Milvus 混合检索 + MySQL 多路召回 + BGE-reranker 精排，第 4 步不含生成。"""

import argparse                               # 导入 argparse，用于解析命令行参数
import json                                   # 导入 json，用于读取父块文件
import time                                   # 导入 time，用于统计耗时
from pathlib import Path                      # 导入 Path，用于定位数据文件
from typing import List                       # 导入 List，用于声明列表类型

import pymysql                                # 导入 pymysql，用于查询 MySQL 历史对话
import config                                 # 导入配置模块，权重、阈值、条数都从这里读
import vector_store                           # 导入向量库模块，复用其中的 Milvus 客户端与向量化
from logger import get_logger                 # 导入日志工具，用于记录检索过程

logger = get_logger("retrieval")              # 创建本模块的 logger 实例

DENSE_FIELD = "vector"                        # 稠密向量字段名
SPARSE_FIELD = "sparse_vector"                # 稀疏向量字段名
OUTPUT_FIELDS = ["text", "source", "summary", "parent_id"]   # 检索时需要取回的标量字段
PARENT_JSON = Path(__file__).resolve().parent / "data" / "enriched_chunks.json"   # 父块来源文件
# PARENT_JSON 保存chunk父子关系：大文档切分成父块，再细切成子块（检索用子块，拿到子块后补父块全文）
_RERANKER = None                              # 重排模型单例缓存，避免重复加载
_PARENT_CACHE = None                          # 父块内存缓存，首次使用时加载一次
# _PARENT_CACHE 全局单例，只加载一次，常驻内存，避免反复读json文件IO

def _load_parent_cache() -> dict:              # 内部函数：加载父块缓存
    """加载父子块文件，返回 {父块索引: 父块文本}，只加载一次并常驻内存。"""
    global _PARENT_CACHE                       # 声明使用模块级缓存变量
    if _PARENT_CACHE is not None:              # 已经加载过
        return _PARENT_CACHE                   # 直接返回缓存
    mapping = {}                               # 父块索引到文本的映射
    try:                                       # 文件缺失或损坏时降级为空映射
        payload = json.loads(PARENT_JSON.read_text(encoding="utf-8"))   # 读取增强结果
        for index, item in enumerate(payload.get("chunks", [])):        # 逐条遍历
            if item.get("is_parent"):          # 只收父块
                mapping[index] = item.get("text", "")   # 记录索引到文本
        logger.info("父块缓存已加载：%d 个父块", len(mapping))   # 记录日志
    except Exception as exc:                   # 读取失败
        logger.warning("父块缓存加载失败，回溯功能降级：%s", exc)   # 记录降级原因
    _PARENT_CACHE = mapping                    # 写入缓存
    return mapping                             # 返回映射


def attach_parent_text(results: List[dict]) -> List[dict]:
    # attach_parent_text：检索命中子块后，根据parent_id去缓存拿到父块文本，放进extra.parent_text
    """给命中子块的结果附加父块全文，写到 extra.parent_text 字段；无父块则留空。"""
    cache = _load_parent_cache()               # 取得父块映射
    if not cache:                              # 没有父块可用
        return results                         # 原样返回
    for item in results:                       # 逐条处理
        parent_id = int(item.get("parent_id", -1))   # 取出父块索引
        if parent_id >= 0 and parent_id in cache:    # 有父块且能找到
            item.setdefault("extra", {})["parent_text"] = cache[parent_id]   # 附加父块全文
    return results                             # 返回结果
# chunk 切片策略是父子块。检索粒度用小的子块提升召回精度；拿到结果后回填父块，给 LLM 完整上下文，避免切片丢失上下文

# ===================== 一、Milvus 混合检索 =====================

def _search_one_field(client, query_vec, field: str, limit: int, metric: str) -> List[dict]:   # 单字段检索
    # _search_one_field：通用Milvus检索函数，可以指定向量字段、度量方式
    """对指定向量字段做一次检索，返回带名次的候选列表（内部函数）。"""
    outcome = client.search(                   # 调用 Milvus 检索
        collection_name=config.MILVUS_COLLECTION,   # 集合名称
        data=[query_vec],                      # 查询向量，单条
        anns_field=field,                      # 指定检索哪个向量字段
        search_params={"metric_type": metric, "params": {}},   # 度量方式
        limit=limit,                           # 返回条数
        output_fields=OUTPUT_FIELDS,           # 需要取回的字段
    )                                          # 检索调用结束
    hits = outcome[0] if outcome else []       # 取出第一条查询的结果
    results = []                               # 保存整理后的候选
    for rank, hit in enumerate(hits):          # 按名次遍历（rank 从 0 开始）
        entity = hit.get("entity", {})         # 取出实体字段
        results.append({                       # 组装候选结构
            "id": hit.get("id"),               # 主键
            "text": entity.get("text", ""),    # 原文
            "source": entity.get("source", ""),   # 来源文件名
            "summary": entity.get("summary", ""), # 摘要
            "parent_id": int(entity.get("parent_id", -1)),   # 所属父块索引，供回溯上下文
            "rank": rank,                      # 本路中的名次，供 RRF 使用
            "raw_score": hit.get("distance", 0.0),   # 本路的原始相似度分
        })                                     # 候选组装结束
    return results                             # 返回本路候选


def rrf_fuse(dense_hits: List[dict], sparse_hits: List[dict], k: int = None) -> List[dict]:   # RRF 融合
    """用 RRF 公式融合 dense 与 sparse 两路结果：score = Σ weight/(k + rank)。"""
    k_value = config.RRF_K if k is None else k   # 未指定时从配置读 k 值
    fused = {}                                 # 用主键聚合两路结果
    for hits, tag in ((dense_hits, "dense"), (sparse_hits, "sparse")):   # 依次处理两路
        for hit in hits:                       # 遍历该路候选
            key = hit["id"]                    # 以主键作为聚合键
            if key not in fused:               # 第一次见到这个主键
                fused[key] = {**hit, "rrf_score": 0.0, "from": []}   # 初始化融合记录
            fused[key]["rrf_score"] += 1.0 / (k_value + hit["rank"] + 1)   # 累加 RRF 分数
            fused[key]["from"].append(tag)     # 记录命中了哪一路
    merged = list(fused.values())              # 转成列表
    merged.sort(key=lambda item: item["rrf_score"], reverse=True)   # 按 RRF 分数降序
    for item in merged:                        # 逐条整理输出
        item["score"] = item["rrf_score"]      # 对外统一用 score 字段
        item["from"] = "+".join(item["from"])  # 命中的路拼成字符串
    return merged                              # 返回融合结果


# 注意：此处的权重只影响 RRF 融合阶段的排序，最终排序由 rerank 决定。
# MySQL 路降权后，若它仍能进入精排，说明它和 query 的相关度确实高。
# 如果发现 MySQL 路仍压过知识条款，需要在 rerank 前过滤掉 source=mysql 的结果。
def fuse_multi_route(routes: List[tuple]) -> List[dict]:   # 多路加权融合
    """按路加权融合多路召回结果：每路给 (候选列表, 该路权重, 路名)。"""
    fused = {}                                 # 用主键聚合各路结果
    for hits, weight, tag in routes:           # 依次处理每一路
        for hit in hits:                       # 遍历该路候选
            key = (tag, hit.get("id"))         # MySQL 路与向量路的主键可能重号，用「路名+主键」区分
            if key not in fused:               # 第一次见到
                fused[key] = {**hit, "fused_score": 0.0, "from": []}   # 初始化记录
            rank = hit.get("rank", 0)          # 该候选在本路中的名次
            fused[key]["fused_score"] += weight / (config.RRF_K + rank + 1)   # 加权累加 RRF 分数
            fused[key]["from"].append(tag)     # 记录命中的路
    merged = list(fused.values())              # 转成列表
    merged.sort(key=lambda item: item["fused_score"], reverse=True)   # 按融合分数降序
    for item in merged:                        # 逐条整理
        item["score"] = round(item["fused_score"], 6)   # 对外统一用 score
        item["from"] = "+".join(item["from"])  # 命中的路拼成字符串
    return merged                              # 返回融合结果


def hybrid_search_milvus(query: str, top_k: int = None) -> List[dict]:
    """Milvus 混合检索：dense 与 sparse 各查一次，再用 RRF 融合，返回候选列表。"""
    limit = top_k or config.HYBRID_TOP_K       # 未指定条数时从配置读
    dense_vec, sparse_vec = vector_store.embed_query(query)   # 生成查询的两种向量
    client = vector_store.get_milvus_client()  # 获取 Milvus 客户端
    client.load_collection(config.MILVUS_COLLECTION)   # 加载集合到内存
    each = limit * 2                           # 每路多取一倍候选，供融合后仍有足够结果
    dense_hits = _search_one_field(client, dense_vec, DENSE_FIELD, each, "COSINE")   # 稠密路
    sparse_hits = _search_one_field(client, sparse_vec, SPARSE_FIELD, each, "IP")    # 稀疏路
    logger.info("混合检索：dense 召回 %d 条，sparse 召回 %d 条",
                len(dense_hits), len(sparse_hits))   # 记录两路召回条数
    fused = rrf_fuse(dense_hits, sparse_hits)  # RRF 融合
    return fused[:limit]                       # 截取前 top_k 条返回
# 两路并行检索：Dense 稠密向量：`COSINE`余弦相似度，捕捉语义含义（近义词、同义改写）
# Sparse 稀疏向量：`IP`内积，捕捉关键词、术语字面匹配（专业名词、专有词）
# 为什么两路？稠密抓语义，稀疏抓关键词，互补，解决单独稠密丢关键词、单独稀疏不理解语义的问题

# ===================== 二、MySQL 多路召回 =====================

def search_mysql(query: str, top_k: int = None) -> List[dict]:
    """从 MySQL 的 messages 表模糊匹配历史提问，连不上时返回空列表，不抛异常。"""
    limit = top_k or config.MYSQL_RECALL_TOP_K   # 未指定条数时从配置读
    try:                                       # 数据库异常一律吞掉，不影响主流程
        import db                              # 延迟导入，复用已有的连接封装
        conn = db.get_mysql_conn()             # 获取 MySQL 连接
    except Exception as exc:                   # 连接失败
        logger.warning("MySQL 召回跳过（连不上）：%s", exc)   # 记录告警
        return []                              # 返回空列表
    try:                                       # 查询过程同样做保护
        with conn.cursor() as cursor:          # 打开游标
            cursor.execute(                    # 模糊匹配用户历史提问
                "SELECT id, content FROM messages WHERE role = %s AND content LIKE %s "
                "ORDER BY id DESC LIMIT %s",   # 只查用户提问，按时间倒序
                ("user", f"%{query}%", limit),  # 参数化传入，避免拼接 SQL
            )                                  # 查询执行结束
            rows = cursor.fetchall()           # 取出结果
    except Exception as exc:                   # 查询失败
        logger.warning("MySQL 召回失败：%s", exc)   # 记录告警
        return []                              # 返回空列表
    finally:                                   # 无论如何都要关闭连接
        conn.close()                           # 关闭连接
    results = []                               # 保存召回结果
    for rank, row in enumerate(rows):          # 按名次遍历
        results.append({                       # 组装候选结构
            "id": row.get("id"),               # 消息编号
            "text": row.get("content", ""),    # 历史提问内容
            "source": "mysql",                 # 标记来源为 MySQL 路
            "summary": "",                     # 历史对话没有摘要
            "summary": "",                     # 历史对话没有摘要
            "parent_id": -1,                   # 历史对话没有父块
            "rank": rank,                      # 本路名次，供加权融合使用
        })                                     # 候选组装结束
    logger.info("MySQL 召回 %d 条", len(results))   # 记录召回条数
    return results                             # 返回结果
# 从`messages`表，`role='user'`，**LIKE 模糊匹配用户历史提问**，拿到相似历史问题，作为额外召回源

# ===================== 三、多路召回 =====================

def multi_recall(query: str, top_k: int = 30) -> List[dict]:
    """多路召回：Milvus 混合检索 + MySQL 历史对话加权融合，按 text 去重后返回。"""
    milvus_hits = hybrid_search_milvus(query, config.HYBRID_TOP_K)   # 第一路：Milvus
    for index, hit in enumerate(milvus_hits):  # 给 Milvus 路补上名次，供融合使用
        hit["rank"] = index                    # 名次从 0 开始
    mysql_hits = search_mysql(query, config.MYSQL_RECALL_TOP_K)      # 第二路：MySQL
    fused = fuse_multi_route([                 # 按权重融合两路
        (milvus_hits, 1.0, "milvus"),          # 知识路权重 1.0
        (mysql_hits, config.MYSQL_RECALL_WEIGHT, "mysql"),   # 历史对话路降权
    ])                                         # 融合结束
    # **加权 RRF**：和上面普通 RRF 不一样，每一路可以单独设置权重
    merged = []                                # 保存去重后的结果
    seen = set()                               # 记录已出现过的文本，用于去重
    for item in fused:                         # 已按加权分数降序，先出现的分更高
        text = (item.get("text") or "").strip()   # 取出文本并去空白
        if not text or text in seen:           # 空文本或已出现过
            continue                           # 跳过
        seen.add(text)                         # 记录该文本
        merged.append(item)                    # 收进结果
    merged = merged[:top_k]                    # 截取前 top_k 条
    logger.info("多路召回：Milvus %d 条 + MySQL %d 条（权重 %.2f）→ 去重后 %d 条",
                len(milvus_hits), len(mysql_hits), config.MYSQL_RECALL_WEIGHT, len(merged))   # 日志
    return merged                              # 返回合并结果


# ===================== 四、重排 =====================

def load_reranker():                           # 加载重排模型
    # load_reranker：懒加载单例，全局只加载一次模型，不重复加载占用显存
    """加载本地 BGE-reranker-v2-m3 重排模型，使用懒加载单例，避免重复占显存。"""
    global _RERANKER                           # 声明使用模块级单例变量
    if _RERANKER is not None:                  # 已经加载过
        return _RERANKER                       # 直接返回缓存
    from FlagEmbedding import FlagReranker     # 延迟导入，避免未安装时影响模块加载
    model_path = config.RERANKER_PATH          # 从配置读取模型本地路径
    if not model_path:                         # 路径为空
        raise RuntimeError("RERANKER_PATH 未配置")   # 立即报错
    device_ok = vector_store._cuda_available()  # 复用向量库模块的显卡探测
    logger.info("开始加载 Reranker：%s", model_path)   # 记录日志
    start = time.time()                        # 记录加载开始时间
    _RERANKER = FlagReranker(model_path, use_fp16=device_ok)   # 创建重排模型实例
    logger.info("Reranker 加载完成，耗时 %.1f 秒", time.time() - start)   # 记录耗时
    return _RERANKER                           # 返回模型实例


def rerank(query: str, docs: List[dict], top_k: int = None) -> List[dict]:
    """用 BGE-reranker-v2-m3 精排候选，按 sigmoid 分数降序取 top_k 并过滤低分。"""
    limit = top_k or config.RERANK_TOP_K       # 未指定条数时从配置读
    if not docs:                               # 候选为空
        return []                              # 直接返回空列表
    model = load_reranker()                    # 获取重排模型单例
    pairs = [[query, doc.get("text", "")] for doc in docs]   # 组装成 query-文档 对
    scores = model.compute_score(pairs, normalize=True)      # 计算分数，normalize 即 sigmoid 转 0~1
    # model.compute_score(normalize=True) → sigmoid归一到0~1之间的相关性分数
    if not isinstance(scores, list):           # 只有一条时返回的是标量
        scores = [scores]                      # 统一包成列表
    ranked = []                                # 保存带分数的候选
    for doc, score in zip(docs, scores):       # 逐条绑定分数
        item = dict(doc)                       # 复制一份，避免改动原对象
        item["rerank_score"] = float(score)    # 写入精排分数（0~1）
        item["score"] = float(score)           # 对外统一用 score
        ranked.append(item)                    # 收进结果
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)   # 按分数降序
    kept = [item for item in ranked if item["rerank_score"] >= config.RERANK_SCORE_THRESHOLD]   # 过滤低分
    logger.info("精排：输入 %d 条 → 过滤低于 %.2f 的 %d 条 → 保留 %d 条",
                len(ranked), config.RERANK_SCORE_THRESHOLD, len(ranked) - len(kept), len(kept))   # 记录日志
    return kept[:limit]                        # 截取前 top_k 条返回


# ===================== 五、检索总入口 =====================

def retrieve(query: str, top_k: int = None) -> List[dict]:
    """检索总入口：多路召回后精排，返回最终候选，并记录各阶段条数。"""
    limit = top_k or config.RERANK_TOP_K       # 未指定条数时从配置读
    start = time.time()                        # 记录总耗时起点
    candidates = multi_recall(query)           # 第一步：多路召回
    final = rerank(query, candidates, limit)   # 第二步：精排并过滤
    final = attach_parent_text(final)          # 第三步：给命中子块附加父块全文
    logger.info("检索完成：召回 %d 条 → 精排保留 %d 条，耗时 %.2f 秒",
                len(candidates), len(final), time.time() - start)   # 记录整体情况
    return final                               # 返回最终结果


# ===================== 六、命令行入口 =====================

def main() -> None:                            # 命令行入口
    """命令行入口：python -m retrieval "问题"，打印精排结果的分数、来源与文本片段。"""
    parser = argparse.ArgumentParser(description="Milvus 混合检索 + 多路召回 + 精排")   # 创建解析器
    parser.add_argument("query", help="要检索的问题")   # 位置参数：查询问题
    parser.add_argument("--top_k", type=int, default=None, help="返回条数")   # 可选条数
    args = parser.parse_args()                 # 解析命令行参数
    start = time.time()                        # 记录耗时起点
    results = retrieve(args.query, args.top_k)  # 执行检索
    print("=" * 78)                            # 打印分隔线
    print(f"查询：{args.query}")                # 打印查询内容
    print(f"Milvus 版本：{_milvus_version()}")  # 打印 Milvus 版本
    print("稀疏方案：BGE-m3 lexical_weights（学习式稀疏检索）")   # 打印稀疏方案
    print("-" * 78)                            # 打印分隔线
    if not results:                            # 没有结果
        print("没有检索到任何结果")              # 打印提示
    for idx, item in enumerate(results, 1):    # 逐条打印
        text = (item.get("text") or "")[:100]  # 文本截断到 100 字
        print(f"{idx}. score={item.get('score', 0):.4f} | 来源={item.get('source', '')[:40]}")   # 分数与来源
        print(f"   {text}")                    # 打印文本片段
    print("-" * 78)                            # 打印分隔线
    print(f"返回 {len(results)} 条，总耗时 {time.time() - start:.2f} 秒")   # 打印汇总
    print("=" * 78)                            # 打印分隔线


def _milvus_version() -> str:                  # 内部函数：取 Milvus 版本
    """取 Milvus 服务端版本号，取不到时返回 unknown。"""
    try:                                       # 版本接口失败不影响主流程
        from pymilvus import connections, utility   # 导入连接池与工具类
        for alias, handler in connections.list_connections():   # 遍历已注册的连接
            if handler is not None:            # 只挑真正建立起来的连接
                return str(utility.get_server_version(using=alias))   # 用该别名取版本
    except Exception:                          # 取版本失败
        pass                                   # 忽略，继续返回占位值
    return "unknown"                           # 返回占位字符串


if __name__ == "__main__":                     # 支持 python -m retrieval 直接运行
    main()                                     # 执行命令行入口
