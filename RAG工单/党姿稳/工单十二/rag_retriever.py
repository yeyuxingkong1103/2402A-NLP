# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：传统扁平 RAG 检索——向量召回 + BM25 召回 + RRF 融合 + MMR 去冗余（对照组基线）

import numpy as np                                              # 导入 numpy 用于向量相似度计算
import config                                                   # 导入本工单配置模块
import embedder                                                 # 导入嵌入编码模块
import fusion                                                   # 导入 RRF 与 MMR 融合算法
from logger import LOGGER                                       # 导入统一日志对象


class RagIndex(object):                                         # 定义扁平 RAG 索引类
    """扁平 RAG 的检索索引：语料块、向量矩阵与 BM25 倒排索引。"""

    def __init__(self, chunks, vecs, bm25_idx):                 # 构造方法
        self.chunks = chunks                                    # 语料块列表（行号与向量矩阵对齐）
        self.vecs = vecs                                        # 归一化后的语料块向量矩阵
        self.bm25 = bm25_idx                                    # BM25 倒排索引

    def vector_search(self, query_vec, top_k=None):             # 定义向量召回方法
        """按余弦相似度返回 [(行号, 分数), ...] 降序排列。"""
        top_k = top_k or config.VECTOR_TOP_K                    # 取配置的召回数量
        scores = self.vecs.dot(query_vec)                       # 归一化向量点积即余弦相似度
        order = np.argsort(-scores)[:top_k]                     # 取相似度最高的若干行号
        return [(int(i), float(scores[int(i)])) for i in order]  # 返回行号与分数

    def retrieve(self, model, question, top_k=None):            # 定义检索主方法
        """执行扁平 RAG 检索，返回排序后的结果列表（含每个候选的来源与得分）。"""
        top_k = top_k or config.FINAL_TOP_K                     # 最终返回的上下文块数量
        qv = embedder.encode(model, [question], is_query=True)[0]  # 编码查询（加官方前缀）
        vec_hits = self.vector_search(qv)                       # 第一路：向量召回
        bm25_hits = self.bm25.search(question, top_k=config.BM25_TOP_K)  # 第二路：BM25 全文召回
        vec_rank = [i for i, _ in vec_hits]                     # 向量侧排序（仅行号）
        bm25_rank = [i for i, _ in bm25_hits]                   # BM25 侧排序（仅行号）
        fused, scores = fusion.rrf_fuse([vec_rank, bm25_rank])  # 两路 RRF 融合
        selected = fusion.mmr_dedup(fused, scores, self.vecs, top_k)  # MMR 去冗余并截断
        fusion.retrieve_summary("扁平RAG", len(fused), len(selected))  # 输出候选数量变化
        results = []                                            # 初始化结果列表
        for rank, i in enumerate(selected):                     # 遍历最终选中的块
            c = self.chunks[i]                                  # 取出语料块
            results.append({"rank": rank,                        # 在结果中的名次（从 0 开始）
                            "idx": i,                            # 语料块行号（与向量矩阵对齐）
                            "id": c["id"],                       # 语料块编号
                            "source": c["source"],               # 来源说明书
                            "page": c["page"],                   # 来源页码
                            "text": c["text"],                   # 块正文
                            "score": round(scores.get(i, 0.0), 6),  # RRF 融合得分
                            "via": "vector+bm25"})               # 召回来源标记
        LOGGER.info("扁平RAG 检索完成：返回 %d 块", len(results))  # 输出检索结果规模
        return results                                          # 返回结果列表


def build_index(chunks, model, force=False):                    # 定义索引构建函数
    """构建扁平 RAG 索引：编码语料块并建立 BM25 索引（均带磁盘缓存）。"""
    import bm25                                                 # 延迟导入避免循环依赖
    vecs = embedder.encode_chunks(model, chunks, config.EMBED_CACHE_FILE, force=force)  # 编码或读取向量
    idx = bm25.build_or_load_index(chunks, force=force)         # 构建或读取 BM25 索引
    return RagIndex(chunks, vecs, idx)                          # 组装成索引对象返回
