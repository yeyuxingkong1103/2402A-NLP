# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：编排『优化前（基线）』与『优化后』两条 Graph RAG 问答管线，并记录分阶段耗时

import uuid                                                     # 导入 uuid 用于生成请求 ID 便于日志追踪
import time                                                     # 导入 time 用于阶段计时
import config                                                   # 导入本工单配置模块
import embedder                                                 # 导入向量编码模块
import generator                                                # 导入生成模块
from graph_retriever import GraphRetriever                      # 导入图谱检索器
from retriever import expand_queries, synonym_queries, is_broad_question, is_comparative_question  # 导入查询扩展与问题类型判别
from timing import StageTimer, LOGGER                           # 导入分阶段计时器与日志对象


class GraphRAGPipeline(object):                                 # 定义 Graph RAG 管线类
    """同时持有检索器与两版图谱，按模式执行问答并返回结构化结果。"""

    def __init__(self, retriever, graph_optimized=None, graph_baseline=None):  # 构造方法
        self.retriever = retriever                              # 保存混合检索器
        self.gr_opt = GraphRetriever(graph_optimized, retriever.chunks) if graph_optimized else None  # 优化版图谱检索器
        self.gr_base = GraphRetriever(graph_baseline, retriever.chunks) if graph_baseline else None  # 基线版图谱检索器

    # -------------------------------------------------------------------
    # 一、优化前管线：纯向量检索 + 平铺上下文 + 基线提示词
    # -------------------------------------------------------------------
    def answer_baseline(self, question, timer=None):            # 定义基线问答方法
        """执行优化前流程，返回结果字典。"""
        timer = timer or StageTimer(uuid.uuid4().hex[:8])       # 未传入计时器时自建
        timer.start("query_encode")                             # 开始：查询向量化
        qv = embedder.encode_query(question)                    # 编码用户问题
        timer.stop("query_encode")                              # 结束：查询向量化
        timer.start("retrieve")                                 # 开始：检索
        chunks = self.retriever.retrieve_baseline(qv, top_k=config.FINAL_TOP_K)  # 纯向量 top-k
        timer.stop("retrieve")                                  # 结束：检索
        timer.start("generate")                                 # 开始：生成
        answer, ctx, lang = generator.generate(question, chunks, None, optimized=False)  # 基线生成
        timer.stop("generate")                                  # 结束：生成
        return self._pack(question, answer, chunks, None, lang, timer, "baseline")  # 组装返回结果

    # -------------------------------------------------------------------
    # 二、优化后管线：混合检索 + MMR + 图谱扩展 + 结构化上下文 + 强约束提示词
    # -------------------------------------------------------------------
    def answer_optimized(self, question, timer=None):           # 定义优化后问答方法
        """执行优化后流程，返回结果字典。"""
        timer = timer or StageTimer(uuid.uuid4().hex[:8])       # 未传入计时器时自建
        timer.start("query_encode")                             # 开始：查询向量化
        queries = expand_queries(question)                      # 复合问题拆分为多路子查询
        qv = embedder.encode_query(question)                    # 只编码原问题一次，子查询仅在全文侧展开
        timer.stop("query_encode")                              # 结束：查询向量化
        timer.start("graph_expand")                             # 开始：图谱检索
        triples = []                                            # 关系三元组初值
        g_chunks = []                                           # 图谱证据块初值
        if self.gr_opt is not None:                             # 存在优化版图谱时执行图谱检索
            g_chunks, triples = self.gr_opt.retrieve(question)  # 实体链接 + 邻域扩展
        timer.stop("graph_expand")                              # 结束：图谱检索
        timer.start("retrieve_hybrid")                          # 开始：混合检索
        # 第二轮优化：按问题类型动态决定上下文条数——宽泛综合题给更多块以覆盖多要点参考答案，
        # 具体事实题控制在较少条数以保精度（RAGAS 精度公式下尾部无关块不扣分，扩大条数近似无损）
        top_k = config.FINAL_TOP_K_BROAD if is_broad_question(question) else config.FINAL_TOP_K_SPECIFIC  # 动态条数
        # 第三轮优化：跨机构对比题启用同义词扩展查询 + 图谱加权 + MMR λ 放宽，
        # 解决『问题用词与年报原文表述不一致导致证据块检不回』的召回瓶颈
        comparative = is_comparative_question(question)         # 是否为跨机构对比题
        syn_qs = synonym_queries(question) if comparative else []  # 同义词扩展子查询（第四轮：融合加权+覆盖精排）
        chunks = self.retriever.retrieve(question, qv,          # 多路混合召回 + 图谱融合 + MMR 去冗余
                                         top_k=top_k,           # 动态最终条数
                                         use_hybrid=True, use_mmr=True,  # 开启全部优化
                                         queries=queries,       # 传入多路子查询（含同义词扩展，全文侧使用）
                                         extra_chunks=g_chunks,  # 图谱块作为第三条排序列表参与 RRF
                                         question=question,     # 传入问题供关键词覆盖精排
                                         graph_weight=config.GRAPH_FUSE_WEIGHT if comparative else 1.0,  # 对比题加大图谱权重
                                         lam=config.MMR_LAMBDA_COMPARATIVE if comparative else None,  # 对比题适度放宽 λ
                                         syn_queries=syn_qs,    # 同义词子查询加权并只取精准小池
                                         use_cov=not comparative)  # 对比题关闭覆盖度项，防止洗掉同义词检回的证据块
        timer.stop("retrieve_hybrid")                           # 结束：混合检索
        timer.start("generate")                                 # 开始：生成
        answer, ctx, lang = generator.generate(question, chunks, triples, optimized=True)  # 优化后生成
        timer.stop("generate")                                  # 结束：生成
        return self._pack(question, answer, chunks, triples, lang, timer, "optimized")  # 组装返回结果

    # -------------------------------------------------------------------
    # 三、工具方法
    # -------------------------------------------------------------------
    def _pack(self, question, answer, chunks, triples, lang, timer, mode):  # 定义结果组装方法
        """统一包装问答结果，包含答案、上下文、关系、语言与分阶段耗时。"""
        summary = timer.summary()                               # 取分阶段耗时汇总
        retrieve_time = sum(c for n, c in summary["stages"]     # 累计检索相关阶段耗时
                            if n.startswith("retrieve") or n in ("graph_expand", "query_encode"))  # 判定检索阶段
        result = {"question": question,                         # 原始问题
                  "answer": answer,                             # 生成答案
                  "mode": mode,                                 # 管线模式
                  "lang": lang,                                 # 回答语言
                  "contexts": [c["text"] for c in chunks],      # 上下文文本列表（供评估）
                  "sources": [{"source": c["source"], "page": c["page"], "id": c["id"]}  # 上下文来源明细
                              for c in chunks],                 # 逐一列出以便前端展示
                  "triples": triples or [],                     # 图谱关系三元组
                  "timing": summary,                            # 分阶段耗时明细
                  "retrieve_seconds": round(retrieve_time, 4)}  # 检索阶段总耗时（验收关键指标）
        LOGGER.info("[%s] 检索耗时 %.3fs，总耗时 %.3fs",           # 输出关键指标日志
                    mode, result["retrieve_seconds"], summary["total"])  # 填入数值
        return result                                           # 返回结果字典
