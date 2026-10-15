# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：问答服务编排——索引构建/缓存、检索、生成与问答历史持久化（验证数据卷）

import os                                                       # 导入 os 用于文件读写与路径处理
import json                                                     # 导入 json 用于索引与历史序列化
import time                                                     # 导入 time 用于耗时统计
import pickle                                                   # 导入 pickle 用于 BM25 索引序列化
import numpy as np                                              # 导入 numpy 用于向量矩阵存取
import config                                                   # 导入本工单配置模块
import dataset                                                  # 导入语料与分块模块
import embedder                                                 # 导入向量化模块
import llm_client                                               # 导入大模型调用模块
from retriever import Retriever, BM25                           # 导入检索器与 BM25
from logging_setup import LOGGER                                # 导入统一日志对象


class QAService(object):                                        # 定义问答服务类
    """对外提供问答能力；内部持有检索器与全部文本块，索引结果缓存在数据卷。"""

    def __init__(self):                                         # 构造方法
        self.chunks = []                                        # 文本块列表
        self.retriever = None                                   # 检索器（延迟构建）
        self.ready = False                                      # 就绪标记
        self.build_seconds = 0.0                                # 索引构建耗时

    # -------------------------------------------------------------------
    # 一、索引构建与缓存
    # -------------------------------------------------------------------
    def build(self):                                            # 定义索引构建方法
        """构建或从数据卷加载索引；重复调用幂等。"""
        if self.ready:                                          # 已就绪则直接返回
            return self                                         # 返回自身
        t0 = time.time()                                        # 记录起始时间
        config.ensure_dirs()                                    # 确保数据目录存在
        self.chunks = self._load_chunks()                       # 加载或生成分块
        bm25 = self._load_bm25()                                # 加载或构建 BM25 索引
        vectors = self._load_vectors()                          # 加载或构建向量矩阵
        self.retriever = Retriever(self.chunks, vectors, bm25=bm25)  # 组装检索器
        self.build_seconds = round(time.time() - t0, 2)         # 记录构建耗时
        self.ready = True                                       # 标记就绪
        LOGGER.info("索引就绪：%d 块，向量检索 %s，耗时 %.2fs",    # 输出就绪日志
                    len(self.chunks), "开启" if self.retriever.embed_enabled else "关闭",
                    self.build_seconds)                         # 填入统计信息
        return self                                             # 返回自身

    def _load_chunks(self):                                     # 定义分块加载函数
        """优先读取索引缓存，缺失时解析语料并落盘。"""
        cache = os.path.join(config.DATA_DIR, "chunks.json")    # 分块缓存路径
        if os.path.isfile(cache):                               # 缓存存在
            try:                                                # 尝试读取
                with open(cache, "r", encoding="utf-8") as f:   # 以 UTF-8 打开
                    chunks = json.load(f)                       # 反序列化
                LOGGER.info("命中分块缓存：%d 块", len(chunks))  # 输出日志
                return chunks                                   # 返回缓存结果
            except Exception as e:                              # 缓存损坏
                LOGGER.warning("分块缓存损坏，将重建：%r", e)     # 输出告警
        docs = dataset.load_corpus()                            # 解析语料
        chunks = dataset.make_chunks(docs)                      # 分块
        try:                                                    # 尝试写入缓存
            with open(cache, "w", encoding="utf-8") as f:       # 以 UTF-8 打开
                json.dump(chunks, f, ensure_ascii=False)        # 序列化
        except Exception as e:                                  # 写入失败
            LOGGER.warning("分块缓存写入失败：%r", e)            # 仅告警
        return chunks                                           # 返回分块

    def _load_bm25(self):                                       # 定义 BM25 加载函数
        """优先读取 BM25 索引缓存，缺失时重建。"""
        cache = os.path.join(config.DATA_DIR, "bm25.pkl")       # 缓存路径
        if os.path.isfile(cache):                               # 缓存存在
            try:                                                # 尝试读取
                with open(cache, "rb") as f:                    # 二进制读取
                    obj = pickle.load(f)                        # 反序列化
                if obj.n_docs == len(self.chunks):              # 与当前分块数量一致
                    LOGGER.info("命中 BM25 索引缓存：%d 文档", obj.n_docs)  # 输出日志
                    return obj                                  # 返回缓存索引
                LOGGER.info("BM25 缓存与分块不一致，将重建")      # 说明不一致
            except Exception as e:                              # 读取失败
                LOGGER.warning("BM25 缓存不可用，将重建：%r", e)  # 输出告警
        bm = BM25()                                             # 新建索引
        bm.fit([c["text"] for c in self.chunks])                # 构建索引
        try:                                                    # 尝试落盘
            with open(cache, "wb") as f:                        # 二进制写入
                pickle.dump(bm, f, protocol=4)                  # 序列化（协议 4 兼容 Python 3.8）
        except Exception as e:                                  # 写入失败
            LOGGER.warning("BM25 缓存写入失败：%r", e)           # 仅告警
        return bm                                               # 返回索引

    def _load_vectors(self):                                    # 定义向量加载函数
        """加载或构建向量矩阵；模型不可用时返回 None（自动降级为 BM25）。"""
        if not embedder.available():                            # 模型不可用
            return None                                         # 返回 None 触发降级
        cache = config.VECTOR_FILE                              # 向量缓存路径
        if os.path.isfile(cache):                               # 缓存存在
            try:                                                # 尝试读取
                data = np.load(cache)                           # 加载 npz
                vecs = data["vectors"]                          # 取出向量矩阵
                if vecs.shape[0] == len(self.chunks):           # 与分块数量一致
                    LOGGER.info("命中向量缓存：%s", vecs.shape)  # 输出日志
                    return vecs                                 # 返回缓存
            except Exception as e:                              # 读取失败
                LOGGER.warning("向量缓存不可用，将重建：%r", e)  # 输出告警
        LOGGER.info("开始编码 %d 个文本块（首次启动较慢）", len(self.chunks))  # 输出提示
        try:                                                    # 尝试编码
            vecs = embedder.encode([c["text"] for c in self.chunks])  # 批量编码
            np.savez_compressed(cache, vectors=vecs)            # 压缩保存到数据卷
            return vecs                                         # 返回向量矩阵
        except Exception as e:                                  # 编码失败
            LOGGER.warning("向量构建失败，降级为 BM25 检索：%r", e)  # 输出告警
            return None                                         # 返回 None 触发降级

    # -------------------------------------------------------------------
    # 二、检索与问答
    # -------------------------------------------------------------------
    def retrieve(self, question, mode="hybrid", top_k=None):    # 定义检索方法
        """执行检索并返回结果块与检索耗时。"""
        t0 = time.time()                                        # 记录起始时间
        qv = None                                               # 查询向量初值
        if self.retriever.embed_enabled and mode != "bm25":     # 需要向量时先编码查询
            try:                                                # 尝试编码
                qv = embedder.encode_query(question)            # 编码问题
            except Exception as e:                              # 编码失败
                LOGGER.warning("查询向量化失败，降级为 BM25：%r", e)  # 输出告警
                mode = "bm25"                                   # 降级为纯 BM25
        chunks = self.retriever.retrieve(question, qv, top_k=top_k, mode=mode)  # 执行检索
        return chunks, round(time.time() - t0, 4)               # 返回结果与耗时

    def answer(self, question, mode="hybrid", top_k=None):      # 定义问答方法
        """执行完整问答流程：检索 + 生成，并写入问答历史。"""
        chunks, retrieve_s = self.retrieve(question, mode=mode, top_k=top_k)  # 检索
        t0 = time.time()                                        # 记录生成起始时间
        answer = llm_client.generate_answer(question, chunks)   # 生成答案
        generate_s = round(time.time() - t0, 4)                 # 生成耗时
        result = {"question": question,                         # 问题
                  "answer": answer,                             # 答案
                  "mode": mode,                                 # 检索模式
                  "retrieve_seconds": retrieve_s,               # 检索耗时
                  "generate_seconds": generate_s,               # 生成耗时
                  "contexts": [{"source": c["source"], "page": c["page"],  # 上下文来源
                                "text": c["text"][:300]} for c in chunks]}  # 上下文片段（截断）
        self._append_history(result)                            # 写入问答历史（验证卷持久化）
        return result                                           # 返回结果

    # -------------------------------------------------------------------
    # 三、历史与统计
    # -------------------------------------------------------------------
    def _append_history(self, record):                          # 定义历史写入方法
        """把问答记录追加写入 JSONL 历史文件（位于挂载卷，重启后不丢失）。"""
        try:                                                    # 尝试写入
            with open(config.HISTORY_FILE, "a", encoding="utf-8") as f:  # 追加模式打开
                f.write(json.dumps(record, ensure_ascii=False) + "\n")  # 写入一行 JSON
        except Exception as e:                                  # 写入失败
            LOGGER.warning("问答历史写入失败：%r", e)            # 仅告警，不影响回答

    def history_count(self):                                    # 定义历史计数方法
        """统计历史问答条数，用于验证数据卷持久化。"""
        if not os.path.isfile(config.HISTORY_FILE):             # 文件不存在
            return 0                                            # 返回 0
        with open(config.HISTORY_FILE, "r", encoding="utf-8") as f:  # 打开文件
            return sum(1 for _ in f)                            # 统计行数

    def stats(self):                                            # 定义统计信息方法
        """返回服务运行状态摘要，供 /stats 接口使用。"""
        return {"chunks": len(self.chunks),                     # 语料块数
                "sources": len(set(c["source"] for c in self.chunks)),  # 来源文件数
                "vector_enabled": bool(self.retriever and self.retriever.embed_enabled),  # 向量开关
                "bm25_terms": len(self.retriever.bm25.df) if self.retriever else 0,  # BM25 词表规模
                "history_count": self.history_count(),          # 历史问答条数
                "build_seconds": self.build_seconds,            # 索引构建耗时
                "version": config.APP_VERSION}                  # 服务版本
