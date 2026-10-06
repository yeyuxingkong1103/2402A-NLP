# -*- coding: utf-8 -*-
"""本地向量化 / 重排序模型封装（FlagEmbedding）。模型加载较慢，进程内单例复用。"""
import threading
# 解析：线程模块（懒加载锁）


class LazyModelProxy:
    """模型懒加载代理：构造零成本；首次访问属性才加载；并发下线程安全只加载一次。"""

    def __init__(self, factory):
        # 解析：构造——只保存工厂函数，不加载模型
        self._factory = factory
        # 解析：保存模型创建函数
        self._instance = None
        # 解析：真实模型实例（懒加载）
        self._lock = threading.Lock()
        # 解析：互斥锁（保证并发只加载一次）

    def _get(self):
        # 解析：取真实实例
        if self._instance is None:
            # 解析：未加载（快速路径检查）
            with self._lock:
                # 解析：加锁（并发安全）
                if self._instance is None:
                    # 解析：双重检查——锁内再查一次（其他线程可能已加载）
                    self._instance = self._factory()
                    # 解析：调用工厂加载模型（约 25s/2.5GB，只发生一次）
        return self._instance
        # 解析：返回实例

    def __getattr__(self, name):
        # 解析：属性访问转发——proxy.xxx 等价于真实模型.xxx
        return getattr(self._get(), name)
        # 解析：先取实例再取属性（首次访问触发加载）


class BGEM3Embedder:
    """BGE-m3 稠密向量（1024 维），CPU 推理。"""

    def __init__(self, model_path: str):
        # 解析：构造——加载 BGE-m3 模型
        from FlagEmbedding import BGEM3FlagModel
        # 解析：延迟导入 FlagEmbedding（重依赖）

        self._model = BGEM3FlagModel(model_path, use_fp16=False)
        # 解析：创建模型实例（本机无 GPU，fp16 关闭走 CPU）

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 解析：批量向量化文档
        output = self._model.encode(
            # 解析：调用编码
            texts, batch_size=8, max_length=512, return_dense=True
            # 解析：8 条/批（内存可控）、最长 512 token、只取稠密向量
        )
        # FlagEmbedding 1.x 返回键名为 dense_vecs
        dense = output.get("dense_vecs", output.get("dense"))
        # 解析：键名兼容新旧版本（1.x 是 dense_vecs，旧版是 dense）
        return dense.tolist()
        # 解析：numpy 转 Python 列表返回

    def embed_query(self, text: str) -> list[float]:
        # 解析：单条查询向量化
        return self.embed_documents([text])[0]
        # 解析：复用批量方法取第一条（查询与文档同一编码空间）


class BGEReranker:
    """BGE-reranker-v2-m3 重排序：返回按相关性降序的文本列表。"""

    def __init__(self, model_path: str):
        # 解析：构造——加载重排序模型
        from FlagEmbedding import FlagReranker
        # 解析：延迟导入

        self._model = FlagReranker(model_path, use_fp16=False)
        # 解析：创建模型实例（CPU 推理）

    def rerank(self, query: str, passages: list[str]) -> list[str]:
        # 解析：对候选文本重排序
        if not passages:
            # 解析：无候选
            return []
            # 解析：直接返回空列表
        scores = self._model.compute_score(
            # 解析：批量计算相关性分数
            [[query, passage] for passage in passages], normalize=True
            # 解析：构造 (查询, 候选) 对列表，分数归一化到 0~1
        )
        ordered = sorted(zip(passages, scores), key=lambda pair: -pair[1])
        # 解析：(文本, 分数) 配对后按分数降序排列
        return [passage for passage, _ in ordered]
        # 解析：只返回排序后的文本（丢弃分数）
