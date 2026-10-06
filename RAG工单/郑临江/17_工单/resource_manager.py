# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单
资源池化与单例模式：重量级组件（文档解析器、重排模型、VLM）全局单例；
数据库/向量库连接使用连接池并正确归还，修复资源泄漏。
"""
import threading

import config


# ---- 重量级组件单例 ----
_singletons = {}
_lock = threading.Lock()


class DeepDocParser:
    """文档解析器（重量级）。"""
    def __init__(self):
        # 模拟初始化耗时
        self.loaded = True

    def parse(self, path):
        return f"[parsed]{path}"


class ReRankModel:
    """重排模型（重量级）。"""
    def __init__(self):
        self.loaded = True

    def rerank(self, docs, query):
        return docs


class VLMModel:
    """视觉语言模型（GPU 常驻，释放不及时会占显存）。"""
    def __init__(self):
        self.loaded = True

    def generate(self, prompt):
        return f"[vlm]{prompt[:20]}"


def _get_singleton(name, factory):
    if name not in _singletons:
        with _lock:
            if name not in _singletons:
                _singletons[name] = factory()
    return _singletons[name]


def get_parser():
    if config.SINGLETON_PARSER:
        return _get_singleton("parser", DeepDocParser)
    return DeepDocParser()   # 未优化：每请求新建


def get_reranker():
    return _get_singleton("reranker", ReRankModel)


def get_vlm():
    if config.SINGLETON_VLM:
        return _get_singleton("vlm", VLMModel)
    return VLMModel()


# ---- 连接池 ----
class ConnectionPool:
    """连接池：连接用后正确归还，修复连接泄漏。"""
    def __init__(self, factory, size=10):
        self._factory = factory
        self._size = size
        self._pool = [factory() for _ in range(size)]
        self._cond = threading.Condition()

    def acquire(self):
        with self._cond:
            while not self._pool:
                self._cond.wait()
            return self._pool.pop()

    def release(self, conn):
        with self._cond:
            self._pool.append(conn)
            self._cond.notify()

    def __enter__(self):
        self._conn = self.acquire()
        return self._conn

    def __exit__(self, *a):
        self.release(self._conn)


class FakeConn:
    def __init__(self):
        self.id = id(self)

    def query(self, sql):
        return []


DB_POOL = ConnectionPool(FakeConn, size=config.DB_POOL_SIZE)
VECTOR_POOL = ConnectionPool(FakeConn, size=config.VECTOR_POOL_SIZE)
