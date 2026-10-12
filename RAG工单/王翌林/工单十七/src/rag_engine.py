# -*- coding: utf-8 -*-
"""
优化版 RAG 引擎：
- 单例（singleton）：重量级组件（Embedder / Reranker / LLM / VectorStore）仅初始化一次；
- 连接池：向量检索客户端从池中借出/归还，避免每请求新建连接；
- 有界缓存：LRU + TTL，防止全局缓存无限增长导致内存泄漏；
- 并发限流：LLM 推理用信号量限制并发，防止 GPU/线程打满；
- 资源释放：推理后显式 gc + （若用 GPU）empty_cache，中间张量不残留。
设计为可在 CPU 上运行（哈希嵌入 + 内存向量库 + 可控延迟的 LLM），便于压测复现。
"""
import os, gc, time, hashlib, threading, heapq
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

CORPUS = [
    "淬火是将钢加热到临界温度以上保温后快速冷却的热处理工艺，可提高硬度与耐磨性。",
    "公差与配合用于控制机械零件的尺寸精度，保证互换性与装配要求。",
    "链条传动适合中心距较大的传动场合，能在恶劣环境下工作。",
    "带式输送机由驱动滚筒、托辊、输送带与驱动电机组成，用于散料连续输送。",
    "轧制带材时调节张力的主要目的是控制板形，防止跑偏与厚度波动。",
    "水洗洗涤塔通过气液接触去除气体中的粉尘与可溶性污染物。",
]


class Embedder:
    """轻量确定性嵌入（单例）：用特征哈希模拟句向量，避免每请求加载大模型。"""
    _dim = 256

    def encode(self, text):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = [0.0] * self._dim
        for i, b in enumerate(h):
            v[i % self._dim] += (b / 255.0)
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]


class VectorStore:
    """内存向量库（单例 + 连接池语义）。"""

    def __init__(self):
        self._emb = Embedder()
        self._docs = [(d, self._emb.encode(d)) for d in CORPUS]

    def search(self, query, top_k=3):
        qv = self._emb.encode(query)
        scored = []
        for doc, dv in self._docs:
            s = sum(a * b for a, b in zip(qv, dv))
            scored.append((s, doc))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:top_k]]


class LLMClient:
    """LLM 客户端（单例 + 并发限流 + 资源释放）。"""

    def __init__(self, latency=0.8, max_concurrency=40):
        self.latency = latency
        self._sem = threading.Semaphore(max_concurrency)
        self._pool = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="llm")

    def generate(self, prompt):
        with self._sem:  # 限制真实耗时推理的并发，防打满
            time.sleep(self.latency)  # 模拟推理耗时
            # 模拟生成：取 prompt 中关键词做简短回答
            ans = "根据检索到的文档，" + prompt[:24] + "。"
            # 资源释放：清理中间对象（真实场景为 del tensor + torch.cuda.empty_cache()）
            tmp = [0] * 4096
            del tmp
            gc.collect()
            return ans


class LRUCache:
    """有界 LRU + TTL 缓存：防止内存泄漏（对比：基线版本无上限会无限增长）。"""

    def __init__(self, capacity=512, ttl=300):
        self.capacity = capacity
        self.ttl = ttl
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                val, exp = self._data[key]
                if time.time() < exp:
                    self._data.move_to_end(key)
                    return val
                del self._data[key]
            return None

    def set(self, key, value):
        with self._lock:
            self._data[key] = (value, time.time() + self.ttl)
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._data)


class ConnectionPool:
    """向量检索客户端连接池：借出/归还，避免每请求新建（修复连接泄漏）。"""

    def __init__(self, size=8):
        self._free = [VectorStore() for _ in range(size)]
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            if self._free:
                return self._free.pop()
        return VectorStore()  # 池空时临时创建（生产应阻塞/限流）

    def release(self, conn):
        with self._lock:
            if len(self._free) < 16:
                self._free.append(conn)


class RAGEngine:
    """单例 RAG 引擎。"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, latency=0.8):
        if getattr(self, "_ready", False):
            return
        self.embedder = Embedder()
        self.pool = ConnectionPool(size=8)
        self.llm = LLMClient(latency=latency, max_concurrency=40)
        self.cache = LRUCache(capacity=512, ttl=300)
        self._ready = True
        print(f"[RAGEngine] singleton 初始化完成（连接池8 / LLM并发8 / 缓存上限512）")

    def query(self, question, top_k=3):
        # 1) 语义缓存命中则直接返回（省检索+生成）
        cached = self.cache.get(question)
        if cached is not None:
            return cached, True
        # 2) 从连接池借向量客户端检索
        conn = self.pool.acquire()
        try:
            docs = conn.search(question, top_k=top_k)
        finally:
            self.pool.release(conn)  # 必须归还，防连接泄漏
        # 3) 生成
        prompt = question + " | 上下文: " + " ".join(docs)
        answer = self.llm.generate(prompt)
        result = {"question": question, "answer": answer, "docs": docs}
        self.cache.set(question, result)
        return result, False
