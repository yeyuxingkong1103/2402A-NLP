# -*- coding: utf-8 -*-
"""
本地向量库
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：轻量级向量存储（numpy + JSON 持久化），余弦相似度检索。
     数据规模（两份招股说明书约千级分块）下性能足够，且零外部依赖。
"""
import os        # 创建索引目录、拼接文件路径
import json      # JSON 持久化（文本/元数据可读，便于检查）
import numpy as np  # 向量矩阵运算与相似度计算

from config import INDEX_DIR        # 索引持久化目录
from ollama_client import client    # 全局单例：负责调 Ollama 做向量化


class VectorStore:
    """基于 numpy 的本地向量库"""

    def __init__(self):
        self.texts = []      # 分块文本
        self.metadatas = []  # 元数据 [{"page":.., "source":..}, ...]
        self.vecs = None     # np.ndarray [n, dim]

    # 三条列表/数组按下标一一对应：texts[i] / metadatas[i] / vecs[i] 是同一条数据

    # ── 构建 ───────────────────────────────────────────────
    def add(self, chunks, source_name):
        """向量化并写入
        参数 chunks: [{"text": str, "page": int}, ...]
        """
        print(f"[VectorStore] 向量化 {len(chunks)} 个分块 (来源: {source_name}) ...")
        # 批量向量化：Ollama 无批量接口，embed_batch 内部逐条调用
        vecs = client.embed_batch([c["text"] for c in chunks])
        for c, v in zip(chunks, vecs):
            self.texts.append(c["text"])  # 文本与向量按下标对齐存入
            self.metadatas.append({"page": c.get("page"), "source": source_name})
        # 转成 float32：比默认 float64 省一半内存，精度损失对相似度排序无影响
        new = np.array(vecs, dtype=np.float32)
        # 首次写入直接赋值；之后用 vstack 沿第 0 维（行）追加，保持 [n, dim] 形状
        self.vecs = new if self.vecs is None else np.vstack([self.vecs, new])
        print(f"[VectorStore] 完成，当前总量: {len(self.texts)}")

    # ── 检索 ───────────────────────────────────────────────
    @staticmethod
    def _cosine(a, b):
        """余弦相似度: a [n,d] 与 b [d]"""
        # a 按行求范数得 [n]（每个分块向量的模），b 是标量（查询向量的模）
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b)
        # 分母加 1e-8 防"零向量导致除零"产生 NaN（bge-m3 正常输出不为零，属防御性写法）
        return (a @ b) / (na * nb + 1e-8)

    def search(self, query, top_k=5):
        """查询向量化后做余弦相似度检索
        返回: [{"text", "page", "source", "score"}, ...]
        """
        # 空库直接返回，避免对 None/空数组做矩阵运算报错
        if self.vecs is None or len(self.texts) == 0:
            return []
        qv = np.array(client.embed(query), dtype=np.float32)  # 查询用同一模型向量化，保证维度一致
        scores = self._cosine(self.vecs, qv)  # 一次矩阵乘法算出全部 n 条相似度
        # 取负再 argsort 即降序排列，截取前 top_k 个下标
        idx = np.argsort(-scores)[:top_k]
        return [
            {
                "text": self.texts[i],
                "page": self.metadatas[i].get("page"),
                "source": self.metadatas[i].get("source"),
                "score": float(scores[i]),  # np.float32 转原生 float，否则 JSON 序列化报错
            }
            for i in idx
        ]

    # ── 持久化 ─────────────────────────────────────────────
    def save(self, name):
        # exist_ok=True：目录已存在时不抛异常（重复保存场景）
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            # numpy 数组不能直接 json.dump，先 .tolist() 转成嵌套 list
            "vecs": self.vecs.tolist() if self.vecs is not None else None,
        }
        path = os.path.join(INDEX_DIR, f"{name}.json")
        # encoding="utf-8" 必须：Windows 默认 gbk 编码写中文会乱码/报错
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[VectorStore] 已保存: {path}")

    @classmethod
    def load(cls, name):
        path = os.path.join(INDEX_DIR, f"{name}.json")
        if not os.path.exists(path):
            return None  # 索引不存在返回 None，由调用方决定是否先构建
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls()  # cls() 是类方法工厂写法：子类继承时也能正确实例化
        store.texts = data["texts"]
        store.metadatas = data["metadatas"]
        # 向量转回 float32 的 ndarray；vecs 为 None（空库）时保持 None
        store.vecs = np.array(data["vecs"], dtype=np.float32) if data["vecs"] else None
        print(f"[VectorStore] 已加载 {len(store.texts)} 条: {path}")
        return store
