"""src/offline/embedder.py —— 向量化适配层（dense + sparse 双路）。

在链路中的位置（离线侧与在线侧共用）：
    src/offline/chunkers.py 的文本块 → 【本文件】 → src/offline/milvus_store.py 入库
    在线检索时 src/online/retriever.py 也用本文件把查询编码成同一空间的向量

三个后端，通过配置 EMBEDDING_BACKEND 切换（这是本项目的核心降级设计）：
    flag（默认） 本地 BGE-M3 模型。效果最好，但要下载模型、需要 GPU/大内存
    api         远程 HTTP 向量服务。把算力放到远端
    hash        纯哈希假向量。**不依赖任何模型和网络** —— 让整条链路在
                没有模型、没有 GPU、甚至没装 FlagEmbedding 的机器上也能跑通，
                用于验证"注册→上传→检索→问答"的流程是否连通
                （README 里"资源不足时可在 .env 中使用降级配置"指的就是它）

输出的 Embeddings 同时包含两路向量，对应 Milvus 里的两个字段：
    dense  稠密向量（1024 维浮点数组）—— 管语义相似，"回充设备"≈"气体重新充入装置"
    sparse 稀疏向量（词 id -> 权重）—— 管字面精确匹配，像 BM25 但能存进 Milvus
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import requests

from configs.settings import get_settings


@dataclass
class Embeddings:
    """一次向量化的结果，两路向量按下标与输入文本一一对应。

    字段：
        dense:  稠密向量列表，每项 1024 维
        sparse: 稀疏向量列表，每项是 {词哈希: 权重}
    """

    dense: list[list[float]]
    sparse: list[dict[int, float]]


def _hash_embedding(text: str, dimension: int = 1024) -> list[float]:
    """生成确定性的"假"稠密向量（降级后端，不含任何语义信息）。

    参数：
        text: 待编码文本
        dimension: 向量维度，必须与 Milvus 的字段维度一致
    返回：
        单位化（模长为 1）的向量。

    做法（哈希投影法）：
        把文本按每 2 个字符一组切片，对每组算 SHA-256，
        用摘要的前 4 字节决定"落在哪一维"，第 5 字节决定"加多少"（映射到 -0.5~+0.5）。
        最后做归一化。

    为什么需要它：
        这是"让项目在没有模型的机器上也能跑通"的关键。
        它的向量没有语义质量（近义词不会靠近），但满足两个必要条件：
        向量维度对、模长为 1，所以 Milvus 能正常存取、余弦相似度能正常算。
        流程因此完全打通，只是检索质量差 —— 作为"验证链路连通性"足够了。

    为什么必须归一化：
        检索用 COSINE 余弦相似度，它只看向量方向、不看长度。
        不归一化的话，长文本因为累加了更多组、向量模长特别大，
        相似度计算会出现难以解释的结果。

    `or 1.0` 的作用：
        norm 为 0（空文本）时用作除数会抛 ZeroDivisionError，
        兜底成 1.0 让空文本返回全零向量而不崩溃。
    """
    values = [0.0] * dimension
    for index in range(0, len(text), 2):
        digest = hashlib.sha256(text[index : index + 2].encode("utf-8")).digest()
        position = int.from_bytes(digest[:4], "big") % dimension  # 前 4 字节取模 -> 落到哪一维
        values[position] += (digest[4] / 255.0) - 0.5             # 第 5 字节映射到 -0.5~+0.5 的加权
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


def _sparse(text: str) -> dict[int, float]:
    """生成稀疏向量：词 -> 出现次数（降级版的关键词路）。

    参数：
        text: 待编码文本
    返回：
        {词哈希: 权重}，权重就是该词在文本中的出现次数。

    两个刻意的简化：
        1. 直接按空格 split 分词，不走 jieba —— 因为没有中文分词的情况下
           "字面匹配"这条路本来就只能做到这个程度，作为降级方案够用
        2. 用 MD5 前 4 字节当词 id，不做去冲突处理 ——
           哈希冲突的概率极低（2^32 空间），且冲突的后果只是两个无关词共享权重，
           对"流程能否跑通"这个目标无影响

    正常（非降级）路径下，稀疏向量由 BGE-M3 的 lexical_weights 提供，质量高得多。
    """
    result: dict[int, float] = {}
    for token in text.split():
        index = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big")
        result[index] = result.get(index, 0.0) + 1.0
    return result


class BGEEmbedder:
    """向量化器：按配置在 BGE-M3 / 远程 API / 哈希降级之间切换。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None  # 模型惰性加载：不真正编码就不去占内存/显存

    def _load(self) -> Any:
        """惰性加载 BGE-M3 模型（只在首次真正需要编码时加载）。

        返回：
            BGEM3FlagModel 实例。

        use_fp16 与 device 的联动：
            device 不是 cpu 时才开半精度（fp16），因为 CPU 不支持 fp16 运算。
            不这样判断的话，在无 GPU 的机器上会直接报错而不是自动降级。

        为什么把 import 写在函数里：
            FlagEmbedding 是个重依赖，装在 import 阶段会让模块加载变慢。
            放进函数体后，用 hash/api 后端时根本不 import 它。
        """
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.settings.embedding_model, use_fp16=self.settings.embedding_device != "cpu", device=self.settings.embedding_device)
        return self._model

    def encode(self, texts: list[str]) -> Embeddings:
        """把一批文本编码成两路向量。

        参数：
            texts: 文本列表
        返回：
            Embeddings(dense, sparse)，长度与输入一致。

        四条分支的处理顺序（按"代价从低到高"排）：
            1. 空输入直接返回空 —— 避免无意义的计算和接口调用
            2. hash 后端 —— 纯本地计算，最快
            3. api 后端 —— 远程调用，失败会把异常抛给调用方（配置错了就该立刻暴露）
            4. 默认走到 BGE-M3 本地模型，且**失败时静默降级到 hash**（见下面的 except）

        为什么只有本地模型这条路径有 try/except 降级：
            模型加载失败最常见的原因是"没下载下来/内存不够"，属于环境问题而非配置错误。
            这种情况下继续用 hash 跑通流程，比让整次构建失败更有价值；
            而 api 后端配错 URL 属于配置错误，应该让用户看到报错。

        BGE-M3 的 lexical_weights 键是字符串：
            要先 int(key) 转成整数，才能匹配 Milvus 稀疏向量字段的要求。
        """
        if not texts:
            return Embeddings([], [])
        if self.settings.embedding_backend == "hash":
            return Embeddings([_hash_embedding(text) for text in texts], [_sparse(text) for text in texts])
        if self.settings.embedding_backend == "api":
            response = requests.post(self.settings.embedding_api_url, json={"model": self.settings.embedding_model, "input": texts}, timeout=600)
            response.raise_for_status()
            dense = response.json().get("embeddings", [])
            # 远程接口只给稠密向量，稀疏路仍用本地哈希算法补上
            return Embeddings(dense, [_sparse(text) for text in texts])
        try:
            output = self._load().encode(texts, return_dense=True, return_sparse=True)
            sparse = []
            for item in output.get("lexical_weights", []):
                sparse.append({int(key): float(value) for key, value in item.items()})
            return Embeddings(output["dense_vecs"].tolist(), sparse)
        except Exception:
            # 模型不可用时的兜底：宁可检索质量差，也不让整个构建流程中断
            return Embeddings([_hash_embedding(text) for text in texts], [_sparse(text) for text in texts])


# 模块级单例：模型加载很贵，全进程只应加载一次
_embedder = BGEEmbedder()


def embed(texts: list[str]) -> Embeddings:
    """向量化的对外入口（全项目统一从这里调）。

    参数：
        texts: 文本列表
    返回：
        Embeddings(dense, sparse)
    """
    return _embedder.encode(texts)
