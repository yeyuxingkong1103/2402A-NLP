# -*- coding: utf-8 -*-
"""记忆层：短期（Redis 会话历史，TTL 过期）+ 长期（Milvus 摘要存储与召回）。"""
import json  # 消息序列化：Redis 只存字符串，dict 要靠 json 互转
from typing import List  # 类型标注，让返回值结构可读

import numpy as np  # 向量计算：embedding 转 float32 数组
import redis  # 短期记忆依赖：Redis 内存数据库客户端，key-value 存取 O(1)
from pymilvus import MilvusClient, DataType  # 长期记忆依赖：Milvus 向量数据库客户端 + 字段类型枚举
from sentence_transformers import SentenceTransformer  # 文本向量化模型：把摘要/问题编码成稠密向量

from src.config import settings  # 统一配置中心：Redis 地址、Milvus 集合名、TTL 都从 .env 读


class ShortTermMemory:  # 短期记忆：Redis List 存最近 20 轮对话，面向"会话级"高频读写
    """基于 Redis List 的多轮对话短期记忆。"""

    def __init__(self, ttl_seconds: int | None = None):  # 初始化连接与过期时间；传 None 则用全局配置
        self.redis = redis.Redis(  # 建立 Redis 连接（内存数据库，读写微秒级）
            host=settings.redis_host,  # 主机地址
            port=settings.redis_port,  # 端口
            db=settings.redis_db,  # 逻辑库编号：按业务隔离 key 空间
            decode_responses=True,  # 返回 str 而非 bytes，省去手动 decode
        )
        self.ttl = ttl_seconds or settings.session_ttl  # 会话过期时间（默认 1800s）：30 分钟不活跃自动释放内存
        self.max_turns = 20  # 最多保留 20 条消息（10 轮）

    def _key(self, session_id: str) -> str:  # 统一 key 命名，避免散落硬编码
        return f"session:{session_id}"  # 带业务前缀的 key：session:xxx，便于按前缀排查和清理

    def add(self, session_id: str, role: str, content: str) -> None:  # 追加一条消息到会话尾部
        """追加一条消息，超过上限时裁剪最旧的。"""
        key = self._key(session_id)  # 先算好 key，下面三条命令复用
        msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)  # 序列化成 JSON 字符串；ensure_ascii=False 保留中文原文
        self.redis.rpush(key, msg)  # RPUSH 尾部追加：List 天然按时间有序，O(1)
        self.redis.expire(key, self.ttl)  # 每次写入刷新 TTL（滑动过期）：持续对话不过期，停下 30 分钟自动清

        # 滑动窗口裁剪
        while self.redis.llen(key) > self.max_turns:  # LLEN 查长度 O(1)，超出上限就裁
            self.redis.lpop(key)  # LPOP 弹掉最旧一条，保持只留最近 20 条

    def get_history(self, session_id: str) -> List[dict]:  # 读取完整会话历史
        """获取完整会话历史（LangChain 格式）。"""
        key = self._key(session_id)  # 算 key
        raw = self.redis.lrange(key, 0, -1)  # LRANGE 0 -1 取全部元素，一次命令拿整段历史
        return [json.loads(m) for m in raw]  # 反序列化回 dict 列表，供拼提示词用

    def clear(self, session_id: str) -> None:  # 清空会话
        """清空会话（结束会话时调用）。"""
        self.redis.delete(self._key(session_id))  # DEL 整个 key：结束会话时释放内存


def _to_python_floats(raw) -> list:  # 向量类型转换工具函数
    """把 numpy 向量强制转成纯 Python float 列表（pymilvus struct.pack 硬性要求）。"""
    return np.asarray(raw, dtype=np.float32).reshape(-1).astype(float).tolist()  # numpy float32 → 原生 float 列表，否则写入会报类型错


class LongTermMemory:  # 长期记忆：Milvus 向量化存用户病史摘要，语义检索召回
    """用户级别的长期记忆，存 Milvus 的 memory collection。"""

    def __init__(self, embedder: SentenceTransformer):  # 注入共享的向量化模型，与知识库检索同模型保证向量空间一致
        self.client = MilvusClient(uri=settings.milvus_uri, timeout=10)  # 连接 Milvus（Lite 模式本地文件即可），超时 10s 快速失败
        self.embedder = embedder  # 保存向量化模型引用
        self.collection = settings.milvus_memory_collection  # 记忆专用 collection 名，与知识库 collection 分开
        self._ensure_collection()  # 幂等建表：不存在才建，重复启动不报错

    def _ensure_collection(self) -> None:  # 确保 collection 存在
        """确保 memory collection 存在。"""
        if self.client.has_collection(self.collection):  # 已存在直接返回
            return  # 幂等退出
        dim = len(_to_python_floats(self.embedder.encode(["test"])[0]))  # 试编码一个词拿向量维度（bge 模型通常 1024 维）
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)  # 建表结构：主键自增，禁止动态字段保证结构可控
        schema.add_field("id", DataType.INT64, is_primary=True)  # 主键 id
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)  # 向量字段：维度必须与模型输出一致
        schema.add_field("user_id", DataType.VARCHAR, max_length=100)  # 用户隔离字段：召回时按用户过滤
        schema.add_field("summary", DataType.VARCHAR, max_length=2000)  # 摘要正文字段
        index = self.client.prepare_index_params()  # 准备索引参数
        index.add_index(field_name="vector", index_type="FLAT", metric_type="IP")  # FLAT 暴力精确检索（记忆数据量小够准）+ 内积 IP 度量（向量已归一化时等价余弦相似度）
        self.client.create_collection(self.collection, schema=schema, index_params=index)  # 建表带索引，一步到位

    def save(self, user_id: str, summary: str) -> None:  # 保存一条用户病史摘要
        """保存用户会话摘要到长期记忆。"""
        vec = _to_python_floats(self.embedder.encode(summary, normalize_embeddings=True))  # 摘要向量化并归一化：归一化后内积=余弦相似度
        self.client.insert(self.collection, [{  # 单行插入
            "vector": vec,  # 向量
            "user_id": str(user_id),  # 归属用户
            "summary": summary,  # 摘要原文（召回时直接取，不用反向解码）
        }])  # 插入完成

    def recall(self, user_id: str, query: str, top_k: int = 3) -> List[str]:  # 语义召回：按当前问题找该用户最相关的历史摘要
        """根据当前 query 召回用户相关的历史摘要。"""
        filter_expr = f'user_id == "{user_id}"'  # 标量过滤表达式：先按用户隔离，再做向量检索（多用户互不串味）
        qv = _to_python_floats(self.embedder.encode(query, normalize_embeddings=True))  # 查询向量化，与写入侧同样归一化，保证度量一致
        res = self.client.search(  # 向量相似度检索
            self.collection, data=[qv], limit=top_k,  # Top-K=3：取最相关的 3 条病史
            output_fields=["summary"], filter=filter_expr,  # 只回传摘要字段，并带用户过滤条件
        )  # 检索完成
        return [hit["entity"]["summary"] for hit in res[0]] if res[0] else []  # 提取摘要文本；无命中返回空列表不报错
