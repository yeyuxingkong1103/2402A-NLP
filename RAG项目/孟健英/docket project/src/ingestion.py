# -*- coding: utf-8 -*-
"""知识库入库管线：按页切分 → 切块 → BGE-m3 向量化 → Milvus 入库。"""
import logging  # 日志：记录向量化与入库条数，建库过程可追踪、可复盘
import re  # 正则：识别解析器写出的 [第N页] 页码标记，按页切分全靠它
from typing import List  # 类型注解：让函数签名自解释，便于 IDE 静态检查

from pymilvus import DataType, MilvusClient  # Milvus 官方客户端：DataType 定义字段类型，MilvusClient 负责连库与读写；Milvus 是生产级向量数据库
from sentence_transformers import SentenceTransformer  # 加载 BGE 系列 Embedding 模型，把文本块编码成稠密向量（语义检索的基础）

from src.config import settings  # 统一配置中心：模型名、Milvus 地址、集合名等都从 .env 读

logger = logging.getLogger(__name__)  # 本模块日志器
_embedder = None  # 延迟加载，避免导入就拉模型

# 解析器写出的页码标记，形如“[第12页]”
_PAGE_MARK = re.compile(r"\[第(\d+)页\]")  # 预编译正则：捕获组抓页码数字，编译一次反复复用


def get_embedder() -> SentenceTransformer:  # 单例工厂：全进程共享同一个模型实例
    """单例模式获取向量化模型（首次调用才加载）。"""
    global _embedder  # 声明操作的是模块级变量，否则赋值会变成新建同名局部变量
    if _embedder is None:  # 懒加载：模型几百 MB，首次真正使用时才载入内存
        _embedder = SentenceTransformer(
            settings.embedding_model, device=settings.embedding_device  # 模型名与设备(cpu/cuda)走配置；入库与在线检索必须用同一个 Embedding 模型，否则向量空间不匹配、检索结果全是噪声
        )
    return _embedder  # 之后的调用直接返回已加载实例，不重复加载


def split_pages(text: str) -> List[tuple]:  # 切页：把整篇文本按页码标记拆开
    """按“[第N页]”标记切页，返回 [(页码, 该页正文), ...]。"""
    parts = _PAGE_MARK.split(text)  # re.split 带捕获组时，捕获内容（页码）也会留在结果里，形成交错数组
    pages = []  # 存放 (页码, 正文) 二元组
    # split 结果形如 [前言, 页号1, 正文1, 页号2, 正文2, ...]
    for i in range(1, len(parts) - 1, 2):  # 步长 2 跳取：奇数位是页码，紧随其后的偶数位是该页正文
        body = parts[i + 1].strip()  # 取页码后面的正文并去首尾空白
        if body:  # 跳过只有页码没有内容的空页
            pages.append((int(parts[i]), body))  # 页码转 int 存好，引用展示要按页码定位
    return pages


def chunk_pages(pages: List[tuple], source: str, chunk_size: int = 55, overlap: int = 8) -> List[dict]:  # 分块：滑窗把每页切成定长小块，块是向量化与检索的最小单元
    """按页切块，块记录必须带来源与页码（引用展示要用）。"""
    records = []  # 块记录列表，每块是 {text, source, page} 字典
    step = max(1, chunk_size - overlap)  # 步长=块长-重叠；chunk_overlap 防关键信息被切在边界——边界句在相邻块里还能完整出现一次
    for page_no, body in pages:  # 逐页处理，页码一路带进块记录
        clean = body.strip()  # 去首尾空白再切，避免块里混入无意义空字符
        for start in range(0, len(clean), step):  # 按步长滑窗切块
            chunk = clean[start:start + chunk_size].strip()  # 定长切片：块长是语义完整性与向量表达精度的权衡
            if len(chunk) < 10:  # 过短的块（多为页尾残渣）语义不完整，丢弃
                continue
            records.append({"text": chunk, "source": source, "page": int(page_no)})  # 来源文件名+页码随块入库：答案引用溯源要用
    return records


def _ensure_collection(client: MilvusClient, collection: str, dim: int) -> None:  # 建集合：Collection≈关系库的表
    """确保 Milvus collection 存在（幂等）。"""
    if client.has_collection(collection):  # 幂等：已存在就直接返回，重复执行不报错
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)  # 定义表结构：auto_id 主键自增；关闭动态字段，schema 外的字段一律拒收，防脏数据
    schema.add_field("id", DataType.INT64, is_primary=True)  # Field≈列：主键 id
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)  # 向量字段：维度必须与 Embedding 模型输出一致（BGE-m3 为 1024 维）
    schema.add_field("text", DataType.VARCHAR, max_length=4000)  # 块原文：检索命中后作为上下文喂给 LLM
    schema.add_field("source", DataType.VARCHAR, max_length=200)  # 来源文件名：回答里标注出自哪份指南
    schema.add_field("page", DataType.INT64)  # 页码：回答里标注出自第几页
    schema.add_field("chunk_index", DataType.INT64)  # 同一来源内的块序号：需要时可按序回读上下文

    index = client.prepare_index_params()  # 索引参数容器
    index.add_index(field_name="vector", index_type="FLAT", metric_type="IP")  # FLAT=暴力全扫，数据量不大时精度最高；IP=内积，配合归一化向量等价于余弦相似度
    client.create_collection(collection, schema=schema, index_params=index)  # schema 与索引一起下发，真正建集合


def reset_collection(collection: str | None = None) -> bool:  # 删集合：全量重建前的清场动作
    """删掉旧集合，便于重建知识库（重建前先确认）。"""
    client = MilvusClient(uri=settings.milvus_uri)  # 连 Milvus：uri 可以是本地文件(milvus lite)或远程服务地址
    collection = collection or settings.milvus_collection  # 未指定则用配置里的默认集合名
    if client.has_collection(collection):  # 存在才删，避免对不存在的集合报错
        client.drop_collection(collection)  # 整个集合连数据带索引一起删
        return True  # 确实删了返回 True
    return False  # 本来就没有，返回 False


def embed_and_insert(records: List[dict], collection: str | None = None) -> int:  # 入库主入口：向量化 → 写入 Milvus 一条龙
    """向量化并入库 Milvus，返回插入条数。"""
    collection = collection or settings.milvus_collection  # 默认写入配置里的集合
    embedder = get_embedder()  # 取 Embedding 模型单例（首次调用触发加载）
    client = MilvusClient(uri=settings.milvus_uri)  # 连接 Milvus

    texts = [r["text"] for r in records]  # 抽出全部块文本：批量编码比逐条编码快得多
    logger.info("向量化 %d 个块...", len(texts))  # 建库耗时大头在向量化，先记一笔
    raw = embedder.encode(texts, normalize_embeddings=True).astype("float32").tolist()  # 批量编码并 L2 归一化：归一化后内积=余弦相似度；float32 是 Milvus 向量字段的要求
    vectors = [[float(x) for x in v] for v in raw]  # numpy 标量转原生 float，避免客户端序列化兼容问题

    _ensure_collection(client, collection, dim=len(vectors[0]))  # 首次入库时按向量实际维度建集合

    # chunk_index：同批次内按来源递增的块序号（真实集合已有该非空字段）
    seq: dict = {}  # 每个来源各自的块计数器
    data = []  # 待插入数据列表：每条是一个 Entity，Entity≈表里的一行
    for v, r in zip(vectors, records):  # 向量与块记录一一配对
        src = r.get("source", "")  # 来源文件名，缺省空串
        seq[src] = seq.get(src, -1) + 1  # 该来源的块序号从 0 开始递增
        data.append({"vector": v, "text": r["text"], "source": src,  # 组装一行 Entity：向量+原文+来源
                     "page": int(r.get("page", 0)), "chunk_index": int(seq[src])})  # 页码与块序号一并入库，引用溯源用
    result = client.insert(collection, data)  # 批量插入：一次网络往返写入整批，远快于逐条
    count = result["insert_count"]  # 从返回结果里取实际插入条数
    logger.info("入库 %d 条到 %s", count, collection)  # 记日志便于核对建库规模
    return count
