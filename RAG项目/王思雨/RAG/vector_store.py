# -*- coding: utf-8 -*-
"""向量库模块：BGE-M3 向量化 + Milvus 入库，第 3 步只做写入，检索与重排留到下一步。"""

import argparse                               # 导入 argparse，用于解析命令行参数
import json                                   # 导入 json，用于读取解析结果文件
import time                                   # 导入 time，用于生成时间戳与统计耗时
from pathlib import Path                      # 导入 Path，用于路径处理
from typing import List                       # 导入 List，用于声明列表类型

from pymilvus import DataType, MilvusClient   # 导入 Milvus 客户端与字段类型枚举
import config                                 # 导入配置模块，连接参数与模型路径都从这里取
from logger import get_logger                 # 导入日志工具，用于记录向量库日志

logger = get_logger("vector_store")           # 创建本模块的 logger 实例

VECTOR_DIM = 1024                             # BGE-M3 输出向量维度，固定 1024 维
MAX_LEN_TEXT = 8192                           # text 字段最大字符数（向量对应原文）
MAX_LEN_SUMMARY = 2048                        # summary 字段最大字符数（摘要）
MAX_LEN_SOURCE = 512                          # source 字段最大字符数（文档来源）
EMBED_BATCH_SIZE = 16                         # 向量化的批大小
INSERT_BATCH_SIZE = 64                        # 写入 Milvus 的批大小
EMBED_MAX_TOKENS = 512                        # 单条文本最大 token 数，块本身很短，512 足够
LOG_EVERY = 100                               # 每处理多少条打印一次进度日志

CHUNK_JSON = Path(__file__).resolve().parent / "data" / "enriched_chunks.json"   # 增强结果文件

# Collection 必须包含的字段名，供建表与测试校验使用
REQUIRED_FIELDS = (                           # 字段名元组开始
    "id",                                     # 主键，自增编号
    "vector",                                 # 稠密向量字段，用于语义相似度检索
    "sparse_vector",                          # 稀疏向量字段，用于关键词稀疏检索
    "text",                                   # 向量对应的原文，供混合检索使用
    "summary",                                # 片段摘要
    "parent_id",                              # 所属父块索引
    "source",                                 # 文档来源
    "created_at",                             # 创建时间，时间戳（秒）
    "updated_at",                             # 修改时间，时间戳（秒）
)                                             # 字段名元组结束

_BGE_M3 = None                                # BGE-M3 模型单例缓存，避免重复加载


# ===================== Milvus：连接与 Collection =====================

def get_milvus_client() -> MilvusClient:       # 返回 Milvus 客户端对象
    """获取 Milvus 客户端，地址来自 config（即 .env），代码中不写死。"""
    if not config.MILVUS_HOST:                                  # 主机地址为空说明 .env 还没配好
        raise RuntimeError("MILVUS_HOST 未配置，请复制 .env.example 为 .env 并填写")  # 立即报错
    uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"   # 拼出 Milvus 服务地址
    return MilvusClient(uri=uri, timeout=config.MILVUS_TIMEOUT)  # 创建并返回客户端


def build_schema():                            # 返回 Collection 的字段定义
    """构建 Collection 字段定义：id、vector、text、summary、source、created_at、updated_at。"""
    schema = MilvusClient.create_schema(       # 创建字段集合（schema）
        auto_id=True,                          # 主键由 Milvus 自动生成，不需要手工传
        enable_dynamic_field=False,            # 关闭动态字段，字段必须严格按定义
    )                                          # schema 创建结束
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)  # 主键，INT64 自增
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)  # 稠密向量，1024 维
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量，BGE-M3 学习式稀疏
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=MAX_LEN_TEXT)  # 原文
    schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=MAX_LEN_SUMMARY)  # 摘要
    schema.add_field(field_name="parent_id", datatype=DataType.INT64)   # 所属父块索引，无父块为 -1
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=MAX_LEN_SOURCE)  # 来源
    schema.add_field(field_name="created_at", datatype=DataType.INT64)   # 创建时间戳
    schema.add_field(field_name="updated_at", datatype=DataType.INT64)   # 修改时间戳
    return schema                              # 返回构建好的 schema


def get_field_names() -> List[str]:            # 返回字段名列表
    """返回 Collection 的字段名列表，不需要连接 Milvus 即可调用，方便测试校验。"""
    return [field.name for field in build_schema().fields]   # 从 schema 中取出所有字段名


def create_collection(recreate: bool = False) -> None:       # 创建向量集合
    """创建 Collection；recreate=True 时先删除再重建，保证入库前集合是干净的。"""
    client = get_milvus_client()                             # 获取 Milvus 客户端
    name = config.MILVUS_COLLECTION                          # 集合名称来自 config
    if client.has_collection(name):                          # 如果集合已经存在
        if not recreate:                                     # 且不要求重建
            logger.info("集合 %s 已存在，跳过创建", name)      # 记录日志后直接返回
            return                                           # 结束函数
        client.drop_collection(name)                         # 要求重建则先删除旧集合
        logger.info("集合 %s 已删除，准备重建", name)          # 记录删除日志
    index_params = client.prepare_index_params()             # 创建索引参数对象
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")  # 稠密向量索引
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")  # 稀疏倒排索引
    client.create_collection(                                # 创建集合
        collection_name=name,                                # 集合名称
        schema=build_schema(),                               # 使用上面定义的字段
        index_params=index_params,                           # 挂上向量索引
    )                                                        # 创建集合结束
    logger.info("集合 %s 创建完成", name)                     # 记录创建完成日志


def drop_collection() -> None:                               # 删除向量集合
    """删除 Collection，方便开发期重建，集合不存在时不做任何事。"""
    client = get_milvus_client()                             # 获取 Milvus 客户端
    name = config.MILVUS_COLLECTION                          # 集合名称来自 config
    if client.has_collection(name):                          # 只有存在才删除
        client.drop_collection(name)                         # 执行删除
        logger.info("集合 %s 已删除", name)                   # 记录删除日志


# ===================== BGE-M3 向量化 =====================

def _cuda_available() -> bool:                 # 内部函数：判断有没有可用显卡
    """判断当前环境是否有可用的 CUDA 显卡，失败时按无显卡处理。"""
    try:                                       # 尝试导入 torch 并查询显卡
        import torch                           # 延迟导入，没装 torch 时不影响其他功能
        return bool(torch.cuda.is_available())  # 返回显卡是否可用
    except Exception:                          # 没装 torch 或查询失败
        return False                           # 按无显卡处理


def load_bge_m3():                             # 加载 BGE-M3 向量模型
    """加载本地 BGE-M3 向量模型，使用懒加载单例，避免重复加载占显存。"""
    global _BGE_M3                             # 声明使用模块级单例变量
    if _BGE_M3 is not None:                    # 已经加载过
        return _BGE_M3                         # 直接返回缓存
    from FlagEmbedding import BGEM3FlagModel    # 延迟导入，避免未安装时影响模块加载
    model_path = config.BGE_M3_PATH            # 从配置读取模型本地路径
    if not model_path or not Path(model_path).exists():   # 路径为空或不存在
        raise RuntimeError(f"BGE-M3 模型路径不存在：{model_path}")   # 立即报错
    device = "cuda" if _cuda_available() else "cpu"       # 有显卡用显卡，否则用 CPU
    logger.info("开始加载 BGE-M3：%s（设备：%s）", model_path, device)   # 记录日志
    start = time.time()                        # 记录加载开始时间
    _BGE_M3 = BGEM3FlagModel(                  # 创建模型实例
        model_path,                            # 模型本地路径
        use_fp16=(device == "cuda"),           # 显卡上用半精度，加速推理
        devices=device,                        # 指定运行设备（该库参数名为 devices）
    )                                          # 模型创建结束
    logger.info("BGE-M3 加载完成，耗时 %.1f 秒", time.time() - start)   # 记录加载耗时
    return _BGE_M3                             # 返回模型实例


def _to_milvus_sparse(weights) -> dict:        # 内部函数：转换稀疏向量格式
    """把 BGE-M3 的 lexical_weights 转成 Milvus 要求的 {整数词项: 浮点权重} 格式。"""
    return {int(token): float(weight) for token, weight in weights.items()}   # 逐项转换


def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH_SIZE):
    """把文本批量转成 dense 与 sparse 两种向量，输入为空时返回两个空列表。"""
    if not texts:                              # 输入为空
        return [], []                          # 返回两个空列表
    model = load_bge_m3()                      # 获取模型单例
    dense_all = []                             # 保存稠密向量
    sparse_all = []                            # 保存稀疏向量
    total = len(texts)                         # 文本总条数
    last_logged = 0                            # 上次打印进度时的条数
    for start in range(0, total, batch_size):  # 按批遍历
        batch = texts[start:start + batch_size]              # 取出当前批次
        outcome = model.encode(                              # 调用模型编码
            batch,                                           # 当前批文本
            batch_size=batch_size,                           # 批大小
            max_length=EMBED_MAX_TOKENS,                     # 单条最大 token 数
            return_dense=True,                               # 要稠密向量
            return_sparse=True,                              # 要稀疏权重（学习式稀疏检索）
            return_colbert_vecs=False,                       # 不要 ColBERT 向量
        )                                                    # 编码调用结束
        dense_all.extend([vec.tolist() for vec in outcome["dense_vecs"]])   # 收集稠密向量
        sparse_all.extend([_to_milvus_sparse(w) for w in outcome["lexical_weights"]])   # 收集稀疏向量
        done = min(start + batch_size, total)                # 已处理条数
        if done - last_logged >= LOG_EVERY or done == total:  # 每满 100 条或最后一批
            logger.info("向量化进度：%d/%d", done, total)     # 打印进度日志
            last_logged = done                                # 记录本次打印位置
    return dense_all, sparse_all               # 返回稠密与稀疏两组向量


def embed_query(query: str):
    """把单条查询文本转成 dense 与 sparse 向量，供检索时使用。"""
    dense_list, sparse_list = embed_texts([query])   # 复用批量向量化接口
    return dense_list[0], sparse_list[0]       # 返回这一条的两个向量


# ===================== 读取解析结果 =====================

def load_chunks_from_json(json_path: str = None) -> List[dict]:   # 读取解析结果
    """读 data/parsed_chunks.json，摊平所有块并去重，返回待入库的块列表。"""
    path = Path(json_path) if json_path else CHUNK_JSON   # 未指定路径时用默认路径
    if not path.exists():                      # 文件不存在
        raise FileNotFoundError(f"解析结果文件不存在：{path}")   # 立即报错
    payload = json.loads(path.read_text(encoding="utf-8"))       # 读取并解析 JSON
    raw_chunks = payload.get("chunks")         # 优先按增强结果的新格式取块
    if raw_chunks is None:                     # 不是新格式（第 2 步的解析结果）
        raw_chunks = []                        # 收集旧格式的块
        for result in payload.get("results", []):   # 遍历每个 PDF
            for item in result.get("chunks", []):   # 遍历该 PDF 的每个块
                item.setdefault("source", result.get("source", ""))   # 补齐来源字段
                raw_chunks.append(item)        # 收进列表
    chunks = []                                # 保存整理后的块
    seen = set()                               # 记录已出现过的文本，用于去重
    skipped_parent = 0                         # 跳过的父块数量
    for item in raw_chunks:                    # 逐块整理
        if item.get("is_parent"):              # 父块只用于回溯上下文，不入库
            skipped_parent += 1                # 计数
            continue                           # 跳过
        text = (item.get("text") or "").strip()   # 取出块文本并去空白
        if not text or text in seen:           # 空文本或重复文本
            continue                           # 跳过
        seen.add(text)                         # 记录该文本
        chunks.append({                        # 组装待入库结构
            "text": text,                      # 块正文
            "source": item.get("source", ""),  # 文件名
            "summary": (item.get("summary") or "")[:MAX_LEN_SUMMARY],   # 第 7 步生成的摘要
            "parent_id": int(item.get("parent_id", -1)),   # 所属父块索引，无父块为 -1
        })                                     # 结构组装结束
    logger.info("读取 %s：%d 块（跳过父块 %d 个）", path.name, len(chunks), skipped_parent)   # 记录日志
    return chunks                              # 返回块列表


# ===================== 写入 Milvus =====================

def insert_chunks(chunks: List[dict], recreate: bool = False) -> int:   # 批量写入向量
    """把块写入 Milvus：向量化后按批插入，recreate=True 时先重建集合，返回成功条数。"""
    if recreate:                               # 需要重建集合
        drop_collection()                      # 先删除旧集合
        create_collection()                    # 再创建新集合
    client = get_milvus_client()               # 获取 Milvus 客户端
    name = config.MILVUS_COLLECTION            # 集合名称
    now = int(time.time())                     # 当前时间戳，写入时间字段
    inserted = 0                               # 已成功写入条数
    total = len(chunks)                        # 待写入总条数
    for start in range(0, total, INSERT_BATCH_SIZE):        # 按批遍历
        batch = chunks[start:start + INSERT_BATCH_SIZE]     # 取出当前批次
        dense_vecs, sparse_vecs = embed_texts([item["text"] for item in batch])   # 批量向量化
        rows = []                              # 组装本批要写入的行
        for item, dvec, svec in zip(batch, dense_vecs, sparse_vecs):   # 逐条组装
            rows.append({                      # 一行对应一个块
                "vector": dvec,                # 稠密向量字段
                "sparse_vector": svec,         # 稀疏向量字段
                "text": item["text"],          # 原文
                "summary": item.get("summary", ""),   # 摘要，来自离线增强
                "parent_id": int(item.get("parent_id", -1)),   # 所属父块索引
                "source": item.get("source", ""),     # 来源文件名
                "created_at": now,             # 创建时间戳
                "updated_at": now,             # 修改时间戳
            })                                 # 行组装结束
        client.insert(collection_name=name, data=rows)      # 批量写入 Milvus
        inserted += len(rows)                  # 累加成功条数
        logger.info("写入进度：%d/%d", inserted, total)      # 打印进度日志
    logger.info("写入完成，共 %d 条", inserted)              # 记录完成日志
    return inserted                            # 返回成功条数


def verify_insert(expected: int = 0, wait_seconds: int = 30) -> dict:   # 校验入库结果
    """连接 Milvus 校验入库结果，返回集合名、总行数与前 3 条文本样例。"""
    client = get_milvus_client()               # 获取 Milvus 客户端
    name = config.MILVUS_COLLECTION            # 集合名称
    if not client.has_collection(name):        # 集合不存在
        return {"collection": name, "row_count": 0, "sample": []}   # 返回空结果
    client.load_collection(name)               # 加载集合到内存才能查询
    row_count = 0                              # 初始化行数
    deadline = time.time() + wait_seconds      # 轮询截止时间
    while True:                                # Milvus 是最终一致的，需要轮询等待
        counted = client.query(collection_name=name, filter="", output_fields=["count(*)"])   # 统计
        row_count = int(counted[0].get("count(*)", 0)) if counted else 0   # 取出统计值
        if row_count >= expected or time.time() >= deadline:   # 达到期望值或已超时
            break                              # 结束轮询
        time.sleep(1)                          # 等待 1 秒后重试
    rows = client.query(                       # 查询前 3 条记录
        collection_name=name,                  # 集合名称
        filter="id >= 0",                      # 主键自增非负，等价于全部
        output_fields=["text", "source"],      # 只取需要的字段
        limit=3,                               # 只要 3 条
    )                                          # 查询结束
    sample = [                                 # 组装样例列表
        {"source": r.get("source", ""), "text": (r.get("text", "") or "")[:50]}   # 文本截断到 50 字
        for r in rows                          # 遍历查询结果
    ]                                          # 样例组装结束
    return {"collection": name, "row_count": row_count, "sample": sample}   # 返回校验结果


# ===================== 命令行入口 =====================

def main() -> None:                            # 命令行入口
    """命令行入口：读解析结果 → 写入 Milvus → 校验结果，支持 --rebuild 重建集合。"""
    parser = argparse.ArgumentParser(description="BGE-M3 向量化并写入 Milvus")   # 创建参数解析器
    parser.add_argument("--rebuild", action="store_true", help="先删除集合再重建")   # 重建开关
    args = parser.parse_args()                 # 解析命令行参数
    start = time.time()                        # 记录总耗时起点
    chunks = load_chunks_from_json()           # 读取增强结果并整理（已跳过父块）
    print("=" * 70)                            # 打印分隔线
    print(f"数据来源：{CHUNK_JSON.name} | 待入库子块：{len(chunks)} 条")   # 打印来源与总数
    print("=" * 70)                            # 打印分隔线
    inserted = insert_chunks(chunks, recreate=args.rebuild)   # 写入 Milvus
    result = verify_insert(expected=inserted)  # 校验写入结果，按写入条数轮询等待
    print(f"集合名　：{result['collection']} | 行数：{result['row_count']}")   # 打印集合与行数
    print("前 3 条样例：")                      # 打印小标题
    for i, item in enumerate(result["sample"], 1):        # 逐条打印样例
        print(f"  {i}. [{item['source'][:28]}] {item['text']}")   # 打印来源与文本片段
    print(f"本次写入：{inserted} 条 | 总耗时：{time.time() - start:.1f} 秒")   # 打印写入数与耗时
    logger.info("入库流程结束，写入 %d 条，集合 %s 共 %d 行",
                inserted, result["collection"], result["row_count"])   # 记录结束日志


if __name__ == "__main__":                     # 支持 python -m vector_store 直接运行
    main()                                     # 执行命令行入口
