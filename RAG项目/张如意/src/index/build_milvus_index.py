# -*- coding: utf-8 -*-
"""把 chunks_embedded.jsonl 灌进 Milvus（稠密 + 稀疏 + 标量过滤）。

## 连接方式：Milvus Lite（嵌入式）
本机没有 Milvus 服务、Docker 守护进程也没起。实测 **milvus-lite 3.2.1 支持
HNSW/COSINE、SPARSE_INVERTED_INDEX、标量 INVERTED、RRFRanker** 全部需求，
且无需任何外部服务——203 条数据用嵌入式版完全够。
换独立部署时只改 `--uri` 即可（`http://host:19530`），schema 不用动。

## 索引
| 字段 | 类型 | 索引 | 度量 |
| --- | --- | --- | --- |
| `dense_vector` | FLOAT_VECTOR(1024) | HNSW | COSINE |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | SPARSE_INVERTED_INDEX | IP |
| `crop` / `std_no` / `knowledge_type` / `type_group` / `pest_kind` / `region` | VARCHAR | INVERTED | — |
| `is_high_risk` / `needs_verification_hint` | BOOL | INVERTED | — |
| `year` | INT64 | INVERTED | — |

## 用法
    python src/index/build_milvus_index.py --limit 1     # 冒烟
    python src/index/build_milvus_index.py               # 全量
    python src/index/build_milvus_index.py --recreate    # 删表重建
"""
# ============================================================================
# 【本文件在流水线里的位置】索引第 3 步（共 3 步），也是最后一步：
#   build_chunks.py → embed_chunks.py → build_milvus_index.py（本文件）
# 【做什么】建表(schema) → 建索引(向量 + 标量倒排) → 分批插入 → 自检实体数
# 【检索时用到的】稠密 HNSW/COSINE + 稀疏 SPARSE_INVERTED/IP 负责召回，
#   11 个标量字段的 INVERTED 索引负责作物/知识类型/附录B 这类过滤
# ============================================================================
import argparse
import json
import os
import sys
import time

# 输入向量文件；Milvus Lite 库文件路径（嵌入式，不需要启动服务）
IN = os.path.join("data", "chunks", "chunks_embedded.jsonl")
# collection 名；检索脚本必须用同一个名字
DEFAULT_URI = os.path.join("data", "index", "milvus_rag.db")   # Milvus Lite 本地文件
COLLECTION = "agri_knowledge"

# 标量字段：(名, 类型, 长度)
# 标量字段清单：(字段名, VARCHAR 最大长度) —— 每个都要建 INVERTED 索引
VARCHAR_FIELDS = [("crop", 16), ("std_no", 32), ("knowledge_type", 32),
                  ("type_group", 16), ("pest_kind", 8), ("region", 32),
                  ("subtype", 64), ("source_section", 32)]
BOOL_FIELDS = ["is_high_risk", "needs_verification_hint"]
INT_FIELDS = ["year"]
# 症状覆盖层字段：存一份供调试/展示，**不建倒排索引**（没有任何路由过滤用它）
RAW_FIELDS = [("sym_text", 2048)]
RAW_BOOL_FIELDS = ["has_symptoms"]


# 读向量文件；--limit 用于冒烟
def load(path, limit=0):
    """逐行读取 chunks_embedded.jsonl，返回 dict 列表。

    path：输入 JSONL 路径；limit>0 时只取前 limit 条（冒烟测试用），0 表示全量。
    """
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


# 稀疏向量格式反向转换：落盘是 indices/values，Milvus 要的是 {token_id: 权重}
def to_sparse(sv):
    """JSON 里的 {indices:[], values:[]} -> Milvus 要的 {token_id: weight}"""
    return {int(i): float(v) for i, v in zip(sv["indices"], sv["values"])}


# 建表结构：主键 + 两路向量 + 标量过滤字段 + 文本字段
def build_schema(client, DataType):
    # auto_id=False：主键用我们自己的 chunk_id；开动态字段以备扩展
    s = client.create_schema(auto_id=False, enable_dynamic_field=True)
    # 主键：chunk_id（如 GBZ26583-2011-p20-c03），与卡片 ID 一致
    s.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    # 稠密向量维度必须与 BGE-M3 输出一致（1024）
    s.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
    # 稀疏向量：BGE-M3 的 lexical weights
    s.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # 逐个登记标量字段（作物/标准号/知识类型/…），供 WHERE 式过滤
    for name, ln in VARCHAR_FIELDS:
        s.add_field(name, DataType.VARCHAR, max_length=ln)
    # 布尔字段：高危标记、核实提示标记
    for name in BOOL_FIELDS:
        s.add_field(name, DataType.BOOL)
    # 整数字段：标准发布年
    for name in INT_FIELDS:
        s.add_field(name, DataType.INT64)
    # 正文检索字段也存进去，BM25 之外的场景/调试要用
    # 两个文本字段也存一份，便于调试与前端展示
    s.add_field("embed_text", DataType.VARCHAR, max_length=2048)
    s.add_field("bm25_text", DataType.VARCHAR, max_length=8192)
    # 症状覆盖层：只存不索引
    for name, ln in RAW_FIELDS:
        s.add_field(name, DataType.VARCHAR, max_length=ln)
    for name in RAW_BOOL_FIELDS:
        s.add_field(name, DataType.BOOL)
    return s


# 建索引：向量路用近似最近邻索引，标量路用倒排索引
def build_index(client):
    idx = client.prepare_index_params()
    # 稠密：HNSW + 余弦距离（M/efConstruction 是建图参数，越大越准也越慢）
    idx.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    # 稀疏：倒排 + 内积（稀疏向量用 IP 度量）
    idx.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX",
                  metric_type="IP")
    # 每个标量字段都建 INVERTED，否则过滤会退化成全表扫描
    for name, _ in VARCHAR_FIELDS:
        idx.add_index(field_name=name, index_type="INVERTED")
    for name in BOOL_FIELDS + INT_FIELDS:
        idx.add_index(field_name=name, index_type="INVERTED")
    return idx


# 把一行 JSONL 转成 Milvus 能插入的记录（字段名与 schema 严格对应）
def record(r):
    """单条 chunk 记录 → Milvus 插入行。

    r：chunks_embedded.jsonl 里的一行（dict）。字段名/类型必须与 build_schema
    严格对应；None 补默认值（空串/False/0），VARCHAR 超长截断。
    """
    m = r["meta"]
    card = r.get("card", {})
    return {
        # 主键 + 两路向量
        "chunk_id": r["chunk_id"],
        "dense_vector": r["dense_vector"],
        "sparse_vector": to_sparse(r["sparse_vector"]),
        # 标量字段：null 统一补空串 / False / 0，Milvus 不接受 None
        "crop": m.get("crop") or "",
        "std_no": m.get("std_no") or "",
        "knowledge_type": m.get("knowledge_type") or "",
        "type_group": m.get("type_group") or "",
        "pest_kind": m.get("pest_kind") or "",
        # 地区：卡片没写就按本批口径填「全国」
        "region": card.get("region") or "全国",
        # VARCHAR 有长度上限，超长截断（上限见 schema）
        "subtype": (m.get("subtype") or "")[:64],
        "source_section": (m.get("source_section") or "")[:32],
        "is_high_risk": bool(m.get("is_high_risk")),
        "needs_verification_hint": bool(m.get("needs_verification_hint")),
        "year": int(card.get("year") or 0),
        # 两个文本字段同样按 schema 上限截断
        "embed_text": (r["embed_text"] or "")[:2048],
        "bm25_text": (r["bm25_text"] or "")[:8192],
        # 症状覆盖层（只存不索引）
        "sym_text": (r.get("sym_text") or "")[:2048],
        "has_symptoms": bool(m.get("has_symptoms")),
    }


# 主流程：连接 → （可选）删表 → 建表建索引 → 打印生效索引 → 批量插入 → 自检
def main():
    """建库主流程。返回退出码：0 成功，1 数量不符/已存在，2 缺输入文件。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=DEFAULT_URI, help="Milvus Lite 文件路径，或 http://host:19530")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--input", default=IN)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--keep", action="store_true", help="只建表不删已有 collection")
    a = ap.parse_args()

    # 前置检查：向量文件必须存在
    if not os.path.exists(a.input):
        print(f"缺 {a.input}，先跑 src/index/embed_chunks.py"); return 2
    # 建好输出目录（Lite 是本地文件，远端是 URL）
    if a.uri.startswith("http"):
        os.makedirs("data/index", exist_ok=True)
    else:
        os.makedirs(os.path.dirname(a.uri) or ".", exist_ok=True)

    from pymilvus import MilvusClient, DataType
    from pymilvus.milvus_client.index import IndexParams  # noqa: F401  (确保依赖可用)

    # 连接 Milvus（Lite 版直接打开本地文件，不依赖任何外部服务）
    print(f"=== 连接 Milvus ===")
    print(f"  uri: {a.uri}" + ("（嵌入式 Milvus Lite，无需服务）" if not a.uri.startswith("http") else ""))
    t0 = time.time()
    client = MilvusClient(a.uri)
    print(f"  ✓ 连接成功 {time.time()-t0:.2f}s")

    # 判断 collection 是否已存在，据此决定是新建还是追加
    exists = client.has_collection(a.collection)
    # --recreate：先删旧表再重建（schema 或语料变了时用）
    if a.recreate and exists:
        client.drop_collection(a.collection)
        print(f"  ✓ 已删除旧 collection「{a.collection}」")
        exists = False
    # 已存在又没让重建：默认提示并退出，防止误叠加重复数据
    if exists:
        if not a.keep and a.limit == 0:
            print(f"  collection「{a.collection}」已存在。用 --recreate 重建，或 --keep 保留后追加。")
            return 1
        print(f"  collection「{a.collection}」已存在，追加插入")
    else:
        t0 = time.time()
        # 新建表 + 建索引一次性完成
        client.create_collection(a.collection, schema=build_schema(client, DataType),
                                 index_params=build_index(client))
        print(f"  ✓ 建表 + 建索引 {time.time()-t0:.2f}s")

    # 打印实际生效的索引清单，便于核对（不靠猜）
    print("\n=== 实际生效的索引 ===")
    for name in sorted(client.list_indexes(a.collection)):
        d = client.describe_index(a.collection, name)
        print(f"  {name:<24} {d.get('index_type'):<24} metric={d.get('metric_type')}")

    rows = load(a.input, a.limit)
    print(f"\n=== 插入 {len(rows)} 条 ===")
    t0 = time.time()
    # 分批插入（batch_size 默认 64），每批打印进度
    for i in range(0, len(rows), a.batch_size):
        client.insert(a.collection, [record(r) for r in rows[i:i + a.batch_size]])
        print(f"  已插入 {min(i+a.batch_size, len(rows))}/{len(rows)}", flush=True)
    print(f"  ✓ 插入完成 {time.time()-t0:.2f}s")

    # 插入完必须 load，collection 才能被检索
    client.load_collection(a.collection)
    # 自检：库内实体数必须等于本次插入条数
    n = client.query(a.collection, filter="", output_fields=["count(*)"])[0]["count(*)"]
    print(f"\n=== 自检 ===")
    print(f"  collection 内实体数: {n}")
    if n != len(rows):
        print(f"  ✗ 数量不符（期望 {len(rows)}）"); return 1
    print(f"  ✓ 数量相符")
    # 打印下一步命令，形成完整流水线闭环
    print(f"\n下一步：python src/retrieve/search.py --query \"黄瓜起腻虫了打什么药\" --crop 黄瓜")
    return 0


if __name__ == "__main__":
    sys.exit(main())
