"""bge-m3 双向量编码。

存在的理由（技术方案 4.3）：dense 与 sparse 必须来自**同一次前向**——跑两遍模型
既费显存又费时间，而 bge-m3 的词权重本就是同一次前向的副产物。

模型固定用原版 D:\\Model\\bge-m3：同目录下的 bge-m3-ft-v1 / -r32 缺 sparse_linear.pt，
FlagEmbedding 找不到该文件会**随机初始化** sparse 头（实测同一文本两次加载得到
[17,17] 与 [10,10]），会让混合检索的稀疏路灌入噪声。详见设计文档 4.3。

编码走 CPU：实测原版 2.1 块/秒，3388 块约 27.5 分钟，可接受；不为它改 torch 装 CUDA。
"""
from __future__ import annotations

import json
import pathlib
import re

# 环境规避（pandas 先于 sklearn）：与精排侧共用同一份实现，
# 不能依赖 db.milvus → pymilvus 顺手带进 pandas 这个没人记账的巧合
from app.compat import ensure_import_order
from app.db.milvus import entity_count, ensure_collection, get_client, insert_chunks
from app.db.mysql import connect
from app.ingest.load_mysql import LAW_ID

import numpy as np

DEFAULT_MODEL_PATH = r"D:\Model\bge-m3"

# 批大小取 16：CPU 上再大收益有限，而 8GB 显存的目标机上 16 也留有余量
ENCODE_BATCH = 16


def to_sparse_dict(raw) -> dict[int, float]:
    """把 FlagEmbedding 的 defaultdict{token_id: weight} 转成 Milvus 要的 {int: float}。

    必须滤掉零权重项：Milvus 的稀疏倒排索引会为每个 key 建 posting，
    权重 0 的项占空间又永不命中。转换后为空要抛错——那说明该文本的稀疏路
    彻底失效，属于数据异常，硬门槛要求中断而不是静默跳过。
    """
    out = {int(k): float(v) for k, v in raw.items() if float(v) > 0}
    if not out:
        raise ValueError("sparse 转换后为空：该文本的稀疏检索路会失效")
    return out


def normalize_dense(vecs) -> np.ndarray:
    """L2 归一化。技术方案 4.3：保证 COSINE 与内积一致。

    全零向量会让除法产生 nan，nan 一旦写进 Milvus 后续极难排查，故当场拦下。
    """
    arr = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("出现全零 dense 向量，无法归一化")
    return arr / norms


def load_model(model_path: str = DEFAULT_MODEL_PATH, use_fp16: bool = False):
    """加载 bge-m3。use_fp16 默认关：本机 torch 是 CPU 版，fp16 在 CPU 上更慢。"""
    # 必须先于 FlagEmbedding 调用：它内部"先 torch 后 sklearn"会踩中本机
    # 环境缺陷，此前不崩只是因为 pymilvus 顺手带了 pandas，属未记账的巧合
    ensure_import_order()
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(model_path, use_fp16=use_fp16, device="cpu")


def encode_texts(model, texts: list[str]) -> list[tuple[list[float], dict[int, float]]]:
    """一次前向同时取 dense 与 sparse，返回与输入等长的 (dense, sparse) 列表。

    顺序与条数必须与输入严格对齐：调用方按位置拼装，错位会静默把 A 条的向量
    写到 B 条上。故此处断言条数。
    """
    out = model.encode(texts, batch_size=ENCODE_BATCH,
                       return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense = normalize_dense(out["dense_vecs"])
    sparse = [to_sparse_dict(w) for w in out["lexical_weights"]]
    if len(dense) != len(texts) or len(sparse) != len(texts):
        raise ValueError(f"编码条数不符：输入 {len(texts)}、"
                         f"dense {len(dense)}、sparse {len(sparse)}")
    return list(zip([d.tolist() for d in dense], sparse))


def build_milvus_row(chunk: dict, meta: dict, dense: list, sparse: dict) -> dict:
    """拼装一行 Milvus 数据。

    分工：块级字段（款号/项号/父指针/正文/类型）来自 law_chunks.jsonl，
    条级元数据（law_id / 版本 / 效力 / 生效日 / 指纹）来自 MySQL——这是
    设计文档 4.1 定的口径，两者在此汇合成集合需要的一整行。
    """
    return {
        "chunk_id": chunk["chunk_id"],
        "dense": list(dense),
        "sparse": sparse,
        "law_id": meta["law_id"],
        "law_version": meta["law_version"],
        "article_no": chunk["article_no"],
        "article_no_cn": meta["article_no_cn"],
        "paragraph_no": chunk["paragraph_no"],
        "item_no": meta.get("item_no"),
        "path": chunk["path"],
        "status": meta["status"],
        "effective_date": meta["effective_date"],
        "parent_id": chunk["parent_id"],
        "chunk_type": chunk["chunk_type"],
        "source_hash": meta["source_hash"],
        "text": chunk["text"],
    }


# 项号前缀：「（一）」或「(一)」。只认中文数字，避免把「（以下称）」误判成项
ITEM_PREFIX = re.compile(r"^[（(]([一二三四五六七八九十]+)[）)]")

CHUNKS_PATH = pathlib.Path(__file__).resolve().parents[3] / "data" / "parsed" / "law_chunks.jsonl"

# 每条法条查一次库换成本次全量取回：1260 行、几百 KB，
# 比 3388 次单条查询少 3388 次往返。
# JOIN law_version 取版本/效力/生效日：本期只有一行版本，但多版本时这里不必改。
# 此处是**唯一**一处元数据 SQL——article_meta 直接执行它，
# 不再往里内联第二份，否则改这里会毫无效果。
META_SQL = """
SELECT a.article_no, a.article_no_cn, a.source_hash, v.law_version, v.status, v.effective_date
FROM article a JOIN law_version v ON a.law_id = v.law_id
WHERE a.law_id = %s
"""


def parse_item_no(text: str) -> str | None:
    """从项块正文取出光杆中文数字（存「一」而非「（一）」）。

    解析不出返回 None 而不抛错：chunk.py 判项用的是同一套前缀正则，
    个别正文以括号数字开头却并非项号的情况不值得中断整条入库。
    """
    m = ITEM_PREFIX.match(text.strip())
    return m.group(1) if m else None


def article_meta(conn, law_id: str = LAW_ID) -> dict[int, dict]:
    """取全部条级元数据，返回 {条号: 元数据}。

    law_version / status / effective_date 来自 law_version 表，与 article 表做一次
    交叉连接——本期只有一行版本，但它决定了将来多版本时不必改这里。
    """
    with conn.cursor() as cur:
        cur.execute(META_SQL, (law_id,))
        rows = cur.fetchall()
    meta = {}
    for art_no, art_no_cn, sha, version, status, eff in rows:
        meta[int(art_no)] = {
            "law_id": law_id, "law_version": version, "status": status,
            "effective_date": str(eff), "source_hash": sha, "article_no_cn": art_no_cn,
        }
    return meta


def build_rows(chunks: list[dict], meta_by_no: dict[int, dict], model) -> list[dict]:
    """把块列表转成 Milvus 行列表。编码按批进行，批间打印进度。

    每块的 item_no 在条级元数据基础上按块补齐——它属于块级信息，
    不进 MySQL 的 article 表（那张表本期只写条级）。
    """
    rows: list[dict] = []
    for start in range(0, len(chunks), ENCODE_BATCH):
        batch = chunks[start:start + ENCODE_BATCH]
        encoded = encode_texts(model, [c["text"] for c in batch])
        for chunk, (dense, sparse) in zip(batch, encoded):
            # KeyError 是刻意的：元数据缺失说明 ETL 没跑或条号对不上，
            # 属于硬门槛，不能降级成空值继续
            meta = dict(meta_by_no[chunk["article_no"]])
            meta["item_no"] = (parse_item_no(chunk["text"])
                               if chunk["chunk_type"] == "item" else None)
            rows.append(build_milvus_row(chunk, meta, dense, sparse))
        print(f"  编码进度 {min(start + ENCODE_BATCH, len(chunks))}/{len(chunks)}", flush=True)
    return rows


def main(conn=None, client=None, chunks_path: pathlib.Path = CHUNKS_PATH) -> dict[str, int]:
    """端到端：读块 → 查元数据 → 编码 → 写 Milvus。返回统计。

    条数校验放在写库之前：编码条数 ≠ 输入条数说明有块被静默丢弃，
    这时还没写库，中断的代价最小。
    """
    own_conn, own_client = conn is None, client is None
    if own_conn:
        conn = connect()
    if own_client:
        client = get_client()
    try:
        with open(chunks_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        meta_by_no = article_meta(conn)
        ensure_collection(client)
        model = load_model()
        rows = build_rows(chunks, meta_by_no, model)
        if len(rows) != len(chunks):
            raise ValueError(f"组装条数不符：输入 {len(chunks)}、产出 {len(rows)}")
        written = insert_chunks(client, rows)
        stats = entity_count(client)
        return {"chunks": len(chunks), "meta": len(meta_by_no),
                "written": written, "rows": stats}
    finally:
        if own_client:
            client.close()
        if own_conn:
            conn.close()


if __name__ == "__main__":
    print(f"灌库完成：{main()}")
