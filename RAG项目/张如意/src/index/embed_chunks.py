# -*- coding: utf-8 -*-
"""BGE-M3 向量化：稠密（1024 维）+ 稀疏（lexical weights）。

输入：data/chunks/chunks.jsonl
输出：data/chunks/chunks_embedded.jsonl —— 每个 chunk 追加
      `dense_vector`（1024 维 float 列表）与 `sparse_vector`（{indices, values}）

## 两路各喂什么（**这里有一处我改了你的口径，见下**）

| 向量 | 喂的字段 | 理由 |
| --- | --- | --- |
| 稠密 | `embed_text + sym_text` | 标题+防治对象+防治适期+药剂名，**不含高危数值**——数字干扰语义相似度 |
| 稀疏 | `bm25_text + sym_text` | **含剂量/安全间隔期**——稀疏向量是拿来**替代 BM25** 的，BM25 的职责正是精确匹配数值（「代森锰锌多少倍」）。若稀疏也不喂数值，它替代不了 BM25，两路都答不了数值问句 |

⚠️ **症状覆盖层（`sym_text`）只在编码这一刻并入两路**：`chunks.jsonl` 里 `embed_text` /
   `bm25_text` 保持卡本体口径，症状**不进基底**——`search.py` 在导入时要用正则扫
   `bm25_text` 现算实体表（兜底第三档「正文提到过」的唯一判据），症状文本是外部二级来源，
   写进基底就等于把「手册里提到过」记成「国标正文提到过」，兜底档位会莫名漂移。

⚠️ **偏离说明**：你写的是「输入：按规则拼接…不喂高危数值」，指向稠密那一路。
   稀疏若照此办理就失去替代 BM25 的能力，所以我让它吃 `bm25_text`。
   若你本意就是两路同源，把 `SPARSE_FIELD` 改成 `"embed_text"` 即可。

## 用法
    python src/index/embed_chunks.py --limit 1     # 冒烟：只跑一条
    python src/index/embed_chunks.py               # 全量
"""
# ============================================================================
# 【本文件在流水线里的位置】索引第 2 步（共 3 步）：
#   build_chunks.py → embed_chunks.py（本文件）→ build_milvus_index.py
# 【做什么】用 BGE-M3 给每个 chunk 生成两路向量：
#   dense_vector  （1024 维，语义相似）   ← 喂 embed_text
#   sparse_vector （token 权重，精确匹配） ← 喂 bm25_text
#   语义型问法走稠密，含数值的问法（「多少倍」）走稀疏
# ============================================================================
import argparse
import json
import os
import sys
import time

# 本地 BGE-M3 权重目录；输入 chunk 文件；输出是追加了两个向量的新文件
DEFAULT_MODEL = r"D:\zg6\bge-m3"        # 本地权重（2.3G，含 sparse_linear.pt）
IN = os.path.join("data", "chunks", "chunks.jsonl")
OUT = os.path.join("data", "chunks", "chunks_embedded.jsonl")
# 稀疏路喂哪个字段——必须是含数值的 bm25_text，否则它替代不了 BM25
SPARSE_FIELD = "bm25_text"              # 见上表；改成 "embed_text" 即两路同源


# 逐行读 JSONL，跳过空行
def load_chunks(path):
    """读取 chunk 文件，返回 dict 列表（跳过空行）。"""
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


# FlagEmbedding 输出的 {token_id: 权重} → 转成向量库通用的 indices/values 格式
def to_sparse(d):
    """FlagEmbedding 的 lexical_weights 是 {token_id: weight}，转成 indices/values。

    用 indices/values 而不是 {token: weight} 是主流向量库（Milvus/Qdrant）的稀疏格式，
    直接能喂；要人读的话用 tokenizer 把 indices 映回 token 即可。
    """
    # 空稀疏向量给空结构，后面自检会统计出来
    if not d:
        return {"indices": [], "values": []}
    # 按 token_id 排序，保证同一份输入每次落盘结果一致
    items = sorted((int(k), float(v)) for k, v in d.items())
    return {"indices": [i for i, _ in items], "values": [v for _, v in items]}


# 主流程：读 chunk → 两路编码 → 自检 → 落盘
def main():
    """编码主流程：读 chunk → BGE-M3 两路编码 → 维度/非空自检 → 追加向量落盘。

    返回退出码：0 成功，1 自检不过，2 缺输入文件或模型目录。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", default=IN)
    ap.add_argument("--output", default=OUT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)   # 编码时的最大 token 截断长度
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条（冒烟用）")
    a = ap.parse_args()

    # 前置检查：chunk 文件与模型权重都在，否则早退
    if not os.path.exists(a.input):
        print(f"缺 {a.input}，先跑 src/index/build_chunks.py"); return 2
    if not os.path.isdir(a.model):
        print(f"缺模型目录 {a.model}"); return 2

    import torch
    from FlagEmbedding import BGEM3FlagModel

    # 自动选设备：有 CUDA 就用 GPU，并开 fp16
    use_cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} ｜ 设备 {'cuda' if use_cuda else 'cpu'}")
    print(f"模型 {a.model}")

    t0 = time.time()
    model = BGEM3FlagModel(a.model, use_fp16=use_cuda, devices="cuda" if use_cuda else "cpu")
    print(f"模型加载 {time.time()-t0:.1f}s")

    # 读入全部 chunk（--limit >0 时只取前 N 条做冒烟）
    chunks = load_chunks(a.input)
    if a.limit:
        chunks = chunks[: a.limit]
    print(f"待编码 {len(chunks)} 个 chunk ｜ batch_size={a.batch_size}\n")

    # 症状覆盖层：只在**编码这一刻**与两路基底拼在一起，chunk 文件里两个字段保持卡本体口径
    def _with_sym(c, base):
        sym = (c.get("sym_text") or "").strip()
        return f"{base} {sym}".strip() if sym else base

    # 稠密路输入 = embed_text（只含语义线索）+ 症状文本
    dense_in = [_with_sym(c, c["embed_text"]) for c in chunks]
    # 稀疏路输入 = bm25_text（含数值）+ 症状文本；字段缺失时退回 embed_text
    sparse_in = [_with_sym(c, c.get(SPARSE_FIELD) or c["embed_text"]) for c in chunks]

    t0 = time.time()
    # 第一趟前向：只取稠密向量
    dense_out = model.encode(dense_in, batch_size=a.batch_size, max_length=a.max_length,
                             return_dense=True, return_sparse=False,
                             return_colbert_vecs=False)["dense_vecs"]
    t_dense = time.time() - t0
    print(f"稠密编码完成 {t_dense:.1f}s ｜ shape={getattr(dense_out, 'shape', None)}")

    t0 = time.time()
    # 第二趟前向：只取稀疏权重（BGE-M3 的 lexical_weights）
    sparse_out = model.encode(sparse_in, batch_size=a.batch_size, max_length=a.max_length,
                              return_dense=False, return_sparse=True,
                              return_colbert_vecs=False)["lexical_weights"]
    t_sparse = time.time() - t0
    print(f"稀疏编码完成 {t_sparse:.1f}s ｜ 第一条非零项 {len(sparse_out[0])} 个")

    # ---- 自检 ----
    # 自检 1：稠密维度必须是 1024（与 Milvus schema 对齐）
    dim = len(dense_out[0])
    if dim != 1024:
        print(f"\n✗ 稠密维度是 {dim}，期望 1024"); return 1
    # 自检 2：不能有 chunk 的稀疏向量为空
    nonzero = sum(1 for s in sparse_out if s)
    if nonzero != len(chunks):
        print(f"\n✗ 有 {len(chunks)-nonzero} 条稀疏向量为空"); return 1
    print(f"\n✓ 稠密维度 {dim} ｜ 稀疏非空 {nonzero}/{len(chunks)}")

    # ---- 落盘 ----
    # 落盘：原 chunk 字段全保留，追加两个向量 + 一份 embedding 元信息
    with open(a.output, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            rec = dict(c)
            # 稠密向量保留 6 位小数，控制文件体积
            rec["dense_vector"] = [round(float(x), 6) for x in dense_out[i]]
            # 稀疏向量转为 indices/values
            rec["sparse_vector"] = to_sparse(sparse_out[i])
            # 记录编码器与字段口径，以后能追溯这一版索引是怎么来的
            rec["embedding_meta"] = {
                "model": "BAAI/bge-m3",
                "dense_dim": dim,
                "dense_source_field": "embed_text + sym_text",
                "sparse_source_field": SPARSE_FIELD + " + sym_text",
                "sparse_type": "lexical_weights",
                "symptom_source_field": "sym_text",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 打印输出文件大小与两路耗时，便于判断要不要调 batch_size
    sz = os.path.getsize(a.output) / 1024 / 1024
    print(f"\n→ {a.output}  {len(chunks)} 行  {sz:.1f} MB")
    print(f"  稠密 {t_dense:.1f}s ｜ 稀疏 {t_sparse:.1f}s ｜ 合计 {time.time()-t0:.1f}s")
    if len(chunks) > 1:
        print(f"  平均 {1000*(t_dense+t_sparse)/len(chunks):.0f} ms/chunk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
