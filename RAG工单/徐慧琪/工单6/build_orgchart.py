# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
# 工单编号：人工智能NLP-RAG-混合检索任务
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""把组织结构图结构化块**增量**入库（不重建工单3 的既有索引）。

用法：
    python build_orgchart.py            # 幂等：已入库则跳过
    python build_orgchart.py --rebuild  # 先删该 source 的 orgchart 点再入
    python build_orgchart.py --skip-ocr # 跳过 PaddleOCR-VL 交叉验证

设计要点：
  · 入库逻辑全部收在 `ingest_orgchart()` 里：build_index.py 建完主索引后会**自动
    调用同一个函数**把 recreate 冲掉的 30 个 orgchart 块补回来（F1），两条路径
    共用一份实现，不会分叉；
  · 只 upsert 本任务生成的 30 个 orgchart 块（1 整树 + 29 节点），其余点原样保留；
  · 点 ID 由 `source#chunk_id` 稳定哈希而来（vector_store._stable_id）→ 重跑
    不会产生重复点；--rebuild 先按 source+block_type 精确删除本图旧点；
  · 结构校验（29 节点 / 28 边 / 单根）不过则**拒绝写入**：宁可失败，
    也不把错误的树入进库里静默答错；
  · embedding / Qdrant 调用序列照抄工单3 的 build_index.py（embedder.embed_texts
    → vector_store.upsert_chunks），collection 取 `vector_store.default_name()`
    —— **跟随当前活动嵌入模型**，与 upsert 的目标库严格一致（踩坑记录见
    `ingest_orgchart` 内的注释）。

入口必须走 bootstrap.run_with_large_stack：本脚本触碰 embedder（transformers），
Windows 主线程 1MB 栈会段错误（见 src/bootstrap.py 的踩坑记录）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402  —— 必须最先导入


def _orgchart_filter(source: str):
    """只匹配「该文档的 orgchart 点」的 Qdrant Filter（计数/删除共用）。"""
    from qdrant_client.http import models as qm

    return qm.Filter(must=[
        qm.FieldCondition(key="source", match=qm.MatchValue(value=source)),
        qm.FieldCondition(key="block_type", match=qm.MatchValue(value="orgchart")),
    ])


def _target_collection_name() -> str:
    """orgchart 增量入库的目标库 —— 必须与 `vector_store.upsert_chunks` 的缺省库一致。

    为什么单列一个函数：upsert_chunks 不传 name 时走 `vector_store.default_name()`
    （跟随**活动嵌入模型**）。删除/计数若写死 `config.COLLECTION_OPT`，切模型建库
    （`build_index.py --embedding-model ...`）时就会「删 A 库、写 B 库」——实测把主库
    的 30 个 orgchart 点删掉且没补回（8010 → 7980，id5 证据块丢失）。
    tests/test_orgchart_target.py 钉住这条不变式。
    """
    from src import vector_store

    return vector_store.default_name()


def _orgchart_points(client, name: str, source: str) -> int:
    """该文档已入库的 orgchart 点数（读失败按 0 处理，交给幂等 upsert 兜底）。"""
    try:
        res = client.count(collection_name=name, count_filter=_orgchart_filter(source),
                           exact=True)
        return int(res.count)
    except Exception:  # noqa: BLE001 —— 老版本不支持 count_filter 时退化为 0
        return 0


def ingest_orgchart(rebuild: bool = False, skip_ocr: bool = True) -> int:
    """解析 → 结构闸门 → （OCR 交叉验证）→ 增量入库 → 结构落盘。

    返回 0 成功；1 找不到 PDF；2 结构校验未通过（拒绝写入错树）；
    3 collection 不存在（需先跑 build_index.py）。embedding / Qdrant 异常
    原样抛出：CLI 由 main 冒泡，被 build_index.py 串联时由调用方降级成告警
    （补入失败不得拖垮主索引建库）。
    【注意】本函数不关闭 Qdrant 客户端——调用方（build_index.py）还要接着用它。
    """
    import json

    from src import config, embedder, orgchart_parser, vector_store

    pdf = os.path.join(config.ROOT, config.ORGCHART_SOURCE)
    if not os.path.isfile(pdf):
        print(f"找不到 PDF：{pdf}", file=sys.stderr)
        return 1

    # ---- 1. 解析 + 结构闸门（错了就不写库）----
    chart = orgchart_parser.parse_orgchart(pdf, config.ORGCHART_PAGE_IDX)
    v = chart.verify
    print(f"解析：[{chart.source} p{chart.page_idx}] 节点 {v['n_nodes']} 边 {v['n_edges']} "
          f"根={v['roots']} 单根={v['single_root']}")
    if (not v["single_root"]
            or v["n_nodes"] != orgchart_parser.ORGCHART_EXPECTED_NODES
            or v["n_edges"] != orgchart_parser.ORGCHART_EXPECTED_EDGES):
        print("结构校验未通过，拒绝入库（宁可失败也不写入错树）", file=sys.stderr)
        return 2

    ocr = {"ok": False, "reason": "skipped"}
    if not skip_ocr:
        ocr = orgchart_parser.verify_with_ocr(chart, pdf)
        print("OCR 交叉验证：" + json.dumps(
            {k: val for k, val in ocr.items() if k not in ("matched", "expected")},
            ensure_ascii=False))

    chunks = orgchart_parser.build_chunks(chart)
    print(f"生成块：{len(chunks)}（1 整树 + {len(chunks) - 1} 节点）")

    # ---- 2. 增量入库（照抄 build_index.py 的调用序列）----
    client = vector_store.get_client()
    # 【必须与 upsert 同库】`vector_store.upsert_chunks(chunks, vectors)` 不传 name，
    # 走 `default_name()`（跟随**活动嵌入模型**）。此处曾经写死 `config.COLLECTION_OPT`，
    # 于是 `build_index.py --embedding-model bge-m3-ft-v1` 这类切模型建库时会出现
    # 「删除的是主库、写入的是另一个库」：主库的 30 个 orgchart 点被删掉且没有补回
    # （实测 8010 → 7980，id5 的证据块丢失）。统一取 default_name() 后，两条路径
    # 永远作用于同一个 collection。
    name = _target_collection_name()
    if not vector_store.collection_exists(name):
        print(f"collection {name} 不存在：请先执行 python build_index.py 建库，"
              f"本脚本只做组织结构图的增量入库", file=sys.stderr)
        return 3
    before = vector_store.count(name)
    existing = _orgchart_points(client, name, chart.source)

    do_upsert = True
    if rebuild:
        if existing:
            client.delete(collection_name=name,
                          points_selector=_orgchart_filter(chart.source), wait=True)
            print(f"已删除旧的 orgchart 点：{existing}")
        else:
            print("无旧 orgchart 点，无需删除")
    elif existing == len(chunks):
        do_upsert = False
        print(f"已入库 {existing} 点，跳过（--rebuild 可强制重入）")

    if do_upsert:
        vectors = embedder.embed_texts([c["text"] for c in chunks], show_progress=True)
        print(f"向量 {len(vectors)} 条（dim={embedder.get_dim()}）")
        written = vector_store.upsert_chunks(chunks, vectors)
        after = vector_store.count(name)
        print(f"入库 {written} 点 | collection {name}：{before} → {after}")

    # ---- 3. 结构落盘（供答案定位报告/专项说明复用）----
    out = os.path.join(config.TABLE_DIR, "招股说明书2_orgchart.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"verify": v, "ocr": ocr,
                   "nodes": {k: {"bbox": n.bbox, "parent": n.parent,
                                 "children": n.children, "depth": n.depth}
                             for k, n in chart.nodes.items()}},
                  fh, ensure_ascii=False, indent=1)
    print(f"结构落盘：{out}")
    return 0


def main() -> int:
    import argparse

    from src import vector_store

    ap = argparse.ArgumentParser(description="工单05 组织结构图增量入库")
    ap.add_argument("--rebuild", action="store_true",
                    help="先删除该文档已入库的 orgchart 点再重新写入")
    ap.add_argument("--skip-ocr", action="store_true",
                    help="跳过 PaddleOCR-VL 交叉验证（仅影响结构落盘的 ocr 字段）")
    args = ap.parse_args()

    rc = ingest_orgchart(rebuild=args.rebuild, skip_ocr=args.skip_ocr)
    if rc == 0:
        vector_store.close()      # 本地嵌入式模式：显式释放存储锁
    return rc


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
