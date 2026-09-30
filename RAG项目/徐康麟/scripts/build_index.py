# -*- coding: utf-8 -*-
"""构建知识库索引（离线链路入口）：``PDF -> 分块 -> Ollama bge-m3 -> Milvus``。

用法
----
    # 真实后端（本阶段主线）：Ollama 嵌入 + Milvus 向量库
    python scripts/build_index.py --source knowledge/ --provider milvus --embedding ollama --rebuild

    # 角色知识：knowledge/<role_id>/ 会**自动按角色分区入库**
    # （knowledge/lawyer/ -> role_id=lawyer，其它角色检索不到，反之亦然）
    python scripts/build_index.py --source knowledge/lawyer --provider milvus --embedding ollama

    # 显式指定角色（把所有来源都打上同一角色）
    python scripts/build_index.py --source knowledge/lawyer --role-id lawyer

    # 加一份中文 PDF 一起入，并当场做一次检索命中自检
    python scripts/build_index.py --source knowledge/ --source uploads/x.pdf ^
        --provider milvus --embedding ollama --check-retrieval "民间借贷利率上限"

    # 维度不匹配时允许重建本项目专用 collection（⚠️ 会删该 collection 数据）
    python scripts/build_index.py --source knowledge/ --provider milvus --recreate-collection

    # 一键离线兜底（不连任何真实后端）
    python scripts/build_index.py --source knowledge/ --offline --rebuild

幂等性
------
不带 ``--rebuild`` 连续跑两次：第二次全部命中 MD5 指纹，输出「新增 0 / 跳过 N」，
Milvus 里向量总数不变。

⚠️ 注意：指纹只算文件 MD5，**不含 role_id**。改了角色归属必须加 ``--rebuild``，
否则会被判定为「md5 未变」而跳过，角色标签不会更新。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# 允许 `python scripts/build_index.py` 直接运行时找到 legal_rag 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.config import KNOWLEDGE_DIR, RagConfig  # noqa: E402
from legal_rag.embedding.base import build_embedder  # noqa: E402
from legal_rag.ingest.pipeline import KnowledgeBasePipeline, group_sources_by_role  # noqa: E402
from legal_rag.logging_setup import get_logger, setup_logging  # noqa: E402
from legal_rag.roles import DEFAULT_ROLE_ID  # noqa: E402
from legal_rag.store.base import build_store  # noqa: E402

logger = get_logger("scripts.build_index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建法律知识库索引")
    parser.add_argument("--source", action="append", default=None,
                        help="待入库的文件或目录，可重复；默认 knowledge/")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：内存向量库 + 零依赖向量化 + 余弦重排")
    parser.add_argument("--provider", default=None, choices=["memory", "chroma", "milvus"],
                        help="向量库后端（覆盖配置）；真实后端用 milvus")
    parser.add_argument("--embedding", default=None, choices=["offline", "bge_m3", "ollama"],
                        help="向量化后端（覆盖配置）；本阶段真实后端用 ollama（bge-m3）")
    parser.add_argument("--collection", default=None,
                        help="Milvus collection 名（覆盖 MILVUS_COLLECTION）")
    parser.add_argument("--recreate-collection", action="store_true",
                        help="[注意] 维度不匹配时允许 DROP 并重建本项目专用 collection（会丢该表数据）")
    parser.add_argument("--no-milvus-fallback", action="store_true",
                        help="Milvus 连不上时不要降级为内存库，直接报错退出")
    parser.add_argument("--strategy", default=None,
                        choices=["fixed", "sentence", "paragraph", "article"],
                        help="分块策略；article=法条感知（按「第X条」一条一块，默认）")
    parser.add_argument("--chunk-size", type=int, default=None, help="子块目标字符数")
    parser.add_argument("--overlap", type=int, default=None, help="子块之间重叠字符数")
    parser.add_argument("--parent-size", type=int, default=None, help="父块目标字符数")
    parser.add_argument("--role-id", default="", help="把这些知识归属到某个角色（分区隔离用）")
    parser.add_argument("--rebuild", action="store_true", help="忽略 MD5 指纹，全量重建")
    parser.add_argument("--check-retrieval", default=None, metavar="QUERY",
                        help="入库后立刻用该查询做一次向量检索自检，打印命中的文本片段")
    parser.add_argument("--check-top-k", type=int, default=3, help="自检返回条数（默认 3）")
    return parser


def _configure_retrieval_probe(store, embedder, query: str, top_k: int) -> int:
    """入库后自检：真嵌入一次查询词 -> 走向量库检索 -> 打印命中。"""
    print("-" * 60)
    print(f"检索自检：{query!r}（top_k={top_k}）")
    started = None
    import time as _time
    started = _time.perf_counter()
    vector = embedder.embed_query(query)
    hits = store.search(vector, top_k=top_k, where={"is_parent": False})
    elapsed = (_time.perf_counter() - started) * 1000.0
    print(f"  查询向量维度 {len(vector)}，命中 {len(hits)} 条，耗时 {elapsed:.1f}ms")
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.chunk.text.split())[:80]
        print(f"  [{rank}] score={hit.score:.4f} source={hit.chunk.source} :: {snippet}")
    if not hits:
        print("  [注意] 没有任何命中：请确认入库成功、collection 名与维度一致（见上方日志）")
        return 1
    return 0


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    if args.offline:
        os.environ["RAG_OFFLINE"] = "1"
    if args.recreate_collection:
        # config.py 已冻结，这里通过环境变量打开「维度不匹配允许重建」开关
        os.environ["MILVUS_RECREATE_ON_DIM_MISMATCH"] = "true"

    config = RagConfig.from_env()
    config.ensure_dirs()

    if args.provider:
        config.vector_store = args.provider
    if args.embedding:
        config.embedding_provider = args.embedding
    elif config.vector_store.lower() in ("milvus", "milvus_grpc", "zilliz") \
            and config.embedding_provider.lower() in ("offline", "hash", "fallback"):
        # 真实向量库 + 兜底哈希向量是「维度对不上」的组合（兜底 256 维 vs
        # bge-m3 1024 维的 collection 名），因此连真实 Milvus 时默认改用 Ollama
        # bge-m3；要复现旧行为显式加 --embedding offline。
        config.embedding_provider = "ollama"
        print("[提示] --provider milvus 未指定 --embedding，默认改用 ollama/bge-m3"
              "（真实嵌入后端）；如需兜底哈希向量请显式 --embedding offline")
    if args.collection:
        config.milvus.collection = args.collection
    if args.strategy:
        config.chunk.strategy = args.strategy
    if args.chunk_size is not None:
        config.chunk.chunk_size = args.chunk_size
    if args.overlap is not None:
        config.chunk.overlap = args.overlap
    if args.parent_size is not None:
        config.chunk.parent_size = args.parent_size

    sources = args.source or [str(KNOWLEDGE_DIR)]

    print(f"向量库后端 : {config.vector_store}")
    print(f"向量化后端 : {config.embedding_provider}")
    print(f"Ollama     : {config.ollama.base_url}（模型 {config.ollama.embedding_model}）")
    if config.vector_store.lower() in ("milvus", "milvus_grpc", "zilliz"):
        print(f"Milvus     : {config.milvus_uri} collection={config.milvus.collection} "
              f"metric={config.milvus.metric_type} recreate_on_mismatch="
              f"{config.milvus.recreate_on_dim_mismatch}")
    print(f"分块策略   : {config.chunk.strategy} (chunk_size={config.chunk.chunk_size}, "
          f"overlap={config.chunk.overlap}, parent_size={config.chunk.parent_size})")
    print(f"索引目录   : {config.index_dir}")
    print(f"知识来源   : {', '.join(sources)}")
    print("-" * 60)

    model_hint = (config.ollama.embedding_model
                  if config.embedding_provider.lower() in ("ollama", "ollama_bge_m3")
                  else config.embedding_model)
    embedder = build_embedder(config.embedding_provider, model_hint,
                              config.embedding_dim, config=config)
    print(f"嵌入后端已就绪: {embedder.name}（model={getattr(embedder, 'model', model_hint)}）")

    dim = 0
    if config.vector_store.lower() in ("milvus", "milvus_grpc", "zilliz"):
        try:
            dim = embedder.ensure_dim()      # 维度只能从真实响应实测取得
        except Exception as exc:  # noqa: BLE001 - 换成可执行的修复提示再抛
            raise SystemExit(
                f"\n[错误] 无法向嵌入后端实测向量维度（embedder={embedder.name}）："
                f"{type(exc).__name__}: {exc}\n"
                f"修复建议：\n"
                f"  1) 确认本机 Ollama 在跑：curl {config.ollama.base_url}/api/tags\n"
                f"  2) 确认已拉取嵌入模型：ollama pull {config.ollama.embedding_model}\n"
                f"  3) 或改用兜底后端（仅用于连通性验证）：--embedding offline"
            ) from exc
        print(f"实测嵌入维度 : {dim}（不取代码写死值；bge-m3 = 1024）")

    store = build_store(config.vector_store, persist_dir=config.index_dir,
                        collection=config.milvus.collection, config=config,
                        dim=dim, embedder=embedder,
                        fallback=not args.no_milvus_fallback)
    if getattr(store, "primary", None) == "milvus":
        print(f"[注意] 向量库已降级为 {store.name}（原因：{getattr(store, 'fallback_reason', '')}）")
    else:
        print(f"向量库已就绪: {store.name}"
              + (f" collection={getattr(store, 'collection', '')}"
                 f" dim={getattr(store, 'dim', 0)}"
                 if store.name == "milvus" else ""))
    print("-" * 60)

    pipeline = KnowledgeBasePipeline(embedder, store, config)
    if args.role_id:
        groups: list[tuple[str, list[str]]] = [(args.role_id, sources)]
    else:
        # 未显式指定角色时按 knowledge/<role_id>/ 目录分组：
        # 一把入库「无归属」的语料是安静的隔离漏洞（任何角色都看不见它），
        # 所以这里沿用服务启动时的同一条规则，未标注来源回退默认角色并告警。
        groups = group_sources_by_role(sources, default_role=DEFAULT_ROLE_ID)
        print(f"角色分组入库 : {[(role or '(未标注)', len(paths)) for role, paths in groups]}")

    print("-" * 60)
    total_chunks = total_skipped = total_failed = total_docs = 0
    for role_id, paths in groups:
        report = pipeline.ingest(paths, rebuild=args.rebuild, role_id=role_id)
        total_chunks += report.chunks
        total_skipped += report.skipped
        total_failed += report.failed
        total_docs += report.documents
        print(f"[role_id={role_id or '(未标注)'}] {report.summary()}")
        for item in report.files:
            print(f"  [{item['status']}] {item.get('role_id', role_id)} | {item['file']}"
                  + (f" -> {item.get('chunks')} chunks" if item.get("chunks") else "")
                  + (f" ({item.get('reason')})" if item.get("reason") else ""))
    print(f"合计       : 文件 {total_docs} / 新增 chunk {total_chunks} / "
          f"跳过 {total_skipped} / 失败 {total_failed}")
    stats = pipeline.stats()
    print(f"索引内 chunk 总数: {stats['chunks_in_store']}，已登记文档: {stats['documents']}")

    status = 0 if total_failed == 0 else 1
    if args.check_retrieval:
        status = max(status, _configure_retrieval_probe(
            store, embedder, args.check_retrieval, args.check_top_k))
    return status


if __name__ == "__main__":
    logging.getLogger(__name__).debug("build_index 启动参数：%s", sys.argv[1:])
    sys.exit(main())
