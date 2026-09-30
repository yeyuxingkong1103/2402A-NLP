#!/usr/bin/env python
"""离线入库 CLI：解析 -> 分块 -> （摘要/去重）-> 向量化 -> 写入 Milvus -> 登记 SQL。

用法：
    python scripts/ingest.py --dir data/corpus/lawyer --role lawyer
    python scripts/ingest.py --file data/corpus/psychologist/mental_health_law.md --role psychologist
    python scripts/ingest.py --dir data/corpus --role lawyer --strategy title --summary --re-ingest

前置条件：
    - Milvus、关系库、Ollama（bge-m3 向量化）都要可达。**跑之前必须先停掉 web 服务**：
      Milvus Lite 是单进程独占文件，服务开着会直接 DataDirLockedError 起不来。
    - 不开 --summary 时不需要 LLM；开了就是每个 chunk 一次 LLM 调用，很慢。

副作用与幂等：
    - 整体是只增不删的：会写 Milvus 和关系库 documents 表，但不清库、不动别的角色。
    - 默认幂等：已登记过 source 的文档整篇跳过；--re-ingest 才先删该 source 的旧向量
      与旧登记再重灌，其它文档不受影响。
    - 重灌的删除动作排在向量化之后（见 app/core/document/ingest.py 的说明），
      所以 Ollama 抖动时不会出现「旧的删了、新的没写」。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.document import (  # noqa: E402
    CHUNK_OVERLAP_MAX,
    CHUNK_OVERLAP_MIN,
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_MIN,
    STRATEGIES,
    Chunker,
    DocumentParser,
)
from app.core.document.enhance import filter_low_quality  # noqa: E402
from app.core.document.ingest import build_chunks, store_chunks  # noqa: E402
from app.core.embedding import EmbeddingClient  # noqa: E402
from app.core.llm import LLMClient  # noqa: E402
from app.core.logging_config import get_logger, setup_logging  # noqa: E402
from app.core.store.milvus_store import MilvusStore  # noqa: E402
from app.core.store.sql_store import SQLStore  # noqa: E402

log = get_logger("ingest")


def ingest_path(
    path: Path,
    role_id: str,
    chunk_size: int,
    overlap: int,
    strategy: str,
    re_ingest: bool,
    summary: bool,
    dedup: bool,
) -> int:
    """把单个文件或目录递归入库到指定角色，返回本次实际写入的 chunk 总数。

    跳过的三种情况都不算失败、也不影响退出码：解析器拒收的文件（ValueError）、
    过滤后没有有效文本、以及已登记 source 的文档（默认去重时）。所以「入库 0 条」
    既可能是全都已入库，也可能是文件全被解析器拒了——要看日志区分。
    """
    # 离线链路组件：解析 → 分块 → （摘要/去重）→ 向量化 → 向量库，外加关系库登记
    parser = DocumentParser()                                  # ① 解析：PDF/图片/md/txt → 纯文本
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap, strategy=strategy)  # ② 分块
    embedding = EmbeddingClient(settings)                      # ③ 向量化：Ollama bge-m3 → 1024 维
    milvus = MilvusStore(settings)                             # ④ 向量库：Milvus
    sql = SQLStore(settings)                                   # ⑤ 关系库：登记文档元信息
    sql.connect()
    llm = LLMClient(settings) if summary else None

    if path.is_file():
        files = [path]
    else:
        # 排序只为让入库顺序稳定：chunk_index 与日志顺序随之可复现，
        # 多次跑的日志能直接对比（rglob 本身的顺序依赖文件系统）
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in parser.SUPPORTED_EXTS)

    # 文档级去重：已登记过的 source 直接跳过（除非 --re-ingest 显式重灌）
    # 一次查全量登记、循环里查内存集合：否则每个文件一次 SQL 往返
    registered = set()
    if dedup and not re_ingest:
        registered = {d.source for d in sql.list_documents(role_id)}

    total = 0
    for f in files:
        try:
            doc = parser.parse_file(f)          # ① 解析：文件 → 纯文本
        except ValueError as exc:
            log.warning("跳过 %s: %s", f, exc)
            continue

        if dedup and not re_ingest and doc["source"] in registered:
            log.info("跳过 %s: 已入库（--re-ingest 可重灌）", f.name)
            continue

        texts = chunker.chunk(doc["text"], embed=embedding.embed_texts if strategy == "semantic" else None)
        texts = filter_low_quality(texts, settings.min_chunk_chars)
        if not texts:
            log.warning("跳过 %s: 无有效文本", f)
            continue

        # ③ 先算向量（会联网，可能失败）。放在删除之前：Ollama 抖动时不能把旧数据先删了
        chunks = build_chunks(
            texts,
            source=doc["source"],
            title=doc["title"],
            embedding=embedding,
            summary=summary,
            llm=llm,
            dedup=dedup,
        )

        if re_ingest:
            # 按 source 整篇删（不是按 chunk 差量更新）：文档改过之后新旧 chunk 对不上，
            # 差量更新会留下改前改后的混合版本。粒度就是「一个文件」。
            deleted = milvus.delete_by_source(role_id, doc["source"])
            log.info("重复入库，先删除旧数据 %d 条（source=%s）", deleted, doc["source"])
            # 关系库的旧登记也要清，否则重灌几次就有几条一模一样的记录（实测过）
            sql.dedup_document(role_id, doc["source"])

        # ④ 写向量库 → ⑤ 登记关系库（这一段与接口/seed 共用同一实现）
        count = store_chunks(
            chunks,
            role_id=role_id,
            source=doc["source"],
            title=doc["title"],
            milvus=milvus,
            sql=sql,
        )
        total += count
        log.info("入库 %s -> %d chunks", f.name, count)

    log.info("完成，共入库 %d chunks", total)
    return total


def _bounded_int(name: str, lo: int, hi: int):
    """argparse 的整数类型 + 范围校验。

    用 argparse 自己的报错路径（退出码 2、打印 usage）而不是抛异常：这是命令行入口，
    参数写错应该在动手写 Milvus 之前就停下来，并给出可读的原因。
    """

    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} 必须是整数：{raw!r}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"{name} 必须在 [{lo}, {hi}] 内，收到 {value}")
        return value

    return parse


def main() -> None:
    setup_logging(settings)
    ap = argparse.ArgumentParser(description="RAG 知识库离线入库")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="目录（递归扫描）")
    group.add_argument("--file", help="单个文件")
    ap.add_argument("--role", default="psychologist", help="角色 id")
    # 与 /knowledge/upload 同一组边界，见 app/core/document/chunker.py。
    # 这条离线入口直接写 Milvus，绕过接口校验，所以边界必须在这里也拦一道：
    # 超大的 chunk 会被检索整段塞进系统消息，把 prompt 顶穿上下文。
    ap.add_argument(
        "--chunk-size",
        type=_bounded_int("chunk-size", CHUNK_SIZE_MIN, CHUNK_SIZE_MAX),
        default=500,
        metavar=f"[{CHUNK_SIZE_MIN}-{CHUNK_SIZE_MAX}]",
        help="分块字符数（上界受上下文预算约束）",
    )
    ap.add_argument(
        "--overlap",
        type=_bounded_int("overlap", CHUNK_OVERLAP_MIN, CHUNK_OVERLAP_MAX),
        default=50,
        metavar=f"[{CHUNK_OVERLAP_MIN}-{CHUNK_OVERLAP_MAX}]",
        help="相邻 chunk 重叠字符数",
    )
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="paragraph", help="分块策略")
    # default=None 是**故意**留出的第三态：只有 None（没传这个 flag）才落到 .env 的
    # SUMMARY_ENABLED。store_true 只会产出 True，所以这里无需区分"显式关"；但若把默认
    # 值写成 False，就分不清「用户显式关掉」和「用户没说话」，后者会再也跟随不了全局开关。
    ap.add_argument(
        "--summary",
        action="store_true",
        default=None,
        help="用 LLM 给每个 chunk 生成摘要（费 LLM，慢）；不传则跟随 .env 的 SUMMARY_ENABLED",
    )
    # 去重默认开启，故开关做成反向的 --no-dedup：不加参数就是安全的那一侧
    ap.add_argument("--no-dedup", action="store_true", help="关闭去重（默认按 source + chunk 内容去重）")
    ap.add_argument("--re-ingest", action="store_true", help="重复入库时先按 source 删除旧数据")
    args = ap.parse_args()

    # 没显式传 --summary 就跟随全局开关（.env 的 SUMMARY_ENABLED），与上传接口同一套语义
    summary = args.summary if args.summary is not None else settings.summary_enabled

    path = Path(args.dir or args.file)
    ingest_path(
        path,
        args.role,
        args.chunk_size,
        args.overlap,
        args.strategy,
        args.re_ingest,
        summary,
        not args.no_dedup,
    )


if __name__ == "__main__":
    main()
