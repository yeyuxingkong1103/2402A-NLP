"""src/offline/pipeline.py —— 离线构建管线总入口（CLI）。

在链路中的位置（新架构离线侧的总编排）：
    目录中的原始文档 → 【本文件】 → 解析 → 分块 → 向量化 → Milvus 入库 + 元数据登记
上游：命令行/运维脚本；下游：src/offline 下的 parsers / chunkers / embedder /
      milvus_store / metadata_store，以及 src/models/database.py 建表

用法（README 里的示例）：
    python -m src.offline.pipeline --role lawyer --input data/samples --chunk semantic --rebuild

正式数据按角色分目录存放：
    data/raw/lawyer/、data/raw/psychologist/

与 backend/pipeline.py 的分工：
    backend 那条主线是"上传一个 PDF → 网页上看进度"的交互式构建；
    本文件是"把一个目录整批灌进某个角色的知识库"的运维式构建，一次处理多份文档。
    两者最终都落到同一个 Milvus 集合结构上，只是入口和使用场景不同。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from configs.settings import get_settings
from src.models.database import init_db
from src.offline.chunkers import build_chunks
from src.offline.embedder import embed
from src.offline.milvus_store import store
from src.offline.metadata_store import create_document, mark_document, replace_chunks
from src.offline.parsers import parse_directory


def build_role(role: str, input_dir: str | Path, strategy: str = "semantic", rebuild: bool = False, tenant_id: str | None = None) -> dict[str, int]:
    """把一个目录下的全部文档构建进某个角色的知识库。

    参数：
        role: 角色 id，向量和元数据都按它隔离（同一份文档可以给不同角色各建一份）
        input_dir: 待处理目录，会递归扫描
        strategy: 分块策略，传 chunkers.build_chunks 支持的五种之一
        rebuild: 是否先删掉该文档的旧向量（同名文档重建时用）
        tenant_id: 租户 id，不传则取配置里的默认值（多租户隔离用）
    返回：
        {"documents": 处理文档数, "chunks": 分块总数, "vectors": 实际入库向量数}

    每份文档的处理顺序（不可调换）：
        1. 建文档记录（拿到 doc.id，后面所有数据都挂在它下面）
        2. 解析 → 分块 → 向量化
        3. 需要重建就先删除旧向量
        4. 写入向量库
        5. 写入分块元数据
        6. 把文档标记为 ready

    为什么"先写向量、再写元数据、最后才标 ready"：
        ready 是给查询侧看的信号。如果顺序反过来，文档刚标成 ready 而向量还没写完，
        检索侧就可能去查一个空文档 —— 表现为"构建完成了但搜不到内容"。
        把 ready 放在最后，能保证"看到 ready 就意味着数据齐了"。

    `chunk.parent_id or 0`：
        向量库要求 parent_id 是数字，而顶层块的 parent_id 是 None，
        用 `or 0` 兜底成 0（约定 0 表示"没有父块"）。
    """
    settings = get_settings()
    tenant_id = tenant_id or settings.tenant_id
    init_db()  # 确保元数据库和表已存在（幂等）
    total = {"documents": 0, "chunks": 0, "vectors": 0}
    # parse_directory 返回 [(文件路径, 按页解析结果)]，逐个文件独立处理
    for path, pages in parse_directory(input_dir):
        # 第一页的 metadata 里带着"用了哪个解析器"，记录下来便于事后排查解析质量问题
        doc = create_document(role, tenant_id, str(path), {"parser": pages[0].get("metadata", {}) if pages else {}})
        chunks = build_chunks(pages, strategy=strategy)
        records = []
        # 一次批量向量化整个文档的所有块，比逐块调用快得多
        embeddings = embed([chunk.content for chunk in chunks])
        now = int(time.time())
        # embed 返回 dense/sparse 两路向量，与 chunks 一一对应（顺序一致，用 zip 配对）
        for chunk, dense, sparse in zip(chunks, embeddings.dense, embeddings.sparse):
            # 各字段都做了长度截断，防止超长文本超出存储字段上限导致整批写入失败
            records.append({"dense_vector": dense, "sparse_vector": sparse, "content": chunk.content[:8192], "summary": chunk.summary[:1024], "parent_id": chunk.parent_id or 0, "doc_id": doc.id, "role_id": role, "tenant_id": tenant_id, "doc_source": str(path)[:512], "page": chunk.page, "create_time": now, "update_time": now, "user_id": ""})
        if rebuild:
            # 只在显式要求重建时才删：默认行为是追加，避免误删已有的其他数据
            store.delete_document(doc.id, role, tenant_id)
        count = store.insert_chunks(records)
        replace_chunks(doc.id, role, tenant_id, [{"content": c.content, "summary": c.summary, "parent_id": c.parent_id, "page": c.page, "metadata": c.metadata} for c in chunks])
        # ready 放在最后一步，保证"标记就绪"等于"数据完整可用"
        mark_document(doc.id, "ready", {"chunks": count, "strategy": strategy})
        total["documents"] += 1
        total["chunks"] += len(chunks)
        total["vectors"] += count
    return total


def main() -> int:
    """命令行入口。

    返回：
        0（成功）。

    四个参数：
        --role    角色 id（必填）
        --input   输入目录（必填）
        --chunk   分块策略，默认 semantic
        --rebuild 是否重建（`action="store_true"` 表示是个开关，不带值）

    打印构建统计而不是静默成功：
        批处理跑完必须能一眼看到"处理了几份、切了多少块、入了多少向量"，
        出现"文档数为 0"这类异常时才能立刻发现是路径写错了。
    """
    parser = argparse.ArgumentParser(description="RAG 离线文档构建管线")
    parser.add_argument("--role", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--chunk", choices=["fixed", "sentence", "paragraph", "heading", "semantic"], default="semantic")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    print(build_role(args.role, args.input, args.chunk, args.rebuild))
    return 0


if __name__ == "__main__":
    # 用 argparse 的 main 作为脚本入口，方便 python -m src.offline.pipeline 直接调用
    raise SystemExit(main())
