# -*- coding: utf-8 -*-
"""增量更新知识库：追加/刷新「图形语义块」（组织结构图）。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

说明：完整重建需对 ~2350 个切片重新向量化（约 8 分钟）；图形语义块数量少，
这里做**增量更新**——仅编码新增的图形块并追加到已有索引，秒级完成。
（如需完全重建，直接运行 `build_kb.py --rebuild` 亦可，效果等价。）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np   # noqa: E402

from src import config   # noqa: E402
from src.chunking import Chunk   # noqa: E402
from src.embedding import Embedder, normalize   # noqa: E402
from src.figure_semantics import extract_all_figure_semantics   # noqa: E402
from src.knowledge_base import _tokenize   # noqa: E402


def main() -> None:
    if not config.CHUNKS_PATH.exists():
        raise SystemExit("未找到知识库，请先运行 build_kb.py")

    # 1) 读回已有切片（剔除旧的 figure 块，避免重复）
    chunks = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("kind") == "figure":
                continue
            chunks.append(Chunk(id=d["id"], text=d["text"], source=d["source"],
                                page=d["page"], kind=d["kind"]))
    old_vecs = np.load(config.EMB_PATH)
    if len(chunks) != old_vecs.shape[0]:
        raise SystemExit(f"索引与切片不一致：chunks={len(chunks)} vectors={old_vecs.shape[0]}")

    # 2) 抽取图形语义块
    sems = extract_all_figure_semantics(config.PDF_PATHS)
    cid = max((c.id for c in chunks), default=-1) + 1
    fig_chunks = []
    for s in sems:
        if not s.semantic_text:
            continue
        fig_chunks.append(Chunk(id=cid, text=s.semantic_text, source=s.source,
                                page=s.page, kind="figure"))
        cid += 1
    print(f"新增图形语义块 {len(fig_chunks)} 个")

    # 3) 仅编码新增块并追加
    if fig_chunks:
        vecs = normalize(Embedder().encode([c.text for c in fig_chunks]))
        all_vecs = np.vstack([old_vecs, vecs]).astype(np.float32)
    else:
        all_vecs = old_vecs
    all_chunks = chunks + fig_chunks

    # 4) 落盘
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(config.EMB_PATH, all_vecs)

    meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    meta["chunks"] = len(all_chunks)
    meta["kinds"] = {k: sum(1 for c in all_chunks if c.kind == k)
                     for k in sorted({c.kind for c in all_chunks})}
    config.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"完成：块数 {len(all_chunks)}，维度 {all_vecs.shape}，类型 {meta['kinds']}")
    for c in fig_chunks:
        print(f"  figure id={c.id} {c.source} p.{c.page}: {c.text[:80]}")


if __name__ == "__main__":
    main()