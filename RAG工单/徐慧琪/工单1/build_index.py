# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
脚本：一键构建知识库（解析 -> 分块 -> 向量化 -> 入 Qdrant）

用法：
    python build_index.py              # 全流程（已建过索引则跳过向量化，除非 --rebuild）
    python build_index.py --rebuild    # 删除 collection 全量重建
    python build_index.py --skip-parse # 复用已有 content_list.json
    python build_index.py --limit 50   # 只处理前 50 页（调试）

对应工单第三步：向量化 + Qdrant 入库
  - 用本机 embedding 模型（local_files_only=True）
  - VectorParams(size=模型维度, distance=COSINE)
  - payload: text、page_idx、source
  - 批量写入 batch_size=100
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

from src import chunker, config, embedder, pdf_parser, vector_store


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 RAG 知识库")
    ap.add_argument("--pdf", default=config.SOURCE_PDF)
    ap.add_argument("--rebuild", action="store_true", help="删除并重建 collection")
    ap.add_argument("--skip-parse", action="store_true", help="复用已解析结果")
    ap.add_argument("--skip-embed", action="store_true", help="仅分块，不做向量化")
    ap.add_argument("--limit", type=int, default=None, help="只解析前 N 页（调试）")
    args = ap.parse_args()

    t_all = time.time()
    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    md_path = os.path.join(config.PARSED_DIR, f"{stem}.md")
    cl_path = os.path.join(config.PARSED_DIR, f"{stem}_content_list.json")
    ck_path = os.path.join(config.CHUNK_DIR, "chunks.json")

    # ---------------- 1) PDF 解析 ----------------
    if not args.skip_parse or not os.path.isfile(cl_path):
        _banner("第 1 步 / 4：解析 PDF")
        content, markdown = pdf_parser.parse_pdf(args.pdf, limit_pages=args.limit)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(markdown)
        with open(cl_path, "w", encoding="utf-8") as fh:
            json.dump(content, fh, ensure_ascii=False, indent=1)
        n_tab = sum(1 for c in content if c["type"] == "table")
        print(f"  解析完成：{len(content)} 块（表格 {n_tab}）-> {cl_path}")
    else:
        _banner("第 1 步 / 4：解析 PDF（跳过，复用已有结果）")
        with open(cl_path, encoding="utf-8") as fh:
            content = json.load(fh)
        print(f"  复用 {cl_path}（{len(content)} 块）")

    # ---------------- 2) 分块 ----------------
    _banner("第 2 步 / 4：分块")
    chunks = chunker.chunk_content_list(content)
    os.makedirs(os.path.dirname(ck_path), exist_ok=True)
    with open(ck_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=1)
    st = chunker.summarize(chunks)
    print(f"  {st['total']} 块（正文 {st['text']} / 表格 {st['table']}），"
          f"平均 {st['avg_chars']} 字符 -> {ck_path}")

    if args.skip_embed:
        print("\n已指定 --skip-embed，流程结束。")
        return 0

    # ---------------- 3) 向量化 ----------------
    _banner("第 3 步 / 4：向量化（本地 bge-m3）")
    dim = embedder.get_dim()
    print(f"  模型: {config.EMBEDDING_MODEL_PATH}")
    print(f"  维度: {dim} | 设备: {embedder.info()['device']} | local_files_only=True")

    t0 = time.time()
    vectors = embedder.embed_texts([c["text"] for c in chunks],
                                   show_progress=False)
    embed_s = time.time() - t0
    print(f"  编码完成：{len(vectors)} 条，用时 {embed_s:.1f}s "
          f"（{len(vectors) / max(embed_s, 1e-6):.0f} 块/秒）")

    # ---------------- 4) 入 Qdrant ----------------
    _banner("第 4 步 / 4：写入 Qdrant")
    vector_store.ensure_collection(dim, recreate=args.rebuild)
    # 重建或空库时才需要重写；否则先清空同名 collection 避免重复点
    if not args.rebuild and vector_store.count() > 0:
        print("  检测到已有数据，重建 collection 以保证一致性 …")
        vector_store.ensure_collection(dim, recreate=True)

    n = vector_store.upsert_chunks(chunks, vectors,
                                   batch_size=config.UPSERT_BATCH_SIZE)
    info = vector_store.info()
    print(f"  写入 {n} 点 | collection={info['collection']} "
          f"| 当前点数={info['points']} | 模式={info['mode']}")
    if info["points"] != n:
        # 本地模式的历史脏数据会表现为"点数 > 写入数"，必须显式报警而不是静默通过
        print(f"  ⚠ 点数与写入数不一致（差 {info['points'] - n}），"
              f"可能存在旧数据残留，请执行 --rebuild 全量重建")

    vector_store.close()

    _banner(f"完成，总耗时 {time.time() - t_all:.1f}s")
    print(f"  知识库位置: {config.QDRANT_LOCAL_PATH if config.QDRANT_MODE == 'local' else config.QDRANT_URL}")
    print(f"  下一步    : streamlit run app.py")
    return 0


if __name__ == "__main__":
    # 放进大栈线程执行：Windows 主线程栈只有约 1MB，深层导入链会把它撑爆
    # （表现为进程直接消失、Python 层无法捕获）。见 src/bootstrap.py 说明。
    sys.exit(bootstrap.run_with_large_stack(main))
