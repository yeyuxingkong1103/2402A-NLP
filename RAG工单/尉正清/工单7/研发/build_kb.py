# 工单编号：人工智能NLP-RAG-功能测试及评估
"""为 ccf_competition 的 9 份年报建统一知识库

工单7 要求「使用 01-06 任务工单实现的 RAG 系统进行测试」，所以这里不新写检索
逻辑，只调用 01-06 已有的 document.load_pdf / build_chunks 和
vector_store.BGEM3VectorStore —— 解析、分块、向量化全部沿用被测系统本身。

与 01-06 的两点差别，都是为了「测试」这个目标：

1. 9 份文档建**一个**知识库，而不是 9 个分立知识库。
   这批年报覆盖 3 个行业（银行/保险/证券），示例问题里有跨公司比较，
   分成 9 个库就没法互相检索了。每块带一个 doc 字段标明出处，
   报告里据此统计召回都落在了哪几份文档上。

2. 不做图像（图表页）多模态解析。工单4 的那条链路要对每份文档最多 40 页图表
   逐页调多模态大模型，9 份文档要几百次请求，对一个 1 人日的测试工单不划算。
   本工单的 10 个问题都落在正文和财务表格上，用不到图里的信息。
   （这个限制写进了测试报告的「测试范围」一节，不作隐瞒。）

用法：
    python build_kb.py                 # 全量 9 份（约 2800 页）
    python build_kb.py --limit 1       # 只建第一份，快速验证流程
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ccf_docs import DOCS, available, label, resolve        # noqa: E402
from config import CACHE_ROOT                                # noqa: E402
from document import build_chunks, load_pdf                  # noqa: E402
from vector_store import BGEM3VectorStore                    # noqa: E402

KB_NAME = "ccf_competition"
MANIFEST = "docs.json"


def doc_chunks(doc):
    """解析一份年报并切成块，给每块打上 doc 标记。"""
    pdf_path = doc["path"]
    started = time.time()
    pages, tables = load_pdf(pdf_path)
    chunks = build_chunks(pages, tables)
    for chunk in chunks:
        chunk["doc"] = doc["key"]
        chunk["company"] = doc["short"]
        chunk["year"] = doc["year"]
    print(f"[解析] {label(doc)}：{len(pages)} 页正文 + {len(tables)} 张表 "
          f"→ {len(chunks)} 块（{time.time() - started:.1f}s）", flush=True)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只建前 N 份，0 表示全部")
    parser.add_argument("--no-cache", action="store_true", help="忽略已有缓存，强制重建")
    args = parser.parse_args()

    ready = available()
    if args.limit:
        ready = ready[:args.limit]
    if not ready:
        print(f"[错误] 在 {resolve(DOCS[0]).parent} 下没找到任何 PDF，请先解压附件")
        return 1

    out_dir = Path(CACHE_ROOT) / KB_NAME
    if out_dir.exists() and not args.no_cache:
        store = BGEM3VectorStore()
        if store.load(out_dir):
            print(f"[跳过] 知识库已存在且可用：{out_dir}（{len(store.chunks)} 块）。"
                  f"要重建请加 --no-cache")
            return 0

    all_chunks = []
    for doc in ready:
        all_chunks += doc_chunks(doc)

    print(f"[编码] 共 {len(all_chunks)} 块，交给 BGE-M3 ...", flush=True)
    started = time.time()
    store = BGEM3VectorStore()
    store.build(all_chunks)
    print(f"[编码] 完成，耗时 {time.time() - started:.1f}s", flush=True)

    store.save(out_dir)
    manifest = {
        "name": KB_NAME,
        "embedding_model": store.model_name,
        "docs": [{k: (str(v) if isinstance(v, Path) else v)
                  for k, v in d.items() if k != "path"} for d in ready],
        "chunks": len(all_chunks),
    }
    (Path(out_dir) / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 知识库已写入 {out_dir}（{len(all_chunks)} 块，{len(ready)} 份文档）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
