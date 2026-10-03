# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
"""
多文档入库：按 `app/core/doc_profiles.py` 里的配置逐份 PDF **追加入库**。

  python scripts/ingest_all.py                  # 入库所有已配置但还没入库的文档
  python scripts/ingest_all.py --only liyuan    # 只入库某一份（按 key）
  python scripts/ingest_all.py --status         # 只看库里现在有哪些文档
  python scripts/ingest_all.py --rebuild --yes  # 清库重灌（危险，见下）

【为什么默认是"追加"而不是"重建"】
`--rebuild` 会 **drop 整个 collection**，把**所有**文档一起删掉 —— 不是只删你指定的
那一份。工单02 交付后库里已有招股书1 的 1116 条，误用一次就得重新嵌入全部内容。
所以这里把 `--rebuild` 做成必须显式加 `--yes` 才执行，并且在执行前打印将丢失多少行。

【为什么入库前先打印行数】
解析/分块/嵌入这条链上任何一环出错，最终表现都是"库里少了几百条"，
而不是报错。先记下入库前的行数，入库后复查增量，才能确认真的写进去了。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                          # noqa: E402
from app.core.doc_profiles import DOC_PROFILES                           # noqa: E402


def _raw(name: str) -> Path:
    return settings.data_path / "raw" / name


def show_status() -> int:
    from app.core.vectorstore import VectorStore
    store = VectorStore()
    ok, msg = store.health()
    print(f"Milvus：{msg}" if ok else f"[错误] Milvus 不可用：{msg}")
    if not ok:
        return 1
    total = store.count()
    print(f"collection `{store.collection}` 共 {total} 行\n")
    print(f"{'文档':<24}{'块数':>8}{'doc_id':>18}  文件在否")
    seen: dict[str, int] = {}
    try:
        rows = store.client.query(
            collection_name=store.collection, filter="id >= 0",
            output_fields=["doc_name", "doc_id"], limit=16384)
        for r in rows:
            key = (r.get("doc_name", "?"), r.get("doc_id", "?"))
            seen[key] = seen.get(key, 0) + 1
    except Exception as e:  # noqa: BLE001
        print(f"  （按文档统计失败：{e}）")
    for (name, did), n in sorted(seen.items()):
        exists = "✅" if _raw(name).exists() else "❌ 文件不在"
        print(f"{name:<24}{n:>8}{did:>18}  {exists}")
    return 0


async def ingest_one(key: str, limit_pages: int | None = None) -> dict:
    from app.core.pipeline import IngestPipeline

    prof = DOC_PROFILES[key]
    pdf = _raw(prof.doc_name)
    if not pdf.exists():
        return {"key": key, "error": f"PDF 不存在：{pdf}"}

    pipe = IngestPipeline()
    seen: list[str] = []

    async def run():
        task = asyncio.create_task(
            pipe.run(pdf, rebuild=False, limit_pages=limit_pages))
        while not task.done():
            st = pipe.status()
            if not seen or st.stage != seen[-1]:
                seen.append(st.stage)
                print(f"    [{st.stage}] {st.message}", flush=True)
            await asyncio.sleep(1.5)
        return await task

    st = await run()
    return {"key": key, "status": st}


def main() -> int:
    ap = argparse.ArgumentParser(description="工单03 · 多文档追加式入库")
    ap.add_argument("--only", default=None,
                    help=f"只入库某一份：{'/'.join(DOC_PROFILES)}")
    ap.add_argument("--status", action="store_true", help="只打印库里的文档分布")
    ap.add_argument("--limit-pages", type=int, default=None, help="只解析前 N 页")
    ap.add_argument("--rebuild", action="store_true",
                    help="⚠️ 清空整个 collection 后重灌（会删掉所有文档）")
    ap.add_argument("--yes", action="store_true", help="配合 --rebuild，确认执行")
    args = ap.parse_args()

    if args.status:
        return show_status()

    from app.core.vectorstore import VectorStore

    keys = [args.only] if args.only else list(DOC_PROFILES)
    bad = [k for k in keys if k not in DOC_PROFILES]
    if bad:
        print(f"[错误] 未知文档 {bad}，可选：{list(DOC_PROFILES)}", file=sys.stderr)
        return 2

    store = VectorStore()
    ok, msg = store.health()
    if not ok:
        print(f"[错误] Milvus 不可用：{msg}", file=sys.stderr)
        return 1

    before = store.count()
    print(f"入库前 collection `{store.collection}` 共 {before} 行\n")

    if args.rebuild:
        if not args.yes:
            print("[拒绝执行] --rebuild 会 drop 整个 collection，"
                  "把**所有**文档一起删掉（不是只删指定的那份）。")
            print(f"          当前库里有 {before} 行。确认要重灌请再加 --yes。")
            return 3
        print(f"[重建] 正在清空 collection（将丢失全部 {before} 行）…")
        store.drop_collection()
        store.create_collection()
        before = 0

    for key in keys:
        prof = DOC_PROFILES[key]
        print(f">>> {prof.doc_name}（key={key}）")
        res = asyncio.run(ingest_one(key, args.limit_pages))
        st = res.get("status")
        if st is None or st.error:
            print(f"    [失败] {res.get('error') or st.error}")
            continue
        for k, v in st.stats.items():
            print(f"      {k}: {v}")

    after = store.count()
    print(f"\n入库后共 {after} 行（本次写入 {after - before} 行）")
    print()
    show_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
