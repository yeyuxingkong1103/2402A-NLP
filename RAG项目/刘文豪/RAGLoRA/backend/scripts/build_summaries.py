# -*- coding: utf-8 -*-
"""为 chunk 生成摘要，填充 Milvus 的 `summary` 字段。

背景：`summary` 字段在建表时按清单要求创建了，但**一直是空的** ——
属于「名义已满足、实际未填充」。本脚本把它真正填上。

摘要的用途
----------
1. **快速浏览**：前端展示来源时，摘要比整段 800 字正文更适合一眼扫过；
2. **上下文压缩**：长文档场景下可用摘要代替正文进 prompt，省 token。

摘要**不参与检索打分** —— 检索仍用正文的 dense + sparse 向量。
（若要让摘要参与检索，需要额外的摘要向量字段，那是另一件事。）

成本（务必先看）
----------------
本脚本对每个 chunk 调一次 LLM。实测速率约 **1.5-2 条/秒**（qwen2.5:7b，CPU 推理）：

    kb_medical   636 条   ≈  6-8 分钟
    kb_legal  14319 条   ≈  2-3 小时

因此**默认只跑 kb_medical**，全量跑属明确决策，需 `--collection kb_legal` 显式指定。

⚠️ 关于「数据增强」的其他两项（实测数据）
----------------------------------------
* **去重**：实测两个集合**完全相同 0 条、前 60 字相同 0 条** ——
  本语料已很干净，去重**没有价值**，故未实现。
* **低质量过滤**：kb_legal 有 654 条 <50 字的短块，但抽样发现**它们都是有效的短条文**
  （如「《广告法》第十三条规定，广告不得贬低其他生产经营者的商品或者服务。」）。
  **按长度过滤会删掉合法内容**，故未实现。
* **父子块**：kb_legal 仅 6/14313 个条文被切成多块（条文本身是原子的），
  kb_medical 有 112/113 页被切成多块 —— **仅医疗侧有适用场景**，未在本次实现。

用法
----
    python scripts/build_summaries.py                    # kb_medical（默认）
    python scripts/build_summaries.py --limit 20         # 只跑 20 条（验证用）
    python scripts/build_summaries.py --collection kb_legal   # 全量，耗时数小时
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import get_logger          # noqa: E402
from app.services import llm, milvus_store       # noqa: E402

log = get_logger("summaries")

PROMPT = """请用一句话概括下面这段内容的要点，不超过 50 字。
只输出概括本身，不要任何前缀、编号或解释。

内容：
{text}
"""


def summarize(text: str) -> str:
    """生成单条摘要。失败返回空串（调用方会跳过，不写入空值）。"""
    try:
        out = llm.chat([{"role": "user", "content": PROMPT.format(text=text[:1500])}],
                       temperature=0.1, max_tokens=80)
        return " ".join(out.split())[:200]
    except Exception as e:
        log.warning("摘要生成失败: %s", str(e)[:120])
        return ""


def run(collection: str, limit: int | None, batch: int = 50) -> dict:
    client = milvus_store.get_client()
    if not client.has_collection(collection):
        return {"ok": False, "error": f"集合不存在: {collection}"}

    # 只取还没有摘要的（幂等：重跑不会重复生成）
    rows: list[dict] = []
    offset = 0
    while True:
        batch_rows = client.query(
            collection_name=collection, filter="summary == ''",
            output_fields=["id", "text"], limit=1000, offset=offset)
        if not batch_rows:
            break
        rows.extend(batch_rows)
        offset += len(batch_rows)
        if len(batch_rows) < 1000:
            break
    if limit:
        rows = rows[:limit]

    log.info("%s 待生成摘要 %d 条", collection, len(rows))
    if not rows:
        return {"ok": True, "generated": 0, "note": "没有缺少摘要的 chunk"}

    t0 = time.time()
    done = 0
    buf: list[tuple[int, str]] = []

    for i, r in enumerate(rows, 1):
        s = summarize(r.get("text") or "")
        if s:
            buf.append((r["id"], s))
            done += 1

        if len(buf) >= batch:
            _flush(collection, buf)
            buf.clear()
            rate = done / max(1e-9, time.time() - t0)
            log.info("  %d/%d 已生成（%.1f 条/秒，预计剩余 %.0f 分钟）",
                     i, len(rows), rate, (len(rows) - i) / max(rate, 1e-9) / 60)

    if buf:
        _flush(collection, buf)

    elapsed = time.time() - t0
    client.flush(collection)
    log.info("%s 摘要完成 | %d 条 | %.0fs | %.2f 条/秒",
             collection, done, elapsed, done / max(elapsed, 1e-9))
    return {"ok": True, "generated": done, "total": len(rows),
            "elapsed_s": round(elapsed, 1)}


def _flush(collection: str, items: list[tuple[int, str]]) -> None:
    """把 (id, summary) 批量写回。用 upsert 会覆盖整行，故改用逐条 update。

    ⚠️ Milvus 的 upsert 是**整行替换**：只给 id+summary 会把 text/向量清空。
    因此必须用「先 query 出整行、改 summary、再 upsert 回去」的方式，
    或者用 update。这里用 query+upsert 以保证不丢字段。
    """
    client = milvus_store.get_client()
    ids = [i for i, _ in items]
    existing = {}
    for i in ids:
        res = client.query(collection_name=collection, filter=f"id == {int(i)}",
                           output_fields=["*"], limit=1)
        if res:
            existing[int(i)] = res[0]

    rows = []
    for i, s in items:
        e = existing.get(int(i))
        if not e:
            continue
        e = dict(e)
        e["summary"] = s
        # query 返回的是只读结构，字段名与 schema 一致，可直接回写
        rows.append(e)
    if rows:
        client.upsert(collection_name=collection, data=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kb_medical")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 条（0=全部缺少摘要的）")
    ap.add_argument("--sample", type=int, default=0,
                    help="抽样验证模式：只跑 N 条并打印速率，供评估全量成本")
    args = ap.parse_args()

    limit = args.sample or args.limit or None
    res = run(args.collection, limit)
    print(res)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
