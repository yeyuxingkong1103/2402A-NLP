"""分块策略对比实验：对 guide.pdf 用 3 种分块策略分别入库，向量召回 top-4 评估命中率。

用法（仓库根目录）：python eval/chunk_experiment.py
依赖真实 Milvus + BGE-M3 + hypertension_guide collection（锚定 ground-truth 原文、筛 source）。
不改任何现有代码；临时 collection 用 exp_ 前缀，跑完清理。
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()  # 先加载 .env 再 import，保证 EMBED_MODEL / MILVUS_* 读对

import chunker  # noqa: E402
import ingest  # noqa: E402
import milvus_store  # noqa: E402
import parser  # noqa: E402

logger = logging.getLogger(__name__)

PDF_PATH = ROOT / "data" / "raw" / "guide.pdf"
EVAL_PATH = ROOT / "eval" / "eval_set.json"
OUT_PATH = ROOT / "docs" / "分块策略对比.md"
TOP_K = 4
OVERLAP_RATIO = 0.5
HEADING_RE = re.compile(r"^\d+\.\d*\s", re.MULTILINE)
RECURSIVE_SEPS = ["\n\n", "\n", "。", "！", "？", "；", ""]


def split_fixed(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """复用 chunker.split_chunks（临时改 chunker 模块常量注入 size/overlap）。"""
    old_size, old_overlap = chunker.CHUNK_SIZE, chunker.CHUNK_OVERLAP
    chunker.CHUNK_SIZE, chunker.CHUNK_OVERLAP = size, overlap
    try:
        chunks = chunker.split_chunks(text, page_no=0)
    finally:
        chunker.CHUNK_SIZE, chunker.CHUNK_OVERLAP = old_size, old_overlap
    return [c["content"] for c in chunks if c["content"].strip()]


def split_by_heading(text: str, max_len: int = 800, size: int = 400) -> list[str]:
    """按章节标题切块；超过 max_len 的章节再按 size 切。"""
    text = text.strip()
    if not text:
        return []
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return split_fixed(text, size=size, overlap=60) if len(text) > max_len else [text]
    sections = [text[: matches[0].start()]] if matches[0].start() > 0 else []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[m.start(): end])
    out = []
    for sec in sections:
        if sec.strip():
            sec = sec.strip()
            out.extend(split_fixed(sec, size=size, overlap=60) if len(sec) > max_len else [sec])
    return out


def split_recursive(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """参考 RecursiveCharacterTextSplitter：按分隔符优先级递归切 + 合并保留 overlap。"""
    return _merge(_recursive(text, RECURSIVE_SEPS, size), size, overlap)


def _recursive(text: str, seps: list[str], size: int) -> list[str]:
    sep = seps[0]
    if sep == "":
        return [text[i: i + size] for i in range(0, len(text), size)]
    pieces = text.split(sep)
    out = []
    for i, p in enumerate(pieces):
        piece = p + (sep if i < len(pieces) - 1 else "")
        if len(piece) > size and len(seps) > 1:
            out.extend(_recursive(piece, seps[1:], size))
        elif piece:
            out.append(piece)
    return out


def _merge(splits: list[str], size: int, overlap: int) -> list[str]:
    docs, cur, total = [], [], 0
    for d in splits:
        if total + len(d) > size and cur:
            docs.append("".join(cur))
            while total > overlap and cur:
                total -= len(cur[0])
                cur.pop(0)
        cur.append(d)
        total += len(d)
    if cur:
        docs.append("".join(cur))
    return [c for c in docs if c.strip()]


def _build_chunks(pages, split_fn, name):
    chunks = []
    for p in pages:
        text = ingest.normalize_whitespace(p["text"])
        for i, content in enumerate(split_fn(text)):
            chunks.append({
                "id": f"exp_{name}_p{p['page']}_c{i}",
                "content": content,
                "page": p["page"],
            })
    return chunks


def _embed_and_insert(collection, chunks, embedder):
    milvus_store.create_collection(collection)
    documents = [c["content"] for c in chunks]
    embeddings = embedder.encode(
        documents, batch_size=ingest.BATCH_SIZE, normalize_embeddings=True
    ).tolist()
    now = int(time.time())
    records = [
        {
            "id": c["id"], "embedding": emb, "content": c["content"], "page": c["page"],
            "source": "guide.pdf", "domain": "",
            "created_at": now, "updated_at": now, "summary": ingest.make_summary(c["content"]),
            "parent_content": c.get("parent_content", ""),
        }
        for c, emb in zip(chunks, embeddings)
    ]
    milvus_store.insert(collection, records)
    logger.info("已入库 %d 条到 %s", len(records), collection)


def _anchor():
    rows = milvus_store.query_all(milvus_store.MILVUS_COLLECTION)
    content_by_id = {r["id"]: r["content"] for r in rows}
    source_by_id = {r["id"]: r.get("source", "") for r in rows}
    eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    selected = []
    for item in eval_set:
        cid = item["source_chunk_id"]
        if source_by_id.get(cid) == "guide.pdf" and content_by_id.get(cid):
            selected.append({"question": item["question"], "source_content": content_by_id[cid],
                             "source_chunk_id": cid})
        else:
            logger.warning("跳过 %s（source=%s，或原文缺失）", cid, source_by_id.get(cid))
    return selected


def _hit(source_content: str, chunk_content: str) -> bool:
    import jieba
    jieba.setLogLevel(logging.WARNING)
    src = set(jieba.lcut(source_content))
    if not src:
        return False
    return len(src & set(jieba.lcut(chunk_content))) / len(src) >= OVERLAP_RATIO


def _run_strategy(name, collection, split_fn, selected, embedder):
    pages = parser.extract_text(str(PDF_PATH))
    chunks = _build_chunks(pages, split_fn, name)
    _embed_and_insert(collection, chunks, embedder)

    hit_count = 0
    sim_sum = 0.0
    for item in selected:
        qvec = embedder.encode([item["question"]], normalize_embeddings=True).tolist()[0]
        hits = milvus_store.search(collection, qvec, TOP_K)
        if hits:
            sim_sum += sum(h["similarity"] for h in hits) / len(hits)
        if any(_hit(item["source_content"], h["content"]) for h in hits):
            hit_count += 1
    total = len(selected)
    return {
        "name": name,
        "chunk_count": len(chunks),
        "avg_len": round(sum(len(c["content"]) for c in chunks) / len(chunks), 1),
        "hit_count": hit_count,
        "total": total,
        "hit_rate": round(hit_count / total, 4) if total else 0.0,
        "avg_sim": round(sim_sum / total, 4) if total else 0.0,
    }


def _build_conclusion(stats):
    best = max(stats, key=lambda s: s["hit_rate"])
    longest = max(stats, key=lambda s: s["avg_len"])
    fewest = min(stats, key=lambda s: s["chunk_count"])
    return (
        f"在 guide.pdf 来源的 {stats[0]['total']} 条问题上，命中率最高的是「{best['name']}」"
        f"（{best['hit_rate']:.1%}，{best['hit_count']}/{best['total']}）。"
        f"chunk 数最少的是「{fewest['name']}」（{fewest['chunk_count']}），"
        f"平均块长最长的是「{longest['name']}」（{longest['avg_len']} 字）。"
        "注意：固定策略的 overlap 只作用于超长单句硬切，普通相邻块无重叠；"
        "本实验只做向量召回 top-4、未用 rerank，且样本量较小，结论仅供分块策略选型参考。"
    )


def _build_report(stats):
    header = ("| 策略 | chunk 数 | 平均长度 | 命中数/总数 | 命中率 | 平均相似度 |\n"
              "|---|---|---|---|---|---|")
    rows = [
        f"| {s['name']} | {s['chunk_count']} | {s['avg_len']} | "
        f"{s['hit_count']}/{s['total']} | {s['hit_rate']:.1%} | {s['avg_sim']:.4f} |"
        for s in stats
    ]
    return ("# 分块策略对比\n\n" + header + "\n" + "\n".join(rows)
            + "\n\n## 结论\n\n" + _build_conclusion(stats) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    embedder = ingest.load_embedder()
    selected = _anchor()
    logger.info("筛出 guide.pdf 来源问题 %d 条", len(selected))
    if not selected:
        raise SystemExit("没有可评测的 guide.pdf 问题，请检查 hypertension_guide collection")

    strategies = [
        ("固定 400/60", "exp_fixed", split_fixed),
        ("按标题", "exp_heading", split_by_heading),
        ("递归", "exp_recursive", split_recursive),
    ]
    stats = []
    try:
        for name, collection, split_fn in strategies:
            logger.info("=== 策略：%s ===", name)
            stats.append(_run_strategy(name, collection, split_fn, selected, embedder))
    finally:
        for _, collection, _ in strategies:
            try:
                milvus_store.drop_collection(collection)
                logger.info("已 drop %s", collection)
            except Exception as e:  # noqa: BLE001
                logger.warning("drop %s 失败: %s", collection, e)

    OUT_PATH.write_text(_build_report(stats), encoding="utf-8")
    for s in stats:
        logger.info(
            "%s: chunk=%d avg_len=%.1f hit=%d/%d rate=%.1f%% sim=%.4f",
            s["name"], s["chunk_count"], s["avg_len"], s["hit_count"], s["total"],
            s["hit_rate"] * 100, s["avg_sim"],
        )
    logger.info("报告已写入 %s", OUT_PATH)


if __name__ == "__main__":
    main()
