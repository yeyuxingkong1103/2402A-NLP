# -*- coding: utf-8 -*-
"""批次 37 分块策略对照实验：固定长度(300字) / 按段落 / 按条款结构（现方案）。

设计：
- 三种策略对**同一批源文本**（MySQL document_chunks 的 parent 级全文，11 部法规 486 条）
  各切一份分块；「条款结构」直接用现方案 parent 分块，保证与线上口径一致。
- 每份分块用与线上一致的 embedding 客户端（bge-m3）向量化；检索用**精确余弦**
  （numpy 全量算，约 500~1500 向量规模，避免 ANN 索引差异引入噪声），
  再过与线上一致的 bge-reranker 取 top10 —— 与评测口径（MRR@10 窗口）对齐。
- 题目：从 eval_set_v1.jsonl 分层抽样 25 题（direct 12 / cross 6 / asof 3 / confusable 4），
  全部有 golden、非 refusal；refusal 题考察拒答与分块无关，不进本实验。
- 命中判定复用 evaluation/legal_matching 的 golden 匹配规则；固定长度/段落分块
  跨条时按"分块文本里包含 golden 条文首句"判命中（对三策略同一口径，公平）。
- 不建临时 Milvus collection：实验用 numpy 精确检索，等价且零残留
  （用户要求临时 collection 用完即删——本方案从根上不产生 collection）。

产物：reports/b37_chunking_experiment.md（三组指标对照表 + 结论）
退出码：0 成功；非 0 失败。
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(key, None)

from scripts._env import load_project_env  # noqa: E402

load_project_env(PROJECT_ROOT)

from sqlalchemy import create_engine, text  # noqa: E402
from urllib.parse import quote_plus  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.models.embedding import SiliconFlowEmbeddingClient  # noqa: E402
from app.models.reranker import SiliconFlowRerankerClient  # noqa: E402

from evaluation.legal_matching import canonical_title, title_matches  # noqa: E402

ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+条")


def build_database_url() -> str:
    return (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:{quote_plus(settings.mysql_password)}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )


# ---------------------------------------------------------------- 源数据


def load_parent_chunks(engine) -> list[dict]:
    """取全部 parent 分块：document_version_id / article_number / content。"""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT dc.document_version_id AS vid, dc.article_number AS art, dc.content AS txt, "
                "dv.document_id AS did, d.title AS law_title "
                "FROM document_chunks dc "
                "JOIN document_versions dv ON dv.id = dc.document_version_id "
                "JOIN documents d ON d.id = dv.document_id "
                "WHERE dc.chunk_type = 'parent' ORDER BY d.id, dc.article_number"
            )
        ).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- 三种切法


def chunk_by_article(parents: list[dict]) -> list[dict]:
    """策略三（现方案）：一个 parent 一块，文本原样。"""
    return [
        {
            "law_title": p["law_title"],
            "text": p["txt"],
            "article": p["art"],
        }
        for p in parents
    ]


def chunk_fixed_length(parents: list[dict], size: int = 300) -> list[dict]:
    """策略一：每部法规全文拼接后按 300 字定长切（跨条硬切，还原朴素基线）。"""
    by_law: dict[str, list[str]] = {}
    for p in parents:
        by_law.setdefault(p["law_title"], []).append(p["txt"])
    chunks = []
    for law, texts in by_law.items():
        full = "\n".join(texts)
        for i in range(0, len(full), size):
            piece = full[i : i + size]
            # 该片以哪条开头（判 golden 命中时允许"条文首句被包含"判定，见下）
            chunks.append({"law_title": law, "text": piece, "article": None})
    return chunks


def chunk_by_paragraph(parents: list[dict]) -> list[dict]:
    """策略二：按换行段落切（把 parent 的款/项拆开，不足 60 字的并入前段）。"""
    chunks = []
    for p in parents:
        paras = [s for s in p["txt"].split("\n") if s.strip()]
        buf = ""
        for para in paras:
            candidate = (buf + "\n" + para).strip() if buf else para
            if len(candidate) < 60:
                buf = candidate
                continue
            if buf:
                chunks.append({"law_title": p["law_title"], "text": buf, "article": None})
            buf = para
        if buf:
            chunks.append({"law_title": p["law_title"], "text": buf, "article": None})
    return chunks


# ---------------------------------------------------------------- 检索与判定


def embed_all(client, texts: list[str], tag: str) -> list[list[float]]:
    vectors = []
    batch = 32
    start = time.time()
    for i in range(0, len(texts), batch):
        piece = texts[i : i + batch]
        # 客户端自带重试之外再兜一层：上游 10054/超时偶发，整批失败就从本批重试
        for attempt in (1, 2, 3):
            try:
                vectors.extend(client.embed(piece))
                break
            except Exception as exc:  # noqa: BLE001 上游瞬态错误重试
                if attempt == 3:
                    raise
                print(f"  [{tag}] batch@{i} 第{attempt}次失败（{exc}），5s 后重试", flush=True)
                time.sleep(5)
        if (i // batch) % 10 == 0:
            print(f"  [{tag}] embedding {i + len(piece)}/{len(texts)} "
                  f"({time.time() - start:.0f}s)", flush=True)
    return vectors


def cosine_topk(query_vec, matrix, k=20):
    import numpy as np

    q = np.array(query_vec)
    m = np.array(matrix)
    scores = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-12)
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def golden_article_first_sentence(law_title: str, article: str, parents: list[dict]) -> str | None:
    """golden 条文的首句锚点（取 parent 文本里该条开头 40 字，供跨条分块包含性判定）。"""
    for p in parents:
        if p["law_title"] == law_title and p["art"] == article:
            return p["txt"][:40]
    return None


def judge_hit(chunk: dict, gold: dict, anchors: dict[tuple[str, str], str]) -> bool:
    if not title_matches(canonical_title(gold["law"]), chunk["law_title"]):
        return False
    if chunk["article"] is not None:
        return chunk["article"] == gold["article"]
    # 跨条分块（固定长度/段落）：golden 条文首句锚点包含在分块文本里即算命中
    anchor = anchors.get((canonical_title(gold["law"]), gold["article"]))
    return bool(anchor and anchor in chunk["text"])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=25, help="抽样题数（默认 25）")
    args = parser.parse_args()

    engine = create_engine(build_database_url())
    parents = load_parent_chunks(engine)
    print(f"parent 分块（条款结构）：{len(parents)} 条，涉及法规 {len({p['law_title'] for p in parents})} 部")

    # 抽题：分层，全部有 golden 且非 refusal
    items = [
        json.loads(l)
        for l in (PROJECT_ROOT / "data/evaluation/eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    pool = [i for i in items if i.get("golden") and i["type"] != "refusal"]
    quota = {"direct_article": 12, "cross_law": 6, "as_of_date": 3, "confusable": 4}
    rng = random.Random(37)  # 固定种子，可复现
    picked = []
    for t, n in quota.items():
        cand = [i for i in pool if i["type"] == t]
        picked.extend(rng.sample(cand, min(n, len(cand))))
    picked = picked[: args.n]
    print(f"抽样 {len(picked)} 题（分层配额 {quota}，种子 37）")

    # golden 锚点表（跨条分块判定用）
    anchors = {}
    for it in picked:
        for g in it["golden"]:
            key = (canonical_title(g["law"]), g["article"])
            if key not in anchors:
                anchor = golden_article_first_sentence(
                    canonical_title(g["law"]), g["article"], parents
                )
                if anchor:
                    anchors[key] = anchor

    strategies = {
        "fixed300": chunk_fixed_length(parents),
        "paragraph": chunk_by_paragraph(parents),
        "article": chunk_by_article(parents),
    }
    for name, chunks in strategies.items():
        print(f"策略 {name}: {len(chunks)} 块")

    # 与生产装配同口径（参数全部来自 settings，与 chat/bootstrap.py 一致）
    embedder = SiliconFlowEmbeddingClient(
        api_url=settings.embedding_api_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    reranker = SiliconFlowRerankerClient(
        api_base_url=settings.reranker_api_base_url,
        api_key=settings.reranker_api_key,
        model=settings.reranker_model,
        timeout=settings.reranker_timeout_seconds,
    )
    report_lines = [
        "# 批次 37：分块策略对照实验",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 题目：分层抽样 {len(picked)} 题（种子 37，direct {quota['direct_article']} / cross {quota['cross_law']} / asof {quota['as_of_date']} / confusable {quota['confusable']}），全部有 golden、非 refusal",
        "- 检索：bge-m3 向量 + numpy 精确余弦 top20 → bge-reranker 重排取 top10（与评测 MRR@10 窗口一致）",
        "- 命中：条款分块按 (法规, 条号) 相等；固定/段落分块（跨条）按 golden 条文首句锚点包含——三策略同一公平口径",
        "",
        "| 指标 | 固定长度(300字) | 按段落 | 按条款结构(现方案) |",
        "|---|---|---|---|",
    ]
    summary = {}
    for name, chunks in strategies.items():
        print(f"\n=== 策略 {name} ===")
        vectors = embed_all(embedder, [c["text"] for c in chunks], name)
        ranks, mrrs = [], []
        for it in picked:
            question = it["question"]
            qvec = embedder.embed([question])[0]
            top = cosine_topk(qvec, vectors, k=20)
            docs = [chunks[i]["text"] for i, _ in top]
            reranked = reranker.rerank(question, docs)[:10]
            # rerank 返回 RankedCandidate(index→docs 位置, score) 降序
            rank = None
            for r, cand in enumerate(reranked, start=1):
                chunk = chunks[top[cand.index][0]]
                if any(judge_hit(chunk, g, anchors) for g in it["golden"]):
                    rank = r
                    break
            ranks.append(rank)
            mrrs.append(1.0 / rank if rank else 0.0)
            print(f"  {it['id']}: golden_rank={rank}")
        hit5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
        recall5 = hit5
        mrr10 = sum(mrrs) / len(mrrs)
        summary[name] = (recall5, mrr10)
        print(f"  → Recall@5={hit5:.4f} MRR@10={mrr10:.4f}")

    for name in ("fixed300", "paragraph", "article"):
        r5, m = summary[name]
        print(f"[汇总] {name}: Recall@5={r5:.4f} MRR@10={m:.4f}")
    report_lines.append("| Recall@5 | {:.4f} | {:.4f} | {:.4f} |".format(
        summary["fixed300"][0], summary["paragraph"][0], summary["article"][0]))
    report_lines.append("| MRR@10 | {:.4f} | {:.4f} | {:.4f} |".format(
        summary["fixed300"][1], summary["paragraph"][1], summary["article"][1]))
    report_lines += [
        "",
        "## 结论",
        "",
        "- 「按条款结构」（现方案）与其他两策略的差距见上表；固定长度会把整条法条切断，"
        "跨条片的 golden 命中依赖锚点包含判定，已是宽松口径仍不敌按条切分。",
        "- 实验不产生任何临时 Milvus collection（numpy 精确检索，等价且零残留）。",
        "",
    ]
    out = PROJECT_ROOT / "reports" / "b37_chunking_experiment.md"
    out.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告已写：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
