# -*- coding: utf-8 -*-
"""T7 诊断：目标证据块在「向量单路 / BM25 单路 / 融合后」的名次与分数。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用途：定位「答案块存在却没进 top-5 / 排到第 5」是**单路没召回**还是**融合后被压**，
为修法（数值锚定、完整度加权、同页去噪）提供实测依据。只读，不改数据。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/diag_ranking.py --ids 207,34,2
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import bm25_index, embedder, retrieval_utils, vector_store  # noqa: E402
from app.core.chunker import Chunk  # noqa: E402
from app.core.config import get_config, model_slug  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402

EVAL_SET = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"

# 关注块（captain 提供的线索）：题 → 期望进前排的块前缀
TARGETS: dict[str, list[str]] = {
    "207": ["招股说明书1_p0490", "招股说明书1_p0479"],
    "34": ["招股说明书1_p0152", "招股说明书1_p0153"],
    "2": ["招股说明书2_p0022", "招股说明书2_p0306", "招股说明书2_p0326"],
}


def rank_of(ids: list[str], prefixes: list[str]) -> dict[str, int]:
    """返回每个前缀在有序 id 列表中的 1-based 名次（未命中记 -1）。"""
    out: dict[str, int] = {}
    for prefix in prefixes:
        pos = next((i + 1 for i, cid in enumerate(ids) if cid.startswith(prefix)), -1)
        out[prefix] = pos
    return out


def main(argv: list[str] | None = None) -> int:
    """打印目标块的三路名次。"""
    parser = argparse.ArgumentParser(description="目标证据块三路名次诊断")
    parser.add_argument("--ids", default="207,34,2")
    parser.add_argument("--depth", type=int, default=60, help="单路考察深度")
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("diag_ranking")
    retriever = build_retriever(cfg=cfg)
    retriever.load()
    model_dir = retriever.model_dir
    vindex = retriever.vector_index
    bm25 = bm25_index.BM25Index.load(model_dir, logger=log)

    items = {str(json.loads(line)["id"]): json.loads(line)
             for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()}
    wanted = [s.strip() for s in args.ids.split(",") if s.strip()]
    for qid in wanted:
        item = items.get(qid)
        if item is None:
            print(f"题 {qid} 不在题集内")
            continue
        question = item["question"]
        prefixes = TARGETS.get(qid, [])
        print("=" * 96)
        print(f"### 题 {qid}：{question}")
        print(f"    golden 证据页：{item.get('evidence_pages')}")
        # 向量单路
        qvec = embedder.embed_query(question, cfg=cfg, logger=log)
        vhits = vector_store.search_vectors(vindex, qvec, args.depth, logger=log)
        vids = [cid for cid, _ in vhits]
        vscore = {cid: s for cid, s in vhits}
        # BM25 单路
        bhits = bm25.search(question, args.depth, logger=log)
        bids = [cid for cid, _ in bhits]
        bscore = {cid: s for cid, s in bhits}
        vrank, brank = rank_of(vids, prefixes), rank_of(bids, prefixes)
        for prefix in prefixes:
            vid = next((c for c in vids if c.startswith(prefix)), "")
            bid = next((c for c in bids if c.startswith(prefix)), "")
            rrf = retrieval_utils.rrf_fuse(vhits[:20], bhits[:20], k=int(cfg.retrieval.rrf_k))
            fused = [cid for cid, _ in rrf]
            print(f"   {prefix:<26} 向量名次={vrank[prefix]:>4} (score={vscore.get(vid, 0):.4f}) "
                  f"BM25名次={brank[prefix]:>4} (score={bscore.get(bid, 0):.3f}) "
                  f"RRF名次={rank_of(fused, [prefix])[prefix]}")
        # 融合+加权+重排后的最终 top-k
        result = retriever.retrieve(question, top_k=cfg.retrieval.top_k, logger=log)
        print("   最终 top-k：" + " | ".join(f"{c.chunk_id}({getattr(c, 'score', 0):.4f})" for c in result.chunks))
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
