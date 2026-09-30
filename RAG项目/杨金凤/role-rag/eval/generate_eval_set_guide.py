"""从 hypertension_guide 的 guide.pdf 来源 chunk 反推问答，生成纯 guide.pdf 评测集。

用法（仓库根目录）：python eval/generate_eval_set_guide.py
依赖真实 DeepSeek + Milvus。复用 eval/generate_eval_set.py 的 prompt 与生成函数（不复制）。
输出覆盖 eval/eval_set.json，字段与 generate_eval_set.py 一致。
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv

load_dotenv()  # 先加载 .env 再 import，保证 MILVUS_COLLECTION / DEEPSEEK_* 读对

import milvus_store  # noqa: E402
import generate_eval_set  # noqa: E402   # 复用 GEN_PROMPT / generate_pair / _get_client

logger = logging.getLogger(__name__)

SOURCE = "guide.pdf"
SAMPLE_SIZE = 30
SEED = 42
OUT_PATH = ROOT / "eval" / "eval_set.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    random.seed(SEED)
    rows = [r for r in milvus_store.query_all(milvus_store.MILVUS_COLLECTION)
            if r.get("source") == SOURCE]
    logger.info("guide.pdf 来源 chunk 共 %d 条", len(rows))
    if len(rows) < SAMPLE_SIZE:
        raise RuntimeError(f"guide.pdf 仅 {len(rows)} 条 chunk，不足 {SAMPLE_SIZE}")

    client = generate_eval_set._get_client()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    eval_set = []
    for i, row in enumerate(random.sample(rows, SAMPLE_SIZE), 1):
        pair = generate_eval_set.generate_pair(client, model, row["content"])
        eval_set.append({"question": pair["question"], "ground_truth": pair["ground_truth"],
                         "source_chunk_id": row["id"]})
        logger.info("[%d/%d] %s", i, SAMPLE_SIZE, pair["question"][:40])

    OUT_PATH.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写入 %s（%d 条）", OUT_PATH, len(eval_set))


if __name__ == "__main__":
    main()
