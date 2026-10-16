# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""步骤2：为块表构建向量索引（bge-m3 1024 维）+ BM25。

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/11_build_index.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag import corpus as corpus_mod                      # noqa: E402
from lightrag.config import load_config                        # noqa: E402
from lightrag.logging_util import get_logger, write_json       # noqa: E402
from lightrag.ollama_client import OllamaClient                # noqa: E402
from lightrag.vector_index import build_index                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("build_index", cfg, console=True)
    chunks_path = os.path.join(cfg.path_str("processed_dir"), "chunks.jsonl")
    if not os.path.exists(chunks_path):
        log.error("未找到块表 %s，请先跑 10_build_corpus.py", chunks_path)
        return 2

    from lightrag.vector_index import index_exists
    if args.skip_if_exists and index_exists(cfg):
        print("索引已存在，跳过（--skip-if-exists）")
        return 0

    chunks = corpus_mod.load_chunks(chunks_path)
    log.info("载入块表 %d 条，开始嵌入（model=%s, dim=%s）",
             len(chunks), cfg.llm.embed_model, cfg.llm.embed_dim)

    client = OllamaClient(cfg, logger=log)
    health = client.health()
    if not health.get("ok"):
        log.warning("Ollama 探活异常：missing=%s", health.get("missing"))
        if health.get("missing"):
            print(f"[ERROR] Ollama 缺少模型：{health['missing']}；请先 ollama pull")
            return 3

    stats = build_index(chunks, cfg=cfg, client=client, logger=log)
    write_json(os.path.join(cfg.path_str("log_dir"), "index_stats.json"), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
