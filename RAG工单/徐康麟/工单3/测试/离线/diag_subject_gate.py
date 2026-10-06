# -*- coding: utf-8 -*-
"""t12 定位诊断：题 4 的主体闸门四集合（expected/allowed/found/leaked）+ 表列语义分桶。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/diag_subject_gate.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import answerability  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.config import discover_issuer_names
from app.core.query_understanding import strip_issuer_names, understand  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402

EVAL_SET = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"

# 题 4 的真实答案（captain 用表结构核出的 7 家）
TRUTH = ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达"]
# 干扰项：存在控制关系的关联方（自然人/控股股东）
DECOY = "赵马克 42.35% 公司控股股东"


def main() -> int:
    """打印题 4 的闸门四集合与表列语义（定位 allowed 是否漏项 / 是否被自然人夹带拦下）。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("diag_subject_gate")
    retriever = build_retriever(cfg=cfg)
    items = {str(json.loads(line)["id"]): json.loads(line)
             for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()}
    item = items["4"]
    question = item["question"]
    info = understand(question, None, cfg=cfg, llm=None, logger=log)
    retrieval = retriever.retrieve(question, top_k=cfg.retrieval.top_k, logger=log)
    pool = list(retrieval.chunks) + list(retrieval.support_chunks)

    print("=" * 104)
    print(f"### 题 4：{question}")
    print(f"  题面分类：field_type={info['field_type']} expects_numeric={info['expects_numeric']} "
          f"expected_subject={info.get('expected_subject')}")
    print(f"  召回块：{[c.chunk_id for c in retrieval.chunks]}")
    print(f"  支持块：{[c.chunk_id for c in retrieval.support_chunks]}")

    issuer_names = discover_issuer_names(logger=log)
    stripped = strip_issuer_names(question, issuer_names)
    expected = answerability.classify_subject_expectation(stripped, issuer_names=None, logger=log)
    print(f"\n  ① classify_subject_expectation(strip_issuer_names(原题)) → expected={expected!r}")
    print(f"     剥离发行人后题面：{stripped!r}")

    tables = [c for c in pool if str(getattr(c, "type", "")) == "table"]
    print(f"\n  ② 参与取值的表格块 {len(tables)} 个：")
    for chunk in tables:
        first_line = str(getattr(chunk, "content", "")).splitlines()[0] if str(getattr(chunk, "content", "")) else ""
        print(f"     {chunk.chunk_id} p{chunk.page}｜首行（表头）：{first_line[:110]!r}")

    allowed, leaked = answerability.allowed_set_from_tables(tables, expected, logger=log)
    print(f"\n  ③ allowed_set_from_tables → allowed({len(allowed)})={list(allowed)}")
    print(f"     leaked({len(leaked)})={list(leaked)}")
    missing = [name for name in TRUTH if name not in allowed]
    print(f"     真值 7 家缺项：{missing or '无（allowed 完整）'}")

    # ④ 表列语义：表头逐列判定
    print("\n  ④ 表头列语义分桶（column_kind）：")
    for chunk in tables:
        header = str(getattr(chunk, "content", "")).splitlines()[0] if str(getattr(chunk, "content", "")) else ""
        cells = [c.strip() for c in header.strip("|").split("|")]
        print(f"     {chunk.chunk_id}: " + " | ".join(f"{c}→{answerability.column_kind(c)}" for c in cells if c))

    # ⑤ 闸门实测：喂 7 家全名单 / 喂 7 家 + 自然人夹带
    for label, text in (("7 家全名单", "、".join(TRUTH)),
                        ("7 家 + 赵马克夹带", f"{'、'.join(TRUTH)}；控股股东为赵马克")):
        gate = answerability.subject_gate(question, text, pool, issuer_names=None, cfg=cfg, logger=log)
        print(f"\n  ⑤ 闸门[{label}] → ok={gate.ok} reason={gate.reason} counted={gate.counted}")
        print(f"     expected={gate.expected!r}")
        print(f"     allowed ({len(gate.allowed)})={list(gate.allowed)}")
        print(f"     found   ({len(gate.found)})={list(gate.found)}")
        print(f"     leaked  ({len(gate.leaked)})={list(gate.leaked)}")

    # ⑥ 真值表内容（逐行）
    conn = sqlite3.connect(str(cfg.paths.index_dir / "rag.sqlite3"))
    rows = conn.execute("SELECT markdown,title FROM tables WHERE file_name='招股说明书2.pdf' AND page=157").fetchall()
    print(f"\n  ⑥ p157 表数 = {len(rows)}")
    for index, (markdown, title) in enumerate(rows, start=1):
        print(f"     表{index} title={title!r}")
        print("       " + "\n       ".join(str(markdown).splitlines()[:10]))
    conn.close()
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
