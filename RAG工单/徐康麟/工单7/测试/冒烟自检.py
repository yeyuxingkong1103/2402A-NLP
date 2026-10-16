# -*- coding: utf-8 -*-
"""工单7：RAG 链路冒烟自检（索引就绪 → 单题检索 → 生成 → 多轮）。

工单编号：人工智能NLP-RAG-功能测试及评估

用途：
    索引刚建完时，用最小代价确认「工单6 基线链路在 CCF 年报语料上确实能跑通」，
    再交给评估框架跑全量 10 题。避免评估脚本里的失败被误判为「判分逻辑 bug」。

用法：
    python 测试/冒烟自检.py
    python 测试/冒烟自检.py --question "平安银行2019年营业收入是多少？"

退出码：0 全部通过；3 索引缺失；4 后端（Ollama）不可用；1 断言失败。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "研发"))

DEFAULT_QUESTION = "平安银行2019年年度报告中，报告期内实现营业收入是多少？"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单7：{WORK_ORDER} —— RAG 链路冒烟自检")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="用于冒烟的单题")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="额外打印机器可读 JSON")
    args = parser.parse_args(argv)

    from app.core.config import get_config, describe_runtime
    from app.core.errors import IndexMissingError, LLMProbeError

    cfg = get_config()
    print("=" * 78)
    print(f"工单7 冒烟自检｜{WORK_ORDER}")
    print(f"仓库    ：{REPO_ROOT}")
    print(f"语料目录：{cfg.paths.raw_dir}")
    print(f"索引目录：{cfg.paths.index_dir}")
    manifest_path = Path(cfg.paths.index_dir) / "index_manifest.json"
    if not manifest_path.is_file():
        print(f"❌ 索引缺失：{manifest_path}（请先运行 研发/scripts/build_index.py）", file=sys.stderr)
        return 3
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"索引    ：{manifest.get('chunk_total')} 块｜嵌入 {manifest.get('embed_model')}"
          f"({manifest.get('embed_dim')}维)｜向量后端 {manifest.get('vector_backend')}")
    print(f"语料    ：{manifest.get('chunk_by_corpus')}")
    print("=" * 78)

    from app.core.qa_engine import QAEngine

    engine = QAEngine(cfg=cfg)
    t0 = time.perf_counter()
    warm = engine.warmup(with_generate=True)
    print(f"预热完成（{time.perf_counter() - t0:.2f}s）：{json.dumps(warm, ensure_ascii=False)[:300]}")

    failed: list[str] = []

    # ---- 1) 单题问答 ----
    print("-" * 78)
    print(f"[1/3] 单题问答：{args.question}")
    result = engine.answer(args.question, lang="auto", top_k=args.top_k,
                           session_id="smoke", return_evidence=True)
    print(f"  答案：{result.answer[:300]}")
    print(f"  引用页：{result.cited_pages}")
    print(f"  耗时：检索 {result.latency_ms.get('retrieve')} ms｜生成 {result.latency_ms.get('generate')} ms"
          f"｜首字 {result.latency_ms.get('first_token')} ms｜总计 {result.latency_ms.get('total')} ms")
    print(f"  后端：{result.backend}:{result.model}｜方式：{result.generation_method}"
          f"｜拒答：{result.refused}")
    if not str(result.answer or "").strip():
        failed.append("单题问答：答案为空")
    if result.latency_ms.get("total") is None:
        failed.append("单题问答：未记录总耗时")

    # ---- 2) 证据块 ----
    print("-" * 78)
    print(f"[2/3] 检索证据（top-{args.top_k}）")
    evidence = list(result.evidence or [])
    if not evidence:
        failed.append("检索证据为空（检索层未返回任何 chunk）")
    for hit in evidence[:5]:
        text = str(hit.get("text") or "").replace("\n", " ")[:100]
        print(f"  ✦ {hit.get('file_name')} p{hit.get('page')} [{hit.get('type')}] "
              f"score={hit.get('final_score')}｜{text}…")

    # ---- 3) 多轮（同一 session 追问）----
    print("-" * 78)
    print("[3/3] 多轮追问（同 session）")
    follow = "那它的净利润是多少？"
    r2 = engine.answer(follow, lang="auto", top_k=args.top_k, session_id="smoke", return_evidence=False)
    print(f"  追问：{follow}")
    print(f"  答案：{r2.answer[:200]}")
    print(f"  耗时 {r2.latency_ms.get('total')} ms｜拒答={r2.refused}｜引用页={r2.cited_pages}")
    if not str(r2.answer or "").strip():
        failed.append("多轮追问：答案为空")

    # ---- 4) 异常输入（不崩溃即可）----
    print("-" * 78)
    print("[额外] 异常输入兜底")
    for probe in ("", "   ", "!@#$%^&*()", "asdkjfhaksjdfh"):
        try:
            rr = engine.answer(probe, lang="auto", top_k=3, session_id="smoke-anom")
            print(f"  输入={probe!r} → 拒答={rr.refused}｜答案={(rr.answer or '')[:60]!r}")
        except Exception as exc:  # noqa: BLE001 —— 冒烟阶段只关心「是否崩溃」
            failed.append(f"异常输入崩溃：{probe!r} → {type(exc).__name__}: {exc}")
            print(f"  ❌ 输入={probe!r} 崩溃：{type(exc).__name__}: {exc}")

    print("=" * 78)
    if failed:
        print("❌ 冒烟自检未通过：")
        for item in failed:
            print(f"  - {item}")
        return 1
    print("✅ 冒烟自检通过：索引加载 / 检索 / 生成 / 多轮 / 异常兜底 全部可跑")
    if args.json:
        print(json.dumps({"answer": result.answer, "cited_pages": result.cited_pages,
                          "latency_ms": result.latency_ms,
                          "evidence": [{"chunk_id": h.get("chunk_id"), "page": h.get("page"),
                                        "score": h.get("final_score")} for h in evidence]},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已中断]", file=sys.stderr)
        raise SystemExit(130)
