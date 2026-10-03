# -*- coding: utf-8 -*-
"""基线检索「精确复现」：用**原封不动的工单1 基线代码 + 基线索引**重跑 10 个工单问题的检索，
拿到评估运行真实送入生成器的 chunk 集合，从而区分「检索失败」与「生成/答案形态失败」。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（T6 核心取证）

为什么必须这样做：`环境事实.md` §4.1.0 定的判据是「golden evidence 原文是否落在**本次返回**的
chunk 中」。日志里 chunk_id 被截断，只能看到页码；本脚本用基线代码本身复现，拿到精确 chunk_id。

隔离措施（**绝不写工单1**）：
  - 基线代码与索引已**复制**到 `优化/基线/repro/`，只读使用；
  - 所有输出路径通过 `RAG_PATHS__*` 环境变量指向 `repro/`，`get_settings()` 的
    `ensure_directories()` 只会在 repro 内建目录；
  - 嵌入模型用 `RAG_EMBEDDING__LOCAL_MODEL_DIR` 指向工单1 的模型目录（只读加载，不写入）。

复现命令（工作目录 = E:\\gao6gongdan\\工单2）：

    pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/reproduce_baseline_retrieval.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
W2 = Path(r"E:\gao6gongdan\工单2")
REPRO = W2 / "优化" / "基线" / "repro"
OUT = W2 / "优化" / "基线" / "baseline_repro_retrieval.json"

# ---- 1. 先设环境变量，保证后续 import 不会指向工单1 ----
os.environ.update(
    {
        "PYTHONDONTWRITEBYTECODE": "1",
        "RAG_PATHS__DATA_RAW": str(REPRO / "data" / "raw"),
        "RAG_PATHS__DATA_PROCESSED": str(REPRO / "data" / "processed"),
        "RAG_PATHS__DATA_INDEX": str(REPRO / "data" / "index"),
        "RAG_PATHS__DATA_EVAL": str(REPRO / "data" / "eval"),
        "RAG_PATHS__EVAL_RESULTS": str(REPRO / "eval_results"),
        "RAG_PATHS__LOGS": str(REPRO / "logs"),
        "RAG_PATHS__SQLITE_PATH": str(REPRO / "data" / "index" / "rag.sqlite3"),
        # 只读加载工单1 的本地嵌入模型；绝不联网
        "RAG_EMBEDDING__LOCAL_MODEL_DIR": str(W1 / "models" / "bge-small-zh-v1.5"),
        "RAG_EMBEDDING__MODEL_NAME": str(W1 / "models" / "bge-small-zh-v1.5"),
        "RAG_EMBEDDING__BACKEND": "numpy",
        # 基线评估运行中 rerank_score 全为 null（交叉编码器不可用），此处等价复现
        "RAG_RETRIEVAL__USE_RERANKER": "false",
    }
)
sys.path.insert(0, str(REPRO))

TRACE = W1 / "logs" / "rag_trace.jsonl"
TS_RE = re.compile(r'"ts"\s*:\s*"([^"]+)"')
WINDOW = ("2026-10-02T19:37", "2026-10-02T19:42")
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"


def norm_punct(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in PUNCT_TO_STRIP)


def load_eval_variants() -> dict[int, dict]:
    """从真实日志中取出评估运行里每题最后一次 retrieve_multi 的查询变体（原样复现输入）。"""
    golden = {
        json.loads(l)["question"]: int(json.loads(l)["id"])
        for l in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    current: int | None = None
    out: dict[int, dict] = {}
    with TRACE.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TS_RE.search(line[:120])
            if not m:
                continue
            ts = m.group(1)
            if not (WINDOW[0] <= ts[:16] <= WINDOW[1]):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mod, fn, event = obj.get("module", ""), obj.get("function", ""), obj.get("event", "")
            if mod == "app.core.qa_engine" and fn == "QAEngine.ask" and event == "enter":
                args = obj.get("args") or []
                current = golden.get(args[0]) if args else None
            elif (
                current is not None
                and mod == "app.core.retriever"
                and fn == "Retriever.retrieve_multi"
                and event == "enter"
            ):
                args = obj.get("args") or []
                variants = args[0] if args else []
                # 日志里 tuple 被序列化成 list；retrieve_multi 期望 (query, kind) 元组
                variants = [tuple(v) if isinstance(v, list) else v for v in variants]
                out[current] = {"ts": ts, "variants": variants}
    return out


def main() -> int:
    from app.core.config import get_settings
    from app.core.retriever import Retriever
    from app.models.schemas import Chunk

    settings = get_settings()
    print(f"PROJECT_ROOT = {settings.paths.project_root}")
    print(f"data_index   = {settings.paths.data_index}")
    print(f"logs         = {settings.paths.logs}")
    assert "工单1" not in str(settings.paths.data_index), "索引路径仍指向工单1，拒绝继续（安全保护）"
    assert str(settings.paths.logs).startswith(str(REPRO)), "日志路径未隔离到 repro，拒绝继续"

    chunks = [Chunk(**json.loads(l)) for l in (REPRO / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    retriever = Retriever()
    ok = retriever.load_index(chunks)
    print(f"索引装载: {ok}; vector={retriever.vector_store.count()} ({retriever.vector_store.name}); bm25={retriever.bm25.size}")
    print(f"嵌入后端: {retriever.embedder.name} dim={retriever.embedder.dimension}")

    golden = [
        json.loads(l) for l in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    golden.sort(key=lambda g: int(g["id"]))
    ev_ids = {
        int(g["id"]): {c.chunk_id for c in chunks if norm_punct(g["evidence"]) in norm_punct(c.content)}
        for g in golden
    }
    ev_pages = {
        int(g["id"]): sorted({c.page for c in chunks if norm_punct(g["evidence"]) in norm_punct(c.content)})
        for g in golden
    }

    variants_map = load_eval_variants()
    rows = []
    for g in golden:
        qid = int(g["id"])
        info = variants_map.get(qid)
        if not info:
            rows.append({"question_id": qid, "error": "日志中未找到该题的 retrieve_multi 变体"})
            continue
        results = retriever.retrieve_multi(info["variants"])
        ctx_ids = [r.chunk.chunk_id for r in results]
        ctx_pages = sorted({r.chunk.page for r in results})
        hit_ids = sorted(set(ctx_ids) & ev_ids[qid])
        rows.append(
            {
                "question_id": qid,
                "variants": info["variants"],
                "ts": info["ts"],
                "context_chunk_ids": ctx_ids,
                "context_pages": ctx_pages,
                "context_size": len(ctx_ids),
                "evidence_chunk_ids": sorted(ev_ids[qid]),
                "evidence_pages": ev_pages[qid],
                "evidence_chunk_in_context": bool(hit_ids),
                "evidence_chunk_ids_in_context": hit_ids,
                "evidence_page_in_context": bool(set(ctx_pages) & set(ev_pages[qid])),
                "top1": (results[0].chunk.chunk_id, results[0].chunk.page) if results else None,
                "scores": [
                    {
                        "rank": i + 1,
                        "chunk_id": r.chunk.chunk_id,
                        "page": r.chunk.page,
                        "score": round(float(r.score), 4),
                        "vector_score": round(float(r.vector_score or 0.0), 4),
                        "bm25_score": round(float(r.bm25_score or 0.0), 4),
                        "source": r.source,
                    }
                    for i, r in enumerate(results)
                ],
            }
        )

    summary = {
        "证据 chunk 进入最终上下文": {
            "count": sum(1 for r in rows if r.get("evidence_chunk_in_context")),
            "question_ids": [r["question_id"] for r in rows if r.get("evidence_chunk_in_context")],
        },
        "证据页进入最终上下文": {
            "count": sum(1 for r in rows if r.get("evidence_page_in_context")),
            "question_ids": [r["question_id"] for r in rows if r.get("evidence_page_in_context")],
        },
        "总题数": len(rows),
    }

    OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
                    "用途": "用基线原始代码+原始索引精确复现评估运行的检索上下文（chunk 级）",
                    "文件口径": "★ 流水线最终上下文：本文件的 context_chunk_ids 就是基线 `Retriever.retrieve_multi` "
                    "返回、并**原样送入生成器**的那 8 条 chunk（= 工单1 config.retrieval.rerank_top_n = 8）。"
                    "它不是纯向量检索结果，也不是候选池；索引级口径见 `baseline_retrieval.json`。两者不可混用。",
                    "context_size": "每题恒为 8 条 chunk（与真实日志 Generator.generate 的 retrieved_count=8 一致）",
                    "baseline_code": str(REPRO / "app"),
                    "index": str(REPRO / "data" / "index"),
                    "variants_source": str(TRACE) + f" 窗口 {WINDOW} 内每题最后一次 retrieve_multi 的入参",
                    "hit_definition": "严格口径：整段 evidence（归一化 B：去空白+中英文标点）落在同一 chunk 内",
                    "保真度校验": "复现的上下文页码与真实日志逐题一致 10/10（见 基线说明.md §4.1）",
                    "reproduce_cmd": "pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/reproduce_baseline_retrieval.py",
                },
                "summary": summary,
                "per_question": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'题号':<6}{'上下文条数':<10}{'上下文页':<40}{'证据chunk在上下文':<18}{'证据页在上下文':<16}{'top1':<22}")
    for r in rows:
        if r.get("error"):
            print(f"{r['question_id']:<6}ERROR {r['error']}")
            continue
        print(
            f"{r['question_id']:<6}{r['context_size']:<10}{str(r['context_pages'])[:38]:<40}"
            f"{('✅ 是' if r['evidence_chunk_in_context'] else '❌ 否'):<16}"
            f"{('✅ 是' if r['evidence_page_in_context'] else '❌ 否'):<14}"
            f"{str(r['top1']):<22}"
        )
    print(f"\n汇总: 证据 chunk 进入上下文 {summary['证据 chunk 进入最终上下文']['count']}/{len(rows)}"
          f"；证据页进入上下文 {summary['证据页进入最终上下文']['count']}/{len(rows)}")
    print(f"输出: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 基线检索复现失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
