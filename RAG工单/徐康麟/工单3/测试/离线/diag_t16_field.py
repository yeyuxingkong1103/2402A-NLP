# -*- coding: utf-8 -*-
"""t16 诊断：字段型问句成句化的触发条件在真实候选块上逐条打印（只读诊断，不改产品逻辑）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用途：解释「阶段⑩ 为什么没触发」。打印 5 项：① 问题分类；② 现有答案正文抽值；
③ 每个候选块抽值；④ 取值逐字比对；⑤ 触发判据。产物落到 优化/评估结果/过程日志/_t16_field_diag.txt。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/diag_t16_field.py --qid 531
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
sys.path.insert(0, str(DEV_DIR))
sys.path.insert(0, str(DEV_DIR / "scripts"))

from app.core import citation as citation_mod  # noqa: E402
from app.core import generator as generator_mod  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import build_engine  # noqa: E402
from app.core.query_understanding import FIELD_KEYWORDS, understand  # noqa: E402
from evaluate import load_golden  # noqa: E402

DEFAULT_GOLDEN = DEV_DIR / "data" / "eval" / "golden_qa.jsonl"
OUT_DIR = REPO_ROOT / "优化" / "评估结果" / "过程日志"


def main(argv: list[str] | None = None) -> int:
    """入口：按 qid 取题 → 实跑一次 → 打印触发判据的每一步。"""
    parser = argparse.ArgumentParser(description="t16 字段型成句化诊断")
    parser.add_argument("--qid", type=int, default=531)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("diag_t16_field")
    lines: list[str] = []

    def emit(text: str) -> None:
        lines.append(text)
        print(text)

    with log.enter("diag", {"qid": args.qid}) as span:
        golden = load_golden(DEFAULT_GOLDEN, logger=log)
        item = next(row for row in golden if int(row["id"]) == args.qid)
        question = str(item["question"])
        engine = build_engine(cfg=cfg, warmup=True, logger=log)
        info = understand(question, [], cfg=cfg, llm=None, logger=log)
        emit(f"问题：{question}")
        emit(f"分类：field_type={info.get('field_type')} expects_numeric={info.get('expects_numeric')} "
             f"expected_subject={info.get('expected_subject')}")
        keywords = [k for k in FIELD_KEYWORDS.get(str(info.get("field_type")), ()) if k in question]
        emit(f"命中字段词（问题里出现的）：{keywords}")
        emit(f"枚举词命中={bool(generator_mod._ENUM_QUESTION_PATTERN.search(question))} "
             f"问号数={question.count('？') + question.count('?')}")
        answer = engine.ask(question, top_k=args.top_k)
        body = citation_mod.answer_body(answer.text)
        emit(f"答案正文（{len(body)} 字）：{body}")
        emit(f"引用：{[c.render() for c in answer.citations]}")
        expects_numeric = bool(info.get("expects_numeric"))
        for keyword in keywords:
            value, style = generator_mod._extract_field_value(body, keyword, expects_numeric=expects_numeric)
            emit(f"答案抽值：keyword={keyword} value={value!r} style={style!r}")
            if not value or style == "copula":
                continue
            pool = [engine.generator.chunk_lookup(c.chunk_id) for c in answer.citations if c.chunk_id]
            retrieval = getattr(answer, "retrieval", None)
            pool += list(getattr(retrieval, "chunks", []) or [])
            pool += list(getattr(retrieval, "support_chunks", []) or [])
            seen: set[str] = set()
            for chunk in pool:
                if chunk is None:
                    continue
                chunk_id = str(getattr(chunk, "chunk_id", ""))
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                evidence_value, evidence_style = generator_mod._extract_field_value(
                    str(getattr(chunk, "content", "") or ""), keyword, expects_numeric=expects_numeric)
                same = (bool(evidence_value)
                        and generator_mod.text_utils.squash_text(evidence_value)
                        == generator_mod.text_utils.squash_text(value))
                emit(f"  块 {chunk_id} p{getattr(chunk, 'page', '?')} type={getattr(chunk, 'type', '?')} "
                     f"取值={evidence_value!r}({evidence_style}) 同值={same}")
                if not same:
                    emit(f"    内容前 160 字：{str(getattr(chunk, 'content', ''))[:160]!r}")
        corpus = {"item": item, "info": info, "answer": answer.text, "body": body}
        # 阶段⑪（尾句裁剪）触发条件的逐步复现：子句 → token 重叠 → 安全网 → 每块可核性
        clauses = generator_mod._split_clauses(body)
        emit(f"[尾句裁剪] 子句数={len(clauses)} 枚举命中={bool(generator_mod._ENUM_QUESTION_PATTERN.search(question))} "
             f"expects_numeric={expects_numeric}")
        if len(clauses) >= 2:
            q_tokens = {t for t in generator_mod.text_utils.tokenize(question) if len(t) >= 2}
            kept, dropped = list(clauses), []
            while len(kept) >= 2:
                last = kept[-1]
                if expects_numeric and __import__("re").search(r"\d", last):
                    emit(f"  尾子句含数字且 expect 数值 → 停止：{last[:40]}")
                    break
                tokens = {t for t in generator_mod.text_utils.tokenize(last) if len(t) >= 2}
                if tokens & q_tokens:
                    emit(f"  尾子句与问题重叠 {sorted(tokens & q_tokens)} → 停止：{last[:40]}")
                    break
                dropped.append(kept.pop())
            emit(f"  可裁尾子句 {len(dropped)} 个：{[d[:30] for d in dropped]}")
            if dropped:
                kept_body = "".join(kept).strip()
                anchors = [k for k in FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ())
                           if k in question]
                emit(f"  保留文本（{len(kept_body)} 字）：{kept_body}")
                emit(f"  安全网 anchors={anchors} 缺失={[a for a in anchors if a not in kept_body]}")
                for chunk in ([engine.generator.chunk_lookup(c.chunk_id) for c in answer.citations
                               if c.chunk_id]
                              + list(getattr(getattr(answer, "retrieval", None), "chunks", []) or [])
                              + list(getattr(getattr(answer, "retrieval", None), "support_chunks", []) or [])):
                    if chunk is None:
                        continue
                    cite = citation_mod.Citation(file_name=str(getattr(chunk, "file_name", "")),
                                                 page=int(getattr(chunk, "page", 0)),
                                                 chunk_id=str(getattr(chunk, "chunk_id", "")) or None)
                    evidence = engine.generator.evidence_lookup(cite)
                    ok, why = citation_mod.answer_support_check(kept_body, evidence)
                    emit(f"    块 {cite.chunk_id} → 可核={ok} {why[:90]}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"_t16_field_diag_{args.qid}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        emit(f"（诊断摘要 JSON 长度 {len(json.dumps(corpus, ensure_ascii=False))}；产物 {out_path}）")
        span.set_output({"qid": args.qid, "keywords": keywords})
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：非零退出，绝不静默
        import traceback

        print(json.dumps({"event": "diag_t16_field.failed", "level": "ERROR",
                          "error_type": type(exc).__name__, "message": str(exc),
                          "stack": traceback.format_exc()}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
