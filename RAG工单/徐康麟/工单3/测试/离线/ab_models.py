# -*- coding: utf-8 -*-
"""T6 生成模型 A/B：同一提示词下比较各本地模型的「正文合规率 / 引用合规率 / 首字延迟」。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

目的：qwen2.5:3b 在实测中经常只回引用行或照抄片段标签（正文为空），需要用数据决定：
    (a) 换更强的本地模型（deepseek-r1:7b 等）是否显著提升合规率且首字仍 ≤3 s；
    (b) 或继续在小模型上叠加确定性兜底。
只做只读生成，不改任何数据；结果打印为表格。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/ab_models.py --models qwen2.5:3b,deepseek-r1:7b --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import citation as citation_mod  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import setup_logging, shutdown_logging, get_logger  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402

EVAL_SET = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"


def main(argv: list[str] | None = None) -> int:
    """逐模型跑同一批题，输出合规率与延迟。"""
    parser = argparse.ArgumentParser(description="T6 生成模型 A/B")
    parser.add_argument("--models", default="qwen2.5:3b,deepseek-r1:7b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-think", default="false", choices=["true", "false"],
                        help="推理模型是否允许思维链（false → 传 think=false 直接作答）")
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("ab_models")
    retriever = build_retriever(cfg=cfg)
    items = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    prompts = {str(it["id"]): (it["question"],
                             retriever.retrieve(it["question"], top_k=cfg.retrieval.top_k, logger=log).chunks)
               for it in items}

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        os.environ["RAG_LLM__OLLAMA_GEN_MODEL"] = model
        os.environ["RAG_LLM__TEMPERATURE"] = str(args.temperature)
        os.environ["RAG_LLM__MAX_TOKENS"] = str(args.max_tokens)
        os.environ["RAG_LLM__ENABLE_THINK"] = args.enable_think
        import importlib

        from app.core import config as config_mod, llm_client, generator as generator_mod

        importlib.reload(config_mod)
        importlib.reload(llm_client)
        importlib.reload(generator_mod)
        cfg2 = config_mod.get_config()
        gen = generator_mod.build_generator(cfg=cfg2, logger=log)
        if gen.llm is None:
            print(f"== {model}: 后端不可用，跳过")
            continue
        # 冷启动预热（不计入首字统计）
        warm = gen.llm.generate_full("预热：只回答「好」。", max_tokens=4, temperature=0.0, logger=log)
        print(f"\n== 模型 {model}（温度 {args.temperature}，预热 {warm.total_ms:.0f} ms）")
        print(f"{'题':<6}{'首字ms':>9}{'总ms':>9}{'正文':>5}{'引用':>5}  正文首行")
        ok_body = ok_cite = 0
        firsts: list[float] = []
        for qid, (question, chunks) in prompts.items():
            prompt = gen.build_prompt(question, chunks, language="zh")
            res = gen.llm.generate_full(prompt, trace_id=f"ab{qid}", logger=log)
            text = res.text
            body = citation_mod.answer_body(text)
            has_cite = bool(citation_mod.parse_citations(text))
            body_ok = not citation_mod.is_empty_answer(text) and body.strip() != citation_mod.UNKNOWN_TEXT
            ok_body += int(body_ok)
            ok_cite += int(has_cite)
            firsts.append(float(res.first_token_ms))
            head = body.splitlines()[0][:46] if body else ""
            print(f"{qid:<6}{res.first_token_ms:9.0f}{res.total_ms:9.0f}{('有' if body_ok else '空'):>5}"
                  f"{('有' if has_cite else '无'):>5}  {head}")
        print(f"   合规率：正文 {ok_body}/{len(prompts)}，引用 {ok_cite}/{len(prompts)}；"
              f"首字 max {max(firsts):.0f} ms")
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
