# -*- coding: utf-8 -*-
"""命令行多轮对话（在线链路入口）。

用法：
    # 离线兜底（不连任何真实后端）
    python scripts/chat_cli.py --offline --once "民间借贷的利率上限是多少"

    # 真实 Ollama（本机 127.0.0.1:11434 的 qwen2.5:3b）
    python scripts/chat_cli.py --provider ollama --once "民间借贷的利率上限是多少？请用一句话回答。" --no-stream
    python scripts/chat_cli.py --provider ollama --once "什么是合同违约？"        # 默认流式

    # 交互式多轮
    python scripts/chat_cli.py --provider ollama --role 律师 --metrics

    # 跨机测试（Windows 用 VM 的 Ollama）
    python scripts/chat_cli.py --provider ollama --base-url http://192.168.188.128:11434 --once "你好"

``--metrics`` 会在退出前打印本次运行的首 token 延迟、两种口径的 tok/s、
预填充/解码/模型加载耗时等指标摘要（日志里另有 ``METRIC {json}`` 完整行）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag import metrics as M  # noqa: E402
from legal_rag.config import KNOWLEDGE_DIR, RagConfig  # noqa: E402
from legal_rag.engine import RagEngine  # noqa: E402
from legal_rag.generate.llm_metrics import last_stats  # noqa: E402
from legal_rag.ingest.pipeline import KnowledgeBasePipeline, group_sources_by_role  # noqa: E402
from legal_rag.logging_setup import setup_logging  # noqa: E402
from legal_rag.observability import bind_request, reset_request  # noqa: E402
from legal_rag.roles import DEFAULT_ROLE_ID, get_role, list_roles  # noqa: E402
from legal_rag.schemas import Message  # noqa: E402

PROVIDER_CHOICES = ("mock", "ollama", "deepseek", "openai_compat")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="法律 RAG 命令行对话")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：不下载模型、不外呼，全程使用兜底实现")
    parser.add_argument("--provider", default=None, choices=PROVIDER_CHOICES,
                        help="大模型后端；默认读 LLM_PROVIDER（本阶段主线：ollama）")
    parser.add_argument("--model", default=None, help="覆盖模型名，例如 qwen2.5:3b")
    parser.add_argument("--base-url", default=None,
                        help="Ollama base_url，例如 http://192.168.188.128:11434")
    parser.add_argument("--role", default=None, help="角色（role_id 或中文名，如 lawyer / 律师）")
    parser.add_argument("--once", default=None, help="只问一个问题就退出")
    parser.add_argument("--session-id", default="cli-session", help="会话 ID")
    parser.add_argument("--user-id", default="cli-user", help="用户 ID")
    parser.add_argument("--top-k", type=int, default=None, help="最终保留的知识条数")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--source", action="append", default=None, help="知识来源，默认 knowledge/")
    parser.add_argument("--metrics", action="store_true", help="结束时打印本次请求的 LLM 指标摘要")
    parser.add_argument("--list-roles", action="store_true", help="列出角色库后退出")
    return parser


def ensure_index(engine: RagEngine, sources: list[str]) -> None:
    """索引为空时先建库（内存库每次进程启动都是空的）。

    与 ``build_index.py`` / 服务启动一致：按 ``knowledge/<role_id>/`` 目录**分角色入库**，
    否则命令行试出来的回答会与实际服务的隔离行为不一致。
    """
    if engine.store.count() > 0:
        return
    print(f"[建库] 索引为空，正在把 {', '.join(sources)} 写入知识库 …")
    pipeline = KnowledgeBasePipeline(engine.embedder, engine.store, engine.config)
    for role_id, paths in group_sources_by_role(sources, default_role=DEFAULT_ROLE_ID):
        report = pipeline.ingest(paths, rebuild=True, role_id=role_id)
        print(f"[建库] role_id={role_id or '(未标注)'} {report.summary()}")


def print_answer(answer, header: str = "助手") -> None:
    print(f"\n{header}[{answer.provider}]：{answer.text}")
    if answer.citations:
        print("\n引用来源：")
        for index, cite in enumerate(answer.citations, start=1):
            print(f"  [{index}] {cite['source']}（相关度 {cite['score']}）")
            print(f"      {cite['snippet'][:100]}")


def print_citations(answer) -> None:
    if not answer.citations:
        return
    print("\n引用来源：")
    for index, cite in enumerate(answer.citations, start=1):
        print(f"  [{index}] {cite['source']}（相关度 {cite['score']}）")


#: 本次请求最值得看的字段（顺序即打印顺序）
_DETAIL_KEYS = (
    "provider", "model", "role_id", "stream", "ok", "degraded",
    "degraded_from", "degraded_to",
    "prompt_chars", "output_chars", "prompt_tokens", "output_tokens",
    "ttft_seconds", "total_seconds", "load_seconds", "prefill_seconds", "decode_seconds",
    "tokens_per_second_official", "tokens_per_second_wall",
    "cold_start", "keep_alive", "attempts", "error",
)


def print_metrics_summary() -> None:
    """打印指标摘要：先本次请求的原始值，再累计分布（P50/P95）。"""
    detail = last_stats() or {}
    print("\n" + "=" * 64)
    print("LLM 指标摘要")
    if detail:
        print("  本次请求：")
        for key in _DETAIL_KEYS:
            if key in detail and detail[key] is not None:
                print(f"    {key} = {detail[key]}")
    else:
        print("  本次请求：无 LLM 指标（未真正调用大模型？）")

    flattened = M.snapshot().values()
    rows = sorted((k, v) for k, v in flattened.items() if k.startswith("llm_"))
    if rows:
        print("  本次进程累计（histogram 取 avg）：")
        for name, value in rows:
            print(f"    {name} = {value:.4g}")
    print("=" * 64)


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    if args.list_roles:
        for role in list_roles():
            print(f"  {role.role_id:<18} {role.name:<10} [{role.domain}]")
        return 0

    if args.offline:
        os.environ["RAG_OFFLINE"] = "1"

    config = RagConfig.from_env()

    if args.offline and args.provider and args.provider != "mock":
        print(f"[离线模式] --offline 强制 LLM_PROVIDER=mock，忽略 --provider {args.provider}")
    elif args.provider:
        config.llm_provider = args.provider

    if args.model:
        config.llm_model = args.model
        config.ollama.llm_model = args.model
    if args.base_url:
        config.ollama.base_url = args.base_url

    config.ensure_dirs()
    sources = args.source or [str(KNOWLEDGE_DIR)]

    role = get_role(args.role)
    engine = RagEngine(config)
    ensure_index(engine, sources)

    print("=" * 64)
    chain = engine.router.chain()
    primary = engine.router.client_for(chain[0])
    print(f"角色：{role.name}（{role.role_id}）  领域：{role.domain}")
    print(f"向量化：{engine.embedder.name}   向量库：{engine.store.name}   "
          f"重排：{engine.reranker.name}   大模型链路：{' -> '.join(chain)}")
    print(f"模型：{primary.model}   base_url：{getattr(primary, 'base_url', 'n/a')}   "
          f"超时：{config.llm_timeout:g}s   流式：{not args.no_stream}")
    print(f"知识库 chunk 数：{engine.store.count()}   会话：{args.session_id}")
    print("=" * 64)

    history: list[Message] = []
    window = config.session_window

    def answer_once(question: str) -> None:
        if args.no_stream:
            answer = engine.ask(question, role=role, session_id=args.session_id,
                                user_id=args.user_id, history=history, top_k=args.top_k)
            print_answer(answer)
        else:
            answer, pieces = engine.ask_stream(question, role=role, session_id=args.session_id,
                                               user_id=args.user_id, history=history,
                                               top_k=args.top_k)
            print(f"\n助手[{answer.provider}]：", end="", flush=True)
            for piece in pieces:
                print(piece, end="", flush=True)
            print()
            print_citations(answer)

        history.append(Message(role="user", content=question))
        history.append(Message(role="assistant", content=answer.text))
        if window > 0 and len(history) > window:
            del history[:-window]

    # 绑定 request 上下文：此后所有日志自动带 request_id，指标也带上 role_id 标签
    token = bind_request(role_id=role.role_id, session_id=args.session_id,
                         user_id=args.user_id, path="chat_cli")
    try:
        if args.once:
            answer_once(args.once)
            return 0

        print("输入问题开始对话；输入 exit / quit 退出，输入 :roles 查看角色。")
        while True:
            try:
                question = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", ":q"):
                break
            if question == ":roles":
                for item in list_roles():
                    print(f"  {item.role_id:<18} {item.name}")
                continue
            answer_once(question)
    finally:
        reset_request(token)
        if args.metrics:
            print_metrics_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
