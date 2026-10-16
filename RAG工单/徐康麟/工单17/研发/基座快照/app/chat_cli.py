# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

多轮对话命令行（CLI）—— 工单5 的交互入口之一（另两个是 Gradio 与标准库 Web）。

用法（工作目录 = 仓库根，统一走 run_py.ps1）：

    # 单次多轮提问（带会话号，可连续追问）
    pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --ask "这个公司的法定代表人是谁？" --session-id s1

    # 交互式多轮（推荐演示用；每轮都会打印「改写后的独立问句」）
    pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --chat

    # 清空会话 / 查看会话状态 / 健康检查
    pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --reset --session-id s1
    pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --state --session-id s1
    pwsh -NoProfile -File run_py.ps1 研发/app/chat_cli.py --health

退出码：``0`` 正常 / ``2`` 参数错误 / ``3`` 索引缺失 / ``4`` 依赖缺失。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

#: 仓库根：研发/app/chat_cli.py → parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

EXIT_OK, EXIT_PARAM, EXIT_INDEX, EXIT_DEPS = 0, 2, 3, 4

#: 交互模式的帮助文本
_HELP = """可用命令：
  /help          显示本帮助
  /reset         清空当前会话（焦点实体与历史一并清空）
  /state         显示当前会话状态（焦点实体 / 上一轮槽位 / 实体栈）
  /history       显示最近几轮问答
  /rewrite on|off  是否显示改写后的独立问句（默认 on）
  /quit  或  /exit  退出
其它任意输入按「提问」处理。"""


def _print_turn(result: Any, *, show_rewrite: bool = True, as_json: bool = False) -> None:
    """打印一轮问答（含改写对照）。"""
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return
    lat = result.latency_ms or {}
    if show_rewrite and result.strategy != "none":
        print(f"  ↳ 改写 : {result.rewritten_query}")
        refs = "、".join(f"{r.get('surface')}→{r.get('replacement')}"
                         for r in (result.rewrite or {}).get("resolved_refs") or [])
        print(f"  ↳ 策略 : {result.strategy}（{refs}）")
    elif show_rewrite:
        print(f"  ↳ 策略 : none（问句已独立，无需改写）")
    print(f"  ↳ 焦点 : {result.focus_entity}")
    print(f"  A: {result.answer}")
    print(f"  ⏱ 改写 {lat.get('rewrite', 0):.1f} ms | 检索 {lat.get('retrieve', 0):.0f} ms | "
          f"生成 {lat.get('generate', 0):.0f} ms | 合计 {lat.get('total', 0):.0f} ms")


def _interactive(engine: Any, *, session_id: str, lang: str,
                 use_cache: bool) -> int:
    """交互式多轮 REPL。"""
    show_rewrite = True
    print(f"多轮对话已就绪（工单：{WORK_ORDER}）")
    print(f"会话号：{session_id}　输入 /help 查看命令，/quit 退出。\n")
    while True:
        try:
            question = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return EXIT_OK
        if not question:
            continue
        low = question.lower()
        if low in ("/quit", "/exit", ":q"):
            print("再见。")
            return EXIT_OK
        if low in ("/help", "/?"):
            print(_HELP)
            continue
        if low == "/reset":
            engine.reset(session_id)
            print("已清空当前会话。\n")
            continue
        if low == "/state":
            print(json.dumps(engine.state(session_id), ensure_ascii=False, indent=1))
            print()
            continue
        if low == "/history":
            for i, t in enumerate(engine.history(session_id), 1):
                print(f"  [{i}] Q: {t['question']}\n      R: {t['rewritten_query']}\n      A: {t['answer'][:120]}")
            print()
            continue
        if low.startswith("/rewrite"):
            show_rewrite = not low.endswith("off")
            print(f"改写显示已{'开启' if show_rewrite else '关闭'}。\n")
            continue
        result = engine.ask(question, session_id=session_id, lang=lang, use_cache=use_cache)
        _print_turn(result, show_rewrite=show_rewrite)
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat_cli.py",
        description=f"多轮对话命令行（工单：{WORK_ORDER}）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_HELP,
    )
    parser.add_argument("--ask", help="单次多轮提问（配合 --session-id 可连续追问）")
    parser.add_argument("--chat", action="store_true", help="进入交互式多轮对话")
    parser.add_argument("--session-id", default="cli", help="会话号（默认 cli）")
    parser.add_argument("--lang", default="auto", choices=["zh", "en", "auto"], help="语言路径")
    parser.add_argument("--json", action="store_true", help="机器可读输出（单行 JSON）")
    parser.add_argument("--no-rewrite", action="store_true", help="不显示改写后的独立问句")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存（延迟测量口径用）")
    parser.add_argument("--reset", action="store_true", help="清空该会话后退出")
    parser.add_argument("--state", action="store_true", help="打印会话状态后退出")
    parser.add_argument("--history", action="store_true", help="打印会话历史后退出")
    parser.add_argument("--health", action="store_true", help="打印健康检查后退出")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（仅调试）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = [bool(args.ask), bool(args.chat), bool(args.reset), bool(args.state),
               bool(args.history), bool(args.health)]
    if sum(actions) == 0:
        build_parser().print_help()
        return EXIT_PARAM
    if sum(actions) > 1:
        print("[参数错误] --ask / --chat / --reset / --state / --history / --health 只能选一个",
              file=sys.stderr)
        return EXIT_PARAM

    try:
        from app.core.errors import IndexMissingError
        from app.multiturn_engine import MultiTurnEngine
    except ImportError as exc:  # noqa: BLE001
        print(f"[依赖缺失] 无法导入多轮引擎：{exc}\n"
              f"请确认工作目录为仓库根（{REPO_ROOT}），并统一用 run_py.ps1 启动。", file=sys.stderr)
        return EXIT_DEPS

    try:
        engine = MultiTurnEngine()
        if not args.no_warmup:
            warm = engine.warmup(with_generate=True)
            if not args.json:
                print(f"[预热] 索引 {warm['base'].get('index_chunks')} 块，"
                      f"实体 {warm['entities']} 个，耗时 {warm['elapsed_ms']:.0f} ms")
    except IndexMissingError as exc:
        print(f"[索引缺失] {exc}", file=sys.stderr)
        return EXIT_INDEX
    except Exception as exc:  # noqa: BLE001
        print(f"[启动失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INDEX

    sid = args.session_id or "cli"
    use_cache = not args.no_cache

    if args.health:
        print(json.dumps(engine.health(), ensure_ascii=False, indent=1))
        return EXIT_OK
    if args.reset:
        engine.reset(sid)
        print(f"已清空会话 {sid}")
        return EXIT_OK
    if args.state:
        print(json.dumps(engine.state(sid), ensure_ascii=False, indent=1))
        return EXIT_OK
    if args.history:
        print(json.dumps(engine.history(sid), ensure_ascii=False, indent=1))
        return EXIT_OK
    if args.chat:
        return _interactive(engine, session_id=sid, lang=args.lang, use_cache=use_cache)

    started = time.perf_counter()
    result = engine.ask(args.ask, session_id=sid, lang=args.lang, use_cache=use_cache)
    if not args.json:
        print(f"Q: {args.ask}")
    _print_turn(result, show_rewrite=not args.no_rewrite, as_json=args.json)
    if not args.json:
        print(f"（墙钟 {((time.perf_counter() - started) * 1000):.0f} ms，会话 {sid}）")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
