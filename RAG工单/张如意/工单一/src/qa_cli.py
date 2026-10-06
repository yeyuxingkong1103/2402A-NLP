# -*- coding: utf-8 -*-
"""
工单01 命令行问答入口（问答引擎）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

功能：
  1. 单问模式：-q/--question 直接提问一次
  2. 交互模式：-i/--interactive 连续问答（支持 :k/:ctx/:help 等命令）
  3. 语音输入：--voice 调用 speech_input.py 识别麦克风语音后提问
  4. 可指定 top_k、检索策略、流水线预设，可打印检索上下文便于溯源

用法：
    python "工单01-基于PDF文档的问答系统/src/qa_cli.py" -q "法定代表人是谁？"
    python .../src/qa_cli.py -i
    python .../src/qa_cli.py -i --preset wo06_hybrid --top-k 8 --show-context
    python .../src/qa_cli.py -q "Who is the legal representative?"   # 英文提问
    python .../src/qa_cli.py --voice -i                              # 语音输入

提示：首次使用请先运行 build_index.py 建立索引。
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

# --- 让脚本可以独立运行：把项目根目录（工单作业/）加入模块搜索路径 ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 同级脚本（speech_input 等）所在目录
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_core import config                        # noqa: E402
from rag_core.pipeline import PRESETS, Pipeline    # noqa: E402

LATENCY_BUDGET_S = 3.0      # 工单性能验收：从提问到返回答案不超过 3 秒


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="工单01：招股说明书问答系统命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="交互模式内置命令：\n"
               "  :k N      调整返回片段数（默认取预设 top_k）\n"
               "  :ctx      开/关检索上下文展示\n"
               "  :help     显示帮助\n"
               "  exit      退出（也可用 :q / quit / Ctrl+C）\n",
    )
    p.add_argument("-q", "--question", default=None, help="单次提问的问题文本")
    p.add_argument("-i", "--interactive", action="store_true", help="进入交互式问答")
    p.add_argument("--top-k", type=int, default=None,
                   help="送入 LLM 的片段数，默认取预设值（工单01 预设为 5）")
    p.add_argument("--preset", default="wo01_baseline", choices=sorted(PRESETS),
                   help="流水线预设，默认 wo01_baseline")
    p.add_argument("--collection", default="prospectus", help="向量库集合名")
    p.add_argument("--strategy", default=None,
                   choices=["vector", "fulltext", "hybrid"],
                   help="覆盖检索策略（默认取预设）")
    p.add_argument("--reranker", default=None,
                   choices=["none", "llm", "tfidf", "adaptive", "cascade"],
                   help="覆盖重排器（默认取预设）")
    p.add_argument("--show-context", action="store_true",
                   help="打印检索到的片段原文（溯源/调试用）")
    p.add_argument("--voice", action="store_true",
                   help="提问前先做一次语音识别（需安装 SpeechRecognition 或 Whisper）")
    p.add_argument("--lang", default="zh-CN",
                   help="语音识别语言，默认 zh-CN（英文用 en-US）")
    return p


# ---------------------------------------------------------------------------
# 流水线装载
# ---------------------------------------------------------------------------
def load_pipeline(args) -> Pipeline | None:
    """装载流水线并校验索引是否就绪，失败时打印可操作的修复提示。"""
    # 复制一份预设再覆盖，避免污染 rag_core.PRESETS 全局字典
    overrides = {}
    if args.strategy:
        overrides["strategy"] = args.strategy
    if args.reranker:
        overrides["reranker"] = args.reranker
    cfg = replace(PRESETS[args.preset], **overrides) if overrides else PRESETS[args.preset]

    pipeline = Pipeline(cfg, collection=args.collection)
    try:
        pipeline.load_index()
        n_vec = pipeline.retriever.vs.count()
    except Exception as e:
        n_vec = 0
        print(f"[警告] 索引装载失败：{e}")

    if n_vec == 0:
        print("[错误] 向量索引为空或不存在，无法问答。")
        print("       请先执行：python "
              '"工单01-基于PDF文档的问答系统/src/build_index.py"')
        return None

    print(f"[就绪] 预设={args.preset} 集合={args.collection} 向量数={n_vec} "
          f"策略={cfg.strategy} 重排={cfg.reranker} top_k="
          f"{args.top_k or cfg.top_k}")
    return pipeline


# ---------------------------------------------------------------------------
# 单问
# ---------------------------------------------------------------------------
def print_answer(result: dict, elapsed: float, show_context: bool = False) -> None:
    """统一格式输出答案、引用、耗时与上下文。"""
    print("\n" + "-" * 66)
    print(f"问题：{result.get('question', '')}")
    print("-" * 66)
    print("答案：")
    print(result.get("answer", "（无答案）").strip())

    citations = result.get("citations") or []
    if citations:
        print("\n引用来源：")
        for c in citations:
            section = f"  章节：{c['section']}" if c.get("section") else ""
            print(f"  · 《{c.get('doc', '')}》第{c.get('page', '?')}页"
                  f"（{c.get('type', 'text')}，{c.get('chunk_id', '')}）{section}")
            if c.get("snippet"):
                print(f"    摘录：{c['snippet']}")
    else:
        print("\n引用来源：（模型未标注来源，可加 --show-context 查看检索片段）")

    timings = result.get("timings", {}) or {}
    retrieve = timings.get("retrieve") or timings.get("total")
    if retrieve is None and timings:
        retrieve = sum(v for k, v in timings.items()
                       if k not in ("generate", "total", "query_understanding"))
    gen = timings.get("generate")
    detail = ""
    if retrieve is not None:
        detail += f"检索 {retrieve * 1000:.0f} ms"
    if gen is not None:
        detail += f" / 生成 {gen * 1000:.0f} ms"
    ok = "[满足 3 秒要求]" if elapsed <= LATENCY_BUDGET_S else "[超出 3 秒预算]"
    print(f"\n耗时：{elapsed * 1000:.0f} ms"
          + (f"（{detail}）" if detail else "") + f"  {ok}")

    if show_context:
        docs = result.get("docs", [])
        print(f"\n检索片段（top {len(docs)}）：")
        for i, d in enumerate(docs, 1):
            score = d.get("final_score", d.get("score", 0))
            print(f"  [片段{i}] 《{d.get('doc', '')}》第{d.get('page', '?')}页 "
                  f"score={float(score):.4f} chunk={d.get('chunk_id', '')}")
            text = (d.get("text") or "").replace("\n", " ")
            print(f"      {text[:200]}{'…' if len(text) > 200 else ''}")
    print("-" * 66)


def ask_once(pipeline: Pipeline, question: str, top_k: int | None,
             show_context: bool = False) -> dict | None:
    """执行一次问答，捕获异常保证交互模式不中断（容错机制）。"""
    question = (question or "").strip()
    if not question:
        print("[提示] 问题为空，请重新输入。")
        return None
    if len(question) > 500:
        print(f"[提示] 问题过长（{len(question)} 字），已截断到 500 字。")
        question = question[:500]

    t0 = time.perf_counter()
    try:
        result = pipeline.ask(question, top_k=top_k, return_trace=True)
    except Exception as e:                       # 网络/模型异常不应导致程序退出
        print(f"[错误] 本次问答失败：{e}")
        print("       请检查 DEEPSEEK_API_KEY、网络连接与索引是否就绪。")
        return None
    elapsed = time.perf_counter() - t0
    result.setdefault("question", question)
    print_answer(result, elapsed, show_context=show_context)
    return result


# ---------------------------------------------------------------------------
# 交互模式
# ---------------------------------------------------------------------------
HELP_TEXT = """可用命令：
  :k N     设置返回片段数（如 :k 8）
  :ctx     开/关检索上下文展示
  :help    显示本帮助
  exit     退出（:q / quit / Ctrl+C 亦可）"""


def interactive_loop(pipeline: Pipeline, args) -> None:
    top_k = args.top_k
    show_context = args.show_context
    print("\n进入交互式问答（输入 :help 查看命令，exit 退出）")
    if args.voice:
        print("[语音模式] 每次回车后开始录音，说完自动识别。")

    while True:
        try:
            raw = (input("\n请输入问题> ")
                   if not args.voice else _voice_question(args))
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if raw is None:
            continue
        text = raw.strip()
        if not text:
            continue
        low = text.lower()
        if low in ("exit", "quit", ":q", ":quit"):
            print("已退出。")
            return
        if low == ":help":
            print(HELP_TEXT)
            continue
        if low.startswith(":k"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = max(1, min(int(parts[1]), 20))
                print(f"[设置] top_k = {top_k}")
            else:
                print("[提示] 用法：:k 8")
            continue
        if low == ":ctx":
            show_context = not show_context
            print(f"[设置] 上下文展示 = {'开' if show_context else '关'}")
            continue
        ask_once(pipeline, text, top_k, show_context=show_context)


def _voice_question(args) -> str | None:
    """调用语音模块获取一句问题；依赖缺失时给出安装提示并退回键盘输入。"""
    try:
        import speech_input
        text = speech_input.listen_once(lang=args.lang, engine="auto")
    except ImportError as e:
        print(f"[提示] 语音模块不可用：{e}")
        print("       请安装：pip install SpeechRecognition pyaudio")
        return None
    except Exception as e:
        print(f"[提示] 语音识别失败：{e}")
        return None
    if not text:
        return None
    print(f"[语音识别] {text}")
    return text


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    if not args.question and not args.interactive:
        build_parser().print_help()
        print("\n[提示] 请用 -q 提问，或加 -i 进入交互模式。")
        return 1
    if args.preset not in PRESETS:
        print(f"[错误] 未知预设：{args.preset}")
        return 2

    pipeline = load_pipeline(args)
    if pipeline is None:
        return 3

    if args.question:
        ask_once(pipeline, args.question, args.top_k,
                 show_context=args.show_context)
    if args.interactive:
        interactive_loop(pipeline, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
