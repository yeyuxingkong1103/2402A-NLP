"""S10 答案生成的命令行入口。**本特性的进程外接口。**

运行方式（MUST 用模块形式，以仓库根为工作目录）：

    D:/zg6_Project/9/med_rag/rag/python.exe -m backend.generate_answer "血压高平时要注意什么"
    D:/zg6_Project/9/med_rag/rag/python.exe -m backend.generate_answer "氢氯噻嗪" clinician

契约见 `docs/05_接口设计.md` §4.3。

---

## 为什么要有它

服务端的生成链路要走 SSE、要起 HTTP、要加载 2.3 GB 权重 —— 调提示词时这些
全是噪声。CLI 把链路缩到最短：**一条问题进去，一段答案出来**，中间的两路
（检索、生成）各自把关键信息打出来。改提示词时用它迭代，比刷新浏览器快得多。

**它和服务端跑的是同一份实现**（`backend/generate/service.py` 的 `Generation`）
—— CLI 只做参数解析与输出渲染，MUST NOT 自己实现生成。
"""

import argparse
import sys
import traceback

from backend.generate import MODES, PromptError
from backend.generate.client import LLMError
from backend.generate.service import run as run_coro
from backend.generate.service import start as start_generation
from backend.retrieve import EXIT_ARGS, EXIT_DATA, EXIT_DEP, EXIT_OK, RetrievalError

__all__ = ["main"]

# 退出码沿用 `docs/05` §5 的分工（1 参数 / 2 数据 / 3 外部依赖），
# 与 `backend/retrieve_search.py`、`backend/query_embed.py` 一致。
EXIT_DESCRIPTION = {
    EXIT_OK: "成功",
    EXIT_ARGS: "参数错误",
    EXIT_DATA: "数据/校验问题",
    EXIT_DEP: "外部依赖不可用",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backend.generate_answer",
        description="医知源 · 检索 → 生成（一条命令跑完整条链路）",
    )
    parser.add_argument("question", help="要提问的问题原文")
    parser.add_argument(
        "mode", nargs="?", default=None,
        choices=list(MODES),
        help="回答模式（默认 patient）",
    )
    parser.add_argument("--top-k", type=int, default=None, help="检索返回条数（覆盖 TOP_K）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="语义路余弦阈值（覆盖 SIMILARITY_THRESHOLD）")
    parser.add_argument("--show-prompt", action="store_true",
                        help="打印渲染后的完整提示词（调提示词时用）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)

    try:
        return _run(args)
    except RetrievalError as exc:
        print("检索失败：%s" % exc.message, file=sys.stderr)
        return exc.code
    except PromptError as exc:
        print("提示词错误：%s" % exc.message, file=sys.stderr)
        return EXIT_DATA
    except LLMError as exc:
        print("生成失败：%s" % exc.message, file=sys.stderr)
        print(
            "\n若提示 invalid_api_key，请在仓库根 .env 里填入真实的 AGICTO_API_KEY。",
            file=sys.stderr,
        )
        return EXIT_DEP
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return EXIT_ARGS
    except Exception as exc:  # noqa: BLE001 —— 兜底，避免把堆栈直接抛给使用者
        print("未预期的错误：%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        traceback.print_exc()
        return EXIT_DATA


def _run(args: argparse.Namespace) -> int:
    import json

    from backend.api.config import AppConfig
    from backend.retrieve import bundle as bundle_module
    from backend.retrieve.cli_support import build_index, encode
    from backend.retrieve.service import search as retrieve_search

    config = AppConfig()

    # ⚠️ 密钥检查放在**最前面**，早于加载 BGE-M3 权重（约 10 s）与连 Milvus。
    #
    # 与 `backend/serve.py` 的"便宜的检查先跑"同一取向。缺密钥时用户要等 10 秒
    # 才被告知一句"该填 .env"—— 而那句话一个字都不依赖模型。
    if not config.agicto_api_key:
        raise LLMError(
            "缺少 AGICTO_API_KEY。请在仓库根创建 .env（可复制 .env.example）并填入：\n"
            "    AGICTO_API_KEY=<你的密钥>"
        )

    # 与 `backend/retrieve_search.py` 同样的装配方式：CLI 每次重建索引
    # （不同进程，不复用服务端的常驻单例）。
    class _Args:
        candidates = rrf_k = admit_rank = min_coverage = None

    index = build_index(_Args())
    bundle_module.set_index(index)

    top_k = args.top_k or config.top_k
    threshold = args.threshold if args.threshold is not None else config.similarity_threshold

    print("检索中…", file=sys.stderr, flush=True)
    retrieval = retrieve_search(args.question, encode(args.question), top_k, threshold)
    print(
        "检索：%d 条，is_empty=%s below_threshold=%s"
        % (len(retrieval.passages), retrieval.is_empty, retrieval.below_threshold),
        file=sys.stderr,
        flush=True,
    )

    if retrieval.is_empty:
        # 与 `docs/05` §4.3 的约束一致：passages 为空 MUST NOT 走到模型
        # （"无上下文生成"是被明令禁止的）。这里直接按拒答处理。
        print("\n检索为空或全部低于阈值 —— 按契约不调用模型，直接拒答。")
        return EXIT_OK

    generation = start_generation(
        args.question, retrieval.passages, mode=args.mode or "patient", config=config
    )

    if args.show_prompt:
        print("=" * 60)
        print(generation.prompt_text)
        print("=" * 60)

    print("生成中…", file=sys.stderr, flush=True)
    run_coro(_consume(generation))

    result = generation.result
    if result is None:  # pragma: no cover —— _consume 正常返回则必然已填
        print("生成未产出结果。", file=sys.stderr)
        return EXIT_DATA

    if args.json:
        print(json.dumps(result.to_record(), ensure_ascii=False, indent=2))
        return EXIT_OK

    print("\n===== 正文 =====")
    print(result.answer_text)
    print("\n===== 来源 =====")
    for source in result.sources:
        pages = (
            "第 %d 页" % source["page_start"]
            if source["page_start"] == source["page_end"]
            else "第 %d–%d 页" % (source["page_start"], source["page_end"])
        )
        print("  【%d】%s  %s  %s"
              % (source["citation_id"], source["file_name"], pages, source["section"] or ""))
    print("\n===== 免责声明 =====")
    print(result.disclaimer)
    print(
        "\nconfidence=%s  引用=%s  引用失败=%s"
        % (result.confidence, result.used_citation_ids, result.is_citation_failure)
    )
    return EXIT_OK


async def _consume(generation) -> None:
    """把流耗光。CLI 不逐段打印 —— 与非流式结果对照时，中间态反而是噪声。"""

    async for _ in generation.deltas():
        pass


def _force_utf8_output() -> None:
    """把标准输出/错误切到 UTF-8。Windows 控制台默认 cp936，中文会乱码。

    与 `backend/serve.py`、`backend/retrieve_search.py` 的同名函数是同一件事，
    刻意不共用（调用时机不同，抽出去只多一条依赖，收益是三行代码）。
    """

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


if __name__ == "__main__":
    raise SystemExit(main())
