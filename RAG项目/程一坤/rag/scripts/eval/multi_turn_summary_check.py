# -*- coding: utf-8 -*-
"""会话摘要 + 查询改写 的「真实多轮」链路检查（不是评测集，是链路探针）。

为什么需要它：
`evaluation/run_eval.py` 每题用独立 session（`eval_<题号>`）且直接调 `ChatService.chat`，
**不走 `api/chat_persistence.persist_turn`** —— 窗口永远不满、Redis 里没有 key，
摘要**不会触发**、改写也拿不到上文。因此"重跑评测"无法证明摘要/改写真的可用。
本脚本按真实多轮顺序连跑 N 轮（真实检索 + 真实 LLM），每轮：
  1. 读当前短期记忆 → 按真实链路顺序做一次查询改写（记录是否改写 + 改写后的查询文本）
  2. 真实问答（`ChatService.chat`）
  3. 等价于 `persist_turn` 的 Redis 回写（append user/assistant）
  4. 等价于 `persist_turn` 后台线程的摘要更新（此处同步调用，便于观察）
跑完打印：① LLM 真实压出来的摘要原文（存 Redis，非预置）
          ② 某一轮实际注入提示词的「# 本次会话前情」段全文
          ③ 每轮改写记录汇总

用法：
    python scripts/eval/multi_turn_summary_check.py                # 默认 12 轮
    python scripts/eval/multi_turn_summary_check.py --rounds 14
    python scripts/eval/multi_turn_summary_check.py --rounds 2 --keep     # 只看前 2 轮改写

退出码：0 = 摘要生成 + 前情注入均观察到；1 = 跑了足够轮数仍没触发（真失败）；
        2 = 前置/参数不满足（开关关闭、轮数不足以触发窗口）；3 = 前置服务不可用。

前置：Redis / MySQL / Milvus 在线 + LLM 可用（先跑 `python scripts/check_services.py`）。
跑完自动删除探针 key（Redis），不写 MySQL（`chat_store=None`）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 项目根 = scripts/eval/ 向上两层
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_project_env  # noqa: E402

# 除 _env 剔除的 4 个外，all_proxy 同样会劫持外网调用
for _name in ("all_proxy", "ALL_PROXY"):
    os.environ.pop(_name, None)

load_project_env(PROJECT_ROOT)

# 本脚本就是要验"开关打开"的行为，显式覆盖（已在环境里的同名变量优先，故用直接赋值）
os.environ["SESSION_SUMMARY_ENABLED"] = "true"

from app.chat.service import ChatService  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.memory.summary_service import maybe_update_session_summary  # noqa: E402
from app.models.llm import build_chat_client_from_settings  # noqa: E402
from app.retrieval.assembly import build_default_retrieval_service, rewrite_query_safely  # noqa: E402
from app.retrieval.query_rewrite import QueryRewriter  # noqa: E402

USER_ID = "probe_b22_user"
SESSION_ID = "probe_b22_multiturn"

# 同一主题的连续追问：让"前情"真的有内容可压。
# 末两条故意写成**省略式追问**（"这个"/"它"）：改写器只在有指代信号时才翻历史，
# 全是自足问题的题目集跑不出改写记录（实测 12 轮全是 改写=no）。
QUESTIONS = [
    "经济补偿怎么算",
    "工作年限怎么认定",
    "月工资超过社平工资三倍怎么办",
    "什么情况下不用支付经济补偿",
    "违法解除劳动合同要赔多少",
    "试用期最长可以多久",
    "试用期工资可以低于正式工资吗",
    "加班费的计算基数是什么",
    "未休年假可以折算工资吗",
    "孕妇被辞退有什么特别保护",
    "那这个要赔多少？",
    "它和经济补偿能一起要吗？",
]

SUMMARY_MARKER = "# 本次会话前情"
SUMMARY_PROMPT_MARKER = "压缩会话前情"

# 实测触发点（改动短期记忆窗口/摘要节流策略后需同步复核）：
#   · short_term 窗口 = 20 条消息 = 10 轮 → 第 10 轮结束时窗口满，当轮即可产出摘要
#   · 摘要写入发生在本轮回答之后 → 前情段最早在**下一轮**（第 11 轮）进入提示词
#   · 若某轮摘要生成失败（LLM 返回空回答），按设计保留旧摘要并顺延，注入轮次随之后移
MIN_ROUNDS_FOR_SUMMARY = 10
MIN_ROUNDS_FOR_INJECTION = 11


class RecordingLLM:
    """包一层真实 LLM，记录实际发出的消息（不改动其行为）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.inner.chat(system_prompt, user_prompt)

    def stream_chat(self, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        return self.inner.stream_chat(system_prompt, user_prompt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="会话摘要 + 查询改写 真实多轮链路检查")
    parser.add_argument(
        "--rounds",
        type=int,
        default=14,
        help="连跑轮数（默认 14 = 触发点 10~11 轮 + 3 轮缓冲，题目循环使用）",
    )
    parser.add_argument("--user-id", default=USER_ID, help=f"探针用户（默认 {USER_ID}）")
    parser.add_argument("--session-id", default=SESSION_ID, help=f"探针会话（默认 {SESSION_ID}）")
    parser.add_argument("--keep", action="store_true", help="保留 Redis 探针 key（默认跑完删除）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        print(f"❌ --rounds 至少为 1（收到 {args.rounds}）")
        return 2

    user_id, session_id = args.user_id, args.session_id
    print("=== 环境 ===")
    print(f"SESSION_SUMMARY_ENABLED(实测) = {settings.session_summary_enabled}")
    print(f"短期记忆窗口 = 20 条消息；摘要触发实测最早第 {MIN_ROUNDS_FOR_SUMMARY} 轮、"
          f"前情注入最早第 {MIN_ROUNDS_FOR_INJECTION} 轮")
    print(f"探针 user_id={user_id}  session_id={session_id}  轮数={args.rounds}")
    print("（前情=n/a 表示本轮在检索侧就拒答、未调用 LLM，提示词无从观测）")
    if not settings.session_summary_enabled:
        print("❌ 会话摘要开关未打开，本检查无意义")
        return 2

    try:
        retrieval_service = build_default_retrieval_service()
    except Exception as error:  # 连不上 Redis/MySQL/Milvus 时不静默继续
        print(f"❌ 检索链路装配失败（前置服务不可用）：{type(error).__name__}: {error}")
        return 3

    store = retrieval_service.short_term_memory
    rewriter = QueryRewriter()
    recorder = RecordingLLM(build_chat_client_from_settings())
    service = ChatService(
        retrieval_service=retrieval_service,
        llm_client=recorder,
        refusal_min_vector_score=settings.refusal_min_vector_score,
        session_summary_enabled=True,
    )

    prompt_with_prior: str | None = None
    first_summary_round: int | None = None
    first_injection_round: int | None = None
    observed_after_summary = 0
    rewrites: list[tuple[int, str, str]] = []
    try:
        for index in range(1, args.rounds + 1):
            question = QUESTIONS[(index - 1) % len(QUESTIONS)]

            # 1) 按真实链路顺序先改写：此时短期记忆里只有前 index-1 轮的消息
            rewrite = rewrite_query_safely(
                question,
                query_rewriter=rewriter,
                short_term_memory=store,
                user_id=user_id,
                session_id=session_id,
            )
            if rewrite.changed:
                rewrites.append((index, rewrite.original_query, rewrite.rewritten_query))

            # 2) 真实问答
            recorder.calls.clear()
            result = service.chat(question, user_id=user_id, session_id=session_id)

            # 3) 等价于 persist_turn 的 Redis 回写（chat_store=None，不落 MySQL）
            store.append_message(user_id, session_id, {"role": "user", "content": question})
            store.append_message(
                user_id, session_id, {"role": "assistant", "content": result.answer}
            )

            # 4) 等价于 persist_turn 末尾的后台线程摘要更新（此处同步，便于观察）
            new_summary = maybe_update_session_summary(
                short_term_memory=store,
                user_id=user_id,
                session_id=session_id,
                llm_client=recorder,
                enabled=True,
            )

            messages = store.read_messages(user_id, session_id)
            current_summary = store.read_summary(user_id, session_id) or ""
            # 提示词观测：先排除"摘要生成"那一次调用，剩下的才是问答调用
            chat_prompts = [p for _, p in recorder.calls if SUMMARY_PROMPT_MARKER not in p]
            if not chat_prompts:
                # 本轮在检索侧就拒答（无候选/相关性过低），根本没调 LLM，
                # 提示词无从观测 —— 记为 n/a，不能当成"没注入前情"
                prior_state = "n/a"
            else:
                prior_state = "YES" if SUMMARY_MARKER in chat_prompts[0] else "no"
                if prior_state == "YES" and first_injection_round is None:
                    first_injection_round = index
                    prompt_with_prior = chat_prompts[0]

            if new_summary and first_summary_round is None:
                first_summary_round = index

            # 摘要写入发生在本轮回答之后，故"可观测注入"只能是更靠后的轮次
            if (
                first_summary_round is not None
                and index > first_summary_round
                and prior_state != "n/a"
            ):
                observed_after_summary += 1

            print(
                f"[{index:2d}] 消息={len(messages):2d} 摘要={len(current_summary):3d}字 "
                f"新摘要={'有' if new_summary else '无'} 前情={prior_state} "
                f"改写={'YES' if rewrite.changed else 'no'} 引用={len(result.sources)}"
            )
            if rewrite.changed:
                print(f"      改写：{rewrite.original_query} → {rewrite.rewritten_query}")

        print()
        print("=" * 78)
        print("### ① LLM 真实压出来的摘要原文（存于 Redis，非预置）")
        print("=" * 78)
        print(store.read_summary(user_id, session_id))

        print()
        print("=" * 78)
        print("### ② 某一轮实际注入提示词的「# 本次会话前情」段全文")
        print("=" * 78)
        if prompt_with_prior is None:
            print("（未捕获到含前情段的提示词）")
        else:
            section_start = prompt_with_prior.index(SUMMARY_MARKER)
            tail = prompt_with_prior[section_start:]
            cut = tail.find("\n# ", 1)
            print(tail[: cut if cut > 0 else len(tail)].rstrip())
            print()
            print(f"（该轮提示词总长 {len(prompt_with_prior)} 字）")

        print()
        print("=" * 78)
        print(f"### ③ 查询改写记录：{len(rewrites)}/{args.rounds} 轮发生改写")
        print("=" * 78)
        for index, original, rewritten in rewrites[:10]:
            print(f"  [第 {index} 轮] {original} → {rewritten}")
        if not rewrites:
            print("  （本次没有任何一轮触发改写：题目之间缺少指代/省略信号）")
    finally:
        if args.keep:
            print()
            print("探针 key 保留（--keep）：", store.read_summary(user_id, session_id) is not None)
        else:
            store.delete_session_memory(user_id, session_id)
            print()
            print("探针 key 已删除：", store.read_summary(user_id, session_id) is None)

    print()
    print("=" * 78)
    print("### 结论")
    print("=" * 78)
    print(f"  首次产出摘要的轮次 = {first_summary_round}")
    print(f"  首次注入前情段的轮次 = {first_injection_round}")
    print(f"  发生改写的轮次数 = {len(rewrites)}/{args.rounds}")

    if first_summary_round is not None and first_injection_round is not None:
        print("  ✅ 摘要生成与前情注入都在真实多轮链路里观测到")
        return 0

    # 轮数不够属于"没验到"，与"验了没通过"必须区分开
    if first_summary_round is None and args.rounds < MIN_ROUNDS_FOR_SUMMARY:
        print(f"  ⚠ 轮数不足（{args.rounds} < {MIN_ROUNDS_FOR_SUMMARY}），尚未到窗口触发点，本次未验到")
        return 2
    if first_summary_round is None:
        print("  ❌ 跑了足够轮数仍未产出摘要（窗口已满却拿不到摘要），链路有问题")
        return 1
    if first_summary_round == args.rounds:
        print(f"  ⚠ 摘要在最后一轮（第 {first_summary_round} 轮）才产出，本轮回答已发完，"
              "还没有下一轮可观测注入；请加大 --rounds")
        return 2
    # 摘要产出后唯一可观测的那些轮次如果都在检索侧拒答，就没有提示词可看
    if observed_after_summary == 0:
        print("  ⚠ 摘要产出后的轮次都在检索侧拒答（未调用 LLM），提示词无从观测；"
              "请加大 --rounds 或换用有候选的题目")
        return 2
    print("  ❌ 摘要已产出且有可观测的后续轮次，但提示词里始终没有前情段，链路有问题")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
