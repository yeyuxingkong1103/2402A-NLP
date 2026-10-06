#!/usr/bin/env python3
"""审计 SFT 数据的**监督信号是否被截断吃掉**（真 tokenizer 实测，不估）。

为什么需要这个脚本（2026-09-28 的教训）：

"把检索到的法条放进提示"这个修复是对的，但它让提示从 ~18 token 涨到 ~3000+ token，
而 `--max-len` 还是 2048；`encode_chat()` 当时是**右截断**，而提示在长、答案在尾 ⇒
答案被整段切掉。实测 **912 条里 514 条（56.4%）只剩 <=1 个监督 token**（中位数 1），
等于白训了 20 分钟还看不出来。

所以：**训练前必须量一次**。用法（在装好 transformers 的环境里跑）：

    python scripts/audit_sft_tokens.py \
        --tokenizer /root/autodl-tmp/models/Qwen--Qwen3-4B \
        --data data/sft/train.neg25.jsonl --data data/sft/train.infer.jsonl \
        --max-len 2048 --out /root/token_audit.json

退出码：0 正常；2 有数据文件读不到；**3 证据被裁得太多（训练提示 != 推理提示）**；
**4 有样本只剩 <=1 个监督 token（答案都没了）**。后两种情况都别急着烧 GPU。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.sft import encode_chat  # noqa: E402


class MissingSupervisionError(RuntimeError):
    """`encode_chat` 没给出监督元数据 ⇒ 环境里是旧代码（不许静默当成 0）。"""


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (1 - (position - low)) + ordered[high] * (position - low)


def prompt_and_full_lengths(messages: list[dict], tokenizer) -> tuple[int, int]:
    """提示 / 全文的 token 数。

    注意：transformers 5.x 里 `apply_chat_template(tokenize=True)` 返回的是 **BatchEncoding**，
    对它取 `len()` 得到的是"键的个数"（=2）——第一版就把 1700 token 的提示量成了 2。
    这里统一走 `tokenize=False` 拿文本、再手工 tokenize。
    """
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                                add_generation_prompt=True)
    eos = getattr(tokenizer, "eos_token", "") or ""
    full_text = f"{prompt_text}{messages[-1].get('content') or ''}{eos}"
    return (len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]),
            len(tokenizer(full_text, add_special_tokens=False)["input_ids"]))


def legacy_right_truncate(messages: list[dict], tokenizer, *, max_len: int) -> int:
    """**旧实现**的监督 token 数（右截断），只为复现历史读数。

    2026-09-28 之前 `encode_chat` 是：全文右截断 ``full_ids[:max_len]``，
    ``prompt_len = min(len(prompt_ids), max(len(full_ids)-1, 0))``。提示一超长，答案就被切光，
    只剩 1 个监督 token。修好之后（只裁证据、保留答案）这个数字**不会再复现**，
    所以留一个开关，让报告里那组历史证据仍然可复算。
    """
    last = messages[-1]
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                                add_generation_prompt=True)
    eos = getattr(tokenizer, "eos_token", "") or ""
    full_text = f"{prompt_text}{last.get('content') or ''}{eos}"
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(full_text, add_special_tokens=False)["input_ids"])[:max_len]
    prompt_len = min(len(prompt_ids), max(len(full_ids) - 1, 0))
    return max(0, len(full_ids) - prompt_len)


def audit_records(records: list[dict], tokenizer, *, max_len: int = 2048,
                  legacy: bool = False) -> dict:
    """逐条算「提示多长 / 全文多长 / 实际监督几个 token」，汇总成一份可写进报告的读数。

    ``legacy=True`` 用**旧实现**（右截断）算监督，用来复现 2026-09-28 那组历史读数；
    默认用**当前实现**，此时还会给出 ``prompt_trimmed_*``（证据被裁掉的比例）——
    那才是修好之后真正该盯的指标：**训练提示与推理提示不一致**。
    """
    prompt_lens: list[int] = []
    full_lens: list[int] = []
    supervised: list[int] = []
    trimmed: list[int] = []
    left_cut = 0
    by_kind: dict[str, list[int]] = {}
    starved = 0
    for record in records:
        messages = record.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            continue
        prompt_len, full_len = prompt_and_full_lengths(messages, tokenizer)
        if legacy:
            n_supervised = legacy_right_truncate(messages, tokenizer, max_len=max_len)
            dropped = 0
            was_left_cut = False
        else:
            encoded = encode_chat(messages, tokenizer, max_len=max_len)
            info = encoded.get("supervision")
            if info is None:
                # **绝不许悄悄当成 0**：旧代码（右截断）不返回 supervision，
                # 静默取 0 会得出"100% 样本没有监督"这种假结论（真踩过）。
                raise MissingSupervisionError(
                    "当前环境里的 encode_chat 不返回 supervision 元数据（说明是修好之前的旧代码）。"
                    "请先同步代码；若就是要复现 2026-09-28 的历史读数，请加 --legacy-right-truncate。")
            n_supervised = int(info.get("supervised_tokens") or 0)
            dropped = int(info.get("dropped_context_chars") or 0)
            was_left_cut = bool(info.get("left_truncated_prompt"))
        prompt_lens.append(prompt_len)
        full_lens.append(full_len)
        supervised.append(n_supervised)
        trimmed.append(dropped)
        if was_left_cut:
            left_cut += 1
        if n_supervised <= 1:
            starved += 1
        kind = str((record.get("meta") or {}).get("kind") or "positive")
        by_kind.setdefault(kind, []).append(n_supervised)
    rows = len(supervised)
    trimmed_rows = sum(1 for value in trimmed if value > 0)
    return {
        "present": rows > 0, "rows": rows, "max_len": max_len, "legacy_right_truncate": legacy,
        "prompt_tokens_mean": round(statistics.fmean(prompt_lens), 1) if rows else None,
        "prompt_tokens_median": round(statistics.median(prompt_lens), 1) if rows else None,
        "prompt_tokens_p95": round(percentile(prompt_lens, 0.95), 1) if rows else None,
        "prompt_tokens_max": max(prompt_lens) if rows else None,
        "prompt_over_maxlen": sum(1 for n in prompt_lens if n >= max_len - 1),
        "full_tokens_mean": round(statistics.fmean(full_lens), 1) if rows else None,
        "full_tokens_p95": round(percentile(full_lens, 0.95), 1) if rows else None,
        "full_tokens_max": max(full_lens) if rows else None,
        "fit_in_2048": sum(1 for n in full_lens if n <= 2048),
        "fit_in_4096": sum(1 for n in full_lens if n <= 4096),
        "fit_in_8192": sum(1 for n in full_lens if n <= 8192),
        "supervised_mean": round(statistics.fmean(supervised), 1) if rows else None,
        "supervised_median": round(statistics.median(supervised), 1) if rows else None,
        "supervised_p05": round(percentile(supervised, 0.05), 1) if rows else None,
        "supervised_starved_rows": starved,
        "supervised_starved_share": round(starved / rows, 4) if rows else None,
        # 修好之后真正该盯的：证据被裁 ⇒ 训练提示 ≠ 推理提示
        "prompt_trimmed_rows": trimmed_rows,
        "prompt_trimmed_share": round(trimmed_rows / rows, 4) if rows else None,
        "dropped_context_chars_total": sum(trimmed),
        "left_truncated_rows": left_cut,
        "supervised_by_kind_mean": {k: round(statistics.fmean(v), 1)
                                    for k, v in by_kind.items() if v},
    }


def load_records(path: Path) -> list[dict]:
    handle = path.open(encoding="utf-8")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 SFT 数据的监督信号是否被截断吃掉")
    parser.add_argument("--tokenizer", required=True, help="tokenizer 目录或模型名")
    parser.add_argument("--data", action="append", required=True, help="SFT jsonl（可多次）")
    parser.add_argument("--max-len", type=int, default=2048, help="训练时的 max_len")
    parser.add_argument("--out", default="", help="把结果写成 JSON（UTF-8，LF）")
    parser.add_argument("--max-starved-share", type=float, default=0.05,
                        help="只剩 <=1 个监督 token 的样本占比超过它就 exit 4")
    parser.add_argument("--max-trimmed-share", type=float, default=0.10,
                        help="证据被裁掉（训练提示 != 推理提示）的样本占比超过它就 exit 3")
    parser.add_argument("--legacy-right-truncate", action="store_true",
                        help="用**旧实现**（右截断）算监督，用来复现 2026-09-28 的历史读数")
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415 - 只在真跑时需要

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    result: dict = {}
    missing: list[str] = []
    for raw in args.data:
        path = Path(raw)
        if not path.is_file():
            missing.append(raw)
            result[path.name] = {"present": False}
            continue
        try:
            result[path.name] = audit_records(load_records(path), tokenizer,
                                              max_len=args.max_len,
                                              legacy=args.legacy_right_truncate)
        except MissingSupervisionError as exc:
            print(f"!! {raw}：{exc}", file=sys.stderr)
            return 5

    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    print(text)

    if missing:
        print(f"!! 数据文件不存在：{', '.join(missing)}", file=sys.stderr)
        return 2
    for name, item in result.items():
        trimmed = item.get("prompt_trimmed_share")
        if trimmed is not None and trimmed > args.max_trimmed_share:
            print(f"!! {name}：{trimmed:.1%} 的样本**证据被裁掉**（上限 "
                  f"{args.max_trimmed_share:.1%}）—— 训练提示与推理提示不一致，"
                  f"训练时看不到的上下文，推理时却会出现。先把 --max-len 加大。",
                  file=sys.stderr)
            return 3
        share = item.get("supervised_starved_share")
        if share is not None and share > args.max_starved_share:
            print(f"!! {name}：{share:.1%} 的样本只剩 <=1 个监督 token"
                  f"（上限 {args.max_starved_share:.1%}）—— 答案都没了，训了也白训。",
                  file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
