#!/usr/bin/env python3
"""把负样本按目标比例混进一份**已经渲染好**的 SFT 训练集（正样本保持不变）。

为什么要这个脚本（2026-09-28 的实验设计）：
`build_sft_data.py` 每次跑都会**继续补正样本**（断点续跑），所以"再跑一次加负样本"
会同时改变两件事：正样本变多 + 混进负样本。那样如果 v5b 比 v5a 好，
**分不清是负样本的功劳还是多出来的正样本的功劳**。
所以实验切成两步：
  1. `build_sft_data.py` 先把正样本（与推理同形的提示）定型 ⇒ `train.infer.jsonl`；
  2. 本脚本只做混合：**同一份正样本**按比例下采样 + 负样本，产出 `train.neg25.jsonl`。
这样 v5a（纯正样本）与 v5b（正样本 + 25% 负样本）**只差负样本**，能干净归因。

负样本从哪里来：`samples.jsonl` 里 `meta.kind == "negative"` 的那些（由
`build_sft_data.py --negatives` 生成）。本脚本**按线上提示格式反解**出 question/answer
（渲染后的记录已经是最终形态，直接解析最不容易与训练格式漂移）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.generate.prompt import QUESTION_HEADER  # noqa: E402
from legal_rag.sft import SftSample  # noqa: E402


def sample_from_record(record: dict) -> SftSample | None:
    """把一条**已渲染**的记录反解成 ``SftSample``（question 取最后一个问题头之后的部分）。

    反解而不是另存一份：渲染后的记录就是模型真正看到的东西，
    另立格式早晚会与训练/推理漂移（这正是上一次事故的成因）。
    """
    messages = record.get("messages") or []
    user = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    answer = ""
    for message in messages:
        if message.get("role") == "assistant":
            answer = str(message.get("content") or "")
    if not user or not answer:
        return None
    question = user.rsplit(QUESTION_HEADER, 1)[-1].strip() if QUESTION_HEADER in user else user.strip()
    if not question:
        return None
    meta = dict(record.get("meta") or {})
    return SftSample(question=question, answer=answer,
                     evidence=[str(x) for x in (record.get("evidence") or [])],
                     source=str(record.get("source") or ""), meta=meta)


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pick_negatives(samples_path: Path, *, limit: int = 0, skip: int = 0) -> list[dict]:
    """从 samples.jsonl 里挑出负样本记录（``meta.kind == "negative"``）。

    ``skip`` 用来让**训练集与验证集各拿一批不同的负样本**：
    验证集先取前 N 条，训练集跳过这 N 条再取 —— 否则同一批负样本两边都有，
    验证损失就不再是"没见过的数据"上的读数。
    """
    picked: list[dict] = []
    seen = 0
    for record in load_records(samples_path):
        if str((record.get("meta") or {}).get("kind") or "") != "negative":
            continue
        seen += 1
        if seen <= skip:
            continue
        picked.append(record)
        if limit and len(picked) >= limit:
            break
    return picked


def mix(positives: list[dict], negatives: list[dict], ratio: float, *, seed: int) -> tuple[list[dict], dict]:
    """按目标比例混合；**用下采样正样本**实现，负样本一条不丢。"""
    negatives = list(negatives)
    positives = list(positives)
    if not negatives or ratio <= 0:
        return positives, {"target_ratio": ratio, "actual_ratio": 0.0,
                           "positives_kept": len(positives), "negatives_kept": 0}
    if ratio >= 1:
        keep = 1
    else:
        keep = max(1, min(len(positives), int(round(len(negatives) * (1 - ratio) / ratio))))
    rng = random.Random(seed)
    kept = rng.sample(positives, keep) if keep < len(positives) else positives
    combined = kept + negatives
    rng.shuffle(combined)
    return combined, {"target_ratio": ratio, "actual_ratio": round(len(negatives) / len(combined), 4),
                      "positives_kept": len(kept), "negatives_kept": len(negatives)}


def main() -> int:
    parser = argparse.ArgumentParser(description="把负样本按比例混进已渲染的 SFT 集")
    parser.add_argument("--positives", required=True, help="已渲染的正样本 jsonl（如 train.infer.jsonl）")
    parser.add_argument("--samples", default="data/sft/samples.jsonl", help="含负样本的 samples.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=float, default=0.25, help="负样本目标占比")
    parser.add_argument("--max-negatives", type=int, default=0, help="最多用多少条负样本（0=不限）")
    parser.add_argument("--skip-negatives", type=int, default=0,
                        help="跳过前 N 条负样本（让验证集先用掉一批，训练集就不再重复用）")
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    positives = load_records(Path(args.positives))
    if not positives:
        print(f"!! 正样本为空：{args.positives}", file=sys.stderr)
        return 2
    negatives = pick_negatives(Path(args.samples), limit=args.max_negatives,
                               skip=max(0, args.skip_negatives))
    if not negatives:
        print(f"!! {args.samples} 里没有负样本（meta.kind == negative）—— "
              f"先跑 build_sft_data.py --negatives N", file=sys.stderr)
        return 3

    # 反解检查：正样本必须解得出来（否则说明格式变了）
    broken = [i for i, record in enumerate(positives) if sample_from_record(record) is None]
    if broken:
        print(f"!! 有 {len(broken)} 条正样本反解不出 question/answer（前 3 个下标 {broken[:3]}）——"
              f"格式可能变了，别拿它训", file=sys.stderr)
        return 4

    combined, report = mix(positives, negatives, args.ratio, seed=args.seed)
    if report["actual_ratio"] > report["target_ratio"] + 0.05:
        # 正样本不够下采样时，目标占比**达不到**（负样本一条不丢是硬约定）
        print(f"[注意] 正样本不够：要达到 {report['target_ratio']:.0%} 需要更多正样本，"
              f"实际 {report['actual_ratio']:.1%}（负样本按约定一条不丢，没删）", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in combined) + "\n",
                   encoding="utf-8")

    kinds: dict[str, int] = {}
    for record in combined:
        kind = str((record.get("meta") or {}).get("kind") or "positive")
        kinds[kind] = kinds.get(kind, 0) + 1
    lengths = [sum(len(str(m.get("content") or "")) for m in (r.get("messages") or []))
               for r in combined]
    print(f"正样本 {len(positives)} 条 + 负样本 {len(negatives)} 条 -> 写出 {len(combined)} 条到 {out}")
    print(f"  实际负样本占比 {report['actual_ratio']}（目标 {report['target_ratio']}）"
          f"｜kind 分布 {kinds}｜平均提示字符 {sum(lengths) / len(lengths):.0f}")
    with_ev = 0
    for record in combined:
        evidence = [str(x) for x in (record.get("evidence") or [])]
        blob = "".join(str(m.get("content") or "") for m in (record.get("messages") or []))
        if evidence and any(ev[:40] and ev[:40] in blob for ev in evidence):
            with_ev += 1
    print(f"  自检：提示里含证据 {with_ev}/{len(combined)}")
    if with_ev != len(combined):
        print(f"!! 有 {len(combined) - with_ev} 条提示里没有证据 —— 不许拿去训", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
