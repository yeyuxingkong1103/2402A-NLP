#!/usr/bin/env python3
"""用教师模型（云端 27B）对语料自蒸馏「引用对齐」的 SFT 数据。

为什么这么做（B 方案）：P8 的天花板是**法条级精度** —— 通用重排 + 向量检索排不进 top-5。
微调小基座的目的不是灌法律知识（知识在检索里），而是教它三件事：
**只依据给定条文作答 / 带条号引用 / 没据就说没据**。所以训练数据必须是
"问题 + 只靠证据作答的答案 + 证据"，而且**引用必须能在证据里找到**。

三重校验（缺一条就会把噪声学进去）：
1. `parse_teacher_reply` 抠不出 question/answer -> 丢（绝不拿原文当答案）；
2. `citation_problems` 非空 -> 丢（答案里出现了证据里没有的法名/条号 = 编条文）；
3. 与 `eval/qa_set.jsonl` 词面重叠 -> 丢（**训练集混进评测题，v4 涨分就是自欺欺人**）。

用法（云端，vLLM 起着）：
    python scripts/build_sft_data.py --source knowledge/lawyer --out data/sft \
        --teacher-url http://127.0.0.1:30000/v1 --teacher-model qwen27b \
        --limit 600 --per-file 3 --concurrency 4

特性：**断点续跑**（已处理的证据块记在 samples/rejects 里，重跑自动跳过）、
每个拒绝都写进 `rejects.jsonl` 带原因（不静默丢）、结尾产出 train/val/stats。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.engine import out_of_scope_reply  # noqa: E402
from legal_rag.sft import (  # noqa: E402
    SftSample, citation_problems, dedupe, dump_jsonl, eval_questions, evidence_blocks,
    load_jsonl, overlap_with_eval, parse_teacher_reply, semantic_overlap, similarity,
    split_train_val, to_inference_messages,
)

TEACHER_PROMPT = """你是中国法律数据标注员。下面给你**一段法条原文**（唯一依据）。

要求：
1. 写一个普通用户会真实问出口的问题（口语、具体，20 字以内），这段法条正好能回答它；
2. 写一段回答：**只能依据上面这段法条**，必须引用法名与条号（形如《中华人民共和国XX法》第X条）；
3. 法条里没写的内容**一个字都不要加**（不要补充常识、不要举例外推）；
4. 严格输出 JSON：{{"question": "...", "answer": "..."}}

法条原文：
---
{evidence}
---
"""

#: 负样本（"有检索结果但答不了" -> 该拒答）的教师提示。
#:
#: 为什么要负样本（backlog 里挂着的根因）：上一轮 LoRA 的 `must_refuse` **退化**了
#: （0.8333 → 0.7500）—— 训练集里全是"问 → 答"，模型没学过"没据就拒答"。
#: 负样本的形态刻意做成 **"有上下文但不充分"**（而不是空上下文）：线上检索总会返回点什么，
#: 模型要学的是"**这些资料答不了这个问题**"，与真实失败模式对齐。
NEGATIVE_TEACHER_PROMPT = """你是中国法律数据标注员。下面给你**一段法条原文**。

要求：
1. 写一个普通用户会真实问出口的**法律**问题（口语、具体，20 字以内）；
2. 这个问题必须是**上面这段法条回答不了**的（问的是这段法条没涉及的事）；
3. 不要输出答案，只要问题；
4. 严格输出 JSON：{{"question": "..."}}

法条原文：
---
{evidence}
---
"""

#: 负样本与评测题的**更严**相似度门槛。
#: 理由：负样本天然是"答不了的法律问题"——与评测集里 12 条 `must_refuse` 是**同一类**，
#: 泄漏风险最高；用通用门槛（0.85）会放过改写型泄漏。
NEGATIVE_EVAL_GUARD = 0.6

#: 负样本的问题与"所给证据"的相似度上限：超过它就说明这段法条其实能沾上边，不算负样本。
NEGATIVE_EVIDENCE_MAX_SIMILARITY = 0.3


def parse_question_reply(reply: str) -> str:
    """从教师的回复里抠出 ``question``（只输出问题的提示词用）。

    宽容一点但不含糊：先按 JSON 取 ``question``；取不到再用正则从 ``"question": "..."`` 里抠；
    都没有就抛 ``ValueError``（**绝不把原文当问题**）。
    """
    text = str(reply or "").strip()
    if not text:
        raise ValueError("教师回复为空")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and str(payload.get("question") or "").strip():
            return str(payload["question"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r'"question"\s*:\s*"([^"]{2,80})"', text)
    if match:
        return match.group(1).strip()
    raise ValueError(f"抠不出 question：{text[:120]!r}")


def build_negative_sample(block: str, source_name: str, question: str) -> SftSample:
    """把"证据 + 它答不了的问题"组装成一条负样本，答案是**引擎自己的拒答文案**。

    答案直接用 `legal_rag.engine.out_of_scope_reply()`：这样训练目标与线上真实拒答
    **逐字一致**（含"请补充关键信息 / 换个更具体的说法"的引导语），
    不会因为我们另写一句"我不知道"而把线上行为带偏。
    """
    return SftSample(question=question, answer=out_of_scope_reply(question),
                     evidence=[block], source=source_name,
                     meta={"kind": "negative", "question": question,
                           "answer": out_of_scope_reply(question),
                           "evidence_hash": sha1(block.encode("utf-8")).hexdigest()[:16]})


def render_record(sample: SftSample, *, inference: bool = True) -> dict:
    """把样本转成落盘记录。

    ``inference=True``（默认）-> 走 :func:`legal_rag.sft.to_inference_messages`：
    system + **检索知识（含法条原文）** + 用户问题 → 答案，
    与线上提示**同形**。``False`` 只写 ``[user, assistant]``（**旧行为，已证明是错的**，
    保留只为复现旧数据）。
    """
    if not inference:
        record = sample.to_chat_record()
        record["meta"] = dict(sample.meta)
        return record
    messages = to_inference_messages(sample)
    messages.append({"role": "assistant", "content": sample.answer})
    return {"messages": messages, "evidence": list(sample.evidence),
            "source": sample.source, "meta": dict(sample.meta)}


def evidence_in_prompt_count(records: list[dict], *, prefix: int = 40) -> int:
    """有多少条记录的 **messages 里真的出现了证据片段**。

    这就是 2026-09-28 那次事故的判据：当时 796/796 条都**不含**证据，
    却没有任何检查发现 —— 所以现在把它变成一个会**报错**的数。
    """
    hits = 0
    for record in records:
        evidence = [str(x) for x in (record.get("evidence") or [])]
        blob = "".join(str(m.get("content") or "") for m in (record.get("messages") or []))
        if evidence and any(ev[:prefix] and ev[:prefix] in blob for ev in evidence):
            hits += 1
    return hits


def apply_negative_ratio(positives: list[SftSample], negatives: list[SftSample],
                         ratio: float, *, seed: int) -> tuple[list[SftSample], dict]:
    """把正负样本按目标比例混合（**用下采样正样本**实现，负样本一条不丢）。

    返回 ``(混合后的样本, 说明)``。``ratio<=0`` 表示不加负样本；
    ``ratio>=1`` 视为"尽量都用负样本"（仍会保留至少一条正样本，免得完全跑偏）。
    """
    negatives = list(negatives)
    positives = list(positives)
    if not negatives or ratio <= 0:
        return positives, {"target_ratio": ratio, "actual_ratio": 0.0,
                           "positives_kept": len(positives), "negatives_kept": 0}
    if ratio >= 1:
        keep = max(1, min(len(positives), 1))
    else:
        # n_neg / (n_neg + n_pos) = ratio  ->  n_pos = n_neg * (1 - ratio) / ratio
        keep = int(round(len(negatives) * (1 - ratio) / ratio))
        keep = max(1, min(len(positives), keep))
    rng = random.Random(seed)
    kept = rng.sample(positives, keep) if keep < len(positives) else positives
    mixed = kept + negatives
    rng.shuffle(mixed)
    total = len(mixed)
    return mixed, {"target_ratio": ratio, "actual_ratio": round(len(negatives) / total, 4),
                   "positives_kept": len(kept), "negatives_kept": len(negatives)}


def _negative_candidates(files: list[Path], *, need: int, per_file: int, min_chars: int,
                         seed: int) -> list[tuple[str, str]]:
    """给负样本挑证据块（与正样本不同的抽样种子，尽量不重叠）。"""
    rng = random.Random(seed)
    picked: list[tuple[str, str]] = []
    for path in files:
        if len(picked) >= need:
            break
        for block, name in _evidence_items(path, per_file=per_file, min_chars=min_chars, rng=rng):
            picked.append((block, name))
            if len(picked) >= need:
                break
    return picked


def _build_negatives(args, *, negative_per_block: int, eval_qs: list[str],
                     rejects_path: Path) -> dict:
    """生成负样本：**有上下文但答不了** -> 目标是引擎自己的拒答文案。

    三道闸门（与正样本同样的"不静默丢"口径，拒绝原因全写进 rejects.jsonl）：
    1. 教师回复里抠得出 question（抠不出就丢，绝不拿原文当问题）；
    2. 问题与**这段证据**的相似度 <= ``NEGATIVE_EVIDENCE_MAX_SIMILARITY``
       （否则这段法条其实沾得上边，就不是负样本了）；
    3. 与评测题的相似度 <= ``NEGATIVE_EVAL_GUARD``（**比通用门槛严**：
       负样本天然与评测集里 12 条 `must_refuse` 同类，泄漏风险最高）。
    """
    files = [Path(p) for p in ([args.source] if Path(args.source).is_file()
                               else sorted(Path(args.source).rglob("*")))]
    files = [p for p in files if p.is_file()]
    need_blocks = max(1, -(-args.negatives // max(1, negative_per_block)))   # 向上取整
    pairs = _negative_candidates(files, need=need_blocks, per_file=max(1, args.per_file),
                                 min_chars=args.min_chars, seed=args.seed + 777)
    print(f"  负样本候选证据块 {len(pairs)} 个（目标 {args.negatives} 条问题）")

    lock = threading.Lock()
    built: list[SftSample] = []
    rejected = {"unparsable": 0, "evidence_too_similar": 0, "eval_overlap": 0,
                "too_long": 0, "teacher_error": 0}

    def _work(item: tuple[str, str]) -> None:
        block, source_name = item
        try:
            reply = _call_teacher(args.teacher_url, args.teacher_model, args.api_key,
                                  NEGATIVE_TEACHER_PROMPT.format(evidence=block),
                                  timeout=args.timeout, max_tokens=args.max_tokens,
                                  temperature=args.temperature)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            with lock:
                rejected["teacher_error"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"kind": "negative", "reason": f"teacher_error: {exc}"},
                                            ensure_ascii=False) + "\n")
            return
        try:
            question = parse_question_reply(reply)
        except ValueError as exc:
            with lock:
                rejected["unparsable"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"kind": "negative", "reason": f"unparsable: {exc}"},
                                            ensure_ascii=False) + "\n")
            return
        if len(question) > 40:
            with lock:
                rejected["too_long"] += 1
            return
        overlap = similarity(question, block)
        if overlap > NEGATIVE_EVIDENCE_MAX_SIMILARITY:
            with lock:
                rejected["evidence_too_similar"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"kind": "negative", "reason": "evidence_too_similar",
                                             "question": question, "similarity": round(overlap, 4)},
                                            ensure_ascii=False) + "\n")
            return
        if eval_qs and overlap_with_eval([question], eval_qs, NEGATIVE_EVAL_GUARD):
            with lock:
                rejected["eval_overlap"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"kind": "negative", "reason": "eval_overlap",
                                             "question": question}, ensure_ascii=False) + "\n")
            return
        sample = build_negative_sample(block, source_name, question)
        with lock:
            built.append(sample)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        list(pool.map(_work, pairs * max(1, negative_per_block)))

    # 只留前 N 条（并发下顺序不定，排序保证可复现）
    built.sort(key=lambda s: s.question)
    built = built[: args.negatives]

    # ⚠️ **语义闸门也要过一遍**：负样本天然与评测集里 12 条 `must_refuse` 同类，
    #    只靠词面相似度（0.6）挡不住"换个说法"的泄漏 -> 有 embedder 就再筛一次。
    semantic_dropped = 0
    if args.semantic_check and built and eval_qs:
        embed_fn = _build_embed_fn(args.semantic_model)
        if embed_fn is not None:
            hits = semantic_overlap([s.question for s in built], eval_qs, embed_fn,
                                    args.semantic_threshold)
            if hits:
                offenders = {hit["question"] for hit in hits}
                with rejects_path.open("a", encoding="utf-8") as handle:
                    for hit in hits:
                        handle.write(json.dumps({"kind": "negative",
                                                 "reason": "semantic_overlap", **hit},
                                                ensure_ascii=False) + "\n")
                built = [s for s in built if s.question not in offenders]
                semantic_dropped = len(hits)
                print(f"  负样本语义近邻剔除 {semantic_dropped} 条")

    print(f"  负样本：成功 {len(built)} 条；拒绝 {rejected}"
          + (f"；语义剔除 {semantic_dropped}" if semantic_dropped else ""))
    return {"requested": args.negatives, "built": len(built), "rejected": rejected,
            "semantic_dropped": semantic_dropped, "samples": built,
            "evidence_max_similarity": NEGATIVE_EVIDENCE_MAX_SIMILARITY,
            "eval_guard": NEGATIVE_EVAL_GUARD}


def _call_teacher(url: str, model: str, api_key: str, prompt: str, *, timeout: float,
                  max_tokens: int, temperature: float) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # 关思考模式：不关的话 Qwen3.x 会把自述推理当正文（真机踩过）
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {api_key or 'EMPTY'}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8", "replace"))
    return str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")


def _build_embed_fn(model_path: str):
    """拿一个「文本列表 → 向量列表」的函数（bge-m3）。拿不到就返回 ``None``（**可见**降级）。

    走规范工厂 ``legal_rag.embedding.base.build_embedder``（与引擎同一条路），
    并在建好后**先探一次**——别等几百条样本跑完才发现嵌入不能用。
    """
    if not model_path:
        return None
    try:
        from legal_rag.embedding.base import build_embedder

        embedder = build_embedder("bge_m3", model_path, 1024)
        embedder.embed_texts(["预热"])
    except Exception as exc:  # noqa: BLE001 - 语义闸门是锦上添花：失败要可见但不能崩
        print(f"  [警告] 语义闸门不可用（{type(exc).__name__}: {exc}）—— 本轮只做词面比对")
        return None

    def _embed(texts: list[str]) -> list[list[float]]:
        return embedder.embed_texts(list(texts))

    return _embed


def _evidence_items(source: Path, *, per_file: int, min_chars: int,
                    rng: random.Random) -> list[tuple[str, str]]:
    """返回 ``[(证据文本, 来源文件名)]``；同一文件内**随机抽** ``per_file`` 段。"""
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    blocks = evidence_blocks(text, min_chars=min_chars)
    if not blocks:
        return []
    rng.shuffle(blocks)
    return [(block, source.name) for block in blocks[:per_file]]


def main() -> int:
    parser = argparse.ArgumentParser(description="自蒸馏生成 SFT 数据")
    parser.add_argument("--source", default="knowledge/lawyer")
    parser.add_argument("--out", default="data/sft")
    parser.add_argument("--eval-file", default="eval/qa_set.jsonl",
                        help="评测题集（用于**防泄漏**比对）")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--teacher-model", default="qwen27b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--limit", type=int, default=600, help="最多处理多少个证据块")
    parser.add_argument("--per-file", type=int, default=3, help="每个语料文件抽几段")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--overlap-threshold", type=float, default=0.85)
    parser.add_argument("--semantic-check", action="store_true",
                        help="额外用 bge-m3 向量近邻抓『纯改写』型泄漏（词面闸门抓不到）")
    parser.add_argument("--semantic-model", default="",
                        help="bge-m3 权重目录；给了才做语义闸门")
    parser.add_argument("--semantic-threshold", type=float, default=0.92)
    parser.add_argument("--legacy-prompt-shape", action="store_true",
                        help="**别用**（只为复现旧数据）：把训练提示写回 [user, assistant] 的老样子。"
                             "老样子**不含检索到的法条** -> 模型学的是'凭记忆作答'，"
                             "这正是 2026-09-28 查出的 LoRA ±0 与 must_refuse 退化的根因")
    parser.add_argument("--negatives", type=int, default=0,
                        help="生成多少条**负样本**（有上下文但答不了 -> 应拒答）")
    parser.add_argument("--negative-ratio", type=float, default=0.0,
                        help="负样本在训练集里的目标占比（例如 0.25）；用下采样正样本实现，负样本不丢")
    parser.add_argument("--negative-per-block", type=int, default=1,
                        help="每个证据块生成几条负样本")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    rejects_path = out_dir / "rejects.jsonl"

    known_hashes: set[str] = set()
    if samples_path.is_file():
        for record in load_jsonl(samples_path):
            known_hashes.add(str((record.get("meta") or {}).get("evidence_hash") or ""))
    if rejects_path.is_file():
        for record in load_jsonl(rejects_path):
            known_hashes.add(str((record.get("meta") or {}).get("evidence_hash") or ""))
    print(f"已有样本 {len(load_jsonl(samples_path)) if samples_path.is_file() else 0} 条，"
          f"已知证据块 {len(known_hashes)} 个（断点续跑会跳过）")

    rng = random.Random(args.seed)
    files = sorted(Path(args.source).rglob("*.md"))
    rng.shuffle(files)
    print(f"语料文件 {len(files)} 个（{args.source}），目标证据块 {args.limit} 个，"
          f"并发 {args.concurrency}")

    pending: list[tuple[str, str]] = []
    for path in files:
        if len(pending) >= args.limit:
            break
        for block, name in _evidence_items(path, per_file=args.per_file,
                                           min_chars=args.min_chars, rng=rng):
            digest = sha1(block.encode("utf-8")).hexdigest()[:16]
            if digest in known_hashes:
                continue
            pending.append((block, name))
            known_hashes.add(digest)
            if len(pending) >= args.limit:
                break
    print(f"本轮待处理 {len(pending)} 个证据块")

    eval_qs = eval_questions(args.eval_file) if Path(args.eval_file).is_file() else []
    print(f"防泄漏基准：评测问题 {len(eval_qs)} 条（来自 {args.eval_file}）")

    lock = threading.Lock()
    counters = {"accepted": 0, "rejected": 0, "error": 0}
    started = time.perf_counter()

    def _work(item: tuple[str, str]) -> None:
        block, source_name = item
        digest = sha1(block.encode("utf-8")).hexdigest()[:16]
        meta = {"evidence_hash": digest, "source": source_name}
        try:
            reply = _call_teacher(args.teacher_url, args.teacher_model, args.api_key,
                                  TEACHER_PROMPT.format(evidence=block), timeout=args.timeout,
                                  max_tokens=args.max_tokens, temperature=args.temperature)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            with lock:
                counters["error"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"meta": meta, "reason": f"teacher_error: {exc}"},
                                            ensure_ascii=False) + "\n")
            return
        try:
            question, answer = parse_teacher_reply(reply)
        except ValueError as exc:
            with lock:
                counters["rejected"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"meta": meta, "reason": f"unparsable: {exc}",
                                             "raw": reply[:400]}, ensure_ascii=False) + "\n")
            return
        problems = citation_problems(answer, [block])
        if problems:
            with lock:
                counters["rejected"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"meta": meta, "reason": "citation: " + "; ".join(problems),
                                             "question": question, "answer": answer[:400]},
                                            ensure_ascii=False) + "\n")
            return
        if eval_qs and overlap_with_eval([question], eval_qs, args.overlap_threshold):
            with lock:
                counters["rejected"] += 1
                with rejects_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"meta": meta, "reason": "eval_overlap",
                                             "question": question}, ensure_ascii=False) + "\n")
            return
        sample = SftSample(question=question, answer=answer, evidence=[block],
                           source=source_name, meta=meta)
        record = sample.to_chat_record()
        record["meta"] = {**meta, "question": question, "answer": answer}
        with lock:
            counters["accepted"] += 1
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            done = sum(counters.values())
            if done % 25 == 0:
                print(f"  进度 {done}/{len(pending)}：接受 {counters['accepted']} / "
                      f"拒绝 {counters['rejected']} / 错误 {counters['error']}")

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        list(pool.map(_work, pending))

    records = load_jsonl(samples_path) if samples_path.is_file() else []
    samples = [SftSample(question=str((r.get("meta") or {}).get("question") or ""),
                         answer=str((r.get("meta") or {}).get("answer") or ""),
                         evidence=[str(x) for x in (r.get("evidence") or [])],
                         source=str(r.get("source") or ""),
                         meta=dict(r.get("meta") or {}))
               for r in records]
    samples = [s for s in samples if s.question and s.answer]
    # ⚠️ negatives 也写在同一个 samples.jsonl 里（断点续跑）-> 这里要**按 kind 拆开**，
    #    否则第二次跑会把上次的负样本当成正样本、再加一批新的（负样本翻倍且比例失控）。
    existing_negatives = [s for s in samples if str(s.meta.get("kind") or "") == "negative"]
    samples = [s for s in samples if str(s.meta.get("kind") or "") != "negative"]
    if existing_negatives:
        print(f"（复用已有负样本 {len(existing_negatives)} 条；正样本 {len(samples)} 条）")

    semantic_dropped = 0
    if args.semantic_check and samples and eval_qs:
        print("语义闸门（抓纯改写型泄漏）：")
        embed_fn = _build_embed_fn(args.semantic_model)
        if embed_fn is not None:
            hits = semantic_overlap([s.question for s in samples], eval_qs, embed_fn,
                                    args.semantic_threshold)
            if hits:
                offenders = {hit["question"] for hit in hits}
                with rejects_path.open("a", encoding="utf-8") as handle:
                    for hit in hits:
                        handle.write(json.dumps({"reason": "semantic_overlap", **hit},
                                                ensure_ascii=False) + "\n")
                samples = [s for s in samples if s.question not in offenders]
                semantic_dropped = len(hits)
                print(f"  语义上过近、已剔除 {semantic_dropped} 条（详见 rejects.jsonl）")
                for hit in hits[:5]:
                    print(f"    {hit['ratio']} 「{hit['question']}」 ≈ 「{hit['eval_question']}」")
            else:
                print("  没有语义近邻（阈值 %.2f）" % args.semantic_threshold)

    samples, dropped = dedupe(samples, args.overlap_threshold)

    # ------------------------------------------------------------------ 负样本（可选）
    negative_report: dict = {"requested": args.negatives, "built": 0, "reused": len(existing_negatives),
                             "samples": list(existing_negatives)}
    if args.negatives > 0:
        missing = args.negatives - len(existing_negatives)
        if missing <= 0:
            print(f"\n=== 负样本已够（{len(existing_negatives)} >= {args.negatives}），不再向教师请求")
        else:
            print(f"\n=== 生成负样本（再要 {missing} 条：有上下文但答不了 -> 应拒答）")
            fresh = _build_negatives(args, negative_per_block=args.negative_per_block,
                                     eval_qs=eval_qs, rejects_path=rejects_path)
            negative_report.update({k: v for k, v in fresh.items() if k != "samples"})
            negative_report["samples"] = list(existing_negatives) + list(fresh.get("samples") or [])
            negative_report["built"] = len(fresh.get("samples") or [])
            # 把新生成的负样本也落进 samples.jsonl（断点续跑）
            with samples_path.open("a", encoding="utf-8") as handle:
                for sample in fresh.get("samples") or []:
                    handle.write(json.dumps(render_record(sample, inference=True),
                                            ensure_ascii=False) + "\n")

    negative_samples = list(negative_report.pop("samples", []))
    mixed, mix_report = apply_negative_ratio(samples, negative_samples, args.negative_ratio,
                                             seed=args.seed)
    print(f"混合：正样本 {len(samples)} + 负样本 {mix_report['negatives_kept']} -> "
          f"{len(mixed)} 条（实际负样本占比 {mix_report['actual_ratio']}）")

    train, val = split_train_val(mixed, val_ratio=args.val_ratio, seed=args.seed)
    record_shape = not args.legacy_prompt_shape
    train_records = [render_record(s, inference=record_shape) for s in train]
    val_records = [render_record(s, inference=record_shape) for s in val]
    dump_jsonl(train_records, out_dir / "train.jsonl")
    dump_jsonl(val_records, out_dir / "val.jsonl")
    # 顺带把"重渲染好的"完整样本写一份，便于人工抽查提示长什么样
    dump_jsonl([render_record(s, inference=record_shape) for s in mixed], out_dir / "samples_rendered.jsonl")

    # ⚠️ **自检**：训练提示里到底有没有证据（2026-09-28 的事故就是这里没人查：
    #    796/796 条都没把法条放进提示，模型等于在学"凭记忆作答"）
    train_with_evidence = evidence_in_prompt_count(train_records)
    val_with_evidence = evidence_in_prompt_count(val_records)
    print(f"\n[自检] 提示里含证据的训练样本：{train_with_evidence}/{len(train_records)}"
          f"（验证集 {val_with_evidence}/{len(val_records)}）")
    if record_shape and train_records and train_with_evidence == 0:
        print("!! 训练提示里**一条证据都没有** —— 训练与推理不同形，模型只会学'凭记忆作答'。"
              "不要拿这份数据去训（这正是 2026-09-28 查出的根因）", file=sys.stderr)
        return 3

    stats = {
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "pending": len(pending), "accepted": counters["accepted"],
        "rejected": counters["rejected"], "errors": counters["error"],
        "samples_total": len(samples), "deduped": dropped,
        "semantic_dropped": semantic_dropped,
        "train": len(train), "val": len(val),
        "prompt_shape": "inference" if record_shape else "legacy(user/assistant)",
        "train_with_evidence_in_prompt": train_with_evidence,
        "val_with_evidence_in_prompt": val_with_evidence,
        "negatives": {**negative_report, **mix_report},
        "teacher": f"{args.teacher_url} ({args.teacher_model})",
        "eval_questions": len(eval_qs), "seed": args.seed,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("\n=== 汇总 ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"产出：{out_dir/'train.jsonl'}（{len(train)}） / {out_dir/'val.jsonl'}（{len(val)}）"
          f" / {out_dir/'rejects.jsonl'}")
    if counters["accepted"] == 0 and pending:
        print("!! 一条都没接受：先看 rejects.jsonl 的原因分布（教师端点/提示词/校验门槛）",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
