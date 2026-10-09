# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""用大模型从语料生成问答对，扩充训练数据

为什么需要这一步：
  只靠 FiQA 的 qrels 只能凑出 3,980 条三元组（453 个 query），
  实测在这么小的数据上微调，模型会从「已经很好的预训练状态」漂移出去 ——
  训练步数越多、检索指标越低（详见 优化/过程问题记录.md 问题 8）。
  补训练数据的办法是从语料本身生成问答对：拿一篇文档让大模型写一个
  「这篇文档能回答的问题」，(问题, 文档) 就成了一条训练正例。

防污染：生成用的文档**排除掉评估集里任何 query 的标准答案文档**，
否则等于把考题的答案提前喂给模型。

用法：python gen_qa.py [--n 2000] [--workers 8]
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

import data as D
from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL

OUT = D.DATA_DIR
DOCS_PER_CALL = 4          # 一次请求让模型处理几篇文档（省调用次数）
MAX_DOC_CHARS = 900        # 文档太长就截断，FiQA 文档中位 517 字符

# 提示词用英文写：语料是英文、模型是英文模型（bge-base-en-v1.5），
# 一开始用中文提示词，模型就把问题也生成成中文了 —— 那样会直接被
# 英文分词器切成 [UNK]，训练数据全废。语言必须和语料保持一致。
PROMPT = """You are annotating data for a financial question-answering system. \
Below are {k} documents from a finance forum, numbered from 0.

{docs}

For EACH document, write one question that the document answers. Requirements:
- The question must be specific and genuinely answerable from that document.
- Write it the way a real user would ask, 5-25 words.
- Do not say "according to the document" or "above".
- Do not copy whole sentences from the document.
- Write the question in ENGLISH.

Output ONLY a JSON array of exactly {k} strings, like:
["question 1", "question 2", "question 3", "question 4"]
No other text."""

_client = requests.Session()


def _request(prompt, max_tokens, timeout):
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,          # 生成任务要一点多样性，不然问题都一个腔调
        "max_tokens": max_tokens,
    }
    resp = _client.post(
        f"{LLM_API_BASE.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content") or ""


def _chat(prompt, timeout=120, max_tokens=2048):
    """调大模型，空正文就加大额度重试。

    deepseek-flash 是推理模型，reasoning token 先于正文消耗额度；
    额度被吃光时正文是空串。工单 01-10 的 rag_engine.chat() 里踩过同一个坑，
    这里沿用同样的处理：翻倍重试，最多三次。
    """
    for _ in range(3):
        content = _request(prompt, max_tokens, timeout).strip()
        if content:
            return content
        max_tokens = min(max_tokens * 2, 16384)
    return ""


def _parse_questions(raw, k):
    """从模型输出里抠出 JSON 数组。抠不出来就返回空，由调用方跳过。"""
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [str(x).strip() for x in arr[:k] if str(x).strip()]


def build_batch(docs):
    parts = [f"[文档 {i}]\n{d['text'][:MAX_DOC_CHARS]}" for i, d in enumerate(docs)]
    return PROMPT.format(k=len(docs), docs="\n\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="目标问答对数量")
    ap.add_argument("--workers", type=int, default=8, help="并发请求数")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    if not LLM_API_BASE or not LLM_API_KEY:
        print("[错误] 未配置 DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY")
        return 1

    print("=" * 68)
    print(f"生成问答对（目标 {args.n} 条，并发 {args.workers}）")
    print("=" * 68)

    corpus = D._read_jsonl(OUT / "corpus.jsonl")
    eval_set = json.loads((OUT / "eval_set.json").read_text(encoding="utf-8"))

    # 防污染：评估集里任何 query 的标准答案文档都不能用来生成
    forbidden = {d for ids in eval_set["qrels"].values() for d in ids}
    pool = [c for c in corpus if c["_id"] not in forbidden]
    print(f"语料 {len(corpus):,} 篇，排除评估集标准答案 {len(forbidden)} 篇 "
          f"→ 可用 {len(pool):,} 篇")

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_batches = (args.n + DOCS_PER_CALL - 1) // DOCS_PER_CALL
    batches = [pool[i * DOCS_PER_CALL:(i + 1) * DOCS_PER_CALL]
               for i in range(n_batches)]

    pairs, calls, failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_chat, build_batch(b)): b for b in batches}
        for fut in as_completed(futs):
            batch = futs[fut]
            calls += 1
            try:
                qs = _parse_questions(fut.result(), len(batch))
            except Exception:                                   # noqa: BLE001
                failed += 1
                qs = []
            for doc, q in zip(batch, qs):
                pairs.append({"query": q, "positive_id": doc["_id"]})
            if calls % 50 == 0:
                print(f"  已发 {calls}/{len(batches)} 次请求，"
                      f"得到 {len(pairs)} 条问答对（失败 {failed}）", flush=True)

    D.write_jsonl(OUT / "gen_pairs.jsonl", pairs)
    print(f"\n完成：{len(pairs)} 条问答对（{calls} 次调用，失败 {failed}）")
    print(f"已写入 {OUT / 'gen_pairs.jsonl'}")
    for p in pairs[:3]:
        print(f"  例：{p['query'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
