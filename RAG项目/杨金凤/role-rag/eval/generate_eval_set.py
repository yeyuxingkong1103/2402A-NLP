"""从 hypertension_guide 随机抽 30 条 chunk，用 DeepSeek 反推问答对，写 eval/eval_set.json。

用法（仓库根目录）：python eval/generate_eval_set.py
依赖真实 DeepSeek + Milvus。不 import rag，避免连带 Redis/MySQL/检索链路。
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()  # 先加载 .env 再 import milvus_store，保证 MILVUS_COLLECTION 读对

import milvus_store  # noqa: E402
from openai import OpenAI  # noqa: E402

logger = logging.getLogger(__name__)

COLLECTION = milvus_store.MILVUS_COLLECTION  # 默认 hypertension_guide
SAMPLE_SIZE = 30
SEED = 42
OUT_PATH = Path(__file__).resolve().parent / "eval_set.json"

GEN_PROMPT = """你是一名评测集标注员。请根据下面这段《中国高血压防治指南》文本，生成一道用于评测 RAG 问答系统的问答对。

要求：
1. 问题必须能由该片段直接回答，具体、事实性（数值、定义、诊断标准、用药、剂量、禁忌等），不要问"这段讲了什么"这类空泛问题。
2. 标准答案(ground_truth)必须完整、准确，严格依据片段内容，不得添加片段之外的信息。
3. 只输出 JSON，不要任何多余文字、不要 markdown 代码块：
{"question": "...", "ground_truth": "..."}

文本片段：
$chunk"""


def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY1")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY1，请在 .env 中填写")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _parse_json(content: str) -> dict | None:
    """剥掉代码块围栏后解析 JSON，失败再取第一个 {...} 块，仍失败返回 None。"""
    text = re.sub(r"^```[a-zA-Z]*\s*", "", content.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def generate_pair(client: OpenAI, model: str, chunk: str) -> dict:
    """调 DeepSeek 反推一条 (question, ground_truth)，解析失败重试一次。"""
    prompt = Template(GEN_PROMPT).substitute(chunk=chunk)
    last = ""
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        last = resp.choices[0].message.content or ""
        parsed = _parse_json(last)
        if parsed and parsed.get("question") and parsed.get("ground_truth"):
            return {
                "question": parsed["question"].strip(),
                "ground_truth": parsed["ground_truth"].strip(),
            }
        logger.warning("第 %d 次解析失败，重试", attempt + 1)
    raise RuntimeError(f"连续 2 次解析失败：{last[:200]}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    random.seed(SEED)
    rows = milvus_store.query_all(COLLECTION)
    if len(rows) < SAMPLE_SIZE:
        raise RuntimeError(f"{COLLECTION} 仅 {len(rows)} 条 chunk，不足 {SAMPLE_SIZE}")

    client = _get_client()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    eval_set = []
    for i, row in enumerate(random.sample(rows, SAMPLE_SIZE), 1):
        pair = generate_pair(client, model, row["content"])
        eval_set.append({
            "question": pair["question"],
            "ground_truth": pair["ground_truth"],
            "source_chunk_id": row["id"],
        })
        logger.info("[%d/%d] %s", i, SAMPLE_SIZE, pair["question"][:40])

    OUT_PATH.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写入 %s（%d 条）", OUT_PATH, len(eval_set))


if __name__ == "__main__":
    main()
