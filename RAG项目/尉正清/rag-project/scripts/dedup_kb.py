# -*- coding: utf-8 -*-
"""知识库去重：清理完全重复与高度近似的切片。

    .venv/bin/python -m scripts.dedup_kb            # 只统计，不动数据
    .venv/bin/python -m scripts.dedup_kb --apply    # 执行清理

判重分两级：
    完全重复  归一化（去空白与标点）后文本一模一样 —— 直接删多余的那几条
    高度近似  同来源内字符 shingle 的 Jaccard 相似度超阈值 —— 默认只报告不删，
              因为「近似」在法条这类语料里可能是两条真正不同的条文
              （例如刑法第133条与第133条之一），误删代价高

保留策略：同一组里保留**先入库的那条**（id 最小），其余删除。
"""
import argparse
import re
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from app.config.logging_conf import setup_logging
from app.config import settings
from app.db.milvus_conn import get_milvus

logger = logging.getLogger(__name__)

# 本脚本靠 logger.info 输出统计结果，必须配好日志：
# stdlib 的 root logger 默认只把 WARNING 以上送到 stderr，
# 不调这一步的话所有 info 都会被静默丢弃、终端上看不到任何输出
setup_logging()

SHINGLE = 12            # 字符 shingle 长度
NEAR_THRESHOLD = 0.92   # 近似判定的 Jaccard 阈值


def normalize(text: str) -> str:
    """去空白与标点，只留下实义字符。"""
    s = re.sub(r"[\s　]+", "", text or "")
    return re.sub(r"[，。、；：（）()\[\]【】《》\-—_·.,;:!?！？\"'“”‘’]", "", s)


def shingles(text: str, k: int = SHINGLE) -> set:
    if len(text) <= k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def fetch_rows(milvus) -> List[Dict]:
    """取回全部切片的 id / 文本 / 来源。"""
    rows, offset, batch = [], 0, 2000
    while True:
        part = milvus.client.query(
            collection_name=settings.MILVUS_COLLECTION,
            filter='role_id != ""',
            output_fields=["text", "role_id", "source"],
            limit=batch, offset=offset)
        if not part:
            break
        rows.extend(part)
        if len(part) < batch:
            break
        offset += batch
    return rows


def find_exact_dups(rows: List[Dict]) -> List[List[int]]:
    """返回 [[保留的 id, 待删 id, ...], ...]"""
    groups = defaultdict(list)
    for r in rows:
        groups[normalize(r.get("text", ""))].append(r["id"])
    return [[sorted(ids)[0]] + sorted(ids)[1:]
            for ids in groups.values() if len(ids) > 1]


def find_near_dups(rows: List[Dict], threshold: float = NEAR_THRESHOLD,
                   max_bucket: int = 200) -> List[Tuple[int, int, float]]:
    """同来源内按 shingle 倒排找近似对。

    只在**同一 source** 内比较：跨来源的相似内容往往是有意保留的
    （例如同一法条在法条库和问答库里各出现一次，用途不同）。
    """
    by_source = defaultdict(list)
    for r in rows:
        by_source[r.get("source", "?")].append(r)

    pairs = []
    for src, items in by_source.items():
        if len(items) < 2:
            continue
        # 先用首个 shingle 分桶，桶太大就跳过（避免 O(n²)）
        buckets = defaultdict(list)
        for r in items:
            sh = shingles(normalize(r.get("text", "")))
            if not sh:
                continue
            buckets[sorted(sh)[0]].append((r["id"], sh))

        for bucket in buckets.values():
            if len(bucket) < 2 or len(bucket) > max_bucket:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    sim = jaccard(bucket[i][1], bucket[j][1])
                    if sim >= threshold:
                        pairs.append((bucket[i][0], bucket[j][0], sim))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正执行删除")
    ap.add_argument("--near", action="store_true", help="连近似重复一起删")
    ap.add_argument("--threshold", type=float, default=NEAR_THRESHOLD)
    args = ap.parse_args()

    milvus = get_milvus()
    collection = settings.MILVUS_COLLECTION

    logger.info("读取集合 %s ...", collection)
    rows = fetch_rows(milvus)
    logger.info("共 %s 条切片", len(rows))

    # ---- 完全重复 ----
    exact = find_exact_dups(rows)
    exact_ids = [i for g in exact for i in g[1:]]
    logger.info("-" * 56)
    logger.info("完全重复：%s 组，冗余 %s 条", len(exact), len(exact_ids))
    for g in exact[:10]:
        text = next(r["text"] for r in rows if r["id"] == g[0])
        logger.info("  ×%s  %s", len(g), normalize(text)[:56])

    # ---- 近似重复 ----
    near = find_near_dups(rows, args.threshold)
    near_ids = sorted({b for _, b, _ in near})
    logger.info("-" * 56)
    logger.info("高度近似（Jaccard ≥ %s）：%s 对", args.threshold, len(near))
    for a, b, s in near[:10]:
        text = next(r["text"] for r in rows if r["id"] == a)
        logger.info("  %.3f  %s", s, normalize(text)[:56])

    to_delete = list(exact_ids)
    if args.near:
        to_delete += [i for i in near_ids if i not in to_delete]

    logger.info("=" * 56)
    logger.info("待删除合计 %s 条（占 %.2f%%）",
                len(to_delete), 100.0 * len(to_delete) / max(len(rows), 1))

    if not args.apply:
        logger.info("这是统计模式，未改动数据。确认后加 --apply 执行删除")
        return
    if not to_delete:
        logger.info("没有需要清理的数据")
        return

    for i in range(0, len(to_delete), 500):
        milvus.client.delete(collection_name=collection,
                             ids=to_delete[i:i + 500])
    milvus.client.flush(collection)
    logger.info("已删除 %s 条，集合现有 %s 条", len(to_delete),
                milvus.count(collection))


if __name__ == "__main__":
    main()
