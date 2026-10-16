# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""测试一：数据集格式与防泄漏校验。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

必填字段 = {"id", "格式", "anchor", "positive", "来源", "文档", "正向分块"}
合法格式 = {"正例对", "三元组", "相似度分数对"}


def test_数据集文件齐全(数据集目录: Path):
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "留出测试集.jsonl", "数据集统计.json"):
        assert (数据集目录 / name).exists(), f"缺少 {name}"


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_每行schema合法(split: str, 数据集目录: Path, 读jsonl函数):
    rows = 读jsonl函数(数据集目录 / f"{split}.jsonl")
    assert rows, f"{split}.jsonl 为空"
    for i, r in enumerate(rows):
        missing = 必填字段 - set(r)
        assert not missing, f"{split} 第{i+1}行缺字段 {missing}"
        assert r["格式"] in 合法格式, f"{split} 第{i+1}行格式非法：{r['格式']}"
        assert isinstance(r["anchor"], str) and r["anchor"].strip(), f"{split} 第{i+1}行 anchor 为空"
        assert isinstance(r["positive"], str) and r["positive"].strip(), f"{split} 第{i+1}行 positive 为空"


def test_三种格式均有覆盖(数据集目录: Path, 读jsonl函数, 数据集统计: dict):
    rows = 读jsonl函数(数据集目录 / "train.jsonl")
    格式集 = {r["格式"] for r in rows}
    assert 格式集 == 合法格式, f"训练集格式覆盖不全：{格式集}"
    分布 = 数据集统计["各split格式分布"]["train"]
    for f in 合法格式:
        assert 分布.get(f, 0) > 0, f"统计中 {f} 数量为 0"


def test_三元组必须有负例且不等于正例(数据集目录: Path, 读jsonl函数):
    rows = 读jsonl函数(数据集目录 / "train.jsonl")
    tri = [r for r in rows if r["格式"] == "三元组"]
    assert tri, "训练集无三元组"
    for r in tri:
        assert r.get("negative"), f"三元组 {r['id']} 缺负例"
        assert r["negative"].strip() != r["positive"].strip(), f"三元组 {r['id']} 负例与正例相同"


def test_相似度分数对取值范围(数据集目录: Path, 读jsonl函数):
    rows = 读jsonl函数(数据集目录 / "train.jsonl")
    sim = [r for r in rows if r["格式"] == "相似度分数对"]
    assert sim, "训练集无相似度分数对"
    for r in sim:
        s = r.get("score")
        assert isinstance(s, (int, float)), f"{r['id']} score 缺失"
        assert 0.0 <= float(s) <= 1.0, f"{r['id']} score 越界：{s}"
    # 正例对应为 1.0
    pos = [r for r in sim if r.get("负例类型") == "正例"]
    assert pos and all(abs(float(r["score"]) - 1.0) < 1e-9 for r in pos), "正例对的相似度分数必须为 1.0"


def test_留出测试集与训练集无交集(数据集目录: Path, 读jsonl函数):
    """防泄漏核心断言：留出测试集的金标准分块，在训练集中既不做正例也不做负例。

    注：多条查询可以指向同一个金标准分块（同一分块含多个指标），故只要求
    金标准唯一率 ≥ 90%，不做严格唯一性断言。
    """
    hold = 读jsonl函数(数据集目录 / "留出测试集.jsonl")
    assert hold, "留出测试集为空"
    gold = {r["gold_chunk_id"] for r in hold}
    assert len(gold) >= len(hold) * 0.9, f"金标准分块重复过多：{len(gold)}/{len(hold)}"
    train = 读jsonl函数(数据集目录 / "train.jsonl")
    used = set()
    for r in train:
        used.add(r["正向分块"])
        if r.get("负向分块"):
            used.add(r["负向分块"])
    overlap = gold & used
    assert not overlap, f"训练集出现留出金标准分块（泄漏）：{list(overlap)[:5]}"


def test_留出测试集字段完整(数据集目录: Path, 读jsonl函数):
    hold = 读jsonl函数(数据集目录 / "留出测试集.jsonl")
    for r in hold:
        assert r.get("query", "").strip(), "缺失 query"
        assert r.get("gold_chunk_id", "").strip(), "缺失 gold_chunk_id"
        assert r.get("文档"), "缺失 文档"


def test_统计MD5与实际文件一致(数据集目录: Path, 数据集统计: dict):
    for name, md5 in 数据集统计["文件MD5"].items():
        p = 数据集目录 / name
        assert p.exists(), f"统计里登记的 {name} 不存在"
        actual = hashlib.md5(p.read_bytes()).hexdigest()
        assert actual == md5, f"{name} 的 MD5 与统计不一致（文件被改动过）：{actual} != {md5}"


def test_锚点文本不含答案数值(数据集目录: Path, 读jsonl函数):
    """指标数值类锚点的问题里不得出现该数值（否则是抄答案）。

    只检查长度 ≥3 的数值串，避免 "1" 命中 "2019" 这类子串误判。
    """
    rows = 读jsonl函数(数据集目录 / "train.jsonl")
    hit = []
    for r in rows:
        v = r.get("数值")
        if v and r.get("指标"):
            num = str(v).rstrip("%").replace("亿元", "").replace("万元", "").replace("元", "")
            if len(num) >= 3 and num in r["anchor"]:
                hit.append((r["指标"], v, r["anchor"]))
    assert not hit, f"有 {len(hit)} 条锚点把答案数值写进了问题，例如：{hit[:3]}"
