# -*- coding: utf-8 -*-
"""独立块 content 拼接与保护性截断测试"""

from scripts.chunk_split import compose_block_content


def test_无描述时原样返回图注():
    assert compose_block_content("图 2 软件质量模型", "", max_chars=2000) == "图 2 软件质量模型"


def test_无图注时返回描述():
    assert compose_block_content("", "一段描述", max_chars=2000) == "一段描述"


def test_两者都有时拼接并带标记():
    result = compose_block_content("图 2 软件质量模型", "树状结构图", max_chars=2000)
    assert result == "图 2 软件质量模型\n[视觉解析] 树状结构图"


def test_超长时图注保持完整():
    caption = "图 2 软件质量模型"
    desc = "很长的描述" * 1000
    result = compose_block_content(caption, desc, max_chars=100)
    assert result.startswith(caption + "\n[视觉解析] ")
    assert len(result) <= 100


def test_超长时描述被截断并带省略号():
    caption = "图注"
    desc = "描述内容" * 500
    result = compose_block_content(caption, desc, max_chars=50)
    assert result.endswith("…")
    assert len(result) <= 50


def test_不超长时不加省略号():
    result = compose_block_content("图注", "短描述", max_chars=2000)
    assert not result.endswith("…")


def test_图注本身超长时硬截断():
    caption = "超长图注" * 1000
    result = compose_block_content(caption, "描述", max_chars=20)
    assert len(result) <= 20


def test_边界_max_chars_恰好等于拼接长度():
    caption = "a" * 10
    desc = "b" * 10
    # "\n[视觉解析] " 共 8 个字符
    result = compose_block_content(caption, desc, max_chars=28)
    assert result == caption + "\n[视觉解析] " + desc
    assert len(result) == 28


def test_预算为零或负时只保留图注():
    result = compose_block_content("abcdefghij", "描述" * 100, max_chars=5)
    assert result == "abcde"


def test_去除首尾空白():
    assert compose_block_content("  图注  ", "  描述  ", max_chars=2000) == "图注\n[视觉解析] 描述"


def test_结果永不超限_多组参数():
    """属性式校验：任意输入下输出长度都不超过 max_chars"""
    import random

    rng = random.Random(20260920)          # 固定种子，保证可复现
    for _ in range(500):
        caption = "图" * rng.randint(0, 60)
        desc = "述" * rng.randint(0, 400)
        max_chars = rng.randint(1, 300)
        result = compose_block_content(caption, desc, max_chars=max_chars)
        assert len(result) <= max_chars, (
            f"超限：len={len(result)} max_chars={max_chars} "
            f"caption_len={len(caption)} desc_len={len(desc)}"
        )
