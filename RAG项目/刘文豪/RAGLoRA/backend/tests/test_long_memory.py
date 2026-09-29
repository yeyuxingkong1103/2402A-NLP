# -*- coding: utf-8 -*-
"""长期记忆的测试。

⚠️ 本文件守护的是一条**隐私级** bug
==================================
2026-09-20 实测发现：`recall(user_id=999999)` 返回了 `user_id=41` 的记录 ——
**任何用户都能读到别人的长期记忆**。

根因：`MilvusClient.hybrid_search(...)` **没有 `filter` 形参**，
传进去会被 `**kwargs` 静默吞掉。过滤必须挂在**每个 `AnnSearchRequest`** 的
`expr` 上。这与之前在 Qdrant 上修过的是同一类问题
（`prefetch + FusionQuery` 下顶层 `query_filter` 被忽略）——
**融合只合并排名，不重新过滤**，过滤必须在各路检索时就生效。

这个坑在本项目出现了**两次**（milvus_store 一次、long_memory 一次），
所以必须用测试锁住，而不是靠记性。

需要 Milvus；未启动时整文件跳过（长期记忆本就依赖它）。
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import long_memory  # noqa: E402


def _milvus_ready() -> bool:
    try:
        from app.services import milvus_store
        return milvus_store.health().get("ok", False)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _milvus_ready(), reason="需要 Milvus")


# ---------------------------------------------------------------- 用户隔离
def test_memory_is_isolated_per_user():
    """核心回归：一个用户绝不能召回另一个用户的记忆。

    这条断言极简 —— 但正是当初被违反的性质。
    """
    u1 = int(time.time() * 1000) % 10 ** 9
    u2 = u1 + 1

    assert long_memory.remember(u1, 1, "我的手机号是13800000000", "好的，已记录"), \
        "写入失败，无法验证隔离"

    # 另一个用户用**完全相同的问题**去召回，应当一条都拿不到
    other = long_memory.recall(u2, "我的手机号是多少", k=5)
    assert other == [], (
        f"用户 {u2} 召回了 {len(other)} 条本属于 {u1} 的记忆 —— 用户隔离失效！"
        "常见原因：hybrid_search 的 filter 被 kwargs 吞掉，"
        "没有下推到每个 AnnSearchRequest 的 expr。"
    )

    # 而本人要能取到
    mine = long_memory.recall(u1, "我的手机号是多少", k=5)
    assert mine, f"本人 {u1} 反而取不到自己的记忆"

    long_memory.forget(u1)


def test_exclude_conversation_filters_current():
    """当前会话的内容不该被长期记忆再召回一次（走短期记忆即可，重复注入白占上下文）。"""
    u = int(time.time() * 1000) % 10 ** 9 + 7
    long_memory.remember(u, 100, "我在服用氨氯地平", "好的")
    long_memory.remember(u, 200, "我在服用氨氯地平，剂量5mg", "好的")

    got = long_memory.recall(u, "我服用什么药", k=5, exclude_conversation=100)
    conv_ids = {m["conversation_id"] for m in got}
    assert 100 not in conv_ids, f"被排除的会话 100 仍出现在结果里: {conv_ids}"

    long_memory.forget(u)


def test_forget_removes_only_that_user():
    u1 = int(time.time() * 1000) % 10 ** 9 + 11
    u2 = u1 + 1
    long_memory.remember(u1, 1, "甲用户的秘密内容", "好")
    long_memory.remember(u2, 1, "乙用户的秘密内容", "好")

    long_memory.forget(u1)
    assert long_memory.recall(u1, "秘密内容", k=5) == [], "forget 未清干净"
    assert long_memory.recall(u2, "秘密内容", k=5), "forget 误删了别的用户"

    long_memory.forget(u2)


# ---------------------------------------------------------------- 纯逻辑
def test_format_for_prompt_empty():
    assert long_memory.format_for_prompt([]) == ""


def test_format_for_prompt_has_no_citation_markers():
    """记忆片段必须**明确告知模型它不是检索资料** —— 否则模型会拿历史对话当引用来源。"""
    txt = long_memory.format_for_prompt([{"text": "用户上周提过对 ACEI 过敏"}])
    assert "ACEI" in txt
    assert "不要" in txt or "不是" in txt


def test_max_id_stays_in_signed_int64():
    """主键必须落在 Milvus 的有符号 INT64 范围内。

    直接取 sha1 前 16 位是 64 位**无符号**，约半数会越界并报
    `Value out of range`（与 milvus_store.to_milvus_id 同一个坑）。
    """
    for i in range(200):
        mid = long_memory._mid(1, 2, f"文本{i}")
        assert 0 <= mid <= 2 ** 63 - 1
