# -*- coding: utf-8 -*-
"""精排并发安全的回归测试。

本模块守护的 bug（2026-09-20 压测暴露）
======================================
并发 8 压测时，**27% 的请求失败**（12×HTTP 500 + 10×ReadError），
日志里刷满：

    Expected all tensors to be on the same device,
    but found at least two devices, cpu and cuda:0!

根因：`_model` 是进程内共享单例，而 `nn.Module.cuda()/.cpu()` 是**原地修改**。
精排为了给 Ollama 让显存，每次请求都要「搬上 GPU → 推理 → 搬回 CPU」，
但这段**没有锁**：

    线程 A: model.cuda()  →  正在 GPU 上推理
    线程 B: 取 _model、输入放到 cuda  →  此时 A 已 model.cpu()
    → 设备不一致

修复：用 `_gpu_lock` 把**整段设备切换+推理**串行化。
（只锁推理那一段不够 —— 设备切换本身才是竞态所在。）

⚠️ 为什么这个测试必须存在
------------------------
该 bug 的**失效是静默的**：`except` 会把异常吞掉并「降级为粗排顺序」，
接口照样返回 200，只是**结果质量悄悄退化成未精排**。
不并发跑就发现不了，而单测单线程也永远触发不了。
"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import rerank  # noqa: E402


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_gpu_lock_exists():
    """锁必须存在 —— 它是并发安全的唯一保障。"""
    assert hasattr(rerank, "_gpu_lock"), "缺少 _gpu_lock，并发设备切换会竞态"
    assert isinstance(rerank._gpu_lock, type(threading.Lock()))


def test_rerank_is_serialized_under_concurrency():
    """并发调用精排，必须**不出现设备不一致**、且结果全部带上 rerank_score。

    修复前：大量调用降级（rerank_score 全为 None）并刷错误日志；
    修复后：全部正常评分。
    """
    cands = [{"text": f"高血压的诊断标准第{i}条内容", "source": f"doc{i}"}
             for i in range(8)]

    errors: list[str] = []
    results: list[list[dict]] = []
    lock = threading.Lock()

    def call():
        cs = [dict(c) for c in cands]          # 每个线程独立副本，避免互相覆盖
        try:
            out = rerank.rerank("高血压的诊断标准是什么", cs, top_k=3)
            with lock:
                results.append(out)
        except Exception as e:
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: call(), range(16)))

    assert not errors, f"并发精排抛异常: {errors[:2]}"

    # 修复前这里的 rerank_score 会大量是 None（降级路径）
    degraded = sum(1 for r in results if r and r[0].get("rerank_score") is None)
    assert degraded == 0, (
        f"{degraded}/{len(results)} 次调用走了降级路径（rerank_score 为 None）—— "
        "说明精排仍然失败，很可能是设备切换又失去了锁保护"
    )

    # 每条结果都应有分数
    for r in results:
        for c in r:
            assert isinstance(c.get("rerank_score"), float)


def test_rerank_result_is_sorted():
    """并发不该破坏排序正确性。"""
    cands = [{"text": t, "source": f"d{i}"} for i, t in enumerate([
        "高血压的诊断标准是收缩压大于等于140毫米汞柱",
        "今天天气不错适合出门散步",
        "血压测量应在安静状态下进行",
    ])]
    out = rerank.rerank("高血压诊断标准", cands, top_k=3)
    scores = [c["rerank_score"] for c in out]
    assert scores == sorted(scores, reverse=True), f"结果未按精排分降序: {scores}"
