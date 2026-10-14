# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""显存闸门的规格测试。

背景（本机实测，2026-10-14）：`torch.cuda.mem_get_info()` 在本机**不反映
Ollama 进程占用的显存** —— 先加载 embedder 后，无论 Ollama 是否常驻，
它都报 6009MB 空闲（而 nvidia-smi 显示整卡只剩 4014MB）。
旧闸门据此认为「显存充足」直接放行，随后精排模型加载触发**进程级段错误**
（Windows/CUDA 边界，Python 无法捕获，连 try/except 都来不及）。
实测：不释放 Ollama 时 3/3 崩；先释放再加载 3/3 成功。

因此规格是：**加载本地模型前必须无条件请 Ollama 释放显存**，
不得以 torch 的读数作为跳过释放的依据。
"""
import time

from src import llm, rag


def _fake_cuda(monkeypatch, free_bytes):
    """伪造一个「读数充裕」的 CUDA 环境。"""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda: (free_bytes, 8 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)   # 测试不真等


def test_headroom_calls_unload_even_when_reading_looks_plenty(monkeypatch):
    """读数充裕（8G 全空）时也必须释放 —— 该读数不可信。"""
    calls: list[int] = []
    monkeypatch.setattr(llm, "unload", lambda: calls.append(1))
    _fake_cuda(monkeypatch, free_bytes=8 * 1024 ** 3)

    rag._ensure_gpu_headroom(need_mb=3000)

    assert calls, ("显存读数显示充裕也必须先让 Ollama 释放：本机 mem_get_info "
                   "不反映 Ollama 占用，旧实现会因此跳过释放并段错误")


def test_headroom_calls_unload_when_reading_is_low(monkeypatch):
    """读数不足时同样要释放（原有行为不能回退）。"""
    calls: list[int] = []
    monkeypatch.setattr(llm, "unload", lambda: calls.append(1))
    _fake_cuda(monkeypatch, free_bytes=1024 * 1024 * 1024)

    rag._ensure_gpu_headroom(need_mb=3000)

    assert calls


def test_headroom_no_unload_without_cuda(monkeypatch):
    """无 CUDA（纯 CPU 机器）时不必打扰 Ollama。"""
    import torch
    calls: list[int] = []
    monkeypatch.setattr(llm, "unload", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    rag._ensure_gpu_headroom(need_mb=3000)

    assert calls == []


def test_headroom_survives_unload_failure(monkeypatch):
    """Ollama 没启动时 unload 内部吞异常，闸门本身不得抛错。"""
    def boom():
        raise RuntimeError("Ollama 没在跑")
    monkeypatch.setattr(llm, "unload", boom)
    _fake_cuda(monkeypatch, free_bytes=8 * 1024 ** 3)

    rag._ensure_gpu_headroom(need_mb=3000)      # 不抛即通过
