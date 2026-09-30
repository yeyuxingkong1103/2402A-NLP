# -*- coding: utf-8 -*-
"""精排：bge-reranker-v2-m3（cross-encoder）。

显存策略（M0 实测结论，这是本项目最关键的一处工程取舍）：
    精排模型**不能常驻 GPU** —— 它会抢走 Ollama 的 KV cache，
    生成吞吐从 47.8 tok/s 暴跌到 5.2 tok/s（8G 显存红线）。

    正确做法：权重常驻 **CPU 内存**，每次精排时搬入 GPU、算完搬回。
    实测单轮开销：搬入 148ms + 前向 150ms + 搬回 115ms ≈ 413ms，
    而生成始终保有完整显存。因为精排发生在生成**之前**，两者本就不重叠。
"""
import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core import config
from ..core.logging import get_logger

log = get_logger("rerank")

_lock = threading.Lock()        # 保护「惰性加载」这一个动作
_tok = None
_model = None      # 常驻 CPU 内存（fp16）
_available: bool | None = None

# 保护「搬入 GPU → 推理 → 搬回 CPU」这一整段（见 rerank() 里的说明）。
# 与 _lock 分开：加载只发生一次，而设备切换每次请求都要做，粒度不同。
_gpu_lock = threading.Lock()


def _ensure():
    """惰性加载：模型放在 CPU，不占显存。"""
    global _tok, _model, _available
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        try:
            log.info("加载精排模型(CPU): %s", config.RERANK_MODEL_PATH)
            _tok = AutoTokenizer.from_pretrained(config.RERANK_MODEL_PATH)
            _model = AutoModelForSequenceClassification.from_pretrained(
                config.RERANK_MODEL_PATH, torch_dtype=torch.float16
            ).eval()          # 注意：不 .cuda()，常驻 CPU
            _available = True
            log.info("精排模型就绪（权重驻内存，用时才搬 GPU）")
        except Exception as e:
            _available = False
            log.error("精排模型加载失败，将降级为不精排: %s", e)


def is_available() -> bool:
    if _available is None:
        _ensure()
    return bool(_available)


def rerank(query: str, candidates: list[dict],
           top_k: int | None = None) -> list[dict]:
    """对候选做精排。

    入参 candidates 每项需含 "text"。返回按精排分降序的新列表，
    每项追加 rerank_score 字段。任何异常都降级为原序返回。
    """
    top_k = top_k or config.RERANK_TOP_K
    if not candidates:
        return []

    _ensure()
    if not _available:
        for c in candidates:
            c["rerank_score"] = None
        return candidates[:top_k]

    use_cuda = torch.cuda.is_available()
    model = _model

    # ⚠️ 整个「搬入 GPU → 推理 → 搬回 CPU」必须**串行**（2026-09-20 压测暴露）
    # ------------------------------------------------------------------
    # `_model` 是进程内共享单例，而 `nn.Module.cuda()/.cpu()` 是**原地修改**。
    # 无锁并发时会出现：
    #
    #     线程 A: model.cuda()  →  正在 GPU 上推理
    #     线程 B: 取 _model、输入放到 cuda  →  此时 A 已 model.cpu()
    #     → RuntimeError: Expected all tensors to be on the same device,
    #       but found at least two devices, cpu and cuda:0!
    #
    # 实测后果：并发 8 压测时**四分之一以上的请求精排失败**，
    # 虽然 except 兜底降级为粗排顺序（不至于 500），但**结果质量悄悄退化成未精排**，
    # 而且日志被同一条 ERROR 刷屏。
    #
    # GPU 本就是独占资源，并发精排无法真正并行，串行化没有损失。
    # 锁的粒度必须覆盖**设备切换**，而不只是推理那一段。
    with _gpu_lock:
        try:
            pairs = [
                (query, (c.get("text") or "")[:config.RERANK_MAX_SEQ * 4])
                for c in candidates
            ]
            enc = _tok(pairs, padding=True, truncation=True,
                       max_length=config.RERANK_MAX_SEQ, return_tensors="pt")

            if use_cuda:
                model = model.cuda()                  # 搬入

            with torch.inference_mode():
                inputs = {k: v.to("cuda" if use_cuda else "cpu")
                          for k, v in enc.items()}
                scores = model(**inputs).logits.view(-1).float().cpu().tolist()

            if use_cuda:
                model = model.cpu()                   # 搬回，把显存还给 Ollama
                torch.cuda.empty_cache()

        except Exception as e:
            log.error("精排失败，降级为粗排顺序: %s", e)
            try:
                if use_cuda:
                    _model.cpu()
                    torch.cuda.empty_cache()
            except Exception:
                pass
            for c in candidates:
                c["rerank_score"] = None
            return candidates[:top_k]

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 6)

    ranked = sorted(candidates, key=lambda c: -c["rerank_score"])
    log.info("精排完成 | %d 候选 -> top%d | 最高分 %.3f 最低分 %.3f",
             len(candidates), top_k, ranked[0]["rerank_score"], ranked[-1]["rerank_score"])
    return ranked[:top_k]


def warmup() -> None:
    """预热：把精排模型加载进内存（不占显存），避免首个请求等待。"""
    _ensure()


def release() -> None:
    """彻底释放模型（进程退出前调用）。

    ⚠️ 也要拿 `_gpu_lock`：否则若此刻正有请求在做精排，
    `_model = None` 会把正在推理的模型引用置空（设备切换期间尤其危险）。
    """
    global _model, _tok
    with _gpu_lock, _lock:
        _model = _tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
