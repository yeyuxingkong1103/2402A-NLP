import logging
from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.app.core.config import settings
from backend.app.rerank.base import RerankClient

logger = logging.getLogger(__name__)


class BgeRerankClient(RerankClient):
    def __init__(self, model_path: str = settings.BGE_RERANKER_MODEL_PATH, model_dtype: str = settings.MODEL_DTYPE) -> None:
        # 保存本地模型路径，初始化时严格从本地加载。
        self.model_path = model_path
        # 保存精度配置，不允许静默降级。
        self.model_dtype = model_dtype
        # 将配置字符串转换为 torch dtype，当前任务只接受 FP16。
        self.torch_dtype = _resolve_torch_dtype(model_dtype)
        # 加载 tokenizer 时强制 local_files_only，防止联网下载。
        logger.info("loading bge reranker tokenizer", extra={"local_only": True})
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # 加载分类模型到 CUDA，并使用 FP16 权重。
        logger.info("loading bge reranker model", extra={"device": "cuda", "dtype": model_dtype})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=self.torch_dtype,
        )
        # 显式移动到 CUDA；如果 CUDA 或显存不可用，应由 readiness 测试暴露硬件阻塞。
        self.model.to("cuda")
        # 推理模式下关闭 dropout 等训练行为。
        self.model.eval()
        # 加载完成只记录状态，不输出路径或文本。
        logger.info("bge reranker model loaded", extra={"device": "cuda"})

    def score(self, query: str, documents: list[str]) -> list[float]:
        # 空文档无需模型推理，直接返回空分数。
        if not documents:
            return []
        # 记录脱敏输入统计，不输出 query 或 document 内容。
        _log_score_request(query, documents)
        # BGE reranker 接收 query/document 成对输入，顺序与 documents 保持一致。
        pairs = [[query, document] for document in documents]
        # tokenizer 生成张量并移动到 CUDA；截断避免超出模型最大长度。
        encoded = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to("cuda")
        # 推理阶段禁用梯度，降低显存占用但不改变硬件要求。
        with torch.no_grad():
            outputs = self.model(**encoded)
        # 二分类/单分数模型统一取 logits 展平后的分数。
        scores = outputs.logits.view(-1).float().detach().cpu().tolist()
        # 只记录分数数量，不记录具体分数或文本。
        logger.info("reranker scores produced", extra={"count": len(scores)})
        return scores


def _resolve_torch_dtype(model_dtype: str) -> torch.dtype:
    # 当前任务要求 FP16；其他值视为配置错误而不是自动降级。
    if model_dtype != "float16":
        raise ValueError("BGE models require MODEL_DTYPE=float16")
    # 返回 PyTorch 对应的半精度类型。
    return torch.float16


def _log_score_request(query: str, documents: Sequence[str]) -> None:
    # 统计长度用于排查异常输入，同时避免泄露用户文本全文。
    document_lengths = [len(document) for document in documents]
    # 只输出数量和长度，不输出 query/document 原文。
    logger.info(
        "reranker input received",
        extra={
            "query_chars": len(query),
            "document_count": len(documents),
            "min_document_chars": min(document_lengths),
            "max_document_chars": max(document_lengths),
        },
    )
