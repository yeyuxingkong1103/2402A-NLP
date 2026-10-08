"""BGE-M3 编码器 —— **全项目唯一的编码口径定义**（D3）。

批处理脚本（S5）与将来的后端查询（读路径）**共用本模块**，口径只有一处，
"抄错口径"的风险因此为零。改这里 = 改全链路，指纹会跟着变。

编码口径（docs/04 §8 锁定）：
    稠密向量 = 最后一层的 CLS（第 0 个 token）+ L2 归一化，1024 维，float32。
    归一化后 ‖v‖₂ = 1，余弦相似度退化为内积，与 Milvus 的 COSINE 度量一致。
"""

from __future__ import annotations

import hashlib
import os

from . import (
    BATCH_SIZE, DIM, EXIT_MODEL, MAX_LENGTH, MODEL_DIR, RULE_VERSION,
    WEIGHT_FILE, EmbedError,
)

# D2 裁决：指纹 = docs/04 §8 原定义（config 哈希 + 权重大小）+ 编码参数。
# 后四项是必须的——§8 的原定义挡不住"改了 max_length 却没重建库"，
# 而那恰恰是 §8 自己列为头号风险的漂移。
POOLING, NORMALIZATION, DTYPE = "cls", "l2", "float32"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(model_dir: str = MODEL_DIR, max_length: int = MAX_LENGTH) -> dict:
    """模型与编码参数的可辨识摘要。同一输入必须逐字符可复现。"""
    weight = os.path.join(model_dir, WEIGHT_FILE)
    config = os.path.join(model_dir, "config.json")
    if not os.path.isdir(model_dir):
        raise EmbedError(EXIT_MODEL, "权重目录不存在：%s" % model_dir)
    if not os.path.isfile(weight):
        raise EmbedError(EXIT_MODEL, "权重文件缺失：%s\n（下载不完整？完整约 2.27 GB）" % weight)
    if not os.path.isfile(config):
        raise EmbedError(EXIT_MODEL, "配置文件缺失：%s" % config)

    size = os.path.getsize(weight)
    if size < 1 << 30:                       # < 1 GB 基本可以断定是残缺的
        raise EmbedError(EXIT_MODEL,
                         "权重文件只有 %.1f MB，疑似下载不完整：%s" % (size / 1e6, weight))
    return {
        "model_dir": os.path.abspath(model_dir),
        "config_sha256": _sha256(config),
        "weight_file": WEIGHT_FILE,
        "weight_bytes": size,
        "max_length": max_length,
        "pooling": POOLING,
        "normalization": NORMALIZATION,
        "dtype": DTYPE,
        "dim": DIM,
        "rule_version": RULE_VERSION,
    }


class Encoder:
    """加载一次，反复编码。构造即校验权重与指纹。"""

    def __init__(self, model_dir: str = MODEL_DIR, max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EmbedError(EXIT_MODEL, "缺 torch/transformers：%s" % exc) from exc

        self.torch = torch
        self.batch_size, self.max_length = batch_size, max_length
        self.fingerprint = fingerprint(model_dir, max_length)
        try:
            self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        except Exception as exc:             # 权重损坏 / 格式不符
            raise EmbedError(EXIT_MODEL, "模型加载失败：%s\n  %s" % (model_dir, exc)) from exc
        self.model.eval()

    def token_lengths(self, texts: list[str]) -> list[int]:
        """逐条真实 token 数（含特殊符）。用于排序分批与超长报告。"""
        return [len(self.tok(t, add_special_tokens=True)["input_ids"]) for t in texts]

    def encode_documents(self, texts: list[str], on_batch=None):
        """批量编码。返回 float32 的 (N, 1024) 已归一化矩阵。

        按 token 长度**降序分批再还原顺序**：否则一个 3260 token 的长 chunk
        会把同批其余 15 条都 padding 到 3260，白白慢十几倍。

        on_batch(done, total) 是给调用方报进度用的可选回调；后端查询不需要它。
        """
        torch, out = self.torch, [None] * len(texts)
        ordered = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        with torch.no_grad():
            for start in range(0, len(ordered), self.batch_size):
                idx = ordered[start : start + self.batch_size]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state[:, 0]      # CLS
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
                for slot, i in enumerate(idx):
                    out[i] = hidden[slot]
                if on_batch is not None:
                    on_batch(min(start + self.batch_size, len(ordered)), len(ordered))
        return torch.stack(out).to(torch.float32)

    def encode_query(self, text: str):
        """编码单条用户问题。

        BGE-M3 的稠密检索对 query 与 passage 用**同一套编码**（CLS + L2，无指令
        前缀），故当前实现与 encode_documents 一致。接口单独留着，是为了将来
        换成需要 query 前缀的模型（如 bge-large-zh）时不必改调用方。
        """
        return self.encode_documents([text])[0]
