# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""嵌入模型封装：本地离线加载 bge-small-zh-v1.5 + CLS 池化 + L2 归一化。

要点（实测，勿改）：
  - 底座目录 ``工单1\\models\\bge-small-zh-v1.5`` 的 ``1_Pooling/config.json`` 标明
    ``pooling_mode_cls_token=true``、``pooling_mode_mean_tokens=false``，即 **CLS 池化**；
    微调必须沿用同一池化方式，否则前后向量空间不可比。
  - 本机无 accelerate/peft，故不依赖 sentence-transformers 的 Trainer，
    仅用 transformers 的 ``AutoModel`` + 原生 PyTorch。
  - 默认只解冻「最后 2 层 + 池化」，既省 CPU 反向时间，又保留通用语义（防灾难性遗忘）。
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# 离线优先：避免任何联网探测（本机有网但下载会拖慢/挂死）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class EmbeddingModel(nn.Module):
    """可微调的句向量编码器。"""

    def __init__(self, model_dir: str | Path, *, max_length: int = 128, pooling: str = "cls",
                 trainable_layers: str = "最后2层+池化", device: str = "cpu") -> None:
        super().__init__()
        self.model_dir = str(model_dir)
        self.max_length = int(max_length)
        self.pooling = pooling
        self.device = torch.device(device)
        self.backbone = AutoModel.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.trainable_layers = trainable_layers
        self.to(self.device)
        self.apply_freeze_policy(trainable_layers)

    # ------------------------------------------------------------ 前向
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        if self.pooling == "cls":
            emb = last[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            emb = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            raise ValueError(f"不支持的池化方式：{self.pooling}")
        return F.normalize(emb, p=2, dim=-1)

    # ------------------------------------------------------------ 冻结策略
    def apply_freeze_policy(self, policy: str) -> dict[str, int]:
        for p in self.parameters():
            p.requires_grad = True
        n_layers = int(self.backbone.config.num_hidden_layers)
        frozen, total = 0, sum(p.numel() for p in self.parameters())
        if policy in ("全部", "all", "全参数"):
            return {"可训练参数": total, "总参数": total, "冻结参数": 0}
        keep = 2 if "2" in str(policy) else (1 if "1" in str(policy) else 0)
        # 冻结词嵌入
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        # 冻结前 n-keep 层
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < n_layers - keep:
                for p in layer.parameters():
                    p.requires_grad = False
        if getattr(self.backbone, "pooler", None) is not None:
            for p in self.backbone.pooler.parameters():
                p.requires_grad = True
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"可训练参数": total - frozen, "总参数": total, "冻结参数": frozen,
                "冻结前N层": n_layers - keep, "解冻后N层": keep}

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------ 编码
    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, batch_size: int = 64, max_length: int | None = None,
               progress_cb=None) -> np.ndarray:
        """批量编码；**内存不足时自动折半批大小重试**（本机内存紧张，实测会 OOM）。"""
        self.eval()
        ml = int(max_length or self.max_length)
        vecs: list[np.ndarray] = []
        total = len(texts)
        i = 0
        bs = max(1, int(batch_size))
        while i < total:
            batch = [str(t) for t in texts[i : i + bs]]
            try:
                enc = self.tokenizer(batch, padding="longest", truncation=True, max_length=ml,
                                     return_tensors="pt").to(self.device)
                v = self(enc["input_ids"], enc["attention_mask"])
            except (RuntimeError, MemoryError) as exc:  # noqa: PERF203
                msg = str(exc).lower()
                if bs > 1 and ("memory" in msg or "alloc" in msg):
                    bs = max(1, bs // 2)
                    print(f"[编码器][警告] 内存不足，编码批大小降为 {bs} 后重试")
                    gc.collect()
                    continue
                raise
            vecs.append(v.detach().cpu().numpy().astype("float32"))
            i += len(batch)
            if progress_cb is not None:
                progress_cb(min(i, total), total)
        if not vecs:
            return np.zeros((0, self.hidden_size), dtype="float32")
        return np.vstack(vecs)

    def encode_one(self, text: str, max_length: int | None = None) -> np.ndarray:
        return self.encode([text], batch_size=1, max_length=max_length)[0]

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    # ------------------------------------------------------------ 保存
    def save(self, out_dir: str | Path, *, meta: dict | None = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # 注意：safetensors 在本机保存时对共享存储张量会 MemoryError（实测），
        # 故统一用 PyTorch 原生 .bin 序列化（与底座目录中的 pytorch_model.bin 一致）。
        try:
            self.backbone.save_pretrained(str(out), safe_serialization=False)
        except Exception as exc:  # noqa: BLE001 —— 兜底：手工写 state_dict
            print(f"[编码器][警告] save_pretrained 失败，改用手工保存：{exc}")
            torch.save(self.backbone.state_dict(), out / "pytorch_model.bin")
            self.backbone.config.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        # 兼容 sentence-transformers 的目录结构（CLS 池化 + Normalize）
        (out / "1_Pooling").mkdir(exist_ok=True)
        (out / "1_Pooling" / "config.json").write_text(json.dumps({
            "word_embedding_dimension": self.hidden_size,
            "pooling_mode_cls_token": self.pooling == "cls",
            "pooling_mode_mean_tokens": self.pooling == "mean",
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "modules.json").write_text(json.dumps([
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ], ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "sentence_bert_config.json").write_text(json.dumps(
            {"max_seq_length": self.max_length, "do_lower_case": False}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (out / "工单11_微调信息.json").write_text(json.dumps(
            meta or {}, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path, **kw) -> "EmbeddingModel":
        return cls(path, **kw)


def set_threads(n: int) -> None:
    n = max(1, int(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # 已初始化过，忽略
