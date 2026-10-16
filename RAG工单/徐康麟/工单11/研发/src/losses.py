# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""损失函数：TripletLoss / CosineSimilarityLoss / ContrastiveLoss / MatryoshkaLoss。

工单原文《2、损失函数：引导训练过程》点名了四类损失，本模块全部自实现
（本机无 accelerate/peft，不使用 sentence-transformers 的 Trainer）。

约定：所有损失输入均为 **已 L2 归一化** 的句向量（encoder 内已归一化）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """余弦距离 d = 1 - cos ∈ [0, 2]（输入已归一化时等价于 1 - 内积）。"""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def pairwise_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """成对余弦距离矩阵 (N, M)，用于难负例挖掘。"""
    return 1.0 - a @ b.t()


class TripletLoss(nn.Module):
    """三元组损失（工单原文点名）。

    ``loss = relu( d(anchor, positive) - d(anchor, negative) + margin )``

    - ``距离`` 支持 ``cosine``（d = 1 - cos，范围 [0,2]）与 ``euclidean``。
    - ``hard_mining=True`` 时对每个锚点在 batch 内取最难的负例（in-batch hard negative）。
    """

    def __init__(self, margin: float = 0.3, distance: str = "cosine", hard_mining: bool = False) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin 必须 >= 0")
        self.margin = float(margin)
        self.distance = distance
        self.hard_mining = bool(hard_mining)

    def _dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance == "cosine":
            return cosine_distance(a, b)
        if self.distance == "euclidean":
            return torch.pairwise_distance(a, b, p=2)
        raise ValueError(f"不支持的距离度量：{self.distance}")

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor | None = None,
                *, return_parts: bool = False):
        if negative is None:
            # 无显式负例 -> in-batch 负例（同 batch 其它样本的正例作为负例）
            negative = torch.roll(positive, shifts=1, dims=0)
        d_ap = self._dist(anchor, positive)
        if self.hard_mining:
            sim = anchor @ negative.t()  # 余弦相似度矩阵
            idx = sim.argmax(dim=1)  # 最相似 = 最难负例
            d_an = self._dist(anchor, negative[idx])
        else:
            d_an = self._dist(anchor, negative)
        loss = F.relu(d_ap - d_an + self.margin)
        if return_parts:
            return loss.mean(), {"d_ap": d_ap.detach(), "d_an": d_an.detach()}
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    """余弦相似度损失（工单原文点名）：让预测余弦相似度逼近标注分数。

    ``pred = (cos(a,b) + 1) / 2``（映射到 [0,1]，可用 ``是否映射到01`` 关闭）
    ``loss = MSE(pred, score) * 损失缩放``
    """

    def __init__(self, 损失缩放: float = 20.0, 是否映射到01: bool = True) -> None:
        super().__init__()
        self.scale = float(损失缩放)
        self.rescale = bool(是否映射到01)

    def forward(self, a: torch.Tensor, b: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(a, b, dim=-1)
        pred = (cos + 1.0) / 2.0 if self.rescale else cos
        return F.mse_loss(pred, score.float()) * self.scale


class ContrastiveLoss(nn.Module):
    """对比损失（工单原文点名）。

    ``loss = y * d^2 + (1 - y) * relu(margin - d)^2``，默认欧氏距离。
    """

    def __init__(self, margin: float = 0.5, distance: str = "euclidean") -> None:
        super().__init__()
        self.margin = float(margin)
        self.distance = distance

    def forward(self, a: torch.Tensor, b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.distance == "cosine":
            d = cosine_distance(a, b)
        else:
            d = torch.pairwise_distance(a, b, p=2)
        y = label.float()
        pos = y * d.pow(2)
        neg = (1.0 - y) * F.relu(self.margin - d).pow(2)
        return (pos + neg).mean()


class MatryoshkaLoss(nn.Module):
    """套娃损失（工单原文点名）：在多个截断维度上分别计算基础损失并取平均。

    使嵌入向量具备「分层可截断」特性——前 k 维即可单独用于检索。
    """

    def __init__(self, base_loss: nn.Module, dims: tuple[int, ...] = (512, 256, 128, 64),
                 mode: str = "triplet", weights: tuple[float, ...] | None = None) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.dims = tuple(dims)
        self.mode = mode
        self.weights = tuple(weights) if weights else tuple(1.0 for _ in self.dims)

    @staticmethod
    def _truncate(x: torch.Tensor, dim: int) -> torch.Tensor:
        y = x[..., :dim]
        return F.normalize(y, p=2, dim=-1)

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None,
                *, score: torch.Tensor | None = None) -> torch.Tensor:
        total = torch.zeros((), dtype=a.dtype, device=a.device)
        wsum = 0.0
        for dim, w in zip(self.dims, self.weights):
            if dim > a.shape[-1]:
                continue
            if self.mode == "triplet":
                sub = self.base_loss(self._truncate(a, dim), self._truncate(b, dim),
                                     self._truncate(c, dim) if c is not None else None)
            elif self.mode == "cosine":
                sub = self.base_loss(self._truncate(a, dim), self._truncate(b, dim), score)
            else:
                raise ValueError(f"MatryoshkaLoss 不支持的模式：{self.mode}")
            total = total + w * sub
            wsum += w
        return total / max(wsum, 1e-8)


def build_loss(cfg_loss: dict) -> tuple[nn.Module, str, str]:
    """按配置构造损失函数。

    返回 ``(loss_module, 类型键, 说明)``。类型键取值：
    ``triplet`` / ``cosine`` / ``contrastive`` / ``matryoshka-triplet`` / ``combined``。
    ``combined`` 返回的组合模块带 ``.triplet`` / ``.cosine`` 子损失与 ``.w_t`` / ``.w_c`` 权重。
    """
    类型 = str(cfg_loss.get("类型", "triplet")).lower()
    t_cfg = cfg_loss.get("TripletLoss", {}) or {}
    c_cfg = cfg_loss.get("CosineSimilarityLoss", {}) or {}
    n_cfg = cfg_loss.get("ContrastiveLoss", {}) or {}
    w = cfg_loss.get("权重", {}) or {}

    triplet = TripletLoss(margin=t_cfg.get("margin", 0.3), distance=t_cfg.get("距离", "cosine"),
                          hard_mining=t_cfg.get("hard_mining", False))
    cosine = CosineSimilarityLoss(损失缩放=c_cfg.get("损失缩放", 20.0),
                                  是否映射到01=c_cfg.get("是否映射到01", True))
    contrastive = ContrastiveLoss(margin=n_cfg.get("margin", 0.5), distance=n_cfg.get("距离", "euclidean"))

    class Combined(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.triplet = triplet
            self.cosine = cosine
            self.w_t = float(w.get("triplet", 1.0))
            self.w_c = float(w.get("cosine", 1.0))

        def forward(self, anchor, positive, negative=None, score=None):
            parts = {}
            total = torch.zeros((), dtype=anchor.dtype, device=anchor.device)
            if self.w_t and negative is not None:
                lt = self.triplet(anchor, positive, negative)
                total = total + self.w_t * lt
                parts["triplet"] = float(lt.detach())
            if self.w_c and score is not None:
                lc = self.cosine(anchor, positive, score)
                total = total + self.w_c * lc
                parts["cosine"] = float(lc.detach())
            return total, parts

    if 类型 == "triplet":
        return triplet, "triplet", "TripletLoss(margin=%.2f, 距离=%s)" % (triplet.margin, triplet.distance)
    if 类型 == "cosine":
        return cosine, "cosine", f"CosineSimilarityLoss(缩放={cosine.scale})"
    if 类型 == "contrastive":
        return contrastive, "contrastive", f"ContrastiveLoss(margin={contrastive.margin})"
    if 类型 in ("matryoshka", "matryoshka-triplet", "套娃"):
        dims = tuple(t_cfg.get("截断维度", [512, 256, 128, 64]))
        return MatryoshkaLoss(triplet, dims=dims, mode="triplet"), "matryoshka-triplet", f"MatryoshkaLoss(TripletLoss, dims={dims})"
    if 类型 in ("triplet+cosine", "combined", "组合"):
        desc = (f"组合损失(TripletLoss w={w.get('triplet', 1.0)} + "
                f"CosineSimilarityLoss w={w.get('cosine', 1.0)})")
        return Combined(), "combined", desc
    raise ValueError(f"未知损失函数类型：{类型}")
