# -*- coding: utf-8 -*-
"""
4 种损失函数 - Triplet / Contrastive / Cosine / Matryoshka
工单编号: 人工智能 NLP-RAG 项目-Embedding 模型微调任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TripletLoss(nn.Module):
    """
    三元组损失: max(0, d(anchor, positive) - d(anchor, negative) + margin)

    促使 anchor 和 positive 靠近, 和 negative 远离
    """
    def __init__(self, margin: float = 0.5, distance_type: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type

    def forward(self, anchor, positive, negative):
        if self.distance_type == "cosine":
            # cos_sim 越大越近, 所以用 1 - cos_sim
            d_ap = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            d_an = 1 - F.cosine_similarity(anchor, negative, dim=-1)
        else:
            d_ap = F.pairwise_distance(anchor, positive)
            d_an = F.pairwise_distance(anchor, negative)

        loss = torch.clamp(d_ap - d_an + self.margin, min=0.0).mean()
        return loss


class ContrastiveLoss(nn.Module):
    """
    对比损失 (Supervised Contrastive Learning)

    正负例对: y=1 正例, y=0 负例
    loss = y * log(sim) + (1-y) * log(1 - sim)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels=None):
        """
        z1, z2: (batch, dim) 两组嵌入
        labels: (batch,) 1=正例 0=负例, 如果 None 则默认全部正例
        """
        sim = F.cosine_similarity(z1, z2, dim=-1) / self.temperature

        if labels is None:
            # 默认全部正例
            labels = torch.ones(z1.size(0), device=z1.device)

        # BCE with logits
        loss = F.binary_cross_entropy_with_logits(sim, labels.float())
        return loss


class CosineSimilarityLoss(nn.Module):
    """
    余弦相似度损失: 1 - cos_sim(emb1, emb2) 接近 target_score

    用于带相似度分数的句子对训练
    """
    def __init__(self):
        super().__init__()

    def forward(self, emb1, emb2, target_scores):
        """
        emb1, emb2: (batch, dim)
        target_scores: (batch,) 预期相似度 [0, 1]
        """
        pred_sim = F.cosine_similarity(emb1, emb2, dim=-1)
        # MSE loss between predicted and target
        loss = F.mse_loss(pred_sim, target_scores.float())
        return loss


class MatryoshkaLoss(nn.Module):
    """
    套娃损失: 分层可截断的 Triplet Loss

    不同截断维度: [32, 64, 128, 256, full]
    每个维度都计算 Triplet Loss, 加权求和
    """
    def __init__(self, marginal_dims=None, margin: float = 0.5):
        super().__init__()
        self.marginal_dims = marginal_dims or [32, 64, 128, 256]
        self.margin = margin
        self.triplet = TripletLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        total_loss = 0.0
        dim = anchor.size(-1)

        # 截断维度不能超过实际维度
        cut_dims = [d for d in self.marginal_dims if d <= dim]

        for cut in cut_dims:
            a_cut = anchor[:, :cut]
            p_cut = positive[:, :cut]
            n_cut = negative[:, :cut]
            total_loss += self.triplet(a_cut, p_cut, n_cut)

        # 全维度也计算一次
        total_loss += self.triplet(anchor, positive, negative)

        return total_loss / (len(cut_dims) + 1)


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """获取损失函数"""
    losses = {
        "triplet": lambda: TripletLoss(**{k: v for k, v in kwargs.items() if k in ["margin", "distance_type"]}),
        "contrastive": lambda: ContrastiveLoss(**{k: v for k, v in kwargs.items() if k in ["temperature"]}),
        "cosine": lambda: CosineSimilarityLoss(),
        "matryoshka": lambda: MatryoshkaLoss(**{k: v for k, v in kwargs.items() if k in ["marginal_dims", "margin"]}),
    }
    factory = losses.get(name, losses["triplet"])
    return factory()


if __name__ == "__main__":
    import torch

    anchor = torch.randn(4, 64)
    positive = torch.randn(4, 64)
    negative = torch.randn(4, 64)

    for name in ["triplet", "contrastive", "cosine", "matryoshka"]:
        loss_fn = get_loss_function(name)
        if name == "cosine":
            targets = torch.tensor([0.9, 0.8, 0.1, 0.0])
            loss = loss_fn(anchor, positive, targets)
        else:
            loss = loss_fn(anchor, positive, negative)
        print(f"[OK] {name}: loss={loss.item():.4f}")
