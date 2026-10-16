# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""测试二：损失函数数值正确性（手算期望值逐值校验）。"""
from __future__ import annotations

import math

import pytest
import torch

from losses import (ContrastiveLoss, CosineSimilarityLoss, MatryoshkaLoss, TripletLoss,
                    build_loss, cosine_distance)


def t(*vals) -> torch.Tensor:
    return torch.tensor([list(vals)], dtype=torch.float32)


# ------------------------------------------------------------- TripletLoss
def test_三元组损失_完全命中时为零():
    """a==p 且 a⊥n：d(a,p)=0，d(a,n)=1，loss = relu(0-1+0.3) = 0。"""
    a, p, n = t(1, 0), t(1, 0), t(0, 1)
    loss = TripletLoss(margin=0.3)(a, p, n)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_三元组损失_手算值():
    """余弦距离域：d(a,p)=1（正交），d(a,n)=1（正交）→ loss = relu(1-1+0.3) = 0.3。"""
    a, p, n = t(1, 0), t(0, 1), t(0, -1)
    loss = TripletLoss(margin=0.3)(a, p, n)
    assert loss.item() == pytest.approx(0.3, abs=1e-6)


def test_三元组损失_margin生效():
    """margin=0 时同一组样本的损失应为 0（d_ap == d_an）。"""
    a, p, n = t(1, 0), t(0, 1), t(0, -1)
    assert TripletLoss(margin=0.0)(a, p, n).item() == pytest.approx(0.0, abs=1e-6)
    assert TripletLoss(margin=0.8)(a, p, n).item() == pytest.approx(0.8, abs=1e-6)


def test_三元组损失_batch取均值():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])   # 第一条 d=0，第二条 d=1
    n = torch.tensor([[0.0, -1.0], [0.0, -1.0]])  # d 均为 1
    loss = TripletLoss(margin=0.3)(a, p, n)
    assert loss.item() == pytest.approx((0.0 + 0.3) / 2, abs=1e-6)


def test_三元组损失_欧氏距离():
    """欧氏距离下 a=[0,0] p=[0,3] n=[0,4]：relu(3-4+0.5)=0。"""
    a, p, n = t(0, 0), t(0, 3), t(0, 4)
    assert TripletLoss(margin=0.5, distance="euclidean")(a, p, n).item() == pytest.approx(0.0, abs=1e-6)


def test_余弦距离定义():
    d = cosine_distance(t(1, 0), t(0, 1))
    assert d.item() == pytest.approx(1.0, abs=1e-6)
    assert cosine_distance(t(1, 0), t(1, 0)).item() == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------ CosineSimilarityLoss
def test_余弦相似度损失_手算值():
    """cos(a,b)=0 → pred=(0+1)/2=0.5；score=1.0 → MSE=0.25；缩放 1.0 → 0.25。"""
    a, b = t(1, 0), t(0, 1)
    score = torch.tensor([1.0])
    assert CosineSimilarityLoss(损失缩放=1.0)(a, b, score).item() == pytest.approx(0.25, abs=1e-6)


def test_余弦相似度损失_完全一致时为零():
    a = t(1, 0)
    assert CosineSimilarityLoss(损失缩放=1.0)(a, a, torch.tensor([1.0])).item() == pytest.approx(0.0, abs=1e-6)


def test_余弦相似度损失_缩放线性():
    a, b = t(1, 0), t(0, 1)
    score = torch.tensor([0.0])  # pred=0.5 → MSE=0.25
    assert CosineSimilarityLoss(损失缩放=2.0)(a, b, score).item() == pytest.approx(0.5, abs=1e-6)


def test_余弦相似度损失_关闭01映射():
    a, b = t(1, 0), t(0, 1)  # cos=0
    assert CosineSimilarityLoss(损失缩放=1.0, 是否映射到01=False)(a, b, torch.tensor([0.0])).item() \
        == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------- ContrastiveLoss
def test_对比损失_正例手算():
    """欧氏距离 0.3，label=1 → 0.3² = 0.09。"""
    a, b = t(0, 0), t(0.3, 0)
    assert ContrastiveLoss(margin=0.5)(a, b, torch.tensor([1.0])).item() == pytest.approx(0.09, abs=1e-6)


def test_对比损失_负例手算():
    """欧氏距离 0.3，label=0，margin=0.5 → relu(0.5-0.3)² = 0.04。"""
    a, b = t(0, 0), t(0.3, 0)
    assert ContrastiveLoss(margin=0.5)(a, b, torch.tensor([0.0])).item() == pytest.approx(0.04, abs=1e-6)


def test_对比损失_负例足够远时为零():
    a, b = t(0, 0), t(5, 0)
    assert ContrastiveLoss(margin=0.5)(a, b, torch.tensor([0.0])).item() == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------- MatryoshkaLoss
def test_套娃损失_单维度等于基础损失():
    a, p, n = t(1, 0), t(0, 1), t(-1, 0)
    base = TripletLoss(margin=0.3)
    m = MatryoshkaLoss(base, dims=(2,), mode="triplet")
    assert m(a, p, n).item() == pytest.approx(base(a, p, n).item(), abs=1e-6)


def test_套娃损失_多维度有限且为正():
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    n = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    v = MatryoshkaLoss(TripletLoss(margin=0.3), dims=(64, 32, 16), mode="triplet")(a, p, n)
    assert math.isfinite(v.item()) and v.item() >= 0


# ------------------------------------------------------------- build_loss
@pytest.mark.parametrize("类型,期望键", [
    ("triplet", "triplet"),
    ("cosine", "cosine"),
    ("contrastive", "contrastive"),
    ("matryoshka-triplet", "matryoshka-triplet"),
    ("triplet+cosine", "combined"),
])
def test_损失工厂可切换(类型: str, 期望键: str):
    mod, kind, desc = build_loss({"类型": 类型, "TripletLoss": {"margin": 0.3}})
    assert kind == 期望键
    assert isinstance(desc, str) and desc


def test_损失工厂_未知类型报错():
    with pytest.raises(ValueError):
        build_loss({"类型": "不存在的损失"})


def test_组合损失前向可用():
    mod, kind, _ = build_loss({"类型": "triplet+cosine",
                               "TripletLoss": {"margin": 0.3},
                               "CosineSimilarityLoss": {"损失缩放": 1.0},
                               "权重": {"triplet": 1.0, "cosine": 0.5}})
    a, p, n = t(1, 0), t(0, 1), t(-1, 0)
    loss, parts = mod(a, p, n, torch.tensor([0.2]))
    assert loss.item() > 0
    assert set(parts) == {"triplet", "cosine"}


def test_梯度可回传():
    a = torch.tensor([[1.0, 0.0]], requires_grad=True)
    p = torch.tensor([[0.6, 0.8]])
    n = torch.tensor([[-1.0, 0.0]])
    TripletLoss(margin=0.3)(a, p, n).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
