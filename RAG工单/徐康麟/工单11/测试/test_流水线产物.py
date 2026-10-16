# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""测试四：配置、模型落盘与端到端流水线产物自检。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import Config


def test_配置可加载且阈值齐全():
    cfg = Config()
    assert cfg.hyper_get("训练", "学习率") > 0
    assert cfg.hyper_get("训练", "批大小") > 0
    assert cfg.hyper_get("损失函数", "TripletLoss", "margin") >= 0
    assert cfg.hyper_get("数据集", "最大序列长度") > 0
    assert cfg.hyper_get("评估", "ndcg_k") > 0


def test_路径均为绝对且语料存在():
    cfg = Config()
    for key in ("语料分块", "工单7测试问题集", "底座模型"):
        p = cfg.path(key)
        assert p.is_absolute(), f"{key} 不是绝对路径"
        assert p.exists(), f"{key} 不存在：{p}"


def test_配置可被环境变量覆盖(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WO11_CHUNKS", r"E:\某个\临时路径.jsonl")
    cfg = Config()
    assert str(cfg.path("语料分块")).endswith("临时路径.jsonl")


def test_未知路径键报错():
    with pytest.raises(KeyError):
        Config().path("不存在的键")


def test_微调模型已落盘(仓库根: Path):
    d = 仓库根 / "研发" / "data" / "finetuned"
    if not d.exists():
        pytest.skip("尚未完成微调训练")
    for name in ("config.json", "pytorch_model.bin", "vocab.txt", "工单11_微调信息.json"):
        assert (d / name).exists(), f"微调模型缺文件：{name}"
    info = json.loads((d / "工单11_微调信息.json").read_text(encoding="utf-8"))
    if "实际轮次" in info:
        assert info["实际轮次"] >= 1.0, f"训练轮次不足 1 个 epoch：{info['实际轮次']}"
        assert info["损失下降"] > 0, "损失未下降"


def test_训练日志与曲线存在(仓库根: Path):
    log = 仓库根 / "研发" / "data" / "train_log.jsonl"
    png = 仓库根 / "研发" / "data" / "loss_curve.png"
    if not log.exists():
        pytest.skip("尚未完成微调训练")
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) >= 10, "训练日志步数过少"
    for r in rows:
        assert set(("step", "loss", "lr")) <= set(r)
    assert png.exists() and png.stat().st_size > 5000, "损失曲线 PNG 缺失或过小"


def test_对比表存在且字段齐全(仓库根: Path):
    p = 仓库根 / "优化" / "评估结果" / "对比表.csv"
    if not p.exists():
        pytest.skip("尚未完成微调前后评估")
    text = p.read_text(encoding="utf-8-sig").splitlines()
    assert len(text) >= 5, "对比表行数过少"
    header = text[0].split(",")
    for col in ("指标组", "指标", "微调前", "微调后", "变化", "结论"):
        assert col in header, f"对比表缺列：{col}"


def test_截图目录中文命名(仓库根: Path):
    d = 仓库根 / "测试" / "测试截图"
    if not d.exists():
        pytest.skip("尚未生成截图")
    pngs = sorted(d.glob("*.png"))
    assert len(pngs) >= 4, f"截图数量不足：{len(pngs)}"
    for p in pngs:
        name = p.stem
        assert any("\u4e00" <= ch <= "\u9fff" for ch in name), f"截图文件名必须含中文：{p.name}"
        assert p.stat().st_size > 3000, f"截图疑似空图：{p.name}"
