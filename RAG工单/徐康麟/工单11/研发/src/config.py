# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""统一配置加载（路径 + 超参），全部阈值/超参配置化，禁止硬编码。

优先级：环境变量（WO11_*） > 部署/配置/超参.json > 内置默认值。
中文键与英文键同时支持，便于文档与代码各自可读。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# 工单11 仓库根：本文件位于 <根>/研发/src/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "部署" / "配置"

# ---------------------------------------------------------------- 内置默认值
DEFAULT_PATHS: dict[str, Any] = {
    "语料分块": r"E:\gao6gongdan\工单7\研发\data\processed\chunks.jsonl",
    "工单7索引目录": r"E:\gao6gongdan\工单7\研发\data\index\bge-m3_1024",
    "工单7测试问题集": r"E:\gao6gongdan\工单7\研发\评估\测试问题集.jsonl",
    "底座模型": r"E:\gao6gongdan\工单1\models\bge-small-zh-v1.5",
    "数据集目录": r"研发\data\dataset",
    "微调模型目录": r"研发\data\finetuned",
    "评估目录": r"研发\评估",
    "日志目录": r"部署\日志",
    "截图目录": r"测试\测试截图",
    "对比结果目录": r"优化\评估结果",
}

DEFAULT_HYPER: dict[str, Any] = {
    "数据集": {
        "随机种子": 20260826,
        "最大序列长度": 128,
        "最小分块字符数": 24,
        "最大分块字符数": 600,
        "单文档最大采样数": 220,
        "留出每文档上限": 60,
        "每文档配额": {"指标数值": 130, "章节标题": 60, "表格标题": 30},
        "留出每文档配额": {"指标数值": 30, "章节标题": 16, "表格标题": 8},
        "训练集比例": 0.8,
        "验证集比例": 0.1,
        "测试集比例": 0.1,
        "训练三元组上限": 2400,
        "验证三元组上限": 300,
        "测试三元组上限": 300,
        "留出分块比例": 0.12,
        "每个锚点负例数": 2,
        "难负例同指标异公司": True,
        "包含相似度分数对": True,
        "包含正例对": True,
        "包含三元组": True,
    },
    "损失函数": {
        "类型": "triplet+cosine",
        "TripletLoss": {"margin": 0.3, "距离": "cosine", "hard_mining": False},
        "CosineSimilarityLoss": {"损失缩放": 1.0, "是否映射到01": True},
        "ContrastiveLoss": {"margin": 0.5},
        "权重": {"triplet": 1.0, "cosine": 0.5},
    },
    "训练": {
        "学习率": 3e-05,
        "权重衰减": 0.01,
        "批大小": 12,
        "训练轮数": 1,
        "最大步数": 117,
        "训练样本上限": 1400,
        "预热比例": 0.1,
        "梯度裁剪": 1.0,
        "线程数": 6,
        "优化器": "AdamW",
        "学习率调度": "cosine",
        "可训练层": "最后2层+池化",
        "混合精度": False,
        "设备": "cpu",
        "时间预算秒": 1080,
        "每N步评估": 25,
        "每N步记录损失": 1,
        "检查点间隔": 40,
    },
    "评估": {
        "topk": [1, 5, 10],
        "ndcg_k": 10,
        "端到端检索": True,
        "端到端编码批大小": 32,
        "端到端最大分块数": 2500,
        "真实问题topk": [1, 3, 5, 10],
        "相似度间隔阈值": 0.05,
        "bge-m3参照": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 —— 配置损坏不许裸崩
        print(f"[配置][警告] 读取失败，按默认值继续：{path} -> {exc}")
        return {}


def _abs(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


class Config:
    """配置对象：cfg.path("语料分块") / cfg.hyper("训练", "学习率")。"""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.repo_root = REPO_ROOT
        self.config_dir = config_dir or CONFIG_DIR
        self.paths = _deep_merge(DEFAULT_PATHS, _load_json(self.config_dir / "路径.json"))
        self.hyper = _deep_merge(DEFAULT_HYPER, _load_json(self.config_dir / "超参.json"))
        # 环境变量覆盖关键路径（便于换语料/换模型而不改文件）
        env_map = {
            "WO11_CHUNKS": "语料分块",
            "WO11_INDEX_DIR": "工单7索引目录",
            "WO11_QUESTIONS": "工单7测试问题集",
            "WO11_BASE_MODEL": "底座模型",
            "WO11_OUT_DIR": "微调模型目录",
        }
        for env_key, cfg_key in env_map.items():
            val = os.environ.get(env_key)
            if val:
                self.paths[cfg_key] = val

    # ------------------------------------------------------------- 访问器
    def path(self, key: str) -> Path:
        if key not in self.paths:
            raise KeyError(f"未知路径键：{key}；可用：{sorted(self.paths)}")
        return _abs(self.paths[key])

    def hyper_get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.hyper
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def ensure_dirs(self) -> None:
        for key in ("数据集目录", "微调模型目录", "评估目录", "日志目录", "截图目录", "对比结果目录"):
            self.path(key).mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        return {"仓库根": str(self.repo_root), "路径": self.paths, "超参": self.hyper}

    def summary_lines(self) -> list[str]:
        lines = [f"仓库根：{self.repo_root}"]
        lines += [f"路径·{k} = {self.path(k)}" for k in self.paths]
        lines += [
            f"超参·损失函数.类型 = {self.hyper_get('损失函数', '类型')}",
            f"超参·训练.学习率 = {self.hyper_get('训练', '学习率')}",
            f"超参·训练.批大小 = {self.hyper_get('训练', '批大小')}",
            f"超参·训练.最大步数 = {self.hyper_get('训练', '最大步数')}",
        ]
        return lines


def get_config() -> Config:
    return Config()


if __name__ == "__main__":
    cfg = Config()
    print(json.dumps(cfg.as_dict(), ensure_ascii=False, indent=2))
