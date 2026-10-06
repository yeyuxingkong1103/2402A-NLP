# -*- coding: utf-8 -*-
"""评测题库：加载 JSON 并校验字段完整性。"""
import json
# 解析：JSON 模块（读题库）
from dataclasses import dataclass
# 解析：数据类（题库结构）
from pathlib import Path
# 解析：路径处理


@dataclass
# 解析：题库数据结构
# 评测题库数据结构：名称 / 角色 / 样本列表
class EvalDataset:
    # 解析：题库数据类
    name: str
    # 解析：题库名
    role: str
    # 解析：评测角色名
    samples: list[dict]
    # 解析：题目列表


def load_eval_dataset(path: Path) -> EvalDataset:
    """加载题库；字段缺失时抛 ValueError，指明第几题缺什么。"""
    if not path.exists():
        # 解析：文件不存在
        raise FileNotFoundError(f"题库文件不存在: {path}")
        # 解析：报错
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 解析：读 JSON

    if not payload.get("samples"):
        # 解析：无样本
        raise ValueError("题库缺少 samples 字段或为空")
        # 解析：报错

    for i, item in enumerate(payload["samples"], 1):
        # 解析：逐题校验（i 从 1 开始用于报错信息）
        if not item.get("question"):
            # 解析：缺问题
            raise ValueError(f"第 {i} 题缺少 question")
            # 解析：指明第几题缺什么
        if not item.get("reference_answer"):
            # 解析：缺参考答案
            raise ValueError(f"第 {i} 题缺少 reference_answer")
            # 解析：报错
        if not item.get("reference_contexts"):
            # 解析：缺标准上下文
            raise ValueError(f"第 {i} 题缺少 reference_contexts")
            # 解析：报错

    return EvalDataset(
        # 解析：构造题库对象
        name=payload.get("name", "unnamed"),
        # 解析：名称（缺省 unnamed）
        role=payload.get("role", ""),
        # 解析：角色（缺省空串）
        samples=payload["samples"],
        # 解析：样本
    )


def load_samples_from_report(path: Path) -> list[dict]:
    """从 run_eval 保存的报告 JSON 加载已生成样本（用于换 judge 重判，不重新生成）。"""
    if not path.exists():
        # 解析：文件不存在
        raise FileNotFoundError(f"报告文件不存在: {path}")
        # 解析：报错
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 解析：读报告 JSON
    samples = payload.get("samples")
    # 解析：取已生成样本
    if not samples:
        # 解析：无样本
        raise ValueError("报告缺少 samples 字段，无法重判（需用 run_eval 保存的完整报告）")
        # 解析：报错
    return samples
    # 解析：返回样本（重判时复用生成结果，不重新调大模型）
