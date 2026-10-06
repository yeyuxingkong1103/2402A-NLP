"""evaluation/build_dataset.py —— 生成新架构（src/）的评测数据集。

在链路中的位置：
    独立脚本。产出 evaluation/dataset.json，供 evaluation/run_ragas.py 使用。

与 eval/ 目录的区别（两个目录都叫"评测"，但评的是两条不同的主线）：
    eval/        评 backend/ 这条主线的**检索**质量（recall@k / MRR）
    evaluation/  评 src/ 这条主线的**生成**质量（RAGAS）以及角色一致性

用法：
    python evaluation/build_dataset.py

数据集格式：每行 {question, ground_truth, role_id}
    question      测试问题
    ground_truth  标准答案要点（RAGAS 的 context_recall / context_precision 需要）
    role_id       该问题应该由哪个角色回答（供角色一致性评测使用）
"""
from __future__ import annotations

import json
from pathlib import Path

# 当前是一份内置的最小样例集，两题分别覆盖法律和心理两个角色。
#
# 注意这是**示例数据**，规模远不足以得出可信的结论：
# 两题的评测集上任何一个指标变化都可能只是噪声。
# 它的作用是"让整条评测流程能跑起来、格式看得清"，
# 真实评测需要按同样格式补充几十到上百题（覆盖各角色、各文档、以及无依据的拒答情形）。
DATASET = [
    {"question": "合同违约后我应该先准备哪些证据？", "ground_truth": "合同文本、付款凭证、沟通记录、交付记录", "role_id": "lawyer"},
    {"question": "焦虑时可以先做什么？", "ground_truth": "承认感受、呼吸、离开压力源、记录触发因素", "role_id": "psychologist"},
]


def main() -> int:
    """把内置数据集写到 evaluation/dataset.json。

    返回：
        0（成功）。

    用 with_name 而不是写死路径：
        路径相对于本文件所在目录，从任何工作目录执行都能正确落盘。
    """
    out = Path(__file__).with_name("dataset.json")
    out.write_text(json.dumps(DATASET, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
