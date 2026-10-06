"""RAGAS 评测 CLI：批量跑 /eval/ragas 同款评测并落盘报告（G12）。

用法：
    # 单角色全量评测（默认数据集 data/eval/ragas_dataset.jsonl）
    python scripts/eval_ragas.py --persona-id 1

    # 限定条数 + 指定数据集
    python scripts/eval_ragas.py --persona-id 2 --limit 5 --dataset data/eval/ragas_dataset.jsonl

输出：终端打印指标摘要；报告 JSON 落盘到 data/eval/ragas_report_<persona>_ts.json。
评测依赖 LLM 在线接口（Judge），测试环境请勿运行。
"""
import argparse
import json
import os
import sys

# 把项目根目录插入 sys.path 首位，保证从任意目录运行都能 import 到 src.* 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logging import get_logger, setup_logging  # noqa: E402
from src.services import eval_service  # noqa: E402

logger = get_logger("scripts.eval")


def main() -> None:
    # argparse 定义评测参数：角色 ID 必填，其余可选用默认值
    parser = argparse.ArgumentParser(description="RAGAS 评测 CLI")
    parser.add_argument("--persona-id", type=int, required=True, help="心理医生角色 ID")
    parser.add_argument("--limit", type=int, default=10, help="最多评测条数（默认 10）")
    parser.add_argument("--dataset", default=None, help="数据集 JSONL 路径（默认 data/eval/ragas_dataset.jsonl）")
    args = parser.parse_args()

    setup_logging()
    # 调服务层真正执行 RAGAS 评测，返回报告 dict（含 metrics、样本、引擎、落盘路径等）
    report = eval_service.run_eval(args.persona_id, limit=args.limit, dataset_path=args.dataset)

    # 终端打印可读的评测摘要，便于快速查看指标
    metrics = report.get("metrics") or {}
    print("=" * 72)
    print(f"角色 ID   : {args.persona_id}")
    print(f"样本数    : {len(report.get('samples') or [])}")
    print(f"引擎      : {report.get('engine')}")
    for metric, value in metrics.items():
        print(f"{metric:<24}: {value}")
    print(f"报告文件  : {report.get('report_path')}")
    print("=" * 72)
    # 同时把指标写入日志（JSON 格式，ensure_ascii=False 保留中文可读性），便于归档检索
    logger.info("评测完成：%s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
