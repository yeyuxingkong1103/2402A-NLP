#!/bin/bash
cd /home/dabaie/code/工单/工单十六
export PYTHONPATH=$(pwd)
mkdir -p docs/screenshots data/logs
echo "=== 评估 baseline vs finetuned ==="
/home/dabaie/code/my_project/.venv/bin/python3 src/evaluate.py --max-eval 18 2>&1 | tee data/logs/evaluate.log
echo "=== 评估结束 ==="
