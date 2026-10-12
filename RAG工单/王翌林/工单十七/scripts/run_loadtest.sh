#!/bin/bash
cd /home/dabaie/code/工单/工单十七
export PYTHONPATH=$(pwd)
PY=/home/dabaie/code/my_project/.venv/bin/python3
echo "=== 优化版压测 (20并发, 45s) ==="
$PY scripts/load_test.py --mode optimized --port 8017 --out loadtest/result_opt.json --concurrency 20 --duration 45
echo "=== 基线版压测 (20并发, 45s) ==="
$PY scripts/load_test.py --mode baseline --port 8018 --out loadtest/result_base.json --concurrency 20 --duration 45
echo "ALL DONE"
