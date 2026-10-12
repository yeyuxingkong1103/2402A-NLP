#!/bin/bash
cd /home/dabaie/code/工单/工单十六
export PYTHONPATH=$(pwd)
mkdir -p finetune_output data/logs
echo "=== 重训 3 epochs ==="
/home/dabaie/code/my_project/.venv/bin/python3 src/finetune.py \
  --epochs 3 --lr 1e-4 --batch-size 1 --grad-accum 4 --no-4bit \
  2>&1 | tee data/logs/finetune.log
echo "DONE"
