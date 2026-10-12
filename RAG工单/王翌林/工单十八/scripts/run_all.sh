#!/bin/bash
set -e
cd /home/dabaie/code/工单/工单十八
PY=/home/dabaie/code/my_project/.venv/bin/python3
$PY scripts/gen_demo_dataset.py
$PY scripts/run_demo.py
echo DONE
