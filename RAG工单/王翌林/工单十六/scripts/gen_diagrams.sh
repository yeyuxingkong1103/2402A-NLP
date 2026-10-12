#!/bin/bash
# 生成工单十六的设计类截图（架构图、流程图、思维导图、训练loss曲线）
cd /home/dabaie/code/工单/工单十六
mkdir -p docs/screenshots
rm -rf /home/dabaie/.cache/matplotlib 2>/dev/null
/home/dabaie/code/my_project/.venv/bin/python3 scripts/gen_diagrams.py 2>&1 | tail -20
ls docs/screenshots/*.png
