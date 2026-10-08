#!/bin/bash
# 工单四 Step 3：安装 PaddleOCR（CPU 版 paddlepaddle，人工智能NLP-RAG-图像内容解析及检索优化）
set -x
PIP=/home/dabaie/code/my_project/.venv/bin/pip
$PIP install -q paddlepaddle paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -5
/home/dabaie/code/my_project/.venv/bin/python -c "from paddleocr import PaddleOCR; print('PADDLEOCR_OK')"
echo "DONE"
