#!/bin/bash
# 工单四 Step 3：检查模型下载与 PaddleOCR 状态（人工智能NLP-RAG-图像内容解析及检索优化）
du -sh /home/dabaie/models/Qwen2-VL-2B-Instruct/ 2>/dev/null
ls /home/dabaie/models/Qwen2-VL-2B-Instruct/ 2>/dev/null | grep -c 'safetensors$'
/home/dabaie/code/my_project/.venv/bin/python -c "from paddleocr import PaddleOCR; print('PADDLE_OK')" 2>&1 | tail -1
