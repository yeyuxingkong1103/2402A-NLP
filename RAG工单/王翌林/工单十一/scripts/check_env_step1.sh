#!/bin/bash
# 工单四 环境检查脚本（人工智能NLP-RAG-图像内容解析及检索优化）
# 用途：Step 1 方案设计前的环境勘察（GPU/依赖包），非业务代码
PYTHON_BIN="/home/dabaie/code/my_project/.venv/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "NO_GPU"
$PYTHON_BIN - <<'PYEOF'
# 工单四 环境检查（人工智能NLP-RAG-图像内容解析及检索优化）
import importlib.util as u
for m in ["torch","transformers","cn_clip","clip","PIL","fitz","paddleocr","sentence_transformers","pymilvus"]:
    spec = u.find_spec(m)
    if spec:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    else:
        print(m, "MISSING")
import torch
print("cuda_available", torch.cuda.is_available())
PYEOF
