#!/bin/bash
# 工单四 Step 3：下载 Qwen2-VL-2B-Instruct 到本地模型库（人工智能NLP-RAG-图像内容解析及检索优化）
# 优先 modelscope（国内快），失败则走 hf-mirror
set -x
TARGET=/home/dabaie/models/Qwen2-VL-2B-Instruct
if [ -d "$TARGET" ] && ls "$TARGET"/*.safetensors >/dev/null 2>&1; then
    echo "ALREADY_EXISTS"
    exit 0
fi
PIP=/home/dabaie/code/my_project/.venv/bin/pip
$PIP list 2>/dev/null | grep -qi modelscope || $PIP install -q modelscope
/home/dabaie/code/my_project/.venv/bin/python - <<'PYEOF'
# 工单四：modelscope 下载 Qwen2-VL-2B-Instruct（人工智能NLP-RAG-图像内容解析及检索优化）
from modelscope import snapshot_download
p = snapshot_download("Qwen/Qwen2-VL-2B-Instruct",
                      local_dir="/home/dabaie/models/Qwen2-VL-2B-Instruct")
print("DOWNLOADED_TO", p)
PYEOF
echo "DONE"
