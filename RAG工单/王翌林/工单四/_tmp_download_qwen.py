# 临时脚本：续传 Qwen2-VL-2B-Instruct（人工智能NLP-RAG-图像内容解析及检索优化）
from modelscope import snapshot_download

p = snapshot_download(
    "Qwen/Qwen2-VL-2B-Instruct",
    local_dir="/home/dabaie/models/Qwen2-VL-2B-Instruct",
)
print("DOWNLOADED_TO", p, flush=True)
