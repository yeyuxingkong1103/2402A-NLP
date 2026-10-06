"""下载 embedding 模型 bge-m3（跳过示例图片，断点续传）"""
import os
from huggingface_hub import snapshot_download

# 使用国内镜像，加大超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

path = snapshot_download(
    "BAAI/bge-m3",
    ignore_patterns=["imgs/*", "*.jpg", "*.png", "*.jpeg"],
    resume_download=True,
)
print("DONE:", path)
