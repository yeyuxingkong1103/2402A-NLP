"""从 ModelScope 下载 bge-m3 必要文件（跳过 onnx 与示例图片）"""
from modelscope import snapshot_download

target = r"C:\Users\cyh\Desktop\111\rag_roleplay\models\bge-m3"
path = snapshot_download(
    "BAAI/bge-m3",
    local_dir=target,
    ignore_patterns=["onnx/*", "imgs/*", "*.jpg", "*.png", "*.webp", "*.jpeg", ".DS_Store"],
)
print("DONE:", path)
