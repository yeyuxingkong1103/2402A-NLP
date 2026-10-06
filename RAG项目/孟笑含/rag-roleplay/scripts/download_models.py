# -*- coding: utf-8 -*-
"""下载 BGE-m3 与 BGE-reranker-v2-m3 到 models/。

国内网络可用 hf-mirror：
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py
用法：
    python scripts/download_models.py [--rerank]
"""
import os
# 解析：环境变量
import sys
# 解析：命令行参数
from pathlib import Path
# 解析：路径

# 本机环境可能全局设了 HF_HUB_OFFLINE=1，脚本内强制在线
os.environ["HF_HUB_OFFLINE"] = "0"
# 解析：覆盖离线开关（本机全局环境踩过坑）
# 国内网络：未显式设置时默认走 hf-mirror 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 解析：默认走镜像（用户显式设置则尊重）
# 镜像站不代理 Xet CAS 服务器，禁用 Xet 走经典 HTTP 下载
os.environ["HF_HUB_DISABLE_XET"] = "1"
# 解析：禁用 Xet（镜像不支持，401 踩坑修复）

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# 解析：模型目录
MODELS = [
    # 解析：模型清单（仓库ID、本地目录名）
    ("BAAI/bge-m3", "bge-m3"),
    # 解析：向量化模型
    ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
    # 解析：重排模型
]
# 仓库里的图片/演示文件不参与推理，且部分镜像站会对其返回 403
IGNORE_PATTERNS = ["*.DS_Store", "imgs/*", "*.jpg", "*.jpeg", "*.png", "*.gif", "*.md"]
# 解析：忽略清单（图片与文档不下载，防镜像 403 中断）


def _is_complete(dest: Path) -> bool:
    """config.json 可能在失败下载里残留；必须有 safetensors 权重才算完整。"""
    return any(dest.glob("*.safetensors")) or any(dest.glob("model*.bin"))
    # 解析：有权重文件才算下载完成（防半成品目录被误判完成）


# 下载 BGE-m3 / BGE-reranker 模型（幂等：有权重文件则跳过）
def main(download_rerank: bool = True) -> None:
    # 解析：下载主流程
    MODELS_DIR.mkdir(exist_ok=True)
    # 解析：确保目录
    targets = MODELS if download_rerank else MODELS[:1]
    # 解析：下载目标（可选跳过重排模型）
    for repo_id, local_name in targets:
        # 解析：逐模型
        dest = MODELS_DIR / local_name
        # 解析：本地路径
        if _is_complete(dest):
            # 解析：已完整下载
            print(f"[跳过] {repo_id} 已存在：{dest}")
            # 解析：跳过
            continue
            # 解析：下一个
        print(f"[下载] {repo_id} -> {dest}")
        # 解析：提示
        from huggingface_hub import snapshot_download
        # 解析：延迟导入 huggingface_hub

        snapshot_download(repo_id=repo_id, local_dir=str(dest), ignore_patterns=IGNORE_PATTERNS)
        # 解析：下载快照（忽略图片/文档）
        print(f"[完成] {dest}")
        # 解析：完成提示


if __name__ == "__main__":
    # 解析：入口
    main(download_rerank="--rerank" in sys.argv or len(sys.argv) == 1)
    # 解析：默认下载全部；--rerank 显式（兼容旧用法）
