"""本地环境自检：确认依赖、索引、模型、服务是否就绪。

与 `python -m app.main doctor` 的区别：本脚本只检查**运行环境**，
不加载问答引擎，因此即使索引损坏也能给出诊断结论，适合排障第一站。

用法:
    <项目解释器> tools/check_local_env.py
"""

from __future__ import annotations

import importlib
import shutil
import socket
import sys
from pathlib import Path

# 目录结构：<root>/设计/规格说明/check_local_env.py
#   parents[0]=规格说明 parents[1]=设计 parents[2]=项目根
ROOT = Path(__file__).resolve().parents[2]

DEPENDENCIES = [
    ("numpy", "数值计算"),
    ("pymupdf", "PDF 解析"),
    ("loguru", "日志"),
    ("jieba", "中文分词"),
    ("sqlalchemy", "SQLite ORM"),
    ("pydantic", "数据模型"),
    ("sentence_transformers", "向量模型"),
    ("torch", "深度学习后端"),
    ("transformers", "Whisper/翻译模型"),
    ("streamlit", "网页界面"),
    ("pytest", "测试"),
    ("openai", "LLM / 语音接口客户端"),
]

ARTIFACTS = [
    ("data/raw/招股说明书1.pdf", "语料 PDF"),
    ("data/processed/chunks.jsonl", "分块结果"),
    ("data/index/vectors.npy", "向量索引"),
    ("data/index/bm25_index.pkl", "BM25 索引"),
    ("data/index/rag.sqlite3", "SQLite（分块/对话/反馈/评估）"),
    ("data/eval/golden_qa.jsonl", "10 题标准答案"),
    # 评估报告按交付分类归档在「优化」阶段目录
    ("优化/评估结果/eval_results/ragas_report.md", "评估报告"),
    # 源码与配置按交付分类归档
    ("研发/app/core/qa_engine.py", "问答引擎源码"),
    ("研发/scripts/build_index.py", "建索引脚本"),
    ("部署/环境配置/requirements.txt", "依赖清单"),
    ("测试/tests/conftest.py", "测试基础设施"),
    ("设计/文档/TECH_DOC.md", "技术文档"),
]

MODELS = [
    ("models/bge-small-zh-v1.5", "嵌入模型（必需）"),
    ("models/whisper-tiny", "本地语音识别（可选）"),
    ("models/Qwen3-0.6B", "本地翻译/润色（可选）"),
]

PORTS = [
    (8501, "Streamlit 网页界面"),
    (8000, "vLLM LLM 服务"),
    (8001, "Whisper 语音服务"),
]


def human(size: float) -> str:
    return f"{size / 1024 / 1024:.1f} MB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def check_dependencies() -> int:
    print("\n[1/5] Python 依赖")
    print(f"      解释器: {sys.executable}")
    print(f"      版本  : {sys.version.split()[0]}")
    missing = 0
    for module, purpose in DEPENDENCIES:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "")
            print(f"      OK   {module:24} {version:<12} {purpose}")
        except Exception as exc:
            missing += 1
            print(f"      缺失 {module:24} {'':<12} {purpose}  ({type(exc).__name__})")
    return missing


def check_artifacts() -> int:
    print("\n[2/5] 索引与数据产物")
    missing = 0
    for relative, purpose in ARTIFACTS:
        path = ROOT / relative
        if path.exists():
            size = path.stat().st_size if path.is_file() else dir_size(path)
            print(f"      OK   {relative:42} {human(size):>10}  {purpose}")
        else:
            missing += 1
            print(f"      缺失 {relative:42} {'':>10}  {purpose}")
    return missing


def check_models() -> int:
    print("\n[3/5] 本地模型")
    missing_required = 0
    for relative, purpose in MODELS:
        path = ROOT / relative
        weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin")) if path.is_dir() else []
        if path.is_dir() and weights:
            print(f"      OK   {relative:32} {human(dir_size(path)):>10}  {purpose}")
        elif path.is_dir():
            print(f"      不完整 {relative:30} {'':>10}  只有配置文件、缺权重  {purpose}")
            if "必需" in purpose:
                missing_required += 1
        else:
            print(f"      缺失 {relative:32} {'':>10}  {purpose}")
            if "必需" in purpose:
                missing_required += 1
    return missing_required


def check_ports() -> None:
    print("\n[4/5] 服务端口")
    for port, name in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            running = sock.connect_ex(("127.0.0.1", port)) == 0
        print(f"      {'运行中' if running else '未启动'}  {port}  {name}")


def check_disk() -> None:
    print("\n[5/5] 磁盘")
    usage = shutil.disk_usage(ROOT)
    print(f"      可用 {usage.free / 1024**3:.1f} GB / 总计 {usage.total / 1024**3:.1f} GB")


def main() -> int:
    print("=" * 68)
    print(" 本地环境自检 —— 基于 PDF 文档的 RAG 问答系统（工单1）")
    print("=" * 68)
    print(f"项目根目录: {ROOT}")

    missing_deps = check_dependencies()
    missing_artifacts = check_artifacts()
    missing_models = check_models()
    check_ports()
    check_disk()

    print("\n" + "=" * 68)
    print(" 结论")
    print("=" * 68)
    ready = True
    if missing_deps:
        ready = False
        print(f"  ✗ 缺少 {missing_deps} 个 Python 依赖 → 运行 scripts/setup_env.ps1 或 pip install -r requirements.txt")
    if missing_artifacts:
        ready = False
        print(f"  ✗ 缺少 {missing_artifacts} 个数据产物 → 运行 python scripts/build_index.py")
    if missing_models:
        ready = False
        print(f"  ✗ 缺少必需的嵌入模型 → 放入 models/bge-small-zh-v1.5/")
    if ready:
        print("  ✓ 环境就绪。常用命令（在项目根目录执行）：")
        print("      启动界面 : pwsh -File 部署\\run.ps1 web")
        print("      跑测试   : pwsh -File 部署\\run.ps1 test")
        print("      跑评估   : pwsh -File 部署\\run.ps1 eval")
        print("      建索引   : pwsh -File 部署\\run.ps1 index")
        print("      命令行问 : pwsh -File 部署\\run.ps1 ask \"注册资本是多少？\"")
    print()
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
