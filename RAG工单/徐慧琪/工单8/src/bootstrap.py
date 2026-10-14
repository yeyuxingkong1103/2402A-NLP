# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：离线引导（必须最先导入）

职责（顺序敏感）：
  1. 置离线环境变量 —— 任何 huggingface/transformers 导入之前生效，
     保证全进程不可能触发模型下载；
  2. 提供 project_root()，供 config.py 定位工程根目录；
  3. 【保留自 07】Windows 大栈线程修复 run_with_large_stack() ——
     src/fulltext.py / src/retriever.py / src/answer_location.py / build_index.py
     等既有入口调用它，属复用模块依赖的公开接口，不可删除
     （删除后这些入口会 AttributeError；且主线程 1MB 栈实测会段错误）。

本模块复用自工单07（工单07 复用自 01~06），逻辑零改动；工单08 仅补一个
显式 ensure_offline() 入口（与模块导入即置的离线开关行为一致）。
"""
from __future__ import annotations

import os
import sys
import threading


def ensure_offline() -> None:
    """强制离线：HF/Transformers/ModelScope 一律不回源。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    # 防 tokenizers 并行导致的 fork 告警（Windows 上尤其吵）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def project_root() -> str:
    """工程根目录（本文件位于 <root>/src/bootstrap.py）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 模块导入即生效：任何 transformers / sentence_transformers 导入之前
ensure_offline()

# ---------------------------------------------------------------------------
# 以下为工单07 既有逻辑（原样保留）：BLAS 线程数限制 + 导入链预热 + 大栈线程
# ---------------------------------------------------------------------------
# 限制 BLAS/OpenMP 线程数：默认按 CPU 核数开线程，每个线程都要占内存与栈，
# 在内存吃紧的机器上会直接抛出 OpenBLAS 分配失败。本系统的主要算力在 GPU
# （bge-m3 编码、reranker 推理），CPU 侧 BLAS 线程数降下来对速度影响很小。
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4")
# 模型只从本地目录读取
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _preload_import_chain() -> None:
    """预热 sklearn -> pandas 导入链，规避 Windows 上的导入期栈溢出。

    现象（工单07 本机实测）：`from transformers import AutoTokenizer` 后调用
    `from_pretrained()` 触发 sklearn -> pandas 的深层懒加载，在 Windows 上耗尽
    C 栈，表现为 `Windows fatal exception: access violation`（exit code 139）。
    先 `import pandas` / `import sklearn` 再加载 tokenizer 则稳定通过。
    """
    for mod in ("pandas", "sklearn", "scipy",
                "sklearn.utils", "sklearn.metrics"):
        try:
            __import__(mod)
        except Exception:
            # 缺失也不致命：只是失去这层保护，继续运行
            pass


# 16 MB 足够（崩溃时主线程只有约 1 MB），同时避免把内存压垮：
# 线程栈是"预留+按需提交"，但 BLAS 会为每个核起线程，栈开太大
# （实测 64MB）会显著抬高提交内存，反而引出
# "OpenBLAS error: Memory allocation still failed after 10 retries"。
_STACK_SIZE = 16 * 1024 * 1024     # 16 MB


def _enlarge_default_stack() -> None:
    try:
        threading.stack_size(_STACK_SIZE)
    except (ValueError, RuntimeError):
        pass


# 注意：这里**只**设置环境变量与栈大小，不在这里做重量级导入。
# 本模块是被 import 的，而 import 发生在主线程（栈只有约 1 MB）——
# 把 pandas/sklearn/scipy 的预热放在这里，等于把"深链导入"搬回小栈上执行，
# 反而制造崩溃（工单07 实测）。预热改到 run_with_large_stack() 的大栈线程里做。
_enlarge_default_stack()


def run_with_large_stack(func, *args, **kwargs):
    """在 16MB 栈的新线程里执行 func，避免主线程栈溢出把进程打崩。

    用于各命令行入口（__main__）：主线程栈在进程创建时就固定了，改不了，
    只能把真正的活儿挪到子线程里跑。
    """
    box: dict = {}

    def _target() -> None:
        try:
            # 深链导入放在大栈线程里做，避免撑爆主线程的 1MB 栈
            _preload_import_chain()
            box["value"] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - 原样抛回主线程
            box["error"] = exc

    try:
        threading.stack_size(_STACK_SIZE)
    except (ValueError, RuntimeError):
        pass
    thread = threading.Thread(target=_target, name="main-large-stack")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")
