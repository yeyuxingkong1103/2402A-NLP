# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：运行时引导（bootstrap）

【必须在任何 transformers / sentence_transformers 导入之前 import 本模块】

本模块承担两件事：

1. 强制离线环境变量
   工单硬约束：禁止下载任何模型。这里在进程最早期就把 HF/ModelScope 切到离线，
   使得任何漏写 `local_files_only=True` 的调用路径也会直接失败而不是偷偷联网下载。

2. 规避 Windows 上的导入期栈溢出（本机实测崩溃，非猜测）
   现象：`from transformers import AutoTokenizer` 后调用 `from_pretrained()`
        触发 sklearn -> pandas 的深层懒加载，在 Windows 上耗尽 C 栈，
        表现为 `Windows fatal exception: access violation`（进程 exit code 139 / 段错误），
        faulthandler 栈顶停在 importlib._bootstrap_external._path_stat。
   验证：单独 `import pandas` 正常，`import numpy` 后仍崩溃，先 `import pandas`
        或 `import sklearn` 后再加载 tokenizer 则稳定通过。
   结论：这是导入顺序敏感的循环导入导致的栈溢出，不是模型文件损坏、也不是 CUDA 问题。
   对策：在此处提前把 pandas / sklearn 预热进 sys.modules，断开递归导入链。
"""

import os
import sys
import threading

# ---------------------------------------------------------------------------
# 1) 离线模式：任何下载尝试都直接报错，而不是偷偷联网
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
# 关闭 tokenizers 的并行告警噪声，避免多进程 fork 问题
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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

    注意 `__import__("sklearn")` 只加载顶层包，真正会触发深层链的是
    sklearn 的子模块（utils/metrics 等会在首次使用时拉 pandas、scipy）。
    因此这里连子模块一起预热。
    """
    for mod in ("pandas", "sklearn", "scipy",
                "sklearn.utils", "sklearn.metrics"):
        try:
            __import__(mod)
        except Exception:
            # 缺失也不致命：只是失去这层保护，继续运行
            pass


# ---------------------------------------------------------------------------
# 3. 大栈线程
# ---------------------------------------------------------------------------
# Windows 上 Python 主线程的栈默认只有约 1 MB。上面的导入链预热能规避大部分
# 情况，但**不是百分百**：实测同一条命令多次运行，仍会偶发
# "Windows fatal exception: stack overflow"（进程直接消失，Python 层
# 无法 try/except 捕获，faulthandler 连回溯都打印不出来）。
# 根因是 C 层的深层递归调用耗尽主线程栈，因此最可靠的对策是把工作放到
# **大栈线程**里执行。
#   · threading.stack_size() 影响此后创建的所有线程（含 Streamlit 的脚本执行线程）；
#   · 命令行入口则用 run_with_large_stack() 显式包一层。
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
# 反而制造崩溃（实测：加了 scipy/sklearn 子模块预热后，
# `from src import bootstrap` 这一步本身就会偶发段错误）。
# 预热改到 run_with_large_stack() 的**大栈线程里**做。
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


def project_root() -> str:
    """返回项目根目录（src 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
