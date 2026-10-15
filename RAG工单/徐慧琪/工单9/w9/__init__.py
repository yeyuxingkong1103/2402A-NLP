# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

工单09 自有包 `w9`（Graph RAG 优化实现）。

为什么包名是 `w9` 而不是 `src`：
    工单09 以「只读引用」方式复用工单08 的 `src` 包（见 `w9/compat.py`）。
    08 的内部模块使用绝对导入 `from src import config`，且 `bootstrap.project_root()`
    以**文件位置**解析工程根目录 —— 因此只要把 `../工单8` 挂到 `sys.path`，
    `import src` 就精确指向 08 的代码，其 ROOT 也精确解析为 工单8 目录。
    若 09 自己的包也叫 `src`，两个包会争用同名导入，必然互相污染。
    故 09 的包命名为 `w9`（workorder 9），与 08 彻底隔离、互不干扰。
"""

__version__ = "1.0.0"
