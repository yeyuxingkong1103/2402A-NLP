"""S4 分块的实现模块。

对外唯一入口是 backend/chunk_clean.py；本包不被单独执行。
模块划分：core 常量 / embedder 向量 / loader 输入 / decide 判定 / output 产出
"""

RULE_VERSION = "chunk-v1+bge-m3"
