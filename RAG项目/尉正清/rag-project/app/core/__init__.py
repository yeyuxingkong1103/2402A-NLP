# app/core/__init__.py
"""业务核心层。

这里刻意不做 eager import —— 各服务内部都是懒加载单例，
避免 import 包时就触发模型加载或数据库连接。
"""
