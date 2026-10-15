# -*- coding: utf-8 -*-
# singleton_model.py 工单17 重量级模型/解析器单例模式
import threading

class SingletonMeta(type):
    """线程安全单例元类，用于模型、DeepDoc解析器"""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


# ----------------示例用法----------------
# class DeepDocParser(metaclass=SingletonMeta):
#     def __init__(self):
#         print("仅进程启动执行一次初始化")
#
# p1 = DeepDocParser()
# p2 = DeepDocParser()
# assert p1 is p2
