# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：请求级分阶段计时与结构化日志，用于响应时间验收（≤3 秒）与瓶颈定位

import time                                                     # 导入时间模块用于高精度计时
import json                                                     # 导入 json 用于结构化日志序列化
import logging                                                  # 导入标准日志库
from datetime import datetime                                   # 导入日期时间用于日志时间戳
import config                                                   # 导入配置模块（同时触发标准输出 UTF-8 修正）

# 配置全局日志格式：时间 + 级别 + 消息，级别 INFO，输出到控制台
logging.basicConfig(level=logging.INFO,                         # 设置日志最低级别为 INFO
                    format="%(asctime)s [%(levelname)s] %(message)s")  # 设置日志输出格式
LOGGER = logging.getLogger("graph_rag")                         # 创建本模块专用 logger，避免污染根日志配置


class StageTimer(object):                                       # 定义分阶段计时器类
    """按阶段累计耗时，每个请求实例化一个，结束时输出结构化耗时明细。"""

    def __init__(self, request_id):                             # 构造方法：接收请求 ID 用于日志追踪
        self.request_id = request_id                            # 保存请求 ID，便于串联同一请求的所有日志
        self._t0 = time.perf_counter()                          # 记录整体起始时刻（高精度单调时钟）
        self._stages = []                                       # 初始化阶段耗时列表：[(阶段名, 秒), ...]
        self._marks = {}                                        # 记录各阶段起始时刻，防止阶段嵌套错乱

    def start(self, stage):                                     # 开始某个阶段的计时
        self._marks[stage] = time.perf_counter()                # 记录该阶段起始时刻
        return self                                             # 返回自身以支持链式调用

    def stop(self, stage):                                      # 结束某个阶段的计时
        if stage not in self._marks:                            # 容错：若未 start 过则直接返回 0
            return 0.0                                          # 返回 0 表示该阶段无有效计时
        cost = time.perf_counter() - self._marks.pop(stage)     # 计算阶段耗时并移除起始标记
        self._stages.append((stage, cost))                      # 追加到阶段耗时列表
        return cost                                             # 返回本次阶段耗时（秒）

    def total(self):                                            # 计算从开始到现在的总耗时
        return time.perf_counter() - self._t0                   # 返回整体耗时（秒）

    def summary(self):                                          # 生成耗时汇总字典
        gap = self.total() - sum(c for _, c in self._stages)    # 计算未归入任何阶段的间隙开销
        return {                                                # 返回结构化结果
            "request_id": self.request_id,                      # 请求 ID
            "stages": [(n, round(c, 4)) for n, c in self._stages],  # 各阶段耗时（保留 4 位小数）
            "unaccounted": round(gap, 4),                       # 未被阶段覆盖的间隙耗时
            "total": round(self.total(), 4),                    # 总耗时
        }

    def log(self):                                              # 输出结构化日志
        s = self.summary()                                      # 取汇总结果
        detail = ", ".join("%s=%.3fs" % (n, c) for n, c in s["stages"])  # 拼接阶段明细为可读字符串
        LOGGER.info("[%s] %s | unaccounted=%.3fs | total=%.3fs",       # 记录日志（含请求 ID 便于追踪）
                    s["request_id"], detail, s["unaccounted"], s["total"])  # 填入具体数值
        return s                                                # 返回汇总结果供调用方使用


def log_json(obj, prefix="stage"):                              # 定义结构化 JSON 日志辅助函数
    LOGGER.info("%s %s", prefix, json.dumps(obj, ensure_ascii=False))  # 以 JSON 行形式输出，便于后续解析


def now_str():                                                  # 定义获取当前时间字符串的辅助函数
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")         # 返回格式化的本地时间字符串
