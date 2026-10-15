# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：提供控制台日志与「带 request_id 的结构化阶段日志」，用于定位各阶段耗时

import json                                                     # 导入 json 用于序列化结构化日志
import time                                                     # 导入 time 用于计时
import uuid                                                     # 导入 uuid 用于生成请求标识
import logging                                                  # 导入 logging 用于控制台输出
import config                                                   # 导入本工单配置模块


def get_logger(name="perf"):                                    # 定义控制台日志获取函数
    """返回配置好的控制台 logger，格式为「时间 [级别] 消息」，重复调用返回同一实例。"""
    logger = logging.getLogger(name)                            # 按名称取 logger（同名复用）
    if not logger.handlers:                                     # 尚未配置过处理器时才配置
        handler = logging.StreamHandler()                       # 输出到标准错误流
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",  # 时间与级别
                                               datefmt="%H:%M:%S"))  # 只保留时分秒
        logger.addHandler(handler)                              # 挂载处理器
        logger.setLevel(logging.INFO)                           # 日志级别设为 INFO
        logger.propagate = False                                # 避免向根 logger 重复输出
    return logger                                               # 返回 logger


LOGGER = get_logger()                                           # 模块级共享 logger


class StageTracer(object):                                      # 定义阶段追踪器类
    """记录一次请求在五个处理阶段上的耗时，并落盘为带 request_id 的 JSON Lines。"""

    def __init__(self, request_id=None, query="", log_file=None, persist=True):  # 构造方法
        self.request_id = request_id or uuid.uuid4().hex[:12]   # 请求标识（未指定则随机生成）
        self.query = query                                      # 本次请求的查询原文
        self.log_file = log_file or config.LOG_FILE             # 结构化日志文件路径
        self.persist = persist                                  # 是否落盘（基准测试时关闭可减少 I/O 干扰）
        self.records = []                                       # 已记录的阶段明细
        self._stack = []                                        # 当前打开的阶段栈（支持嵌套）
        self._t0 = time.perf_counter()                          # 请求整体起始时刻（高精度计时）

    def begin(self, stage):                                     # 定义阶段开始函数
        """标记某阶段开始，压入计时栈。"""
        self._stack.append([stage, time.perf_counter()])        # 记录阶段名与开始时刻
        return self                                             # 支持链式调用

    def end(self, stage=None, **metrics):                       # 定义阶段结束函数
        """标记某阶段结束，计算耗时并记录；stage 省略时弹出栈顶阶段。"""
        if not self._stack:                                     # 栈为空说明没有进行中的阶段
            return 0.0                                          # 返回 0 表示无法计时
        name, start = self._stack.pop()                         # 弹出栈顶阶段
        if stage is not None and stage != name:                 # 传入阶段名与栈顶不一致时以栈顶为准
            LOGGER.warning("阶段名不匹配：栈顶 %s，传入 %s", name, stage)  # 输出告警
        seconds = time.perf_counter() - start                   # 计算阶段耗时
        return self.record(name, seconds, **metrics)            # 记录该阶段

    def record(self, stage, seconds, **metrics):                # 定义阶段记录函数
        """写入一条阶段记录，并同步落盘结构化日志。"""
        rec = {"request_id": self.request_id,                   # 请求标识，用于串联同一请求的各阶段
               "stage": stage,                                  # 阶段名称
               "seconds": round(seconds, 6),                    # 阶段耗时（秒）
               "query": self.query[:60],                        # 查询前缀（截断便于阅读）
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}  # 完成时刻
        rec.update({k: v for k, v in metrics.items() if v is not None})  # 合并业务指标
        self.records.append(rec)                                # 存入内存明细
        if self.persist:                                        # 需要落盘时写入 JSON Lines
            config.ensure_dirs()                                # 确保目录存在
            with open(self.log_file, "a", encoding="utf-8") as f:  # 追加方式打开日志
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # 一行一条 JSON
        return seconds                                          # 返回阶段耗时

    def total(self):                                            # 定义请求总耗时函数
        """返回从创建追踪器到当前时刻的总耗时（秒）。"""
        return time.perf_counter() - self._t0                   # 高精度差值

    def stage_seconds(self, stage):                             # 定义按阶段取耗时函数
        """返回指定阶段的累计耗时（同名阶段多次记录时求和）。"""
        return round(sum(r["seconds"] for r in self.records if r["stage"] == stage), 6)  # 求和

    def retrieval_seconds(self):                                # 定义检索阶段耗时函数
        """返回「检索结果返回」口径的耗时：查询处理与增强 + 检索 + 上下文组装。"""
        return round(sum(r["seconds"] for r in self.records      # 累加属于检索口径的各阶段
                         if r["stage"] in config.RETRIEVAL_STAGES), 6)  # 使用配置中的阶段清单

    def as_dict(self):                                          # 定义明细导出函数
        """把阶段明细导出为「阶段名 -> 耗时」的字典，便于写入报告。"""
        out = {}                                                # 初始化结果字典
        for r in self.records:                                  # 遍历阶段记录
            out[r["stage"]] = round(out.get(r["stage"], 0.0) + r["seconds"], 6)  # 同名阶段累加
        return out                                              # 返回阶段耗时字典

    def summary(self):                                          # 定义单请求摘要函数
        """返回本次请求的汇总：总耗时、检索口径耗时、阶段明细与是否达标。"""
        ret = self.retrieval_seconds()                          # 检索口径耗时
        return {"request_id": self.request_id,                  # 请求标识
                "total_seconds": round(self.total(), 6),        # 端到端总耗时
                "retrieval_seconds": ret,                       # 检索结果返回耗时
                "within_budget": ret <= config.RETRIEVAL_BUDGET,  # 是否满足 3 秒验收
                "stages": self.as_dict()}                       # 各阶段耗时
