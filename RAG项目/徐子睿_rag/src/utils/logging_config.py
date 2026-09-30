"""src/utils/logging_config.py —— 日志配置与 trace_id 注入。

在链路中的位置：
    src/api/main.py 的启动钩子调用 configure_logging(log_dir) 完成初始化
    src/api/main.py 的 trace_middleware 写入 trace_id_var，本文件的过滤器把它注入每条日志

解决的问题只有一个：**让一次请求的所有日志能被串起来**。
    一个请求会经过中间件、路由、链路编排、多个存储模块，各自打日志。
    若不带上统一标识，出了问题时要在交织的多请求日志里靠时间戳猜测归属，
    并发一高就完全不可用。

方案：ContextVar + logging.Filter
    ContextVar 存当前请求的 trace_id（天然按请求上下文隔离，不受并发影响）
    Filter 在每条日志落盘前把 trace_id 读出来塞进 record
    格式串里用 %(trace_id)s 输出
"""
from __future__ import annotations

import contextvars
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# ContextVar 是"上下文局部变量"：每个请求处理上下文有自己独立的值，
# 多线程/多协程并发时不会互相覆盖（这正是全局变量做不到的）。
# 默认值 "-" 表示"这条日志不属于任何请求"，如应用启动阶段的日志。
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class TraceIdFilter(logging.Filter):
    """把当前上下文的 trace_id 写进日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """在日志记录上附加 trace_id 属性。

        参数：
            record: 待输出的日志记录
        返回：
            恒为 True —— 这个过滤器只做"加工"，从不拦截日志。
            返回 False 会丢弃该条日志，对本用途毫无意义，反而容易造成日志静默丢失。

        为什么要做成 Filter 而不是每次调 logger 时手工传：
            前者是"自动的"，后者要求每个打日志的地方都记得带上 trace_id ——
            只要有一处忘了，那处日志就无法关联，而这种遗漏很难被发现。
        """
        record.trace_id = trace_id_var.get()
        return True


def configure_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """配置根日志器：同时输出到按天滚动的文件和控制台。

    参数：
        log_dir: 日志目录（不存在会自动创建）
        level: 日志级别，默认 INFO

    两个 handler 的分工：
        TimedRotatingFileHandler  写文件，when="midnight" 每天零点切分，
                                  backupCount=14 保留 14 天
                                  —— 限制保留天数是必要的，否则日志会无限增长最终占满磁盘
        StreamHandler             输出到控制台，便于本地开发和容器看日志
                                  （容器场景通常靠收集 stdout，只写文件会导致 docker logs 一片空白）

    encoding="utf-8" 是必需的：
        日志里有中文（角色名、错误信息）。不指定编码时，
        Windows 上会用系统默认编码（GBK）写文件，遇到 UTF-8 中文会抛异常或写成乱码。

    `if not root.handlers` 这层判断防的是"重复添加 handler"：
        uvicorn 的 --reload 模式或测试中多次导入模块时，
        configure_logging 可能被调用多次。
        不加判断的话，handler 会越叠越多，同一条日志被打印 N 遍 ——
        这是日志里最常见的"为什么每行都出现两次"的原因。

    注意 formatter 与 filter 各建了两份（文件和 console 各一套）：
        因为 filter 是挂在 handler 上的、不是挂在 formatter 上的，
        两个 handler 必须各持一个 Filter 实例，否则其中一个 handler 拿不到 trace_id。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [trace=%(trace_id)s] %(name)s %(message)s")
    handler = TimedRotatingFileHandler(log_dir / "app.log", when="midnight", backupCount=14, encoding="utf-8")
    handler.setFormatter(formatter)
    handler.addFilter(TraceIdFilter())
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(TraceIdFilter())
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
        root.addHandler(console)


def get_logger(name: str) -> logging.Logger:
    """按模块名取日志器。

    参数：
        name: 日志器名（惯例传 __name__，日志里就会显示模块路径，便于定位来源）
    返回：
        Logger 实例。

    对 logging.getLogger 的薄封装：
        作用是留一个统一的获取入口 —— 将来若要给全项目的日志器统一加
        处理器或过滤器，改这一处即可，不必改所有调用点。
    """
    return logging.getLogger(name)
