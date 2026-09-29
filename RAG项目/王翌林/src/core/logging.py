"""统一日志：控制台 + 文件（app/error/access），支持 DEBUG/INFO/WARNING/ERROR。

阶段 4：全部日志行携带 request_id（无请求上下文时为 "-"），由 main.py 的
access_log_middleware 通过 set_request_id 注入，便于单条请求全链路检索。
"""
import contextvars
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from src.core.config import settings

# ContextVar 而非全局变量：异步/多线程下每个请求有独立的 request_id，
# 全局变量会被并发请求互相覆盖，ContextVar 天然隔离各协程上下文。
_request_id_var: contextvars.ContextVar = contextvars.ContextVar("request_id", default="-")

# 日志格式固定带 rid 字段；无请求上下文时填 "-"，保证格式永远可被解析。
_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] [rid=%(request_id)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# 幂等开关：setup_logging 可能在多处被重复调用，用它避免重复添加 handler 导致日志重复输出。
_configured = False


def set_request_id(rid: str) -> None:
    """由中间件在每个请求进入时调用，写入当前上下文的 request_id。"""
    # rid 为空时统一存 "-"，防止 None 直接进日志格式串。
    _request_id_var.set(rid or "-")


def get_request_id() -> str:
    """读取当前上下文的 request_id，读取不到时返回默认值 "-"。"""
    return _request_id_var.get()


def _record_factory(*args, **kwargs):
    """LogRecord 工厂：给每条日志记录动态注入 request_id 属性。

    为什么要工厂？因为 request_id 是"每条日志各不相同"的运行期数据，
    只能在这里按当前上下文取值，无法写死在 Formatter 里。
    """
    # 先构造标准 record，再补充自定义字段，最后返回。
    record = _base_factory(*args, **kwargs)
    record.request_id = _request_id_var.get()
    return record


def _level() -> int:
    """把配置里的字符串级别（如 "INFO"）转成 logging 的数字常量。

    用 getattr 兜底：配置写错（如 "infoo"）时回退 INFO，
    而不是让服务因为一个错别字直接启动失败。
    """
    return getattr(logging, settings.log_level.upper(), logging.INFO)


def setup_logging() -> None:
    """一次性初始化日志系统（控制台 + 文件 + 专用 logger）。"""
    # 声明 _base_factory 为全局：下面要保存原始工厂，供 _record_factory 调用链回溯。
    global _configured, _base_factory
    # 幂等保护：重复调用会不断往 root 加 handler，导致同一条日志被打印多次。
    if _configured:
        return

    # 日志目录可能尚未创建（首次启动），先落盘目录再挂文件 handler。
    settings.ensure_dirs()
    # 保存"原生"工厂，再替换成我们的包装版；否则包装里再调用 _record_factory 会无限递归。
    _base_factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(_record_factory)
    root = logging.getLogger()
    root.setLevel(_level())
    # 清空既有 handler：uvicorn 等框架可能已装过 handler，不清会重复输出。
    # 用 list(...) 复制一份再遍历，避免在迭代中修改同一列表（常见踩坑）。
    for h in list(root.handlers):
        root.removeHandler(h)

    # 所有 handler 共用同一 formatter，保证控制台与文件格式一致，便于比对。
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台输出到 stdout（不是 stderr），便于容器日志采集按流分离。
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # app.log：全量日志，按 20MB 切割，保留 5 个历史文件（防止磁盘被写满）。
    app_handler = RotatingFileHandler(
        os.path.join(settings.log_dir, "app.log"),
        maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    app_handler.setFormatter(formatter)
    root.addHandler(app_handler)

    # error.log：仅 ERROR 及以上，方便运维直查故障而不被 INFO 淹没。
    error_handler = RotatingFileHandler(
        os.path.join(settings.log_dir, "error.log"),
        maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root.addHandler(error_handler)

    # 大模型 / RAG 检索专用日志
    # 给具名 logger（llm/rag）单独挂文件：便于把高频、体积大的模型交互日志与主日志分离。
    for name, filename in (("llm", "llm.log"), ("rag", "rag.log")):
        logger = logging.getLogger(name)
        # 二次幂等判断：防止 setup_logging 被绕过或该 logger 已被外部配置过时重复挂 handler。
        if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
            handler = RotatingFileHandler(
                os.path.join(settings.log_dir, filename),
                maxBytes=20 * 1024 * 1024, backupCount=3, encoding="utf-8",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    # 压噪：uvicorn 访问日志平时只留 WARNING，debug 模式才放开；
    # httpx 每次调用 LLM 都会记一条 INFO，必须压到 WARNING，否则日志会被刷爆。
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING if not settings.debug else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # 最后才置位，确保初始化中途异常时下次仍可重新配置。
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """业务代码统一入口：取 logger 前先保证日志系统已初始化。

    这样任何模块直接 get_logger(__name__) 即可，无需关心初始化顺序。
    """
    setup_logging()
    return logging.getLogger(name)


# 模块导入即初始化：保证进程启动早期（如配置校验失败）就有日志可打。
setup_logging()