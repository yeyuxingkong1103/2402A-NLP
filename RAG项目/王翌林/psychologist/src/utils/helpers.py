"""工具函数：文本处理、时间、ID。

这里放的是一批与业务无关、可被各处复用的纯函数（无副作用、易测试）：
时间格式化、字符串截断/规范化、md5、文件名净化、内存分页。
"""
import datetime as dt
# hashlib：提供 md5 等哈希算法；re：正则表达式，用于文本清洗与文件名过滤
import hashlib
import re
# Any/Dict/List/Optional：为入参与返回值提供类型标注，便于阅读与静态检查
from typing import Any, Dict, List, Optional


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回当前本地时间的字符串，默认格式 'YYYY-MM-DD HH:MM:SS'。

    常用于日志、文件名、接口返回中的时间字段。
    """
    return dt.datetime.now().strftime(fmt)


def to_str(value: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
    """把任意值安全地转成字符串。

    - None 直接返回 None（保持“空”的语义，前端可据此判断）；
    - 日期/时间按指定格式格式化；
    - 其它类型回退到 str()。
    这样调用方无需关心字段到底是时间还是普通值。
    """
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date)):
        return value.strftime(fmt)
    return str(value)


def truncate(text: str, max_len: int = 100, suffix: str = "...") -> str:
    """按最大长度截断文本，超长时追加省略号。

    用于列表页摘要展示，避免超长文本撑破界面；空值统一返回空串。
    """
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len] + suffix


def md5(text: str) -> str:
    """计算文本的 MD5 十六进制摘要。

    常用于生成内容指纹/缓存键或做去重标识。注意 MD5 不安全，不可用于密码存储。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    """把连续任意空白（空格/换行/制表符）压成一个空格，并去掉首尾空白。

    文档解析或用户输入常带杂乱空白，统一规范化后便于检索与比较。
    """
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_filename(name: str) -> str:
    """把原始文件名净化为安全文件名。

    仅保留字母数字、下划线、中文、点、连字符，其余替换为下划线，
    再限制到 180 字符——防止路径穿越（如 ../）与非法字符导致的存储/下载问题。
    """
    name = re.sub(r"[^\w\u4e00-\u9fff\.\-]+", "_", name or "file")
    return name[:180]


def paginate(items: List[Any], page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """对内存中的列表做分页，返回统一的分页结构。

    注意：这是“内存分页”，适合数据已全部加载的小规模场景；
    大数据量应改用数据库层面的 LIMIT/OFFSET。
    参数：page 从 1 开始；start 用 max(0, ...) 兜底，防止负数下标取错数据。
    """
    total = len(items)
    start = max(0, (page - 1) * page_size)
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": items[start:start + page_size],
    }