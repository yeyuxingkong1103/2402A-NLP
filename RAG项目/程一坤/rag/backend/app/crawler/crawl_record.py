# 导入数据类装饰器，用于定义采集记录对象
from dataclasses import dataclass

# 导入日期时间类型，用于保存采集完成时间
from datetime import datetime

# 导入 SHA-256 哈希算法，用于识别文档内容版本
from hashlib import sha256


# 计算文本内容的稳定 SHA-256 哈希值
def calculate_content_hash(content: str) -> str:
    # 将文本统一编码为 UTF-8 字节后计算 SHA-256
    content_bytes = content.encode("utf-8")

    # 返回便于保存和比较的十六进制哈希字符串
    return sha256(content_bytes).hexdigest()


# 保存一次官方来源采集结果及其状态
@dataclass(frozen=True)
class CrawlRecord:
    # 保存本次采集任务的唯一标识
    crawl_id: str

    # 保存官方来源配置的唯一标识
    source_id: str

    # 保存实际访问的官方页面地址
    source_url: str

    # 保存采集到的文档标题，失败时可以为空
    document_title: str | None

    # 保存文档类型，失败时可以为空
    document_type: str | None

    # 保存文档内容哈希，失败时可以为空
    content_hash: str | None

    # 保存采集内容格式，例如 HTML 或 PDF
    content_format: str | None

    # 保存采集完成时间
    collected_at: datetime

    # 保存官方服务器返回的 HTTP 状态码
    http_status: int | None

    # 保存采集时是否通过 robots.txt 规则
    robots_allowed: bool

    # 保存本次采集状态，例如 success 或 failed
    crawl_status: str

    # 保存文档版本状态，例如 new、unchanged 或 updated
    version_status: str | None

    # 保存失败原因，成功时为空
    error_message: str | None
