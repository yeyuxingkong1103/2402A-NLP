"""HTTP 响应读取与大小限制。

原 requester.py 并入部分：限制大小地读取响应正文、
按响应头字符集解码，防止超大响应导致内存溢出。
"""

# 定义单次响应允许的最大字节数（50 MB）
MAX_RESPONSE_BYTES = 50 * 1024 * 1024

# 定义分块读取时每次读取的字节数（64 KB）
READ_CHUNK_BYTES = 64 * 1024


# 响应正文超过大小限制时抛出此异常
class ResponseTooLarge(ValueError):
    """响应正文超过允许的最大大小。"""


# 限制大小地读取响应正文，防止内存溢出
def read_limited(response: object) -> bytes:
    # 尝试从响应头获取声明的内容长度
    declared_length = response.headers.get("Content-Length")
    if declared_length is not None:
        try:
            # 解析声明的字节数
            declared_bytes = int(declared_length)
        except ValueError:
            # 解析失败时设为 0
            declared_bytes = 0
        # 如果声明长度超过限制，直接拒绝
        if declared_bytes > MAX_RESPONSE_BYTES:
            raise ResponseTooLarge("官方页面响应超过 50 MB 上限")

    # 用于保存读取的所有内容块
    content_parts: list[bytes] = []
    # 记录已读取的总字节数
    total_bytes = 0
    # 分块读取响应正文
    while True:
        # 读取一个数据块
        chunk = response.read(READ_CHUNK_BYTES)
        # 如果读取到空数据，说明已到达末尾
        if not chunk:
            break
        # 累加已读取字节数
        total_bytes += len(chunk)
        # 如果总字节数超过限制，停止读取并抛出异常
        if total_bytes > MAX_RESPONSE_BYTES:
            raise ResponseTooLarge("官方页面响应超过 50 MB 上限")
        # 保存当前数据块
        content_parts.append(chunk)
    # 拼接所有数据块并返回
    return b"".join(content_parts)
