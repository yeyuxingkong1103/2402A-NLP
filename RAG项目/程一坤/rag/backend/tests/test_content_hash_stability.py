# 导入 HTTP 响应模拟对象，用于构造测试用的响应
from io import BytesIO
# 导入时间模块，用于提供时钟函数
import time

# 导入被测试的请求器类
from app.crawler.requester import OfficialPageRequester
# 导入白名单验证器
from app.crawler.whitelist import OfficialSourceWhitelist
# 导入限速器
from app.crawler.rate_limiter import SourceRateLimiter


# 模拟 HTTP 响应头对象
class _FakeHeaders:
    def __init__(self, content_type: str = "text/html", charset: str = "utf-8") -> None:
        # 保存内容类型
        self._content_type = content_type
        # 保存字符集
        self._charset = charset

    # 模拟获取内容类型
    def get_content_type(self) -> str:
        return self._content_type

    # 模拟获取字符集
    def get_content_charset(self) -> str | None:
        return self._charset

    # 模拟获取 Content-Length 响应头
    def get(self, name: str) -> str | None:
        return None


# 模拟 HTTP 响应对象
class _FakeResponse:
    def __init__(self, content: bytes, content_type: str = "text/html") -> None:
        # 保存响应正文
        self._content = BytesIO(content)
        # 保存 HTTP 状态码
        self._code = 200
        # 构造响应头对象
        self.headers = _FakeHeaders(content_type=content_type, charset="utf-8")

    # 模拟读取响应正文
    def read(self, size: int = -1) -> bytes:
        return self._content.read(size)

    # 模拟获取 HTTP 状态码
    def getcode(self) -> int:
        return self._code

    # 模拟关闭响应
    def close(self) -> None:
        self._content.close()

    # 支持上下文管理器
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# 测试：同一页面不同时间戳的两次采集应产生相同哈希
def test_content_hash_ignores_dynamic_timestamps() -> None:
    # 构造包含动态时间戳的第一次采集结果（模拟第一次访问）
    html_with_timestamp_1 = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>劳动合同法</title>
        <!-- PublishDate=2026-09-17T10:00:00 -->
    </head>
    <body>
        <h1>中华人民共和国劳动合同法</h1>
        <p>第一条 为了完善劳动合同制度...</p>
    </body>
    </html>
    """.encode("utf-8")

    # 构造包含不同时间戳的第二次采集结果（模拟第二次访问）
    html_with_timestamp_2 = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>劳动合同法</title>
        <!-- PublishDate=2026-09-17T11:30:00 -->
    </head>
    <body>
        <h1>中华人民共和国劳动合同法</h1>
        <p>第一条 为了完善劳动合同制度...</p>
    </body>
    </html>
    """.encode("utf-8")

    # 构造测试用的白名单（允许 example.com 域名）
    whitelist = OfficialSourceWhitelist(
        allowed_sources={
            "test-source": {
                "domains": {"example.com"},
                "path_prefixes": {"/"},
            }
        }
    )

    # 构造测试用的限速器（不限速）
    rate_limiter = SourceRateLimiter(
        minimum_interval_seconds=0.0,
        clock=time.time,
    )

    # 记录每次调用时应返回的响应
    responses = [
        _FakeResponse(html_with_timestamp_1),
        _FakeResponse(html_with_timestamp_2),
    ]
    response_index = [0]

    # 模拟 URL 打开函数（返回预设的响应）
    def fake_open_url(request, timeout):
        # 如果是请求 robots.txt，返回允许所有访问的响应
        if "robots.txt" in request.full_url:
            return _FakeResponse(b"User-agent: *\nAllow: /\n")
        # 否则返回预设的页面响应
        idx = response_index[0]
        response_index[0] += 1
        return responses[idx]

    # 创建请求器实例（注入模拟的打开函数）
    requester = OfficialPageRequester(
        whitelist=whitelist,
        rate_limiter=rate_limiter,
        user_agent="test-agent",
        open_url=fake_open_url,
        sleep=lambda _: None,
    )

    # 第一次采集
    result_1 = requester.fetch(
        crawl_id="crawl-1",
        source_id="test-source",
        source_url="https://example.com/law/labor-contract",
    )

    # 第二次采集（仅时间戳不同）
    result_2 = requester.fetch(
        crawl_id="crawl-2",
        source_id="test-source",
        source_url="https://example.com/law/labor-contract",
    )

    # 验证两次采集都成功
    assert result_1.record.crawl_status == "success"
    assert result_2.record.crawl_status == "success"

    # 验证哈希值相同（忽略了时间戳差异）
    assert result_1.record.content_hash == result_2.record.content_hash

    # 验证哈希值不为空
    assert result_1.record.content_hash is not None
    assert len(result_1.record.content_hash) == 64  # SHA-256 十六进制长度


# 测试：正文内容变化时哈希应该改变
def test_content_hash_changes_when_content_changes() -> None:
    # 构造第一个版本的页面
    html_version_1 = """
    <html>
    <body>
        <h1>劳动合同法</h1>
        <p>第一条 为了完善劳动合同制度...</p>
    </body>
    </html>
    """.encode("utf-8")

    # 构造第二个版本的页面（正文内容已修改）
    html_version_2 = """
    <html>
    <body>
        <h1>劳动合同法</h1>
        <p>第一条 为了保护劳动者的合法权益...</p>
    </body>
    </html>
    """.encode("utf-8")

    # 构造测试环境
    whitelist = OfficialSourceWhitelist(
        allowed_sources={
            "test-source": {
                "domains": {"example.com"},
                "path_prefixes": {"/"},
            }
        }
    )
    rate_limiter = SourceRateLimiter(
        minimum_interval_seconds=0.0,
        clock=time.time,
    )

    # 准备两次不同的响应
    responses = [
        _FakeResponse(html_version_1),
        _FakeResponse(html_version_2),
    ]
    response_index = [0]

    def fake_open_url(request, timeout):
        # 如果是请求 robots.txt，返回允许所有访问的响应
        if "robots.txt" in request.full_url:
            return _FakeResponse(b"User-agent: *\nAllow: /\n")
        # 否则返回预设的页面响应
        idx = response_index[0]
        response_index[0] += 1
        return responses[idx]

    # 创建请求器
    requester = OfficialPageRequester(
        whitelist=whitelist,
        rate_limiter=rate_limiter,
        user_agent="test-agent",
        open_url=fake_open_url,
        sleep=lambda _: None,
    )

    # 第一次采集
    result_1 = requester.fetch(
        crawl_id="crawl-1",
        source_id="test-source",
        source_url="https://example.com/law/labor-contract",
    )

    # 第二次采集（正文已变化）
    result_2 = requester.fetch(
        crawl_id="crawl-2",
        source_id="test-source",
        source_url="https://example.com/law/labor-contract",
    )

    # 验证两次采集都成功
    assert result_1.record.crawl_status == "success"
    assert result_2.record.crawl_status == "success"

    # 验证哈希值不同（正文内容已改变）
    assert result_1.record.content_hash != result_2.record.content_hash
