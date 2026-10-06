# 导入可调用类型注解，用于依赖注入
from collections.abc import Callable

# 导入数据类装饰器，用于定义采集结果
from dataclasses import dataclass

# 导入日期时间类型，用于记录采集时间戳
from datetime import datetime, timezone

# 导入时间模块，用于限速等待和重试延迟
import time

# 导入 URL 解析工具，用于构造 robots.txt 缓存键
from urllib.parse import urlunparse

# 导入 HTTP 错误类型，用于捕获和处理网络异常
from urllib.error import HTTPError, URLError

# 导入请求构造和发送函数
from urllib.request import Request, urlopen

# 导入采集记录模型和哈希计算函数
from app.crawler.crawl_record import CrawlRecord, calculate_content_hash

# 导入 HTML 正文提取与清洗逻辑（原 requester 内置，已拆分独立）
from app.crawler.html_text import _extract_cleaned_text

# 导入限速器，用于遵守官方站点的访问频率约束
from app.crawler.rate_limiter import SourceRateLimiter

# 导入响应大小限制异常与限长读取逻辑（原 requester 内置，已拆分独立）
from app.crawler.response_reader import ResponseTooLarge, read_limited

# 导入 robots.txt 策略解析和校验逻辑
from app.crawler.robots_policy import RobotsAccessDenied, RobotsAccessPolicy

# 导入官方来源白名单验证器
from app.crawler.whitelist import OfficialSourceWhitelist, UrlNotAllowed


# 定义遇到临时错误时的最大重试次数
MAX_ATTEMPTS = 3


# 保存单次采集请求的结果和元数据
@dataclass(frozen=True)
class CrawlResult:
    # 保存采集记录（状态、哈希、错误信息等）
    record: CrawlRecord
    # 保存采集到的完整 HTML 内容，失败时为 None
    content: str | None


# 负责从官方来源安全采集页面内容的请求器
class OfficialPageRequester:
    # 限长读取逻辑已拆分至 response_reader 模块，此处绑定以维持类内调用路径
    _read_limited = staticmethod(read_limited)

    # 初始化请求器及其依赖组件
    def __init__(
        self,
        whitelist: OfficialSourceWhitelist,
        rate_limiter: SourceRateLimiter,
        user_agent: str,
        timeout_seconds: float = 30.0,
        open_url: Callable[..., object] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # 保存官方来源白名单验证器
        self._whitelist = whitelist
        # 保存访问频率限速器
        self._rate_limiter = rate_limiter
        # 保存用户代理字符串
        self._user_agent = user_agent
        # 超时秒数必须为正数
        if timeout_seconds <= 0:
            raise ValueError("请求超时必须大于零")
        # 保存单次请求的超时时间
        self._timeout_seconds = timeout_seconds
        # 保存 URL 打开函数（用于测试注入）
        self._open_url = open_url
        # 保存睡眠函数（用于测试注入）
        self._sleep = sleep
        # 缓存已解析的 robots.txt 策略，避免重复请求
        self._robots_policies: dict[str, RobotsAccessPolicy] = {}

    # 从指定 URL 采集页面内容并返回采集结果
    def fetch(self, crawl_id: str, source_id: str, source_url: str) -> CrawlResult:
        # 记录采集开始时间（UTC 时区）
        collected_at = datetime.now(timezone.utc)
        try:
            # 验证 URL 是否在官方来源白名单内
            parsed_url = self._whitelist.validate(source_url)
            # 获取该域名的 robots.txt 策略
            robots_policy = self._get_robots_policy(
                parsed_url.scheme, parsed_url.netloc, source_id
            )
            # 校验当前 User-Agent 是否被允许访问该 URL
            robots_policy.require_allowed(self._user_agent, source_url)

            # 构造 HTTP 请求对象，设置 User-Agent 请求头
            page_request = Request(
                source_url, headers={"User-Agent": self._user_agent}
            )
            # 带重试和限速地打开 URL 并读取响应
            with self._open_with_retry(page_request, source_id) as response:
                # 解码响应正文为字符串
                content = self._decode_response(response)
                # 获取 HTTP 状态码
                http_status = response.getcode()
                # 获取响应内容类型
                content_format = response.headers.get_content_type()

            # 提取清洗后的正文文本（移除动态时间戳等噪声）
            cleaned_text = _extract_cleaned_text(content)
            # 对清洗后的正文计算哈希，确保哈希稳定性
            content_hash = calculate_content_hash(cleaned_text)
            # 返回成功的采集结果
            return CrawlResult(
                record=CrawlRecord(
                    crawl_id=crawl_id,
                    source_id=source_id,
                    source_url=source_url,
                    document_title=None,
                    document_type=None,
                    content_hash=content_hash,
                    content_format=content_format,
                    collected_at=collected_at,
                    http_status=http_status,
                    robots_allowed=True,
                    crawl_status="success",
                    version_status="new",
                    error_message=None,
                ),
                content=content,
            )
        # 捕获白名单验证失败异常
        except UrlNotAllowed as error:
            return self._failure_result(
                crawl_id, source_id, source_url, collected_at, False, str(error)
            )
        # 捕获 robots.txt 禁止访问异常
        except RobotsAccessDenied as error:
            return self._failure_result(
                crawl_id, source_id, source_url, collected_at, False, str(error)
            )
        # 捕获 HTTP 错误响应
        except HTTPError as error:
            return self._failure_result(
                crawl_id,
                source_id,
                source_url,
                collected_at,
                True,
                f"官方页面返回 HTTP 错误：{error.code}",
                error.code,
            )
        # 捕获响应正文超过大小限制异常
        except ResponseTooLarge as error:
            return self._failure_result(
                crawl_id, source_id, source_url, collected_at, True, str(error)
            )
        # 捕获网络连接和超时异常
        except (URLError, TimeoutError, OSError) as error:
            return self._failure_result(
                crawl_id,
                source_id,
                source_url,
                collected_at,
                True,
                f"官方页面请求失败：{error}",
            )

    # 获取指定域名的 robots.txt 访问策略（带缓存）
    def _get_robots_policy(
        self, scheme: str, netloc: str, source_id: str
    ) -> RobotsAccessPolicy:
        # 构造缓存键（scheme + netloc，统一为小写）
        cache_key = urlunparse((scheme.lower(), netloc.lower(), "", "", "", ""))
        # 如果缓存中已有该域名的策略，直接返回
        if cache_key in self._robots_policies:
            return self._robots_policies[cache_key]

        # 构造 robots.txt 的完整 URL
        robots_url = f"{scheme}://{netloc}/robots.txt"
        # 构造请求对象
        robots_request = Request(
            robots_url, headers={"User-Agent": self._user_agent}
        )
        try:
            # 请求 robots.txt（不受限速约束）
            with self._open_with_retry(
                robots_request, source_id, rate_limited=False
            ) as response:
                # 读取并解码 robots.txt 内容
                robots_text = self._read_limited(response).decode(
                    "utf-8", errors="replace"
                )
        # 如果 robots.txt 不存在（404/410），使用默认开放策略
        except HTTPError as error:
            if error.code not in {404, 410}:
                raise
            robots_text = "User-agent: *\nAllow: /\n"

        # 解析 robots.txt 文本为策略对象
        robots_policy = RobotsAccessPolicy.from_text(robots_text)
        # 缓存解析后的策略
        self._robots_policies[cache_key] = robots_policy
        return robots_policy

    # 解码 HTTP 响应正文为字符串
    @classmethod
    def _decode_response(cls, response: object) -> str:
        # 从响应头获取字符集，默认使用 UTF-8
        charset = response.headers.get_content_charset() or "utf-8"
        # 读取字节流并按指定字符集解码
        return cls._read_limited(response).decode(charset, errors="replace")

    # 带重试和限速地打开 URL
    def _open_with_retry(
        self,
        request: Request,
        source_id: str,
        *,
        rate_limited: bool = True,
    ) -> object:
        # 循环尝试最多 MAX_ATTEMPTS 次
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # 如果需要限速（默认开启）
            if rate_limited:
                # 计算需要等待的秒数
                wait_seconds = self._rate_limiter.seconds_until_allowed(source_id)
                # 如果需要等待，则睡眠相应时间
                if wait_seconds > 0:
                    self._sleep(wait_seconds)
                # 记录本次请求，更新限速器状态
                self._rate_limiter.record_request(source_id)
            try:
                # 尝试打开 URL 并返回响应对象
                return self._open_url(request, timeout=self._timeout_seconds)
            # 捕获 HTTP 错误
            except HTTPError as error:
                # 如果是客户端错误（4xx）或已达最大重试次数，不再重试
                if error.code < 500 or attempt == MAX_ATTEMPTS:
                    error.close()
                    raise
                # 服务端错误（5xx）且未达最大重试次数，关闭响应后继续重试
                error.close()
            # 捕获网络连接和超时错误
            except (URLError, TimeoutError, OSError):
                # 如果已达最大重试次数，不再重试
                if attempt == MAX_ATTEMPTS:
                    raise
            # 指数退避等待后重试（1秒、2秒、4秒）
            self._sleep(float(2 ** (attempt - 1)))
        # 如果循环正常结束但没有返回，说明流程异常
        raise RuntimeError("请求重试流程异常结束")

    # 构造失败的采集结果
    @staticmethod
    def _failure_result(
        crawl_id: str,
        source_id: str,
        source_url: str,
        collected_at: datetime,
        robots_allowed: bool,
        error_message: str,
        http_status: int | None = None,
    ) -> CrawlResult:
        # 创建失败状态的采集记录
        record = CrawlRecord(
            crawl_id=crawl_id,
            source_id=source_id,
            source_url=source_url,
            document_title=None,
            document_type=None,
            content_hash=None,
            content_format=None,
            collected_at=collected_at,
            http_status=http_status,
            robots_allowed=robots_allowed,
            crawl_status="failed",
            version_status=None,
            error_message=error_message,
        )
        # 返回失败结果（内容为 None）
        return CrawlResult(record=record, content=None)
