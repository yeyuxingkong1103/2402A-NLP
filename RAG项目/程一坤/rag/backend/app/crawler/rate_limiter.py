# 导入可调用对象类型，用于标注测试时传入的时钟函数
from collections.abc import Callable


# 定义官方来源访问频率限制器
class SourceRateLimiter:
    # 初始化访问频率限制器
    def __init__(
        self,
        minimum_interval_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        # 拒绝负数访问间隔，避免错误配置导致无限频繁访问
        if minimum_interval_seconds < 0:
            raise ValueError("访问间隔不能为负数")

        # 保存同一来源两次请求之间的最小间隔
        self._minimum_interval_seconds = minimum_interval_seconds

        # 保存当前时间函数，便于测试时使用可控时钟
        self._clock = clock

        # 保存每个来源最近一次请求的时间
        self._last_request_times: dict[str, float] = {}

    # 返回指定来源还需要等待的秒数
    def seconds_until_allowed(self, source_id: str) -> float:
        last_request_time = self._last_request_times.get(source_id)
        if last_request_time is None:
            return 0.0

        elapsed_seconds = self._clock() - last_request_time
        return max(0.0, self._minimum_interval_seconds - elapsed_seconds)

    # 判断指定来源当前是否允许发起请求
    def can_request(self, source_id: str) -> bool:
        # 读取指定来源最近一次请求的时间
        last_request_time = self._last_request_times.get(source_id)

        # 来源没有历史请求时，允许第一次请求
        if last_request_time is None:
            return True

        # 获取当前时间
        current_time = self._clock()

        # 计算距离上一次请求经过的时间
        elapsed_seconds = current_time - last_request_time

        # 经过时间达到最小间隔时，允许再次请求
        return elapsed_seconds >= self._minimum_interval_seconds

    # 记录指定来源刚刚完成的一次请求
    def record_request(self, source_id: str) -> None:
        # 获取当前时间并保存为该来源最近一次请求时间
        self._last_request_times[source_id] = self._clock()
