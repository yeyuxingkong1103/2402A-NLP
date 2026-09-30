# 导入 Python 标准库中的 robots.txt 规则解析器
from urllib.robotparser import RobotFileParser


# 定义 robots.txt 禁止采集时使用的业务异常
class RobotsAccessDenied(PermissionError):
    """目标 URL 被 robots.txt 规则禁止访问。"""


# 定义 robots.txt 访问策略
class RobotsAccessPolicy:
    # 保存已经解析完成的 robots.txt 规则
    def __init__(self, robots_parser: RobotFileParser) -> None:
        # 保存解析器，供后续 URL 校验重复使用
        self._robots_parser = robots_parser

    # 根据 robots.txt 文本创建访问策略
    @classmethod
    def from_text(cls, robots_text: str) -> "RobotsAccessPolicy":
        # 创建标准 robots.txt 解析器
        robots_parser = RobotFileParser()

        # 按行拆分 robots.txt 文本并加载访问规则
        robots_parser.parse(robots_text.splitlines())

        # 返回包含已解析规则的访问策略
        return cls(robots_parser)

    # 判断指定爬虫是否允许访问目标 URL
    def can_fetch(self, user_agent: str, source_url: str) -> bool:
        # 根据 User-Agent 和 URL 查询 robots.txt 访问规则
        return self._robots_parser.can_fetch(user_agent, source_url)

    # 强制检查目标 URL，禁止访问时立即终止采集
    def require_allowed(self, user_agent: str, source_url: str) -> None:
        # 判断当前爬虫是否允许访问目标 URL
        if not self.can_fetch(user_agent, source_url):
            # 抛出明确异常，防止后续代码继续发送网络请求
            raise RobotsAccessDenied(f"robots.txt 禁止访问：{source_url}")
