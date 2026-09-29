# 提供不可变的数据类，用于保存官方来源规则
from dataclasses import dataclass

# 提供 URL 解析结果类型和 URL 解析函数
from urllib.parse import ParseResult, urlparse


LABOR_LAW_SOURCE_RULES = {
    "moj_cn_labor_contract_regulation": {
        "domains": {"xzfg.moj.gov.cn"},
        "path_prefixes": {"/front/law/detail"},
    },
    "court_cn_labor_interpretation_2": {
        "domains": {"www.court.gov.cn"},
        "path_prefixes": {"/zixun/xiangqing/472691.html"},
    },
    "court_cn_labor_cases": {
        "domains": {"www.court.gov.cn"},
        "path_prefixes": {"/zixun/xiangqing/472681.html"},
    },
    "moj_cn_work_injury_insurance": {
        "domains": {"xzfg.moj.gov.cn"},
        "path_prefixes": {"/front/law/detail"},
    },
    "moj_cn_annual_leave": {
        "domains": {"xzfg.moj.gov.cn"},
        "path_prefixes": {"/front/law/detail"},
    },
    "moj_cn_female_worker_protection": {
        "domains": {"xzfg.moj.gov.cn"},
        "path_prefixes": {"/front/law/detail"},
    },
    "samr_cn_labor_contract_law": {
        "domains": {"www.samr.gov.cn"},
        "path_prefixes": {"/zw/zfxxgk/fdzdgknr/bgt/art/"},
    },
    "samr_cn_labor_law": {
        "domains": {"www.samr.gov.cn"},
        "path_prefixes": {"/zw/zfxxgk/fdzdgknr/bgt/art/"},
    },
    "gongbao_court_cn_labor_arbitration_law": {
        "domains": {"gongbao.court.gov.cn"},
        "path_prefixes": {"/Details/"},
    },
    "gov_cn_wage_payment_regulation": {
        "domains": {"www.gov.cn"},
        "path_prefixes": {"/zhengce/"},
    },
    "court_cn_labor_interpretation_1": {
        "domains": {"www.court.gov.cn"},
        "path_prefixes": {"/zixun/xiangqing/"},
    },
}


# 定义白名单校验失败时使用的业务异常
class UrlNotAllowed(ValueError):
    """目标 URL 不符合官方来源白名单。"""


# 定义单个官方来源的域名和路径规则
@dataclass(frozen=True)
class OfficialSourceRule:
    # 保存允许访问的完整域名
    domains: frozenset[str]

    # 保存允许访问的 URL 路径前缀
    path_prefixes: frozenset[str]


# 定义官方来源白名单校验器
class OfficialSourceWhitelist:
    # 初始化白名单校验器
    def __init__(
        self,
        allowed_sources: dict[str, dict[str, set[str]]],
    ) -> None:
        # 将每个来源的配置转换成不可变的规则对象
        self._source_rules = {
            # 使用来源 ID 标识当前官方来源
            source_id: OfficialSourceRule(
                # 统一域名格式，避免大小写或首尾空格影响匹配
                domains=frozenset(
                    domain.lower().strip()
                    for domain in source_config["domains"]
                ),
                # 统一路径格式，空路径默认匹配根路径
                path_prefixes=frozenset(
                    prefix.strip() or "/"
                    for prefix in source_config["path_prefixes"]
                ),
            )
            # 读取所有官方来源配置
            for source_id, source_config in allowed_sources.items()
        }

    # 校验传入的 URL 是否允许访问
    def validate(self, source_url: str) -> ParseResult:
        # 将 URL 解析成结构化结果
        parsed_url = urlparse(source_url)

        # 只允许通过 HTTPS 或 HTTP 访问官方来源
        if parsed_url.scheme.lower() not in ("https", "http"):
            # 拒绝 javascript 和其他非 HTTP(S) 协议
            raise UrlNotAllowed("只允许访问 HTTP(S) 官方来源")

        # 检查 URL 是否包含有效域名
        if not parsed_url.hostname:
            # 没有域名时无法确认访问目标，因此拒绝访问
            raise UrlNotAllowed("URL 缺少有效域名")

        # 检查域名和路径是否匹配已配置的官方来源
        if not self._matches_allowed_source(parsed_url):
            # 未匹配白名单时拒绝访问
            raise UrlNotAllowed("URL 不在官方来源白名单中")

        # 返回已经校验通过的 URL 解析结果
        return parsed_url

    # 判断解析后的 URL 是否匹配任意官方来源规则
    def _matches_allowed_source(self, parsed_url: ParseResult) -> bool:
        # 获取 URL 的小写域名，确保域名比较不受大小写影响
        hostname = parsed_url.hostname.lower()

        # 获取 URL 路径，空路径按根路径处理
        path = parsed_url.path or "/"

        # 遍历所有官方来源规则，查找至少一条匹配规则
        return any(
            # 要求域名精确匹配，避免误放行相似域名
            hostname in rule.domains
            # 要求路径以官方配置的路径前缀开头
            and any(
                path == prefix.rstrip("/")
                or path.startswith(f"{prefix.rstrip('/')}/")
                for prefix in rule.path_prefixes
            )
            # 逐个检查来源规则
            for rule in self._source_rules.values()
        )
