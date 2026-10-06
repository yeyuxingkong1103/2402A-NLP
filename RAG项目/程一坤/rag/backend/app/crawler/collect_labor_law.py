"""采集首批劳动法相关官方页面并保存原始文件。

用途：为离线 Pipeline 提供真实数据，替代 mock 内容，便于观察解析和条文切块的真实效果。

运行方式（在 backend 目录下）：

    python -m app.crawler.collect_labor_law

输出位置：
    默认写入项目 data/labor_law_raw 目录；
    部署时可以设置环境变量 LEGAL_RAG_RAW_DATA_DIRECTORY 指向数据盘。

采集行为：
    只访问白名单内的官方域名与路径，遵守 robots.txt，
    同一来源按最小间隔串行访问，不并发、不绕过任何防护。
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.crawler.rate_limiter import SourceRateLimiter
from app.crawler.requester import OfficialPageRequester
from app.crawler.runner import CrawlRunner, SavedCrawlResult
from app.crawler.whitelist import LABOR_LAW_SOURCE_RULES, OfficialSourceWhitelist

# 输出目录不写死绝对路径：优先读环境变量，未设置时按项目根目录推导。
# 这样换机器、换目录、上服务器都不需要改代码。
OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE = "LEGAL_RAG_RAW_DATA_DIRECTORY"

# 本文件位于 backend/app/crawler/ 下，向上四层即项目根目录
PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_DIRECTORY / "data" / "labor_law_raw"

# 同一来源两次请求之间的最小间隔（秒）。
# 官方站点是公共资源，宁可慢一点，也不要给对方造成压力或被判定为异常访问。
MINIMUM_REQUEST_INTERVAL_SECONDS = 3.0

# 请求头里标识自己，方便站点管理员在必要时识别与联系
CRAWLER_USER_AGENT = "legal-rag-bot/0.1"


@dataclass(frozen=True)
class LaborLawTarget:
    """一个待采集的官方页面。"""

    source_id: str
    source_url: str
    document_title: str | None = None  # 人工核对的法规名，优先级高于页面提取


# 首批目标：已确认的官方公开页面，source_id 必须与白名单配置一一对应。
# 白名单在 app/crawler/whitelist.py 的 LABOR_LAW_SOURCE_RULES，新增目标时两处要同时更新。
LABOR_LAW_TARGETS = (
    LaborLawTarget(
        # 司法部行政法规库：劳动合同法实施条例
        "moj_cn_labor_contract_regulation",
        "https://xzfg.moj.gov.cn/front/law/detail?LawID=284",
        document_title="中华人民共和国劳动合同法实施条例",
    ),
    LaborLawTarget(
        # 最高人民法院：劳动争议司法解释
        "court_cn_labor_interpretation_2",
        "https://www.court.gov.cn/zixun/xiangqing/472691.html",
        document_title=None,  # 使用页面 <title> 提取
    ),
    LaborLawTarget(
        # 最高人民法院：劳动争议相关案例
        "court_cn_labor_cases",
        "https://www.court.gov.cn/zixun/xiangqing/472681.html",
        document_title=None,  # 使用页面 <title> 提取
    ),
    LaborLawTarget(
        # 司法部行政法规库：工伤保险条例
        "moj_cn_work_injury_insurance",
        "https://xzfg.moj.gov.cn/front/law/detail?LawID=610",
        document_title="工伤保险条例",
    ),
    LaborLawTarget(
        # 司法部行政法规库：职工带薪年休假条例
        "moj_cn_annual_leave",
        "https://xzfg.moj.gov.cn/front/law/detail?LawID=208",
        document_title="职工带薪年休假条例",
    ),
    LaborLawTarget(
        # 司法部行政法规库：女职工劳动保护特别规定
        "moj_cn_female_worker_protection",
        "https://xzfg.moj.gov.cn/front/law/detail?LawID=343",
        document_title="女职工劳动保护特别规定",
    ),
    LaborLawTarget(
        # 国家市场监督管理总局：劳动合同法
        "samr_cn_labor_contract_law",
        "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_0abfdd261c03417b949df19d869add8d.html",
        document_title="中华人民共和国劳动合同法",
    ),
    LaborLawTarget(
        # 国家市场监督管理总局：劳动法
        "samr_cn_labor_law",
        "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_d9aa750028b14b99a776cb93726a360d.html",
        document_title="中华人民共和国劳动法",
    ),
    LaborLawTarget(
        # 最高人民法院公报：劳动争议调解仲裁法
        "gongbao_court_cn_labor_arbitration_law",
        "http://gongbao.court.gov.cn/Details/997e66171cf55d219c613ec18dc370.html",
        document_title="中华人民共和国劳动争议调解仲裁法",
    ),
    LaborLawTarget(
        # 中国政府网：工资支付暂行规定
        "gov_cn_wage_payment_regulation",
        "https://www.gov.cn/zhengce/2022-08/31/content_5711284.htm",
        document_title="工资支付暂行规定",
    ),
    LaborLawTarget(
        # 最高人民法院：劳动争议司法解释（一）
        "court_cn_labor_interpretation_1",
        "https://www.court.gov.cn/zixun/xiangqing/282121.html",
        document_title="最高人民法院关于审理劳动争议案件适用法律问题的解释（一）",
    ),
)


def resolve_output_root() -> Path:
    """返回本次采集的输出目录；环境变量优先，便于部署时改到数据盘。"""
    configured_directory = os.environ.get(
        OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE, ""
    ).strip()
    if configured_directory:
        return Path(configured_directory)
    return DEFAULT_OUTPUT_ROOT


def build_crawl_runner(
    output_root: Path,
    *,
    open_url: Callable[..., object] | None = None,
) -> CrawlRunner:
    """组装一个受白名单、robots.txt 和限速约束的采集器。

    open_url 用于测试时注入替身，避免测试真的访问官方站点。
    """
    requester_arguments = {
        "whitelist": OfficialSourceWhitelist(LABOR_LAW_SOURCE_RULES),
        "rate_limiter": SourceRateLimiter(
            minimum_interval_seconds=MINIMUM_REQUEST_INTERVAL_SECONDS,
            clock=time.monotonic,
        ),
        "user_agent": CRAWLER_USER_AGENT,
    }
    if open_url is not None:
        requester_arguments["open_url"] = open_url
    return CrawlRunner(
        requester=OfficialPageRequester(**requester_arguments),
        document_root=output_root,
    )


def collect_labor_law(
    output_root: Path | None = None,
    *,
    open_url: Callable[..., object] | None = None,
) -> tuple[SavedCrawlResult, ...]:
    """按顺序采集每个目标页面，返回逐个目标对应的采集结果。

    采集结果里带有状态、HTTP 状态码、内容哈希和落盘路径，
    便于下一步判断是新增、未变化还是失败。
    """
    resolved_output_root = output_root or resolve_output_root()
    crawl_runner = build_crawl_runner(resolved_output_root, open_url=open_url)

    results = []
    for index, target in enumerate(LABOR_LAW_TARGETS, start=1):
        results.append(
            crawl_runner.run(
                # 编号固定且可读，便于在日志中定位是第几个目标
                crawl_id=f"labor-law-{index:03d}",
                source_id=target.source_id,
                source_url=target.source_url,
            )
        )
    return tuple(results)


def main() -> int:
    """命令行入口：打印每个目标的采集结果。

    退出码：全部成功返回 0，任意一个失败返回 1，方便脚本串联时判断。
    """
    results = collect_labor_law()
    for result in results:
        record = result.record
        # 只打印哈希前缀，避免把完整正文或长哈希刷到终端
        content_hash_prefix = (record.content_hash or "")[:16]
        saved_file_path = str(result.file_path) if result.file_path else "-"
        print(
            f"{record.source_id}\t{record.crawl_status}\t"
            f"{record.http_status or '-'}\t{content_hash_prefix}\t{saved_file_path}"
        )
    all_succeeded = all(
        result.record.crawl_status == "success" for result in results
    )
    return 0 if all_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
