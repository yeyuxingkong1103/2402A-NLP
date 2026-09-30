# 导入数据类工具，用于返回带保存路径的采集结果
from dataclasses import dataclass, replace
from tempfile import NamedTemporaryFile

# 导入路径类型，用于安全管理采集文件目录
from pathlib import Path

# 导入当前采集请求器和采集记录
from app.crawler.crawl_record import CrawlRecord
from app.crawler.requester import CrawlResult, OfficialPageRequester


# 保存采集记录以及原始文件保存位置
@dataclass(frozen=True)
class SavedCrawlResult:
    # 保存采集状态和内容哈希
    record: CrawlRecord

    # 保存页面正文，失败时为空
    content: str | None

    # 保存原始文件路径，失败或重复时为空
    file_path: Path | None


# 执行单页面采集，并将成功正文保存到指定目录
class CrawlRunner:
    # 初始化请求器和原始文档保存目录
    def __init__(
        self,
        requester: OfficialPageRequester,
        document_root: Path,
    ) -> None:
        # 保存官方页面请求器
        self._requester = requester
        # 保存原始文档目录并确保目录存在
        self._document_root = document_root
        self._document_root.mkdir(parents=True, exist_ok=True)

    # 采集页面并根据内容哈希保存原始文件
    def run(
        self,
        crawl_id: str,
        source_id: str,
        source_url: str,
    ) -> SavedCrawlResult:
        # 执行白名单、robots.txt、限速和 HTTP 请求流程
        crawl_result = self._requester.fetch(
            crawl_id=crawl_id,
            source_id=source_id,
            source_url=source_url,
        )

        # 采集失败时不创建空文件
        if crawl_result.content is None or crawl_result.record.content_hash is None:
            return SavedCrawlResult(
                record=crawl_result.record,
                content=None,
                file_path=None,
            )

        # 根据内容哈希生成稳定文件名，避免 URL 产生路径风险
        file_path = self._document_path(crawl_result)

        # 已存在相同哈希文件时不重复写入
        if file_path.exists():
            unchanged_record = replace(
                crawl_result.record,
                version_status="unchanged",
            )
            return SavedCrawlResult(
                record=unchanged_record,
                content=crawl_result.content,
                file_path=file_path,
            )

        # 先写入临时文件，完成后再替换目标文件
        temporary_path = self._temporary_path(file_path)
        try:
            temporary_path.write_text(crawl_result.content, encoding="utf-8")
            temporary_path.replace(file_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        # 返回已保存的原始采集结果
        return SavedCrawlResult(
            record=crawl_result.record,
            content=crawl_result.content,
            file_path=file_path,
        )

    # 根据采集内容哈希生成原始文件路径
    def _document_path(self, crawl_result: CrawlResult) -> Path:
        # 当前官方页面按 HTML 格式保存
        suffix = ".html"
        # 使用哈希作为文件名，确保路径不受外部 URL 控制
        return self._document_root / f"{crawl_result.record.content_hash}{suffix}"

    @staticmethod
    def _temporary_path(file_path: Path) -> Path:
        with NamedTemporaryFile(
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=True,
        ) as temporary_file:
            return Path(temporary_file.name)
