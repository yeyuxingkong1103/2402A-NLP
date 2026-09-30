# 导入数据类装饰器，用于定义统一 PDF 解析结果
from dataclasses import dataclass

# 导入路径类型，用于读取待解析的 PDF 文件
from pathlib import Path


# 表示远程 PDF 解析失败
class PdfParsingError(RuntimeError):
    """PDF 远程解析未返回可用正文。"""


# 保存 PDF 解析后的统一结果
@dataclass(frozen=True)
class ParsedPdf:
    # 保存解析后的正文内容
    content: str

    # 保存 PDF 页数
    page_count: int

    # 保存实际提供结果的解析器名称
    parser_name: str


# 调用 MinerU，并在结果不完整时使用 Qwen-VL 补充
def parse_pdf(
    source_path: Path,
    mineru_client: object,
    qwen_vl_client: object,
) -> ParsedPdf:
    # 确认输入文件存在，避免把文件错误伪装成远程解析失败
    if not source_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{source_path}")

    # 先调用 MinerU 获取结构化解析结果
    mineru_result = mineru_client.parse(source_path)

    # 读取 MinerU 返回的正文和页数
    mineru_content = mineru_result.get("content", "")
    mineru_page_count = mineru_result.get("page_count", 0)

    # MinerU 结果完整时直接返回，避免额外消耗 Qwen-VL API
    if mineru_result.get("complete") and mineru_content.strip():
        return ParsedPdf(
            content=mineru_content,
            page_count=mineru_page_count,
            parser_name="mineru",
        )

    # MinerU 结果不完整时调用 Qwen-VL 补充页面内容
    qwen_vl_content = qwen_vl_client.parse(source_path)

    # Qwen-VL 也没有返回正文时明确报告解析失败
    if not qwen_vl_content or not qwen_vl_content.strip():
        error_message = mineru_result.get("error", "远程解析未返回正文")
        raise PdfParsingError(f"PDF 解析失败：{error_message}")

    # 页数来源随分支变化：MinerU 报得出页数就用它的，报不出（0）时改用 Qwen-VL
    # 渲染出的真实页数 —— 否则走兜底分支的文件会被记成"0 页"。
    resolved_page_count = mineru_page_count or _qwen_vl_page_count(qwen_vl_client)

    # 返回 Qwen-VL 补充后的统一 PDF 结果
    return ParsedPdf(
        content=qwen_vl_content,
        page_count=resolved_page_count,
        parser_name="qwen-vl",
    )


# 读取 Qwen-VL 客户端上次解析报告的页数
def _qwen_vl_page_count(qwen_vl_client: object) -> int:
    """取 Qwen-VL 兜底分支的页数；读不到时返回 0。

    兜底分支的页数只能来自客户端自持的解析痕迹：`parse()` 的对外契约是"返回正文文本"
    （见 `app/models/qwen_vl.py`），页数记在 `last_trace["page_count"]`。
    这里用 getattr 容忍任何没实现该属性的替身，读不到就让页数保持 0，不把主流程带崩。
    """
    trace = getattr(qwen_vl_client, "last_trace", None)

    # 痕迹不存在或不是字典（替身/异常实现）时视为没有页数
    if not isinstance(trace, dict):
        return 0

    # 页数可能是 None 或非法字符串，统一收敛成整数 0
    try:
        return int(trace.get("page_count", 0) or 0)
    except (TypeError, ValueError):
        return 0
