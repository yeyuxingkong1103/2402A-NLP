"""检索前的元数据过滤：法域、适用时间点、文书类型、是否现行有效。

为什么必须在「召回之前」过滤，而不是召回后再筛：
1. 召回后再筛会先浪费一次向量检索的算力，还把结果条数挤占掉；
2. 更严重的是"不该出现的历史版本"会混进候选，模型据此作答就是错的答案。
   法律场景里，答错版本比答不出更危险，所以过滤必须前置。

日期字段在 Milvus 里存的是 Unix 时间戳（INT64）：
- effective_date == 0 表示"生效日期未知（待人工补录）"，它不是 1970-01-01
- expiration_date 为空（NULL）表示"尚未失效"
"""

from datetime import date, datetime, timezone

# 生效日期未知时写入的哨兵值（详见 app/db/vector_index_service.py 的说明）
UNKNOWN_DATE_TIMESTAMP = 0


def _escape_literal(value: str) -> str:
    """转义 Milvus 过滤 DSL 字符串字面量中的反斜杠和双引号。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class FilterError(ValueError):
    """过滤条件非法（例如日期格式不对、法域为空字符串）。"""


def to_timestamp(value: date | datetime | str | None) -> int | None:
    """把日期转成 Unix 时间戳；None 原样返回。

    支持三种入参：date、datetime、以及 "YYYY-MM-DD" 文本（命令行与接口层常用）。
    统一按 UTC 零点换算，与写入索引时的换算方式保持一致，
    否则"同一天"的边界判断会前后不一致。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(moment.timestamp())
    if isinstance(value, date):
        return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp())
    if isinstance(value, str):
        try:
            parsed = datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError as error:
            raise FilterError(f"日期格式应为 YYYY-MM-DD，收到：{value!r}") from error
        return int(datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc).timestamp())
    raise FilterError(f"不支持的日期类型：{type(value).__name__}")


def build_filter_expression(
    *,
    as_of_date: date | datetime | str | None = None,
    jurisdiction: str | None = None,
    document_types: list[str] | None = None,
    only_current: bool = False,
    include_unknown_effective_date: bool = True,
) -> str | None:
    """拼出 Milvus 的过滤表达式；没有任何条件时返回 None（表示不过滤）。

    参数含义：
    - as_of_date：适用时间点，例如问的是"2019 年有效的法条"就传 2019-12-31
    - jurisdiction：法域，本项目当前全部为"中国大陆"
    - document_types：限定文书类型，例如只要"法律"和"行政法规"
    - only_current：只要现行有效的条文
    - include_unknown_effective_date：
        生效日期待补录的条文（effective_date == 0）是否纳入结果。
        默认纳入 —— 缺数据不该等于"这条法不存在"，静默隐藏会让用户以为库里没有；
        纳入后由回答层标注"生效日期待补录"，把不确定性交给用户判断。
    """
    conditions: list[str] = []

    # 法域过滤：本批数据全部为中国大陆，传空字符串视为非法条件
    if jurisdiction is not None:
        if not jurisdiction.strip():
            raise FilterError("法域不能是空字符串；不需要限制时请传 None")
        conditions.append(f'jurisdiction == "{_escape_literal(jurisdiction.strip())}"')

    # 文书类型过滤：用 in 语法，避免拼一长串 or
    if document_types:
        cleaned = [item.strip() for item in document_types if item and item.strip()]
        if not cleaned:
            raise FilterError("文书类型列表不能全为空")
        quoted = ", ".join(f'"{_escape_literal(item)}"' for item in cleaned)
        conditions.append(f"document_type in [{quoted}]")

    # 现行有效过滤
    if only_current:
        conditions.append("is_current == true")

    # 适用时间点过滤（时效的核心）
    timestamp = to_timestamp(as_of_date)
    if timestamp is not None:
        # ① 尚未生效的版本必须排除：生效日期明确晚于查询时间点的，在那个时间点还没生效。
        #    生效日期未知（0）的条文按 include_unknown_effective_date 决定是否保留
        if include_unknown_effective_date:
            conditions.append(f"(effective_date == {UNKNOWN_DATE_TIMESTAMP} or effective_date <= {timestamp})")
        else:
            conditions.append(
                f"(effective_date != {UNKNOWN_DATE_TIMESTAMP} and effective_date <= {timestamp})"
            )
        # ② 已失效的版本必须排除：失效日期存在且早于或等于查询时间点
        #    失效日期为空表示尚未失效，应予保留
        conditions.append(f"(expiration_date is null or expiration_date > {timestamp})")

    return " and ".join(conditions) if conditions else None
