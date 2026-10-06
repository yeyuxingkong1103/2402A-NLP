"""条文路径生成规则。

原 legal_meta.py 并入部分：提供条文路径（article_path）生成与编号校验，
供入库映射调用。路径规则与长度上限和数据库字段定义保持一致。
"""


def build_article_path(
    article_number: str,
    paragraph_number: str | None = None,
    item_number: str | None = None,
) -> str:
    """生成条文路径标识。

    规则：
    - 只有条号：返回 "47"
    - 条 + 款：返回 "47-2"
    - 条 + 款 + 项：返回 "47-2-1"
    - "之一"条款：返回 "99之1"（"第九十九条之一"）
    - "之一"条款 + 款：返回 "99之1-1"（"第九十九条之一第一款"）
    - 前导零会被去除："0047" → "47"，"0047-002" → "47-2"

    输入约束（违反任一条均抛 ValueError）：
    - article_number 不能为空或只有空白
    - article_number / paragraph_number / item_number 必须是阿拉伯数字编号（可含「之」），不接受中文形式
    - 传了值（非 None）时，不能是空串或只有空白
    - 有项号时必须有款号（不允许静默丢弃项号）
    - 单个编号长度 ≤ 16 字符（与数据库 VARCHAR(64) 字段定义保持一致）
    - 最终 article_path 长度 ≤ 128 字符（与数据库 VARCHAR(128) 字段定义保持一致）
    - 长度上限与数据库字段定义保持一致，修改其一必须同步修改另一处

    参数：
    - article_number: 条号（阿拉伯数字字符串，如 "47" 或 "99之1"）
    - paragraph_number: 款号（阿拉伯数字字符串，如 "2"，可选）
    - item_number: 项号（阿拉伯数字字符串，如 "1"，可选）

    返回：
    - 条文路径字符串

    异常：
    - ValueError: 输入不符合约束时抛出

    示例：
    >>> build_article_path("47")
    '47'
    >>> build_article_path("47", "2")
    '47-2'
    >>> build_article_path("47", "2", "1")
    '47-2-1'
    >>> build_article_path("0047", "002", "001")
    '47-2-1'
    >>> build_article_path("99之1")
    '99之1'
    >>> build_article_path("99之1", "1")
    '99之1-1'
    >>> build_article_path("")
    Traceback (most recent call last):
        ...
    ValueError: 条号不能为空
    >>> build_article_path("第四十七条")
    Traceback (most recent call last):
        ...
    ValueError: 条号必须是阿拉伯数字编号（可含「之」），入库前必须先规范化，收到："第四十七条"
    >>> build_article_path("47", "")
    Traceback (most recent call last):
        ...
    ValueError: 款号不能为空或只有空白，收到：''
    >>> build_article_path("47", "二")
    Traceback (most recent call last):
        ...
    ValueError: 款号必须是阿拉伯数字编号（可含「之」），入库前必须先规范化，收到：'二'
    >>> build_article_path("47", None, "1")
    Traceback (most recent call last):
        ...
    ValueError: 有项号时必须有款号，不允许跳过款号直接指定项号
    """

    def _validate_and_normalize(value: str, field_name: str) -> str:
        """校验并规范化编号：去除前后空白、去除前导零、校验格式、校验长度。"""
        # 校验非空
        if not value or not value.strip():
            raise ValueError(f"{field_name}不能为空或只有空白，收到：{value!r}")

        value = value.strip()

        # 校验长度上限（与数据库字段 VARCHAR(64) 保持一致，现实法律条号最长 5 位，16 已有充足余量）
        if len(value) > 16:
            raise ValueError(
                f"{field_name}长度不能超过 16 字符（与数据库字段定义一致），收到长度：{len(value)}"
            )

        # 校验格式：只能包含 ASCII 数字 (0-9) 和「之」字，不接受全角数字
        for ch in value:
            if ch != "之" and not ("0" <= ch <= "9"):
                raise ValueError(
                    f"{field_name}必须是阿拉伯数字编号（可含「之」），入库前必须先规范化，收到：{value!r}"
                )

        # 校验「之」的使用规则
        if "之" in value:
            parts = value.split("之")
            # 最多只能有一个「之」
            if len(parts) > 2:
                raise ValueError(
                    f"{field_name}最多只能包含一个「之」字，收到：{value!r}"
                )
            # 「之」前后必须都有数字
            if not parts[0] or not parts[1]:
                raise ValueError(
                    f"{field_name}中「之」前后必须都有数字，收到：{value!r}"
                )

        # 去除前导零
        # 特例：全零字符串（"0"、"00"、"000"）规范化为 "0"
        if value.replace("之", "").replace("0", "") == "":
            # 全是 0 和「之」的组合
            if "之" in value:
                parts = value.split("之")
                normalized_parts = []
                for part in parts:
                    stripped = part.lstrip("0")
                    if not stripped:  # 全是 0
                        raise ValueError(
                            f"{field_name}中「之」前后不能是纯 0，收到：{value!r}"
                        )
                    normalized_parts.append(stripped)
                normalized = "之".join(normalized_parts)
            else:
                # 纯数字且全是 0
                normalized = "0"
        else:
            # 正常情况：去除前导零
            if "之" in value:
                parts = value.split("之")
                normalized_parts = []
                for part in parts:
                    stripped = part.lstrip("0")
                    if not stripped:  # lstrip 后为空说明全是 0，但整体不是全 0（前面已处理）
                        raise ValueError(
                            f"{field_name}中「之」前后不能是纯 0，收到：{value!r}"
                        )
                    normalized_parts.append(stripped)
                normalized = "之".join(normalized_parts)
            else:
                normalized = value.lstrip("0")
                if not normalized:  # 不应该到这里，因为全 0 已在上面处理
                    normalized = "0"

        return normalized

    # 校验条号非空（在 _validate_and_normalize 之前单独检查，以提供更明确的错误信息）
    if not article_number or not article_number.strip():
        raise ValueError("条号不能为空")

    # 校验并规范化条号
    article_number = _validate_and_normalize(article_number, "条号")

    # 校验款项关系：有项必有款
    if item_number is not None and paragraph_number is None:
        raise ValueError("有项号时必须有款号，不允许跳过款号直接指定项号")

    # 校验并规范化款号
    if paragraph_number is not None:
        paragraph_number = _validate_and_normalize(paragraph_number, "款号")

    # 校验并规范化项号
    if item_number is not None:
        item_number = _validate_and_normalize(item_number, "项号")

    # 生成路径
    if paragraph_number is None:
        path = article_number
    elif item_number is None:
        path = f"{article_number}-{paragraph_number}"
    else:
        path = f"{article_number}-{paragraph_number}-{item_number}"

    # 校验最终路径长度（与数据库字段 VARCHAR(128) 保持一致）
    if len(path) > 128:
        raise ValueError(
            f"article_path 长度不能超过 128 字符（与数据库字段定义一致），生成的路径长度：{len(path)}"
        )

    return path
