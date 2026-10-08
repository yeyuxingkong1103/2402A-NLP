"""退出码与异常。

退出码必须区分失败类别，供上层人工/脚本判断（FR-007）。
"""

EXIT_OK = 0
EXIT_FAIL = 1            # 未归类的失败
EXIT_BAD_INPUT = 2       # 输入不存在 / 缺产物 / JSON 损坏 / 无待处理文档
EXIT_UNKNOWN_TYPE = 3    # 出现 docs/04 §5 未列出的块类型
EXIT_PROTECT_REGRESSION = 4  # 保护类判据归零（FR-018）
EXIT_EMPTY_OUTPUT = 5    # 清洗后块数为 0
EXIT_DEP_MISSING = 6     # 运行环境缺依赖


class CleanError(Exception):
    """所有可预期的失败都以此抛出，携带退出码。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
