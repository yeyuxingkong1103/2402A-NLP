"""会话标题生成规则（任务 5.4）。

独立成文件：标题清洗是纯函数职责，与 MySQL 存储无关；
单独放置便于单测覆盖各种边界（空白、超长、空提问）。
"""

# 导入日期时间类型（兜底文案使用本地时间）
from datetime import datetime

# 会话标题的最大显示长度（不含省略号；与产品约定"首条提问前 20 字"一致）
TITLE_MAX_CHARS = 20


def derive_title(question: str | None, now: datetime | None = None) -> str:
    """从首条提问生成会话标题；提问为空时用"新会话 + 本地时间"兜底。

    规则（项目负责人批复）：
    1. 去掉首尾空白与换行
    2. 中间连续空白（空格/制表/换行）压缩为一个空格
    3. 取前 20 字；超长截断后追加"…"（不硬切语义，只做长度截断）
    4. 清洗后为空（例如只发了空白串）→ "新会话 YYYY-MM-DD HH:mm"

    参数：
    - question: 首条提问原文（可能为 None 或空白）
    - now: 兜底文案使用的时间；测试可注入，生产用当前本地时间

    返回：
    - 会话标题字符串（最长 TITLE_MAX_CHARS + 1 个省略号字符）
    """
    # 兜底时间：未注入时取当前本地时间（标题是给人看的展示文本，用本地时间更直观）
    now = now or datetime.now()

    # 第一步：去掉首尾空白与换行
    cleaned = (question or "").strip()

    # 第二步：中间连续空白（空格/制表符/换行）压缩为单个空格
    # str.split() 不带参数会按任意空白（含 \n \t）切分，join 回单空格即完成压缩
    cleaned = " ".join(cleaned.split())

    # 第三步：清洗后为空 → 使用"新会话 + 本地时间"兜底，绝不允许空标题
    if not cleaned:
        # 格式示例：新会话 2026-09-18 20:05
        return f"新会话 {now.strftime('%Y-%m-%d %H:%M')}"

    # 未超过上限时直接返回清洗结果（不补省略号）
    if len(cleaned) <= TITLE_MAX_CHARS:
        return cleaned

    # 第四步：超长截断取前 20 字，追加省略号标识"未显示完全"
    return cleaned[:TITLE_MAX_CHARS] + "…"
