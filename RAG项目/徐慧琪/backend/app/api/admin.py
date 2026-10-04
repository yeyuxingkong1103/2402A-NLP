"""管理侧路由：审计导出（设计 §四 的第 86 行 / §五 / FR-6.4）。

只有一条端点，且它是**律师侧专用路径**里最敏感的一条：导出的每一行都可能带着
律师侧的问句原文（query_text）。故鉴权取设计 §四 写死的 partner 档
（`require_roles(Role.PARTNER)`），而角色不够与未认证**同一种响应（404）**——
那条语义在 `deps.require_roles` 里已经落好（PermissionDeniedError 属 AuthError
家族 → main 的处理器出 404），本模块不再自己判一次：判据两处写就有一处分叉的
余地，而分叉的方向是「越权者拿到了导出」。

**必须有范围与上限**（本任务的裁决，理由设计 §一「单机自足」的同一口径）：
不给参数就导出全表 = 一次请求把库拖垮（也是给越权者留的放大器），故缺省窗口取
最近 24 小时、缺省上限 1000 行、上限硬顶 5000 行（超了判 400 而不是静默截到 5000：
「你要的比给得出的多」是调用方该知道的事，silent 截断会让「导全了」与「被截了」
长得一样）。取值与理由见下面三个常量。

**返回 JSON 数组**（brief 的裁决：本轮不做 CSV 文件下载）。代价写进报告：数组里
没有「是否被截断」的标记，客户端只能按 `len(rows) == limit` 自查 —— 想做成
`{"rows": [...], "truncated": true}` 的话要动设计 §四 的契约，故留作存疑项。
"""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query, Request

from app.api import deps
from app.core import errors
from app.core.security import CurrentUser, Role
from app.db import audit as audit_db

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# 缺省时间窗：最近 24 小时。为什么不给「全表」或更长：审计表是只增表（每天上万行
# 就会到），而缺省值会被最不经意地用到（前端不传参数的那次调用）—— 让「不传」
# 落到一个能一次答完的范围上。要更早的账就显式给 start。
DEFAULT_WINDOW = dt.timedelta(hours=24)
# 缺省窗口右端的 1 秒富余。ts 是 DATETIME(0)：写入时小数秒**四舍五入**进位
# （实测 20:45:04.700311 在库里存成 20:45:05），而「刚刚发生的那次请求」恰恰是
# 排查时最想看到的一行 —— 边界正好落在它身上时，时间窗会把它切掉，且看起来与
# 「没有这次请求」一模一样。1 秒富余的代价是缺省窗口多包 1 秒，对审计导出无影响。
# 只加在**缺省**右端：调用方显式给的 end 按字面理解（它要的是那个边界）。
END_SLACK = dt.timedelta(seconds=1)
# 缺省行数与硬顶。1000 行 ≈ 一次几百 KB 的 JSON，单机服务扛得住；5000 是「再多
# 就该收窄时间窗分批导」的线。两个都是字面量常量而不是环境变量：它们描述的是
# 「这条端点一次能安全答多少」，是接口契约的一部分，不该按机器漂。
DEFAULT_LIMIT = 1000
MAX_LIMIT = 5000


def parse_time(raw: str, *, is_end: bool) -> dt.datetime:
    """解析时间参数；不合法判 400（不是 500，也不是静默忽略）。

    接受 ISO 8601 的两种形态（`datetime.fromisoformat` 的输入）：
      - 纯日期 `2026-09-30`：**start 取当天 00:00:00，end 取当天 23:59:59.999999**
        —— 日期在区间两端的心智模型不对称（人说的「到 9 月 30 日」包含全天），
        两端同取 0 点会让 end 那天整个漏掉，而漏掉的那天在结果里没有任何痕迹。
      - 完整时刻 `2026-09-30T14:05:00`（空格分隔也收，fromisoformat 本身支持）。
    带时区偏移的输入（`+08:00`）判 400：库里的 ts 是 MySQL 的 CURRENT_TIMESTAMP，
    按会话时区给的**本地钟**，拿一个带时区的时刻去比要先把两边换算成同一个基准，
    而换算的口径（会话时区取什么）不在这条端点的输入里 —— 与其猜一个，不如让
    调用方给本地时间。非法值一律 400，文案走 errors.PUBLIC_ERRORS[400]（不含
    异常原文）。
    """
    text = raw.strip()
    # 先按字符挡偏移，再解析：`fromisoformat` 对「2026-09-30+08:00」不报错 ——
    # 它把 `+08:00` 当成**时刻**解析（实测：得到 2026-09-30 08:00，tzinfo 为
    # None），于是只靠 tzinfo 判会静默收下一个与调用方本意完全不同的时间
    if "+" in text:
        raise errors.BadRequest("时间不支持时区偏移，请给本地时间")
    try:
        moment = dt.datetime.fromisoformat(text)
    except ValueError:
        raise errors.BadRequest(f"时间格式不合法：{raw!r}") from None
    if moment.tzinfo is not None:
        raise errors.BadRequest("时间不支持时区偏移，请给本地时间")
    if is_end and len(text) == 10:
        return moment.replace(hour=23, minute=59, second=59, microsecond=999999)
    return moment


def _db_now(conn) -> dt.datetime:
    """库侧当前时刻 —— 缺省窗口两端的**唯一**来源。

    为什么不用 `datetime.now()`（本任务实测的坑）：ts 列填的是库侧的
    CURRENT_TIMESTAMP，而本机容器里的 MySQL 走 UTC，比 Windows 本地钟早 8 小时 ——
    拿应用时钟去比库里的 ts，窗口会整体错位 8 小时（24 小时的窗还看得见，一天的
    窗就空了一半），而「空结果」与「这段时间没有记录」完全不可分。窗口与列同源
    之后，这条端点只认一条时间轴；显式给的 start/end 也按这条轴理解（部署要求：
    库的时区与会话时区一致，见报告）。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT NOW()")
        return cur.fetchone()[0]


@router.get("/audit/export")
def export_audit(request: Request, start: str | None = None,
                 end: str | None = None,
                 limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
                 _user: CurrentUser = Depends(deps.require_roles(Role.PARTNER))
                 ) -> list[dict]:
    """按时间范围导出审计行（FR-6.4），新→旧，最多 limit 行（列见 db/audit.py）。

    鉴权：partner（`require_roles(Role.PARTNER)`）—— 角色不够与未认证都是 404，
    由 deps 那一条落好。请求被本端点自己审计成 action="export"（中间件在收尾写，
    与其它 action 同一条路径），故「谁导了审计」本身也可追溯（设计 §五 的 action
    清单里就有 export，不是可选项）。

    连接用 `Services.conn`：与审计写入同一条（写入选择的理由见 core/audit.py）。
    代价：这条读走业务的 REPEATABLE READ 会话，读视图取决于这条连接上一次提交的
    时刻 —— 本实例每次请求收尾都会 commit 一次审计，故实际新鲜度是「上一个请求」；
    代价与账目见任务 7 报告的存疑项。

    时间比较交给 MySQL（`fetch_between` 的 WHERE）：应用侧不把 datetime 转成字符串
    再比，避免第二套格式化口径（两处格式化就是两个口径，差一秒就漏行）。
    """
    conn = deps.services_of(request).conn
    now = _db_now(conn)
    start_at = parse_time(start, is_end=False) if start is not None \
        else now - DEFAULT_WINDOW
    end_at = parse_time(end, is_end=True) if end is not None \
        else now + END_SLACK
    if start_at > end_at:
        # 反向区间判 400 而不是回空数组：空结果是「这段时间没有记录」的诚实答案，
        # 用它来回答「你把区间写反了」会让调用方以为导出成功
        raise errors.BadRequest("start 晚于 end：时间范围写反了")
    return audit_db.fetch_between(conn, start=start_at, end=end_at, limit=limit)
