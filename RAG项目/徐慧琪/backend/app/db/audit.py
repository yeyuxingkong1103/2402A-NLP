"""审计表（设计 §五 的 audit_log）与它的读写。

与 users 同形：**一张表一个模块** —— 建表语句、写入、查询、口径注释同处一处。
`db/users.py` 的 docstring 就写着「审计表由后续任务照同一形态另开一个」，本模块
就是那一个；`db/mysql.py` 只放连接与执行，不认识任何具体表。

**为什么没有塞进 `deploy/mysql/schema.sql`**（设计 §五 那张表的家在哪，本任务的
裁决）：schema.sql 是三张**入库侧**数据表（law / law_version / article）的家，
由 `apply_schema` 在灌数据前执行；users 与 fee_generation_log 这两张运行期生成的
表都不在那里，各自在自己的模块里 `ensure_table`。audit_log 属后一类（由服务在
运行期写），照先例跟 users 走。更要紧的是**分叉**：同一段 DDL 存两份，改一处
漏一处是迟早的事（本项目反复吃「两处各存一份口径」的亏），而两份 DDL 分叉后的
形态是「老库按旧 DDL 建、新库按新 DDL 建」，只有全新部署才看得见。故 DDL 的唯一
落点就是本模块的 `DDL` 常量；test_audit 有一条用例钉住 schema.sql 里不出现它。

**不在这里校验任何业务口径**（与 users 的 role、fee_log 的 status 同一条）：
「公众侧 query_text 恒 NULL」的判据落在 `core/audit.py` 的**代码路径**上（公众侧
那一支根本不取问句），不靠写入方「记得传 None」—— 后者一个漏传就把公众提问写进
库了，而这一层即使加了断言也只是第二道，且会让「谁负责」变模糊。本模块只保证：
传什么写什么、列与设计逐列一致、ts 由库侧给。
"""
from __future__ import annotations

import datetime as dt

# 照抄设计 §五（字段名、类型、索引、行内说明逐字不改）。行内 `--` 注释是设计稿
# 原文，也是这张表最要紧的两条边界的**库侧留档**（公众侧不记身份、不记提问），
# 排查时对着 SHOW CREATE TABLE 就能看到，不必回代码里找。注释文本刻意不含半角
# 逗号：test_audit 的 DDL 解析按顶层逗号切定义块，注释里混进逗号会切出噪声碎片。
DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  request_id VARCHAR(64) NOT NULL,
  user_id BIGINT NULL,          -- 公众侧为 NULL：不采集身份
  role VARCHAR(16) NULL,
  team_id VARCHAR(64) NULL,
  action VARCHAR(32) NOT NULL,  -- login / qa / search / article / nav / recommend / export
  query_text VARCHAR(500) NULL, -- 仅律师侧写入；公众侧恒 NULL
  recall_path VARCHAR(255) NULL,
  verify_result VARCHAR(64) NULL,
  status_code INT NOT NULL,
  latency_ms INT NOT NULL,
  INDEX idx_ts (ts),
  INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 读取列的固定顺序（fetch_between 与用例的取行共用）：加列时改这里就够。
# id 与 ts 在写入列里**都不出现**：id 由自增给，ts 由库侧 DEFAULT 给 —— 时间只有
# 这一个来源，应用侧再塞一个 datetime.now() 就是第二个时钟，两者早晚分叉
# （NTP 回拨、时区设置、跨进程毫秒差），而审计的时间列一旦分叉，事后按时间窗
# 串证据链就会漏行。
_COLUMNS = ("id", "ts", "request_id", "user_id", "role", "team_id", "action",
            "query_text", "recall_path", "verify_result", "status_code",
            "latency_ms")
_WRITE_COLUMNS = _COLUMNS[2:]


def ensure_table(conn) -> None:
    """建表（幂等）—— 表不存在时由第一次写入前调用（时机见 core/audit.py）。

    失败**不让异常沉默**：与 users/fee_log 同口径，建表失败往上抛，由调用方
    （core/audit.py 的写入胶水）翻成一条 warning 且不阻断请求 —— 审计是旁路，
    但旁路坏了必须留痕。
    """
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def record(conn, *, request_id: str, action: str, status_code: int,
           latency_ms: int, user_id: int | None = None, role: str | None = None,
           team_id: str | None = None, query_text: str | None = None,
           recall_path: str | None = None, verify_result: str | None = None) -> None:
    """追加一行审计；缺的字段写 NULL（None 就是 None，不兜默认值）。

    **单条 INSERT + commit**：一行 = 一次请求的收尾，没有多语句事务的窗口，
    所以调用方用哪条连接都能成立（选哪条与代价见 core/audit.py 的注释与任务 7
    报告）。不返回 lastrowid：调用方一个都不用它，返回一个「碰巧正确」的 id 只会
    诱使将来有人拿它当关联键。
    """
    placeholders = ", ".join(["%s"] * len(_WRITE_COLUMNS))
    sql = (f"INSERT INTO audit_log ({', '.join(_WRITE_COLUMNS)}) "
           f"VALUES ({placeholders})")
    with conn.cursor() as cur:
        cur.execute(sql, (request_id, user_id, role, team_id, action, query_text,
                          recall_path, verify_result, status_code, latency_ms))
    conn.commit()


def fetch_between(conn, *, start: dt.datetime, end: dt.datetime,
                  limit: int) -> list[dict]:
    """按时间窗取行（含两端），**新→旧**，最多 limit 行。

    排序取新→旧而不是旧→新：LIMIT 是硬顶（上限的理由见 api/admin.py），一旦触顶，
    丢掉的是窗口里**最早**那几行 —— 排查时最近的事件才是要看的，而早于窗口的本来
    就不在这次导出里。代价写进报告：截断在响应上只表现为 len(rows) == limit，
    客户端据此收窄时间窗重导。

    时间比较交给 MySQL（ts 是库侧的本地时钟），应用侧不再做一次时区换算：两处
    换算就是两个口径。返回 dict 而非元组：十二列按位置取值，插一列就会静默错位。
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM audit_log "
            "WHERE ts >= %s AND ts <= %s ORDER BY ts DESC, id DESC LIMIT %s",
            (start, end, limit))
        return [dict(zip(_COLUMNS, row)) for row in cur.fetchall()]
