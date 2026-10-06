"""费用生成留痕（AC-21 的证据）。

为什么要落库：AC-21 要求费用展示「可追溯」—— 出问题时要知道某个区间是据
哪段语料、哪个模型、哪次请求生成的。**被回查拒绝的区间同样要记**：
「拒绝过什么」和「输出过什么」一样是审计材料。

AC-19 约束：本表不记录用户问题原文、IP 或任何身份信息，故表结构里
根本不设这些列（测试钉住这一点）。
"""
from __future__ import annotations

# ts 用数据库默认值：避免应用时钟与数据库时钟不一致
DDL = """
CREATE TABLE IF NOT EXISTS fee_generation_log (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  cause VARCHAR(64) NULL COMMENT '案由标签；刻意不存用户问题原文（AC-19）',
  snippet_id VARCHAR(128) NULL COMMENT '命中的收费口径片段 id',
  range_low DECIMAL(12,2) NULL,
  range_high DECIMAL(12,2) NULL,
  model VARCHAR(64) NULL,
  request_id VARCHAR(64) NULL,
  status VARCHAR(16) NOT NULL COMMENT 'ok / rejected',
  INDEX idx_ts (ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 插入列的固定顺序：record 与 make_logger 共用，改这里就够
_COLUMNS = ("cause", "snippet_id", "range_low", "range_high",
            "model", "request_id", "status")


def ensure_table(conn) -> None:
    """建表（幂等）—— 启动时无条件调用。"""
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def record(conn, **fields) -> None:
    """写一行留痕。只认 _COLUMNS 里的字段，缺的记 NULL。

    缺的记 NULL 而不是抛错，是因为**判拒那一行天生就缺字段**（无区间、无
    snippet_id、甚至没问到模型）—— 拒用同样在 AC-21 的追溯范围里，若这里
    要求字段齐全，反而会把最需要审计的那类行挡在库外。
    字段名不匹配时按「不认」处理（丢弃，不写列也不报错）：这是 AC-19 在
    结构之外的第二道保证 —— 调用方即便手滑多传一个 question，也不会有任何
    一列接得住它。
    """
    placeholders = ", ".join(["%s"] * len(_COLUMNS))
    values = tuple(fields.get(name) for name in _COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO fee_generation_log ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders})", values)
    conn.commit()


def make_logger(conn):
    """返回可直接传给 fees.estimate(log_fn=...) 的可调用对象。

    做成适配器的理由：estimate 的契约是 log_fn(**fields)（全部按关键字传），
    而 record 要的第一个位置参数是连接 —— 直接把 record 传进去会因缺 conn
    在真跑时报 TypeError，而这类错在静态检查里看不出来，故用闭包把连接绑死。
    """
    def log(**fields) -> None:
        record(conn, **fields)
    return log
