# audit_log 表本身的判据：DDL 与列集合、写入的逐格落位、取材函数（召回路径/截断）。
#
# 为什么与 test_audit.py 分成两个文件：那一半判的是**中间件接线**（AC-19 的落库
# 检查、留痕失败不阻断、action 与路径的映射），这一半判的是**表与写入**；两半都
# 真连库，合在一个文件里会越过本项目 300 行的单文件闸门（test_api_surface 拆分的
# 同款理由）。每个用例自建行、按唯一 request_id / cause 精确清行。
import datetime as dt
import json
import pathlib
import uuid
from unittest.mock import MagicMock

import pytest

from app.core import audit
from app.db import audit as audit_db
from app.db.mysql import SCHEMA_PATH, connect

# 设计 §五 的列，一个不多一个不少（白名单：黑名单天然漏项，而这张表要装身份
# 与问句 —— AC-19 的落库检查全靠它）
EXPECTED_COLUMNS = {"id", "ts", "request_id", "user_id", "role", "team_id", "action",
                    "query_text", "recall_path", "verify_result", "status_code",
                    "latency_ms"}


# 本用例写过的 request_id（_rid 登记）；清理走 conn 夹具的收尾（失败的用例也要清）
_WRITTEN: list[str] = []


@pytest.fixture()
def conn():
    """真连接；建表、收尾清理、关闭都在这里（理由同 test_audit.py）。"""
    connection = connect()
    audit_db.ensure_table(connection)
    _WRITTEN.clear()
    yield connection
    _forget(connection, *_WRITTEN)
    _WRITTEN.clear()
    connection.close()


def _rid(prefix: str) -> str:
    """本次运行唯一的标记值（request_id 列）。写死值的话，「查得到」可能来自
    上一次跑剩的陈货 —— 而那种绿恰恰掩盖「本次一行都没写」。登记进 _WRITTEN，
    由 conn 夹具在收尾（含失败路径）统一删。"""
    rid = f"t7db-{prefix}-{uuid.uuid4().hex[:12]}"
    _WRITTEN.append(rid)
    return rid


def _row(conn, request_id: str) -> dict | None:
    """按唯一 request_id 取本用例写的那一行（列序 = db/audit._COLUMNS）。"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(audit_db._COLUMNS)} FROM audit_log "
                    "WHERE request_id = %s ORDER BY id DESC LIMIT 1", (request_id,))
        row = cur.fetchone()
    return None if row is None else dict(zip(audit_db._COLUMNS, row))


def _forget(conn, *request_ids: str) -> None:
    """删掉本用例写的行（只删自己那几个，绝不动别的行）。"""
    if not request_ids:
        return
    with conn.cursor() as cur:
        cur.executemany("DELETE FROM audit_log WHERE request_id = %s",
                        [(rid,) for rid in request_ids])
    conn.commit()


# ---- 表与写入（db/audit.py）----

def test_ensure_table_executes_the_ddl_and_commits():
    """上面那些用例只能证明「跑两次不报错」：本机这张表早就建好了，ensure_table
    改成 `pass` 之后，其余用例查的仍是那张老表，**整份测试照样全绿**，要等哪次
    全新部署才炸。故用假连接钉住：DDL 确实交给了游标执行、也确实提交了
    （与 test_users / test_fee_log 同一款）。
    """
    fake = MagicMock()
    audit_db.ensure_table(fake)
    fake.cursor.return_value.__enter__.return_value.execute.assert_called_once_with(
        audit_db.DDL)
    fake.commit.assert_called_once()


def _ddl_columns(ddl: str) -> set[str]:
    """从 DDL 文本里刮列名：取括号块，再按**顶层逗号**切定义块（同 test_users）。

    为什么要看 DDL 文本而不是只看库里的表：DDL 是 IF NOT EXISTS —— 表一旦建出来，
    之后谁把 DDL 改成多一列（比如加回一个存 IP 的列），老表纹丝不动、查表结构的
    用例照绿，要等哪次重建库才生效，而重建只会发生在全新部署上。
    """
    start = ddl.index("(")
    depth, body = 0, ""
    for offset, char in enumerate(ddl[start:], start):
        depth += 1 if char == "(" else -1 if char == ")" else 0
        if char == ")" and depth == 0:
            body = ddl[start + 1:offset]
            break
    # 行内 `-- 注释` 先剥掉：设计稿把「公众侧为 NULL」这类说明写在列定义行尾，
    # 而注释跟在逗号后面时会被顶层逗号切分当成**下一个**定义块的开头（首 token
    # 是 "--"），于是真正的列名（role）被吞掉、噪声 "--" 混进集合 —— 实测的初版
    # 就是这样红的。MySQL 自己也是先剥注释再解析，这里与它同口径
    body = "\n".join(line.split("--")[0] for line in body.splitlines())
    names, current, depth = set(), [], 0
    for char in body + ",":
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            tokens = "".join(current).split()
            head = tokens[0].lower() if tokens else ""
            if head and head not in {"primary", "index", "key", "unique", "constraint"}:
                names.add(head)
            current = []
            continue
        current.append(char)
    return names


def test_ddl_declares_exactly_the_designs_columns():
    """设计 §五 的列，一个不多一个不少（brief：逐列照抄，不要自行增删列）。"""
    assert _ddl_columns(audit_db.DDL) == EXPECTED_COLUMNS


def test_live_table_has_exactly_the_designs_columns(conn):
    """DDL 对、库里的表也要对 —— 两者是两份会漂移的东西（本机这张表可能建于改
    DDL 之前）。"""
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM audit_log")
        columns = {row[0].lower() for row in cur.fetchall()}
    assert columns == EXPECTED_COLUMNS


def test_the_ddl_keeps_the_one_clock_source_and_the_two_indexes():
    """ts 的 DEFAULT CURRENT_TIMESTAMP 与两条索引直接钉在 DDL 文本上：DEFAULT 没了
    等于每行的 ts 得由应用侧塞 —— 两个时钟来源，而它们分叉时按时间窗串证据链就会
    漏行。索引没了不会报错，只会让导出与按用户查在大表上慢慢变慢。
    """
    ddl = " ".join(audit_db.DDL.split())
    assert "ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP" in ddl
    assert "INDEX idx_ts (ts)" in ddl and "INDEX idx_user (user_id)" in ddl


def test_ts_and_id_are_never_written_by_the_application():
    """写入列集合里不许出现 ts / id（时间只有库侧一个来源）。上面那条钉的是 DDL
    文本，这条钉的是 INSERT 的列清单 —— 两处都要成立才叫「一处取时间」。
    """
    assert "ts" not in audit_db._WRITE_COLUMNS
    assert "id" not in audit_db._WRITE_COLUMNS
    assert set(audit_db._WRITE_COLUMNS) == EXPECTED_COLUMNS - {"id", "ts"}


def test_schema_sql_does_not_carry_a_second_copy_of_the_ddl():
    """建表语句的落点（brief 点名的裁决）：唯一落点是 db/audit.py 的 DDL，
    deploy/mysql/schema.sql 里不放第二份 —— 两份 DDL 是迟早分叉的东西，而分叉后
    的形态是「老库按旧 DDL、新库按新 DDL」，只有全新部署才看得见。这条用例把
    「不许再抄一份」变成可执行的判据。
    """
    schema = pathlib.Path(SCHEMA_PATH).read_text(encoding="utf-8")
    assert "audit_log" not in schema
    assert "CREATE TABLE" in schema, "schema.sql 读错了？三张入库侧的表该在里面"


def test_record_lands_every_field_in_its_own_column_and_commits(conn):
    """逐格比一遍 + **另开连接读回**：十二列按位置取值，串一格（role 写进 team_id）
    只有逐格比才看得出；而没提交的行在同一条连接里照样可见，所以「提交没提交」
    必须换一条会话验（test_fee_log 吃过这个亏）。
    """
    rid = _rid("record")
    when = dt.datetime(2026, 1, 2, 3, 4, 5)
    # 直插一行 ts 在过去的行：导出用例要按时间窗筛，而中间件写的行永远是「刚刚」
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO audit_log (ts, request_id, user_id, role, team_id, action,"
            " query_text, recall_path, verify_result, status_code, latency_ms)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (when, rid, 9, "partner", "team-z", "search", "押金", "584,585", "ok",
             200, 42))
    conn.commit()
    other = connect()
    try:
        row = _row(other, rid)
    finally:
        other.close()
    assert row is not None, "本连接看得见、别的连接看不见 —— 没有提交"
    assert row["ts"] == when
    assert (row["user_id"], row["role"], row["team_id"]) == (9, "partner", "team-z")
    assert row["action"] == "search" and row["query_text"] == "押金"
    assert row["recall_path"] == "584,585" and row["verify_result"] == "ok"
    assert (row["status_code"], row["latency_ms"]) == (200, 42)
    _forget(conn, rid)


def test_record_leaves_ts_to_the_database():
    """ts 由库侧 DEFAULT 给：直插一行（不带 ts 参数）后它是「刚刚」，不是 NULL、
    也不是应用侧塞的值。假连接上同时钉住 INSERT 的列清单里没有 ts。
    """
    fake = MagicMock()
    audit_db.record(fake, request_id="r", action="qa", status_code=200,
                    latency_ms=7)
    sql = fake.cursor.return_value.__enter__.return_value.execute.call_args[0][0]
    assert " ts" not in sql.split("VALUES")[0], f"INSERT 里带了 ts：{sql}"
    fake.commit.assert_called_once()


# ---- 取材（不给库）:召回路径与问句截断 ----

def _capture(action: str, response: dict | None) -> "audit._Capture":
    capture = audit._Capture()
    capture.status_code = 200
    capture.response_body = b"" if response is None else json.dumps(response).encode()
    return capture


def test_search_takes_its_recall_path_from_blocks_not_sources():
    """`/search` 的响应键是 blocks（RetrievalResult.blocks），qa 的是 sources
    （QAResult.sources）—— 读错键时 recall_path 恒 NULL 且毫无症状。故两个 action
    各钉一条：键名串了，这一对必红。
    """
    search = {"question": "押金", "blocks": [{"article_no": 584}, {"article_no": 577}],
              "exact_nos": [584]}
    assert audit._answer_facts("search", _capture("search", search)) == ("584,577", None)
    qa = {"status": "abstain", "sources": [{"article_no": 1}]}
    assert audit._answer_facts("qa", _capture("qa", qa)) == ("1", "abstain")


def test_the_question_is_capped_at_the_column_width_and_a_blank_is_null():
    """超长问句截到列宽（写不进去 = 那一行整个丢失）；空串/非字符串记 NULL（「没
    取到」与「问了个空」在事后追溯里是两回事）；逗号连接超列宽时按边界截。
    """
    capture = audit._Capture()
    capture.request_body = json.dumps({"question": "长" * 700}).encode()
    assert len(audit._question_text(capture)) == 500
    capture.request_body = json.dumps({"question": ""}).encode()
    assert audit._question_text(capture) is None
    capture.request_body = json.dumps({"question": 42}).encode()
    assert audit._question_text(capture) is None
    assert len(audit._join_capped([str(i) for i in range(1000)])) <= 255

