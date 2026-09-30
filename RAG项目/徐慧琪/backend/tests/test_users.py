# users 表的测试**真连 MySQL**（项目惯例：能用真依赖就不用替身）—— 它同时是
# 「建表语句与库里的表一致」的集成测。每个用例自建账号、按唯一用户名精确删除，
# 重复跑不会互相干扰，也不会在库里留垃圾（模式照 test_fee_log.py）。
import uuid
from unittest.mock import MagicMock

import pymysql
import pytest

from app.core import security
from app.db import users
from app.db.mysql import connect

# 设计 §五 说的列，一个不多一个不少。白名单而不是黑名单：黑名单天然漏项
# （换成 user_question / client_ip 就查不出来），而这张表是要装身份信息的。
EXPECTED_COLUMNS = {"id", "username", "password_hash", "role", "team_id",
                    "is_active", "created_at"}


def _unique() -> str:
    """每次跑都换一个用户名：库里可能留着上一次跑剩的行，「查得到」必须来自本次插入。"""
    return f"t-{uuid.uuid4().hex[:12]}"


def _forget(conn, names) -> None:
    """删掉本用例建的账号（只删自己那几个，绝不动别的行）。"""
    if not names:
        return
    with conn.cursor() as cur:
        cur.executemany("DELETE FROM users WHERE username = %s",
                        [(name,) for name in names])
    conn.commit()


@pytest.fixture()
def conn():
    connection = connect()
    users.ensure_table(connection)
    yield connection
    connection.close()


@pytest.fixture()
def make_account(conn):
    """账号工厂：建一个唯一用户名的账号，用例结束按用户名删掉。

    清理放 finally（yield 之后）而不是各用例末尾：断言失败的用例也要被清干净，
    否则留下的行会在下一次跑时冒充「本次插入」（test_fee_log 那边吃过这个亏）。
    """
    created: list[str] = []

    def make(role: str = "lawyer", team_id: str = "team-a",
             password: str = "pw-123456", active: bool = True):
        username = _unique()
        created.append(username)
        user_id = users.create(conn, username, security.hash_password(password),
                               role, team_id)
        if not active:
            users.set_active(conn, username, False)
        return username, user_id, password

    yield make
    _forget(conn, created)


# ---- 建表 ----

def test_ensure_table_is_idempotent(conn):
    """跑两次不报错 —— 启动时与运维脚本里都会无条件调用它。"""
    users.ensure_table(conn)


def test_ensure_table_executes_the_ddl_and_commits():
    """上一行只能证明「跑两次不报错」：表在本机早就建好了，ensure_table 改成 `pass`
    之后，其余用例查的仍是那张老表，**整份测试照样全绿**，要等哪次全新部署才炸。
    故用假连接钉住：DDL 确实交给了游标执行，也确实提交了（fee_log 的 K8 同款）。
    """
    fake = MagicMock()
    users.ensure_table(fake)
    fake.cursor.return_value.__enter__.return_value.execute.assert_called_once_with(
        users.DDL)
    fake.commit.assert_called_once()


def _ddl_columns(ddl: str) -> set[str]:
    """从 DDL 文本里刮出列名：先取括号块，再按**顶层逗号**切出每条定义。

    与 test_fee_log 那份同法、但分开写：一份针对 fee_log.DDL、一份针对 users.DDL，
    合起来就得为了两侧的形状差异加参数，反而更难读。
    为什么要看 DDL 文本而不是只看库里的表：DDL 是 IF NOT EXISTS —— 表一旦建出来，
    之后谁把 DDL 改坏（比如给 team_id 去掉 NOT NULL），老表纹丝不动，查表结构的
    用例照绿，要等哪次重建库才出问题，而重建只会发生在全新部署上。
    """
    start = ddl.index("(")
    depth, body = 0, ""
    for offset, char in enumerate(ddl[start:], start):
        depth += 1 if char == "(" else -1 if char == ")" else 0
        if char == ")" and depth == 0:
            body = ddl[start + 1:offset]
            break
    names, current, depth = set(), [], 0
    for char in body + ",":
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            tokens = "".join(current).split()
            head = tokens[0].lower() if tokens else ""
            # PRIMARY KEY / INDEX 这类是索引声明，不是列
            if head and head not in {"primary", "index", "key", "unique",
                                     "constraint"}:
                names.add(head)
            current = []
            continue
        current.append(char)
    return names


def test_ddl_declares_exactly_the_documented_columns():
    """设计 §五 的列，多一列少一列都要过裁决（这一列列会被读进 JWT 载荷）。"""
    assert _ddl_columns(users.DDL) == EXPECTED_COLUMNS


def test_live_table_has_exactly_the_documented_columns(conn):
    """DDL 对、库里的表也要对 —— 两者是两份会漂移的东西（本机这张表可能建于改 DDL 之前）。"""
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM users")
        columns = {row[0].lower() for row in cur.fetchall()}
    assert columns == EXPECTED_COLUMNS


def test_ddl_text_keeps_the_four_constraints():
    """直接在 DDL **文本**上钉四条：username UNIQUE、team_id NOT NULL、is_active
    DEFAULT 1、created_at DEFAULT CURRENT_TIMESTAMP。

    为什么非要看文本：DDL 是 IF NOT EXISTS —— 本机这张表早就建好了，之后谁把 DDL 改松
    （去掉 UNIQUE、去掉 NOT NULL），库里的老表纹丝不动、上面那些查表结构的用例照绿，
    要等哪次全新部署才生效，而那时新库比旧库松，没人会注意到。空格归一化只是为了
    让折行不挡断言。
    """
    ddl = " ".join(users.DDL.split())
    assert "username VARCHAR(64) NOT NULL UNIQUE" in ddl
    assert "team_id VARCHAR(64) NOT NULL" in ddl
    assert "is_active TINYINT(1) NOT NULL DEFAULT 1" in ddl
    assert "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP" in ddl


def test_role_column_is_not_enum(conn):
    """设计 §五 的裁决：role 不用 ENUM（加角色时 ENUM 要 ALTER TABLE）。"""
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM users LIKE 'role'")
        column_type = cur.fetchone()[1].lower()
    assert column_type.startswith("varchar"), f"role 列变成了 {column_type}"


def _column_facts(conn, name: str) -> dict:
    """一列的 SHOW COLUMNS 事实（Field/Type/Null/Key/Default/Extra），按名取值。"""
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM users LIKE %s", (name,))
        field, type_, nullable, key, default, extra = cur.fetchone()
    return {"type": type_, "nullable": nullable, "key": key, "default": default,
            "extra": extra}


def test_username_uniqueness_is_enforced_by_the_database(conn):
    """重名必须由**库**挡：UNIQUE 不在时重名账号会并存，登录按用户名取一行就变成
    「取到哪行看运气」——那是越权，且应用层的「先查后插」挡不住并发。"""
    username = _unique()
    try:
        users.create(conn, username, security.hash_password("pw-123456"),
                     "lawyer", "team-a")
        with pytest.raises(pymysql.err.IntegrityError):
            users.create(conn, username, security.hash_password("pw-654321"),
                         "partner", "team-b")
    finally:
        _forget(conn, [username])


def test_team_id_is_not_nullable_and_has_no_default(conn):
    """三层隔离的过滤依据不许缺席：team_id 允许 NULL 或给得出默认值时，一行账号
    可以在没有团队的情况下存在，而按团队过滤的谓词会**静默漏掉**它 —— 过滤「漏掉」
    与「查无此案」在行为上分不开，这正是隔离失效最难发现的形态。"""
    facts = _column_facts(conn, "team_id")
    assert facts["nullable"] == "NO", "team_id 允许 NULL 了"
    assert facts["default"] is None, f"team_id 有了默认值 {facts['default']!r}"
    # 光看表结构不够：真插一行缺 team_id 的，库必须当场拒（而不是塞个空串进去）
    with pytest.raises(pymysql.err.MySQLError):
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) "
                "VALUES (%s, %s, %s)", (_unique(), "x", "lawyer"))
        conn.commit()


def test_is_active_defaults_to_enabled(conn):
    """设计 §五 的 DEFAULT 1：新建的账号默认可用，停用是一次明确的动作。"""
    assert _column_facts(conn, "is_active")["default"] == "1"


# ---- 读写 ----

def test_create_then_read_lands_every_field_in_its_own_column(conn, make_account):
    """逐格比一遍：把 role 写进 team_id 那类串列错误，只查「用户名对不对」是看不出的，
    而这两格决定权限与隔离 —— 串了就是另一回事。"""
    username, user_id, password = make_account(role="partner", team_id="team-z")
    row = users.get_by_username(conn, username)
    assert row is not None, "create 之后按用户名读不到"
    assert row["id"] == user_id
    assert row["username"] == username
    assert row["role"] == "partner"
    assert row["team_id"] == "team-z"
    assert row["is_active"] == 1
    # created_at 不在 get_by_username 的列集合里（登录用不着它），改从 list_all 那一侧
    # 确认库侧默认值真填上了 —— 少了它，审计里「这个账号什么时候开的」就没了
    assert set(row) == {"id", "username", "password_hash", "role", "team_id",
                        "is_active"}, "get_by_username 的列集合变了"
    listed = [r for r in users.list_all(conn) if r["username"] == username][0]
    assert listed["created_at"] is not None, "created_at 没有库侧默认值"
    # 存的是哈希而不是明文：这一格是整张表最不能出事的地方
    assert row["password_hash"] != password
    assert security.verify_password(password, row["password_hash"]) is True


def test_two_accounts_get_distinct_ids(conn, make_account):
    """id 是审计行的外键（audit_log.user_id），两个账号共用一个 id 会让追溯串位。"""
    _, first_id, _ = make_account()
    _, second_id, _ = make_account()
    assert first_id != second_id


def test_unknown_username_returns_none(conn):
    """查无此人是 None 而不是抛：登录路径上它要与「口令不对」走同一个出口。"""
    assert users.get_by_username(conn, _unique()) is None


def _read_from_another_connection(username: str):
    """新开一条连接读这一行再关掉 —— 模拟**另一个会话**（登录端点每次另开连接）。

    两条讲究，缺一条这条判据就不成立：
    ①必须另开连接：pymysql 默认 autocommit=False，未提交的行在同一条连接里可见，
      所以「提交没提交」用同一条连接永远看不出来；
    ②每次读都**重新开**连接，不复用同一个对象：MySQL 默认隔离级别是 REPEATABLE
      READ，连接上第一次 SELECT 就定下了快照，之后本连接再也看不到别处的更新 ——
      复用会使「改口令之后读回」这一步永远读到旧值，测试红得莫名其妙（本次实测踩过）。
    """
    other = connect()
    try:
        return users.get_by_username(other, username)
    finally:
        other.close()


def test_writes_are_committed_and_visible_to_another_connection(conn, make_account):
    """必须提交：否则真实调用方（登录端点，走的是别的连接）根本看不见这个账号。"""
    username, _, password = make_account()
    assert _read_from_another_connection(username) is not None, "create 没提交"
    assert users.set_password(conn, username,
                              security.hash_password("pw-999999")) is True
    row = _read_from_another_connection(username)
    assert row is not None, "改口令之后读不到这一行"
    assert security.verify_password("pw-999999", row["password_hash"]), \
        "set_password 没提交"
    assert security.verify_password(password, row["password_hash"]) is False


def test_set_password_only_hits_the_target_account(conn, make_account):
    """改口令必须精确到人：UPDATE 少写 WHERE（或条件写错）会把全所的口令一起改掉，
    而返回值仍是 1，看起来完全正常。故拿第二个账号当场比对。"""
    target, _, old_password = make_account()
    bystander, _, bystander_password = make_account()
    assert users.set_password(conn, target,
                              security.hash_password("pw-999999")) is True
    target_row = users.get_by_username(conn, target)
    other_row = users.get_by_username(conn, bystander)
    assert security.verify_password("pw-999999", target_row["password_hash"])
    assert security.verify_password(old_password, target_row["password_hash"]) is False
    assert security.verify_password(bystander_password, other_row["password_hash"]), \
        "改口令波及到了别的账号"


def test_set_password_reports_missing_account_without_writing(conn):
    """操作对象不存在必须报 False —— 拼错一个字的 passwd 若返回成功，
    运维会以为口令已重置，而那个人下次登录仍然用旧口令。"""
    assert users.set_password(conn, _unique(),
                              security.hash_password("pw-999999")) is False


def test_set_active_toggles_and_only_the_target(conn, make_account):
    """停用/启用：目标那行真的翻面，别人不受影响，返回 False 表示没这个人。"""
    target, _, _ = make_account()
    bystander, _, _ = make_account()
    assert users.set_active(conn, target, False) is True
    assert users.get_by_username(conn, target)["is_active"] == 0
    assert users.get_by_username(conn, bystander)["is_active"] == 1
    assert users.set_active(conn, target, True) is True
    assert users.get_by_username(conn, target)["is_active"] == 1
    assert users.set_active(conn, _unique(), False) is False


def test_list_all_never_returns_the_password_hash(conn, make_account):
    """列表命令的列集合刻意不含哈希（少一次把哈希抄进终端缓冲/截图的机会）。
    这条用例钉住它 —— 加列是很容易的事，而这一列加回去没有任何症状。

    **键与值都要查**：只查键时，把 SELECT 的列集合换成含哈希的那一份（_COLUMNS）照样
    绿 —— zip 会把哈希塞进 role 那一格，键名一个没变而值全错位（实测本用例的初版就是
    这样漏的）。故逐行断言「没有任何一格的值长得像哈希」。
    """
    username, _, _ = make_account()
    rows = users.list_all(conn)
    assert username in {row["username"] for row in rows}
    mine = [row for row in rows if row["username"] == username][0]
    assert set(mine) == {"id", "username", "role", "team_id", "is_active",
                         "created_at"}
    for row in rows:
        assert "password_hash" not in row
        leaks = [key for key, value in row.items()
                 if isinstance(value, str) and value.startswith("scrypt$")]
        assert not leaks, f"列表里带出了哈希：{leaks}"
