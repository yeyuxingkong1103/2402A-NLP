# 运维脚本的测试：**真连 MySQL**（项目惯例），每个用例自建唯一用户名的账号并在
# 结束时删掉自己那几个。这里验的是脚本的行为（退出码、口令不走 argv、拒绝时不写库），
# 哈希与 token 的判据在 backend/tests/test_security.py 那边。
import argparse
import pathlib
import sys
import uuid

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import manage_users
from manage_users import main

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))
from app.core import security  # noqa: E402  （tools 的测试自己挂路径，见 test_check_env）
from app.db import users  # noqa: E402
from app.db.mysql import connect  # noqa: E402


def _username() -> str:
    return f"t-{uuid.uuid4().hex[:12]}"


def _reader(*passwords: str):
    """口令读取器的替身：按顺序回放给定的口令（脚本每次会读两次）。"""
    values = list(passwords)

    def read():
        return values.pop(0) if values else passwords[-1]

    return read


@pytest.fixture()
def conn():
    connection = connect()
    users.ensure_table(connection)
    yield connection
    connection.close()


@pytest.fixture()
def account(conn):
    """建号并登记待删用户名；用例结束后精确删除（断言失败的用例也要清干净）。"""
    created: list[str] = []

    def make(role: str = "lawyer", team_id: str = "team-a",
             password: str = "pw-123456"):
        username = _username()
        created.append(username)
        users.create(conn, username, security.hash_password(password), role, team_id)
        return username, password

    yield make
    if created:
        with conn.cursor() as cur:
            cur.executemany("DELETE FROM users WHERE username = %s",
                            [(name,) for name in created])
        conn.commit()


# 整棵命令树的选项**白名单**：每个解析器的选项集合必须恰好等于这里写的那个。
# 白名单而不是黑名单：黑名单天然漏项 —— 旧版断言「选项名里不含 pass 子串」，加一个
# `-p`（dest="password"）之后照绿（复审实测），而 argv 恰恰是本脚本唯一不许口令经过
# 的地方。集合相等还顺带钉住反向改动（悄悄删掉/改名一个既有选项）。
# 键是命令路径（顶层用 ""），值是 argparse 实际生成的选项串（含自动的 -h/--help）。
EXPECTED_OPTIONS = {
    "": {"-h", "--help"},
    "create": {"-h", "--help", "--username", "--role", "--team-id"},
    "passwd": {"-h", "--help", "--username"},
    "disable": {"-h", "--help", "--username"},
    "enable": {"-h", "--help", "--username"},
    "list": {"-h", "--help"},
}


def _option_sets() -> dict[str, set[str]]:
    """整棵命令树上每个解析器（顶层 + 各子命令）的选项名集合。

    这里顺带钉住一条集合比对**看不见**的事：树上不许有位置参数。位置参数的
    option_strings 是空列表，故「按 option_strings 收集再比集合」的白名单对它是盲的
    —— 复审实测：给 create 加一个 `add_argument("password", nargs="?")`，本用例仍
    14 passed，而 `parse_args([..., "SECRET-ON-ARGV"])` 真的把明文吃进了 namespace。
    位置参数恰恰是 argv 明文最容易的走私形态（不用记选项名、顺手就写了），也就是本
    用例要拦的那个威胁本身。唯一的例外是子解析器动作（顶层那个 command）：它只是把
    子命令名派发给对应解析器，本身不是收值的格子。
    """
    sets: dict[str, set[str]] = {}
    stack = [("", manage_users.build_parser())]
    while stack:
        path, parser = stack.pop()
        for action in parser._actions:
            # 除子解析器动作外，option_strings 为空 == 位置参数 == argv 明文入口：
            # 它收下的值绕过整个选项白名单直接进 args，本条判负
            assert action.option_strings or isinstance(action, argparse._SubParsersAction), \
                (f"{path or '顶层'} 出现位置参数 {action.dest!r}：argv 上的值可以不经"
                 f"任何选项名直接进库，口令禁止走这条路")
        sets[path] = {name for action in parser._actions
                      for name in action.option_strings}
        for action in parser._actions:
            # 子命令解析器的 choices 是「名字 → 解析器」的映射；其它 action 的 choices
            # 是普通值集合（--role 就是），不能当成子解析器往下走
            if isinstance(action.choices, dict):
                stack.extend(action.choices.items())
    return sets


def test_password_never_arrives_via_the_command_line():
    """本脚本唯一一条硬规矩：口令不走 argv —— argv 会同时进 shell 历史与进程列表
    （同机任何用户可读 /proc/<pid>/cmdline）两处明文，事后追不回。

    断言两层：①命令树的**形状** —— 树上只许有选项、不许有位置参数，且每个子命令的
    选项集合**恰好等于** EXPECTED_OPTIONS（白名单，理由见那里的注释）；②真按
    `--password` 传一个，argparse 必须拒。
    """
    assert _option_sets() == EXPECTED_OPTIONS, \
        f"命令树的选项集合变了，逐项裁决后再更新白名单：{_option_sets()}"
    with pytest.raises(SystemExit) as exc:
        main(["create", "--username", "x", "--role", "lawyer", "--team-id", "t",
              "--password", "secret-value"])
    assert exc.value.code == 2, "argparse 没拦住 --password"


def test_create_writes_a_usable_account(conn, account):
    """建成之后要真能用：角色/团队落对，且存的是能校验通过的口令哈希。"""
    username = _username()
    try:
        assert main(["create", "--username", username, "--role", "assistant",
                     "--team-id", "team-q"],
                    password_reader=_reader("pw-123456", "pw-123456")) == 0
        row = users.get_by_username(conn, username)
        assert row is not None, "脚本报成功却没建出账号"
        assert (row["role"], row["team_id"], row["is_active"]) == (
            "assistant", "team-q", 1)
        assert row["password_hash"] != "pw-123456", "明文进了库"
        assert security.verify_password("pw-123456", row["password_hash"]) is True
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
        conn.commit()


def test_create_refuses_duplicate_without_touching_the_existing_account(conn, account):
    """重名要拒，且**一个字节都不许动**原账号：若实现成「已存在就顺手改口令」，
    一次手滑的 create 会静默重置别人的口令，而调用方以为只是新建失败。"""
    username, password = account()
    before = users.get_by_username(conn, username)["password_hash"]
    code = main(["create", "--username", username, "--role", "partner",
                 "--team-id", "team-x"],
                password_reader=_reader("pw-999999", "pw-999999"))
    assert code == 1
    after = users.get_by_username(conn, username)["password_hash"]
    assert after == before, "重名建号顺手改了原账号的口令"
    assert security.verify_password("pw-999999", after) is False


def test_create_rejects_an_unknown_role(conn):
    """角色白名单来自 Role 枚举：命令行上给不出一个「脚本能建、鉴权不认」的角色。"""
    username = _username()
    with pytest.raises(SystemExit) as exc:
        main(["create", "--username", username, "--role", "root",
              "--team-id", "team-a"], password_reader=_reader("pw-123456"))
    assert exc.value.code == 2
    assert users.get_by_username(conn, username) is None, "被拒的命令仍然建了账号"


@pytest.mark.parametrize("command", ["create", "passwd"])
def test_short_password_rejected_without_writing(command, conn):
    """口令长度是策略、设在进库前的最后一格。不够长时**什么都不写**。

    这一条正是本轮实现踩出来的坑：策略原先写在 read_new_password 里，于是给 main
    注入一个替身读取器就把它整个跳过了（短口令照样建号）。故两个会写口令的命令都要验。
    """
    username = _username()
    if command == "passwd":
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (username, password_hash, role, team_id) "
                        "VALUES (%s, %s, %s, %s)",
                        (username, security.hash_password("pw-123456"), "lawyer",
                         "team-a"))
        conn.commit()
    try:
        before = (users.get_by_username(conn, username) or {}).get("password_hash")
        # 两个子命令的参数不同（passwd 只认 --username），故分别拼 argv
        argv = {"create": ["create", "--username", username, "--role", "lawyer",
                           "--team-id", "team-a"],
                "passwd": ["passwd", "--username", username]}[command]
        assert main(argv, password_reader=_reader("short")) == 1
        after = (users.get_by_username(conn, username) or {}).get("password_hash")
        assert after == before, "口令没通过策略检查，但库里被改动了"
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
        conn.commit()


def test_mismatched_confirmation_is_rejected():
    """两次输入不一致必须拒 —— 改口令输错一个字符，账号就成了一把打不开的锁，
    而库里只有哈希，没有任何入口能把口令读回来。

    这一条直接验交互函数：注入替身时被替换掉的本来就是「两次确认」这层协议
    （对值本身的策略在 check_password_policy，那条由上面两个命令级用例钉）。
    """
    answers = iter(["pw-123456", "pw-123457"])
    with pytest.raises(manage_users.UserInputError):
        manage_users.read_new_password(lambda _prompt: next(answers))


def test_matching_confirmation_returns_the_password():
    """正着走一遍：两次一致就交出口令（否则上面的用例可能因「恒抛」而恒绿）。"""
    answers = iter(["pw-123456", "pw-123456"])
    assert manage_users.read_new_password(lambda _prompt: next(answers)) == "pw-123456"


def test_passwd_changes_the_password(conn, account):
    """改口令：新口令能过、旧口令不能过。只验「新口令能过」会放过「两个都能过」
    这种更糟的实现（等于口令从没被改掉，却报告成功）。"""
    username, old = account()
    assert main(["passwd", "--username", username],
                password_reader=_reader("pw-999999", "pw-999999")) == 0
    stored = users.get_by_username(conn, username)["password_hash"]
    assert security.verify_password("pw-999999", stored) is True
    assert security.verify_password(old, stored) is False


def test_missing_account_returns_nonzero_on_every_mutating_command(conn):
    """拼错用户名时必须非零退出：静默报成功会让运维以为改完了。
    三个会写库的子命令逐个验 —— 漏一个就留一条「报成功但没改」。"""
    ghost = _username()
    for command in ("passwd", "disable", "enable"):
        assert main([command, "--username", ghost],
                    password_reader=_reader("pw-999999", "pw-999999")) == 1, command


def test_disable_and_enable_flip_is_active(conn, account):
    """停用是软删（不 DELETE 行）：审计行里的 user_id 要能指回一个存在过的人。

    每次读都**另开连接**：脚本自己开连接写，而 MySQL 默认 REPEATABLE READ —— 复用
    一条连接时，它第一次 SELECT 定下的快照会让后续读永远看不到脚本刚做的改动
    （本次实测踩过：enable 之后读回仍是被停用的那一版）。
    """
    username, _ = account()
    assert main(["disable", "--username", username]) == 0
    assert _read(username)["is_active"] == 0
    assert main(["enable", "--username", username]) == 0
    assert _read(username)["is_active"] == 1


def _read(username: str) -> dict:
    """新开一条连接读一行再关掉 —— 模拟登录端点那种「每次另开连接」的读法。"""
    other = connect()
    try:
        row = users.get_by_username(other, username)
    finally:
        other.close()
    assert row is not None, f"读不到 {username} 那一行"
    return row


def test_list_prints_accounts_but_never_hashes(conn, account, capsys):
    """列表输出里不许出现哈希（脚本刻意不查这一列）。终端回滚缓冲、截图、工单
    都会把输出带走 —— 哈希不是密钥，但它离口令只差一次离线爆破。"""
    username, _ = account()
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert username in out
    assert "scrypt$" not in out
    assert "password_hash" not in out


def test_no_subcommand_is_a_usage_error_not_success():
    """不给子命令时返回非零：静默返回 0 会让脚本里的拼错子命令看起来像执行成功。"""
    assert main([]) == 2


def test_database_failure_is_reported_without_a_traceback(monkeypatch, capsys):
    """连不上库是环境问题：一行人话 + 非零码。甩栈给运维看只会淹没真正的那一句。"""

    def boom(**_kwargs):
        raise RuntimeError("connect refused")

    monkeypatch.setattr(manage_users, "connect", boom)
    assert main(["list"]) == 1
    assert "连不上 MySQL" in capsys.readouterr().out
