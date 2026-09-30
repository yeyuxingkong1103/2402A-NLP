"""运维脚本：所内账号的创建、改口令、停用/启用与列举（设计 §一：本轮账号由运维创建）。

**口令不接受命令行参数**，这是本脚本唯一一条硬规矩：argv 会进两处明文 —— shell 历史
（bash_history / PSReadLine）与进程列表（同机任何用户都能读 /proc/<pid>/cmdline）。
`--password x` 等于把口令同时抄进两个日志，而且事后追不回（历史已被备份、截图）。
故一律走 getpass 交互读（不回显），口令只存在于内存与库里的 scrypt 哈希中。

用法：
  cd D:/xinzg6/fl
  python tools/manage_users.py create --username zhangsan --role lawyer --team-id team-1
  python tools/manage_users.py passwd  --username zhangsan
  python tools/manage_users.py disable --username zhangsan
  python tools/manage_users.py list
"""
from __future__ import annotations

import argparse
import getpass
import pathlib
import sys

# 本脚本在 tools/ 下而 app 包在 backend/ 下：先挂路径才能 import app.*
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

import pymysql  # noqa: E402  （路径挂好后才能用；它随 app.db.mysql 一起进来）

from app.core import security  # noqa: E402
from app.core.config import load as load_config  # noqa: E402
from app.core.security import Role  # noqa: E402
from app.db import users  # noqa: E402
from app.db.mysql import connect  # noqa: E402

# 本机控制台是 GBK，直接 print 中文会抛 UnicodeEncodeError。显式改 UTF-8 并容忍坏字节：
# 本脚本的输出是给人看的操作结果，不该因为一行编不出来就整个崩掉（up.py 同款处理）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 口令长度门槛定在这里而不是 security.py：那是「机制」（怎么哈希、怎么比），
# 这是「策略」（多长算够），而策略要放在能给人反馈的地方 —— 命令行工具的入口。
MIN_PASSWORD_LEN = 8


class UserInputError(Exception):
    """交互输入不合法。由 main 翻成一行人话 + 退出码，不把栈甩给运维。"""


def check_password_policy(password: str) -> None:
    """口令策略的唯一落点：不论口令从哪条路来，进库前都得过这里。

    为什么独立成一个函数、而不是写在 read_new_password 里：策略挂在**交互**那条路上
    时，任何绕过交互的调用方（测试替身、将来加的非交互入口）都会把策略整个静默跳过。
    本次实现时就踩到过这一点 —— 给 main 注入替身读取器时，短口令照样建出了账号。
    """
    if len(password) < MIN_PASSWORD_LEN:
        raise UserInputError(f"口令至少 {MIN_PASSWORD_LEN} 个字符")


def read_new_password(prompt_fn=None) -> str:
    """交互读两次新口令并核对。prompt_fn 可注入 —— 单测里不必真开终端。

    读两次不是形式：改口令时输错一个字符，账号就成了一把打不开的锁，而运维手上
    没有任何入口能再把口令读出来（库里只有哈希）。这个确认环是唯一的纠错机会。
    注意这一层只管**交互协议**（两次一致）；对值本身的判断在 check_password_policy，
    它不随读取方式改变 —— 注入替身替换掉的是协议，不该连策略一起替换。
    """
    ask = getpass.getpass if prompt_fn is None else prompt_fn
    first = ask("新口令（不回显）：")
    if first != ask("再输一次："):
        raise UserInputError("两次输入不一致，未做任何改动")
    return first


def _password_for_storage(password_reader) -> str:
    """取口令并立刻过策略 —— 建号与改口令的唯一来路。"""
    password = password_reader()
    check_password_policy(password)
    return password


def cmd_create(conn, args, password_reader) -> int:
    """建账号。重名当场拒绝并保持原账号**一个字节都不动**（见下面的捕获）。"""
    password = _password_for_storage(password_reader)
    try:
        new_id = users.create(conn, args.username,
                              security.hash_password(password),
                              args.role, args.team_id)
    except pymysql.err.IntegrityError:
        # 不「已存在就改成新口令」：那会让一次手滑的 create 静默重置别人的口令，
        # 而调用方以为只是新建失败。要改口令有 passwd 这条明确的路。
        print(f"用户名已存在：{args.username}（未做任何改动；改口令请用 passwd）")
        return 1
    print(f"已创建：{args.username}（id={new_id}，角色 {args.role}，"
          f"团队 {args.team_id}）")
    return 0


def cmd_passwd(conn, args, password_reader) -> int:
    """改口令。账号不存在时返回非零 —— 拼错用户名的操作必须能被看出来。"""
    password = _password_for_storage(password_reader)
    if not users.set_password(conn, args.username, security.hash_password(password)):
        print(f"没有这个账号：{args.username}（未做任何改动）")
        return 1
    print(f"已改口令：{args.username}")
    return 0


def cmd_toggle(conn, args, password_reader) -> int:
    """停用/启用（软删）。停用的效果见 security.TOKEN_TTL_S 的注释：立刻挡住新登录，
    但已签发的 token 在有效期（默认 8 小时）内仍然有效。"""
    if not users.set_active(conn, args.username, args.active):
        print(f"没有这个账号：{args.username}（未做任何改动）")
        return 1
    print(f"已{'启用' if args.active else '停用'}：{args.username}")
    return 0


def cmd_list(conn, args, password_reader) -> int:
    """列举账号。**不含 password_hash** —— 运维不需要它，少一次把哈希抄进终端缓冲。"""
    rows = users.list_all(conn)
    if not rows:
        print("（没有任何账号）")
        return 0
    print(f"{'id':>4}  {'用户名':<20} {'角色':<10} {'团队':<16} {'启用':<4} 创建时间")
    for row in rows:
        print(f"{row['id']:>4}  {row['username']:<20} {row['role']:<10} "
              f"{row['team_id']:<16} {'是' if row['is_active'] else '否':<4} "
              f"{row['created_at']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """命令表。--role 的候选直接取自 Role 枚举 —— 角色清单只有那一处真相源。"""
    parser = argparse.ArgumentParser(description="所内账号运维（本轮无管理界面）")
    sub = parser.add_subparsers(dest="command")
    create = sub.add_parser("create", help="新建账号（口令交互输入）")
    create.add_argument("--username", required=True)
    # choices 取枚举值：加角色时这里自动跟着变，不会出现「脚本能建、鉴权不认」的角色
    create.add_argument("--role", required=True, choices=[r.value for r in Role])
    # team_id 必填且无默认值：它是数据层隔离的唯一来源，一个默认团队等于没有隔离
    create.add_argument("--team-id", required=True, dest="team_id")
    create.set_defaults(func=cmd_create)
    for name, active, func in (("passwd", None, cmd_passwd),
                               ("disable", False, cmd_toggle),
                               ("enable", True, cmd_toggle),
                               ("list", None, cmd_list)):
        command = sub.add_parser(name)
        command.set_defaults(func=func, active=active)
        if name != "list":
            command.add_argument("--username", required=True)
    return parser


def main(argv: list[str] | None = None, password_reader=None) -> int:
    """解析、连库、派发，返回进程退出码（0 成功）。password_reader 可注入供单测用。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # 不带子命令时给用法而不是当作成功 —— 静默返回 0 会让脚本里的
        # `manage_users.py create ...`（写错子命令名）看起来像建好了账号
        parser.print_help()
        return 2
    reader = read_new_password if password_reader is None else password_reader
    try:
        # 走 core.config 取连接参数：FL_MYSQL_* 的覆盖在运维脚本里也必须生效，
        # 否则「服务连的库」与「建账号的库」可以是两台机器（而这不会报任何错）
        conn = connect(**load_config().mysql)
    except Exception as exc:  # 连不上是环境问题：一行人话 + 非零码，不甩栈
        print(f"连不上 MySQL：{type(exc).__name__}: {exc}")
        return 1
    try:
        # 建表（幂等）后再动数据：全新环境上第一条 create 不该以「表不存在」告终
        users.ensure_table(conn)
        return args.func(conn, args, reader)
    except UserInputError as exc:
        print(f"输入不合法：{exc}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
