"""创建管理员 CLI（批次7-1）。

背景：users 表 is_admin=1 的行数为 0，审核发布接口（阶段6 的
/api/v1/admin/documents）在真实环境里没人能调用。本 CLI 补上
"把已注册用户升级为管理员"的唯一入口。

用法：
    python -m app.cli.create_admin --email someone@example.com

行为契约（用户批复）：
- 该 email 已注册 → 置 is_admin=1 并打印结果
- 未注册 → 明确报错"该邮箱尚未注册，请先在页面注册"，禁止静默创建账号
- 幂等：已是管理员时重复执行，提示"已是管理员"，退出码 0
- 不打印密码哈希等任何敏感值（本 CLI 根本不读取 password_hash）

文档说明由用户在 docs/部署文档.md 维护（本模块不写 docs）。
"""

import argparse
import sys

from sqlalchemy.orm import Session, sessionmaker

from app.db.sql_models import User

# promote_to_admin 的两种结果标记（测试断言 + CLI 输出共用）
STATUS_PROMOTED = "promoted"
STATUS_ALREADY_ADMIN = "already_admin"


def promote_to_admin(session: Session, email: str) -> str:
    """把指定邮箱的已注册用户升级为管理员（幂等）。

    参数：
    - session: SQLAlchemy 会话（调用方负责提交/关闭的由本函数统一 commit）
    - email: 目标邮箱（大小写不敏感，内部归一为小写）

    返回：
    - STATUS_PROMOTED：本次成功置 is_admin=1
    - STATUS_ALREADY_ADMIN：该用户已是管理员（幂等，不做任何写入）

    异常：
    - ValueError：邮箱未注册（CLI 层转成用户可读的错误输出与非零退出码）
    """
    normalized = email.strip().lower()
    record = session.query(User).filter_by(email=normalized).one_or_none()
    if record is None:
        # 明确报错，绝不静默创建账号（管理员必须是已注册用户）
        raise ValueError(f"该邮箱尚未注册，请先在页面注册：{normalized}")
    if record.is_admin:
        # 幂等：重复执行不报错、不写入
        return STATUS_ALREADY_ADMIN
    record.is_admin = True
    session.commit()
    return STATUS_PROMOTED


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析 --email，执行升级，打印用户可读结果。"""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.create_admin",
        description="把已注册用户升级为管理员（幂等；不创建新账号）",
    )
    parser.add_argument(
        "--email",
        required=True,
        help="目标用户的注册邮箱（必须是页面上已注册的账号）",
    )
    args = parser.parse_args(argv)

    # 会话工厂延迟导入：保证 --help 等纯参数分支不触发数据库连接
    from app.chat.chat_runtime import build_default_session_factory

    session_factory: sessionmaker = build_default_session_factory()
    try:
        status = promote_to_admin(session_factory(), args.email)
    except ValueError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    # 短命 CLI 进程退出时由解释器回收连接池，无需手动关闭

    if status == STATUS_PROMOTED:
        print(f"[成功] 已将 {args.email.strip().lower()} 升级为管理员（is_admin=1）")
    else:
        print(f"[跳过] {args.email.strip().lower()} 已是管理员，无需重复操作")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
