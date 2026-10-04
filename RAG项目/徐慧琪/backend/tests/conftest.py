# 全测试套件共用的环境前置（任务 6）。
#
# 为什么需要一个 conftest：限流加盐值**没有默认值**（设计 §六 的「不存 IP 原文」只有
# 配了盐才成立，见 config.load_rate_salt 的注释），而限流中间件挂在所有公开端点前面
# —— 不设它的话，任何打 /public/qa、/lawyers/recommend、/law/* 的用例都会以
# MissingRateSaltError → 500 的形态失败（丢的是 500，不是 429），失败原因指向限流
# 而不是那条用例真正要判的东西。
#
# 收成 autouse 一条而不是逐个文件 monkeypatch.setenv：公开端点散在 test_api_public /
# test_api_law / test_api_lawyer / test_api_surface / test_main_ratelimit 五个文件里，
# 一处一处抄的话，将来加一条打公开端点的用例而忘了设盐，表现是那条用例 500 —— 而
# 「配置故障」与「护栏被改坏」在红的时候长得一模一样。
#
# 值取 _fakes_http.RATE_SALT（与 JWT_SECRET 并列的共用件）；需要**故意缺盐**的用例
# 在自己的用例体里 monkeypatch.delenv(ENV_RATE_LIMIT_SALT) —— 用例体里的改动发生在
# 本夹具之后，能盖掉这里的设置。
import pytest

from app.core import config, security
from app.db import audit as audit_db
from app.db.mysql import connect
from tests._fakes_http import JWT_SECRET, RATE_SALT


@pytest.fixture(autouse=True)
def _rate_limit_salt(monkeypatch):
    """给每个用例备好限流加盐值。只有缺盐用例会显式删掉它。"""
    monkeypatch.setenv(config.ENV_RATE_LIMIT_SALT, RATE_SALT)


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    """给每个用例备好签名密钥（I-5 起 lifespan 启动自检要求它）。

    与缺盐夹具同形：设置在用例体之前，需要**故意缺密钥**的用例在自己体内
    monkeypatch.delenv 盖掉它。值取 _fakes_http.JWT_SECRET（共用件）—— 逐文件
    各设一份的写法在启动自检面前不再成立：漏设的文件会整片红在启动期。
    """
    monkeypatch.setenv(security.JWT_SECRET_ENV, JWT_SECRET)


@pytest.fixture(scope="session", autouse=True)
def _remove_audit_rows_written_by_the_suite():
    """套件结束时清掉**本次运行新落的**审计行（任务 7 起才需要）。

    为什么需要：审计中间件给每条业务请求写一行，而真连库的 HTTP 用例（test_api_law
    的条文/导航、test_api_deps 的真库登录、test_audit / test_api_admin）会因此往
    audit_log 里落行 —— 单靠各用例按 request_id 清理覆盖不到「别的文件里那些请求
    顺带写下的行」。这里按 id 上界一次性收尾，让「测试不留垃圾」在审计表上也成立。

    前提写在明面上：删的是「套件运行期间写进这台开发库 audit_log 的全部行」。若
    同时有一台**真服务**在写同一个库，它这段时间的审计行也会被删 —— 本项目当前
    没有常驻服务（人工验收时才起），故接受；更保险的做法是给测试单开一个 database，
    那属部署层的事。MySQL 不在线时不打扰（真连库的用例自己会跳过）。
    """
    try:
        conn = connect()
        audit_db.ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(id), 0) FROM audit_log")
            high = int(cur.fetchone()[0])
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM audit_log WHERE id > %s", (high,))
            conn.commit()
        finally:
            conn.close()
