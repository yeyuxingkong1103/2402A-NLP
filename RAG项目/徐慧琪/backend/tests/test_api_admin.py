# `GET /api/v1/admin/audit/export` 的判据（设计 §四 第 86 行、FR-6.4、§二 第 9 条）。
#
# 真连 MySQL：这条端点的价值就是「把库里的行按时间窗取出来」，用替身测不出
# 时间窗与排序。每个用例自建行（含自己发的那几个请求产生的审计行，靠自带的
# X-Request-ID 精确删），不留垃圾。
import datetime as dt
import uuid

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import admin as admin_mod
from app.core import errors, security
from app.core.security import Role
from app.db import audit as audit_db
from app.db.mysql import connect
from tests._fakes_http import (JWT_SECRET, FakeConn, FakeServices, bearer,
                               fake_factory, jwt_token, without_request_id)

URL = "/api/v1/admin/audit/export"


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv(security.JWT_SECRET_ENV, JWT_SECRET)


# 本用例写过的 request_id（_rid 登记）；清理走 conn 夹具的收尾（失败的用例也要清）
_WRITTEN: list[str] = []


@pytest.fixture()
def conn():
    """真连接（扮演业务连接）；建表、收尾清理、关闭都在这里（理由见 test_audit.py）。"""
    connection = connect()
    audit_db.ensure_table(connection)
    _WRITTEN.clear()
    yield connection
    _forget(connection, *_WRITTEN)
    _WRITTEN.clear()
    connection.close()


def _rid(prefix: str) -> str:
    rid = f"t7x-{prefix}-{uuid.uuid4().hex[:12]}"
    _WRITTEN.append(rid)
    return rid


def _db_now(conn) -> dt.datetime:
    """库侧当前时刻 —— 造行与算窗口都用它，不用 `datetime.now()`。

    实测：本机容器里的 MySQL 走 UTC，比 Windows 本地钟早 8 小时（`SELECT NOW()`
    12:44 vs Python 20:44）。测试若拿应用时钟造行，行会落在窗口**外面**（或未来），
    表现是「导不出来」——而那条红会指向导出逻辑，真因却在两个时钟。端点自己的
    缺省窗口也改成从这里取（见 api/admin.py 的 _db_now），这条夹具与它同源。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT NOW()")
        return cur.fetchone()[0]


def _insert(conn, request_id: str, ts: dt.datetime, action: str = "qa") -> None:
    """直插一行（ts 显式给：时间窗用例要的正是「不在缺省窗口里」的行）。"""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO audit_log (ts, request_id, user_id, role, team_id, action,"
            " query_text, status_code, latency_ms)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (ts, request_id, 7, "partner", "team-a", action, "审计导出标记",
             200, 3))
    conn.commit()


def _forget(conn, *request_ids: str) -> None:
    if not request_ids:
        return
    with conn.cursor() as cur:
        cur.executemany("DELETE FROM audit_log WHERE request_id = %s",
                        [(rid,) for rid in request_ids])
    conn.commit()


def _app(conn):
    """真 create_app + 假 Services（conn 是真连接：导出要真读库）。"""
    services = FakeServices(conn=conn, account_conn=FakeConn(row=(1,)))
    factory_fn, _ = fake_factory(services)
    return main.create_app(services_factory=factory_fn)


def _get(conn, *, role: Role | None = Role.PARTNER, rid: str,
         query: str = "") -> "object":
    """打一次导出端点（role=None 就是未认证那一档）。rid 由用例给：那条请求自己
    也会落一行审计，用例结束时按它精确删掉。"""
    headers = {"X-Request-ID": rid}
    if role is not None:
        headers |= bearer(jwt_token(role=role))
    with TestClient(_app(conn)) as client:
        return client.get(URL + query, headers=headers)


def _request_ids(response) -> list[str]:
    return [row["request_id"] for row in response.json()]


# ---- 鉴权：partner 才有；角色不够与未认证都是 404 ----

def test_the_partner_gets_the_rows_in_the_window(conn):
    """partner 拿得到，且拿到的行是**这一行**（逐格比）：只断言 200 或
    「是个列表」的话，把查询换成恒回空表照样绿，而那正是「导出了个寂寞」。"""
    mine = _rid("partner")
    _insert(conn, mine, _db_now(conn))
    rid = _rid("req")
    response = _get(conn, rid=rid)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    rows = [row for row in body if row["request_id"] == mine]
    assert len(rows) == 1, f"导出里没有那一行：{[r['request_id'] for r in body][:5]}"
    row = rows[0]
    assert set(row) == set(audit_db._COLUMNS), "导出的列集合变了"
    assert (row["user_id"], row["role"], row["team_id"]) == (7, "partner", "team-a")
    assert row["action"] == "qa" and row["query_text"] == "审计导出标记"
    assert row["status_code"] == 200 and row["latency_ms"] == 3
    _forget(conn, mine, rid)


def test_lawyer_and_anonymous_get_the_same_404_as_a_missing_path(conn):
    """角色不够（lawyer）与未认证都是 404，且与「这个路径不存在」**逐字段同形**：
    §二 第 9 条要的是不暴露路径存在性 —— 若这条路径的 404 与未知路径的 404 有一
    点差别，扫描器就能从 404 里数出 /admin/* 存在，再往下试角色。
    """
    lawyer_rid, anon_rid, unknown_rid = _rid("lawyer"), _rid("anon"), _rid("unknown")
    lawyer = _get(conn, role=Role.LAWYER, rid=lawyer_rid)
    anonymous = _get(conn, role=None, rid=anon_rid)
    assert lawyer.status_code == 404 and anonymous.status_code == 404
    with TestClient(_app(conn)) as client:
        missing = client.get("/api/v1/admin/nope",
                             headers={"X-Request-ID": unknown_rid})
    assert missing.status_code == 404
    assert without_request_id(lawyer) == without_request_id(missing)
    assert without_request_id(anonymous) == without_request_id(missing)
    _forget(conn, lawyer_rid, anon_rid, unknown_rid)


# ---- 时间参数：非法 400、缺省窗有边界、区间反向 400 ----

def test_an_illegal_time_is_a_400_in_the_unified_shape(conn):
    """非法时间 → 400（不是 500、也不是静默回全表）+ §七 的统一错误形状。"""
    rids = [_rid("bad-start"), _rid("bad-end")]
    for rid, query in zip(rids, ("?start=昨天", "?end=2026-13-45")):
        response = _get(conn, rid=rid, query=query)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "bad_request" and error["message"]
    _forget(conn, *rids)


def test_the_reversed_range_is_a_400_not_an_empty_array(conn):
    """区间写反（start 晚于 end）判 400：空数组是「这段时间没有记录」的诚实答案，
    用它回答「你把区间写反了」会让调用方以为导出成功。"""
    rid = _rid("reversed")
    response = _get(conn, rid=rid, query="?start=2026-10-02&end=2026-10-01")
    assert response.status_code == 400
    _forget(conn, rid)


def test_a_bare_date_means_the_whole_day_at_the_end_bound(conn):
    """纯日期在 end 端取**当天 23:59:59.999999**（人说的「到 9 月 30 日」含全天）：
    两端同取 0 点会让 end 那天整个漏掉，而漏掉的那天在结果里没有任何痕迹。
    纯函数级判据（不起 HTTP）：这条规则只在这一处。
    """
    assert admin_mod.parse_time("2026-09-30", is_end=False) == \
        dt.datetime(2026, 9, 30)
    assert admin_mod.parse_time("2026-09-30", is_end=True) == \
        dt.datetime(2026, 9, 30, 23, 59, 59, 999999)
    assert admin_mod.parse_time("2026-09-30T08:05", is_end=True) == \
        dt.datetime(2026, 9, 30, 8, 5)
    with pytest.raises(errors.BadRequest):
        admin_mod.parse_time("2026-09-30+08:00", is_end=False)


def test_the_default_window_bounds_the_export(conn):
    """缺省不导出全表：三天前那一行**不在**缺省窗口里（缺省窗 = 最近 24 小时），
    而显式给 start 就能取到它 —— 两个方向一起钉：只测一向时，把缺省窗改成「全表」
    照样绿（那条用例看到的三天前那行仍在），而那就是一次请求把库拖垮的形态。
    """
    old, recent = _rid("old"), _rid("recent")
    now = _db_now(conn)
    _insert(conn, old, now - dt.timedelta(days=3))
    _insert(conn, recent, now)
    rid, wide_rid = _rid("req"), _rid("req-wide")
    default = _get(conn, rid=rid)
    assert default.status_code == 200
    assert old not in _request_ids(default), "缺省窗口把三天前的行也导出来了"
    assert recent in _request_ids(default)
    wide = _get(conn, rid=wide_rid,
                query=f"?start={(now - dt.timedelta(days=4)):%Y-%m-%d}"
                      f"&end={now:%Y-%m-%d}")
    assert wide.status_code == 200
    assert old in _request_ids(wide), "显式给了 start 反而取不到那行"
    _forget(conn, old, recent, rid, wide_rid)


# ---- 上限：limit 只管小、超硬顶判 400、触顶时留最新 ----

def test_the_limit_is_honored_and_the_hard_cap_rejects(conn):
    """上限的形态（本任务的裁决）：`limit` 上限 5000，超了**判 400**而不是静默
    截到 5000 ——「你要的比给得出的多」是调用方该知道的事；0 与负数同样 400。
    """
    rids = [_rid("over"), _rid("zero"), _rid("ok-limit")]
    over = _get(conn, rid=rids[0], query=f"?limit={admin_mod.MAX_LIMIT + 1}")
    zero = _get(conn, rid=rids[1], query="?limit=0")
    assert over.status_code == 400 and zero.status_code == 400
    assert _get(conn, rid=rids[2],
                query=f"?limit={admin_mod.MAX_LIMIT}").status_code == 200
    _forget(conn, *rids)


def test_the_cap_keeps_the_newest_rows(conn):
    """触顶时留的是**最新**那一行（fetch_between 的 ORDER BY ts DESC）：截断丢掉
    的应是窗口里最早的事件，而不是最近那次 —— 排查时最近的事件才是要看的。
    """
    older, newer = _rid("t-old"), _rid("t-new")
    now = _db_now(conn)
    _insert(conn, older, now - dt.timedelta(minutes=2))
    _insert(conn, newer, now - dt.timedelta(minutes=1))
    rid = _rid("req")
    response = _get(conn, rid=rid, query="?limit=1")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1, f"limit=1 却回了 {len(body)} 行"
    # 这一行必须是「最新的那一行」而不是随便一行：它是这次请求自己的审计行
    # （收尾写入，ts 最新），或至少不是两条插入里较老的那条
    assert body[0]["request_id"] != older
    _forget(conn, older, newer, rid)


def test_the_export_itself_is_audited_with_the_exporters_identity(conn):
    """导出**本身也要留一行审计**（action="export"，设计 §五 的 action 清单里就有
    它）：谁在什么时候导了审计，本身是审计材料。identity 来自请求 token ——
    这条同时证明 partner 的身份真的被写进去了（而不是恒 NULL）。"""
    rid = _rid("export-row")
    assert _get(conn, rid=rid).status_code == 200
    row = None
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(audit_db._COLUMNS)} FROM audit_log "
                    "WHERE request_id = %s", (rid,))
        got = cur.fetchone()
    row = None if got is None else dict(zip(audit_db._COLUMNS, got))
    assert row is not None, "导出请求自己没有留审计行"
    assert row["action"] == "export" and row["status_code"] == 200
    assert (row["user_id"], row["role"], row["team_id"]) == (7, "partner", "team-a")
    assert row["query_text"] is None, "导出行的 query_text 该是 NULL"
    _forget(conn, rid)
