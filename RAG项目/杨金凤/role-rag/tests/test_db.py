"""MySQL 元数据 CRUD 端到端测试（连 role_rag_test 独立库，标记 slow）。

运行：pytest -m slow tests/test_db.py -v
"""
import os
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

# 关键：db.py 的 engine / SessionLocal 是模块级单例，import 时即用 .env 的 MYSQL_URL 生成。
# 所以必须在 import db 之前，把 MYSQL_URL 的库名替换成独立测试库 role_rag_test。
# 选覆盖环境变量而非 monkeypatch db.engine 的理由：
#   db 的 CRUD 内部直接用全局 engine/SessionLocal，且 create_all 用 engine、get_db 用
#   SessionLocal——monkeypatch 要同时改三处并重建 sessionmaker，易漏；而只改一个
#   MYSQL_URL，db.py 照常初始化，零侵入、最不易出错。
load_dotenv()
_DEFAULT_URL = "mysql+pymysql://root:@127.0.0.1:3306/role_rag?charset=utf8mb4"
# 注意：str(URL) 会把密码掩码成 ***，必须用 render_as_string(hide_password=False)。
TEST_URL = (
    make_url(os.getenv("MYSQL_URL", _DEFAULT_URL))
    .set(database="role_rag_test")
    .render_as_string(hide_password=False)
)
os.environ["MYSQL_URL"] = TEST_URL

import db  # noqa: E402  # 必须在覆盖 MYSQL_URL 之后 import

import pytest  # noqa: E402


@pytest.mark.slow
def test_crud_roundtrip():
    """users / roles / sessions 各做一遍 插入→查询→更新→删除，finally 兜底清理。"""
    db.create_all()

    uid = rid = None
    uname = f"t_user_{uuid4().hex[:12]}"
    rname = f"t_role_{uuid4().hex[:12]}"
    sid = f"t_sess_{uuid4().hex[:12]}"
    new_uname = uname + "_renamed"

    try:
        # ---- 插入 ----
        user = db.create_user(uname)
        uid = user.id
        role = db.create_role(rname, "测试人设", "test_collection")
        rid = role.id
        sess = db.create_session(sid, user_id=uid, role_id=rid)
        assert user.id > 0 and role.id > 0 and sess.id > 0

        # ---- 查询 ----
        assert db.get_user_by_username(uname).id == uid
        assert db.get_role_by_name(rname).id == rid
        got = db.get_session(sid)
        assert got.user_id == uid and got.role_id == rid

        # ---- 更新 ----
        assert db.update_user(uid, new_uname).username == new_uname
        assert db.get_user_by_username(uname) is None
        assert db.get_user_by_username(new_uname).id == uid
        before = db.get_session(sid).last_active_at
        assert db.touch_session(sid).last_active_at >= before

        # ---- 删除 ----
        assert db.delete_session(sid) is True
        assert db.get_session(sid) is None
        assert db.delete_role(rid) is True
        assert db.get_role(rid) is None
        assert db.delete_user(uid) is True
        assert db.get_user(uid) is None
        uid = rid = None  # 已清理干净，finally 无需再删
    finally:
        # 兜底清理（按 FK 依赖顺序：session → role → user），断言失败也不留数据。
        try:
            db.delete_session(sid)
        except Exception:
            pass
        if rid is not None:
            try:
                db.delete_role(rid)
            except Exception:
                pass
        if uid is not None:
            try:
                db.delete_user(uid)
            except Exception:
                pass
