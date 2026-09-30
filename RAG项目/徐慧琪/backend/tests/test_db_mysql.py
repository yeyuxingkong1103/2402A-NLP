# MySQL 模块的单测。需要 MySQL 在线——离线时整文件跳过，
# 因为本模块的价值就在"连得上、表建得对"，用 mock 测等于没测。
import threading

import pytest

from app.db.mysql import (
    DEFAULT_CONFIG, SCHEMA_PATH, _split_statements, apply_schema, connect,
    connect_autocommit, insert_many,
)

# 并发用例用的查询：与 healthz 探针同一句真 SQL（编排层会并发打的就是它）。
# 但走的是通用的 conn.cursor()，**不是**探针函数 —— 保护若只包在探针里，下面的
# 用例必须能红（这是「护栏得钉在被测路径上」的那条要求）。
PROBE_SQL = "SELECT 1 FROM article LIMIT 1"


def _mysql_available() -> bool:
    # 连一下就知道；connect_timeout 已设 3 秒，离线时不会拖慢收集
    try:
        connect().close()
        return True
    except Exception:
        return False


requires_mysql = pytest.mark.skipif(not _mysql_available(), reason="MySQL 未在线")


def test_split_statements_keeps_statement_after_comment():
    # 注释与语句往往在同一段里（"-- 说明\nCREATE TABLE ..."），
    # 若按"以 -- 开头的整句跳过"来写，会把整条建表语句一起丢掉
    sql = "-- 说明\nCREATE TABLE a (id INT);\n\n-- 另一段\nCREATE TABLE b (id INT);"
    stmts = _split_statements(sql)
    assert len(stmts) == 2
    assert stmts[0].startswith("CREATE TABLE a")
    assert stmts[1].startswith("CREATE TABLE b")


def test_split_statements_ignores_blank_and_comment_only_tail():
    sql = "CREATE TABLE a (id INT);\n-- 结尾只有注释\n"
    assert len(_split_statements(sql)) == 1


def test_default_config_uses_utf8mb4():
    # 法条含生僻字与全角标点，utf8 三字节存不下
    assert DEFAULT_CONFIG["charset"] == "utf8mb4"


def test_schema_path_points_at_deploy():
    # 路径写错会让 apply_schema 读不到文件——这条挡的是拼路径时的低级错
    assert SCHEMA_PATH.name == "schema.sql"
    assert SCHEMA_PATH.parent.name == "mysql"
    assert SCHEMA_PATH.exists()


@requires_mysql
def test_apply_schema_creates_three_tables():
    conn = connect()
    try:
        apply_schema(conn, SCHEMA_PATH)
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    assert {"law", "law_version", "article"} <= tables


@requires_mysql
def test_apply_schema_is_idempotent():
    # 重复执行必须安全，否则重跑入库会炸在建表这一步
    conn = connect()
    try:
        apply_schema(conn, SCHEMA_PATH)
        apply_schema(conn, SCHEMA_PATH)
    finally:
        conn.close()


@requires_mysql
def test_insert_many_returns_row_count():
    conn = connect()
    try:
        apply_schema(conn, SCHEMA_PATH)
        n = insert_many(conn, "law",
                        ["law_id", "law_name"],
                        [("__test_law__", "单测占位法")])
        assert n == 1
        # 必须读回来验：insert_many 的返回值是它自己算的 len(rows)，
        # 只断言它等于 1 的话，即使 executemany 一行没写库测试也照样绿
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM law WHERE law_id = %s", ("__test_law__",))
            assert cur.fetchone()[0] == 1
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM law WHERE law_id = %s", ("__test_law__",))
        conn.commit()
        conn.close()


@requires_mysql
def test_insert_many_empty_rows_is_noop():
    # 空列表不该拼出 "VALUES ()" 这种非法 SQL
    conn = connect()
    try:
        assert insert_many(conn, "law", ["law_id", "law_name"], []) == 0
    finally:
        conn.close()


def _hammer(conn, *, threads: int, per_thread: int) -> list[str]:
    """在**同一个**连接上真并发跑 threads × per_thread 次查询，返回失败信息。

    Barrier 让各线程尽量同时冲进去，把竞争窗口放大：并发用例的价值就在「撞上」，
    串行跑 n 次永远绿。
    """
    failures: list[str] = []
    start = threading.Barrier(threads)

    def worker() -> None:
        start.wait()
        for _ in range(per_thread):
            try:
                with conn.cursor() as cur:
                    cur.execute(PROBE_SQL)
                    cur.fetchone()
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return failures


@requires_mysql
def test_concurrent_queries_on_one_connection_never_corrupt_the_protocol():
    """同一连接被多线程共用时，每次查询都必须成功（任务 3 复审的 Critical）。

    判据是**零失败**，不是「有成功的」：改前实测 2 线程 × 60 次成功 119 / 失败 1，
    8 线程 × 20 次成功 5 / 失败 155 —— 任何「至少成功一次」的写法都恒绿。
    失败形态是协议级损坏（InterfaceError: (0, '')、Packet sequence number wrong、
    PyMemoryView_FromBuffer 空指针），而且连接停在损坏状态、后续请求一起挂。
    """
    for threads, per_thread in ((2, 60), (8, 20)):
        conn = connect()
        try:
            failures = _hammer(conn, threads=threads, per_thread=per_thread)
        finally:
            conn.close()
        assert failures == [], f"{threads} 线程 × {per_thread} 次后连接坏了：{failures[:3]}"


@requires_mysql
def test_a_cursor_holds_the_connection_until_it_is_closed():
    """游标没关之前，别的线程查不进去 —— 判据是「排队」而不是「没炸」。

    与上面的压测分工不同：压测证明**结果**对，这条证明**机制**在（锁真的在
    cursor→close 之间持有）。只有压测的话，「保护只包了探针」「锁在某些时序下才
    生效」这类改动可能靠时序蒙混过关。判「还没结束」用 join(timeout) 而不是 sleep
    一个魔数：慢机器上也不会误判成红。
    """
    conn = connect()
    holder = conn.cursor()  # 故意开着不关：锁要到它关闭才还
    entered = threading.Event()
    outcome: list[str] = []

    def other() -> None:
        entered.set()
        try:
            with conn.cursor() as cur:
                cur.execute(PROBE_SQL)
                cur.fetchone()
        except Exception as exc:  # noqa: BLE001 —— 记下来给主线程断言，不在子线程里抛
            outcome.append(f"{type(exc).__name__}: {exc}")

    blocked = threading.Thread(target=other)
    try:
        blocked.start()
        entered.wait(timeout=5)
        blocked.join(timeout=0.5)
        assert blocked.is_alive(), "游标还开着，另一个线程却查完了 —— 连接上没有串行化"
    finally:
        # 失败路径上更要放锁：不然这个线程会一直等着，整个测试进程挂死在这里
        holder.close()
    blocked.join(timeout=5)
    assert not blocked.is_alive(), "游标关掉后另一个线程还是没查成"
    assert outcome == [], f"排队等的那个查询失败了：{outcome}"


@requires_mysql
def test_connect_supports_the_with_statement_like_pymysql():
    """`with connect() as conn:` 必须能用，且退出即关连接（与裸 pymysql 同语义）。

    这条钉的是**代理对协议的忠实度**：`connect()` 的契约是「调用方写法不变」，
    而 pymysql 的 Connection 是上下文管理器。实测漏过一次 —— 加并发代理时忘了
    接 `__enter__`/`__exit__`，`tools/check_env.py` 的 mysql 项当场整项 FAIL
    （TypeError），而套件全绿：check_env 不被任何用例执行，它踩的正是这个空档。
    故判据落在**真连库**上（本文件的惯例）：需要 MySQL 在线，离线时跳过。

    关没关用 `conn.open` 判，不看代理自己的标志位 —— 标志位是我们要验的东西的
    影子，用它当判据等于自证（把 close 写成只置标志就是恒绿）。
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)
        assert conn.open is True, "with 体内连接就该是开着的"
    assert conn.open is False, "退出 with 必须真的关掉连接（pymysql 的 __exit__ 语义）"


@requires_mysql
def test_connect_autocommit_turns_autocommit_on_at_the_server():
    """账号连接必须真的开着 autocommit，业务连接必须没有 —— 判据取服务端状态位。

    为什么在**这一层**钉：autocommit 是 `connect_autocommit` 自己的产物
    （`connect(autocommit=True, **overrides)`），在工厂那一层根本看不见它。
    曾有一条在工厂断言 `kwargs["autocommit"] is True` 的用例，量错了地方 ——
    工厂递下去的是 cfg.mysql，那个键是函数内部加的，于是它恒红（不是恒绿，
    但同样不承重：它测不出任何真实的改动）。

    用 `get_autocommit()` 而不是客户端记的变量：它读的是握手后服务端回的
    SERVER_STATUS_AUTOCOMMIT 状态位，是这条连接在**服务端**的真实模式。
    两个方向都钉：只钉 True 的话，把 connect_autocommit 写成 return connect(...)
    仍然是... 不，那会红；但把 connect() 也改成 autocommit=True 就没人拦了，
    而业务连接开 autocommit 会让入库的批量写失去事务语义。
    """
    account = connect_autocommit()
    business = connect()
    try:
        assert account.get_autocommit() is True
        assert business.get_autocommit() is False
    finally:
        account.close()
        business.close()
