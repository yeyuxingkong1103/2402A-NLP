"""MySQL 连接与执行。

存在的理由：技术方案 7.1 把法条主数据放在 MySQL，而本期入库需要往那里写 1260 条、
并在编码时读回元数据。技术方案 8.1 本模块还承担「团队条件统一注入」，但那是 API 层
的多租户需求，本期无多租户，等接口层开工时再补，此处不留空壳。

驱动选 PyMySQL 而非 mysql-connector：纯 Python、无编译依赖，装起来最省事；
本机 MySQL 实测 8.4.11，默认 caching_sha2_password，所以必须一并装 cryptography。

**并发保护（2026-09-29 任务 3 复审补，Critical）**：一个 PyMySQL 连接就是一条 TCP
会话，同一时刻只能有一段「发语句、读回包」在进行。此前只有 CLI（单进程单线程）用它，
任务 3 把 `Services` 放进 app.state 供多线程端点共享后，协议损坏第一次可达。实测同一
连接上跑 `SELECT 1 FROM article LIMIT 1`：2 线程 × 60 次 → 成功 119 / 失败 1；
8 线程 × 20 次 → 成功 5 / 失败 155，异常形如 `InterfaceError: (0, '')`、
`InternalError: Packet sequence number wrong - got 1 expected 2`、
`ValueError: PyMemoryView_FromBuffer(): info->buf must not be NULL`，还有
`OperationalError: MySQL server has gone away` —— 连接被留在损坏状态且没有重连逻辑，
后续业务请求跟着一起挂；而 /healthz 正是编排层会周期并发打的那个端点。

故 `connect()` 返回的连接把每条 SQL 对话串行化（见 _Connection）。选串行化而不是连接池
或「每线程一个连接」：单机部署、并发上限 20（设计 §六）、查询都是索引命中的短查询，
排队的吞吐代价可接受，而正确性是硬要求；每线程一个连接则要另做生命周期与泄漏管理，
在这个规模上「多出来的出错方式」比它省下的等待更贵。

**将来上真连接池的边界**：本类只保证「同一个连接上同一时刻只有一段对话」。它**不**
解决跨语句的事务隔离（两次 execute 之间仍可能插进别的线程）、连接数上限、以及
wait_timeout 到期后的重连 —— 那些是建池时要一并做的（DBUtils / SQLAlchemy QueuePool）。
届时应把 `connect()` 整个换掉（调用方无感），而不是在这个类上继续加东西：它已经在了
「一层代理」的复杂度上，再叠池化会变成两套生命周期互相打架。
"""
from __future__ import annotations

import pathlib
import threading

import pymysql

# 开发环境默认值，与 deploy/docker-compose.yml 的 mysql 服务一致。
# 生产按技术方案附录 B 走 .env，真实值不入仓库。
DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "devroot",
    "database": "fl_law",
    "charset": "utf8mb4",
    # 连不上时快速失败：单测用它在收集阶段判断要不要跳过，不能拖慢整个测试
    "connect_timeout": 3,
}

# schema.sql 在仓库根下，不在 backend 里——它是部署产物，与容器配置同处 deploy/
SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "mysql" / "schema.sql"


class _Cursor:
    """游标代理：替调用方持锁，直到游标关闭。

    锁的范围是**整个 with 块**而不是单次 execute：一次查询由「发语句」与「读回包」
    两段组成，execute 与随后的 fetchone 之间若插进另一个线程的语句，读回来的就是
    别人的结果集 —— 复现里的 InterfaceError 与 Packet sequence number wrong 正是
    这么来的。所以串行化的单位是「一段对话」，不是「一条语句」。
    """

    def __init__(self, cursor, lock) -> None:
        self._cursor = cursor
        self._lock = lock
        self._released = False

    def __enter__(self) -> "_Cursor":
        # 锁在 cursor() 里已经拿到，入场没有额外动作；返回代理本身，
        # 于是 with 体内的 cur.execute / fetchone 走 __getattr__ 透传到底层游标
        return self

    def __exit__(self, *exc_info) -> bool:
        # 不论 with 体内抛不抛都要放锁；返回 False 让异常照常往外传，
        # 这里不是"吞掉异常"的地方
        self.close()
        return False

    def close(self) -> None:
        """关游标并放锁。幂等：显式 close 之后 with 退出时不会重复 release。"""
        if self._released:
            return
        self._released = True
        try:
            self._cursor.close()
        finally:
            # finally 而不是顺序执行：关游标本身失败也必须放锁，否则这把锁永远
            # 不还，这条连接上后续所有查询静默卡死（比抛错难查得多）
            self._lock.release()

    def __getattr__(self, name):
        # execute / executemany / fetchone / fetchall / rowcount / description…
        # 全部原样透传，调用方看到的仍是 pymysql 游标的接口面
        return getattr(self._cursor, name)


class _Connection:
    """连接代理：同一个连接上的并发查询在锁上排队。

    用 RLock 而不是 Lock：同一线程里嵌套开游标（读着一行再按它的 id 查一次）是合法
    用法，Lock 会自己锁死自己；RLock 只对**持有者线程**可重入，跨线程照样互斥。

    **调用约定：游标一律 `with conn.cursor() as cur:`**。锁在游标关闭时释放，拿了
    不关 = 锁不放，别的线程会一直等（形态是静默卡住，不是报错）。本项目所有调用点
    （healthz 探针、db/users、retrieval、recommend、ingest）都守这条。

    只做「一段对话独占连接」这一件事；跨语句的事务隔离、连接数上限、wait_timeout
    后的重连都不在这里，见模块 docstring 末尾说的池化边界。

    **对将来路由的提醒**：别在 async 路由里跨 `await` 持游标 —— 锁是**按线程**可重入
    的，同一个事件循环上的另一个请求在同线程里能直接拿到它，串行化会静默失效。
    pymysql 本来就是阻塞驱动，查询放同步路由（FastAPI 丢线程池）或 `to_thread` 里。
    """

    def __init__(self, conn) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    def __enter__(self) -> "_Connection":
        """支持 `with connect() as conn:`（与裸 pymysql 连接同形）。

        代理必须连上下文管理协议一起接过来：pymysql 的 Connection 就是上下文管理器
        （源里 `__exit__` 调 close），而 `connect()` 的对外契约是「调用方写法不变」——
        漏了这两条，`with connect() as conn:` 当场 TypeError。实测放过一次这形态：
        `tools/check_env.py` 的 mysql 项整项 FAIL，而它在 pytest 视野之外
        （check_env 不在任何测试里执行），套件全绿也照漏。故下面 test_db_mysql
        有一条真连库的用例钉住它。
        """
        return self

    def __exit__(self, *exc_info) -> bool:
        """退出即关连接 —— 与 pymysql 的 `__exit__` 同语义。

        返回 False 让 with 体内的异常照常外传：这里不是吞异常的地方
        （与 _Cursor.__exit__ 同一口径）。
        """
        self.close()
        return False

    def cursor(self, *args, **kwargs) -> _Cursor:
        """开游标：先拿锁再开。开游标失败也必须放锁，否则一个异常就把连接锁死了。"""
        self._lock.acquire()
        try:
            cursor = self._conn.cursor(*args, **kwargs)
        except BaseException:
            # 收 BaseException 而不是 Exception：加载/中断期间走的 KeyboardInterrupt
            # 同样会从这里穿过去，漏掉它等于把锁留在手里
            self._lock.release()
            raise
        return _Cursor(cursor, self._lock)

    def commit(self) -> None:
        """提交也是一段对话（要先往 socket 写 COMMIT 再读它的回包），同样排队。"""
        with self._lock:
            self._conn.commit()

    def rollback(self) -> None:
        """回滚同上。本项目当前没有回滚点，留着它是为了事务语义完整、不逼调用方
        绕回裸连接（绕回去就绕过了锁）。"""
        with self._lock:
            self._conn.rollback()

    def close(self) -> None:
        """关闭底层连接。已关闭时 pymysql 自己会抛，这里不吞：关闭是显式动作，
        重复关闭是调用方的错，静默掉只会把"谁关的"这条线索抹掉。"""
        with self._lock:
            self._conn.close()

    def __getattr__(self, name):
        # 其余属性（open、ping、server_version…）透传，让代理对调用方尽量无形。
        # 但透传出来的方法**不排队**：底层还有 conn.query() 这类低层入口，
        # 新的查询一律走 cursor()/commit()，别从这个口子绕过去
        return getattr(self._conn, name)


def connect(**overrides) -> _Connection:
    """建立连接。charset 固定 utf8mb4——法条含生僻字与全角标点，utf8 三字节存不下。

    返回的是带并发保护的代理（理由与复现数字见模块 docstring）：调用方写法不变，
    但拿到的不是裸 pymysql 连接。
    """
    return _Connection(pymysql.connect(**{**DEFAULT_CONFIG, **overrides}))


def connect_autocommit(**overrides) -> _Connection:
    """建一条**逐语句自动提交**的连接：每次读都看得到别处已提交的最新值。

    为什么需要第二条连接（2026-09-29 本任务实测，不是推理）：pymysql 默认
    autocommit=False，于是这条连接上的第一次 SELECT 就开启事务并定下
    REPEATABLE READ 的读视图。实测（users 表，两条连接）：

        服务连接读 is_active → True
        运维连接 UPDATE is_active=0 并 commit
        服务连接再读 → **仍是 True**（读的是旧快照）
        服务连接 commit() 之后再读 → False

    后果是「停用即时生效」这类判据在真库上**静默失效**：鉴权依赖逐请求查
    is_active 也照样读到启动那一刻的快照（服务连接的第一次读发生在启动时的
    healthz 探针上，那个事务一直不结束）。同一条连接上「运维刚建好的账号登录
    不上」也是同一形态 —— 而它看起来与「口令错」一模一样。

    所以账号路径（登录查行 + 逐请求查 is_active）**不用**业务连接，用它自己
    这条 autocommit 的连接：每条语句自成一个事务，读视图不跨语句存活。
    业务连接一字不动（改它的隔离级别会牵动 retrieval/ingest 的既有语义，
    那是另一件事，见报告的存疑条目）。

    写入语义与 connect() 不同，别拿它做多语句写：每条语句当场提交，没有
    「出错一起回滚」的窗口。本项目的写路径（ingest、fee_log、审计）都用
    connect()，本函数只服务账号读路径。
    """
    return connect(autocommit=True, **overrides)


def _split_statements(sql: str) -> list[str]:
    """按分号切句并剥掉注释行。

    不能简单地"以 -- 开头的整句跳过"——注释与语句往往在同一段里
    （"-- 法条主数据\nCREATE TABLE article (...)"），那样会把整条建表语句
    一起丢掉，而且是静默丢：表没建成，后续插入才报错，排查方向会被带偏。
    """
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def apply_schema(conn, schema_path: pathlib.Path = SCHEMA_PATH) -> list[str]:
    """执行 DDL 文件，返回执行的语句列表。DDL 全部写成 IF NOT EXISTS，重复执行安全。"""
    statements = _split_statements(schema_path.read_text(encoding="utf-8"))
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    conn.commit()
    return statements


def insert_many(conn, table: str, columns: list[str], rows: list[tuple]) -> int:
    """批量插入，返回插入行数。

    用 executemany 而非拼多值 INSERT：千行级时拼字符串会超 max_allowed_packet，
    且 executemany 由驱动分批下发，出错时定位到具体行更容易。

    table 与 columns 是直接拼进 SQL 的——本项目的调用方全部传字面量常量，
    不接受外部输入；若将来要接用户输入，必须改成白名单校验。
    """
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)
