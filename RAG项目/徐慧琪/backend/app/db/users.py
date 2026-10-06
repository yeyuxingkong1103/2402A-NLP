"""所内账号表（设计 §五 的 users）与它的读写。

为什么单独一个模块而不是把这段塞进 db/mysql.py：与 recommend/fee_log.py 同形 ——
**一张表一个模块**，建表语句、读写函数、口径注释同处一处，改表时不会漏看别的表；
mysql.py 自己的注释也写着「本包只放连接与执行」，它提供 connect/insert_many 这类
通用件，不认识任何具体表。设计 §三 那句「db/mysql.py 新增 users」指的是「落在 db/
包里」，落点取这个更细的模块（审计表由后续任务照同一形态另开一个）。

**role 的合法集合不在这里校验**：与 fee_log 不校验 status 同理（那个判据在 fees.py
的 STATUS_VALID），角色的判据属应用层（core/security.py 的 Role）。本模块若自己也
存一份角色清单，加角色时就有两处真相源 —— 而这一列是要被读进 JWT 载荷的，分叉即
越权。同理，密码强度也不在这一层：这里只存**已经算好的哈希**（谁算的见 security.py）。
"""
from __future__ import annotations

# 照抄设计 §五（字段名与注释语义不改）。两处 `COMMENT` 是把设计稿里的行内说明搬进
# 库：MinerU 那种「注释只活在文档里」的做法在真库里查不到，而这两个语义（哈希格式、
# team_id 的来路）恰恰是排查时要对着 SHOW FULL COLUMNS 看的。COMMENT 文本里刻意
# **不用半角逗号**：本仓库解析 DDL 的测试按顶层逗号切分定义块，注释里混进半角逗号
# 会切出噪声碎片（fee_log 那边的注释已记过这一条）。
# role 用 VARCHAR 而不是 ENUM：加角色时 ENUM 要 ALTER TABLE，而设计 §五 明写这列
# 会被读进 JWT 载荷 —— 改表这种事不该是加一个角色的前置条件。
DDL = """
CREATE TABLE IF NOT EXISTS users (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(64) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL COMMENT 'scrypt 自描述格式：算法$n$r$p$盐$哈希',
  role VARCHAR(16) NOT NULL COMMENT 'lawyer / assistant / partner；应用层校验',
  team_id VARCHAR(64) NOT NULL COMMENT '三层隔离里 team_id 的唯一来源',
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 读取列的固定顺序：get_by_username 与 list_all 共用，加列时改这里就够
_COLUMNS = ("id", "username", "password_hash", "role", "team_id", "is_active")

# 列表命令用的列集合：**故意不含 password_hash** —— 运维不需要它，少一次把哈希抄进
# 终端回滚缓冲、截图或工单的机会（哈希不是密钥，但它离口令只差一次离线爆破）
_LIST_COLUMNS = ("id", "username", "role", "team_id", "is_active", "created_at")


def ensure_table(conn) -> None:
    """建表（幂等）—— 由**建账号的那条路径**无条件调用（运维脚本；测试夹具同）。

    为什么不放进 HTTP lifespan / 装配（终审 m-4 的裁决，2026-10-02）：MySQL 挂住
    时服务必须照常启动、由 /healthz 报 503（test_main 的既有判据），启动期 DDL
    会把它变成「起不来」—— 而账号能存在就说明建号路径（manage_users 走的正是
    这里）已经建过表。失败让异常往上抛：与 fee_log 同口径，静默吞掉建表失败只会
    让故障推迟到第一次登录，而那时看到的是「账号不存在」这种指向完全错误的报错。

    （本 docstring 曾声称「任务 3 的 HTTP lifespan 启动时」会调用，与实现不符 ——
    终审 m-4 抓出，已按实际改写；app/ 内确实没有第二个调用点。）
    """
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def get_by_username(conn, username: str) -> dict | None:
    """按用户名取一行，没有则 None（不是抛）。

    登录路径上「查无此人」是**正常结果**而不是异常：它要与「口令不对」走出同一个
    对外响应（设计 §七：未认证对外一律 404，不区分原因）。返回 dict 而非元组是
    给调用方按列名取值 —— 本表有七列，按位置取值时插一列就会静默错位。
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM users WHERE username = %s",
            (username,))
        row = cur.fetchone()
    return None if row is None else dict(zip(_COLUMNS, row))


def is_active(conn, user_id: int) -> bool:
    """按主键查这一格：停用/启用，返回布尔（**查无此人也是 False**）。

    为什么登录之外还要按 id 查一次（任务 2 复审的交接、本任务落地）：token 一旦
    签发就不再查库，于是「停用」的生效窗口等于 token 剩余有效期（最长 8 小时）。
    而停用的真实触发点是离职与设备丢失 —— 恰恰最需要立刻生效。逐请求查一次把它
    变成即时：本机 MySQL 一次主键命中是亚毫秒级，被它门住的是秒级检索 + 推理。

    只取 is_active 一列、不返回整行：这条查询在每个受保护请求上都要跑，而
    password_hash 是这张表里最不该被无谓搬运的东西（R3 的 list_all 也是同一口径）。
    查无此人返回 False 而不是 None：调用方要的判据只有「能不能放行」一个，
    多一种返回值就多一条会被写错的分支 —— 而写错的方向是**放行**（fail-open），
    与停用判断的立意相反，故在类型上就没有第二条路。

    不用 `select 1 from users where id=%s and is_active=1`：那种写法在「行不存在」
    与「行存在但停用」上返回同一个空集，事后想区分（审计说清是哪种）就得多查一次。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    return bool(row[0]) if row is not None else False


def create(conn, username: str, password_hash: str, role: str,
           team_id: str) -> int:
    """插入一个账号，返回它的 id。

    重名不在这里「先查后插」：那中间有竞态窗口，两个运维同时建同一账号时会双双
    通过检查、双双插入。UNIQUE 约束是唯一可靠的判重，撞上去时 pymysql 抛
    IntegrityError —— 由调用方（运维脚本）翻成人话，本层不吞异常。
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, team_id) "
            "VALUES (%s, %s, %s, %s)", (username, password_hash, role, team_id))
        new_id = cur.lastrowid
    conn.commit()
    return int(new_id)


def set_password(conn, username: str, password_hash: str) -> bool:
    """改口令，返回**是否真的改到了一行**。

    返回值不是装饰：命令行里 `passwd --username zhagnsan`（拼错一个字）若静默返回
    「成功」，运维会以为口令已重置，而那个人下次登录仍用旧口令 —— 这类「操作对象
    不存在却报成功」是运维脚本最该避免的形态。同理适用于 set_active。
    """
    return _update(conn, "password_hash", username, password_hash)


def set_active(conn, username: str, is_active: bool) -> bool:
    """停用/启用账号（软删，不 DELETE 行）。

    为什么不真删：设计 §六 的审计只记 user_id，删了行之后历史审计行会指回一个不
    存在的人；而「这个人曾经存在过、什么时候被停的」本身也是审计材料。
    is_active 显式转成 1/0 再下发：TINYINT(1) 收 bool 也行，但那样「库里存的就是
    0/1」这件事要押在驱动的隐式转换上，而下面读回来的是 0/1 而不是 True/False ——
    写入与读回的形状不对称时，测试与调用方各要记一套。
    """
    return _update(conn, "is_active", username, 1 if is_active else 0)


def _update(conn, column: str, username: str, value) -> bool:
    """按用户名改一列，返回是否命中一行。

    column 由本模块的三个调用点传入字面量、不接受外部输入（与 mysql.insert_many
    的既有约定一致）；真要接用户输入时必须改成白名单。
    """
    with conn.cursor() as cur:
        cur.execute(f"UPDATE users SET {column} = %s WHERE username = %s",
                    (value, username))
        changed = cur.rowcount
    conn.commit()
    return changed == 1


def list_all(conn) -> list[dict]:
    """全部账号，供运维脚本的 list 命令用（不含 password_hash，见 _LIST_COLUMNS）。"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(_LIST_COLUMNS)} FROM users ORDER BY id")
        return [dict(zip(_LIST_COLUMNS, row)) for row in cur.fetchall()]
