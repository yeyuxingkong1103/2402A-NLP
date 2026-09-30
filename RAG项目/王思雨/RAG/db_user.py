# -*- coding: utf-8 -*-
"""用户数据模块：MySQL 中用户、角色、会话、消息的增删查改，第 6 步不含 HTTP 接口。"""

import hashlib                                # 导入 hashlib，用于计算密码哈希
import pymysql                                # 导入 pymysql，用于捕获唯一键冲突异常
import config                                 # 导入配置模块，盐值从这里读
import db                                     # 导入数据库模块，复用连接封装
from logger import get_logger                 # 导入日志工具，用于记录数据操作

logger = get_logger("db_user")                # 创建本模块的 logger 实例

DEFAULT_ROLES = (                             # 系统预置的两个角色
    ("power_repair_expert", "电力维修专家"),   # 电力维修专家角色
    ("general_assistant", "通用助手"))         # 通用助手角色
MSG_ROLE_USER = "user"                        # 消息说话方：用户提问
MSG_ROLE_ASSISTANT = "assistant"              # 消息说话方：助手回答


def _connect():                               # 内部函数：获取数据库连接
    """获取 MySQL 连接，连不上时统一抛 RuntimeError，方便上层捕获。"""
    try:                                      # 尝试建立连接
        return db.get_mysql_conn()            # 复用 db 模块的连接封装
    except Exception as exc:                  # 连接失败
        raise RuntimeError(f"MySQL 连接失败：{exc}") from exc   # 统一转成 RuntimeError


# ===================== 一、密码哈希 =====================

def hash_password(password: str) -> str:      # 计算密码哈希
    """用 sha256 + 固定盐值计算密码哈希；教学简化版，生产应使用 bcrypt。"""
    salted = f"{config.PASSWORD_SALT}{password}"   # 盐值拼在密码前面，避免明文直接入哈希
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()   # 返回十六进制哈希串


def verify_password(password: str, password_hash: str) -> bool:   # 校验密码
    """比对明文密码与数据库中的哈希值是否一致。"""
    return hash_password(password) == password_hash   # 用同样方式算一遍再比较


# ===================== 二、用户 =====================

def register_user(username: str, password: str):   # 注册用户
    """注册新用户，用户名已存在时返回 None，成功返回新用户编号。"""
    conn = _connect()                         # 获取连接，失败会抛 RuntimeError
    try:                                      # 无论成败都要关闭连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 参数化插入新用户
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, hash_password(password)))   # 用户名与密码哈希
            user_id = cursor.lastrowid        # 取新用户编号
        conn.commit()                         # 提交事务
        logger.info("注册成功：%s（id=%s）", username, user_id)   # 记录日志
        return user_id                        # 返回新用户编号
    except pymysql.err.IntegrityError:        # 唯一键冲突说明用户名已存在
        logger.warning("注册失败，用户名已存在：%s", username)   # 记录告警
        return None                           # 返回 None
    finally:                                  # 收尾
        conn.close()                          # 关闭连接


def login_user(username: str, password: str):  # 登录
    """校验用户名与密码，成功返回 {"user_id", "username"}，失败返回 None。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 参数化按用户名查询
                "SELECT id, username, password_hash FROM users WHERE username = %s",
                (username,))                  # 用户名参数
            row = cursor.fetchone()           # 取一行
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    if not row:                               # 用户不存在
        logger.warning("登录失败，用户不存在：%s", username)   # 记录告警
        return None                           # 返回 None
    if not verify_password(password, row["password_hash"]):   # 密码不匹配
        logger.warning("登录失败，密码错误：%s", username)     # 记录告警
        return None                           # 返回 None
    logger.info("登录成功：%s", username)      # 记录日志
    return {"user_id": row["id"], "username": row["username"]}   # 返回用户信息


def get_user(user_id: int):                   # 按编号查用户
    """按编号查询用户，不存在时返回 None。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 参数化按主键查询
                "SELECT id, username, created_at FROM users WHERE id = %s",
                (user_id,))                   # 用户编号
            row = cursor.fetchone()           # 取一行
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    if not row: return None                   # 没查到返回 None
    return {"user_id": row["id"], "username": row["username"],
            "created_at": str(row["created_at"])}   # 返回用户信息


# ===================== 三、角色 =====================

def init_default_roles() -> int:              # 初始化预置角色
    """插入两个预置角色，幂等：已存在则更新描述，返回处理条数。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            for role_name, description in DEFAULT_ROLES:   # 逐个角色处理
                cursor.execute(               # 幂等插入：冲突则更新描述
                    "INSERT INTO roles (role_name, description) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE description = VALUES(description)",
                    (role_name, description))  # 角色名与描述
        conn.commit()                         # 提交事务
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    logger.info("预置角色初始化完成：%d 条", len(DEFAULT_ROLES))   # 记录日志
    return len(DEFAULT_ROLES)                 # 返回处理条数


def list_roles() -> list:                     # 列出全部角色
    """列出所有角色，按编号升序。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute("SELECT id, role_name, description FROM roles ORDER BY id")   # 查询
            rows = cursor.fetchall()          # 取全部
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    return [{"role_id": r["id"], "role_name": r["role_name"],
             "description": r["description"]} for r in rows]   # 整理成列表返回


def get_role_by_name(role_name: str):         # 按名称查角色
    """按角色名查询角色，不存在时返回 None。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 参数化按角色名查询
                "SELECT id, role_name, description FROM roles WHERE role_name = %s",
                (role_name,))                 # 角色名参数
            row = cursor.fetchone()           # 取一行
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    if not row: return None                   # 没查到返回 None
    return {"role_id": row["id"], "role_name": row["role_name"],
            "description": row["description"]}   # 返回角色信息


# ===================== 四、会话 =====================

def create_conversation(user_id: int, role_id: int, title: str) -> int:   # 创建会话
    """创建一个会话，返回新会话编号。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 参数化插入会话
                "INSERT INTO conversations (user_id, role_id, title) VALUES (%s, %s, %s)",
                (user_id, role_id, title))    # 所属用户、角色与标题
            conversation_id = cursor.lastrowid   # 取新会话编号
        conn.commit()                         # 提交事务
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    logger.info("创建会话：id=%s，用户=%s，角色=%s", conversation_id, user_id, role_id)   # 日志
    return conversation_id                    # 返回新会话编号


def list_conversations(user_id: int) -> list:   # 列出某用户的会话
    """列出指定用户的全部会话，按编号倒序（最新的在前）。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 按用户查询，只查自己的，最新在前
                "SELECT id, user_id, role_id, title, created_at FROM conversations "
                "WHERE user_id = %s ORDER BY id DESC", (user_id,))   # 用户编号
            rows = cursor.fetchall()          # 取全部
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    return [{"conversation_id": r["id"], "user_id": r["user_id"], "role_id": r["role_id"],
             "title": r["title"], "created_at": str(r["created_at"])} for r in rows]   # 整理返回


def get_conversation(conversation_id: int):    # 按编号查会话
    """按编号查询会话，不存在时返回 None。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 参数化按主键查询
                "SELECT id, user_id, role_id, title FROM conversations WHERE id = %s",
                (conversation_id,))           # 会话编号
            row = cursor.fetchone()           # 取一行
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    if not row: return None                   # 没查到返回 None
    return {"conversation_id": row["id"], "user_id": row["user_id"],
            "role_id": row["role_id"], "title": row["title"]}   # 返回会话信息


def delete_conversation(conversation_id: int, user_id: int) -> int:   # 删除会话（级联删消息）
    """删除自己的会话及其全部消息；user_id 不匹配则什么都不删，返回删除的会话条数。

    第 9.6 步修复：原来只删 conversations 行，messages 没有外键级联，
    删完消息变孤儿，已删会话的消息还能被 GET /message/{id} 查到。
    """
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 第一步：先确认该会话确实属于这个用户
                "SELECT id FROM conversations WHERE id = %s AND user_id = %s FOR UPDATE",   # 加行锁
                (conversation_id, user_id))   # 会话编号与用户编号，双条件保证不越权
            if not cursor.fetchone():         # 不存在或不属于该用户
                conn.rollback()               # 没改数据，回滚释放行锁
                logger.warning("删除会话失败（不存在或不属于该用户）：id=%s，用户=%s",
                               conversation_id, user_id)   # 记录告警
                return 0                      # 一条都不删
            cursor.execute("DELETE FROM messages WHERE conversation_id = %s", (conversation_id,))   # 先删消息
            removed_msgs = cursor.rowcount    # 记下顺带删掉的消息条数
            cursor.execute(                   # 第三步：再删会话本身
                "DELETE FROM conversations WHERE id = %s AND user_id = %s",
                (conversation_id, user_id))   # 再带一次 user_id，双保险
            affected = cursor.rowcount        # 取实际删除的会话条数
        conn.commit()                         # 两条删除一次提交，要么都成要么都不成
    except Exception as exc:                  # 出错
        conn.rollback()                       # 回滚，避免删一半
        logger.error("删除会话失败，已回滚：%s", exc)   # 记录日志
        raise                                 # 抛给上层，由 app.py 统一转 500
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    logger.info("删除会话：id=%s，用户=%s，同时删除消息 %d 条",
                conversation_id, user_id, removed_msgs)   # 记录日志
    return affected                           # 返回删除的会话条数


# ===================== 五、消息 =====================

def save_message(conversation_id: int, role: str, content: str) -> int:   # 保存消息
    """保存一条消息，role 取 user 或 assistant，返回新消息编号。"""
    if role not in (MSG_ROLE_USER, MSG_ROLE_ASSISTANT):   # 说话方取值非法
        raise ValueError(f"role 只能是 {MSG_ROLE_USER} 或 {MSG_ROLE_ASSISTANT}")   # 立即报错
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 参数化插入消息
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                (conversation_id, role, content))   # 会话编号、说话方、正文
            message_id = cursor.lastrowid     # 取新消息编号
        conn.commit()                         # 提交事务
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    return message_id                         # 返回新消息编号


def list_messages(conversation_id: int, limit: int = 50) -> list:   # 列出会话消息
    """按编号升序返回某会话的消息，默认最多 50 条。"""
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 按会话查询，编号升序取前 N 条
                "SELECT id, role, content, created_at FROM messages "
                "WHERE conversation_id = %s ORDER BY id ASC LIMIT %s",
                (conversation_id, limit))     # 会话编号与条数上限
            rows = cursor.fetchall()          # 取全部
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
    return [{"message_id": r["id"], "role": r["role"], "content": r["content"],
             "created_at": str(r["created_at"])} for r in rows]   # 整理返回


def cleanup_orphan_conversations(expected_ids=None) -> dict:   # 一次性清理工具（第 9.6 步）
    """一次性清理半截会话：删掉「只有 user、没有 assistant」的会话及其消息。

    这是第 9.6 步的一次性工具，专门清理第 6 步 save_turn 短路 bug 造出的半截会话，
    跑完即弃，不接入主流程，也不被 app.py 调用。
    expected_ids 是预期会话编号清单，实际结果与它不一致时中止，避免误删别人的数据。
    """
    conn = _connect()                         # 获取连接
    try:                                      # 收尾关连接
        with conn.cursor() as cursor:         # 打开游标
            cursor.execute(                   # 按会话分组，筛出只有 user、没有 assistant 的
                "SELECT conversation_id, COUNT(*) AS n FROM messages "
                "GROUP BY conversation_id HAVING SUM(role = %s) > 0 AND SUM(role = %s) = 0",
                (MSG_ROLE_USER, MSG_ROLE_ASSISTANT))   # 两个角色常量
            targets = cursor.fetchall()       # 待清理会话及其消息条数
            ids = [r["conversation_id"] for r in targets]   # 会话编号清单
            logger.info("清理前核对：待清理会话与消息数 %s", {r["conversation_id"]: r["n"] for r in targets})   # 打印
            if expected_ids is not None and sorted(ids) != sorted(expected_ids):   # 与预期不符
                logger.warning("实际 %s 与预期 %s 不一致，已中止以免误删", ids, expected_ids)   # 告警
                return {"conversations": [], "messages": 0}   # 一条都不删
            if not ids: return {"conversations": [], "messages": 0}   # 没有半截会话，直接返回
            pairs = [(c,) for c in ids]       # executemany 要求「每行一个元组」
            cursor.executemany("DELETE FROM messages WHERE conversation_id = %s", pairs)   # 先删消息
            deleted_msgs = cursor.rowcount    # executemany 的 rowcount 是累计影响行数
            cursor.executemany("DELETE FROM conversations WHERE id = %s", pairs)   # 再删会话
        conn.commit()                         # 两条删除一起提交
        logger.info("清理完成：删除会话 %s，共删除消息 %d 行", ids, deleted_msgs)   # 打印结果
        return {"conversations": ids, "messages": deleted_msgs}   # 返回明细
    except Exception as exc:                  # 出错
        conn.rollback()                       # 回滚，避免删一半
        logger.error("清理半截会话失败，已回滚：%s", exc)   # 记录日志
        raise                                 # 抛给调用方
    finally:                                  # 收尾
        conn.close()                          # 关闭连接
