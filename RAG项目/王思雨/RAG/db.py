# -*- coding: utf-8 -*-
"""数据库模块：MySQL 建表 + Redis 常用操作，第 1 步只做基础连接，不含业务逻辑。"""

import pymysql                        # 导入 pymysql，用于连接 MySQL 数据库
import redis                          # 导入 redis，用于连接 Redis 缓存
import config                         # 导入配置模块，连接参数全部从这里取
from logger import get_logger         # 导入日志工具，用于记录连接与建表日志

logger = get_logger("db")             # 创建本模块的 logger 实例

CHARSET = "utf8mb4"                   # MySQL 统一字符集，支持中文
SHORT_MEMORY_LIMIT = 10               # 短期记忆最多保留的最近聊天条数
SHORT_MEMORY_TTL = 60 * 60 * 24       # 短期记忆过期时间：1 天，单位秒
CONNECT_TIMEOUT = 5                   # 连接超时秒数，避免服务没起时长时间卡住

# ===================== 建表语句（共 4 张表） =====================

# users 表：存放登录用户
CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (              -- 用户表，已存在时不重复创建
    id INT AUTO_INCREMENT PRIMARY KEY,          -- 主键，自增编号
    username VARCHAR(64) NOT NULL UNIQUE,       -- 用户名，唯一且非空
    password_hash VARCHAR(255) NOT NULL,        -- 密码哈希值，只存哈希不存明文
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 创建时间，默认当前时间
    KEY idx_users_username (username)           -- 用户名索引，加速按用户名查询
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;        -- 使用 InnoDB 引擎与 utf8mb4 字符集
"""

# roles 表：存放问答角色（如检修工、值班员）
CREATE_ROLES_SQL = """
CREATE TABLE IF NOT EXISTS roles (              -- 角色表，已存在时不重复创建
    id INT AUTO_INCREMENT PRIMARY KEY,          -- 主键，自增编号
    role_name VARCHAR(64) NOT NULL UNIQUE,      -- 角色名称，唯一且非空
    description VARCHAR(255),                   -- 角色说明，可为空
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 创建时间，默认当前时间
    KEY idx_roles_name (role_name)              -- 角色名索引，加速按角色名查询
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;        -- 使用 InnoDB 引擎与 utf8mb4 字符集
"""

# conversations 表：存放会话（一次完整对话）
CREATE_CONVERSATIONS_SQL = """
CREATE TABLE IF NOT EXISTS conversations (      -- 会话表，已存在时不重复创建
    id INT AUTO_INCREMENT PRIMARY KEY,          -- 主键，自增编号
    user_id INT NOT NULL,                       -- 所属用户编号，对应 users.id
    role_id INT NOT NULL,                       -- 使用的角色编号，对应 roles.id
    title VARCHAR(255),                         -- 会话标题，可为空
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 创建时间，默认当前时间
    KEY idx_conv_user (user_id),                -- 用户索引，加速查某人的会话
    KEY idx_conv_role (role_id)                 -- 角色索引，加速查某角色的会话
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;        -- 使用 InnoDB 引擎与 utf8mb4 字符集
"""

# messages 表：存放会话中的每条消息
CREATE_MESSAGES_SQL = """
CREATE TABLE IF NOT EXISTS messages (           -- 消息表，已存在时不重复创建
    id INT AUTO_INCREMENT PRIMARY KEY,          -- 主键，自增编号
    conversation_id INT NOT NULL,               -- 所属会话编号，对应 conversations.id
    role VARCHAR(32) NOT NULL,                  -- 说话方：user 提问 / assistant 回答
    content TEXT,                               -- 消息正文，可为空
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 创建时间，默认当前时间
    KEY idx_msg_conv (conversation_id)          -- 会话索引，加速查某会话的消息
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;        -- 使用 InnoDB 引擎与 utf8mb4 字符集
"""


# ===================== MySQL：连接与建表 =====================

def get_mysql_conn():
    """获取 MySQL 连接，参数全部来自 config（即 .env），代码中不写死任何密码。"""
    if not config.MYSQL_HOST:                     # 主机地址为空说明 .env 还没配好
        raise RuntimeError("MYSQL_HOST 未配置，请复制 .env.example 为 .env 并填写")  # 立即报错
    return pymysql.connect(                       # 创建并返回 MySQL 连接对象
        host=config.MYSQL_HOST,                   # 主机地址，来自 .env
        port=config.MYSQL_PORT,                   # 端口，来自 .env
        user=config.MYSQL_USER,                   # 用户名，来自 .env
        password=config.MYSQL_PASSWORD,           # 密码，来自 .env，不写死在代码里
        database=config.MYSQL_DB,                 # 数据库名，来自 .env
        charset=CHARSET,                          # 字符集，保证中文正常存取
        cursorclass=pymysql.cursors.DictCursor,   # 查询结果以字典返回，便于取值
        autocommit=False                          # 关闭自动提交，由调用方显式提交
    )                                             # 连接创建结束


def init_mysql_tables() -> None:
    """初始化 4 张表：users、roles、conversations、messages。"""
    conn = get_mysql_conn()                       # 获取一个数据库连接
    try:                                          # 无论成功失败都要关闭连接
        with conn.cursor() as cursor:             # 打开游标，with 结束后自动释放
            cursor.execute(CREATE_USERS_SQL)          # 执行建表：users
            cursor.execute(CREATE_ROLES_SQL)          # 执行建表：roles
            cursor.execute(CREATE_CONVERSATIONS_SQL)  # 执行建表：conversations
            cursor.execute(CREATE_MESSAGES_SQL)       # 执行建表：messages
        conn.commit()                             # 提交事务，建表才真正生效
        logger.info("MySQL 4 张表初始化完成")      # 记录建表完成日志
    finally:                                      # 无论是否异常
        conn.close()                              # 关闭连接，回收资源


def close_quietly(conn) -> None:
    """安静地关闭一个数据库连接，忽略关闭过程中的异常。"""
    if conn is None:                              # 连接为空时
        return                                    # 直接返回，什么都不做
    try:                                          # 尝试关闭连接
        conn.close()                              # 真正执行关闭
    except Exception:                             # 关闭失败也不影响主流程
        logger.warning("关闭连接时出现异常，已忽略")  # 记录一条告警


# ===================== Redis：连接与演示 =====================

def get_redis_client():
    """获取 Redis 客户端，参数全部来自 config（即 .env）。"""
    if not config.REDIS_HOST:                     # 主机地址为空说明 .env 还没配好
        raise RuntimeError("REDIS_HOST 未配置，请复制 .env.example 为 .env 并填写")  # 避免空地址卡住
    return redis.Redis(                           # 创建并返回 Redis 客户端
        host=config.REDIS_HOST,                   # 主机地址，来自 .env
        port=config.REDIS_PORT,                   # 端口，来自 .env
        db=config.REDIS_DB,                       # 库编号，来自 .env
        decode_responses=True,                    # 自动把字节解码成字符串，便于阅读
        protocol=2,                               # 使用 RESP2 协议，兼容 Redis 5.x 老版本
        socket_connect_timeout=CONNECT_TIMEOUT    # 连接超时，避免服务没起时卡住
    )                                             # 客户端创建结束


def redis_demo() -> dict:
    """演示 Redis 5 种常用数据类型：string、list、hash、set、zset。"""
    client = get_redis_client()                   # 获取 Redis 客户端
    client.set("demo:string", "变压器漏油处理")     # string：最基础的键值对
    str_value = client.get("demo:string")         # 读取 string 值
    client.delete("demo:list")                    # 先清空 list，保证演示可重复执行
    client.rpush("demo:list", "断电", "验电", "拆解")  # list：有序可重复，适合最近记录
    list_value = client.lrange("demo:list", 0, -1)    # 读取 list 全部元素
    client.hset("demo:hash", mapping={"设备": "变压器", "故障": "漏油"})  # hash：一键多字段
    hash_value = client.hgetall("demo:hash")      # 读取 hash 全部字段
    client.delete("demo:set")                     # 先清空 set，保证演示可重复执行
    client.sadd("demo:set", "绝缘老化", "过热", "绝缘老化")  # set：无序自动去重
    set_value = client.smembers("demo:set")       # 读取 set 全部成员（重复项已被去掉）
    client.delete("demo:zset")                    # 先清空 zset，保证演示可重复执行
    client.zadd("demo:zset", {"变压器": 9.5, "断路器": 8.0})  # zset：带分数可排序
    zset_value = client.zrange("demo:zset", 0, -1, withscores=True)  # 按分数升序读取
    logger.info("Redis 5 种数据类型演示完成")      # 记录演示完成日志
    return {                                      # 返回演示结果，方便测试与查看
        "string": str_value,                      # string 示例结果
        "list": list_value,                       # list 示例结果
        "hash": hash_value,                       # hash 示例结果
        "set": list(set_value),                   # set 示例结果（转成列表便于展示）
        "zset": zset_value,                       # zset 示例结果
    }                                             # 演示结果字典结束


# ===================== Redis：短期记忆与缓存 =====================

def save_short_memory(user_id: int, content: str) -> int:
    """把一条聊天内容追加进短期记忆，只保留最近 10 条，返回当前条数。"""
    client = get_redis_client()                             # 获取 Redis 客户端
    key = f"chat:memory:{user_id}"                          # 每个用户一个 list 键
    client.rpush(key, content)                              # 从右侧追加最新一条记录
    client.ltrim(key, -SHORT_MEMORY_LIMIT, -1)              # 只保留最后 10 条，旧的丢弃
    client.expire(key, SHORT_MEMORY_TTL)                    # 设置 1 天过期，避免占用内存
    count = client.llen(key)                                # 读取当前记忆条数
    logger.info("用户 %s 短期记忆已更新，当前 %s 条", user_id, count)  # 记录日志
    return count                                            # 返回当前条数


def get_short_memory(user_id: int) -> list:
    """读取某个用户的短期记忆，按时间从旧到新返回列表。"""
    client = get_redis_client()                             # 获取 Redis 客户端
    key = f"chat:memory:{user_id}"                          # 与写入时使用同一个键
    return client.lrange(key, 0, -1)                        # 返回全部记录（最多 10 条）


def cache_set(key: str, value: str, expire: int = 300) -> None:
    """写缓存：O(1) 常数时间复杂度，expire 为过期秒数，默认 5 分钟。"""
    client = get_redis_client()                             # 获取 Redis 客户端
    client.set(key, value, ex=expire)                       # 写入键值并设置过期时间
    logger.info("缓存已写入：%s，%s 秒后过期", key, expire)   # 记录写入日志


def cache_get(key: str):
    """读缓存：O(1) 常数时间复杂度，不存在时返回 None。"""
    client = get_redis_client()                             # 获取 Redis 客户端
    return client.get(key)                                  # 按键取值，未命中返回 None
