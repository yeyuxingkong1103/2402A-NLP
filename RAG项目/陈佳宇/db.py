import pymysql
import hashlib
import uuid
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

DB_CONFIG = {
    "host": MYSQL_HOST,
    "port": MYSQL_PORT,
    "user": MYSQL_USER,
    "password": MYSQL_PASSWORD,
    "database": MYSQL_DATABASE,
    "charset": "utf8mb4"
}

# 简单连接池：维护一个连接，断开自动重连
_pool_conn = None

def get_conn():
    global _pool_conn
    try:
        if _pool_conn is None:
            _pool_conn = pymysql.connect(**DB_CONFIG)
        else:
            _pool_conn.ping(reconnect=True)
        return _pool_conn
    except Exception:
        _pool_conn = pymysql.connect(**DB_CONFIG)
        return _pool_conn

def hash_password(password, salt=None):
    if salt is None:
        salt = uuid.uuid4().hex
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password, stored):
    if ":" not in stored:
        return password == stored
    salt, hashed = stored.split(":", 1)
    return hash_password(password, salt) == f"{salt}:{hashed}"

def get_character(character_name):
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM characters WHERE name = %s"
            cursor.execute(sql, (character_name,))
            return cursor.fetchone()
    finally:
        pass

def get_character_by_id(char_id):
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM characters WHERE id = %s"
            cursor.execute(sql, (char_id,))
            return cursor.fetchone()
    finally:
        pass

def list_characters():
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT id, name, description FROM characters"
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        pass

def add_character(name, description, system_prompt, collection_name):
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT id FROM characters WHERE name = %s"
            cursor.execute(sql, (name,))
            if cursor.fetchone():
                return False, "角色名已存在"
            sql = "INSERT INTO characters (name, description, system_prompt, collection_name) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (name, description, system_prompt, collection_name))
            conn.commit()
            return True, "添加成功"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        pass

def update_character(char_id, name=None, description=None, system_prompt=None, collection_name=None):
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            updates = []
            params = []
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if description is not None:
                updates.append("description = %s")
                params.append(description)
            if system_prompt is not None:
                updates.append("system_prompt = %s")
                params.append(system_prompt)
            if collection_name is not None:
                updates.append("collection_name = %s")
                params.append(collection_name)
            if not updates:
                return False, "没有需要更新的字段"
            params.append(char_id)
            sql = f"UPDATE characters SET {', '.join(updates)} WHERE id = %s"
            cursor.execute(sql, params)
            conn.commit()
            if cursor.rowcount == 0:
                return False, "角色不存在"
            return True, "更新成功"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        pass

def delete_character(char_id):
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM characters WHERE id = %s"
            cursor.execute(sql, (char_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return False, "角色不存在"
            return True, "删除成功"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        pass

def verify_user(username, password):
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM users WHERE username = %s"
            cursor.execute(sql, (username,))
            user = cursor.fetchone()
            if user and verify_password(password, user[2]):
                return user
            return None
    finally:
        pass

def register_user(username, password):
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT id FROM users WHERE username = %s"
            cursor.execute(sql, (username,))
            if cursor.fetchone():
                return False, "用户名已存在"
            hashed_pwd = hash_password(password)
            sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
            cursor.execute(sql, (username, hashed_pwd))
            conn.commit()
            return True, "注册成功"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        pass
