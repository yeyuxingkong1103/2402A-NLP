"""在线核心的配套工具：密码处理和非核心网页/API 路由。"""

import hashlib
import hmac
import secrets

import mysql.connector
from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class Account(BaseModel):
    """注册和登录共用的请求格式。"""

    username: str = Field(min_length=2, max_length=20)
    password: str = Field(min_length=4, max_length=50)


def password_hash(password, salt=None):
    """把明文密码变成带随机盐的 PBKDF2 摘要。"""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 260_000
    ).hex()
    return f"pbkdf2_sha256$260000${salt}${digest}"


def password_matches(password, stored):
    """重新计算摘要并安全比较；损坏的数据库记录直接视为不匹配。"""
    try:
        salt = stored.split("$")[2]
        calculated = password_hash(password, salt)
        return hmac.compare_digest(calculated, stored)
    except (ValueError, IndexError, TypeError):
        return False


def run_query(open_database, sql, values=(), fetch=None):
    """用传入的数据库连接函数执行事务，供教学入口保留一个简单 query。"""
    connection = open_database()
    try:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(sql, values)
            if fetch == "one":
                result = cursor.fetchone()
            elif fetch == "all":
                result = cursor.fetchall()
            else:
                result = cursor.lastrowid
            connection.commit()
            return result
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def prepare_database(query, doctor_prompt):
    """幂等创建两张 MySQL 表，并插入唯一医生角色。"""
    query("CREATE TABLE IF NOT EXISTS users(id BIGINT PRIMARY KEY AUTO_INCREMENT,username VARCHAR(50) UNIQUE NOT NULL,password_hash VARCHAR(255) NOT NULL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    query("CREATE TABLE IF NOT EXISTS roles(id BIGINT PRIMARY KEY AUTO_INCREMENT,name VARCHAR(50) UNIQUE NOT NULL,persona_prompt TEXT NOT NULL,description VARCHAR(255))")
    query(
        "INSERT IGNORE INTO roles(name,persona_prompt,description) VALUES(%s,%s,%s)",
        ("医生", doctor_prompt, "通用健康医生；当前知识库主要覆盖高血压指南"),
    )


def check_services(httpx, milvus_class, collection, setting, memory_class, query, engine):
    """检查在线模型、Milvus、Redis和MySQL，返回前端需要的状态字典。"""
    model_ready = False
    knowledge_ready = False
    try:
        response = httpx.get(
            setting("LLM_BASE_URL").rstrip("/") + "/models",
            timeout=5,
            headers={"Authorization": "Bearer " + setting("LLM_API_KEY")},
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        model_ready = any(item.get("id") == setting("LLM_MODEL") for item in models)
        client = milvus_class(
            uri=setting("RAG_MILVUS_URI", "http://127.0.0.1:19530"), timeout=3
        )
        try:
            exists = client.has_collection(collection)
            knowledge_ready = exists and int(
                client.get_collection_stats(collection)["row_count"]
            ) > 0
        finally:
            client.close()
        memory_class().client.ping()
        query("SELECT 1", fetch="one")
    except Exception:
        knowledge_ready = False
    return {
        "ready": model_ready and knowledge_ready,
        "model_ready": model_ready,
        "knowledge_ready": knowledge_ready,
        "provider": "deepseek",
        "model": setting("LLM_MODEL"),
        "rerank_state": engine.rerank_state if engine else "not_loaded",
        "message": "首次提问会加载本地检索模型" if model_ready and knowledge_ready else "请检查模型及数据库服务",
    }


def install_common_routes(
    app, base, query, doctor, memory_class, status_callback, mysql_connector
):
    """安装网页、账号、角色、历史和状态路由，避免挤占答辩核心篇幅。"""
    app.mount(
        "/static",
        StaticFiles(directory=base / "app" / "static"),
        name="static",
    )

    @app.get("/")
    def root():
        return {"service": "知愈医疗RAG", "status": "ok", "entry": "app.single_app"}

    @app.get("/chat", include_in_schema=False)
    def page():
        return FileResponse(base / "app" / "static" / "index.html")

    @app.get("/chat/", include_in_schema=False)
    def page_redirect():
        return RedirectResponse("../chat", status_code=307)

    @app.get("/learn", include_in_schema=False)
    def learn():
        return FileResponse(base / "app" / "static" / "learn.html")

    @app.get("/api/status")
    def status():
        return status_callback()

    @app.post("/api/register")
    def register(body: Account):
        try:
            user_id = query(
                "INSERT INTO users(username,password_hash) VALUES(%s,%s)",
                (body.username, password_hash(body.password)),
            )
        except mysql_connector.IntegrityError as error:
            raise HTTPException(400, "用户名已存在") from error
        return {"user_id": user_id, "username": body.username, "message": "注册成功"}

    @app.post("/api/login")
    def login(body: Account):
        user = query(
            "SELECT * FROM users WHERE username=%s", (body.username,), "one"
        )
        if not user or not password_matches(body.password, user["password_hash"]):
            raise HTTPException(401, "用户名或密码错误")
        return {
            "user_id": user["id"],
            "username": user["username"],
            "message": "登录成功",
        }

    @app.get("/api/roles")
    def roles():
        return query(
            "SELECT id,name,description FROM roles WHERE name='医生'", fetch="all"
        )

    @app.get("/api/history")
    def history(user_id: int, role_id: int):
        doctor(role_id)
        return {"history": memory_class().history(user_id, role_id)}

    @app.delete("/api/history")
    def clear_history(user_id: int, role_id: int):
        doctor(role_id)
        memory_class().clear(user_id, role_id)
        return {"message": "对话记忆已清空"}
