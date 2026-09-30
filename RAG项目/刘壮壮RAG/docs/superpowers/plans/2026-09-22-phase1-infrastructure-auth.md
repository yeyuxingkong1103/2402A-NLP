# Phase 1: 基础设施 + 认证 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 RAG 智能助手项目的可运行骨架（目录结构、配置、Docker Compose 中间件、MySQL 数据访问、FastAPI 应用），并实现多用户注册/登录/JWT 认证。

**Architecture:** FastAPI 异步应用，SQLAlchemy 2.0 (async) + aiomysql 访问 MySQL，bcrypt 做密码哈希，PyJWT 签发无状态 token。业务逻辑分层为 `models`（ORM）/ `schemas`（请求响应）/ `services`（业务）/ `api`（路由与依赖注入）。测试用 SQLite(aiosqlite) 内存库，不依赖外部中间件。

**Tech Stack:** Python 3.11, FastAPI, uvicorn, SQLAlchemy 2.0 (async), aiomysql, aiosqlite(测试), pydantic v2, pydantic-settings, bcrypt, PyJWT, pytest, pytest-asyncio, httpx

## Global Constraints

- Python 版本下限：3.11
- 依赖通过 `requirements.txt` 管理，`pip install -r requirements.txt`
- 所有接口路径以 `/api` 为前缀
- 密码必须 bcrypt 哈希存储，绝不明文入库
- JWT 使用 HS256，secret 从环境变量读取（默认值仅用于本地开发）
- 数据库 URL 通过环境变量注入，测试时覆盖为 SQLite
- 所有 SQLAlchemy 模型继承同一个 `Base`
- 测试命令统一为 `pytest -v`（`pytest.ini` 中已设 `asyncio_mode=auto`）

---

## 文件结构总览

```
RAG_angent/
├── .gitignore
├── .env.example
├── requirements.txt
├── pytest.ini
├── docker-compose.yml          # MySQL / Redis / Milvus(etcd+minio)
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 实例 + 路由注册 + 启动建表
│   ├── config.py               # Settings (pydantic-settings)
│   ├── database.py             # async engine / session / Base
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # declarative Base
│   │   └── user.py             # User 模型
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py             # UserCreate / UserRead
│   │   └── auth.py             # Token
│   ├── core/
│   │   ├── __init__.py
│   │   └── security.py         # hash/verify password, JWT
│   ├── services/
│   │   ├── __init__.py
│   │   └── user_service.py     # 用户业务逻辑
│   └── api/
│       ├── __init__.py
│       ├── deps.py             # get_db / get_current_user
│       └── routes/
│           ├── __init__.py
│           ├── health.py       # GET /api/health
│           └── auth.py         # register / login / me
└── tests/
    ├── __init__.py
    ├── conftest.py             # 测试 fixtures + SQLite 覆盖
    ├── test_health.py
    └── test_auth.py
```

---

### Task 1: 项目骨架、配置与健康检查

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `app/main.py`
- Create: `app/api/__init__.py`
- Create: `app/api/routes/__init__.py`
- Create: `app/api/routes/health.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_health.py`

**Interfaces:**
- Consumes: 无（本任务为起点）
- Produces:
  - `app.main:app` — FastAPI 应用实例（后续所有路由都挂到它上面）
  - `app.config:get_settings()` → `Settings`（含 `app_name` 等字段）
  - `GET /api/health` → `{"status": "ok"}`

- [ ] **Step 1: 初始化 git 仓库**

```powershell
git init
```

- [ ] **Step 2: 写 `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
sqlalchemy[asyncio]==2.0.35
aiomysql==0.2.0
aiosqlite==0.20.0
pydantic==2.9.2
pydantic-settings==2.5.2
bcrypt==4.2.0
PyJWT==2.9.0
python-multipart==0.0.9
httpx==0.27.2
pytest==8.3.3
pytest-asyncio==0.24.0
```

- [ ] **Step 3: 写 `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 4: 写 `.gitignore`**

```
__pycache__/
*.pyc
.env
.venv/
venv/
.pytest_cache/
*.db
```

- [ ] **Step 5: 写 `.env.example`**

```dotenv
# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=rag
MYSQL_PASSWORD=rag_password
MYSQL_DATABASE=rag_assistant

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# JWT
JWT_SECRET=dev-secret-change-me
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080
```

- [ ] **Step 6: 写 `app/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Assistant"

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "rag"
    mysql_password: str = "rag_password"
    mysql_database: str = "rag_assistant"

    redis_host: str = "localhost"
    redis_port: int = 6379

    milvus_host: str = "localhost"
    milvus_port: int = 19530

    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 7: 写健康检查路由 `app/api/routes/health.py`**

```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 8: 写 `app/main.py`**

```python
from fastapi import FastAPI

from app.api.routes import health

app = FastAPI(title="RAG Assistant")
app.include_router(health.router, prefix="/api", tags=["health"])
```

- [ ] **Step 9: 写测试 `tests/conftest.py`**

```python
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

- [ ] **Step 10: 写测试 `tests/test_health.py`**

```python
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 11: 运行测试验证通过**

Run: `pytest -v`
Expected: PASS (1 test)

- [ ] **Step 12: 提交**

```powershell
git add .
git commit -m "feat: scaffold project skeleton with config and health check"
```

---

### Task 2: Docker Compose 中间件编排

**Files:**
- Create: `docker-compose.yml`

**Interfaces:**
- Consumes: 无
- Produces: 本地中间件 —— MySQL(3306)、Redis(6379)、Milvus(19530)，供开发与后续阶段使用

- [ ] **Step 1: 写 `docker-compose.yml`**

```yaml
services:
  mysql:
    image: mysql:8.0
    container_name: rag-mysql
    environment:
      MYSQL_ROOT_PASSWORD: root_password
      MYSQL_DATABASE: rag_assistant
      MYSQL_USER: rag
      MYSQL_PASSWORD: rag_password
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-prag_password"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: rag-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - etcd_data:/etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: rag-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data --console-address ":9001"
    volumes:
      - minio_data:/minio_data
    ports:
      - "9000:9000"
      - "9001:9001"

  milvus:
    image: milvusdb/milvus:v2.4.0
    container_name: rag-milvus
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio

volumes:
  mysql_data:
  redis_data:
  etcd_data:
  minio_data:
  milvus_data:
```

- [ ] **Step 2: 启动中间件并验证**

```powershell
docker compose up -d
docker compose ps
```

Expected: mysql / redis / etcd / minio / milvus 均为 running。MySQL 需等待 healthcheck 通过（约 30s）。

- [ ] **Step 3: 提交**

```powershell
git add docker-compose.yml
git commit -m "feat: add docker-compose for MySQL, Redis, Milvus"
```

---

### Task 3: 数据库连接与 User 模型

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/base.py`
- Create: `app/models/user.py`
- Create: `app/database.py`
- Modify: `app/main.py`（启动时建表 + lifespan）
- Modify: `tests/conftest.py`（SQLite 覆盖）

**Interfaces:**
- Consumes: `app.config:get_settings()`（Task 1）
- Produces:
  - `app.database:engine` — `AsyncEngine`，后续所有 DB 访问用它
  - `app.database:get_session()` — async context manager，yield `AsyncSession`
  - `app.database:Base` — SQLAlchemy declarative base
  - `app.models.user:User` — ORM 模型（`id`, `username`, `password_hash`, `created_at`）

- [ ] **Step 1: 写 `app/models/base.py`**

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

- [ ] **Step 2: 写 `app/models/user.py`**

```python
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
```

- [ ] **Step 3: 写 `app/models/__init__.py`**

```python
from app.models.user import User

__all__ = ["User"]
```

- [ ] **Step 4: 写 `app/database.py`**

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.models.base import Base

settings = get_settings()

_engine_kwargs: dict = {}
if settings.database_url.startswith("sqlite"):
    _engine_kwargs = {
        "poolclass": StaticPool,
        "connect_args": {"check_same_thread": False},
    }

engine = create_async_engine(settings.database_url, echo=False, **_engine_kwargs)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


@asynccontextmanager
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
```

- [ ] **Step 5: 修改 `app/main.py` 加入 lifespan 建表**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with init_db():
        yield


app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.include_router(health.router, prefix="/api", tags=["health"])
```

- [ ] **Step 6: 修改 `tests/conftest.py` 用 SQLite 覆盖 + 建表**

```python
import os

# 必须在导入 app 前设置覆盖，engine 才会指向内存 SQLite
os.environ["DATABASE_URL_OVERRIDE"] = "sqlite+aiosqlite:///:memory:"

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

- [ ] **Step 7: 让 `config.py` 支持 URL 覆盖**

在 `app/config.py` 的 `Settings` 中新增字段并调整 `database_url`：

```python
    # 在 Settings 内新增（放在 model_config 前）：
    database_url_override: str | None = None

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )
```

- [ ] **Step 8: 运行测试验证建表与连接逻辑不报错**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 9: 提交**

```powershell
git add app/models app/database.py app/main.py app/config.py tests/conftest.py
git commit -m "feat: add async database layer and User model"
```

---

### Task 4: 安全模块（密码哈希 + JWT）

**Files:**
- Create: `app/core/__init__.py`
- Create: `app/core/security.py`
- Create: `tests/test_security.py`

**Interfaces:**
- Consumes: `app.config:get_settings()`（Task 1）
- Produces:
  - `hash_password(plain: str) -> str`
  - `verify_password(plain: str, hashed: str) -> bool`
  - `create_access_token(user_id: int) -> str`
  - `decode_access_token(token: str) -> int | None`（有效返回 user_id，无效返回 None）

- [ ] **Step 1: 写失败测试 `tests/test_security.py`**

```python
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password():
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_create_and_decode_token():
    token = create_access_token(42)
    assert decode_access_token(token) == 42


def test_decode_invalid_token_returns_none():
    assert decode_access_token("not-a-valid-token") is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_security.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.security'`

- [ ] **Step 3: 写 `app/core/security.py`**

```python
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import get_settings


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: int) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_security.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 提交**

```powershell
git add app/core tests/test_security.py
git commit -m "feat: add password hashing and JWT utilities"
```

---

### Task 5: 用户服务层与注册接口

**Files:**
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/user.py`
- Create: `app/schemas/auth.py`
- Create: `app/services/__init__.py`
- Create: `app/services/user_service.py`
- Create: `app/api/deps.py`
- Create: `app/api/routes/auth.py`
- Modify: `app/main.py`（注册 auth 路由）
- Modify: `tests/conftest.py`（提供 DB session 依赖覆盖）
- Create: `tests/test_auth.py`

**Interfaces:**
- Consumes: `User`（Task 3）、`hash_password` / `create_access_token`（Task 4）
- Produces:
  - `app.services.user_service:create_user(db, username, password) -> User`（用户名重复抛 `ValueError`）
  - `app.services.user_service:get_user_by_username(db, username) -> User | None`
  - `app.api.deps:get_db()` — FastAPI 依赖，yield `AsyncSession`
  - `POST /api/auth/register` — 入参 `{username, password}`，出参 `UserRead`

- [ ] **Step 1: 写 `app/schemas/user.py`**

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class UserRead(BaseModel):
    id: int
    username: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 2: 写 `app/schemas/auth.py`**

```python
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

- [ ] **Step 3: 写 `app/schemas/__init__.py`**

```python
from app.schemas.user import UserCreate, UserRead
from app.schemas.auth import Token

__all__ = ["UserCreate", "UserRead", "Token"]
```

- [ ] **Step 4: 写 `app/services/user_service.py`**

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.user import User


async def create_user(db: AsyncSession, username: str, password: str) -> User:
    existing = await get_user_by_username(db, username)
    if existing is not None:
        raise ValueError("用户名已存在")
    user = User(username=username, password_hash=hash_password(password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()
```

- [ ] **Step 5: 写 `app/api/deps.py`**

```python
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import SessionLocal


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
```

- [ ] **Step 6: 写 `app/api/routes/auth.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.schemas.user import UserCreate, UserRead
from app.services import user_service

router = APIRouter()


@router.post("/auth/register", response_model=UserRead, status_code=201)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        return await user_service.create_user(db, payload.username, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
```

- [ ] **Step 7: 修改 `app/main.py` 注册 auth 路由**

```python
from app.api.routes import auth, health

# ... lifespan 定义保持不变 ...

app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
```

- [ ] **Step 8: 修改 `tests/conftest.py`，让测试用独立内存库并覆盖依赖**

```python
import os

os.environ["DATABASE_URL_OVERRIDE"] = "sqlite+aiosqlite:///:memory:"

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
async def _setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

> 注意：`app/database.py` 在 import 时就会创建 engine。`conftest.py` 顶部先设置 `DATABASE_URL_OVERRIDE` 环境变量，再 import `app.main`，可确保 engine 指向内存 SQLite。`_setup_db` 用 `StaticPool` 保证每次测试用同一个内存库。

- [ ] **Step 9: 写测试 `tests/test_auth.py`**

```python
async def test_register_success(client):
    resp = await client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "secret123"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["username"] == "alice"
    assert "password" not in body


async def test_register_duplicate_username(client):
    payload = {"username": "alice", "password": "secret123"}
    await client.post("/api/auth/register", json=payload)
    resp = await client.post("/api/auth/register", json=payload)
    assert resp.status_code == 409
```

- [ ] **Step 10: 运行测试确认通过**

Run: `pytest -v`
Expected: PASS（health 1 + security 3 + auth 2 = 6 tests）

- [ ] **Step 11: 提交**

```powershell
git add app/schemas app/services app/api/deps.py app/api/routes/auth.py app/main.py tests/conftest.py tests/test_auth.py
git commit -m "feat: add user registration endpoint"
```

---

### Task 6: 登录接口与当前用户接口

**Files:**
- Modify: `app/services/user_service.py`（加 `authenticate_user` / `issue_token`）
- Modify: `app/schemas/auth.py`（加 `LoginRequest`）
- Modify: `app/api/deps.py`（加 `get_current_user`）
- Modify: `app/api/routes/auth.py`（加 login / me）
- Modify: `tests/test_auth.py`（加测试）

**Interfaces:**
- Consumes: `get_user_by_username`（Task 5）、`verify_password` / `create_access_token` / `decode_access_token`（Task 4）
- Produces:
  - `app.services.user_service:authenticate_user(db, username, password) -> User | None`
  - `app.services.user_service:issue_token(user_id) -> str`
  - `app.api.deps:get_current_user(credentials, db) -> User`（缺失/无效凭证抛 401）
  - `POST /api/auth/login` → `Token`（入参 `LoginRequest`）
  - `GET /api/auth/me` → `UserRead`

- [ ] **Step 1: 写失败测试（在 `tests/test_auth.py` 追加）**

```python
async def test_login_success(client):
    await client.post(
        "/api/auth/register",
        json={"username": "bob", "password": "secret123"},
    )
    resp = await client.post(
        "/api/auth/login",
        json={"username": "bob", "password": "secret123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


async def test_login_wrong_password(client):
    await client.post(
        "/api/auth/register",
        json={"username": "bob", "password": "secret123"},
    )
    resp = await client.post(
        "/api/auth/login",
        json={"username": "bob", "password": "wrongpass"},
    )
    assert resp.status_code == 401


async def test_me_requires_token(client):
    resp = await client.get("/api/auth/me")
    assert resp.status_code == 401


async def test_me_returns_user(client):
    await client.post(
        "/api/auth/register",
        json={"username": "carol", "password": "secret123"},
    )
    login = await client.post(
        "/api/auth/login",
        json={"username": "carol", "password": "secret123"},
    )
    token = login.json()["access_token"]
    resp = await client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "carol"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_auth.py -v`
Expected: FAIL（login/me 相关测试报 404/未定义）

- [ ] **Step 3: 在 `app/schemas/auth.py` 追加 `LoginRequest`**

```python
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

> 说明：登录入参用独立的 `LoginRequest`，不做长度校验（区别于注册的 `UserCreate`），这样错误密码统一走 401 而非 422。

- [ ] **Step 4: 在 `app/services/user_service.py` 追加 `authenticate_user` / `issue_token`**

将文件顶部 import 改为（合并，避免重复导入）：

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
```

在文件末尾追加：

```python
async def authenticate_user(
    db: AsyncSession, username: str, password: str
) -> User | None:
    user = await get_user_by_username(db, username)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def issue_token(user_id: int) -> str:
    return create_access_token(user_id)
```

- [ ] **Step 5: 在 `app/api/deps.py` 追加 `get_current_user`**

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少认证凭证")
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="无效或过期的凭证")
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user
```

- [ ] **Step 6: 在 `app/api/routes/auth.py` 追加 login / me**

将文件顶部 import 改为：

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.schemas.auth import LoginRequest, Token
from app.schemas.user import UserCreate, UserRead
from app.services import user_service
```

在文件末尾追加：

```python
@router.post("/auth/login", response_model=Token)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await user_service.authenticate_user(
        db, payload.username, payload.password
    )
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return Token(access_token=user_service.issue_token(user.id))


@router.get("/auth/me", response_model=UserRead)
async def me(current_user=Depends(get_current_user)):
    return current_user
```

- [ ] **Step 7: 运行测试确认通过**

Run: `pytest -v`
Expected: PASS（共 10 tests）

- [ ] **Step 8: 提交**

```powershell
git add app/services/user_service.py app/schemas/auth.py app/api/deps.py app/api/routes/auth.py tests/test_auth.py
git commit -m "feat: add login and current-user endpoints"
```

---

### Task 7: 端到端验证与阶段收尾

**Files:**
- Modify: `README.md`（Create，项目说明与启动步骤）

**Interfaces:**
- Consumes: 全部前述任务
- Produces: 可运行的阶段 1 交付物 + 启动文档

- [ ] **Step 1: 写 `README.md`**

```markdown
# RAG Assistant

多用户、多角色的 RAG 智能聊天助手。

## 启动（开发环境）

1. 启动中间件：

```powershell
docker compose up -d
```

2. 安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. 复制配置：

```powershell
Copy-Item .env.example .env
```

4. 启动应用：

```powershell
uvicorn app.main:app --reload
```

访问 http://localhost:8000/docs 查看接口文档。

## 测试

```powershell
pytest -v
```
```

- [ ] **Step 2: 手动验证完整流程（可选，需 MySQL 已启动）**

```powershell
.\.venv\Scripts\python.exe -c "from app.database import engine; import asyncio; from app.models.base import Base; asyncio.run(engine.begin().__aenter__())"
```

或直接启动 uvicorn 后，用 `/docs` 交互式调用 register → login → me。

- [ ] **Step 3: 提交**

```powershell
git add README.md
git commit -m "docs: add project README with startup steps"
```

---

## 阶段 1 完成后自检清单

- [ ] `pytest -v` 全绿（10 tests）
- [ ] `docker compose up -d` 起 MySQL/Redis/Milvus 无报错
- [ ] 本地 `uvicorn` 启动后可注册、登录、访问 `/api/auth/me`

---

## 后续阶段高层路线图（暂不展开，执行前再细化）

- **Phase 2 — 模型接入抽象层 + 角色管理**：三类模型统一抽象（LLM / 文本 embedding / 视觉 embedding，OpenAI 兼容），角色 CRUD + 预设模板 + 角色级模型配置，角色表与数据隔离。
- **Phase 3 — 知识库入库**：文件上传（txt/pdf/图片）、文本解析分块、OCR、视觉向量、Milvus 写入与删除、异步任务队列。
- **Phase 4 — 对话 + RAG 检索 + 记忆**：会话管理、SSE 流式对话、知识库+长期记忆联合检索、Redis 短期记忆 + 摘要压缩、长期记忆异步沉淀。
- **Phase 5 — 前端 + 部署收尾**：登录页、角色管理页、知识库管理页、聊天页（流式渲染）、App 容器化进 Docker Compose、端到端集成测试与文档。
