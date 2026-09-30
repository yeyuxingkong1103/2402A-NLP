# Phase 2: 模型接入抽象层 + 角色管理 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立三类模型的统一接入抽象（文本 LLM、文本 Embedding、视觉 Embedding），并实现角色（Character）的 CRUD、预设模板与角色级模型配置，所有数据按用户隔离。

**Architecture:** LLM 与文本 Embedding 走 OpenAI 兼容协议，统一用 `openai` 库的 `AsyncOpenAI`（`base_url` 可覆盖，故 Ollama / vLLM / 云端 API 通用）。视觉 Embedding 走独立的多模态 API，用 `httpx` 实现可替换抽象。角色配置持久化在 MySQL `characters` 表，模板硬编码为 Python 常量。

**Tech Stack:** 新增 `openai==1.51.0`。复用 Phase 1 的 FastAPI / SQLAlchemy / pydantic / pytest。

## Global Constraints

- Python 版本下限：3.11
- 所有接口路径以 `/api` 为前缀
- API Key 一律从环境变量读取，绝不入 MySQL、不入前端、不入日志
- 角色相关接口全部需要 `get_current_user` 鉴权，且查询强制带 `user_id` 过滤
- 模型客户端构造统一走 `build_*_from_character()` 工厂函数，路由层不直接 new 客户端
- 测试命令统一为 `pytest -v`

## 与 Phase 1 的接口约定（本阶段复用，不再重新定义）

- `app.config:get_settings()` → `Settings`（本阶段在其上新增字段）
- `app.database:SessionLocal` / `engine` / `Base`
- `app.models.user:User`（`id`, `username`, `password_hash`, `created_at`）
- `app.api.deps:get_db` / `get_current_user`
- `app.models.base:Base`（所有新模型继承它）

---

## 文件结构总览（本阶段新增/修改）

```
app/
├── config.py                        # 修改：新增模型相关字段
├── models/
│   ├── character.py                 # Character ORM
│   └── __init__.py                  # 修改：导出 Character
├── core/
│   ├── llm.py                       # LLMClient + build_llm_from_character
│   ├── embeddings.py                # EmbeddingClient
│   ├── vision.py                    # VisionEmbeddingClient
│   └── character_templates.py       # CHARACTER_TEMPLATES 常量
├── schemas/
│   ├── character.py                 # CharacterCreate/Update/Read
│   └── __init__.py                  # 修改：导出
├── services/
│   └── character_service.py         # 角色业务逻辑
└── api/routes/
    ├── characters.py                # 角色 CRUD + 模板
    └── __init__.py                  # 修改
tests/
├── test_llm.py
├── test_embeddings.py
├── test_vision.py
└── test_characters.py
.env.example                          # 修改：新增模型字段
requirements.txt                      # 修改：新增 openai
```

---

### Task 1: 模型配置项与依赖

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `app/config.py`

**Interfaces:**
- Consumes: `Settings`（Phase 1）
- Produces: `Settings` 新增字段 `llm_base_url` / `llm_api_key` / `embedding_model` / `embedding_base_url` / `embedding_api_key` / `vision_embedding_model` / `vision_embedding_base_url` / `vision_embedding_api_key`

- [ ] **Step 1: 在 `requirements.txt` 追加 openai**

```powershell
Add-Content requirements.txt "openai==1.51.0"
```

（结果文件末尾新增一行 `openai==1.51.0`，其余不变。）

- [ ] **Step 2: 在 `.env.example` 追加模型配置**

```powershell
Add-Content .env.example @'

# LLM (OpenAI 兼容；本地 Ollama 可设 http://localhost:11434/v1, api_key 填 ollama)
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key

# 文本 Embedding
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-key

# 视觉 Embedding（多模态 API，用于图片向量）
VISION_EMBEDDING_MODEL=your-vision-embedding-model
VISION_EMBEDDING_BASE_URL=https://your-vision-api.example.com/v1
VISION_EMBEDDING_API_KEY=your-vision-key
'@
```

- [ ] **Step 3: 修改 `app/config.py`，在 `Settings` 中新增字段**

在 `jwt_expire_minutes` 之后、`model_config` 之前插入：

```python
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""

    vision_embedding_model: str = ""
    vision_embedding_base_url: str = ""
    vision_embedding_api_key: str = ""
```

- [ ] **Step 4: 运行既有测试确保未破坏**

Run: `pytest -v`
Expected: PASS（10 tests）

- [ ] **Step 5: 提交**

```powershell
git add requirements.txt .env.example app/config.py
git commit -m "feat: add model-related settings for LLM and embeddings"
```

---

### Task 2: LLM 客户端（流式生成）

**Files:**
- Create: `app/core/llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Consumes: `app.config:get_settings()`（Phase 1）
- Produces:
  - `class LLMClient` — `__init__(self, *, model, base_url, api_key, temperature=0.7, top_p=1.0, max_tokens=2048)`
  - `LLMClient.chat_stream(self, messages: list[dict]) -> AsyncIterator[str]`（messages 为 OpenAI 格式）
  - `build_llm_from_character(character) -> LLMClient`（character 需含 `model_name`/`base_url`/`temperature`/`top_p`/`max_tokens` 属性）

- [ ] **Step 1: 写失败测试 `tests/test_llm.py`**

```python
import pytest

from app.core.llm import LLMClient, build_llm_from_character


class _FakeCharacter:
    model_name = "gpt-test"
    base_url = "http://localhost:11434/v1"
    temperature = 0.3
    top_p = 0.9
    max_tokens = 512


async def test_build_llm_from_character_reads_fields(monkeypatch):
    captured = {}

    class _FakeOpenAI:
        def __init__(self, base_url, api_key):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr("app.core.llm.AsyncOpenAI", _FakeOpenAI)
    client = build_llm_from_character(_FakeCharacter())
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert client.model == "gpt-test"
    assert client.temperature == 0.3
    assert client.top_p == 0.9
    assert client.max_tokens == 512


async def test_chat_stream_yields_content(monkeypatch):
    class _Chunk:
        def __init__(self, text):
            self.choices = [_Choice(text)]
            self.choices[0].delta.content = text

    class _Choice:
        def __init__(self, text):
            self.delta = type("D", (), {"content": text})()

    class _Stream:
        def __aiter__(self):
            self._items = [_Chunk("你"), _Chunk("好")]
            self._i = 0
            return self

        async def __anext__(self):
            if self._i >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._i]
            self._i += 1
            return item

    class _Completions:
        async def create(self, **kwargs):
            return _Stream()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr("app.core.llm.AsyncOpenAI", _FakeOpenAI)
    client = LLMClient(model="gpt-test", base_url="http://x", api_key="k")
    out = []
    async for piece in client.chat_stream([{"role": "user", "content": "hi"}]):
        out.append(piece)
    assert out == ["你", "好"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.llm'`

- [ ] **Step 3: 写 `app/core/llm.py`**

```python
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import get_settings


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def build_llm_from_character(character) -> LLMClient:
    settings = get_settings()
    base_url = character.base_url or settings.llm_base_url
    return LLMClient(
        model=character.model_name,
        base_url=base_url,
        api_key=settings.llm_api_key,
        temperature=character.temperature,
        top_p=character.top_p,
        max_tokens=character.max_tokens,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```powershell
git add app/core/llm.py tests/test_llm.py
git commit -m "feat: add OpenAI-compatible LLM client with streaming"
```

---

### Task 3: 文本 Embedding 客户端

**Files:**
- Create: `app/core/embeddings.py`
- Create: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `app.config:get_settings()`（Phase 1）
- Produces:
  - `class EmbeddingClient` — `__init__(self, *, model, base_url, api_key)`
  - `EmbeddingClient.embed(self, texts: list[str]) -> list[list[float]]`

- [ ] **Step 1: 写失败测试 `tests/test_embeddings.py`**

```python
from app.core.embeddings import EmbeddingClient


async def test_embed_returns_vectors(monkeypatch):
    class _Embeddings:
        async def create(self, **kwargs):
            data = [type("D", (), {"embedding": [0.1, 0.2]})() for _ in kwargs["input"]]
            return type("R", (), {"data": data})()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.embeddings = _Embeddings()

    monkeypatch.setattr("app.core.embeddings.AsyncOpenAI", _FakeOpenAI)
    client = EmbeddingClient(model="m", base_url="http://x", api_key="k")
    result = await client.embed(["a", "b"])
    assert result == [[0.1, 0.2], [0.1, 0.2]]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.embeddings'`

- [ ] **Step 3: 写 `app/core/embeddings.py`**

```python
from openai import AsyncOpenAI


class EmbeddingClient:
    def __init__(self, *, model: str, base_url: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: 提交**

```powershell
git add app/core/embeddings.py tests/test_embeddings.py
git commit -m "feat: add text embedding client"
```

---

### Task 4: 视觉 Embedding 客户端

**Files:**
- Create: `app/core/vision.py`
- Create: `tests/test_vision.py`

**Interfaces:**
- Consumes: `app.config:get_settings()`（Phase 1）
- Produces:
  - `class VisionEmbeddingClient` — `__init__(self, *, model, base_url, api_key)`
  - `VisionEmbeddingClient.embed_image(self, image_path: str) -> list[float]`

> 说明：视觉嵌入走多模态 API。本实现采用「OpenAI embeddings 风格 + base64 图片输入」的通用协议（`POST {base_url}/embeddings`，`input` 为 base64 字符串）。若厂商协议不同，只需替换本类的 `embed_image` 实现，不影响调用方。

- [ ] **Step 1: 写失败测试 `tests/test_vision.py`**

```python
import base64

from app.core.vision import VisionEmbeddingClient


async def test_embed_image_sends_base64(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG fake")

    captured = {}

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [0.5, 0.6]}]}

    async def _fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return _Response()

    class _FakeClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]

        async def post(self, url, **kwargs):
            return await _fake_post(url, **kwargs)

    monkeypatch.setattr("app.core.vision.httpx.AsyncClient", _FakeClient)
    client = VisionEmbeddingClient(model="m", base_url="http://v", api_key="k")
    result = await client.embed_image(str(img))

    assert result == [0.5, 0.6]
    assert captured["url"] == "http://v/embeddings"
    sent = captured["json"]["input"]
    assert base64.b64decode(sent) == b"\x89PNG fake"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_vision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.vision'`

- [ ] **Step 3: 写 `app/core/vision.py`**

```python
import base64

import httpx


class VisionEmbeddingClient:
    def __init__(self, *, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def embed_image(self, image_path: str) -> list[float]:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(
                "/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": b64},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_vision.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: 提交**

```powershell
git add app/core/vision.py tests/test_vision.py
git commit -m "feat: add vision embedding client"
```

---

### Task 5: Character 模型与角色模板

**Files:**
- Create: `app/models/character.py`
- Modify: `app/models/__init__.py`
- Create: `app/core/character_templates.py`
- Create: `tests/test_character_templates.py`

**Interfaces:**
- Consumes: `app.models.base:Base`（Phase 1）、`app.models.user:User`（Phase 1）
- Produces:
  - `app.models.character:Character`（`id`, `user_id`, `name`, `system_prompt`, `model_name`, `base_url`, `temperature`, `top_p`, `max_tokens`, `created_at`, `updated_at`）
  - `app.core.character_templates:CHARACTER_TEMPLATES` — `list[dict]`，每项含 `key`/`name`/`description`/`system_prompt`

- [ ] **Step 1: 写失败测试 `tests/test_character_templates.py`**

```python
from app.core.character_templates import CHARACTER_TEMPLATES


def test_templates_cover_required_roles():
    keys = {t["key"] for t in CHARACTER_TEMPLATES}
    assert {
        "npc_friend",
        "virtual_friend",
        "tcm_doctor",
        "psychologist",
        "lawyer",
        "financial_advisor",
    }.issubset(keys)


def test_each_template_has_required_fields():
    for t in CHARACTER_TEMPLATES:
        assert t["key"]
        assert t["name"]
        assert t["system_prompt"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_character_templates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.character_templates'`

- [ ] **Step 3: 写 `app/models/character.py`**

```python
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)
    top_p: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    max_tokens: Mapped[int] = mapped_column(Integer, default=2048, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
```

- [ ] **Step 4: 修改 `app/models/__init__.py` 导出 Character**

```python
from app.models.user import User
from app.models.character import Character

__all__ = ["User", "Character"]
```

- [ ] **Step 5: 写 `app/core/character_templates.py`**

```python
CHARACTER_TEMPLATES = [
    {
        "key": "npc_friend",
        "name": "NPC 朋友",
        "description": "有鲜明性格与背景故事的虚拟人物，可陪你闲聊。",
        "system_prompt": (
            "你是一个有鲜明性格的虚拟朋友，请始终以这个人设的口气说话，"
            "保持人物性格一致，自然、生动、有情绪。"
        ),
    },
    {
        "key": "virtual_friend",
        "name": "虚拟朋友",
        "description": "像 Character.ai 那样的陪伴型聊天对象。",
        "system_prompt": (
            "你是用户的好朋友，友好、耐心、善解人意。"
            "认真倾听并给出真诚的回应，让对话自然流动。"
        ),
    },
    {
        "key": "tcm_doctor",
        "name": "中医",
        "description": "中医健康咨询助手，结合知识库给出建议。",
        "system_prompt": (
            "你是一位经验丰富的中医师，基于用户提供的知识库内容，"
            "用通俗的语言解答健康与养生问题。遇到超出知识库或需要就医的情况，"
            "务必提醒用户咨询线下专业医生，不得给出确定性的诊断。"
        ),
    },
    {
        "key": "psychologist",
        "name": "心理医生",
        "description": "心理健康支持与倾听。",
        "system_prompt": (
            "你是一位温和、专业的心理支持者，善于倾听和共情。"
            "基于知识库提供心理健康方面的信息与建议；遇到危机情形，"
            "应建议用户寻求专业机构或紧急帮助。"
        ),
    },
    {
        "key": "lawyer",
        "name": "律师",
        "description": "法律咨询助手。",
        "system_prompt": (
            "你是一位严谨的法律顾问，基于用户提供的知识库与通用法律常识回答问题，"
            "但不得构成正式法律意见，重要事项应建议用户咨询执业律师。"
        ),
    },
    {
        "key": "financial_advisor",
        "name": "金融理财师",
        "description": "理财规划与金融知识助手。",
        "system_prompt": (
            "你是一位持证金融理财师，基于知识库提供理财规划与金融知识，"
            "不得做出具体的投资承诺，并应提示投资风险。"
        ),
    },
]
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_character_templates.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: 提交**

```powershell
git add app/models/character.py app/models/__init__.py app/core/character_templates.py tests/test_character_templates.py
git commit -m "feat: add Character model and preset templates"
```

---

### Task 6: 角色 CRUD 接口

**Files:**
- Create: `app/schemas/character.py`
- Modify: `app/schemas/__init__.py`
- Create: `app/services/character_service.py`
- Create: `app/api/routes/characters.py`
- Modify: `app/api/routes/__init__.py`
- Modify: `app/main.py`
- Create: `tests/test_characters.py`

**Interfaces:**
- Consumes: `Character`（Task 5）、`get_db` / `get_current_user`（Phase 1）
- Produces:
  - `app.services.character_service:create_character(db, user_id, data) -> Character`
  - `app.services.character_service:list_characters(db, user_id) -> list[Character]`
  - `app.services.character_service:get_character(db, user_id, character_id) -> Character | None`
  - `app.services.character_service:update_character(db, user_id, character_id, data) -> Character | None`
  - `app.services.character_service:delete_character(db, user_id, character_id) -> bool`
  - `GET /api/characters/templates` → 模板列表
  - `POST /api/characters` → `CharacterRead`（可传 `template_key`）
  - `GET /api/characters` → `list[CharacterRead]`
  - `GET /api/characters/{id}` → `CharacterRead`
  - `PATCH /api/characters/{id}` → `CharacterRead`
  - `DELETE /api/characters/{id}` → 204

- [ ] **Step 1: 写 `app/schemas/character.py`**

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CharacterCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    system_prompt: str = Field(min_length=1)
    model_name: str = Field(min_length=1, max_length=128)
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048
    template_key: str | None = None


class CharacterUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    system_prompt: str | None = Field(default=None, min_length=1)
    model_name: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class CharacterRead(BaseModel):
    id: int
    name: str
    system_prompt: str
    model_name: str
    base_url: str | None
    temperature: float
    top_p: float
    max_tokens: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 2: 修改 `app/schemas/__init__.py`**

```python
from app.schemas.user import UserCreate, UserRead
from app.schemas.auth import Token
from app.schemas.character import CharacterCreate, CharacterRead, CharacterUpdate

__all__ = [
    "UserCreate",
    "UserRead",
    "Token",
    "CharacterCreate",
    "CharacterRead",
    "CharacterUpdate",
]
```

- [ ] **Step 3: 写 `app/services/character_service.py`**

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.character_templates import CHARACTER_TEMPLATES
from app.models.character import Character
from app.schemas.character import CharacterCreate, CharacterUpdate


async def create_character(
    db: AsyncSession, user_id: int, data: CharacterCreate
) -> Character:
    system_prompt = data.system_prompt
    if data.template_key and not data.system_prompt:
        for t in CHARACTER_TEMPLATES:
            if t["key"] == data.template_key:
                system_prompt = t["system_prompt"]
                break
    char = Character(
        user_id=user_id,
        name=data.name,
        system_prompt=system_prompt,
        model_name=data.model_name,
        base_url=data.base_url,
        temperature=data.temperature,
        top_p=data.top_p,
        max_tokens=data.max_tokens,
    )
    db.add(char)
    await db.commit()
    await db.refresh(char)
    return char


async def list_characters(db: AsyncSession, user_id: int) -> list[Character]:
    result = await db.execute(
        select(Character).where(Character.user_id == user_id).order_by(Character.id)
    )
    return list(result.scalars().all())


async def get_character(
    db: AsyncSession, user_id: int, character_id: int
) -> Character | None:
    result = await db.execute(
        select(Character).where(
            Character.id == character_id, Character.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def update_character(
    db: AsyncSession, user_id: int, character_id: int, data: CharacterUpdate
) -> Character | None:
    char = await get_character(db, user_id, character_id)
    if char is None:
        return None
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(char, field, value)
    await db.commit()
    await db.refresh(char)
    return char


async def delete_character(db: AsyncSession, user_id: int, character_id: int) -> bool:
    char = await get_character(db, user_id, character_id)
    if char is None:
        return False
    await db.delete(char)
    await db.commit()
    return True
```

- [ ] **Step 4: 写 `app/api/routes/characters.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.character_templates import CHARACTER_TEMPLATES
from app.models.user import User
from app.schemas.character import CharacterCreate, CharacterRead, CharacterUpdate
from app.services import character_service

router = APIRouter()


@router.get("/characters/templates")
async def list_templates(current_user: User = Depends(get_current_user)):
    return [
        {
            "key": t["key"],
            "name": t["name"],
            "description": t["description"],
            "system_prompt": t["system_prompt"],
        }
        for t in CHARACTER_TEMPLATES
    ]


@router.post("/characters", response_model=CharacterRead, status_code=201)
async def create_character(
    payload: CharacterCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await character_service.create_character(db, current_user.id, payload)


@router.get("/characters", response_model=list[CharacterRead])
async def list_characters(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await character_service.list_characters(db, current_user.id)


@router.get("/characters/{character_id}", response_model=CharacterRead)
async def get_character(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return char


@router.patch("/characters/{character_id}", response_model=CharacterRead)
async def update_character(
    character_id: int,
    payload: CharacterUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.update_character(
        db, current_user.id, character_id, payload
    )
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return char


@router.delete("/characters/{character_id}", status_code=204)
async def delete_character(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ok = await character_service.delete_character(db, current_user.id, character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="角色不存在")
```

- [ ] **Step 5: 修改 `app/main.py` 注册 characters 路由**

将 import 与 include_router 改为：

```python
from app.api.routes import auth, characters, health

# ... lifespan 定义保持不变 ...

app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(characters.router, prefix="/api", tags=["characters"])
```

- [ ] **Step 6: 写失败测试 `tests/test_characters.py`**

```python
async def _register_and_login(client, username):
    await client.post(
        "/api/auth/register", json={"username": username, "password": "secret123"}
    )
    resp = await client.post(
        "/api/auth/login", json={"username": username, "password": "secret123"}
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def test_create_and_list_character(client):
    headers = await _register_and_login(client, "charuser")
    resp = await client.post(
        "/api/characters",
        json={"name": "中医", "system_prompt": "你是中医", "model_name": "gpt-test"},
        headers=headers,
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "中医"

    resp = await client.get("/api/characters", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1


async def test_create_from_template(client):
    headers = await _register_and_login(client, "charuser2")
    resp = await client.post(
        "/api/characters",
        json={"name": "医生", "system_prompt": "", "model_name": "gpt-test", "template_key": "tcm_doctor"},
        headers=headers,
    )
    assert resp.status_code == 201
    assert "中医" in resp.json()["system_prompt"]


async def test_character_isolation_between_users(client):
    h1 = await _register_and_login(client, "user_a")
    h2 = await _register_and_login(client, "user_b")
    await client.post(
        "/api/characters",
        json={"name": "A的角色", "system_prompt": "x", "model_name": "gpt-test"},
        headers=h1,
    )
    resp = await client.get("/api/characters", headers=h2)
    assert resp.json() == []
```

- [ ] **Step 7: 运行测试确认失败**

Run: `pytest tests/test_characters.py -v`
Expected: FAIL（路由 404）

- [ ] **Step 8: 运行全部测试确认通过**

Run: `pytest -v`
Expected: PASS（10 + 2 + 1 + 1 + 2 + 3 = 19 tests）

- [ ] **Step 9: 提交**

```powershell
git add app/schemas/character.py app/schemas/__init__.py app/services/character_service.py app/api/routes/characters.py app/main.py tests/test_characters.py
git commit -m "feat: add character CRUD endpoints with per-user isolation"
```

---

## Phase 2 完成后自检清单

- [ ] `pytest -v` 全绿（19 tests）
- [ ] 可创建角色（含从模板创建）、列表、详情、更新、删除
- [ ] 两个不同用户的角色列表互不可见
- [ ] LLM / 文本 embedding / 视觉 embedding 三个客户端可独立构造，参数正确
