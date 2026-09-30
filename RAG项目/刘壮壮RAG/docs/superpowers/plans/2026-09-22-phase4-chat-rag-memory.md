# Phase 4: 对话 + RAG 检索 + 记忆 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现多会话对话：会话 CRUD、SSE 流式多轮对话、知识库 + 长期记忆联合检索、Redis 短期记忆（含摘要压缩）、长期记忆异步沉淀。

**Architecture:** 对话内容存 Redis（短期记忆），会话元数据存 MySQL。每轮：读 Redis 历史 → 检索 Milvus（知识库 `knowledge_chunks` + 长期记忆 `long_term_memory`）→ 组装 prompt → LLM 流式 SSE 返回 → 写回 Redis → 达到阈值时异步投递 `extract_memory` 任务。摘要压缩在窗口超限时用 LLM 压缩最早消息。

**Tech Stack:** 新增 `redis==5.0.8`、`fakeredis==2.23.2`（测试）。复用 Phase 2 的 `LLMClient` / `EmbeddingClient`，Phase 3 的 `VectorStore`（本阶段扩展 memory collection）。

## Global Constraints

- 所有接口路径以 `/api` 为前缀，需 `get_current_user` 鉴权 + `user_id` 过滤
- 对话消息存 Redis，key 格式 `chat:{conversation_id}:messages`，值为 JSON 字符串（`{"role","content"}`）
- 短期记忆窗口默认 20 轮（40 条消息），超限时压缩最早消息为摘要
- 长期记忆沉淀阈值：会话内用户消息累积达 10 条时触发一次 `extract_memory`
- 检索 top_k 默认 4；知识库与长期记忆分别检索后合并去重
- Redis 客户端统一经 `app.core.redis_client:get_redis()` 获取，测试用 fakeredis 替换

## 与 Phase 1/2/3 的接口约定（复用，不重新定义）

- `get_settings()`、`SessionLocal`、`get_db` / `get_current_user`
- `User` / `Character` / `KnowledgeFile` 模型
- `LLMClient`（`chat_stream`）、`build_llm_from_character`、`EmbeddingClient`（`embed`）
- `VectorStore`（`search_text` / `insert_text_chunks` / `delete_by_file` / `delete_by_character` / `ensure_collections`）
- arq worker：`app.worker.tasks.process_file`、`python -m app.worker.main`

---

## 文件结构总览（本阶段新增/修改）

```
app/
├── models/
│   ├── conversation.py              # Conversation ORM
│   └── __init__.py                  # 修改：导出
├── core/
│   ├── redis_client.py              # get_redis()
│   ├── memory.py                    # ShortTermMemory
│   ├── retriever.py                 # 检索器
│   └── vector_store.py              # 修改：新增 memory collection 方法
├── schemas/
│   ├── conversation.py              # ConversationCreate/Read + ChatRequest
│   └── __init__.py                  # 修改：导出
├── services/
│   ├── conversation_service.py      # 会话 CRUD
│   └── chat_service.py              # 组装/检索/流式生成/压缩
├── worker/
│   └── tasks.py                     # 修改：新增 extract_memory
├── api/routes/
│   ├── conversations.py             # 会话 CRUD
│   └── chat.py                      # SSE 对话
└── main.py                          # 修改：注册路由
tests/
├── test_memory.py
├── test_retriever.py
├── test_conversations.py
└── test_chat.py
```

---

### Task 1: 依赖与 Conversation 模型

**Files:**
- Modify: `requirements.txt`
- Create: `app/models/conversation.py`
- Modify: `app/models/__init__.py`

**Interfaces:**
- Consumes: `Base`（Phase 1）
- Produces: `app.models.conversation:Conversation`（`id`, `user_id`, `character_id`, `title`, `created_at`）

- [ ] **Step 1: 追加依赖**

```powershell
Add-Content requirements.txt @'
redis==5.0.8
fakeredis==2.23.2
'@
```

- [ ] **Step 2: 写 `app/models/conversation.py`**

```python
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    character_id: Mapped[int] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(128), default="新对话", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
```

- [ ] **Step 3: 修改 `app/models/__init__.py`**

```python
from app.models.user import User
from app.models.character import Character
from app.models.knowledge_file import KnowledgeFile
from app.models.conversation import Conversation

__all__ = ["User", "Character", "KnowledgeFile", "Conversation"]
```

- [ ] **Step 4: 运行既有测试**

Run: `pytest -v`
Expected: PASS（32 tests）

- [ ] **Step 5: 提交**

```powershell
git add requirements.txt app/models/conversation.py app/models/__init__.py
git commit -m "feat: add Conversation model and redis dependency"
```

---

### Task 2: Redis 客户端与短期记忆封装

**Files:**
- Create: `app/core/redis_client.py`
- Create: `app/core/memory.py`
- Create: `tests/test_memory.py`

**Interfaces:**
- Consumes: `get_settings()`（Phase 1）
- Produces:
  - `app.core.redis_client:get_redis() -> redis.asyncio.Redis`
  - `class ShortTermMemory` — `__init__(self, redis_client)`
  - `append(self, conversation_id: int, role: str, content: str)`
  - `get(self, conversation_id: int) -> list[dict]`（返回 `[{"role","content"}]`）
  - `clear(self, conversation_id: int)`
  - `len(self, conversation_id: int) -> int`

- [ ] **Step 1: 写 `app/core/redis_client.py`**

```python
import redis.asyncio as aioredis

from app.config import get_settings


def get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.Redis(
        host=settings.redis_host, port=settings.redis_port, decode_responses=True
    )
```

- [ ] **Step 2: 写失败测试 `tests/test_memory.py`**

```python
import fakeredis.aioredis

from app.core.memory import ShortTermMemory


async def test_append_and_get():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    mem = ShortTermMemory(redis)
    await mem.append(1, "user", "你好")
    await mem.append(1, "assistant", "你好呀")
    history = await mem.get(1)
    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]


async def test_clear():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    mem = ShortTermMemory(redis)
    await mem.append(1, "user", "x")
    await mem.clear(1)
    assert await mem.get(1) == []
    assert await mem.len(1) == 0
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.memory'`

- [ ] **Step 4: 写 `app/core/memory.py`**

```python
import json


class ShortTermMemory:
    def __init__(self, redis_client):
        self.redis = redis_client

    def _key(self, conversation_id: int) -> str:
        return f"chat:{conversation_id}:messages"

    async def append(self, conversation_id: int, role: str, content: str) -> None:
        await self.redis.rpush(
            self._key(conversation_id), json.dumps({"role": role, "content": content})
        )

    async def get(self, conversation_id: int) -> list[dict]:
        raw = await self.redis.lrange(self._key(conversation_id), 0, -1)
        return [json.loads(x) for x in raw]

    async def clear(self, conversation_id: int) -> None:
        await self.redis.delete(self._key(conversation_id))

    async def len(self, conversation_id: int) -> int:
        return await self.redis.llen(self._key(conversation_id))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_memory.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: 提交**

```powershell
git add app/core/redis_client.py app/core/memory.py tests/test_memory.py
git commit -m "feat: add redis client and short-term memory"
```

---

### Task 3: VectorStore 扩展长期记忆集合

**Files:**
- Modify: `app/core/vector_store.py`
- Modify: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `VectorStore`（Phase 3）
- Produces（新增）:
  - `VectorStore.MEMORY_COLLECTION = "long_term_memory"`
  - `insert_memory(self, user_id, character_id, text, vector) -> int`
  - `search_memory(self, character_id, query_vector, top_k) -> list[dict]`
  - `ensure_collections` 额外创建 memory collection

- [ ] **Step 1: 追加失败测试到 `tests/test_vector_store.py`**

```python
def test_ensure_creates_memory_collection(monkeypatch):
    calls = []

    class _FakeMilvus:
        def __init__(self, uri):
            pass

        def create_collection(self, name, dimension):
            calls.append((name, dimension))

    monkeypatch.setattr("app.core.vector_store.MilvusClient", _FakeMilvus)
    store = VectorStore(uri="http://m", text_dim=8, image_dim=4)
    store.ensure_collections()
    assert ("long_term_memory", 8) in calls


def test_search_memory_uses_memory_collection(monkeypatch):
    seen = {}

    class _FakeMilvus:
        def __init__(self, uri):
            pass

        def search(self, collection_name, data, filter, limit, output_fields):
            seen["collection"] = collection_name
            return [[{"entity": {"text": "用户喜欢咖啡"}}]]

    monkeypatch.setattr("app.core.vector_store.MilvusClient", _FakeMilvus)
    store = VectorStore(uri="http://m", text_dim=2, image_dim=2)
    result = store.search_memory(character_id=7, query_vector=[0.1, 0.2], top_k=2)
    assert seen["collection"] == "long_term_memory"
    assert result[0]["text"] == "用户喜欢咖啡"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_vector_store.py -v`
Expected: FAIL（`search_memory` 未定义 / collection 缺失）

- [ ] **Step 3: 修改 `app/core/vector_store.py`**

在 `VectorStore` 类中新增常量与方法：

```python
    MEMORY_COLLECTION = "long_term_memory"
```

`ensure_collections` 改为：

```python
    def ensure_collections(self) -> None:
        self._client.create_collection(self.KNOWLEDGE_COLLECTION, dimension=self._text_dim)
        self._client.create_collection(self.IMAGE_COLLECTION, dimension=self._image_dim)
        self._client.create_collection(self.MEMORY_COLLECTION, dimension=self._text_dim)
```

在 `search_text` 之后新增：

```python
    def insert_memory(
        self, user_id: int, character_id: int, text: str, vector: list[float]
    ) -> int:
        rows = [
            {
                "user_id": user_id,
                "character_id": character_id,
                "text": text,
                "vector": vector,
            }
        ]
        result = self._client.insert(self.MEMORY_COLLECTION, rows)
        if isinstance(result, dict) and result.get("ids"):
            return result["ids"][0]
        return 0

    def search_memory(
        self, character_id: int, query_vector: list[float], top_k: int
    ) -> list[dict]:
        result = self._client.search(
            self.MEMORY_COLLECTION,
            data=[query_vector],
            filter=f"character_id == {character_id}",
            limit=top_k,
            output_fields=["text"],
        )
        hits = result[0] if result else []
        return [h["entity"] for h in hits]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_vector_store.py -v`
Expected: PASS（5 tests）

- [ ] **Step 5: 提交**

```powershell
git add app/core/vector_store.py tests/test_vector_store.py
git commit -m "feat: add long-term memory collection to VectorStore"
```

---

### Task 4: 检索器

**Files:**
- Create: `app/core/retriever.py`
- Create: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `EmbeddingClient`（Phase 2）、`VectorStore`（Phase 3/Task 3）
- Produces:
  - `search_knowledge(character_id: int, query: str, top_k: int = 4) -> list[str]`
  - `search_memory(character_id: int, query: str, top_k: int = 4) -> list[str]`
  - `retrieve_context(character_id: int, query: str, top_k: int = 4) -> tuple[list[str], list[str]]`（返回 `(knowledge, memory)`）

- [ ] **Step 1: 写失败测试 `tests/test_retriever.py`**

```python
from app.core import retriever


async def test_retrieve_context_merges(monkeypatch):
    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class _FakeStore:
        def search_text(self, character_id, query_vector, top_k):
            return [{"text": "知识库片段A"}]

        def search_memory(self, character_id, query_vector, top_k):
            return [{"text": "用户偏好：咖啡"}]

    monkeypatch.setattr(retriever, "EmbeddingClient", lambda **kw: _FakeEmbedder())
    monkeypatch.setattr(retriever, "build_vector_store", lambda: _FakeStore())

    knowledge, memory = await retriever.retrieve_context(1, "你好")
    assert knowledge == ["知识库片段A"]
    assert memory == ["用户偏好：咖啡"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.retriever'`

- [ ] **Step 3: 写 `app/core/retriever.py`**

```python
import asyncio

from app.config import get_settings
from app.core.embeddings import EmbeddingClient
from app.core.vector_store import VectorStore


def _build_embedder() -> EmbeddingClient:
    settings = get_settings()
    return EmbeddingClient(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )


def build_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        uri=settings.milvus_uri,
        text_dim=settings.text_embedding_dim,
        image_dim=settings.image_embedding_dim,
    )


async def search_knowledge(character_id: int, query: str, top_k: int = 4) -> list[str]:
    embedder = _build_embedder()
    store = build_vector_store()
    vec = (await embedder.embed([query]))[0]
    hits = await asyncio.to_thread(store.search_text, character_id, vec, top_k)
    return [h["text"] for h in hits]


async def search_memory(character_id: int, query: str, top_k: int = 4) -> list[str]:
    embedder = _build_embedder()
    store = build_vector_store()
    vec = (await embedder.embed([query]))[0]
    hits = await asyncio.to_thread(store.search_memory, character_id, vec, top_k)
    return [h["text"] for h in hits]


async def retrieve_context(
    character_id: int, query: str, top_k: int = 4
) -> tuple[list[str], list[str]]:
    knowledge, memory = await asyncio.gather(
        search_knowledge(character_id, query, top_k),
        search_memory(character_id, query, top_k),
    )
    return knowledge, memory
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: 提交**

```powershell
git add app/core/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever for knowledge and long-term memory"
```

---

### Task 5: 会话 CRUD 接口

**Files:**
- Create: `app/schemas/conversation.py`
- Modify: `app/schemas/__init__.py`
- Create: `app/services/conversation_service.py`
- Create: `app/api/routes/conversations.py`
- Modify: `app/main.py`
- Create: `tests/test_conversations.py`

**Interfaces:**
- Consumes: `Conversation`（Task 1）、`get_db`/`get_current_user`（Phase 1）
- Produces:
  - `create_conversation(db, user_id, character_id, title="新对话") -> Conversation`
  - `list_conversations(db, user_id, character_id) -> list[Conversation]`
  - `get_conversation(db, user_id, character_id, conversation_id) -> Conversation | None`
  - `delete_conversation(db, user_id, character_id, conversation_id) -> bool`
  - `POST /api/characters/{character_id}/conversations` → `ConversationRead`
  - `GET /api/characters/{character_id}/conversations` → `list[ConversationRead]`
  - `DELETE /api/characters/{character_id}/conversations/{conversation_id}` → 204

- [ ] **Step 1: 写 `app/schemas/conversation.py`**

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationCreate(BaseModel):
    title: str = "新对话"


class ConversationRead(BaseModel):
    id: int
    character_id: int
    title: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    message: str
```

- [ ] **Step 2: 修改 `app/schemas/__init__.py`**

在现有导入后追加：

```python
from app.schemas.conversation import ChatRequest, ConversationCreate, ConversationRead
```

并把 `ConversationCreate` / `ConversationRead` / `ChatRequest` 加入 `__all__`。

- [ ] **Step 3: 写 `app/services/conversation_service.py`**

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation


async def create_conversation(
    db: AsyncSession, user_id: int, character_id: int, title: str = "新对话"
) -> Conversation:
    conv = Conversation(user_id=user_id, character_id=character_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def list_conversations(
    db: AsyncSession, user_id: int, character_id: int
) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.character_id == character_id,
        )
        .order_by(Conversation.id.desc())
    )
    return list(result.scalars().all())


async def get_conversation(
    db: AsyncSession, user_id: int, character_id: int, conversation_id: int
) -> Conversation | None:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.character_id == character_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_conversation(
    db: AsyncSession, user_id: int, character_id: int, conversation_id: int
) -> bool:
    conv = await get_conversation(db, user_id, character_id, conversation_id)
    if conv is None:
        return False
    await db.delete(conv)
    await db.commit()
    return True
```

- [ ] **Step 4: 写 `app/api/routes/conversations.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.conversation import ConversationCreate, ConversationRead
from app.services import character_service, conversation_service

router = APIRouter()


@router.post(
    "/characters/{character_id}/conversations",
    response_model=ConversationRead,
    status_code=201,
)
async def create_conversation(
    character_id: int,
    payload: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return await conversation_service.create_conversation(
        db, current_user.id, character_id, payload.title
    )


@router.get(
    "/characters/{character_id}/conversations",
    response_model=list[ConversationRead],
)
async def list_conversations(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await conversation_service.list_conversations(db, current_user.id, character_id)


@router.delete(
    "/characters/{character_id}/conversations/{conversation_id}", status_code=204
)
async def delete_conversation(
    character_id: int,
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ok = await conversation_service.delete_conversation(
        db, current_user.id, character_id, conversation_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
```

- [ ] **Step 5: 修改 `app/main.py` 注册 conversations 路由**

```python
from app.api.routes import auth, characters, conversations, health, knowledge

# ... 追加
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
```

- [ ] **Step 6: 写失败测试 `tests/test_conversations.py`**

```python
async def _setup(client, username):
    await client.post(
        "/api/auth/register", json={"username": username, "password": "secret123"}
    )
    login = await client.post(
        "/api/auth/login", json={"username": username, "password": "secret123"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    resp = await client.post(
        "/api/characters",
        json={"name": "助手", "system_prompt": "x", "model_name": "gpt-test"},
        headers=headers,
    )
    return headers, resp.json()["id"]


async def test_create_and_list_conversation(client):
    headers, cid = await _setup(client, "convuser")
    resp = await client.post(
        f"/api/characters/{cid}/conversations", json={"title": "第一次聊天"}, headers=headers
    )
    assert resp.status_code == 201
    assert resp.json()["title"] == "第一次聊天"

    listed = await client.get(f"/api/characters/{cid}/conversations", headers=headers)
    assert len(listed.json()) == 1


async def test_delete_conversation(client):
    headers, cid = await _setup(client, "convuser2")
    created = await client.post(
        f"/api/characters/{cid}/conversations", json={"title": "x"}, headers=headers
    )
    cvid = created.json()["id"]
    resp = await client.delete(
        f"/api/characters/{cid}/conversations/{cvid}", headers=headers
    )
    assert resp.status_code == 204
```

- [ ] **Step 7: 运行测试确认失败**

Run: `pytest tests/test_conversations.py -v`
Expected: FAIL（路由 404）

- [ ] **Step 8: 运行全部测试确认通过**

Run: `pytest -v`
Expected: PASS（32 + 2 + 2 + 1 + 2 = 39 tests）

- [ ] **Step 9: 提交**

```powershell
git add app/schemas/conversation.py app/schemas/__init__.py app/services/conversation_service.py app/api/routes/conversations.py app/main.py tests/test_conversations.py
git commit -m "feat: add conversation CRUD endpoints"
```

---

### Task 6: 对话服务（组装 + 压缩）

**Files:**
- Create: `app/services/chat_service.py`
- Create: `tests/test_chat_service.py`

**Interfaces:**
- Consumes: `retrieve_context`（Task 4）、`LLMClient`（Phase 2）、`ShortTermMemory`（Task 2）
- Produces:
  - `build_messages(system_prompt, history, knowledge, memory, user_message) -> list[dict]`
  - `maybe_compress(memory: ShortTermMemory, conversation_id, llm, max_messages=40) -> None`（窗口超限时把最早消息压缩为摘要）

- [ ] **Step 1: 写失败测试 `tests/test_chat_service.py`**

```python
from app.services import chat_service


def test_build_messages_assembles_order():
    msgs = chat_service.build_messages(
        system_prompt="你是助手",
        history=[{"role": "user", "content": "之前的问题"}],
        knowledge=["知识1"],
        memory=["记忆1"],
        user_message="当前问题",
    )
    assert msgs[0] == {"role": "system", "content": "你是助手"}
    # 检索上下文以 system 消息注入
    assert any("知识1" in m["content"] for m in msgs if m["role"] == "system")
    assert msgs[-1] == {"role": "user", "content": "当前问题"}
    # 历史消息在 user 消息之前
    roles = [m["role"] for m in msgs]
    assert roles[-2:] == ["assistant", "user"] or roles[-2:] == ["user", "user"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_chat_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.chat_service'`

- [ ] **Step 3: 写 `app/services/chat_service.py`**

```python
from app.core.memory import ShortTermMemory
from app.core.retriever import retrieve_context


def build_messages(
    system_prompt: str,
    history: list[dict],
    knowledge: list[str],
    memory: list[str],
    user_message: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    context_parts = []
    if knowledge:
        context_parts.append("【知识库】\n" + "\n".join(f"- {k}" for k in knowledge))
    if memory:
        context_parts.append("【长期记忆】\n" + "\n".join(f"- {m}" for m in memory))
    if context_parts:
        messages.append(
            {
                "role": "system",
                "content": "以下是可供参考的上下文信息：\n\n" + "\n\n".join(context_parts),
            }
        )

    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


async def maybe_compress(
    memory: ShortTermMemory, conversation_id: int, llm, max_messages: int = 40
) -> None:
    history = await memory.get(conversation_id)
    if len(history) <= max_messages:
        return
    to_compress = history[:-6]
    recent = history[-6:]
    prompt = (
        "请把以下对话压缩为一段简洁的摘要，保留关键信息（人物、话题、结论）：\n\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
    )
    summary = ""
    async for piece in llm.chat_stream([{"role": "user", "content": prompt}]):
        summary += piece
    await memory.clear(conversation_id)
    await memory.append(conversation_id, "system", f"[对话摘要] {summary}")
    for m in recent:
        await memory.append(conversation_id, m["role"], m["content"])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_chat_service.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: 提交**

```powershell
git add app/services/chat_service.py tests/test_chat_service.py
git commit -m "feat: add chat message assembly and summary compression"
```

---

### Task 7: 长期记忆提取 worker

**Files:**
- Modify: `app/worker/tasks.py`
- Modify: `app/worker/main.py`

**Interfaces:**
- Consumes: `LLMClient`（Phase 2）、`EmbeddingClient`（Phase 2）、`VectorStore.insert_memory`（Task 3）、`ShortTermMemory`（Task 2）
- Produces: `app.worker.tasks:extract_memory(ctx, conversation_id, character_id, user_id) -> None`

- [ ] **Step 1: 在 `app/worker/tasks.py` 追加 `extract_memory`**

```python
from app.core.memory import ShortTermMemory
from app.core.redis_client import get_redis
from app.core.vector_store import VectorStore


MEMORY_EXTRACT_PROMPT = (
    "从以下对话中提取值得长期记住的重要事实（例如用户身份、偏好、重要结论），"
    "最多 5 条，每条一行。若无值得记忆的内容，输出空字符串。\n\n{conversation}"
)


async def extract_memory(
    ctx: dict, conversation_id: int, character_id: int, user_id: int
) -> None:
    settings = get_settings()
    memory = ShortTermMemory(get_redis())
    history = await memory.get(conversation_id)
    if not history:
        return

    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-12:]
    )
    prompt = MEMORY_EXTRACT_PROMPT.format(conversation=conversation_text)

    from app.core.llm import LLMClient

    llm = LLMClient(
        model="gpt-4o-mini",
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.2,
    )
    reply = ""
    async for piece in llm.chat_stream([{"role": "user", "content": prompt}]):
        reply += piece

    facts = [line.strip("- ").strip() for line in reply.splitlines() if line.strip()]
    facts = [f for f in facts if f][:5]
    if not facts:
        return

    embedder = EmbeddingClient(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )
    store = VectorStore(
        uri=settings.milvus_uri,
        text_dim=settings.text_embedding_dim,
        image_dim=settings.image_embedding_dim,
    )
    vectors = await embedder.embed(facts)
    for fact, vec in zip(facts, vectors):
        await asyncio.to_thread(store.insert_memory, user_id, character_id, fact, vec)
```

- [ ] **Step 2: 在 `app/worker/main.py` 注册 `extract_memory`**

将 `functions=[process_file]` 改为：

```python
from app.worker.tasks import extract_memory, process_file

# ... startup/shutdown 不变 ...

worker = Worker(
    functions=[process_file, extract_memory],
    redis_settings=RedisSettings(host=settings.redis_host, port=settings.redis_port),
    on_startup=startup,
    on_shutdown=shutdown,
)
```

（`main.py` 顶部 import 改为 `from app.worker.tasks import extract_memory, process_file`，并移除 `__import__` 写法。）

- [ ] **Step 3: 运行全部测试**

Run: `pytest -v`
Expected: PASS（40 tests）

- [ ] **Step 4: 提交**

```powershell
git add app/worker/tasks.py app/worker/main.py
git commit -m "feat: add long-term memory extraction worker"
```

---

### Task 8: SSE 流式对话接口

**Files:**
- Create: `app/api/routes/chat.py`
- Modify: `app/main.py`
- Create: `tests/test_chat.py`

**Interfaces:**
- Consumes: `Conversation`（Task 1）、`build_messages` / `maybe_compress`（Task 6）、`retrieve_context`（Task 4）、`build_llm_from_character`（Phase 2）、`ShortTermMemory`（Task 2）、`character_service`（Phase 2）、`conversation_service`（Task 5）
- Produces: `POST /api/characters/{character_id}/conversations/{conversation_id}/chat` → SSE（`data: {"delta": ...}`，结束 `data: [DONE]`）

- [ ] **Step 1: 写 `app/api/routes/chat.py`**

```python
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.config import get_settings
from app.core.llm import build_llm_from_character
from app.core.memory import ShortTermMemory
from app.core.redis_client import get_redis
from app.core.retriever import retrieve_context
from app.models.user import User
from app.schemas.conversation import ChatRequest
from app.services import character_service, chat_service, conversation_service

router = APIRouter()


@router.post("/characters/{character_id}/conversations/{conversation_id}/chat")
async def chat(
    character_id: int,
    conversation_id: int,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    conv = await conversation_service.get_conversation(
        db, current_user.id, character_id, conversation_id
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    user_message = payload.message
    memory = ShortTermMemory(get_redis())
    llm = build_llm_from_character(char)

    async def event_stream():
        knowledge, long_memory = await retrieve_context(character_id, user_message)
        history = await memory.get(conversation_id)
        messages = chat_service.build_messages(
            char.system_prompt, history, knowledge, long_memory, user_message
        )
        await memory.append(conversation_id, "user", user_message)

        full_reply = ""
        try:
            async for piece in llm.chat_stream(messages):
                full_reply += piece
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
        finally:
            if full_reply:
                await memory.append(conversation_id, "assistant", full_reply)
            yield "data: [DONE]\n\n"

        # 达到阈值时异步沉淀长期记忆（不阻塞响应）
        try:
            history_len = await memory.len(conversation_id)
            user_msgs = len([m for m in await memory.get(conversation_id) if m["role"] == "user"])
            if user_msgs >= 10:
                from arq import create_pool
                from arq.connections import RedisSettings

                settings = get_settings()
                redis = await create_pool(
                    RedisSettings(host=settings.redis_host, port=settings.redis_port)
                )
                await redis.enqueue_job(
                    "extract_memory", conversation_id, character_id, current_user.id
                )
                await redis.aclose()
        except Exception:
            pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 2: 修改 `app/main.py` 注册 chat 路由**

```python
from app.api.routes import auth, characters, chat, conversations, health, knowledge

# ... 追加
app.include_router(chat.router, prefix="/api", tags=["chat"])
```

- [ ] **Step 3: 写失败测试 `tests/test_chat.py`**

```python
import fakeredis.aioredis

from app.core.memory import ShortTermMemory


async def _setup(client, username):
    await client.post(
        "/api/auth/register", json={"username": username, "password": "secret123"}
    )
    login = await client.post(
        "/api/auth/login", json={"username": username, "password": "secret123"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    resp = await client.post(
        "/api/characters",
        json={"name": "助手", "system_prompt": "你是助手", "model_name": "gpt-test"},
        headers=headers,
    )
    cid = resp.json()["id"]
    conv = await client.post(
        f"/api/characters/{cid}/conversations", json={"title": "x"}, headers=headers
    )
    return headers, cid, conv.json()["id"]


async def test_chat_streams_response(client, monkeypatch):
    headers, cid, cvid = await _setup(client, "chatuser")

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("app.api.routes.chat.get_redis", lambda: fake_redis)

    async def _fake_retrieve(character_id, query, top_k=4):
        return ["知识A"], ["记忆B"]

    monkeypatch.setattr("app.api.routes.chat.retrieve_context", _fake_retrieve)

    class _FakeLLM:
        def __init__(self, **kw):
            pass

        async def chat_stream(self, messages):
            for piece in ["你", "好"]:
                yield piece

    monkeypatch.setattr("app.api.routes.chat.build_llm_from_character", lambda c: _FakeLLM())

    resp = await client.post(
        f"/api/characters/{cid}/conversations/{cvid}/chat",
        json={"message": "你好"},
        headers=headers,
    )
    assert resp.status_code == 200
    text = resp.text
    assert "你" in text and "好" in text
    assert "[DONE]" in text

    mem = ShortTermMemory(fake_redis)
    history = await mem.get(cvid)
    assert history[0]["role"] == "user"
    assert history[-1]["role"] == "assistant"
```

- [ ] **Step 4: 运行测试确认失败**

Run: `pytest tests/test_chat.py -v`
Expected: FAIL（路由 404）

- [ ] **Step 5: 运行全部测试确认通过**

Run: `pytest -v`
Expected: PASS（40 + 1 = 41 tests）

- [ ] **Step 6: 提交**

```powershell
git add app/api/routes/chat.py app/main.py tests/test_chat.py
git commit -m "feat: add SSE streaming chat endpoint"
```

---

## Phase 4 完成后自检清单

- [ ] `pytest -v` 全绿（41 tests）
- [ ] 可创建/列出/删除会话
- [ ] 对话以 SSE 流式返回，多轮上下文正确（Redis 短期记忆）
- [ ] 检索命中知识库与长期记忆并注入 prompt
- [ ] 窗口超限时触发摘要压缩
- [ ] 用户消息达 10 条时投递 `extract_memory` 任务
- [ ] 手动跑 worker + Milvus + Redis 后，长期记忆可写入并跨会话检索

## 本阶段已知边界（供 Phase 5 衔接）

- 角色删除时尚未级联清理 Milvus 向量与 Redis 会话；Phase 5 收尾时补齐。
- 前端页面尚未接入；Phase 5 实现。
- worker 容器化与整体编排在 Phase 5 完成。
