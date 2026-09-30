# Phase 3: 知识库入库 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现知识库文件的上传、解析（文本 / PDF / OCR 图片）、切片、向量化并写入 Milvus，用 arq 异步任务队列驱动，支持按文件删除与重新索引。

**Architecture:** 上传接口先落盘 + 写 DB（`status=pending`），随后投递 arq 任务 `process_file`。worker 内：按文件类型解析 → 文本分块 → 文本 embedding 写 `knowledge_chunks` collection；图片额外走 OCR（文字入文本集合）+ 视觉 embedding（入 `image_vectors` collection）。Milvus 访问封装在同步类 `VectorStore` 中，异步侧用 `asyncio.to_thread` 调用。

**Tech Stack:** 新增 `pymilvus==2.4.4`、`arq==0.26.1`、`pypdf==4.3.1`、`Pillow==10.4.0`、`rapidocr-onnxruntime==1.3.24`。复用 Phase 2 的 `EmbeddingClient` / `VisionEmbeddingClient`。

## Global Constraints

- 所有接口路径以 `/api` 为前缀，且需 `get_current_user` 鉴权 + `user_id` 过滤
- 上传文件落盘到 `data/uploads/{user_id}/{character_id}/`，文件名前缀带 file_id 防冲突
- 原始文件与向量必须能按 `character_id` / `file_id` 完整删除，不留孤儿
- 文本 embedding 维度默认 1536、视觉 embedding 维度默认 512（可配置），collection 创建时据此固定
- 测试用 `monkeypatch` 替换 Milvus / OCR / embedding 客户端，不依赖真实 Milvus 服务
- worker 入口为 `python -m app.worker.main`；任务函数签名 `async def process_file(ctx, file_id)`

## 与 Phase 1/2 的接口约定（复用，不重新定义）

- `app.config:get_settings()` → `Settings`（新增 `milvus_uri`、`text_embedding_dim`、`image_embedding_dim`）
- `app.database:SessionLocal` / `Base` / `engine`
- `app.models.base:Base`、`app.models.user:User`、`app.models.character:Character`
- `app.api.deps:get_db` / `get_current_user`
- `app.core.embeddings:EmbeddingClient`（`embed(texts) -> list[list[float]]`）
- `app.core.vision:VisionEmbeddingClient`（`embed_image(path) -> list[float]`）

---

## 文件结构总览（本阶段新增/修改）

```
app/
├── config.py                        # 修改：新增 milvus_uri / 维度字段
├── models/
│   ├── knowledge_file.py            # KnowledgeFile ORM
│   └── __init__.py                  # 修改：导出
├── core/
│   ├── file_processing.py           # chunk / extract / ocr / classify
│   └── vector_store.py              # VectorStore（Milvus 封装）
├── schemas/
│   ├── knowledge.py                 # KnowledgeFileRead
│   └── __init__.py                  # 修改：导出
├── worker/
│   ├── __init__.py
│   ├── tasks.py                     # process_file 任务
│   └── main.py                      # arq Worker 入口
├── api/routes/
│   └── knowledge.py                 # 上传/列表/删除/状态
└── main.py                          # 修改：注册 knowledge 路由
tests/
├── test_file_processing.py
├── test_vector_store.py
├── test_knowledge_api.py
└── test_worker.py
data/                                # 运行时上传目录（.gitignore）
```

---

### Task 1: 依赖与 Milvus 配置

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `app/config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `Settings`（Phase 1）
- Produces: `Settings.milvus_uri`（property）、`Settings.text_embedding_dim`、`Settings.image_embedding_dim`

- [ ] **Step 1: 追加依赖到 `requirements.txt`**

```powershell
Add-Content requirements.txt @'
pymilvus==2.4.4
arq==0.26.1
pypdf==4.3.1
Pillow==10.4.0
rapidocr-onnxruntime==1.3.24
'@
```

- [ ] **Step 2: 追加 `.env.example` 配置**

```powershell
Add-Content .env.example @'

# Milvus 向量维度（换 embedding 模型时需同步修改）
TEXT_EMBEDDING_DIM=1536
IMAGE_EMBEDDING_DIM=512
'@
```

- [ ] **Step 3: 修改 `app/config.py`**

在 `vision_embedding_api_key` 之后、`model_config` 之前插入：

```python
    text_embedding_dim: int = 1536
    image_embedding_dim: int = 512
```

并在 `database_url` property 之后、`get_settings` 之前新增：

```python
    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"
```

- [ ] **Step 4: 修改 `.gitignore` 追加上传目录**

```powershell
Add-Content .gitignore "data/"
```

- [ ] **Step 5: 运行既有测试**

Run: `pytest -v`
Expected: PASS（19 tests）

- [ ] **Step 6: 提交**

```powershell
git add requirements.txt .env.example app/config.py .gitignore
git commit -m "feat: add Milvus and ingestion dependencies and config"
```

---

### Task 2: KnowledgeFile 模型

**Files:**
- Create: `app/models/knowledge_file.py`
- Modify: `app/models/__init__.py`

**Interfaces:**
- Consumes: `Base`（Phase 1）
- Produces: `app.models.knowledge_file:KnowledgeFile`（`id`, `user_id`, `character_id`, `filename`, `file_type`, `file_path`, `status`, `created_at`；`file_type ∈ {text, pdf, image}`，`status ∈ {pending, processing, done, failed}`）

- [ ] **Step 1: 写 `app/models/knowledge_file.py`**

```python
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    character_id: Mapped[int] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
```

- [ ] **Step 2: 修改 `app/models/__init__.py`**

```python
from app.models.user import User
from app.models.character import Character
from app.models.knowledge_file import KnowledgeFile

__all__ = ["User", "Character", "KnowledgeFile"]
```

- [ ] **Step 3: 运行既有测试**

Run: `pytest -v`
Expected: PASS（19 tests）

- [ ] **Step 4: 提交**

```powershell
git add app/models/knowledge_file.py app/models/__init__.py
git commit -m "feat: add KnowledgeFile model"
```

---

### Task 3: 文件解析与分块（纯函数）

**Files:**
- Create: `app/core/file_processing.py`
- Create: `tests/test_file_processing.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces:
  - `classify_file_type(filename: str) -> str`（按扩展名返回 `text` / `pdf` / `image`，未识别抛 `ValueError`）
  - `chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]`
  - `extract_text(path: str, file_type: str) -> str`（text 直接读、pdf 用 pypdf）
  - `ocr_image(path: str) -> str`

- [ ] **Step 1: 写失败测试 `tests/test_file_processing.py`**

```python
import pytest

from app.core.file_processing import (
    chunk_text,
    classify_file_type,
    extract_text,
    ocr_image,
)


def test_classify_file_type():
    assert classify_file_type("a.txt") == "text"
    assert classify_file_type("b.md") == "text"
    assert classify_file_type("c.pdf") == "pdf"
    assert classify_file_type("d.png") == "image"
    assert classify_file_type("e.jpg") == "image"
    with pytest.raises(ValueError):
        classify_file_type("f.exe")


def test_chunk_text_basic():
    text = "字" * 20
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert chunks == ["字" * 10, "字" * 10]


def test_chunk_text_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=4)
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ghijklmnop"


def test_extract_text_from_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("你好世界", encoding="utf-8")
    assert extract_text(str(p), "text") == "你好世界"


def test_extract_text_from_pdf(monkeypatch):
    class _Reader:
        def __init__(self, path):
            pass

        @property
        def pages(self):
            return [type("P", (), {"extract_text": lambda self: "第1页"})()]

    monkeypatch.setattr("app.core.file_processing.PdfReader", _Reader)
    assert extract_text("x.pdf", "pdf") == "第1页"


def test_ocr_image(monkeypatch):
    class _Rapid:
        def __init__(self):
            pass

        def __call__(self, img_path):
            return ("[[1, 2, 3, 4], ('识别文字', 0.9)]", None)

    monkeypatch.setattr("app.core.file_processing.RapidOCR", _Rapid)
    assert ocr_image("x.png") == "识别文字"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_file_processing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.file_processing'`

- [ ] **Step 3: 写 `app/core/file_processing.py`**

```python
from pathlib import Path

from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def classify_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"不支持的文件类型: {ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_text(path: str, file_type: str) -> str:
    if file_type == "text":
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    if file_type == "pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"无法提取文本的类型: {file_type}")


_ocr = None


def ocr_image(path: str) -> str:
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    result, _ = _ocr(path)
    if not result:
        return ""
    return "".join(item[1] for item in result)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_file_processing.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: 提交**

```powershell
git add app/core/file_processing.py tests/test_file_processing.py
git commit -m "feat: add file parsing, chunking and OCR utilities"
```

---

### Task 4: VectorStore（Milvus 封装）

**Files:**
- Create: `app/core/vector_store.py`
- Create: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `get_settings()`（Phase 1）
- Produces（同步类，异步侧用 `asyncio.to_thread` 调用）:
  - `class VectorStore` — `__init__(self, uri, text_dim, image_dim)`
  - `ensure_collections(self)` — 创建 `knowledge_chunks` 与 `image_vectors` 两个 collection
  - `insert_text_chunks(self, user_id, character_id, file_id, chunks, vectors) -> list[int]`
  - `insert_image_vector(self, user_id, character_id, file_id, image_path, vector) -> int`
  - `search_text(self, character_id, query_vector, top_k) -> list[dict]`（返回含 `text` 的 dict 列表）
  - `delete_by_file(self, character_id, file_id)`
  - `delete_by_character(self, character_id)`

- [ ] **Step 1: 写失败测试 `tests/test_vector_store.py`**

```python
from app.core.vector_store import VectorStore


def test_ensure_creates_two_collections(monkeypatch):
    calls = []

    class _FakeMilvus:
        def __init__(self, uri):
            self.uri = uri

        def create_collection(self, name, dimension):
            calls.append((name, dimension))

    monkeypatch.setattr("app.core.vector_store.MilvusClient", _FakeMilvus)
    store = VectorStore(uri="http://m", text_dim=8, image_dim=4)
    store.ensure_collections()
    assert ("knowledge_chunks", 8) in calls
    assert ("image_vectors", 4) in calls


def test_insert_text_chunks(monkeypatch):
    inserted = {}

    class _FakeMilvus:
        def __init__(self, uri):
            pass

        def insert(self, collection_name, data):
            inserted[collection_name] = data

    monkeypatch.setattr("app.core.vector_store.MilvusClient", _FakeMilvus)
    store = VectorStore(uri="http://m", text_dim=2, image_dim=2)
    store.insert_text_chunks(1, 2, 3, ["你好"], [[0.1, 0.2]])
    rows = inserted["knowledge_chunks"]
    assert rows[0]["user_id"] == 1
    assert rows[0]["character_id"] == 2
    assert rows[0]["file_id"] == 3
    assert rows[0]["text"] == "你好"
    assert rows[0]["vector"] == [0.1, 0.2]


def test_search_text(monkeypatch):
    class _FakeMilvus:
        def __init__(self, uri):
            pass

        def search(self, collection_name, data, filter, limit, output_fields):
            assert collection_name == "knowledge_chunks"
            assert filter == 'character_id == 2'
            return [[{"entity": {"text": "匹配内容"}}]]

    monkeypatch.setattr("app.core.vector_store.MilvusClient", _FakeMilvus)
    store = VectorStore(uri="http://m", text_dim=2, image_dim=2)
    result = store.search_text(character_id=2, query_vector=[0.1, 0.2], top_k=3)
    assert result[0]["text"] == "匹配内容"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.vector_store'`

- [ ] **Step 3: 写 `app/core/vector_store.py`**

```python
from pymilvus import MilvusClient


class VectorStore:
    KNOWLEDGE_COLLECTION = "knowledge_chunks"
    IMAGE_COLLECTION = "image_vectors"

    def __init__(self, uri: str, text_dim: int, image_dim: int):
        self._client = MilvusClient(uri=uri)
        self._text_dim = text_dim
        self._image_dim = image_dim

    def ensure_collections(self) -> None:
        self._client.create_collection(self.KNOWLEDGE_COLLECTION, dimension=self._text_dim)
        self._client.create_collection(self.IMAGE_COLLECTION, dimension=self._image_dim)

    def insert_text_chunks(
        self,
        user_id: int,
        character_id: int,
        file_id: int,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> list[int]:
        rows = [
            {
                "user_id": user_id,
                "character_id": character_id,
                "file_id": file_id,
                "text": text,
                "vector": vector,
            }
            for text, vector in zip(chunks, vectors)
        ]
        result = self._client.insert(self.KNOWLEDGE_COLLECTION, rows)
        return result["ids"] if isinstance(result, dict) else []

    def insert_image_vector(
        self,
        user_id: int,
        character_id: int,
        file_id: int,
        image_path: str,
        vector: list[float],
    ) -> int:
        rows = [
            {
                "user_id": user_id,
                "character_id": character_id,
                "file_id": file_id,
                "image_path": image_path,
                "vector": vector,
            }
        ]
        result = self._client.insert(self.IMAGE_COLLECTION, rows)
        if isinstance(result, dict) and result.get("ids"):
            return result["ids"][0]
        return 0

    def search_text(self, character_id: int, query_vector: list[float], top_k: int) -> list[dict]:
        result = self._client.search(
            self.KNOWLEDGE_COLLECTION,
            data=[query_vector],
            filter=f"character_id == {character_id}",
            limit=top_k,
            output_fields=["text", "file_id"],
        )
        hits = result[0] if result else []
        return [h["entity"] for h in hits]

    def delete_by_file(self, character_id: int, file_id: int) -> None:
        self._client.delete(
            self.KNOWLEDGE_COLLECTION,
            filter=f"character_id == {character_id} and file_id == {file_id}",
        )
        self._client.delete(
            self.IMAGE_COLLECTION,
            filter=f"character_id == {character_id} and file_id == {file_id}",
        )

    def delete_by_character(self, character_id: int) -> None:
        self._client.delete(
            self.KNOWLEDGE_COLLECTION, filter=f"character_id == {character_id}"
        )
        self._client.delete(
            self.IMAGE_COLLECTION, filter=f"character_id == {character_id}"
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_vector_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 提交**

```powershell
git add app/core/vector_store.py tests/test_vector_store.py
git commit -m "feat: add Milvus-backed VectorStore wrapper"
```

---

### Task 5: 文件上传接口

**Files:**
- Create: `app/schemas/knowledge.py`
- Modify: `app/schemas/__init__.py`
- Create: `app/services/knowledge_service.py`
- Create: `app/api/routes/knowledge.py`
- Modify: `app/main.py`
- Create: `tests/test_knowledge_api.py`

**Interfaces:**
- Consumes: `KnowledgeFile`（Task 2）、`classify_file_type`（Task 3）、`get_db`/`get_current_user`（Phase 1）
- Produces:
  - `app.services.knowledge_service:create_file(db, user_id, character_id, filename, file_type, file_path) -> KnowledgeFile`
  - `app.services.knowledge_service:list_files(db, user_id, character_id) -> list[KnowledgeFile]`
  - `app.services.knowledge_service:get_file(db, user_id, character_id, file_id) -> KnowledgeFile | None`
  - `app.services.knowledge_service:delete_file(db, user_id, character_id, file_id) -> KnowledgeFile | None`
  - `POST /api/characters/{character_id}/knowledge`（multipart 上传）→ `KnowledgeFileRead`
  - `GET /api/characters/{character_id}/knowledge` → `list[KnowledgeFileRead]`
  - `DELETE /api/characters/{character_id}/knowledge/{file_id}` → 204

- [ ] **Step 1: 写 `app/schemas/knowledge.py`**

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class KnowledgeFileRead(BaseModel):
    id: int
    filename: str
    file_type: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 2: 修改 `app/schemas/__init__.py`**

```python
from app.schemas.user import UserCreate, UserRead
from app.schemas.auth import Token
from app.schemas.character import CharacterCreate, CharacterRead, CharacterUpdate
from app.schemas.knowledge import KnowledgeFileRead

__all__ = [
    "UserCreate",
    "UserRead",
    "Token",
    "CharacterCreate",
    "CharacterRead",
    "CharacterUpdate",
    "KnowledgeFileRead",
]
```

- [ ] **Step 3: 写 `app/services/knowledge_service.py`**

```python
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_file import KnowledgeFile


async def create_file(
    db: AsyncSession,
    user_id: int,
    character_id: int,
    filename: str,
    file_type: str,
    file_path: str,
) -> KnowledgeFile:
    f = KnowledgeFile(
        user_id=user_id,
        character_id=character_id,
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        status="pending",
    )
    db.add(f)
    await db.commit()
    await db.refresh(f)
    return f


async def list_files(
    db: AsyncSession, user_id: int, character_id: int
) -> list[KnowledgeFile]:
    result = await db.execute(
        select(KnowledgeFile)
        .where(
            KnowledgeFile.user_id == user_id,
            KnowledgeFile.character_id == character_id,
        )
        .order_by(KnowledgeFile.id)
    )
    return list(result.scalars().all())


async def get_file(
    db: AsyncSession, user_id: int, character_id: int, file_id: int
) -> KnowledgeFile | None:
    result = await db.execute(
        select(KnowledgeFile).where(
            KnowledgeFile.id == file_id,
            KnowledgeFile.user_id == user_id,
            KnowledgeFile.character_id == character_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_file(
    db: AsyncSession, user_id: int, character_id: int, file_id: int
) -> KnowledgeFile | None:
    f = await get_file(db, user_id, character_id, file_id)
    if f is None:
        return None
    Path(f.file_path).unlink(missing_ok=True)
    await db.delete(f)
    await db.commit()
    return f
```

- [ ] **Step 4: 写 `app/api/routes/knowledge.py`**

```python
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.file_processing import classify_file_type
from app.models.user import User
from app.schemas.knowledge import KnowledgeFileRead
from app.services import character_service, knowledge_service

router = APIRouter()

UPLOAD_ROOT = Path("data/uploads")


@router.post(
    "/characters/{character_id}/knowledge",
    response_model=KnowledgeFileRead,
    status_code=201,
)
async def upload_file(
    character_id: int,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    char = await character_service.get_character(db, current_user.id, character_id)
    if char is None:
        raise HTTPException(status_code=404, detail="角色不存在")

    try:
        file_type = classify_file_type(file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 先建 DB 记录拿到 file_id
    record = await knowledge_service.create_file(
        db, current_user.id, character_id, file.filename or "unknown", file_type, ""
    )

    dir_path = UPLOAD_ROOT / str(current_user.id) / str(character_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{record.id}_{file.filename}"
    content = await file.read()
    file_path.write_bytes(content)

    record.file_path = str(file_path)
    await db.commit()
    await db.refresh(record)
    return record


@router.get(
    "/characters/{character_id}/knowledge",
    response_model=list[KnowledgeFileRead],
)
async def list_files(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await knowledge_service.list_files(db, current_user.id, character_id)


@router.delete(
    "/characters/{character_id}/knowledge/{file_id}", status_code=204
)
async def delete_file(
    character_id: int,
    file_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    f = await knowledge_service.delete_file(db, current_user.id, character_id, file_id)
    if f is None:
        raise HTTPException(status_code=404, detail="文件不存在")
```

- [ ] **Step 5: 修改 `app/main.py` 注册 knowledge 路由**

```python
from app.api.routes import auth, characters, health, knowledge

# ... lifespan 不变 ...

app.include_router(knowledge.router, prefix="/api", tags=["knowledge"])
```

- [ ] **Step 6: 写失败测试 `tests/test_knowledge_api.py`**

```python
import pytest


async def _register_and_login(client, username):
    await client.post(
        "/api/auth/register", json={"username": username, "password": "secret123"}
    )
    resp = await client.post(
        "/api/auth/login", json={"username": username, "password": "secret123"}
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def _create_character(client, headers):
    resp = await client.post(
        "/api/characters",
        json={"name": "助手", "system_prompt": "x", "model_name": "gpt-test"},
        headers=headers,
    )
    return resp.json()["id"]


async def test_upload_txt_file(client, monkeypatch):
    headers = await _register_and_login(client, "kuploader")
    cid = await _create_character(client, headers)
    files = {"file": ("note.txt", "知识库内容".encode("utf-8"), "text/plain")}
    resp = await client.post(
        f"/api/characters/{cid}/knowledge", files=files, headers=headers
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["file_type"] == "text"
    assert body["status"] == "pending"


async def test_upload_rejects_unknown_type(client):
    headers = await _register_and_login(client, "kuploader2")
    cid = await _create_character(client, headers)
    files = {"file": ("x.exe", b"data", "application/octet-stream")}
    resp = await client.post(
        f"/api/characters/{cid}/knowledge", files=files, headers=headers
    )
    assert resp.status_code == 400


async def test_list_and_delete_file(client):
    headers = await _register_and_login(client, "kuploader3")
    cid = await _create_character(client, headers)
    files = {"file": ("a.txt", "内容".encode("utf-8"), "text/plain")}
    created = await client.post(
        f"/api/characters/{cid}/knowledge", files=files, headers=headers
    )
    fid = created.json()["id"]

    listed = await client.get(f"/api/characters/{cid}/knowledge", headers=headers)
    assert len(listed.json()) == 1

    deleted = await client.delete(
        f"/api/characters/{cid}/knowledge/{fid}", headers=headers
    )
    assert deleted.status_code == 204
```

- [ ] **Step 7: 运行测试确认失败**

Run: `pytest tests/test_knowledge_api.py -v`
Expected: FAIL（路由 404）

- [ ] **Step 8: 运行全部测试确认通过**

Run: `pytest -v`
Expected: PASS（19 + 6 + 3 + 3 = 31 tests）

- [ ] **Step 9: 提交**

```powershell
git add app/schemas/knowledge.py app/schemas/__init__.py app/services/knowledge_service.py app/api/routes/knowledge.py app/main.py tests/test_knowledge_api.py
git commit -m "feat: add knowledge file upload/list/delete endpoints"
```

---

### Task 6: arq 异步任务（process_file）

**Files:**
- Create: `app/worker/__init__.py`
- Create: `app/worker/tasks.py`
- Create: `app/worker/main.py`
- Modify: `app/api/routes/knowledge.py`（上传后投递任务）
- Create: `tests/test_worker.py`

**Interfaces:**
- Consumes: `KnowledgeFile`（Task 2）、`EmbeddingClient`（Phase 2）、`VisionEmbeddingClient`（Phase 2）、`VectorStore`（Task 4）、`extract_text`/`chunk_text`/`ocr_image`（Task 3）、`SessionLocal`（Phase 1）
- Produces:
  - `app.worker.tasks:process_file(ctx: dict, file_id: int) -> None`（将文件解析→切片→向量化→写 Milvus，并把 `KnowledgeFile.status` 置为 `done`/`failed`）
  - `app.worker.tasks:build_vector_store() -> VectorStore`
  - `python -m app.worker.main` 启动 worker

- [ ] **Step 1: 写失败测试 `tests/test_worker.py`**

```python
import asyncio

from app.database import Base, engine
from app.models.knowledge_file import KnowledgeFile


async def _make_pending_file(tmp_path):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.database import SessionLocal

    p = tmp_path / "1_note.txt"
    p.write_text("这是测试知识内容", encoding="utf-8")
    async with SessionLocal() as db:
        f = KnowledgeFile(
            user_id=1,
            character_id=1,
            filename="note.txt",
            file_type="text",
            file_path=str(p),
            status="pending",
        )
        db.add(f)
        await db.commit()
        await db.refresh(f)
        return f.id


async def test_process_file_text(tmp_path, monkeypatch):
    from app.core import embeddings, vector_store

    fid = await _make_pending_file(tmp_path)

    class _FakeEmbedder:
        async def embed(self, texts):
            return [[0.1] * 2 for _ in texts]

    inserted = {}

    class _FakeVectorStore:
        def insert_text_chunks(self, user_id, character_id, file_id, chunks, vectors):
            inserted["chunks"] = chunks

    monkeypatch.setattr(embeddings, "EmbeddingClient", lambda **kw: _FakeEmbedder())
    monkeypatch.setattr(
        "app.worker.tasks.build_vector_store", lambda: _FakeVectorStore()
    )

    from app.worker.tasks import process_file

    await process_file({}, fid)

    assert inserted["chunks"]  # 有切片被写入
    from app.database import SessionLocal

    async with SessionLocal() as db:
        f = await db.get(KnowledgeFile, fid)
        assert f.status == "done"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.worker.tasks'`

- [ ] **Step 3: 写 `app/worker/tasks.py`**

```python
import asyncio

from sqlalchemy import select

from app.config import get_settings
from app.core.embeddings import EmbeddingClient
from app.core.file_processing import chunk_text, extract_text, ocr_image
from app.core.vector_store import VectorStore
from app.core.vision import VisionEmbeddingClient
from app.database import SessionLocal
from app.models.knowledge_file import KnowledgeFile


def build_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        uri=settings.milvus_uri,
        text_dim=settings.text_embedding_dim,
        image_dim=settings.image_embedding_dim,
    )


async def process_file(ctx: dict, file_id: int) -> None:
    settings = get_settings()
    async with SessionLocal() as db:
        f = await db.get(KnowledgeFile, file_id)
        if f is None:
            return
        f.status = "processing"
        await db.commit()

        store = build_vector_store()
        try:
            if f.file_type == "image":
                # OCR 文字 → 文本向量
                ocr_text = await asyncio.to_thread(ocr_image, f.file_path)
                if ocr_text:
                    chunks = chunk_text(ocr_text)
                    embedder = EmbeddingClient(
                        model=settings.embedding_model,
                        base_url=settings.embedding_base_url,
                        api_key=settings.embedding_api_key,
                    )
                    vectors = await embedder.embed(chunks)
                    await asyncio.to_thread(
                        store.insert_text_chunks,
                        f.user_id,
                        f.character_id,
                        f.id,
                        chunks,
                        vectors,
                    )
                # 视觉向量
                vision = VisionEmbeddingClient(
                    model=settings.vision_embedding_model,
                    base_url=settings.vision_embedding_base_url,
                    api_key=settings.vision_embedding_api_key,
                )
                vec = await vision.embed_image(f.file_path)
                await asyncio.to_thread(
                    store.insert_image_vector,
                    f.user_id,
                    f.character_id,
                    f.id,
                    f.file_path,
                    vec,
                )
            else:
                text = await asyncio.to_thread(extract_text, f.file_path, f.file_type)
                chunks = chunk_text(text)
                embedder = EmbeddingClient(
                    model=settings.embedding_model,
                    base_url=settings.embedding_base_url,
                    api_key=settings.embedding_api_key,
                )
                vectors = await embedder.embed(chunks)
                await asyncio.to_thread(
                    store.insert_text_chunks,
                    f.user_id,
                    f.character_id,
                    f.id,
                    chunks,
                    vectors,
                )

            f.status = "done"
            await db.commit()
        except Exception as e:
            f.status = "failed"
            await db.commit()
            raise
```

- [ ] **Step 4: 写 `app/worker/main.py`**

```python
from arq.connections import RedisSettings
from arq.worker import Worker

from app.config import get_settings
from app.worker.tasks import build_vector_store, process_file


async def startup(ctx):
    build_vector_store().ensure_collections()


async def shutdown(ctx):
    pass


if __name__ == "__main__":
    settings = get_settings()
    worker = Worker(
        functions=[process_file],
        redis_settings=RedisSettings(
            host=settings.redis_host, port=settings.redis_port
        ),
        on_startup=startup,
        on_shutdown=shutdown,
    )
    worker.run()
```

> 说明：worker 启动时用 `startup` 钩子调用 `ensure_collections()` 确保 Milvus 集合存在。

- [ ] **Step 5: 修改 `app/api/routes/knowledge.py`，上传后投递任务**

在文件顶部新增 import：

```python
from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings
```

在 `upload_file` 函数的 `return record` 之前插入：

```python
    try:
        redis = await create_pool(
            RedisSettings(
                host=get_settings().redis_host, port=get_settings().redis_port
            )
        )
        await redis.enqueue_job("process_file", record.id)
        await redis.aclose()
    except Exception:
        # 投递失败不阻塞上传，文件保持 pending，后续可手动重试
        pass
```

- [ ] **Step 6: 运行 worker 测试确认通过**

Run: `pytest tests/test_worker.py -v`
Expected: PASS (1 test)

- [ ] **Step 7: 运行全部测试**

Run: `pytest -v`
Expected: PASS（32 tests）

- [ ] **Step 8: 提交**

```powershell
git add app/worker tests/test_worker.py app/api/routes/knowledge.py
git commit -m "feat: add arq worker task for file ingestion"
```

---

## Phase 3 完成后自检清单

- [ ] `pytest -v` 全绿（32 tests）
- [ ] 上传 txt 文件返回 201，落盘 + DB `pending`
- [ ] 上传未知类型返回 400
- [ ] 手动跑 `python -m app.worker.main`（需 Redis + Milvus），上传文件后 `status` 变为 `done`
- [ ] 删除文件同时删除磁盘文件与 DB 记录（Milvus 向量删除在 Phase 4 检索链路中一并覆盖）

## 本阶段已知边界（供 Phase 4 衔接）

- 删除接口目前只删磁盘 + DB，**尚未删 Milvus 向量**；Phase 4 会在检索链路与角色删除时补齐 `VectorStore.delete_by_file` / `delete_by_character` 的调用。
- worker 需独立进程运行，容器化编排在 Phase 5 完成。
