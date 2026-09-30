# Phase 5: 前端 + 部署收尾 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可用的纯 HTML + JS 前端（登录、角色管理、知识库管理、SSE 流式聊天），将应用与 worker 容器化并入 Docker Compose，补齐角色删除的级联清理，并完成端到端验证。

**Architecture:** 前端为无框架的单页应用，由 FastAPI 通过 `StaticFiles` 托管。`login.html` 负责登录/注册（JWT 存 `localStorage`），`index.html` 为主应用（角色侧栏 + 会话 + 聊天 + 知识库）。SSE 通过 `fetch` + `ReadableStream` 消费。应用与 worker 共用同一镜像，用不同 `command` 区分。

**Tech Stack:** 原生 HTML/CSS/JS。Docker（`Dockerfile`）、Docker Compose。无新增 Python 依赖。

## Global Constraints

- 前端静态文件位于 `app/static/`，由 FastAPI `StaticFiles(html=True)` 托管于 `/`
- JWT 存浏览器 `localStorage`，所有请求带 `Authorization: Bearer <token>`
- 前端不硬编码任何 API Key；所有敏感配置只在后端环境变量
- 前端通过 `fetch` + `ReadableStream` 消费 SSE（`EventSource` 不支持 POST）
- 应用与 worker 用同一 `Dockerfile` 镜像，`docker-compose` 中覆盖 `command`
- 容器内服务互访用服务名（`mysql` / `redis` / `milvus`），由 compose 的 `environment` 覆盖

## 与 Phase 1–4 的接口约定（复用，不重新定义）

- 全部后端路由：`/api/health`、`/api/auth/*`、`/api/characters/*`、`/api/characters/{id}/knowledge`、`/api/characters/{id}/conversations`、`/api/characters/{id}/conversations/{cid}/chat`
- `python -m app.worker.main` 启动 worker
- `VectorStore.delete_by_character`（Phase 3 已定义，本阶段接入级联清理）

---

## 文件结构总览（本阶段新增/修改）

```
app/
├── static/
│   ├── login.html
│   ├── index.html
│   ├── style.css
│   └── app.js
├── main.py                          # 修改：挂载 StaticFiles
└── api/routes/
    └── characters.py                # 修改：删除角色时级联清理
Dockerfile                           # Create
docker-compose.yml                   # 修改：新增 app / worker 服务
tests/test_static.py                 # Create：验证静态托管
README.md                            # 修改：补充完整启动说明
```

---

### Task 1: 静态文件托管

**Files:**
- Create: `app/static/login.html`
- Create: `app/static/index.html`
- Create: `app/static/style.css`
- Create: `app/static/app.js`
- Modify: `app/main.py`
- Create: `tests/test_static.py`

**Interfaces:**
- Consumes: FastAPI `app`（Phase 1）
- Produces: `GET /` 返回 `index.html`；`GET /login.html` 返回登录页

- [ ] **Step 1: 写占位前端文件（先通过托管测试）**

`app/static/index.html`：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>RAG Assistant</title></head>
<body>RAG Assistant</body>
</html>
```

`app/static/login.html`：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>登录</title></head>
<body>登录页</body>
</html>
```

`app/static/style.css`（占位，后续任务填充）：

```css
body { font-family: sans-serif; }
```

`app/static/app.js`（占位）：

```js
// 主应用逻辑，后续任务填充
```

- [ ] **Step 2: 修改 `app/main.py` 挂载静态文件**

```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, characters, chat, conversations, health, knowledge
from app.database import init_db

STATIC_DIR = Path(__file__).parent / "static"

# ... lifespan 定义不变 ...

app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(characters.router, prefix="/api", tags=["characters"])
app.include_router(knowledge.router, prefix="/api", tags=["knowledge"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

- [ ] **Step 3: 写失败测试 `tests/test_static.py`**

```python
async def test_index_served(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "RAG Assistant" in resp.text
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_static.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: 提交**

```powershell
git add app/static app/main.py tests/test_static.py
git commit -m "feat: serve static frontend via FastAPI"
```

---

### Task 2: 登录页

**Files:**
- Modify: `app/static/login.html`
- Modify: `app/static/style.css`

**Interfaces:**
- Consumes: `POST /api/auth/register`、`POST /api/auth/login`
- Produces: 可用登录/注册页，成功后将 token 存 `localStorage` 并跳转 `/index.html`

- [ ] **Step 1: 重写 `app/static/login.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>登录 - RAG Assistant</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body class="auth-body">
  <div class="auth-card">
    <h1>RAG Assistant</h1>
    <p class="subtitle">多角色智能助手</p>
    <form id="auth-form">
      <input id="username" placeholder="用户名" autocomplete="username" required>
      <input id="password" type="password" placeholder="密码" autocomplete="current-password" required>
      <button id="submit-btn" type="submit">登录</button>
      <button id="toggle-btn" type="button" class="link-btn">没有账号？注册</button>
    </form>
    <p id="error" class="error"></p>
  </div>
  <script>
    let mode = 'login';
    const form = document.getElementById('auth-form');
    const submitBtn = document.getElementById('submit-btn');
    const toggleBtn = document.getElementById('toggle-btn');
    const errorEl = document.getElementById('error');

    toggleBtn.addEventListener('click', () => {
      mode = mode === 'login' ? 'register' : 'login';
      submitBtn.textContent = mode === 'login' ? '登录' : '注册';
      toggleBtn.textContent = mode === 'login' ? '没有账号？注册' : '已有账号？登录';
      errorEl.textContent = '';
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      errorEl.textContent = '';
      const username = document.getElementById('username').value;
      const password = document.getElementById('password').value;
      const resp = await fetch(`/api/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        errorEl.textContent = body.detail || '操作失败';
        return;
      }
      if (mode === 'register') {
        mode = 'login';
        submitBtn.textContent = '登录';
        toggleBtn.textContent = '没有账号？注册';
        errorEl.textContent = '注册成功，请登录';
        return;
      }
      const body = await resp.json();
      localStorage.setItem('token', body.access_token);
      location.href = '/index.html';
    });
  </script>
</body>
</html>
```

- [ ] **Step 2: 在 `app/static/style.css` 追加登录页样式**

```css
.auth-body {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  margin: 0;
  background: #f0f2f5;
}
.auth-card {
  background: #fff;
  padding: 40px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  width: 320px;
  text-align: center;
}
.auth-card h1 { margin: 0 0 4px; font-size: 22px; }
.subtitle { color: #888; margin: 0 0 24px; }
.auth-card input {
  width: 100%; padding: 10px; margin-bottom: 12px;
  border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box;
}
.auth-card button[type="submit"] {
  width: 100%; padding: 10px; background: #4f6ef7; color: #fff;
  border: none; border-radius: 6px; cursor: pointer; font-size: 15px;
}
.link-btn { background: none; border: none; color: #4f6ef7; cursor: pointer; margin-top: 12px; }
.error { color: #d93025; min-height: 20px; margin-top: 8px; }
```

- [ ] **Step 3: 手动验证（可选）**

启动 uvicorn 后访问 `http://localhost:8000/login.html`，确认能登录/注册并跳转。

- [ ] **Step 4: 提交**

```powershell
git add app/static/login.html app/static/style.css
git commit -m "feat: add login/register page"
```

---

### Task 3: 主应用页（角色 + 会话 + 聊天 + 知识库）

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/app.js`
- Modify: `app/static/style.css`

**Interfaces:**
- Consumes: 全部 `/api/characters/*`、`/api/characters/{id}/knowledge`、`/api/characters/{id}/conversations`、`/api/characters/{id}/conversations/{cid}/chat`
- Produces: 可用的聊天界面

- [ ] **Step 1: 重写 `app/static/index.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>RAG Assistant</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body class="app-body">
  <div class="app">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h2>我的角色</h2>
        <button id="new-char" class="icon-btn">＋</button>
      </div>
      <ul id="char-list" class="char-list"></ul>
      <div class="sidebar-footer">
        <span id="me-name"></span>
        <button id="logout" class="link-btn">退出</button>
      </div>
    </aside>

    <main class="main">
      <div id="no-char" class="empty">请选择或创建一个角色</div>

      <div id="chat-panel" class="chat-panel hidden">
        <header class="chat-header">
          <div>
            <h3 id="char-name"></h3>
            <p id="char-prompt" class="char-prompt"></p>
          </div>
          <button id="new-conv" class="icon-btn">新对话</button>
        </header>

        <div class="chat-body">
          <aside class="conv-list">
            <ul id="conv-list"></ul>
          </aside>
          <div class="chat-area">
            <div id="messages" class="messages"></div>
            <form id="chat-form" class="chat-input">
              <input id="chat-input" placeholder="输入消息…" autocomplete="off">
              <button type="submit">发送</button>
            </form>
          </div>
        </div>
      </div>

      <div id="kb-panel" class="kb-panel">
        <h4>知识库</h4>
        <input id="kb-file" type="file" multiple>
        <button id="kb-upload">上传</button>
        <ul id="kb-list"></ul>
      </div>
    </main>
  </div>

  <dialog id="char-dialog">
    <form id="char-form">
      <h3>创建角色</h3>
      <label>名称<input id="c-name" required></label>
      <label>模板
        <select id="c-template"><option value="">自定义</option></select>
      </label>
      <label>系统提示词<textarea id="c-prompt" rows="4"></textarea></label>
      <label>模型名<input id="c-model" value="gpt-4o-mini"></label>
      <label>Base URL（可选）<input id="c-baseurl" placeholder="留空使用默认"></label>
      <div class="dialog-actions">
        <button type="button" id="c-cancel">取消</button>
        <button type="submit">创建</button>
      </div>
    </form>
  </dialog>

  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 重写 `app/static/app.js`**

```js
const token = localStorage.getItem('token');
if (!token) location.href = '/login.html';

const state = { characters: [], currentCharId: null, conversations: [], currentConvId: null };

function authHeaders(extra = {}) {
  return { 'Authorization': `Bearer ${token}`, ...extra };
}

async function api(path, options = {}) {
  const resp = await fetch(path, { ...options, headers: authHeaders(options.headers || {}) });
  if (resp.status === 401) { location.href = '/login.html'; return; }
  return resp;
}

// ---------- 用户 ----------
async function loadMe() {
  const resp = await api('/api/auth/me');
  if (!resp || !resp.ok) return;
  document.getElementById('me-name').textContent = (await resp.json()).username;
}

// ---------- 角色 ----------
async function loadCharacters() {
  const resp = await api('/api/characters');
  if (!resp || !resp.ok) return;
  state.characters = await resp.json();
  renderCharacters();
}

function renderCharacters() {
  const list = document.getElementById('char-list');
  list.innerHTML = '';
  state.characters.forEach(c => {
    const li = document.createElement('li');
    li.textContent = c.name;
    li.className = c.id === state.currentCharId ? 'active' : '';
    li.onclick = () => selectCharacter(c.id);
    list.appendChild(li);
  });
}

async function selectCharacter(id) {
  state.currentCharId = id;
  state.currentConvId = null;
  renderCharacters();
  const c = state.characters.find(x => x.id === id);
  document.getElementById('no-char').classList.add('hidden');
  document.getElementById('chat-panel').classList.remove('hidden');
  document.getElementById('char-name').textContent = c.name;
  document.getElementById('char-prompt').textContent = c.system_prompt.slice(0, 60);
  document.getElementById('messages').innerHTML = '';
  await loadConversations();
  await loadKnowledge();
}

// ---------- 会话 ----------
async function loadConversations() {
  const resp = await api(`/api/characters/${state.currentCharId}/conversations`);
  if (!resp || !resp.ok) return;
  state.conversations = await resp.json();
  renderConversations();
}

function renderConversations() {
  const list = document.getElementById('conv-list');
  list.innerHTML = '';
  state.conversations.forEach(cv => {
    const li = document.createElement('li');
    li.textContent = cv.title;
    li.className = cv.id === state.currentConvId ? 'active' : '';
    li.onclick = () => selectConversation(cv.id);
    list.appendChild(li);
  });
}

function selectConversation(id) {
  state.currentConvId = id;
  renderConversations();
  document.getElementById('messages').innerHTML = '';
}

async function createConversation() {
  if (!state.currentCharId) return;
  const resp = await api(`/api/characters/${state.currentCharId}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: '新对话' }),
  });
  if (!resp || !resp.ok) return;
  await loadConversations();
  const last = state.conversations[0];
  if (last) selectConversation(last.id);
}

// ---------- 聊天（SSE）----------
function appendMessage(role, content) {
  const box = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = content;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

async function sendMessage(text) {
  if (!state.currentConvId) await createConversation();
  if (!state.currentConvId) return;
  appendMessage('user', text);
  const assistantDiv = appendMessage('assistant', '');
  const resp = await fetch(
    `/api/characters/${state.currentCharId}/conversations/${state.currentConvId}/chat`,
    {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ message: text }),
    }
  );
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') continue;
      try {
        assistantDiv.textContent += JSON.parse(data).delta;
      } catch (_) {}
    }
    document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
  }
}

// ---------- 知识库 ----------
async function loadKnowledge() {
  const resp = await api(`/api/characters/${state.currentCharId}/knowledge`);
  if (!resp || !resp.ok) return;
  const files = await resp.json();
  const list = document.getElementById('kb-list');
  list.innerHTML = '';
  files.forEach(f => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${f.filename} <em>${f.status}</em></span>`;
    const del = document.createElement('button');
    del.textContent = '删除';
    del.onclick = async () => {
      await api(`/api/characters/${state.currentCharId}/knowledge/${f.id}`, { method: 'DELETE' });
      loadKnowledge();
    };
    li.appendChild(del);
    list.appendChild(li);
  });
}

async function uploadKnowledge() {
  const input = document.getElementById('kb-file');
  for (const file of input.files) {
    const fd = new FormData();
    fd.append('file', file);
    await api(`/api/characters/${state.currentCharId}/knowledge`, { method: 'POST', body: fd });
  }
  input.value = '';
  loadKnowledge();
}

// ---------- 创建角色对话框 ----------
async function loadTemplates() {
  const resp = await api('/api/characters/templates');
  if (!resp || !resp.ok) return;
  const sel = document.getElementById('c-template');
  (await resp.json()).forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.key;
    opt.textContent = t.name;
    sel.appendChild(opt);
  });
}

document.getElementById('char-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    name: document.getElementById('c-name').value,
    system_prompt: document.getElementById('c-prompt').value,
    model_name: document.getElementById('c-model').value,
    base_url: document.getElementById('c-baseurl').value || null,
    template_key: document.getElementById('c-template').value || null,
  };
  const resp = await api('/api/characters', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (resp && resp.ok) {
    document.getElementById('char-dialog').close();
    loadCharacters();
  }
});

// ---------- 事件绑定 ----------
document.getElementById('logout').onclick = () => {
  localStorage.removeItem('token');
  location.href = '/login.html';
};
document.getElementById('new-char').onclick = () => document.getElementById('char-dialog').showModal();
document.getElementById('c-cancel').onclick = () => document.getElementById('char-dialog').close();
document.getElementById('new-conv').onclick = createConversation;
document.getElementById('kb-upload').onclick = uploadKnowledge;
document.getElementById('chat-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendMessage(text);
});

loadMe();
loadCharacters();
loadTemplates();
```

- [ ] **Step 3: 在 `app/static/style.css` 追加主应用样式**

```css
.app-body { margin: 0; font-family: sans-serif; }
.app { display: flex; height: 100vh; }
.sidebar { width: 220px; background: #1f2430; color: #fff; display: flex; flex-direction: column; }
.sidebar-header { display: flex; justify-content: space-between; align-items: center; padding: 16px; }
.char-list, .conv-list { list-style: none; padding: 0; margin: 0; }
.char-list li, .conv-list li { padding: 10px 16px; cursor: pointer; }
.char-list li.active, .conv-list li.active { background: #4f6ef7; }
.char-list li:hover, .conv-list li:hover { background: #2a3142; }
.sidebar-footer { margin-top: auto; padding: 16px; display: flex; justify-content: space-between; }
.main { flex: 1; display: flex; flex-direction: column; }
.empty { margin: auto; color: #999; }
.chat-panel { flex: 1; display: flex; flex-direction: column; }
.hidden { display: none !important; }
.chat-header { padding: 12px 16px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
.char-prompt { color: #888; font-size: 12px; margin: 0; }
.chat-body { flex: 1; display: flex; min-height: 0; }
.conv-list { width: 200px; border-right: 1px solid #eee; overflow-y: auto; }
.chat-area { flex: 1; display: flex; flex-direction: column; }
.messages { flex: 1; overflow-y: auto; padding: 16px; }
.msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 8px; max-width: 70%; white-space: pre-wrap; }
.msg.user { background: #4f6ef7; color: #fff; margin-left: auto; }
.msg.assistant { background: #f0f2f5; }
.chat-input { display: flex; padding: 12px; border-top: 1px solid #eee; }
.chat-input input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 6px; }
.chat-input button { margin-left: 8px; padding: 10px 20px; background: #4f6ef7; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
.kb-panel { padding: 16px; border-top: 1px solid #eee; }
.kb-panel ul { list-style: none; padding: 0; }
.kb-panel li { display: flex; justify-content: space-between; padding: 6px 0; }
.icon-btn { background: #4f6ef7; color: #fff; border: none; border-radius: 6px; padding: 6px 12px; cursor: pointer; }
dialog form { display: flex; flex-direction: column; gap: 10px; min-width: 360px; }
dialog label { display: flex; flex-direction: column; gap: 4px; font-size: 14px; }
.dialog-actions { display: flex; justify-content: flex-end; gap: 8px; }
em { color: #999; font-size: 12px; }
```

- [ ] **Step 4: 运行后端测试确保静态托管仍通过**

Run: `pytest -v`
Expected: PASS（42 tests）

- [ ] **Step 5: 手动验证（可选）**

启动 uvicorn + 中间件 + worker 后，访问 `http://localhost:8000/`，走通「登录 → 创建角色 → 上传文件 → 对话」全流程。

- [ ] **Step 6: 提交**

```powershell
git add app/static/index.html app/static/app.js app/static/style.css
git commit -m "feat: add main chat interface"
```

---

### Task 4: 角色删除的级联清理

**Files:**
- Modify: `app/api/routes/characters.py`

**Interfaces:**
- Consumes: `VectorStore.delete_by_character`（Phase 3）、`ShortTermMemory`（Phase 4）、`conversation_service`（Phase 4）
- Produces: 删除角色时同步清理 Milvus 向量与 Redis 会话

- [ ] **Step 1: 修改 `app/api/routes/characters.py` 的 `delete_character`**

```python
import asyncio

from app.core.memory import ShortTermMemory
from app.core.redis_client import get_redis
from app.core.vector_store import VectorStore
from app.config import get_settings


@router.delete("/characters/{character_id}", status_code=204)
async def delete_character(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ok = await character_service.delete_character(db, current_user.id, character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="角色不存在")

    # 级联清理：清空该角色的会话 Redis 记忆
    convs = await conversation_service.list_conversations(db, current_user.id, character_id)
    mem = ShortTermMemory(get_redis())
    for c in convs:
        await mem.clear(c.id)

    # 级联清理：删除该角色的 Milvus 向量
    settings = get_settings()
    store = VectorStore(
        uri=settings.milvus_uri,
        text_dim=settings.text_embedding_dim,
        image_dim=settings.image_embedding_dim,
    )
    await asyncio.to_thread(store.delete_by_character, character_id)
```

同时在文件顶部补充 import（`conversation_service`、`ShortTermMemory`、`get_redis`、`VectorStore`、`get_settings`、`asyncio`）。

- [ ] **Step 2: 运行全部测试确认无回归**

Run: `pytest -v`
Expected: PASS（42 tests）

- [ ] **Step 3: 提交**

```powershell
git add app/api/routes/characters.py
git commit -m "feat: cascade-clean vectors and sessions on character delete"
```

---

### Task 5: 容器化与编排

**Files:**
- Create: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `requirements.txt`、`python -m app.worker.main`、`uvicorn app.main:app`
- Produces: `docker compose up` 一键启动全部服务

- [ ] **Step 1: 写 `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 修改 `docker-compose.yml` 追加 app / worker 服务**

在 `services:` 下追加（与 mysql/redis 平级）：

```yaml
  app:
    build: .
    container_name: rag-app
    ports:
      - "8000:8000"
    environment:
      MYSQL_HOST: mysql
      REDIS_HOST: redis
      MILVUS_HOST: milvus
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_started
      milvus:
        condition: service_started

  worker:
    build: .
    container_name: rag-worker
    command: python -m app.worker.main
    environment:
      MYSQL_HOST: mysql
      REDIS_HOST: redis
      MILVUS_HOST: milvus
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_started
      milvus:
        condition: service_started
```

- [ ] **Step 3: 追加 `.env.example` 容器内说明**

```powershell
Add-Content .env.example @'

# 容器部署时 MYSQL_HOST/REDIS_HOST/MILVUS_HOST 会被 compose 覆盖为服务名
# 本地开发保持 localhost 即可
'@
```

- [ ] **Step 4: 构建并启动（需 Docker）**

```powershell
docker compose build
docker compose up -d
```

Expected: 所有服务 running；`http://localhost:8000/` 可访问。

- [ ] **Step 5: 提交**

```powershell
git add Dockerfile docker-compose.yml .env.example
git commit -m "feat: containerize app and worker with docker compose"
```

---

### Task 6: 端到端验证与文档

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: 全部
- Produces: 完整启动文档 + 验证记录

- [ ] **Step 1: 更新 `README.md`**

```markdown
# RAG Assistant

多用户、多角色的 RAG 智能聊天助手。支持本地（Ollama/vLLM）与云端大模型、
知识库文件（txt/pdf/图片）、多轮流式对话、Redis 短期记忆、Milvus 长期记忆。

## 一键启动（Docker）

1. 复制配置并填写模型 API Key：

```powershell
Copy-Item .env.example .env
```

2. 启动全部服务（MySQL / Redis / Milvus / App / Worker）：

```powershell
docker compose up -d --build
```

3. 访问 http://localhost:8000/

## 本地开发

1. 启动中间件：

```powershell
docker compose up -d mysql redis etcd minio milvus
```

2. 安装依赖并启动应用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

3. 另起一个终端启动 worker：

```powershell
python -m app.worker.main
```

## 测试

```powershell
pytest -v
```

## 配置说明

- `LLM_BASE_URL` / `LLM_API_KEY`：对话模型（OpenAI 兼容；本地 Ollama 填 `http://localhost:11434/v1`，key 填 `ollama`）
- `EMBEDDING_MODEL` / `EMBEDDING_API_KEY`：文本向量模型
- `VISION_EMBEDDING_*`：图片视觉向量多模态 API
- `TEXT_EMBEDDING_DIM` / `IMAGE_EMBEDDING_DIM`：向量维度，须与所选模型一致
```

- [ ] **Step 2: 端到端手工验证清单（记录结果）**

按顺序验证并记录：
1. `docker compose up -d --build` 全部 healthy
2. 注册 → 登录 → 创建「中医」角色（用模板）
3. 上传一个 txt 文件，观察 worker 日志中 `status=done`
4. 发消息，观察 SSE 流式回复，且回复引用了知识库内容
5. 刷新页面，会话列表与多轮上下文仍在（Redis）
6. 连续对话触发长期记忆后，新会话提问可命中记忆
7. 删除角色后，其会话与向量被清理

- [ ] **Step 3: 提交**

```powershell
git add README.md
git commit -m "docs: finalize README with full setup and verification"
```

---

## Phase 5 完成后自检清单

- [ ] `pytest -v` 全绿（42 tests）
- [ ] `docker compose up -d --build` 一键启动，前端可访问
- [ ] 完整链路：登录 → 建角色 → 上传知识库 → 流式多轮对话 → 记忆沉淀 → 角色删除清理

## 全部阶段完成后的总验收

对照需求规格 `docs/superpowers/specs/2026-09-22-rag-assistant-requirements.md` 第 8 节验收标准逐条核对，确认 10 条标准全部满足。
