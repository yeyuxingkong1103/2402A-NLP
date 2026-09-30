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
        const parsed = JSON.parse(data);
        if (parsed.error) { assistantDiv.textContent = '[错误] ' + parsed.error; }
        else assistantDiv.textContent += parsed.delta || '';
      } catch (_) {}
    }
    const box = document.getElementById('messages');
    box.scrollTop = box.scrollHeight;
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
