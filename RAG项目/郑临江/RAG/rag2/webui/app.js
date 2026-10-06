/* RAG2 在线问答前端 —— 零依赖原生 JS（不使用任何 CDN） */
'use strict';

const state = {
  token: localStorage.getItem('rag2_token') || '',
  user: null,
  roles: [],
  roleId: localStorage.getItem('rag2_role') || '',
  sessionId: null,
  sessions: [],
  streaming: false,
};

const $ = (id) => document.getElementById(id);

const SCORE_LABELS = {
  faithfulness: '忠实度', answer_relevancy: '答案相关性',
  context_precision: '上下文精确率', context_recall: '上下文召回率',
  context_relevance: '上下文相关性', answer_correctness: '答案正确性',
  context_entity_recall: '上下文实体召回', semantic_similarity: '语义相似度',
};

/* ------------------------------------------------------------------ 请求 */
function authHeaders(extra) {
  const headers = Object.assign({}, extra || {});
  if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
  return headers;
}

async function api(path, options) {
  const opts = options || {};
  opts.headers = authHeaders(opts.headers);
  if (opts.body && typeof opts.body !== 'string') {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { message: text }; }
  if (res.status === 401) { showLogin('登录已过期，请重新登录'); throw new Error('未登录'); }
  if (!res.ok) throw new Error(data.detail || data.message || ('HTTP ' + res.status));
  return data;
}

/* ------------------------------------------------------------ SSE 流式 */
async function streamChat(payload, handlers) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: 'HTTP ' + res.status }));
    throw new Error(err.detail || err.message || '请求失败');
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let index;
    while ((index = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      let event = 'message';
      let data = '';
      block.split('\n').forEach((line) => {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      });
      if (!data) continue;
      let parsed = {};
      try { parsed = JSON.parse(data); } catch (e) { parsed = { raw: data }; }
      if (handlers[event]) handlers[event](parsed);
    }
  }
}

/* ------------------------------------------------------------ 登录/退出 */
function showLogin(message) {
  $('login-mask').hidden = false;
  $('layout').hidden = true;
  if (message) $('login-error').textContent = message;
}

function showApp() {
  $('login-mask').hidden = true;
  $('layout').hidden = false;
}

$('login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('login-error').textContent = '';
  $('login-btn').disabled = true;
  try {
    const data = await api('/api/auth/login', {
      method: 'POST',
      body: { username: $('login-user').value.trim() },
    });
    state.token = data.token;
    state.user = data.user;
    localStorage.setItem('rag2_token', data.token);
    await boot();
  } catch (err) {
    $('login-error').textContent = err.message;
  } finally {
    $('login-btn').disabled = false;
  }
});

$('btn-logout').addEventListener('click', async () => {
  try { await api('/api/auth/logout', { method: 'POST' }); } catch (e) { /* 忽略 */ }
  state.token = '';
  state.user = null;
  localStorage.removeItem('rag2_token');
  localStorage.removeItem('rag2_role');
  showLogin('');
});

/* ------------------------------------------------------------------ 启动 */
async function boot() {
  const me = await api('/api/auth/me');
  state.user = me.user;
  $('user-line').textContent = me.user.display_name + '（当前用户）';
  const roleData = await api('/api/roles');
  state.roles = roleData.roles;
  if (!state.roles.some((role) => role.role_id === state.roleId)) {
    state.roleId = state.roles.length ? state.roles[0].role_id : '';
  }
  renderRoles();
  await loadSessions();
  showApp();
}

function renderRoles() {
  const box = $('role-picker');
  box.innerHTML = '';
  state.roles.forEach((role) => {
    const button = document.createElement('button');
    button.className = 'role-chip' + (role.role_id === state.roleId ? ' active' : '');
    button.innerHTML = '<span>' + role.avatar + ' ' + escapeHtml(role.name) + '</span>';
    button.title = role.description || '';
    button.addEventListener('click', () => {
      state.roleId = role.role_id;
      state.sessionId = null;
      localStorage.setItem('rag2_role', role.role_id);
      renderRoles();
      renderWelcome();
    });
    box.appendChild(button);
  });
}

function currentRole() {
  return state.roles.find((role) => role.role_id === state.roleId) || null;
}

/* ------------------------------------------------------------------ 会话 */
async function loadSessions() {
  try {
    const data = await api('/api/sessions?limit=30');
    state.sessions = data.sessions;
  } catch (err) { state.sessions = []; }
  renderSessions();
}

function renderSessions() {
  const box = $('session-list');
  box.innerHTML = '';
  if (!state.sessions.length) {
    box.innerHTML = '<div class="tiny" style="padding:6px">还没有历史会话</div>';
    return;
  }
  state.sessions.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'session-item' + (item.session_id === state.sessionId ? ' active' : '');
    const when = new Date(parseFloat(item.last_active) * 1000).toLocaleString('zh-CN', { hour12: false });
    div.innerHTML = '<div class="st">' + escapeHtml(item.title || '新会话') + '</div>'
      + '<div class="sm"><span class="tiny">' + (item.role_id || '') + ' · ' + (item.turns || 0) + ' 轮</span>'
      + '<span class="tiny">' + when + '</span></div>';
    div.addEventListener('click', () => openSession(item.session_id));
    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '删除';
    del.addEventListener('click', async (event) => {
      event.stopPropagation();
      await api('/api/sessions/' + item.session_id, { method: 'DELETE' });
      if (state.sessionId === item.session_id) { state.sessionId = null; renderWelcome(); }
      await loadSessions();
    });
    div.querySelector('.sm').appendChild(del);
    box.appendChild(div);
  });
}

async function openSession(sessionId) {
  const data = await api('/api/sessions/' + sessionId);
  state.sessionId = sessionId;
  if (data.session.role_id && data.session.role_id !== state.roleId
      && state.roles.some((role) => role.role_id === data.session.role_id)) {
    state.roleId = data.session.role_id;
    renderRoles();
  }
  $('chat').innerHTML = '';
  data.messages.forEach((message) => {
    if (message.role === 'user') {
      addMessage('user', message.content, {});
    } else {
      const target = addMessage('assistant', message.content, { citations: message.citations || [] });
      target.wrap.dataset.question = message.question || '';
      attachEvaluateButton(target.wrap, []);
    }
  });
  renderSessions();
  scrollBottom();
}

$('btn-new').addEventListener('click', () => {
  state.sessionId = null;
  renderSessions();
  renderWelcome();
});

/* ------------------------------------------------------------------ 渲染 */
function escapeHtml(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderWelcome() {
  const role = currentRole();
  const chat = $('chat');
  chat.innerHTML = '';
  if (!role) { chat.innerHTML = '<div class="welcome">当前账号没有可用角色</div>'; return; }
  const box = document.createElement('div');
  box.className = 'welcome';
  box.innerHTML = '<h2>' + role.avatar + ' ' + escapeHtml(role.name) + '</h2>'
    + '<div>' + escapeHtml(role.description) + '</div>'
    + '<div class="tiny" style="margin-top:8px">多轮对话 + 混合检索（Milvus + BM25）+ Redis 短期记忆</div>';
  const list = document.createElement('div');
  list.className = 'followups';
  (role.followups || []).forEach((question) => {
    const button = document.createElement('button');
    button.textContent = '· ' + question;
    button.addEventListener('click', () => { $('input').value = question; send(); });
    list.appendChild(button);
  });
  box.appendChild(list);
  chat.appendChild(box);
}

function addMessage(who, content, options) {
  const opts = options || {};
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + who;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  wrap.appendChild(bubble);
  if (opts.citations && opts.citations.length) {
    wrap.appendChild(renderCitations(opts.citations));
  }
  $('chat').appendChild(wrap);
  scrollBottom();
  return { wrap, bubble };
}

function renderCitations(citations) {
  const bar = document.createElement('div');
  bar.className = 'cite-bar';
  citations.forEach((cite) => {
    const chip = document.createElement('span');
    chip.className = 'cite';
    chip.textContent = '[' + cite.index + '] ' + (cite.source || cite.doc_id || '来源');
    chip.title = (cite.snippet || '') + '\n\n来源：' + (cite.source || '') + (cite.page ? '（第 ' + cite.page + ' 页）' : '');
    bar.appendChild(chip);
  });
  return bar;
}

function attachEvaluateButton(wrap, contexts) {
  const btn = document.createElement('button');
  btn.className = 'eval-btn';
  btn.textContent = '评估本条';
  btn.addEventListener('click', async () => {
    const question = wrap.dataset.question || '';
    const answer = wrap.querySelector('.bubble').textContent || '';
    let ctxs = contexts;
    if (!ctxs.length && wrap.dataset.contexts) {
      try { ctxs = JSON.parse(wrap.dataset.contexts); } catch (e) { ctxs = []; }
    }
    let box = wrap.querySelector('.eval-box');
    if (!box) {
      box = document.createElement('div');
      box.className = 'eval-box';
      wrap.appendChild(box);
    }
    box.innerHTML = '评估中，请稍候…';
    try {
      const data = await api('/api/evaluate', {
        method: 'POST',
        body: { question, answer, retrieved_contexts: ctxs },
      });
      box.innerHTML = renderScores(data.scores);
    } catch (err) {
      box.innerHTML = '评估失败：' + escapeHtml(err.message);
    }
  });
  wrap.appendChild(btn);
}

function renderScores(scores) {
  const rows = Object.entries(scores || {}).map(([key, value]) => {
    const label = SCORE_LABELS[key] || key;
    const val = value == null ? '—' : (typeof value === 'number' ? value.toFixed(3) : value);
    return '<div class="score-row"><span>' + escapeHtml(label) + '</span><span>' + escapeHtml(String(val)) + '</span></div>';
  });
  return '<h4>RAGAS 评分（0~1，越接近 1 越好）</h4>' + rows.join('');
}

function scrollBottom() {
  const chat = $('chat');
  chat.scrollTop = chat.scrollHeight;
}

/* ------------------------------------------------------------------ 发送 */
async function send() {
  if (state.streaming) return;
  const text = $('input').value.trim();
  if (!text) return;
  if (!state.roleId) { alert('请先选择角色'); return; }
  state.streaming = true;
  $('btn-send').disabled = true;
  $('input').value = '';
  if ($('chat').querySelector('.welcome')) $('chat').innerHTML = '';
  addMessage('user', text, {});
  const target = addMessage('assistant', '', {});
  target.wrap.dataset.question = text;
  let buffer = '';
  let contexts = [];
  let citations = [];

  try {
    await streamChat({
      question: text,
      role_id: state.roleId,
      session_id: state.sessionId,
      stream: true,
    }, {
      meta: (data) => {
        state.sessionId = data.session_id;
        contexts = data.contexts || [];
        citations = data.citations || [];
      },
      delta: (data) => { buffer += data.text; target.bubble.textContent = buffer; scrollBottom(); },
      done: (data) => {
        target.bubble.textContent = data.answer || buffer;
        citations = data.citations || citations;
        if (citations.length) {
          target.wrap.insertBefore(renderCitations(citations), target.bubble.nextSibling);
        }
        target.wrap.dataset.contexts = JSON.stringify(contexts);
        attachEvaluateButton(target.wrap, contexts);
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = '生成耗时 ' + ((data.timings && data.timings.generation) || 0).toFixed(1) + 's';
        target.wrap.appendChild(meta);
      },
      error: (data) => {
        target.bubble.textContent = '出错了：' + (data.message || '未知错误');
        target.bubble.style.color = 'var(--danger)';
      },
    });
  } catch (err) {
    target.bubble.textContent = '出错了：' + err.message;
    target.bubble.style.color = 'var(--danger)';
  } finally {
    state.streaming = false;
    $('btn-send').disabled = false;
    $('input').focus();
    loadSessions();
    scrollBottom();
  }
}

$('btn-send').addEventListener('click', send);
$('input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); }
});

/* ------------------------------------------------------------------ 入口 */
(async function start() {
  if (!state.token) { showLogin(''); return; }
  try {
    await boot();
  } catch (err) {
    showLogin('自动登录失败：' + err.message);
  }
})();
