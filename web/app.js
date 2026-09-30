/* Role RAG_try 前端：零依赖原生 JS（不使用任何 CDN） */
'use strict';

const state = {
  token: localStorage.getItem('role_rag_token') || '',
  user: null,
  roles: [],
  roleId: localStorage.getItem('role_rag_role') || '',
  sessionId: null,
  sessions: [],
  streaming: false,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ 请求 */
function authHeaders(extra) {
  const headers = Object.assign({}, extra || {});
  if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
  return headers;
}

async function api(path, options) {
  const opts = options || {};
  opts.headers = authHeaders(opts.headers);
  if (opts.body && typeof opts.body !== 'string' && !(opts.body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, opts);
  if (res.status === 401) { showLogin('登录已过期，请重新登录'); throw new Error('未登录'); }
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { message: text }; }
  if (!res.ok) throw new Error(data.message || ('HTTP ' + res.status));
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
    throw new Error(err.message || '请求失败');
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
      body: { username: $('login-user').value.trim(), password: $('login-pass').value },
    });
    state.token = data.token;
    state.user = data.user;
    localStorage.setItem('role_rag_token', data.token);
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
  localStorage.removeItem('role_rag_token');
  showLogin('');
});

/* ------------------------------------------------------------------ 启动 */
async function boot() {
  const me = await api('/api/auth/me');
  state.user = me.user;
  $('user-line').textContent = me.user.display_name + ' · ' + me.user.username
    + (me.user.is_admin ? ' · 管理员' : '');
  const roleData = await api('/api/roles');
  state.roles = roleData.roles;
  if (!state.roles.some((role) => role.id === state.roleId)) {
    state.roleId = state.roles.length ? state.roles[0].id : '';
  }
  renderRoles();
  await loadSessions();
  showApp();
  refreshState();
}

function renderRoles() {
  const box = $('role-picker');
  box.innerHTML = '';
  state.roles.forEach((role) => {
    const button = document.createElement('button');
    button.className = 'role-chip' + (role.id === state.roleId ? ' active' : '');
    button.innerHTML = '<span>' + role.avatar + ' ' + role.name + '</span>'
      + '<span class="cat">' + role.category + '</span>';
    button.title = role.tagline;
    button.addEventListener('click', () => {
      state.roleId = role.id;
      state.sessionId = null;
      localStorage.setItem('role_rag_role', role.id);
      renderRoles();
      renderWelcome();
    });
    box.appendChild(button);
  });
}

function currentRole() {
  return state.roles.find((role) => role.id === state.roleId) || null;
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
    box.innerHTML = '<div class="tiny">还没有历史会话</div>';
    return;
  }
  state.sessions.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'session-item' + (item.session_id === state.sessionId ? ' active' : '');
    const when = new Date(parseFloat(item.last_active) * 1000).toLocaleString('zh-CN', { hour12: false });
    div.innerHTML = '<div class="st">' + escapeHtml(item.title || '新会话') + '</div>'
      + '<div class="sm"><span>' + (item.role_id || '') + ' · ' + (item.turns || 0) + ' 轮</span>'
      + '<span>' + when + '</span></div>';
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
      && state.roles.some((role) => role.id === data.session.role_id)) {
    state.roleId = data.session.role_id;
    renderRoles();
  }
  $('chat').innerHTML = '';
  data.messages.forEach((message) => {
    addMessage(message.role, message.content, {
      citations: message.citations || [],
      meta: message.role === 'assistant' ? '历史消息' : '',
    });
  });
  if (data.summary) {
    const notice = document.createElement('div');
    notice.className = 'notice';
    notice.textContent = '短期记忆摘要：' + data.summary;
    $('chat').prepend(notice);
  }
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
    + '<div>' + escapeHtml(role.tagline) + '</div>'
    + '<div class="tiny" style="margin-top:10px">知识域：' + escapeHtml((role.kb_dirs || []).join(' / '))
    + ' + shared · 检索：稠密 + 稀疏 + BM25 加权 RRF</div>';
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
  if (opts.meta) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = opts.meta;
    wrap.appendChild(meta);
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
    chip.textContent = '[' + cite.index + '] 《' + cite.doc_title + '》'
      + (cite.section ? ' › ' + String(cite.section).split(' › ').pop() : '');
    chip.title = (cite.snippet || '') + '\n\n来源：' + (cite.source || '') + '\n知识域：' + (cite.scope || '');
    bar.appendChild(chip);
  });
  const hint = document.createElement('span');
  hint.className = 'tiny';
  hint.textContent = '（悬停查看原文片段）';
  bar.appendChild(hint);
  return bar;
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
  let buffer = '';

  try {
    await streamChat({
      question: text,
      role_id: state.roleId,
      session_id: state.sessionId,
      mode: $('mode-select').value,
      top_k: parseInt($('topk-input').value, 10) || 6,
      stream: true,
    }, {
      meta: (data) => {
        state.sessionId = data.session_id;
        const routes = data.retrieval && data.retrieval.candidates
          ? Object.entries(data.retrieval.candidates).map(([key, value]) => key + ':' + value).join(' ')
          : '';
        target.wrap.querySelector('.meta') || target.wrap.appendChild(document.createElement('div'));
        const meta = target.wrap.lastElementChild;
        meta.className = 'meta';
        meta.textContent = '角色 ' + data.role_name + ' · 候选 ' + (routes || '-')
          + (data.retrieval && data.retrieval.cached ? ' · 命中缓存' : '')
          + (data.safety_notice ? ' · 已触发安全纠正' : '');
        if (data.safety_notice) {
          const notice = document.createElement('div');
          notice.className = 'notice';
          notice.textContent = '该问题触及角色红线，已要求模型先纠正前提再作答。';
          target.wrap.insertBefore(notice, target.bubble);
        }
      },
      delta: (data) => { buffer += data.text; target.bubble.textContent = buffer; scrollBottom(); },
      done: (data) => {
        target.bubble.textContent = data.answer || buffer;
        if (data.citations && data.citations.length) {
          target.wrap.insertBefore(renderCitations(data.citations), target.bubble.nextSibling);
        }
        const meta = target.wrap.lastElementChild;
        if (meta && meta.className === 'meta') {
          meta.textContent += ' · ' + (data.usage && data.usage.completion_tokens ? data.usage.completion_tokens : 0)
            + ' tokens · ' + ((data.timings && data.timings.generation) || 0).toFixed(1) + 's 生成'
            + (data.cited ? '' : ' · 未标注引用');
        }
        if (data.guardrails && data.guardrails.triggered) {
          const notice = document.createElement('div');
          notice.className = 'notice';
          notice.textContent = '命中角色红线词：' + data.guardrails.hits.join('、') + '（已追加免责声明）';
          target.wrap.appendChild(notice);
        }
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
    refreshTab('state');
    scrollBottom();
  }
}

$('btn-send').addEventListener('click', send);
$('input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); }
});

/* ------------------------------------------------------------------ 右侧面板 */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((item) => item.classList.remove('active'));
    tab.classList.add('active');
    ['state', 'memory', 'debug', 'logs'].forEach((name) => {
      $('tab-' + name).hidden = name !== tab.dataset.tab;
    });
    refreshTab(tab.dataset.tab);
  });
});

function kv(label, value) {
  return '<div class="kv"><span>' + escapeHtml(label) + '</span><span>' + escapeHtml(value) + '</span></div>';
}

async function refreshState() { return refreshTab('state'); }

async function refreshTab(name) {
  try {
    if (name === 'state') {
      const health = await api('/api/health');
      const stats = await api('/api/stats');
      let html = '<div class="card"><h4>组件</h4>';
      Object.entries(health.components || {}).forEach(([key, value]) => {
        html += kv(key, value.ok ? '正常' : ('异常：' + (value.error || '')));
      });
      html += kv('Milvus 版本', (health.components.milvus || {}).version || '-');
      html += kv('知识块数量', (health.components.milvus || {}).kb_rows || 0);
      html += '</div><div class="card"><h4>模型（本地）</h4>';
      html += kv('嵌入模型', health.models.embedder);
      html += kv('生成模型', health.models.llm);
      html += '</div><div class="card"><h4>运行指标</h4>';
      const total = (stats.redis || {}).total || {};
      Object.entries(total).forEach(([key, value]) => { html += kv(key, value); });
      html += kv('缓存命中', ((stats.redis || {}).cache || {}).hit || 0);
      html += kv('缓存未命中', ((stats.redis || {}).cache || {}).miss || 0);
      html += kv('在线用户', ((stats.redis || {}).online) || 0);
      html += kv('kb_version', ((stats.redis || {}).kb_version) || 0);
      html += '</div>';
      (stats.bm25 || []).forEach((item) => {
        html += '<div class="card"><h4>BM25 索引 · ' + item.scope + '</h4>'
          + kv('文档块', item.docs) + kv('对应 kb_version', item.version) + '</div>';
      });
      $('tab-state').innerHTML = html;
    } else if (name === 'memory') {
      const data = await api('/api/memory');
      let html = '<div class="card"><h4>用户画像</h4>';
      Object.entries(data.profile || {}).forEach(([key, value]) => { html += kv(key, value); });
      html += kv('Redis 键分组', JSON.stringify((data.storage || {}).groups || {}));
      html += '</div>';
      (data.roles || []).forEach((role) => {
        html += '<div class="card"><h4>' + escapeHtml(role.role_id) + '</h4>';
        if (!(role.milvus_memories || []).length && !(role.redis_facts || []).length) {
          html += '<div class="tiny">暂无记忆</div>';
        }
        (role.milvus_memories || []).forEach((item) => {
          html += '<div class="kv"><span>' + escapeHtml(item.kind) + '</span><span>'
            + escapeHtml(item.text) + '</span></div>';
        });
        html += '</div>';
      });
      $('tab-memory').innerHTML = html;
    } else if (name === 'logs') {
      const data = await api('/api/logs?limit=120');
      $('tab-logs').innerHTML = (data.logs || []).map((item) =>
        '<div class="log-line ' + item.level + '"><span class="lv">' + item.ts + ' ' + item.level
        + '</span> ' + escapeHtml(item.message) + '</div>').join('');
    }
  } catch (err) {
    $('tab-' + name).innerHTML = '<div class="tiny">加载失败：' + escapeHtml(err.message) + '</div>';
  }
}

$('btn-debug').addEventListener('click', async () => {
  const query = $('debug-query').value.trim();
  if (!query) return;
  $('debug-result').innerHTML = '<div class="tiny">检索中…</div>';
  try {
    const data = await api('/api/retrieval/compare', {
      method: 'POST',
      body: { query: query, role_id: state.roleId, top_k: 5 },
    });
    const report = data.report;
    let html = '<div class="card"><h4>查询稀疏词权重</h4>';
    html += (report.sparse_terms || []).map((item) => item[0] + ':' + item[1]).join('、');
    html += '</div>';
    Object.entries(report.modes).forEach(([mode, payload]) => {
      html += '<div class="card debug-col"><h4>' + mode + '</h4>';
      html += '<div class="tiny">候选 ' + JSON.stringify(payload.candidates) + '</div>';
      (payload.results || []).forEach((hit, index) => {
        const routes = Object.entries(hit.routes || {})
          .map(([name, value]) => name + '#' + value.rank + '(' + value.raw.toFixed(3) + ')').join(' ');
        html += '<div class="hit">' + (index + 1) + '. <b>' + hit.score.toFixed(5) + '</b> 《'
          + escapeHtml(hit.doc_title) + '》<br><span class="tiny">' + escapeHtml(routes)
          + '</span><br>' + escapeHtml(hit.preview || '') + '…</div>';
      });
      html += '</div>';
    });
    $('debug-result').innerHTML = html;
  } catch (err) {
    $('debug-result').innerHTML = '<div class="tiny">失败：' + escapeHtml(err.message) + '</div>';
  }
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
