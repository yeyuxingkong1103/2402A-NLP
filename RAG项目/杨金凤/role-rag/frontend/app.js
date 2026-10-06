const chatEl = document.getElementById('chat');
const roleEl = document.getElementById('role');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const clearBtn = document.getElementById('clear');
const segEl = document.getElementById('seg');
const welcomeEl = document.getElementById('welcome');
const SESSION_KEY = 'web_session_id';
let sending = false;

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
function newSessionId() { return 'web_' + Math.random().toString(36).slice(2, 10); }
function getSessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) { id = newSessionId(); localStorage.setItem(SESSION_KEY, id); }
  return id;
}
let sessionId = getSessionId();

function clearSession() {
  sessionId = newSessionId();
  localStorage.setItem(SESSION_KEY, sessionId); // 覆盖值，不删 key，刷新后沿用新会话
  chatEl.querySelectorAll('.msg').forEach(el => el.remove());
  updateWelcome();
}
function scrollBottom() { chatEl.scrollTop = chatEl.scrollHeight; }
function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}
function updateWelcome() {
  welcomeEl.style.display = chatEl.querySelector('.msg') ? 'none' : '';
}
function fixMarkdownTables(text) {
  let t = text.replace(/([：。！？；])\s*\|/g, '$1\n\n|');
  // 表格内行拆分：紧贴的 || → |\n|
  // 只在含 |---| 分隔行时执行，降低误伤正文的风险
  if (/\|\s*-{2,}\s*\|/.test(t)) {
    t = t.replace(/\|\|/g, '|\n|');
  }
  return t;
}
function simClass(v) {
  return v >= 0.8 ? 'high' : (v >= 0.5 ? 'mid' : 'low');
}

function addUser(msg) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = '<div class="avatar">我</div><div class="bubble">' + esc(msg) + '</div>';
  chatEl.appendChild(div);
  scrollBottom();
  updateWelcome();
}
function addAssistant() {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="avatar">医</div>' +
    '<div class="content"><div class="bubble"><span class="text"></span>' +
    '<span class="typing"><i></i><i></i><i></i></span></div>' +
    '<details class="sources"><summary>📚 来源</summary></details></div>';
  chatEl.appendChild(div);
  const a = {
    bubble: div.querySelector('.bubble'),
    textEl: div.querySelector('.bubble .text'),
    typing: div.querySelector('.bubble .typing'),
    sources: div.querySelector('.sources'),
    summary: div.querySelector('.sources summary'),
    raw: '',
    append(t) {
      if (this.typing) { this.typing.remove(); this.typing = null; }
      this.raw += t;
      this.textEl.textContent = this.raw;
      scrollBottom();
    },
    error(msg) {
      if (this.typing) { this.typing.remove(); this.typing = null; }
      this.raw += '\n⚠️ ' + msg;
      this.textEl.textContent = this.raw;
      scrollBottom();
    },
    done() {
      const normalized = fixMarkdownTables(this.raw);
      const html = (window.marked && window.DOMPurify)
        ? window.DOMPurify.sanitize(window.marked.parse(normalized))
        : esc(this.raw);
      this.bubble.innerHTML = html;
      scrollBottom();
    }
  };
  return a;
}
function renderSources(a, list) {
  if (!Array.isArray(list) || list.length === 0) return;
  a.sources.style.display = '';
  a.summary.textContent = '📚 引用来源（' + list.length + ' 条）';
  list.forEach(s => {
    const sim = Number(s.similarity) || 0;
    const item = document.createElement('div');
    item.className = 'source-item';
    item.innerHTML = '<div class="source-head">' +
      '<span class="page-badge">第 ' + s.page + ' 页</span>' +
      '<span class="sim ' + simClass(sim) + '">相似度 ' + sim.toFixed(3) + '</span>' +
      '</div>' +
      '<div class="source-text">' + esc(String(s.content || '').slice(0, 120)) + '</div>';
    a.sources.appendChild(item);
  });
}
function handleFrame(frame, a) {
  const lines = frame.split('\n');
  let event = 'message';
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  const data = dataLines.join('\n');
  if (event === 'sources') {
    try { renderSources(a, JSON.parse(data)); } catch (e) { /* 忽略解析失败 */ }
  } else if (event === 'error') {
    a.error('生成中断');
  } else if (data.trim() === '[DONE]') {
    a.done();
  } else {
    a.append(data);
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || sending) return;
  inputEl.value = '';
  autoResize();
  addUser(text);
  const a = addAssistant();
  sending = true;
  sendBtn.disabled = true;
  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId, role: roleEl.value })
    });
    if (!resp.ok) {
      let detail = '请求失败（' + resp.status + '）';
      try { detail = (await resp.json()).detail || detail; } catch (e) { /* 忽略 */ }
      a.error(detail);
      a.done();
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleFrame(frame, a);
      }
    }
    a.done();
  } catch (e) {
    a.error('网络错误');
    a.done();
  } finally {
    sending = false;
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', send);
clearBtn.addEventListener('click', clearSession);
inputEl.addEventListener('input', autoResize);
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

// 角色胶囊按钮（隐藏 select 保底：只改 roleEl.value，send() 逻辑不动）
segEl.querySelectorAll('button').forEach(btn => {
  btn.addEventListener('click', () => {
    roleEl.value = btn.dataset.role;
    segEl.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === btn));
  });
});

// 快捷问题卡片：点击即发送
welcomeEl.querySelectorAll('.quick button').forEach(btn => {
  btn.addEventListener('click', () => {
    inputEl.value = btn.dataset.q;
    send();
  });
});
