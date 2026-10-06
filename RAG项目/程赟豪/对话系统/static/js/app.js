/* ===== RAG 角色扮演系统 - 前端逻辑 ===== */
(function () {
  'use strict';

  const API = {
    chat: '/api/chat',
    roleList: '/api/role/list',
    roleCreate: '/api/role/create',
    session: (id) => `/api/session/${id}`,
    health: '/api/health',
  };

  // 状态
  const state = {
    sessionId: '',
    selectedRoleId: null,
    roles: [],
    availableTypes: [],
    sending: false,
  };

  // DOM 引用
  const $ = (id) => document.getElementById(id);
  const els = {
    messages: $('messages'),
    messageInput: $('messageInput'),
    sendBtn: $('sendBtn'),
    roleSelect: $('roleSelect'),
    roleInfo: $('roleInfo'),
    roleTypeSelect: $('roleTypeSelect'),
    roleCreateForm: $('roleCreateForm'),
    toggleRoleForm: $('toggleRoleForm'),
    newRoleId: $('newRoleId'),
    newRoleName: $('newRoleName'),
    newRolePrompt: $('newRolePrompt'),
    roleFormMsg: $('roleFormMsg'),
    sessionId: $('sessionId'),
    newSessionBtn: $('newSessionBtn'),
    clearSessionBtn: $('clearSessionBtn'),
    healthDot: $('healthDot'),
    healthText: $('healthText'),
    chatTitle: $('chatTitle'),
    chatSubtitle: $('chatSubtitle'),
  };

  /* ---------- 工具 ---------- */
  function generateSessionId() {
    return 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
  }

  async function request(url, options = {}) {
    const resp = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const body = await resp.json();
        detail = body.detail || detail;
      } catch (e) { /* ignore */ }
      throw new Error(detail);
    }
    return resp.json();
  }

  function escapeHtml(str) {
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function scrollToBottom() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  /* ---------- 消息渲染 ---------- */
  function addMessage(role, text, sources) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = role === 'user' ? '我' : '助手';

    wrap.appendChild(bubble);
    wrap.appendChild(meta);

    if (role === 'assistant' && sources && sources.length) {
      wrap.appendChild(renderSources(sources));
    }

    els.messages.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function renderSources(sources) {
    const box = document.createElement('div');
    box.className = 'sources';

    const toggle = document.createElement('button');
    toggle.className = 'sources-toggle';
    toggle.type = 'button';
    toggle.textContent = `📚 参考来源 (${sources.length})`;
    toggle.addEventListener('click', () => {
      list.classList.toggle('hidden');
      toggle.textContent = list.classList.contains('hidden')
        ? `📚 参考来源 (${sources.length})`
        : '📚 收起来源';
    });

    const list = document.createElement('div');
    list.className = 'sources-list hidden';
    sources.forEach((s, i) => {
      const item = document.createElement('div');
      item.className = 'source-item';
      const head = document.createElement('div');
      head.className = 'source-head';
      head.textContent = `来源 ${i + 1}：${s.source || '未知'}${s.page ? '（第 ' + s.page + ' 页）' : ''}`;
      const text = document.createElement('div');
      text.className = 'source-text';
      text.textContent = s.text || '';
      item.appendChild(head);
      item.appendChild(text);
      list.appendChild(item);
    });

    box.appendChild(toggle);
    box.appendChild(list);
    return box;
  }

  function addTypingIndicator() {
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    wrap.id = 'typingIndicator';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
    wrap.appendChild(bubble);
    els.messages.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function removeTypingIndicator() {
    const t = $('typingIndicator');
    if (t) t.remove();
  }

  function addError(text) {
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant error';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = '⚠️ ' + text;
    wrap.appendChild(bubble);
    els.messages.appendChild(wrap);
    scrollToBottom();
  }

  /* ---------- 健康检查 ---------- */
  async function checkHealth() {
    try {
      const r = await request(API.health);
      if (r.status === 'ok') {
        els.healthDot.className = 'health-dot online';
        els.healthText.textContent = '服务在线';
      } else {
        throw new Error('bad status');
      }
    } catch (e) {
      els.healthDot.className = 'health-dot offline';
      els.healthText.textContent = '服务离线';
    }
  }

  /* ---------- 角色 ---------- */
  async function loadRoles() {
    try {
      const data = await request(API.roleList);
      state.roles = data.roles || [];
      state.availableTypes = data.available_types || [];

      // 填充角色选择下拉
      els.roleSelect.innerHTML = '<option value="">— 不使用角色（通用助手）—</option>';
      state.roles.forEach((r) => {
        const opt = document.createElement('option');
        opt.value = r.role_id;
        opt.textContent = `${r.name} (${r.role_id})`;
        els.roleSelect.appendChild(opt);
      });

      // 填充角色类型下拉
      els.roleTypeSelect.innerHTML = '';
      state.availableTypes.forEach((t) => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        els.roleTypeSelect.appendChild(opt);
      });

      // 恢复之前选中的角色
      const saved = localStorage.getItem('selected_role_id');
      if (saved && state.roles.some((r) => r.role_id === saved)) {
        els.roleSelect.value = saved;
      }
      updateRoleInfo();
    } catch (e) {
      els.roleSelect.innerHTML = '<option value="">— 角色加载失败 —</option>';
    }
  }

  function updateRoleInfo() {
    const roleId = els.roleSelect.value;
    state.selectedRoleId = roleId || null;
    localStorage.setItem('selected_role_id', roleId || '');

    if (!roleId) {
      els.roleInfo.classList.add('hidden');
      els.chatTitle.textContent = '通用助手';
      els.chatSubtitle.textContent = '你可以向我提问，我会基于知识库检索回答并标注来源';
      return;
    }

    const role = state.roles.find((r) => r.role_id === roleId);
    if (role) {
      els.roleInfo.classList.remove('hidden');
      els.roleInfo.innerHTML =
        `<div class="role-name">${escapeHtml(role.name)}</div>` +
        `<div class="role-desc">${escapeHtml(role.description || '')}</div>`;
      els.chatTitle.textContent = role.name;
      els.chatSubtitle.textContent = role.description || '已选择角色';
    }
  }

  async function createRole(e) {
    e.preventDefault();
    const roleType = els.roleTypeSelect.value;
    const roleId = els.newRoleId.value.trim();
    const name = els.newRoleName.value.trim();
    const customPrompt = els.newRolePrompt.value.trim();

    if (!roleType) return showRoleMsg('请选择角色类型', false);
    if (!roleId) return showRoleMsg('请填写角色 ID', false);

    try {
      await request(API.roleCreate, {
        method: 'POST',
        body: JSON.stringify({
          role_type: roleType,
          role_id: roleId,
          name: name || null,
          custom_prompt: customPrompt || null,
        }),
      });
      showRoleMsg('✅ 角色创建成功', true);
      els.roleCreateForm.reset();
      els.roleCreateForm.classList.add('hidden');
      await loadRoles();
      els.roleSelect.value = roleId;
      updateRoleInfo();
    } catch (err) {
      showRoleMsg('创建失败：' + err.message, false);
    }
  }

  function showRoleMsg(text, ok) {
    els.roleFormMsg.textContent = text;
    els.roleFormMsg.className = 'form-msg ' + (ok ? 'ok' : 'err');
    els.roleFormMsg.classList.remove('hidden');
  }

  /* ---------- 会话 ---------- */
  function newSession() {
    state.sessionId = generateSessionId();
    localStorage.setItem('session_id', state.sessionId);
    els.sessionId.textContent = state.sessionId;
    els.messages.innerHTML =
      '<div class="welcome"><div class="welcome-icon">💬</div>' +
      '<h2>已开启新会话</h2><p>输入你的问题开始对话。</p></div>';
  }

  async function clearSession() {
    try {
      await request(API.session(state.sessionId), { method: 'DELETE' });
      els.messages.innerHTML =
        '<div class="welcome"><div class="welcome-icon">🧹</div>' +
        '<h2>会话记忆已清空</h2><p>短期记忆已清除，可以继续新对话。</p></div>';
    } catch (err) {
      addError('清空会话失败：' + err.message);
    }
  }

  /* ---------- 聊天 ---------- */
  async function sendMessage() {
    if (state.sending) return;
    const text = els.messageInput.value.trim();
    if (!text) return;

    state.sending = true;
    els.sendBtn.disabled = true;
    els.messageInput.value = '';
    autoResizeInput();

    addMessage('user', text);
    const typing = addTypingIndicator();

    try {
      const data = await request(API.chat, {
        method: 'POST',
        body: JSON.stringify({
          message: text,
          session_id: state.sessionId,
          role_id: state.selectedRoleId || null,
          use_history: true,
        }),
      });

      removeTypingIndicator();
      addMessage('assistant', data.message || '(空回复)', data.sources || []);
    } catch (err) {
      removeTypingIndicator();
      addError(err.message);
    } finally {
      state.sending = false;
      els.sendBtn.disabled = false;
      els.messageInput.focus();
    }
  }

  function autoResizeInput() {
    els.messageInput.style.height = 'auto';
    els.messageInput.style.height = Math.min(els.messageInput.scrollHeight, 140) + 'px';
  }

  function onInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  /* ---------- 初始化 ---------- */
  function init() {
    // 会话
    state.sessionId = localStorage.getItem('session_id') || generateSessionId();
    localStorage.setItem('session_id', state.sessionId);
    els.sessionId.textContent = state.sessionId;
    els.sessionId.title = '点击复制 Session ID';
    els.sessionId.addEventListener('click', () => {
      navigator.clipboard.writeText(state.sessionId).catch(() => {});
    });

    // 事件绑定
    els.sendBtn.addEventListener('click', sendMessage);
    els.messageInput.addEventListener('keydown', onInputKeydown);
    els.messageInput.addEventListener('input', autoResizeInput);
    els.roleSelect.addEventListener('change', updateRoleInfo);
    els.toggleRoleForm.addEventListener('click', () => {
      els.roleCreateForm.classList.toggle('hidden');
    });
    els.roleCreateForm.addEventListener('submit', createRole);
    els.newSessionBtn.addEventListener('click', newSession);
    els.clearSessionBtn.addEventListener('click', clearSession);

    // 加载数据
    loadRoles();
    checkHealth();
    setInterval(checkHealth, 30000);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
