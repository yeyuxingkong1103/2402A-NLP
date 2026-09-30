/* ============================================================
   static/js/roleplay.js —— 「角色扮演」页签

   在链路中的位置：
       浏览器 → 本文件 → /api/roleplay/roles、/api/roleplay/sessions、/api/roleplay/chat

   状态说明：
       消息列表只在前端内存里维护，刷新页面即清空；
       服务端另有落库（角色/会话/消息都在 SQLite），只是这个页面没做"加载历史会话"的入口。
       页面底部的元信息会把"这次回答是怎么来的"摊开：
       模型来源（ollama/fallback）、短期记忆后端、长期记忆命中数、引用条数 —— 降级时一眼可见。
   ============================================================ */

/* ================= 角色扮演 ================= */
// 当前角色扮演会话的 id（后端新建会话后回填）与本地消息列表。
// 消息只在前端内存里维护，刷新页面即清空；服务端另有落库（会话与消息都在 SQLite），
// 只是这个页面没有做"加载历史会话"的入口
let rpSessionId = null;
let rpMessages = [];
// 拉取角色列表（GET /api/roleplay/roles）填充下拉框，选项文本带上角色描述便于辨认。
// 失败时给出一个可用的兜底选项（friend）而不是让下拉框空白 ——
// 角色加载失败不该导致整个聊天功能不可用
async function loadRoleplayRoles() {
  try {
    const j = await fetchJson('/api/roleplay/roles');
    const roles = j.roles || [];
    $('rpRole').innerHTML = roles.map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.name)} · ${escapeHtml(r.description || '')}</option>`).join('');
  } catch (err) {
    $('rpRole').innerHTML = '<option value="friend">角色加载失败</option>';
    $('rpMeta').textContent = err.message;
  }
}
// 重绘整个角色对话区：按 speaker 区分左右样式（我 / 角色）。
// 每轮全量重绘而不是追加节点 —— 消息量小，全量重绘代码更简单、也不会出现重复渲染的 bug。
// 渲染完把滚动条置到底部（scrollTop = scrollHeight），让最新一条可见
function renderRoleplayMessages() {
  const box = $('rpMessages');
  box.innerHTML = rpMessages.length
    ? rpMessages.map(m => `<div class="rp-msg ${m.speaker === 'user' ? 'user' : 'assistant'}"><span class="who">${m.speaker === 'user' ? '我' : '角色'}</span>${escapeHtml(m.content)}</div>`).join('')
    : '<div class="empty">选择角色并开始聊天</div>';
  box.scrollTop = box.scrollHeight;
}
// "新建会话"：向后端建一个会话并清空本地消息。
// 用户标识默认取 anonymous（页面上的输入框留空时），方便不登录直接演示。
// 注意后端新建会话后返回的 id 存在 j.id（不是 session_id）—— 与发送消息接口的返回字段不同名
$('rpNew').onclick = async () => {
  try {
    const j = await fetchJson('/api/roleplay/sessions', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ user_id: $('rpUser').value.trim() || 'anonymous', role_id: $('rpRole').value })
    });
    rpSessionId = j.id;
    rpMessages = [];
    renderRoleplayMessages();
    $('rpMeta').textContent = '已新建会话：' + rpSessionId;
  } catch (err) { $('rpMeta').textContent = err.message; }
};
// 换角色即作废当前会话并清空对话：一个会话绑定一个角色，跨角色继续聊会让角色人格错乱
$('rpRole').onchange = () => { rpSessionId = null; rpMessages = []; renderRoleplayMessages(); };
// 发送一条消息（POST /api/roleplay/chat）。
// 先把用户消息推入本地列表并立刻重绘（乐观更新）—— 不用等后端返回，输入体验更跟手。
// 首次发送时 rpSessionId 为 null，后端会自动建会话并把 id 放在 session_id 里返回。
// 底部那行元信息把"这次回答是怎么来的"摊开：模型来源（ollama/fallback）、
// 短期记忆后端、长期记忆命中数、引用条数 —— 降级发生时能一眼看出来
$('rpSend').onclick = async () => {
  const message = $('rpInput').value.trim();
  if (!message) return;
  $('rpSend').disabled = true;
  rpMessages.push({speaker: 'user', content: message});
  renderRoleplayMessages();
  $('rpInput').value = '';
  try {
    const j = await fetchJson('/api/roleplay/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ user_id: $('rpUser').value.trim() || 'anonymous', role_id: $('rpRole').value, session_id: rpSessionId, message })
    });
    rpSessionId = j.session_id;
    rpMessages.push({speaker: 'assistant', content: j.answer || ''});
    renderRoleplayMessages();
    const mem = j.memory || {};
    $('rpMeta').textContent = `会话 ${rpSessionId} · ${j.model_source || 'unknown'} · 短期 ${mem.short_backend || '-'} · 长期命中 ${mem.long_term_hits ?? 0} · 引用 ${(j.citations || []).length} 条`;
  } catch (err) {
    rpMessages.push({speaker: 'assistant', content: '请求失败：' + err.message});
    renderRoleplayMessages();
  } finally { $('rpSend').disabled = false; }
};
$('rpInput').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('rpSend').click(); } });

