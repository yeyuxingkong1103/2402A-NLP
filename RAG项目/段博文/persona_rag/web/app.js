// ============================================================================
// PersonaRAG · 多角色智能顾问系统 · 前端逻辑（原生 JS，无需构建）
// 视图：对话（流式）/ 检索台（纯检索）/ 知识库管理（入库·来源·删除·重建）
// ============================================================================

// API 地址：由后端托管时用同源；直接双击 html 打开时回退到 localhost:8000
const API = location.protocol.startsWith("http") ? location.origin : "http://localhost:8000";

const $ = id => document.getElementById(id);
const LS = { userId: "rag.userId", topK: "rag.topK", roleId: "rag.roleId", view: "rag.view" };

// 角色 key → 头像字与配色（绿色系深浅区分：深绿 / 墨绿 / 橄榄绿）
const ROLE_STYLE = {
  lawyer: { letter: "法", bg: "#14532d", fg: "#86efac" },
  psychologist: { letter: "心", bg: "#065f46", fg: "#6ee7b7" },
  virtual_friend: { letter: "伴", bg: "#3f6212", fg: "#bef264" },
};
const USER_STYLE = { letter: "我", bg: "#1f2a22", fg: "#a7f3c8" };  // 用户头像（中性绿灰）
const ROLE_SUBTITLE = {
  lawyer: "法律顾问在线，可咨询法条、案例分析",
  psychologist: "心理专家在线，提供情绪支持与专业建议",
  virtual_friend: "虚拟朋友在线，随时陪你聊聊天",
};

// 全局状态
const state = {
  view: "chat",
  roles: [],
  role: null,      // 当前角色对象（/roles 返回的原始结构）
  userId: 1,
  topK: 5,
  sending: false,
  abort: null,     // 流式请求的 AbortController
  files: [],       // 待上传文件
  openSource: null // 当前展开的来源名
};

// ============================================================================
// 一、基础工具
// ============================================================================

/** 统一请求封装：非 2xx 抛出带后端 detail 的错误 */
async function api(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch { /* 非 JSON 响应忽略 */ }
    throw new Error(detail);
  }
  return res.json();
}

const postJSON = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/** HTML 转义：所有外部文本先转义再拼接，避免 XSS */
function escapeHtml(t) {
  return String(t ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** 行内标记：`代码`、**加粗**、【1】引用角标（已转义过，注入标签安全） */
function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/【(\d+)】/g, "<strong>【$1】</strong>");
}

/** 轻量 Markdown 渲染：标题、无序列表、空行分段（够角色回答用，不引第三方库） */
function renderRich(text) {
  const out = [];
  let inList = false;
  const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };

  for (const raw of escapeHtml(text).split("\n")) {
    const line = raw.trimEnd();
    const h = line.match(/^#{1,6}\s+(.+)$/);
    const li = line.match(/^[-*]\s+(.+)$/);
    if (h) { closeList(); out.push(`<h4>${inlineMd(h[1])}</h4>`); continue; }
    if (li) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inlineMd(li[1])}</li>`);
      continue;
    }
    closeList();
    out.push(line.trim() ? inlineMd(line) : "");
  }
  closeList();
  return out.join("\n");
}

/** 左下角轻提示 */
function toast(msg, type = "") {
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

const roleStyle = key => ROLE_STYLE[key] || { letter: "AI", bg: "#14532d", fg: "#86efac" };
const shortSize = n => n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(1) + " KB" : (n / 1048576).toFixed(1) + " MB";

// ============================================================================
// 二、初始化
// ============================================================================

async function init() {
  // 恢复本地设置
  state.userId = parseInt(localStorage.getItem(LS.userId) || "1");
  state.topK = parseInt(localStorage.getItem(LS.topK) || "5");
  state.view = localStorage.getItem(LS.view) || "chat";
  $("userId").value = state.userId;
  $("topK").value = state.topK;
  $("searchTopK").value = state.topK;

  bindEvents();
  switchView(state.view);
  checkHealth();
  setInterval(checkHealth, 30000);
  await loadRoles();
}

/** 健康检查 + 在线人数 */
async function checkHealth() {
  const dot = $("statusDot");
  try {
    const d = await api("/health");
    dot.className = "dot online";
    $("statusText").textContent = "服务在线";
    $("onlineText").textContent = d.online_users ? `· ${d.online_users} 人在线` : "";
  } catch {
    dot.className = "dot offline";
    $("statusText").textContent = "服务离线";
    $("onlineText").textContent = "";
  }
}

/** 拉取角色列表，渲染侧边栏并恢复上次选中的角色 */
async function loadRoles() {
  try {
    const d = await api("/roles");
    state.roles = d.roles || [];
    renderRoleList();

    const savedId = parseInt(localStorage.getItem(LS.roleId) || "0");
    const target = state.roles.find(r => r.id === savedId) || state.roles[0];
    if (target) selectRole(target);
  } catch (e) {
    $("roleList").innerHTML = `<div class="table-empty">角色加载失败：${escapeHtml(e.message)}</div>`;
  }
}

function renderRoleList() {
  const box = $("roleList");
  box.innerHTML = "";
  for (const r of state.roles) {
    const st = roleStyle(r.role_key);
    const card = document.createElement("button");
    card.className = "role-card" + (state.role && state.role.id === r.id ? " active" : "");
    card.dataset.id = r.id;
    card.innerHTML =
      `<span class="role-avatar" style="background:${st.bg};color:${st.fg}">${escapeHtml(st.letter)}</span>
       <span class="rc-text">
         <span class="rc-name">${escapeHtml(r.name)}</span>
         <span class="rc-desc">${escapeHtml(r.description || "")}</span>
       </span>`;
    card.addEventListener("click", () => selectRole(r));
    box.appendChild(card);
  }
}

/** 切换角色：同步头部、建议问题、检索台与知识库 */
function selectRole(role) {
  state.role = role;
  localStorage.setItem(LS.roleId, role.id);

  const st = roleStyle(role.role_key);
  document.querySelectorAll(".role-card").forEach(c =>
    c.classList.toggle("active", parseInt(c.dataset.id) === role.id));

  const avatar = $("chatAvatar");
  avatar.textContent = st.letter;
  avatar.style.background = st.bg;
  avatar.style.color = st.fg;
  avatar.classList.add("lg");
  $("chatRoleName").textContent = role.name;
  $("chatRoleDesc").textContent = role.description || "";
  $("emptyTitle").textContent = `和「${role.name}」聊聊`;
  $("emptySub").textContent = ROLE_SUBTITLE[role.role_key] || "提问后系统会先检索知识库再作答";

  renderSuggestions(role.suggestions || []);
  if (state.view === "kb") loadSources();
  if (state.view === "search") $("searchCollection").textContent = "集合：查询中…";
}

function renderSuggestions(list) {
  const box = $("suggestList");
  box.innerHTML = "";
  for (const s of list) {
    const card = document.createElement("button");
    card.className = "suggest-card";
    card.innerHTML = `<span class="sc-tag">${escapeHtml(s.tag || "提问")}</span>
                      <span class="sc-q">${escapeHtml(s.question || "")}</span>`;
    card.addEventListener("click", () => {
      $("input").value = s.question || "";
      autoGrow($("input"));
      onSend();
    });
    box.appendChild(card);
  }
}

// ============================================================================
// 三、事件绑定与视图切换
// ============================================================================

function bindEvents() {
  // 顶部导航
  document.querySelectorAll(".nav-item").forEach(btn =>
    btn.addEventListener("click", () => switchView(btn.dataset.view)));

  // 对话
  $("sendBtn").addEventListener("click", onSend);
  $("stopBtn").addEventListener("click", () => state.abort && state.abort.abort());
  $("clearBtn").addEventListener("click", onClearMemory);
  $("userId").addEventListener("change", () => {
    state.userId = Math.max(1, parseInt($("userId").value) || 1);
    $("userId").value = state.userId;
    localStorage.setItem(LS.userId, state.userId);
  });
  $("topK").addEventListener("change", () => {
    state.topK = Math.min(20, Math.max(1, parseInt($("topK").value) || 5));
    $("topK").value = state.topK;
    localStorage.setItem(LS.topK, state.topK);
  });

  const input = $("input");
  input.addEventListener("input", () => { autoGrow(input); refreshSendBtn(); });
  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
  });

  // 检索台
  $("searchBtn").addEventListener("click", onSearch);
  $("searchQuery").addEventListener("keydown", e => { if (e.key === "Enter") onSearch(); });

  // 知识库
  $("kbRefresh").addEventListener("click", loadSources);
  $("kbRebuild").addEventListener("click", onRebuild);
  $("uploadBtn").addEventListener("click", onUpload);
  $("textIngestBtn").addEventListener("click", onIngestText);
  bindDropZone();
}

function switchView(view) {
  state.view = view;
  localStorage.setItem(LS.view, view);
  document.querySelectorAll(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + view));
  if (view === "kb") loadSources();
  if (view === "chat") scrollToBottom(true);
}

const autoGrow = el => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 140) + "px"; };
const refreshSendBtn = () => { $("sendBtn").disabled = state.sending || !$("input").value.trim(); };
const scrollToBottom = (force = false) => {
  const box = $("chatScroll");
  if (force || box.scrollHeight - box.scrollTop - box.clientHeight < 120) box.scrollTop = box.scrollHeight;
};

// ============================================================================
// 四、对话（流式）
// ============================================================================

async function onSend() {
  const query = $("input").value.trim();
  if (!query || state.sending || !state.role) return;

  state.sending = true;
  $("input").value = "";
  autoGrow($("input"));
  refreshSendBtn();
  $("sendBtn").classList.add("hidden");
  $("stopBtn").classList.remove("hidden");
  $("composerHint").textContent = "检索知识库中…";
  $("chatEmpty").classList.add("hidden");

  addMessage("user", query);
  const bot = addMessage("bot", "");
  const contentEl = bot.querySelector(".msg-content");
  const cursor = document.createElement("span");
  cursor.className = "cursor-blink";
  contentEl.appendChild(cursor);

  const payload = {
    user_id: state.userId,
    role_id: state.role.id,
    query,
    top_k: state.topK,
    use_local_llm: false,
    role_key: state.role.role_key || ""
  };

  state.abort = new AbortController();
  let answer = "";
  let stopped = false;

  try {
    const res = await fetch(API + "/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.abort.signal
    });
    if (!res.ok) throw new Error("HTTP " + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      answer += decoder.decode(value, { stream: true });
      contentEl.innerHTML = renderRich(answer);
      contentEl.appendChild(cursor);
      $("composerHint").textContent = "";
      scrollToBottom();
    }
  } catch (e) {
    stopped = e.name === "AbortError";
    if (!stopped) {
      contentEl.textContent = "[请求失败] " + e.message;
      $("composerHint").textContent = "";
    }
  } finally {
    cursor.remove();
    state.sending = false;
    state.abort = null;
    $("stopBtn").classList.add("hidden");
    $("sendBtn").classList.remove("hidden");
    refreshSendBtn();
    if (stopped) $("composerHint").textContent = "已停止生成（该轮回答仍可能已写入记忆）";
  }

  // 引用资料：走 /cases 纯检索（不调大模型，也不会重复写记忆）
  if (!stopped && answer) await attachSources(bot, query);
  scrollToBottom();
}

/** 拉取本轮引用的资料并折叠展示 */
async function attachSources(botEl, query) {
  try {
    const d = await api("/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: state.topK, role_key: state.role.role_key || "" })
    });
    const docs = d.documents || [];
    if (!docs.length) {
      $("composerHint").textContent = "本次未召回参考资料（该角色知识库可能为空）";
      return;
    }

    const box = document.createElement("div");
    box.className = "msg-sources";
    box.innerHTML = `
      <div class="msg-sources-head">
        <span>引用资料 ${docs.length} 条</span>
        <span class="muted">点击展开</span>
      </div>
      <div class="src-list">
        ${docs.map(doc => `
          <div class="src-item">
            <div class="si-head">[${doc.index}] ${escapeHtml(doc.source || "未知来源")}
              ${doc.section ? "· " + escapeHtml(doc.section) : ""}</div>
            <div class="si-text">${escapeHtml(doc.content || "")}</div>
          </div>`).join("")}
      </div>`;
    box.querySelector(".msg-sources-head").addEventListener("click", () => box.classList.toggle("open"));
    botEl.querySelector(".msg-body").appendChild(box);
  } catch (e) {
    $("composerHint").textContent = "引用资料获取失败：" + e.message;
  }
}

/** 清空当前角色 + 用户的短期记忆 */
async function onClearMemory() {
  if (!state.role) return toast("请先选择角色", "err");
  if (!confirm(`清空「${state.role.name}」的对话记忆？清空后本轮上下文不再带入。`)) return;
  try {
    const d = await postJSON("/memory/clear", { user_id: state.userId, role_id: state.role.id });
    $("messages").innerHTML = "";
    $("chatEmpty").classList.remove("hidden");
    renderSuggestions(state.role.suggestions || []);
    toast(d.message || "已清空", "ok");
  } catch (e) {
    toast("清空失败：" + e.message, "err");
  }
}

/** 追加一条消息，返回消息元素 */
function addMessage(role, text) {
  const st = role === "user" ? USER_STYLE : roleStyle(state.role?.role_key);
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = `
    <div class="msg-avatar" style="background:${st.bg};color:${st.fg}">${escapeHtml(st.letter)}</div>
    <div class="msg-body"><div class="msg-content">${role === "user" ? escapeHtml(text) : renderRich(text)}</div></div>`;
  $("messages").appendChild(el);
  scrollToBottom();
  return el;
}

// ============================================================================
// 五、检索台
// ============================================================================

async function onSearch() {
  const query = $("searchQuery").value.trim();
  if (!query) return toast("请输入查询内容", "err");
  if (!state.role) return toast("请先在左侧选择角色", "err");

  const topK = Math.min(20, Math.max(1, parseInt($("searchTopK").value) || 5));
  $("searchBtn").disabled = true;
  $("searchResults").innerHTML = `<div class="hint">检索中…</div>`;
  try {
    const d = await postJSON("/cases", { query, top_k: topK, role_key: state.role.role_key || "" });
    $("searchCollection").textContent = "集合：" + (state.role.role_key || "-");
    $("searchHint").textContent = `命中 ${d.count} 条 · 角色「${state.role.name}」`;
    renderDocs(d.documents || []);
  } catch (e) {
    $("searchResults").innerHTML = `<div class="hint">检索失败：${escapeHtml(e.message)}</div>`;
  } finally {
    $("searchBtn").disabled = false;
  }
}

function renderDocs(docs) {
  const box = $("searchResults");
  if (!docs.length) {
    box.innerHTML = `<div class="hint">没有召回结果：知识库可能是空的，或分数都被阈值过滤了</div>`;
    return;
  }
  box.innerHTML = docs.map(d => `
    <div class="doc-card">
      <div class="dc-head">
        <span class="dc-index">[${d.index}]</span>
        <span>${escapeHtml(d.source || "未知来源")}</span>
        ${d.section ? `<span>· ${escapeHtml(d.section)}</span>` : ""}
      </div>
      <div class="dc-body">${escapeHtml(d.content || "")}</div>
    </div>`).join("");
}

// ============================================================================
// 六、知识库管理
// ============================================================================

async function loadSources() {
  if (!state.role) return;
  const table = $("sourceTable");
  table.innerHTML = `<div class="table-empty">加载中…</div>`;
  try {
    const d = await api(`/kb/sources?role_key=${encodeURIComponent(state.role.role_key || "")}`);
    $("kbMeta").textContent = `角色「${state.role.name}」· 集合 ${d.collection} · 共 ${d.total_sources} 个来源`;
    $("kbTotal").textContent = d.total_sources;
    renderSources(d.sources || []);
  } catch (e) {
    table.innerHTML = `<div class="table-empty">加载失败：${escapeHtml(e.message)}</div>`;
  }
}

function renderSources(sources) {
  const table = $("sourceTable");
  if (!sources.length) {
    table.innerHTML = `<div class="table-empty">知识库还是空的，用上面的「上传文件」或「粘贴文本」入库</div>`;
    return;
  }

  table.innerHTML = `
    <div class="tr head"><span>来源</span><span>切块数</span><span style="text-align:right">操作</span></div>
    ${sources.map(s => `
      <div class="tr" data-src="${escapeHtml(s.source)}">
        <span class="src-name" title="${escapeHtml(s.source)}">${escapeHtml(s.source)}</span>
        <span class="src-chunks">${s.chunks} 条</span>
        <span class="src-ops">
          <button class="btn sm ghost act-view">查看</button>
          <button class="btn sm danger act-del">删除</button>
        </span>
      </div>`).join("")}`;

  table.querySelectorAll(".tr[data-src]").forEach(row => {
    const src = row.dataset.src;
    row.querySelector(".act-view").addEventListener("click", () => toggleSourceDetail(row, src));
    row.querySelector(".act-del").addEventListener("click", () => onDeleteSource(src));
  });
}

/** 展开/收起某个来源的 chunk 预览 */
async function toggleSourceDetail(row, source) {
  const next = row.nextElementSibling;
  if (next && next.classList.contains("chunk-preview")) {   // 已展开 → 收起
    next.remove();
    state.openSource = null;
    return;
  }
  const box = document.createElement("div");
  box.className = "chunk-preview";
  box.textContent = "加载中…";
  row.after(box);
  state.openSource = source;

  try {
    const d = await api(`/kb/sources/${encodeURIComponent(source)}?role_key=${encodeURIComponent(state.role.role_key || "")}`);
    const chunks = d.chunks || [];
    box.innerHTML = chunks.length
      ? chunks.map(c => `<div class="cp-item"><b>#${c.chunk_index ?? "-"}</b> ${escapeHtml(c.text || "")}</div>`).join("")
      : "该来源没有 chunk";
  } catch (e) {
    box.textContent = "加载失败：" + e.message;
  }
}

async function onDeleteSource(source) {
  if (!confirm(`删除来源「${source}」的全部切块？此操作不可撤销。`)) return;
  try {
    const d = await api(`/kb/sources/${encodeURIComponent(source)}?role_key=${encodeURIComponent(state.role.role_key || "")}`,
      { method: "DELETE" });
    toast(d.message || "已删除", "ok");
    loadSources();
  } catch (e) {
    toast("删除失败：" + e.message, "err");
  }
}

/** 全量重建：按现有文本重新切分写回（不改内容，只重切块与索引） */
async function onRebuild() {
  if (!confirm(`将删除「${state.role.name}」的集合并按现有文本重新切分写入，过程较慢，确认继续？`)) return;
  const btn = $("kbRebuild");
  btn.disabled = true;
  btn.textContent = "重建中…";
  try {
    const d = await postJSON(`/kb/rebuild?role_key=${encodeURIComponent(state.role.role_key || "")}`, { confirm: true });
    toast(`重建完成：写入 ${d.inserted ?? d.chunks ?? 0} 条`, "ok");
    loadSources();
  } catch (e) {
    toast("重建失败：" + e.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "全量重建";
  }
}

// 文件选择 / 拖拽
function bindDropZone() {
  const zone = $("dropZone");
  const input = $("fileInput");
  input.addEventListener("change", () => addFiles([...input.files]));
  ["dragenter", "dragover"].forEach(ev => zone.addEventListener(ev, e => {
    e.preventDefault();
    zone.classList.add("over");
  }));
  ["dragleave", "drop"].forEach(ev => zone.addEventListener(ev, e => {
    e.preventDefault();
    zone.classList.remove("over");
  }));
  zone.addEventListener("drop", e => addFiles([...e.dataTransfer.files]));
}

function addFiles(list) {
  const allow = /\.(txt|md|docx|wps|doc|pdf)$/i;
  for (const f of list) {
    if (!allow.test(f.name)) { toast(`跳过不支持的文件：${f.name}`, "err"); continue; }
    state.files.push(f);
  }
  renderFileList();
}

function renderFileList() {
  $("fileList").innerHTML = state.files.map((f, i) => `
    <div class="file-item">
      <span>${escapeHtml(f.name)}</span>
      <span class="fi-size">${shortSize(f.size)}
        <button class="btn sm ghost" data-i="${i}">移除</button></span>
    </div>`).join("");
  $("fileList").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    state.files.splice(parseInt(b.dataset.i), 1);
    renderFileList();
  }));
  $("uploadBtn").disabled = state.files.length === 0;
  $("uploadBtn").textContent = state.files.length ? `开始入库（${state.files.length} 个文件）` : "开始入库";
}

async function onUpload() {
  if (!state.files.length || !state.role) return;
  const fd = new FormData();
  state.files.forEach(f => fd.append("files", f));

  const btn = $("uploadBtn");
  btn.disabled = true;
  btn.textContent = "解析并入库中…";
  try {
    const d = await api(`/ingest/files?role_key=${encodeURIComponent(state.role.role_key || "")}`,
      { method: "POST", body: fd });
    toast(`入库成功：${d.files.length} 个文件 / ${d.chunks} 条切块，集合共 ${d.corpus_total} 条`, "ok");
    state.files = [];
    renderFileList();
    loadSources();
  } catch (e) {
    toast("入库失败：" + e.message, "err");
  } finally {
    btn.disabled = state.files.length === 0;
    btn.textContent = "开始入库";
  }
}

async function onIngestText() {
  const source = $("textSource").value.trim() || "手动输入";
  const text = $("textBody").value.trim();
  const strategy = $("textStrategy").value;
  if (!text) return toast("粘贴内容为空", "err");
  if (!state.role) return toast("请先选择角色", "err");
  if (text.length < 50 && !confirm("内容较短，可能切不出有效切块，仍要入库？")) return;

  const btn = $("textIngestBtn");
  btn.disabled = true;
  btn.textContent = "入库中…";
  try {
    const d = await postJSON("/ingest/text", {
      source, text, strategy, role_key: state.role.role_key || ""
    });
    toast(`入库成功：${d.chunks} 条切块，集合共 ${d.corpus_total} 条`, "ok");
    $("textBody").value = "";
    loadSources();
  } catch (e) {
    toast("入库失败：" + e.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "入库";
  }
}

// ============================================================================
init();