const API_BASE = "http://127.0.0.1:8000/api";

const state = {
  token: localStorage.getItem("legal_rag_token") || "",
  username: localStorage.getItem("legal_rag_username") || "",
  currentKbId: null,
  currentSessionId: null,
};

const el = (id) => document.getElementById(id);

function authHeaders() {
  return state.token ? { Authorization: `Bearer ${state.token}` } : {};
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...authHeaders(),
    },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "请求失败" }));
    throw new Error(error.detail || "请求失败");
  }
  return response.json();
}

function updateView() {
  const loggedIn = Boolean(state.token);
  el("loginPage").classList.toggle("hidden", loggedIn);
  el("chatPage").classList.toggle("hidden", !loggedIn);
  el("userInfo").textContent = loggedIn ? `已登录：${state.username}` : "未登录";
  el("loginStatus").textContent = loggedIn ? `已登录：${state.username}` : "未登录";
}

function setRetrievalStatus(text, status = "idle") {
  const statusBox = el("retrievalStatus");
  statusBox.textContent = text;
  statusBox.className = `retrieval-status muted ${status}`;
}

function formatDuration(startTime) {
  return ((performance.now() - startTime) / 1000).toFixed(2);
}

function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.*)$/gm, "<h2>$1</h2>");
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/^- (.*)$/gm, "<div class=\"md-list-item\">• $1</div>");
  html = html.replace(/^(\d+)\. (.*)$/gm, "<div class=\"md-list-item\">$1. $2</div>");
  return html.replace(/\n/g, "<br />");
}

function appendMessage(role, content) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  if (role === "assistant") {
    message.innerHTML = renderMarkdown(content);
  } else {
    message.textContent = content;
  }
  el("chatMessages").appendChild(message);
  el("chatMessages").scrollTop = el("chatMessages").scrollHeight;
}

function appendAssistantAnswer(content, references, durationSeconds) {
  const message = document.createElement("div");
  message.className = "message assistant";
  const referenceCount = references ? references.length : 0;
  message.innerHTML = `
    <div class="answer-meta">检索完成，用时 ${durationSeconds} 秒，参考 ${referenceCount} 条法律依据</div>
    <div class="answer-body">${renderMarkdown(content)}</div>
  `;
  el("chatMessages").appendChild(message);
  el("chatMessages").scrollTop = el("chatMessages").scrollHeight;
}

function renderReferences(references) {
  const box = el("referenceList");
  box.innerHTML = "";
  if (!references || references.length === 0) {
    box.innerHTML = "<p class='muted'>暂无引用依据</p>";
    return;
  }
  references.forEach((ref) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `<strong>【依据${ref.index}】${ref.filename}</strong><br />页码：${ref.page_number || "未知"}<br />标题：${ref.title_path || "无"}<br /><br />${escapeHtml(ref.content)}`;
    box.appendChild(item);
  });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
}

async function loginOrRegister(path) {
  const username = el("usernameInput").value.trim();
  const password = el("passwordInput").value.trim();
  if (!username || !password) {
    alert("请输入用户名和密码");
    return;
  }
  const data = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  state.token = data.access_token;
  state.username = data.username;
  localStorage.setItem("legal_rag_token", state.token);
  localStorage.setItem("legal_rag_username", state.username);
  updateView();
  await loadKnowledgeBases();
}

function logout() {
  state.token = "";
  state.username = "";
  state.currentKbId = null;
  state.currentSessionId = null;
  localStorage.removeItem("legal_rag_token");
  localStorage.removeItem("legal_rag_username");
  el("chatMessages").innerHTML = "";
  el("sessionList").innerHTML = "";
  el("taskList").innerHTML = "";
  renderReferences([]);
  updateView();
}

function startNewSession() {
  state.currentSessionId = null;
  el("chatMessages").innerHTML = "";
  renderReferences([]);
  setRetrievalStatus("等待提问");
  el("questionInput").focus();
  loadSessions().catch(() => undefined);
}

async function loadKnowledgeBases() {
  if (!state.token) return;
  const list = await request("/knowledge-bases");
  const select = el("kbSelect");
  select.innerHTML = "";
  list.forEach((kb) => {
    const option = document.createElement("option");
    option.value = kb.id;
    option.textContent = kb.name;
    select.appendChild(option);
  });
  if (list.length > 0) {
    state.currentKbId = Number(select.value || list[0].id);
    await Promise.all([loadDocumentsTasks(), loadSessions()]);
  } else {
    state.currentKbId = null;
    el("sessionList").innerHTML = "<p class='muted'>请先新建知识库</p>";
    el("taskList").innerHTML = "";
  }
}

async function createKnowledgeBase() {
  const name = el("kbNameInput").value.trim();
  const description = el("kbDescInput").value.trim();
  if (!name) {
    alert("请输入知识库名称");
    return;
  }
  await request("/knowledge-bases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
  el("kbNameInput").value = "";
  el("kbDescInput").value = "";
  await loadKnowledgeBases();
}

async function uploadDocument() {
  if (!state.currentKbId) {
    alert("请先创建或选择知识库");
    return;
  }
  const file = el("fileInput").files[0];
  if (!file) {
    alert("请选择 PDF 文件");
    return;
  }
  const form = new FormData();
  form.append("knowledge_base_id", state.currentKbId);
  form.append("file", file);
  await request("/documents/upload", { method: "POST", body: form });
  alert("上传成功，后台正在处理");
  el("fileInput").value = "";
  await loadDocumentsTasks();
}

async function loadDocumentsTasks() {
  if (!state.currentKbId) return;
  const tasks = await request(`/documents/tasks?knowledge_base_id=${state.currentKbId}`);
  const box = el("taskList");
  box.innerHTML = "";
  if (tasks.length === 0) {
    box.innerHTML = "暂无任务";
    return;
  }
  tasks.slice(0, 4).forEach((task) => {
    const item = document.createElement("div");
    const statusClass = task.status === "done" ? "status-done" : task.status === "failed" ? "status-failed" : "";
    item.className = "task-item";
    item.innerHTML = `#${task.id} <span class="${statusClass}">${task.status}</span> ${task.current_step || ""}`;
    box.appendChild(item);
  });
}

async function loadSessions() {
  if (!state.currentKbId) return;
  const sessions = await request(`/chat/sessions?knowledge_base_id=${state.currentKbId}`);
  const box = el("sessionList");
  box.innerHTML = "";
  if (sessions.length === 0) {
    box.innerHTML = "<p class='muted'>暂无历史会话</p>";
    return;
  }
  sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = `list-item clickable session-item${session.id === state.currentSessionId ? " active" : ""}`;
    item.innerHTML = `<div class="session-item-title">${escapeHtml(session.title)}</div><div class="session-item-meta">会话 #${session.id}</div>`;
    item.onclick = () => loadSessionMessages(session.id);
    box.appendChild(item);
  });
}

async function loadSessionMessages(sessionId) {
  state.currentSessionId = sessionId;
  const messages = await request(`/chat/sessions/${sessionId}/messages`);
  el("chatMessages").innerHTML = "";
  renderReferences([]);
  messages.forEach((message) => {
    appendMessage(message.role, message.content);
    if (message.references && message.references.length > 0) {
      renderReferences(message.references);
    }
  });
}

async function askQuestion() {
  if (!state.currentKbId) {
    alert("请先选择知识库");
    return;
  }
  const question = el("questionInput").value.trim();
  if (!question) {
    alert("请输入问题");
    return;
  }
  const startTime = performance.now();
  setRetrievalStatus("正在检索，请稍候...", "searching");
  el("askBtn").disabled = true;
  appendMessage("user", question);
  el("questionInput").value = "";
  try {
    const data = await request("/chat/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ knowledge_base_id: state.currentKbId, question, session_id: state.currentSessionId }),
    });
    state.currentSessionId = data.session_id;
    const durationSeconds = formatDuration(startTime);
    appendAssistantAnswer(data.answer, data.references, durationSeconds);
    renderReferences(data.references);
    setRetrievalStatus(`检索完成，用时 ${durationSeconds} 秒`, "done");
    await loadSessions();
  } catch (error) {
    setRetrievalStatus(`检索失败，用时 ${formatDuration(startTime)} 秒`, "error");
    throw error;
  } finally {
    el("askBtn").disabled = false;
  }
}

el("loginBtn").onclick = () => loginOrRegister("/auth/login").catch((error) => alert(error.message));
el("registerBtn").onclick = () => loginOrRegister("/auth/register").catch((error) => alert(error.message));
el("logoutBtn").onclick = logout;
el("newSessionBtn").onclick = startNewSession;
el("createKbBtn").onclick = () => createKnowledgeBase().catch((error) => alert(error.message));
el("uploadBtn").onclick = () => uploadDocument().catch((error) => alert(error.message));
el("refreshTasksBtn").onclick = () => loadDocumentsTasks().catch((error) => alert(error.message));
el("refreshSessionsBtn").onclick = () => loadSessions().catch((error) => alert(error.message));
el("askBtn").onclick = () => askQuestion().catch((error) => alert(error.message));
el("questionInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion().catch((error) => alert(error.message));
  }
});
document.querySelectorAll(".follow-up-btn").forEach((button) => {
  button.addEventListener("click", () => {
    el("questionInput").value = button.dataset.question || "";
    el("questionInput").focus();
  });
});
el("kbSelect").onchange = async () => {
  state.currentKbId = Number(el("kbSelect").value);
  state.currentSessionId = null;
  el("chatMessages").innerHTML = "";
  renderReferences([]);
  await Promise.all([loadDocumentsTasks(), loadSessions()]);
};

updateView();
renderReferences([]);
loadKnowledgeBases().catch(() => logout());
setInterval(() => loadDocumentsTasks().catch(() => undefined), 5000);
