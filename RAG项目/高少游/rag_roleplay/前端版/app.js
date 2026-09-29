// API 地址：写死绝对路径，双击打开或通过后端访问都能连上
const API = "http://localhost:8000";

const $ = id => document.getElementById(id);
const roleSelect = $("roleSelect");
const userIdEl = $("userId");
const topKEl = $("topK");
const messagesEl = $("messages");
const chatContainer = $("chatContainer");
const userInput = $("userInput");
const sendBtn = $("sendBtn");
const clearBtn = $("clearBtn");
const statusDot = $("statusDot");
const statusText = $("statusText");
const welcomeScreen = $("welcomeScreen");
const suggestionCardsEl = $("suggestionCards");
const welcomeSubtitle = $("welcomeSubtitle");

let sending = false;
let roles = [];
let currentRoleKey = "";

// 每个角色的欢迎语
const ROLE_SUBTITLES = {
  lawyer: "法律顾问在线，可咨询法律条文、案例分析",
  psychologist: "心理专家在线，提供情绪支持和专业建议",
  virtual_friend: "虚拟朋友在线，随时陪你聊天解闷",
};

// 不展示"引用资料"卡片的角色。
// 注意：这里只控制"显示"，不控制"检索"——虚拟朋友仍然会检索 rag_companion
// 里的人设语料，只是不把来源卡片摆在聊天界面上（陪聊角色甩参考资料很出戏）。
const NO_SOURCE_ROLES = ["virtual_friend"];

// 初始化
async function init() {
  await checkHealth();
  await loadRoles();
  bindEvents();
}

// 健康检查
async function checkHealth() {
  try {
    const r = await fetch(API + "/health");
    if (r.ok) { statusDot.className = "status-dot online"; statusText.textContent = "在线"; }
  } catch { statusDot.className = "status-dot offline"; statusText.textContent = "离线"; }
}

// 加载角色
async function loadRoles() {
  try {
    const r = await fetch(API + "/roles");
    const d = await r.json();
    roles = d.roles || [];
    roleSelect.innerHTML = "";
    roles.forEach(r => {
      const o = document.createElement("option");
      o.value = r.id;
      o.textContent = r.name + " - " + r.description;
      o.dataset.roleKey = r.role_key || "";
      o.dataset.suggestions = JSON.stringify(r.suggestions || []);
      roleSelect.appendChild(o);
    });
    // 触发一次切换事件，渲染默认角色的建议
    if (roles.length > 0) {
      onRoleChange();
    }
  } catch { roleSelect.innerHTML = "<option>加载失败</option>"; }
}

// 角色切换：更新建议卡片和欢迎语
function onRoleChange() {
  const selected = roleSelect.options[roleSelect.selectedIndex];
  if (!selected) return;
  currentRoleKey = selected.dataset.roleKey || "";

  // 更新欢迎语
  welcomeSubtitle.textContent = ROLE_SUBTITLES[currentRoleKey] || "选择角色后开始提问";

  // 更新建议卡片
  let suggestions = [];
  try { suggestions = JSON.parse(selected.dataset.suggestions || "[]"); } catch {}
  suggestionCardsEl.innerHTML = "";
  suggestions.forEach(s => {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.q = s.question;
    card.innerHTML = '<span class="card-icon">' + esc(s.tag) + '</span>' +
                     '<span class="card-text">' + esc(s.question) + '</span>';
    card.addEventListener("click", () => { userInput.value = s.question; onSend(); });
    suggestionCardsEl.appendChild(card);
  });
}

// 绑定事件
function bindEvents() {
  sendBtn.addEventListener("click", onSend);
  roleSelect.addEventListener("change", onRoleChange);
  userInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
  });
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
    sendBtn.disabled = userInput.value.trim() && !sending ? false : true;
  });
  clearBtn.addEventListener("click", onClear);
}

// 发送消息（流式）
async function onSend() {
  const q = userInput.value.trim();
  if (!q || sending) return;
  const rid = roleSelect.value;
  if (!rid) { alert("请先选择角色"); return; }
  const uid = parseInt(userIdEl.value);
  const k = parseInt(topKEl.value);

  sending = true;
  sendBtn.disabled = true;
  sendBtn.style.opacity = "0.5";
  userInput.value = "";
  userInput.style.height = "auto";

  // 隐藏欢迎页
  welcomeScreen.classList.add("hidden");

  // 渲染用户消息
  addMsg("user", q);

  // 创建 AI 消息占位
  const botMsg = addMsg("bot", "");
  const contentEl = botMsg.querySelector(".msg-content");
  const cursor = document.createElement("span");
  cursor.className = "cursor-blink";
  contentEl.appendChild(cursor);

  // 流式请求
  try {
    const r = await fetch(API + "/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: uid,
        role_id: parseInt(rid),
        query: q,
        top_k: k,
        use_local_llm: false,
        role_key: currentRoleKey
      })
    });

    if (!r.ok) {
      cursor.remove();
      contentEl.textContent = "[错误] " + r.statusText;
      return;
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    const PREFIX = "###SOURCES###";  // 后端首条消息的引用来源标记
    let fullText = "";
    let buffer = "";
    let sourcesHandled = false;  // 首条来源消息是否已处理

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // 首条消息是引用来源行，先剥离出来渲染，剩下的才是正文
      if (!sourcesHandled) {
        // 标记前缀还没收全（可能被 TCP 分片），等下一块再判断
        if (buffer.length < PREFIX.length && PREFIX.startsWith(buffer)) continue;
        if (buffer.startsWith(PREFIX)) {
          const endIdx = buffer.indexOf("\n");
          if (endIdx === -1) continue;  // 整行 JSON 还没收全，继续等
          let sources = [];
          try { sources = JSON.parse(buffer.substring(PREFIX.length, endIdx)); } catch {}
          buffer = buffer.substring(endIdx + 1);
          // 需要展示引用的角色才渲染卡片（虚拟朋友跳过，避免出戏）
          if (sources.length && !NO_SOURCE_ROLES.includes(currentRoleKey)) {
            renderSources(botMsg, sources);
          }
        }
        sourcesHandled = true;  // 处理完（含"响应里没有标记"的情况）就不再判断
      }

      if (buffer) {
        fullText += buffer;
        buffer = "";
        contentEl.textContent = fullText;
        contentEl.appendChild(cursor);
        scrollToBottom();
      }
    }

    cursor.remove();

  } catch (err) {
    cursor.remove();
    contentEl.textContent = "[网络错误] " + err.message;
  } finally {
    sending = false;
    sendBtn.style.opacity = "1";
    sendBtn.disabled = userInput.value.trim() ? false : true;
    scrollToBottom();
  }
}

// 清空记忆
async function onClear() {
  const rid = roleSelect.value;
  if (!rid) { alert("请先选择角色"); return; }
  if (!confirm("确认清空当前对话记忆？")) return;
  try {
    const r = await fetch(API + "/memory/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: parseInt(userIdEl.value), role_id: parseInt(rid) })
    });
    const d = await r.json();
    messagesEl.innerHTML = "";
    welcomeScreen.classList.remove("hidden");
    onRoleChange();
    alert(d.message);
  } catch (e) { alert("清空失败: " + e.message); }
}

// 添加消息
function addMsg(role, text) {
  const m = document.createElement("div");
  m.className = "msg " + role;
  const avatar = role === "user" ? "我" : "AI";
  m.innerHTML = "<div class='msg-avatar'>" + avatar + "</div>" +
    "<div class='msg-body'><div class='msg-content'>" + esc(text) + "</div></div>";
  messagesEl.appendChild(m);
  scrollToBottom();
  return m;
}

function scrollToBottom() { chatContainer.scrollTop = chatContainer.scrollHeight; }
function esc(t) { const d = document.createElement("div"); d.textContent = t; return d.innerHTML; }
function trunc(s, n) { if (!s) return ""; return s.length > n ? s.substring(0, n) + "..." : s; }

// 渲染"引用资料"卡片（挂在 AI 消息下方）
// 只有需要展示来源的角色才会调用；内容截断 80 字，鼠标悬停可见完整来源名
function renderSources(botMsg, sources) {
  const srcEl = document.createElement("div");
  srcEl.className = "msg-sources";
  srcEl.innerHTML = "<div class='msg-sources-title'>引用资料 (" + sources.length + ")</div>" +
    sources.map(s => "<div class='msg-source-item' title='" + esc(s.source) + "'>[" + s.index + "] " + esc(s.source) + " - " + trunc(s.content, 80) + "</div>").join("");
  botMsg.querySelector(".msg-body").appendChild(srcEl);
  scrollToBottom();
}

init();
