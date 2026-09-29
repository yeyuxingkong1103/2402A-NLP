const $ = (selector) => document.querySelector(selector);
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const roleNames = {"1": "法律助手", "2": "法学老师"};
const storageKey = "lvzhi-workspace-v1";
let state = {user: "user1", role: "1", active: "", sessions: []};
let busy = false;
let toastTimer;

// 页面记录保存在浏览器；用于问答的记忆由后台保存。
try {
  const saved = JSON.parse(localStorage.getItem(storageKey));
  if (saved && /^[A-Za-z0-9_-]{1,32}$/.test(saved.user) && Array.isArray(saved.sessions)) {
    state = saved;
    state.role = roleNames[state.role] ? state.role : "1";
    state.sessions = state.sessions.filter(s => s && Array.isArray(s.messages) && roleNames[s.role]);
    for (const session of state.sessions) {
      for (const message of session.messages) {
        if (message.pending) {
          message.pending = false;
          message.error = "上次连接已中断。如后台仍在处理，请稍后再试。";
        }
      }
    }
  }
} catch (error) { /* 无法读取本地记录时，使用新工作台。 */ }

function toast(text) {
  $("#toast").textContent = text;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 3800);
}

function save() {
  try { localStorage.setItem(storageKey, JSON.stringify(state)); }
  catch (error) { toast("浏览器无法保存记录，请及时导出当前对话。"); }
}

function currentSession() {
  return state.sessions.find(s => s.id === state.active && s.user === state.user && s.role === state.role);
}

function newSession() {
  if (busy) return;
  // 尚未提问的空会话不重复保留。
  state.sessions = state.sessions.filter(s => s.messages.length);
  const session = {id: crypto.randomUUID(), user: state.user, role: state.role,
    title: "新对话", created: Date.now(), messages: []};
  state.sessions.unshift(session);
  state.active = session.id;
  save();
  render();
  closePanels();
  $("#question").value = "";
  updateInput();
  $("#question").focus();
}

function renderHistory() {
  const history = $("#history");
  history.replaceChildren();
  const sessions = state.sessions.filter(s => s.user === state.user && s.messages.length);
  if (!sessions.length) {
    history.innerHTML = '<p class="history-empty">你的思考，从这里开始。<br>提问后，对话会保存在这里。</p>';
    return;
  }
  for (const session of sessions) {
    const button = document.createElement("button");
    button.className = "history-item" + (session.id === state.active ? " selected" : "");
    button.disabled = busy;
    button.innerHTML = icon("chat") + "<span></span><small></small>";
    button.querySelector("span").textContent = session.title;
    button.querySelector("small").textContent = session.role === "2" ? "老师" : "助手";
    button.title = session.title;
    button.onclick = () => {
      state.active = session.id;
      state.role = session.role;
      save(); render(); closePanels();
    };
    history.append(button);
  }
}

// 只渲染常用 Markdown；先转义内容，不执行模型或用户提供的 HTML。
function escapeHTML(text) {
  return String(text).replace(/[&<>"']/g, char => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
}

function inline(text) {
  return escapeHTML(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function markdown(text) {
  let html = "", list = "", code = false;
  for (const line of text.split("\n")) {
    if (line.trim().startsWith("```")) {
      if (list) { html += `</${list}>`; list = ""; }
      html += code ? "</code></pre>" : "<pre><code>";
      code = !code;
      continue;
    }
    if (code) { html += escapeHTML(line) + "\n"; continue; }
    const bullet = line.match(/^\s*[-*]\s+(.+)/);
    const numbered = line.match(/^\s*\d+[.)、]\s*(.+)/);
    const nextList = bullet ? "ul" : numbered ? "ol" : "";
    if (list !== nextList) {
      if (list) html += `</${list}>`;
      if (nextList) html += `<${nextList}>`;
      list = nextList;
    }
    if (nextList) html += `<li>${inline((bullet || numbered)[1])}</li>`;
    else if (/^#{1,6}\s/.test(line)) html += `<h3>${inline(line.replace(/^#{1,6}\s+/, ""))}</h3>`;
    else if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  if (list) html += `</${list}>`;
  if (code) html += "</code></pre>";
  return html;
}

const emptySources = $("#sources").innerHTML;
let firstDocument = "";

function sourceLocation(source) {
  return source.location || (source.page ? `第${source.page}页` : "全文");
}

function sourceLink(source) {
  const name = source.source || firstDocument;
  const url = "/api/document" + (name ? `?source=${encodeURIComponent(name)}` : "");
  return url + (name.toLowerCase().endsWith(".pdf") && source.page > 0 ? `#page=${source.page}` : "");
}

function renderSources(sources = []) {
  $("#source-count").textContent = sources.length;
  $("#sources").replaceChildren();
  if (!sources.length) { $("#sources").innerHTML = emptySources; return; }
  const note = document.createElement("p");
  note.className = "source-caption";
  note.textContent = "以下为检索得到的原文片段，供核对回答。";
  $("#sources").append(note);
  sources.forEach((source, index) => {
    const card = document.createElement("details");
    card.className = "source-card";
    card.open = index === 0;
    card.innerHTML = `<summary><span class="source-index">${String(index + 1).padStart(2,"0")}</span><span class="source-name"></span>${icon("chevron")}</summary><p></p><a target="_blank" rel="noopener">查看原始文件 ${icon("out")}</a>`;
    const label = `${source.source || firstDocument || "资料"} · ${sourceLocation(source)}`;
    card.querySelector(".source-name").textContent = label;
    card.querySelector("summary").title = label;
    card.querySelector("p").textContent = source.text;
    card.querySelector("a").href = sourceLink(source);
    $("#sources").append(card);
  });
}

function renderMessages() {
  const session = currentSession();
  const hasMessages = session && session.messages.length > 0;
  $("#welcome").hidden = hasMessages;
  $("#messages").hidden = !hasMessages;
  $("#messages").replaceChildren();
  if (!hasMessages) return;
  for (const message of session.messages) {
    const article = document.createElement("article");
    article.className = "message " + (message.who === "user" ? "user" : "assistant");
    if (message.who === "user") {
      article.innerHTML = '<div class="user-bubble"></div><span class="message-meta">你</span>';
      article.querySelector(".user-bubble").textContent = message.text;
    } else {
      article.innerHTML = `<div class="assistant-heading"><span class="assistant-logo">律</span>${roleNames[session.role]}<small>基于知识库</small></div>`;
      if (message.pending) {
        article.innerHTML += '<div class="progress" role="status"><span class="spinner"></span><span class="progress-text"></span><small class="elapsed"></small></div>';
        article.querySelector(".progress-text").textContent = message.stage || "正在准备回答";
        article.querySelector(".elapsed").textContent = "本地生成中";
      } else if (message.error) {
        const error = document.createElement("div");
        error.className = "error-box";
        const text = document.createElement("span");
        text.textContent = message.error;
        const retry = document.createElement("button");
        retry.className = "retry-button";
        retry.textContent = "重新提问";
        retry.disabled = busy;
        retry.onclick = () => {
          $("#question").value = message.question || "";
          updateInput(); $("#question").focus();
        };
        error.append(text, retry); article.append(error);
      } else {
        const body = document.createElement("div");
        body.className = "answer-body";
        body.innerHTML = markdown(message.text || "");
        article.append(body);
        const actions = document.createElement("div");
        actions.className = "answer-actions";
        const seen = new Set();
        for (const source of message.sources || []) {
          const key = `${source.source || firstDocument}:${sourceLocation(source)}`;
          if (seen.has(key)) continue;
          seen.add(key);
          const button = document.createElement("button");
          button.className = "source-chip";
          button.innerHTML = icon("file");
          button.append(document.createTextNode(sourceLocation(source)));
          button.title = "在右侧查看引用原文";
          button.onclick = () => {
            renderSources(message.sources);
            document.querySelectorAll(".source-card").forEach((card, i) => {
              card.open = `${message.sources[i].source || firstDocument}:${sourceLocation(message.sources[i])}` === key;
            });
            if (window.innerWidth <= 1250) openPanel("sources");
          };
          actions.append(button);
        }
        const copy = document.createElement("button");
        copy.className = "icon-button";
        copy.innerHTML = icon("copy");
        copy.title = "复制回答"; copy.setAttribute("aria-label", "复制回答");
        copy.onclick = async () => {
          try { await navigator.clipboard.writeText(message.text); toast("回答已复制"); }
          catch (error) { toast("复制失败，请选中文字复制。"); }
        };
        actions.append(copy); article.append(actions);
        if (message.warning) {
          const warning = document.createElement("p");
          warning.className = "answer-warning"; warning.textContent = message.warning;
          article.append(warning);
        }
      }
    }
    $("#messages").append(article);
  }
}

function render() {
  $("#user-name").textContent = state.user;
  $("#avatar").textContent = state.user[0].toUpperCase();
  $("#composer-role").textContent = roleNames[state.role];
  $("#welcome-description").innerHTML = state.role === "2"
    ? "我是你的法学老师。用通俗的语言和简单的例子，<br class='desktop-break'>一起读懂条文背后的含义。"
    : "我是你的法律助手。一起从条文出发，<br class='desktop-break'>找到问题的依据，理清复杂的法律关系。";
  document.querySelectorAll(".role-button").forEach(button => {
    const active = button.dataset.role === state.role;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active);
  });
  renderHistory(); renderMessages();
  const last = [...(currentSession()?.messages || [])].reverse().find(m => m.sources?.length);
  renderSources(last?.sources || []);
  $("#export-button").disabled = !currentSession()?.messages.length;
}

function updateInput() {
  const input = $("#question");
  $("#char-count").textContent = `${input.value.length} / 2000`;
  $("#send-button").disabled = busy || !input.value.trim();
  input.style.height = "auto";
  input.style.height = Math.min(140, Math.max(48, input.scrollHeight)) + "px";
}

function setBusy(value) {
  busy = value;
  for (const button of document.querySelectorAll(".role-button, #new-chat, #user-button, .history-item, #suggestions button, .retry-button")) button.disabled = busy;
  $("#question").disabled = busy;
  updateInput();
}

function scrollDown() {
  $("#chat-scroll").scrollTop = $("#chat-scroll").scrollHeight;
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (busy || !question || question.length > 2000) return;
  if (!currentSession()) newSession();
  const session = currentSession();
  if (!session.messages.length) session.title = question.slice(0, 28);
  session.messages.push({who: "user", text: question});
  const answer = {who: "assistant", text: "", question, pending: true, sources: []};
  session.messages.push(answer);
  save(); render(); setBusy(true); renderSources();
  $("#question").value = ""; updateInput(); scrollDown();
  const started = Date.now();
  const timer = setInterval(() => {
    const elapsed = $(".elapsed");
    if (elapsed) elapsed.textContent = `${Math.floor((Date.now() - started) / 1000)} 秒`;
  }, 1000);
  let complete = false;
  const handleEvent = (data) => {
    if (data.type === "stage") {
      answer.stage = data.text;
      const progress = $(".progress-text");
      if (progress) progress.textContent = data.text;
      $("#announcement").textContent = data.text;
    } else if (data.type === "sources") {
      answer.sources = data.sources; renderSources(data.sources);
    } else if (data.type === "answer") {
      answer.pending = false; answer.text = data.answer;
      answer.sources = data.sources; answer.warning = data.warning;
      complete = true;
      renderMessages(); renderSources(data.sources); save(); scrollDown();
      $("#announcement").textContent = "回答已完成，可以查看参考资料或继续追问。";
    } else if (data.type === "error") throw new Error(data.text);
  };
  try {
    const response = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, user_id: state.user, role: state.role, session_id: session.id})});
    if (!response.ok) {
      const data = await response.json();
      throw new Error(typeof data.detail === "string" ? data.detail : "问题格式不正确，请检查后重试。");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) if (line.trim()) handleEvent(JSON.parse(line));
      if (done) break;
    }
    if (buffer.trim()) handleEvent(JSON.parse(buffer));
    if (!complete) throw new Error("连接已中断，未收到完整回答，请稍后重试。");
  } catch (error) {
    if (!complete) {
      answer.pending = false;
      answer.error = error instanceof TypeError ? "无法连接问答服务，请确认 web_app.py 正在运行。" : error.message;
      $("#announcement").textContent = answer.error;
    }
    checkStatus();
  } finally {
    clearInterval(timer); answer.pending = false;
    setBusy(false); save(); renderMessages(); renderHistory(); scrollDown();
    $("#question").focus();
  }
}

async function checkStatus() {
  const refresh = $("#refresh-status");
  refresh.disabled = true; refresh.classList.add("loading-status");
  try {
    const response = await fetch("/api/status", {signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error("status");
    const data = await response.json();
    $("#connection").className = "connection " + (data.ready ? "ready" : "offline");
    $("#connection span").textContent = data.ready ? "本地服务已就绪" : "部分服务未连接";
    $("#services").replaceChildren();
    const labels = {"Milvus": "知识库", "Redis": "对话记忆", "Ollama": "问答模型"};
    for (const service of data.services) {
      const item = document.createElement("span");
      item.className = "service-item" + (service.ok ? " ok" : "");
      item.title = `${service.name}：${service.detail || (service.ok ? "已连接" : "未连接")}`;
      item.innerHTML = "<i></i>";
      item.append(document.createTextNode(labels[service.name] + (service.ok ? "" : "未就绪")));
      $("#services").append(item);
    }
    const milvus = data.services.find(s => s.name === "Milvus");
    firstDocument = data.document_source || "";
    $("#document-title").textContent = data.document_name || "未指定资料";
    $("#document-title").title = data.document_name || "";
    $("#document-subtitle").textContent = data.document_count > 1 ? `${data.document_count} 个资料文件` : data.document_source || "单个资料文件";
    $("#document-subtitle").title = data.document_source || "";
    $("#document-type").textContent = data.document_count > 1 ? "DATA" : data.document_type || "FILE";
    $("#document-link").href = "/api/document" + (firstDocument ? `?source=${encodeURIComponent(firstDocument)}` : "");
    $("#dataset-label").textContent = data.document_count > 1 ? `${data.document_count} 份` : data.document_type || "资料";
    $("#knowledge-tag").textContent = data.document_count > 1 ? `${data.document_count} 份资料` : "法律知识库";
    $("#document-meta").textContent = milvus?.ok ? `${milvus.count} 个资料片段 · 已入库` : "知识库暂未就绪";
    $("#document-link").setAttribute("aria-disabled", !data.document);
    if (!data.document) $("#document-meta").textContent = "未找到资料文件";
  } catch (error) {
    $("#connection").className = "connection offline";
    $("#connection span").textContent = "服务暂不可用";
    $("#services").textContent = "无法连接，请检查启动窗口后刷新。";
    $("#document-meta").textContent = "暂时无法检查知识库";
  } finally {
    refresh.disabled = false; refresh.classList.remove("loading-status");
  }
}

let panelTrigger;
function closePanels() {
  $(".sidebar").classList.remove("open"); $(".source-panel").classList.remove("open");
  $("#backdrop").hidden = true;
  if (panelTrigger) { panelTrigger.focus(); panelTrigger = null; }
}

function openPanel(name) {
  closePanels(); panelTrigger = document.activeElement;
  const panel = name === "sources" ? $(".source-panel") : $(".sidebar");
  panel.classList.add("open"); $("#backdrop").hidden = false;
  panel.querySelector("button, a").focus();
}

$("#chat-form").onsubmit = sendQuestion;
$("#question").oninput = updateInput;
$("#question").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); $("#chat-form").requestSubmit();
  }
};
$("#new-chat").onclick = newSession;
document.querySelectorAll(".role-button").forEach(button => {
  button.onclick = () => { if (state.role !== button.dataset.role) { state.role = button.dataset.role; newSession(); } };
});
document.querySelectorAll("#suggestions button").forEach(button => {
  button.onclick = () => { $("#question").value = button.dataset.question; updateInput(); $("#question").focus(); };
});
$("#user-button").onclick = () => { $("#user-input").value = state.user; $("#user-dialog").showModal(); };
$("#close-dialog").onclick = () => $("#user-dialog").close();
$("#user-form").onsubmit = (event) => {
  event.preventDefault();
  const user = $("#user-input").value.trim();
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(user)) return;
  if (user !== state.user) { state.user = user; newSession(); }
  $("#user-dialog").close(); render();
};
$("#export-button").onclick = () => {
  const session = currentSession();
  if (!session?.messages.length) return;
  let text = `# ${session.title}\n\n角色：${roleNames[session.role]}\n\n`;
  for (const message of session.messages) {
    if (message.pending) continue;
    text += `## ${message.who === "user" ? "你" : roleNames[session.role]}\n\n${message.error || message.text}\n\n`;
    if (message.sources?.length) text += "参考资料：" + [...new Set(message.sources.map(s => `${s.source || firstDocument} ${sourceLocation(s)}`))].join("；") + "\n\n";
  }
  const url = URL.createObjectURL(new Blob([text], {type: "text/markdown;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = `律知对话-${session.id.slice(0,8)}.md`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000); toast("对话已导出");
};
$("#document-link").onclick = (event) => {
  if ($("#document-link").getAttribute("aria-disabled") === "true") { event.preventDefault(); toast("数据路径中没有找到资料文件。"); }
};
$("#refresh-status").onclick = checkStatus;
$("#menu-button").onclick = () => openPanel("menu");
$("#sources-toggle").onclick = () => openPanel("sources");
$("#close-sources").onclick = closePanels;
$("#backdrop").onclick = closePanels;
document.addEventListener("keydown", event => {
  if (event.key === "Escape") closePanels();
  if (event.key === "Tab" && !$("#backdrop").hidden && !$("#user-dialog").open) {
    const panel = $(".sidebar.open") || $(".source-panel.open");
    const focusable = [...panel.querySelectorAll("button:not(:disabled), a, summary")].filter(el => el.getClientRects().length);
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
});
window.addEventListener("resize", () => { if (window.innerWidth > 1250) closePanels(); });
if (!currentSession()) newSession(); else render();
updateInput(); checkStatus();
