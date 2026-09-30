/* ============ 状态 ============ */
const state = {
  // 解析：前端全局状态对象
  token: localStorage.getItem("rp_token") || "",
  // 解析：登录 token（localStorage 持久化，刷新/重开免登录）
  username: localStorage.getItem("rp_username") || "",
  // 解析：用户名（显示头像首字母用）
  roles: [],
  // 解析：角色列表缓存
  currentRole: null,
  // 解析：当前选中角色
};

const $ = (sel) => document.querySelector(sel);
// 解析：$ 是 querySelector 的简写（类似 jQuery）

/* ============ 登录/注册 ============ */
let authMode = "login";
// 解析：当前认证模式（login/register）

function switchTab(mode) {
  // 解析：切换登录/注册标签页
  authMode = mode;
  // 解析：更新模式
  $("#tab-login").classList.toggle("active", mode === "login");
  // 解析：登录标签高亮切换
  $("#tab-register").classList.toggle("active", mode === "register");
  // 解析：注册标签高亮切换
  $("#auth-btn").textContent = mode === "login" ? "登 录" : "注 册";
  // 解析：按钮文字跟随模式
}
$("#tab-login").onclick = () => switchTab("login");
// 解析：点登录标签
$("#tab-register").onclick = () => switchTab("register");
// 解析：点注册标签

async function handleAuth() {
  // 解析：登录/注册提交
  const username = $("#username").value.trim();
  // 解析：取用户名（去空白）
  const password = $("#password").value;
  // 解析：取密码
  if (!username || !password) {
    // 解析：空输入
    $("#auth-error").textContent = "请输入用户名和密码";
    // 解析：提示
    return;
    // 解析：返回
  }
  const path = authMode === "login" ? "/api/users/login" : "/api/users/register";
  // 解析：按模式选接口
  const resp = await fetch(path, {
    // 解析：请求后端
    method: "POST",
    // 解析：POST
    headers: { "Content-Type": "application/json" },
    // 解析：JSON 头
    body: JSON.stringify({ username, password }),
    // 解析：请求体
  });
  if (!resp.ok) {
    // 解析：请求失败
    const err = await resp.json().catch(() => ({}));
    // 解析：取错误详情（失败时给空对象）
    $("#auth-error").textContent = err.detail || "操作失败";
    // 解析：显示错误
    return;
    // 解析：返回
  }
  const data = await resp.json();
  // 解析：解析响应
  state.token = data.token;
  // 解析：保存 token 到状态
  state.username = data.username;
  // 解析：保存用户名
  localStorage.setItem("rp_token", data.token);
  // 解析：token 持久化
  localStorage.setItem("rp_username", data.username);
  // 解析：用户名持久化
  enterMain();
  // 解析：进入主界面
}
$("#auth-btn").onclick = handleAuth;
// 解析：登录按钮
$("#password").onkeydown = (e) => { if (e.key === "Enter") handleAuth(); };
// 解析：密码框回车提交

function logout() {
  // 解析：退出登录
  state.token = "";
  // 解析：清 token
  localStorage.removeItem("rp_token");
  // 解析：删持久化 token
  $("#main-view").classList.add("hidden");
  // 解析：隐藏主界面
  $("#login-view").classList.remove("hidden");
  // 解析：显示登录页
}

/* ============ 主界面 ============ */
function enterMain() {
  // 解析：进入主界面
  $("#login-view").classList.add("hidden");
  // 解析：隐藏登录页
  $("#main-view").classList.remove("hidden");
  // 解析：显示主界面
  loadRoles();
  // 解析：加载角色列表
}

async function api(path, options = {}) {
  // 解析：统一 API 请求封装（自动带 token、401 自动登出）
  const headers = { ...(options.headers || {}) };
  // 解析：复制请求头
  if (state.token) headers["X-Token"] = state.token;
  // 解析：自动附加鉴权头
  const resp = await fetch(path, { ...options, headers });
  // 解析：请求
  if (resp.status === 401) {
    // 解析：未登录/过期
    logout();
    // 解析：自动登出
    throw new Error("登录已过期");
    // 解析：抛错
  }
  return resp;
  // 解析：返回响应
}

async function loadRoles() {
  // 解析：加载角色列表
  const resp = await api("/api/roles");
  // 解析：请求角色接口
  if (!resp.ok) return;
  // 解析：失败返回
  state.roles = await resp.json();
  // 解析：缓存角色
  const list = $("#role-list");
  // 解析：角色列表容器
  list.innerHTML = "";
  // 解析：清空
  state.roles.forEach((role) => {
    // 解析：逐角色
    const card = document.createElement("div");
    // 解析：建卡片
    card.className = "role-card";
    // 解析：样式类
    card.innerHTML = `
      <div class="avatar">${role.name[0]}</div>
      <div class="role-info">
        <div class="role-name">${role.name}</div>
        <div class="role-category">${role.category}</div>
      </div>`;
    // 解析：卡片内容（首字头像+名称+分类）
    card.onclick = () => selectRole(role, card);
    // 解析：点击选择角色
    list.appendChild(card);
    // 解析：加入列表
  });
}

async function selectRole(role, card) {
  // 解析：选择角色
  state.currentRole = role;
  // 解析：保存当前角色
  document.querySelectorAll(".role-card").forEach((c) => c.classList.remove("active"));
  // 解析：清除全部高亮
  if (card) card.classList.add("active");
  // 解析：高亮当前卡片
  $("#chat-avatar").textContent = role.name[0];
  // 解析：聊天区头像
  $("#chat-role-name").textContent = role.name;
  // 解析：角色名
  $("#chat-role-category").textContent = role.category;
  // 解析：分类
  $("#chat-messages").innerHTML = "";
  // 解析：清空聊天区
  await loadHistory();
  // 解析：加载历史
  await loadDocs();
  // 解析：加载知识库文档
}

/* ============ 聊天 ============ */
function addMessage(sender, text, extra = {}) {
  // 解析：添加消息气泡
  const box = $("#chat-messages");
  // 解析：聊天容器
  document.querySelector(".welcome-tip")?.remove();
  // 解析：移除欢迎提示（可选链安全）
  const wrap = document.createElement("div");
  // 解析：消息行
  wrap.className = `msg ${sender}`;
  // 解析：user/ai 样式
  const avatar = document.createElement("div");
  // 解析：头像
  avatar.className = "avatar";
  // 解析：样式
  avatar.textContent = sender === "user" ? state.username[0] : state.currentRole.name[0];
  // 解析：用户取用户名首字，AI 取角色名首字
  const bubble = document.createElement("div");
  // 解析：气泡
  bubble.className = "bubble";
  // 解析：样式
  wrap.appendChild(avatar);
  // 解析：加头像
  wrap.appendChild(bubble);
  // 解析：加气泡
  box.appendChild(wrap);
  // 解析：加入聊天区
  box.scrollTop = box.scrollHeight;
  // 解析：滚动到底部
  return { wrap, bubble };
  // 解析：返回容器引用（后续追加内容用）
}

function renderExtra(wrap, sources, warnings) {
  // 解析：渲染引用来源与校验提醒
  if (sources && sources.length) {
    // 解析：有来源
    const box = document.createElement("details");
    // 解析：可展开框
    box.className = "sources-box";
    // 解析：样式
    box.innerHTML = `<summary>📖 引用来源（${sources.length}）</summary>` +
      sources.map((s) => `<li>${s}</li>`).join("");
    // 解析：标题与来源列表
    wrap.appendChild(box);
    // 解析：加入消息
  }
  if (warnings && warnings.length) {
    // 解析：有提醒
    const box = document.createElement("details");
    // 解析：可展开框
    box.className = "warnings-box";
    // 解析：样式
    box.open = true;
    // 解析：默认展开（提醒要醒目）
    box.innerHTML = `<summary>⚠️ 校验提醒（${warnings.length}）</summary>` +
      warnings.map((w) => `<div>${w}</div>`).join("");
    // 解析：标题与提醒列表
    wrap.appendChild(box);
    // 解析：加入消息
  }
}

async function sendMessage() {
  // 解析：发送消息（流式）
  const input = $("#chat-input");
  // 解析：输入框
  const text = input.value.trim();
  // 解析：输入内容
  if (!text || !state.currentRole) return;
  // 解析：空输入或未选角色
  input.value = "";
  // 解析：清空输入框
  input.style.height = "auto";
  // 解析：重置高度

  addMessage("user", text);
  // 解析：立即渲染用户气泡

  const { wrap, bubble } = addMessage("ai", "");
  // 解析：预建 AI 气泡（待流式填充）
  const cursor = document.createElement("span");
  // 解析：光标
  cursor.className = "cursor";
  // 解析：闪烁样式
  cursor.textContent = "▌";
  // 解析：光标字符
  bubble.appendChild(cursor);
  // 解析：加到气泡

  try {
    // 解析：流式请求
    const resp = await api("/api/chat/stream", {
      // 解析：SSE 接口
      method: "POST",
      // 解析：POST
      headers: { "Content-Type": "application/json" },
      // 解析：JSON 头
      body: JSON.stringify({ role_id: state.currentRole.id, content: text, use_rag: true }),
      // 解析：角色、内容、开 RAG
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    // 解析：失败抛错

    const reader = resp.body.getReader();
    // 解析：流读取器
    const decoder = new TextDecoder();
    // 解析：字节解码器
    let full = "";
    // 解析：完整回复
    let sources = [];
    // 解析：引用来源
    let warnings = [];
    // 解析：校验提醒
    let buffer = "";
    // 解析：SSE 行缓冲（跨包断行处理）

    while (true) {
      // 解析：读流循环
      const { done, value } = await reader.read();
      // 解析：读一块
      if (done) break;
      // 解析：流结束
      buffer += decoder.decode(value, { stream: true });
      // 解析：解码并入缓冲
      const lines = buffer.split("\n");
      // 解析：按行切分
      buffer = lines.pop();
      // 解析：最后不完整行留缓冲
      for (const line of lines) {
        // 解析：逐完整行
        if (!line.startsWith("data:")) continue;
        // 解析：非数据事件跳过
        const payload = line.slice(5).trim();
        // 解析：取 JSON 载荷
        if (payload === "[DONE]") continue;
        // 解析：结束事件跳过
        try {
          // 解析：解析事件
          const event = JSON.parse(payload);
          // 解析：JSON 解析
          if (event.delta) {
            // 解析：文本增量
            full += event.delta;
            // 解析：拼完整回复
            bubble.textContent = full;
            // 解析：逐字更新气泡（打字机效果）
            bubble.appendChild(cursor);
            // 解析：光标保持在末尾
            $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
            // 解析：滚动到底
          } else if (event.sources) {
            // 解析：来源事件
            sources = event.sources;
            // 解析：保存来源
          } else if (event.warnings) {
            // 解析：提醒事件
            warnings = event.warnings;
            // 解析：保存提醒
          }
        } catch (e) { /* 忽略解析失败 */ }
        // 解析：单行解析失败不中断整体
      }
    }
    cursor.remove();
    // 解析：移除光标
    renderExtra(wrap, sources, warnings);
    // 解析：渲染来源与提醒
  } catch (e) {
    // 解析：流式失败
    cursor.remove();
    // 解析：移除光标
    bubble.textContent = "（对话失败：" + e.message + "）";
    // 解析：显示错误
  }
}

$("#send-btn").onclick = sendMessage;
// 解析：发送按钮
$("#chat-input").addEventListener("keydown", (e) => {
  // 解析：输入框键盘事件
  if (e.key === "Enter" && !e.shiftKey) {
    // 解析：回车且未按 Shift
    e.preventDefault();
    // 解析：阻止默认换行
    sendMessage();
    // 解析：发送
  }
});
$("#chat-input").addEventListener("input", (e) => {
  // 解析：输入自适应高度
  e.target.style.height = "auto";
  // 解析：先重置
  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  // 解析：按内容撑高（上限 120px）
});

/* ============ 历史记录 ============ */
async function loadHistory() {
  // 解析：加载会话历史
  if (!state.currentRole) return;
  // 解析：未选角色
  const resp = await api(`/api/chat/history?role_id=${state.currentRole.id}`);
  // 解析：请求历史
  if (!resp.ok) return;
  // 解析：失败返回
  const msgs = await resp.json();
  // 解析：消息列表
  for (const m of msgs) {
    // 解析：逐消息
    addMessage(m.sender === "user" ? "user" : "ai", m.content);
    // 解析：按发送方渲染气泡
  }
}

/* ============ 知识库 ============ */
$("#kb-upload-btn").onclick = () => $("#kb-file").click();
// 解析：上传按钮触发文件选择
$("#kb-file").onchange = async (e) => {
  // 解析：选中文件后
  const file = e.target.files[0];
  // 解析：取文件
  if (!file || !state.currentRole) return;
  // 解析：无文件或未选角色
  const form = new FormData();
  // 解析：multipart 表单
  form.append("file", file);
  // 解析：附加文件
  const resp = await api(`/api/knowledge/upload?role_id=${state.currentRole.id}`, {
    // 解析：上传接口
    method: "POST",
    // 解析：POST
    body: form,
    // 解析：表单体
  });
  if (resp.ok) {
    // 解析：成功
    const result = await resp.json();
    // 解析：入库结果
    alert(`已入库 ${result.chunks} 块（自动去水印/清洗/去重）`);
    // 解析：提示
    loadDocs();
    // 解析：刷新文档列表
  } else {
    // 解析：失败
    const err = await resp.json().catch(() => ({}));
    // 解析：错误详情
    alert("上传失败：" + (err.detail || resp.status));
    // 解析：提示
  }
  e.target.value = "";
  // 解析：清空选择（同文件可再次上传）
};

async function loadDocs() {
  // 解析：加载知识库文档列表
  if (!state.currentRole) return;
  // 解析：未选角色
  const resp = await api(`/api/knowledge/docs?role_id=${state.currentRole.id}`);
  // 解析：请求文档
  const box = $("#kb-docs");
  // 解析：列表容器
  box.innerHTML = "";
  // 解析：清空
  if (!resp.ok) return;
  // 解析：失败返回
  const docs = await resp.json();
  // 解析：文档列表
  docs.forEach((doc) => {
    // 解析：逐文档
    const item = document.createElement("div");
    // 解析：文档行
    item.className = "kb-doc";
    // 解析：样式
    item.innerHTML = `
      <div>
        <div>📄 ${doc.source}（${doc.chunks} 块）</div>
        <span class="summary">${doc.summary ? doc.summary.slice(0, 40) + "…" : ""}</span>
      </div>
      <button class="del" title="删除文档">✕</button>`;
    // 解析：文件名、块数、摘要、删除按钮
    item.querySelector(".del").onclick = async () => {
      // 解析：删除按钮
      await api(`/api/knowledge/doc?role_id=${state.currentRole.id}&source=${encodeURIComponent(doc.source)}`, { method: "DELETE" });
      // 解析：调删除接口（文件名 URL 编码）
      loadDocs();
      // 解析：刷新
    };
    box.appendChild(item);
    // 解析：加入列表
  });
}

$("#logout-btn").onclick = logout;
// 解析：退出按钮

/* ============ 启动 ============ */
if (state.token) {
  // 解析：已有 token（上次登录过）
  enterMain();
  // 解析：直接进主界面
}
