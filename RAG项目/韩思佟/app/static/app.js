"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "reload-roles", "service-status", "service-message", "service-banner", "role-title", "role-description", "conversation",
    "welcome", "welcome-description", "suggestions", "messages", "pending", "notice", "chat-form",
    "message-input", "send-button", "char-count", "clear-button", "account-button", "account-label",
    "account-dialog", "close-dialog", "account-form", "username", "password", "account-error",
    "auth-submit", "login-tab", "register-tab",
  ].map((id) => [id, byId(id)]));
  const state = { account: null, role: null, busy: false, authBusy: false, serviceBusy: false,
    backendReady: false, modelReady: false, mode: "login", transcripts: new Map() };
  // The document base also handles /chat/ and a Jupyter /proxy/PORT/ prefix.
  const api = (path) => new URL(`api/${path}`, document.baseURI);

  async function request(path, options = {}, timeout = 180000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(api(path), {
        ...options, signal: controller.signal,
        headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers },
      });
      const raw = await response.text();
      let data;
      try { data = JSON.parse(raw); } catch { data = null; }
      if (!response.ok) {
        const detail = data?.detail;
        if (typeof detail === "string") throw new Error(detail);
        if (Array.isArray(detail)) throw new Error(detail.map((item) => item.msg).join("；"));
        throw new Error(`服务返回错误（HTTP ${response.status}）。请检查后端和模型服务日志。`);
      }
      if (data === null) throw new Error("返回的不是预期数据，请确认网页地址与后端服务一致。");
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("请求超时，服务器可能仍在处理。请稍后检查服务状态，避免立即重复提交。");
      if (error instanceof TypeError) throw new Error("连接失败，请检查云实例和服务是否仍在运行。");
      throw error;
    } finally { clearTimeout(timer); }
  }

  function notice(message = "") {
    ui.notice.textContent = message;
    ui.notice.hidden = !message;
  }

  function syncControls() {
    const locked = state.busy || state.authBusy;
    const ready = Boolean(state.account && state.role && state.modelReady && !locked);
    ui["message-input"].disabled = !ready;
    ui["message-input"].placeholder = !state.account ? "先登录或注册，再开始提问…" : !state.role ? "正在连接医生服务…" : !state.modelReady ? "模型正在准备，请等待服务就绪…" : "输入你的问题，或继续追问…";
    ui["send-button"].disabled = !ready || !ui["message-input"].value.trim();
    ui["clear-button"].disabled = !(state.account && state.role && state.backendReady && !locked);
    ui["account-button"].disabled = locked;
    ui["reload-roles"].disabled = locked || state.serviceBusy;
    ui.suggestions.querySelectorAll("button").forEach((button) => { button.disabled = locked; });
    ["username", "password", "auth-submit", "login-tab", "register-tab", "close-dialog"].forEach((key) => { ui[key].disabled = state.authBusy; });
    ui["char-count"].textContent = `${ui["message-input"].value.length} / 500`;
    ui["send-button"].firstChild.textContent = state.busy ? "处理中 " : "发送 ";
  }

  async function checkService() {
    if (state.serviceBusy) return;
    state.serviceBusy = true;
    syncControls();
    try {
      const [health, roles] = await Promise.all([
        request("status", {}, 12000),
        state.role ? Promise.resolve(null) : request("roles", {}, 12000),
      ]);
      if (!state.role) {
        const doctor = Array.isArray(roles) ? roles.find((role) => role.name === "医生") : null;
        if (!doctor) throw new Error("医生服务暂未配置，请检查后端。");
        state.role = doctor;
        ui["role-title"].textContent = "医生 · 知识问答";
        const description = String(doctor.description || "");
        ui["role-description"].textContent = description.includes("通用")
          ? description : "通用健康咨询；当前专属知识库主要覆盖高血压指南";
        renderTranscript();
      }
      state.backendReady = true;
      state.modelReady = health.ready === true;
      const fullyReady = state.modelReady && health.knowledge_ready === true;
      ui["service-status"].textContent = fullyReady ? (health.provider === "deepseek" ? "DeepSeek 在线服务已就绪" : "问答服务已就绪") : state.modelReady ? "模型已就绪" : "模型尚未就绪";
      ui["service-status"].parentElement.classList.toggle("connected", state.modelReady);
      ui["service-message"].textContent = typeof health.message === "string" && health.message
        ? health.message : state.modelReady ? "首次提问需要加载知识库，请稍等片刻。" : "模型仍在启动，请稍后再试；本页会自动检查。";
      ui["service-banner"].hidden = fullyReady;
      ui["reload-roles"].hidden = state.modelReady;
    } catch (error) {
      state.backendReady = false;
      state.modelReady = false;
      ui["service-status"].textContent = "后端未连接";
      ui["service-status"].parentElement.classList.remove("connected");
      ui["reload-roles"].hidden = false;
      ui["service-message"].textContent = error.message;
      ui["service-banner"].hidden = false;
    } finally {
      state.serviceBusy = false;
      syncControls();
    }
  }

  function currentTranscript() {
    const key = `${state.account?.user_id ?? "guest"}:${state.role?.id ?? "none"}`;
    if (!state.transcripts.has(key)) state.transcripts.set(key, []);
    return state.transcripts.get(key);
  }

  function sourcePanel(sources, rewrittenQuery) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `查看检索资料来源 · ${sources.length} 条`;
    details.append(summary);
    const list = document.createElement("ol");
    for (const source of sources) {
      const item = document.createElement("li");
      const filename = String(source.source || "知识库资料").split(/[\\/]/).pop();
      item.textContent = `${filename}${source.chunk_index == null ? "" : ` · 片段编号 ${source.chunk_index}`}`;
      if (typeof source.text === "string" && source.text.trim()) {
        const excerpt = document.createElement("p");
        excerpt.className = "source-excerpt";
        excerpt.textContent = source.text;
        item.append(excerpt);
      }
      list.append(item);
    }
    details.append(list);
    if (rewrittenQuery) {
      const query = document.createElement("p");
      query.className = "rewritten-query";
      query.textContent = `本次检索问题：${rewrittenQuery}`;
      details.append(query);
    }
    return details;
  }

  function messageElement(message) {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${message.kind}`;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = message.kind === "user" ? "我" : message.kind === "error" ? "!" : "✦";
    const content = document.createElement("div");
    content.className = "message-content";
    const label = document.createElement("p");
    label.className = "message-label";
    label.textContent = message.kind === "user" ? "你" : message.kind === "error" ? "这次请求未完成" : "医生";
    const text = document.createElement("div");
    text.className = "message-text";
    text.textContent = message.text;
    content.append(label, text);
    if (message.sources?.length) content.append(sourcePanel(message.sources, message.query));
    wrapper.append(avatar, content);
    return wrapper;
  }

  function scrollToEnd() { ui.conversation.scrollTop = ui.conversation.scrollHeight; }

  function renderTranscript() {
    const transcript = currentTranscript();
    ui.welcome.hidden = transcript.length > 0;
    ui.messages.replaceChildren(...transcript.map(messageElement));
    scrollToEnd();
  }

  function appendMessage(message) {
    currentTranscript().push(message);
    ui.welcome.hidden = true;
    ui.messages.append(messageElement(message));
    scrollToEnd();
  }

  function setBusy(busy) {
    state.busy = busy;
    syncControls();
  }

  ui["chat-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = ui["message-input"].value.trim();
    if (!question || !state.account || !state.role || !state.modelReady || state.busy || state.authBusy) return;
    notice();
    setBusy(true);
    appendMessage({ kind: "user", text: question });
    ui["message-input"].value = "";
    syncControls();
    ui.pending.hidden = false;
    scrollToEnd();
    try {
      const data = await request("chat", { method: "POST", body: JSON.stringify({ user_id: state.account.user_id, role_id: state.role.id, message: question }) });
      if (typeof data.answer !== "string" || !data.answer.trim()) throw new Error("模型没有返回可显示的回答，请查看模型服务日志。");
      appendMessage({ kind: "assistant", text: data.answer, sources: Array.isArray(data.sources) ? data.sources : [], query: data.rewritten_query });
      if (data.rerank_state === "unavailable") {
        notice("本次已使用混合检索；精排模型未就绪，请检查模型文件。来源资料仍可展开查看。");
      }
    } catch (error) {
      appendMessage({ kind: "error", text: error.message });
      ui["message-input"].value = question;
    } finally {
      ui.pending.hidden = true;
      setBusy(false);
      scrollToEnd();
      if (!ui["message-input"].disabled) ui["message-input"].focus();
      checkService();
    }
  });

  ui["message-input"].addEventListener("input", syncControls);
  ui["message-input"].addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      if (!ui["send-button"].disabled) ui["chat-form"].requestSubmit();
    }
  });

  ui.suggestions.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-question]");
    if (!button || state.busy || state.authBusy) return;
    ui["message-input"].value = button.dataset.question;
    if (!state.account) ui["account-dialog"].showModal();
    else ui["message-input"].focus();
    syncControls();
  });

  ui["clear-button"].addEventListener("click", async () => {
    if (!state.account || !state.role || state.busy || state.authBusy) return;
    setBusy(true);
    notice("正在清空对话记忆…");
    try {
      const params = new URLSearchParams({ user_id: state.account.user_id, role_id: state.role.id });
      await request(`history?${params}`, { method: "DELETE" });
      currentTranscript().length = 0;
      renderTranscript();
      notice("对话和服务端记忆已清空，可以开始新的问题。");
    } catch (error) { notice(`未能清空：${error.message}`); }
    finally { setBusy(false); }
  });

  function authMode(mode) {
    if (state.authBusy) return;
    state.mode = mode;
    for (const name of ["login", "register"]) {
      ui[`${name}-tab`].classList.toggle("active", name === mode);
      ui[`${name}-tab`].setAttribute("aria-pressed", String(name === mode));
    }
    ui.password.autocomplete = mode === "login" ? "current-password" : "new-password";
    ui["auth-submit"].textContent = mode === "login" ? "登录并开始" : "注册并开始";
    ui["account-error"].hidden = true;
  }

  ui["login-tab"].addEventListener("click", () => authMode("login"));
  ui["register-tab"].addEventListener("click", () => authMode("register"));
  ui["account-button"].addEventListener("click", () => {
    if (state.busy || state.authBusy) return;
    ui["account-error"].hidden = true;
    ui["account-dialog"].showModal();
  });
  ui["close-dialog"].addEventListener("click", () => ui["account-dialog"].close());
  ui["account-dialog"].addEventListener("cancel", (event) => { if (state.authBusy) event.preventDefault(); });
  ui["account-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.busy || state.authBusy) return;
    const username = ui.username.value.trim();
    if (username.length < 2) {
      ui["account-error"].textContent = "用户名至少需要 2 个字符。";
      ui["account-error"].hidden = false;
      return;
    }
    state.authBusy = true;
    syncControls();
    ui["account-error"].hidden = true;
    ui["auth-submit"].textContent = "正在连接…";
    try {
      const account = await request(state.mode, { method: "POST", body: JSON.stringify({ username, password: ui.password.value }) }, 30000);
      if (!Number.isInteger(account.user_id) || typeof account.username !== "string") throw new Error("账号信息不完整，请检查后端响应。");
      state.account = account;
      ui.password.value = "";
      ui["account-label"].textContent = `${account.username} · 切换账号`;
      ui["account-dialog"].close();
      notice(state.mode === "register" ? "账号创建成功。问答服务就绪后即可提问。" : "登录成功。服务端可能保留近期记忆，如需重新开始，请点击“清空对话”。");
      renderTranscript();
    } catch (error) {
      ui["account-error"].textContent = error.message;
      ui["account-error"].hidden = false;
    } finally {
      state.authBusy = false;
      ui["auth-submit"].textContent = state.mode === "login" ? "登录并开始" : "注册并开始";
      syncControls();
      if (!ui["account-dialog"].open && !ui["message-input"].disabled) ui["message-input"].focus();
    }
  });

  ui["reload-roles"].addEventListener("click", checkService);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) checkService(); });
  setInterval(() => { if (!document.hidden) checkService(); }, 15000);
  syncControls();
  checkService();
})();
