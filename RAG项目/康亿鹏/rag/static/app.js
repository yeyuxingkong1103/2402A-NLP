// 前端交互逻辑：登录/注册/聊天/流式问答/历史
(function () {
  "use strict";

  // ---------- 工具函数 ----------
  const $ = (id) => document.getElementById(id);

  function getToken() { return localStorage.getItem("token"); }
  function setAuth(token) {
    localStorage.setItem("token", token);
  }
  function clearAuth() {
    localStorage.removeItem("token");
  }

  // 视图切换
  function showView(name) {
    ["login-view", "register-view", "chat-view"].forEach((v) => {
      $(v).hidden = (v !== name);
    });
  }

  // 全局提示条
  let toastTimer = null;
  function toast(msg, type) {
    const el = $("toast");
    el.textContent = msg;
    el.className = "toast" + (type === "success" ? " success" : "");
    el.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 3000);
  }

  // 带 JWT 的 fetch 封装；401 自动退出登录
  async function api(url, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (getToken()) headers["Authorization"] = "Bearer " + getToken();
    if (options.body && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const res = await fetch(url, Object.assign({}, options, { headers }));
    if (res.status === 401) {
      clearAuth();
      showView("login-view");
      throw new Error("登录已过期，请重新登录");
    }
    return res;
  }

  // ---------- 登录 ----------
  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = $("login-username").value.trim();
    const password = $("login-password").value;
    if (!username || !password) return;
    try {
      const res = await api("/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "登录失败");
      }
      const data = await res.json();
      setAuth(data.token);
      $("login-form").reset();
      await enterChat();
    } catch (err) {
      toast(err.message);
    }
  });

  // ---------- 注册 ----------
  $("register-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const username = $("reg-username").value.trim();
      const password = $("reg-password").value;
      if (!username || !password) return;
      const res = await api("/register", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "注册失败");
      }
      $("register-form").reset();
      showView("login-view");
      $("login-username").value = username;  // 预填刚注册的用户名，省去重新输入
      $("login-password").focus();  // 聚焦密码框，直接接着输密码
      toast("注册成功，请登录", "success");
    } catch (err) {
      toast(err.message);
    }
  });

  // 视图切换链接
  $("to-register").addEventListener("click", (e) => {
    e.preventDefault();
    showView("register-view");
  });
  $("to-login").addEventListener("click", (e) => {
    e.preventDefault();
    showView("login-view");
  });

  // ---------- 聊天视图 ----------
  let currentDomain = null;
  let currentRole = null;
  let streaming = false;

  // 角色名 -> 中文显示（未列出的直接用原名）
  const ROLE_LABELS = { doctor: "医生", teacher: "教师", admin: "专家" };

  async function loadRoles() {
    const res = await api("/roles");
    const data = await res.json();
    const sel = $("role-select");
    sel.innerHTML = "";
    (data.roles || []).forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r.name;
      opt.textContent = ROLE_LABELS[r.name] || r.name;
      sel.appendChild(opt);
    });
    currentRole = sel.value || null;
  }

  async function enterChat() {
    showView("chat-view");
    try {
      const res = await api("/domains");
      const data = await res.json();
      const sel = $("domain-select");
      sel.innerHTML = "";
      if (!data.domains || data.domains.length === 0) {
        sel.innerHTML = '<option value="">（无可访问领域）</option>';
        return;
      }
      data.domains.forEach((d) => {
        const opt = document.createElement("option");
        opt.value = d.domain;
        opt.textContent = d.domain;
        sel.appendChild(opt);
      });
      currentDomain = data.domains[0].domain;
      sel.value = currentDomain;
      await loadRoles();
      await loadHistory();
    } catch (err) {
      toast(err.message);
    }
  }

  // 领域切换
  $("domain-select").addEventListener("change", async () => {
    currentDomain = $("domain-select").value;
    if (currentDomain) await loadHistory();
    else $("messages").innerHTML = "";
  });

  // 提问对象切换
  $("role-select").addEventListener("change", () => {
    currentRole = $("role-select").value;
  });

  // 渲染历史
  async function loadHistory() {
    $("messages").innerHTML = "";
    try {
      const res = await api("/history?domain=" + encodeURIComponent(currentDomain));
      const data = await res.json();
      const hist = data.history || [];
      hist.forEach((m) => appendMessage(m.role, m.content));
    } catch (err) {
      toast(err.message);
    }
  }

  // 追加一条消息气泡，返回气泡 DOM
  function appendMessage(role, content) {
    const wrap = $("messages");
    const div = document.createElement("div");
    div.className = "msg " + (role === "user" ? "user" : "assistant");
    div.textContent = content || "";
    wrap.appendChild(div);
    wrap.scrollTop = wrap.scrollHeight;
    return div;
  }

  // 发送问题（流式）
  $("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (streaming) return;
    const question = $("question-input").value.trim();
    if (!question) return;
    if (!currentDomain) { toast("请先选择领域"); return; }
    if (!currentRole) { toast("请先选择提问对象"); return; }

    $("question-input").value = "";
    autoResize();
    appendMessage("user", question);
    const bubble = appendMessage("assistant", "");
    bubble.classList.add("streaming");
    streaming = true;
    $("send-btn").disabled = true;

    try {
      const res = await api("/ask", {
        method: "POST",
        body: JSON.stringify({ question, domain: currentDomain, role: currentRole }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "请求失败");
      }
      // 流式读取 text/plain
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        bubble.textContent = buf;
        $("messages").scrollTop = $("messages").scrollHeight;
      }
    } catch (err) {
      bubble.textContent = "（出错）" + err.message;
      toast(err.message);
    } finally {
      bubble.classList.remove("streaming");
      streaming = false;
      $("send-btn").disabled = false;
    }
  });

  // textarea 自适应高度 + 回车发送
  function autoResize() {
    const ta = $("question-input");
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
  }
  $("question-input").addEventListener("input", autoResize);
  $("question-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  // 清空历史
  $("clear-btn").addEventListener("click", async () => {
    if (!currentDomain) return;
    if (!confirm("确定清空当前领域的历史记录？")) return;
    try {
      const res = await api("/clear-history", {
        method: "POST",
        body: JSON.stringify({ domain: currentDomain }),
      });
      if (!res.ok) throw new Error("清空失败");
      $("messages").innerHTML = "";
      toast("历史已清空", "success");
    } catch (err) {
      toast(err.message);
    }
  });

  // 退出
  $("logout-btn").addEventListener("click", () => {
    clearAuth();
    $("messages").innerHTML = "";
    showView("login-view");
  });

  // ---------- 启动 ----------
  (function init() {
    if (getToken()) enterChat();
    else showView("login-view");
  })();
})();
