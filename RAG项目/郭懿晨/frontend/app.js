(() => {
  const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
  const ROUTES = new Set(["upload", "tasks", "documents", "vector-store", "chat"]);
  const STEP_SEQUENCE = [
    { key: "parsing", label: "解析 PARSING", description: "MinerU 解析 PDF" },
    { key: "cleaning", label: "清洗 CLEANING", description: "文档清洗" },
    { key: "chunking", label: "分块 CHUNKING", description: "语义分块" },
    { key: "embedding", label: "向量化 EMBEDDING", description: "生成向量" },
    { key: "upserting", label: "入库 UPSERTING", description: "写入 Qdrant" },
    { key: "completed", label: "完成 COMPLETED", description: "构建完成" },
  ];
  const STEP_INDEX = STEP_SEQUENCE.reduce((map, step, index) => {
    map.set(step.key, index);
    return map;
  }, new Map());
  const LOCAL_KEYS = {
    recentTasks: "neonask.recentTasks",
    conversations: "neonask.conversations",
    activeConversationId: "neonask.activeConversationId",
  };
  const API = {
    health: "/health",
    upload: "/api/files/upload",
    task: (taskId) => `/api/tasks/${encodeURIComponent(taskId)}`,
    startTask: (taskId) => `/api/tasks/${encodeURIComponent(taskId)}/start`,
    documents: "/api/documents",
    vectors: "/api/vector-store",
    categories: "/api/categories",
    chat: "/api/chat",
  };
  const EMPTY_ARTS = {
    upload: `
      <svg viewBox="0 0 120 120" aria-hidden="true">
        <path d="M26 34h26l10 10h28v42H26z" />
        <path d="M52 24v28M44 32l8-8 8 8" />
        <path d="M40 72h40M40 84h22" />
        <path d="M82 28v16h16" />
      </svg>
    `,
    documents: `
      <svg viewBox="0 0 120 120" aria-hidden="true">
        <rect x="30" y="30" width="42" height="56" rx="0" />
        <path d="M40 42h22M40 54h26M40 66h18" />
        <path d="M68 38h14l8 8v40H60" />
        <path d="M82 38v8h8" />
      </svg>
    `,
    vectors: `
      <svg viewBox="0 0 120 120" aria-hidden="true">
        <path d="M30 46 60 30l30 16-30 16z" />
        <path d="M30 46v28l30 16 30-16V46" />
        <path d="M60 62v28" />
        <path d="M42 52 60 62l18-10" />
      </svg>
    `,
    default: `
      <svg viewBox="0 0 120 120" aria-hidden="true">
        <circle cx="60" cy="60" r="28" />
        <path d="M60 32v56M32 60h56" />
      </svg>
    `,
  };

  const el = {
    connectionStatus: document.getElementById("connectionStatus"),
    statTask: document.getElementById("statTask"),
    statDoc: document.getElementById("statDoc"),
    statVec: document.getElementById("statVec"),
    topNav: document.getElementById("topNav"),
    routeViews: Array.from(document.querySelectorAll(".view")),
    navItems: Array.from(document.querySelectorAll(".nav-item")),
    dropzone: document.getElementById("dropzone"),
    uploadFile: document.getElementById("uploadFile"),
    pickFileBtn: document.getElementById("pickFileBtn"),
    uploadBtn: document.getElementById("uploadBtn"),
    pickedFileName: document.getElementById("pickedFileName"),
    uploadProgressFill: document.getElementById("uploadProgressFill"),
    uploadProgressText: document.getElementById("uploadProgressText"),
    uploadResultBody: document.getElementById("uploadResultBody"),
    uploadActionBar: document.getElementById("uploadActionBar"),
    uploadLaunchBtn: document.getElementById("uploadLaunchBtn"),
    uploadLaterBtn: document.getElementById("uploadLaterBtn"),
    recentUploadList: document.getElementById("recentUploadList"),
    taskProgressFill: document.getElementById("taskProgressFill"),
    taskProgressText: document.getElementById("taskProgressText"),
    taskSteps: document.getElementById("taskSteps"),
    taskStatusLine: document.getElementById("taskStatusLine"),
    taskError: document.getElementById("taskError"),
    refreshRecentTasksBtn: document.getElementById("refreshRecentTasksBtn"),
    recentTaskList: document.getElementById("recentTaskList"),
    documentCount: document.getElementById("documentCount"),
    refreshDocumentsBtn: document.getElementById("refreshDocumentsBtn"),
    documentsEmpty: document.getElementById("documentsEmpty"),
    documentGrid: document.getElementById("documentGrid"),
    vectorSummary: document.getElementById("vectorSummary"),
    refreshVectorsBtn: document.getElementById("refreshVectorsBtn"),
    vectorSearchInput: document.getElementById("vectorSearchInput"),
    vectorCategoryChips: document.getElementById("vectorCategoryChips"),
    vectorLimitNote: document.getElementById("vectorLimitNote"),
    vectorRecordList: document.getElementById("vectorRecordList"),
    newConversationBtn: document.getElementById("newConversationBtn"),
    conversationList: document.getElementById("conversationList"),
    copyLastAnswerBtn: document.getElementById("copyLastAnswerBtn"),
    clearConversationBtn: document.getElementById("clearConversationBtn"),
    chatStatus: document.getElementById("chatStatus"),
    chatStream: document.getElementById("chatStream"),
    chatForm: document.getElementById("chatForm"),
    chatInput: document.getElementById("chatInput"),
    chatSendBtn: document.getElementById("chatSendBtn"),
    citationPanel: document.getElementById("citationPanel"),
  };

  let taskPoller = null;

  const state = {
    route: "upload",
    selectedFile: null,
    documents: [],
    documentMap: new Map(),
    vectorRecords: [],
    categories: [],
    selectedCategory: "all",
    vectorSearch: "",
    recentTasks: loadRecentTasks(),
    conversations: loadConversations(),
    activeConversationId: readJSON(LOCAL_KEYS.activeConversationId, null),
    currentTaskId: "",
    currentTask: null,
    chatBusy: false,
  };

  function uid(prefix = "id") {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${prefix}_${window.crypto.randomUUID()}`;
    }
    return `${prefix}_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
  }

  function readJSON(key, fallback) {
    try {
      const raw = window.localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch {
      return fallback;
    }
  }

  function writeJSON(key, value) {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      return;
    }
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => {
      switch (char) {
        case "&":
          return "&amp;";
        case "<":
          return "&lt;";
        case ">":
          return "&gt;";
        case '"':
          return "&quot;";
        case "'":
          return "&#39;";
        default:
          return char;
      }
    });
  }

  function truncate(value, length = 120) {
    const text = String(value ?? "");
    if (text.length <= length) {
      return text;
    }
    return `${text.slice(0, length).trimEnd()}…`;
  }

  function shortId(value, length = 8) {
    const text = String(value ?? "");
    return text ? text.slice(0, length) : "-";
  }

  function formatBytes(bytes) {
    const size = Number(bytes || 0);
    if (!Number.isFinite(size) || size <= 0) {
      return "0 B";
    }
    const units = ["B", "KB", "MB", "GB"];
    let index = 0;
    let amount = size;
    while (amount >= 1024 && index < units.length - 1) {
      amount /= 1024;
      index += 1;
    }
    return `${amount.toFixed(amount >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
  }

  function formatDate(value) {
    if (!value) {
      return "-";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "-";
    }
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(date);
  }

  function formatClock(value) {
    if (!value) {
      return "-";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "-";
    }
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }

  function normalizeRoute(hash = window.location.hash) {
    const raw = hash.replace(/^#\/?/, "").split("?")[0].split("&")[0].trim();
    return ROUTES.has(raw) ? raw : "upload";
  }

  function applyRoute(route) {
    const nextRoute = ROUTES.has(route) ? route : "upload";
    state.route = nextRoute;

    el.routeViews.forEach((view) => {
      view.classList.toggle("is-active", view.dataset.route === nextRoute);
    });

    el.navItems.forEach((button) => {
      const active = button.dataset.route === nextRoute;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-current", active ? "page" : "false");
    });

    const titleMap = {
      upload: "传送门 UPLINK",
      tasks: "构建协议",
      documents: "档案库",
      "vector-store": "数据矩阵",
      chat: "中枢问答",
    };
    document.title = `霓问 NEON-ASK · ${titleMap[nextRoute] || "控制台"}`;
  }

  function navigate(route, { replace = false } = {}) {
    const nextRoute = ROUTES.has(route) ? route : "upload";
    const nextHash = `#/${nextRoute}`;
    if (replace) {
      window.history.replaceState(null, "", nextHash);
    } else if (window.location.hash !== nextHash) {
      window.location.hash = nextHash;
    }
    applyRoute(nextRoute);
  }

  function syncRouteFromHash() {
    const route = normalizeRoute();
    if (!window.location.hash) {
      navigate(route, { replace: true });
      return;
    }
    applyRoute(route);
  }

  function setConnectionStatus(kind, label) {
    if (!el.connectionStatus) {
      return;
    }
    el.connectionStatus.classList.remove("status-online", "status-warning", "status-danger");
    const className = {
      online: "status-online",
      warning: "status-warning",
      danger: "status-danger",
    }[kind] || "status-warning";
    el.connectionStatus.classList.add(className);
    el.connectionStatus.textContent = label;
  }

  function setTaskStatusLine(text, kind = "live") {
    if (!el.taskStatusLine) {
      return;
    }
    el.taskStatusLine.classList.remove("is-live", "is-ok", "is-error");
    el.taskStatusLine.classList.add({
      live: "is-live",
      ok: "is-ok",
      error: "is-error",
    }[kind] || "is-live");
    el.taskStatusLine.textContent = text;
  }

  function setChatStatus(text, kind = "live") {
    if (!el.chatStatus) {
      return;
    }
    el.chatStatus.classList.remove("is-live", "is-ok", "is-error");
    el.chatStatus.classList.add({
      live: "is-live",
      ok: "is-ok",
      error: "is-error",
    }[kind] || "is-live");
    el.chatStatus.textContent = text;
  }

  function setUploadProgress(percent, label) {
    if (el.uploadProgressFill) {
      el.uploadProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
    if (el.uploadProgressText) {
      el.uploadProgressText.textContent = label;
    }
  }

  function setTaskProgress(percent, label) {
    if (el.taskProgressFill) {
      el.taskProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
    if (el.taskProgressText) {
      el.taskProgressText.textContent = label;
    }
  }

  function loadRecentTasks() {
    const items = readJSON(LOCAL_KEYS.recentTasks, []);
    if (!Array.isArray(items)) {
      return [];
    }
    return items
      .filter((item) => item && typeof item === "object" && item.task_id)
      .map((item) => normalizeRecentTask(item))
      .sort(sortRecentTasks)
      .slice(0, 12);
  }

  function saveRecentTasks() {
    writeJSON(LOCAL_KEYS.recentTasks, state.recentTasks);
    updateStats();
  }

  function sortRecentTasks(a, b) {
    const aTime = new Date(a.updated_at || a.last_seen_at || a.created_at || 0).getTime();
    const bTime = new Date(b.updated_at || b.last_seen_at || b.created_at || 0).getTime();
    return bTime - aTime;
  }

  function normalizeRecentTask(task, documentMap = null) {
    const documentId = String(task.document_id || "");
    const documentMeta = documentMap?.get(documentId);
    const fallbackName = documentId ? shortId(documentId) : "未知文档";
    return {
      task_id: String(task.task_id || ""),
      document_id: documentId,
      file_name: String(task.file_name || documentMeta?.file_name || fallbackName),
      status: String(task.status || "pending"),
      step: String(task.step || "pending"),
      progress: Number(task.progress || 0),
      created_at: task.created_at || new Date().toISOString(),
      updated_at: task.updated_at || task.created_at || new Date().toISOString(),
      error_message: task.error_message || null,
      last_seen_at: task.last_seen_at || task.updated_at || task.created_at || new Date().toISOString(),
    };
  }

  function upsertRecentTask(task) {
    const normalized = normalizeRecentTask(task, state.documentMap);
    const existingIndex = state.recentTasks.findIndex((item) => item.task_id === normalized.task_id);
    if (existingIndex >= 0) {
      state.recentTasks[existingIndex] = {
        ...state.recentTasks[existingIndex],
        ...normalized,
        file_name: normalized.file_name || state.recentTasks[existingIndex].file_name,
      };
    } else {
      state.recentTasks.unshift(normalized);
    }
    state.recentTasks = state.recentTasks
      .map((item) => normalizeRecentTask(item, state.documentMap))
      .sort(sortRecentTasks)
      .slice(0, 12);
    saveRecentTasks();
    renderRecentTaskLists();
  }

  function loadConversations() {
    const items = readJSON(LOCAL_KEYS.conversations, []);
    if (!Array.isArray(items) || items.length === 0) {
      return [createConversation("新会话")];
    }
    return items
      .filter((item) => item && typeof item === "object" && item.id)
      .map((item) => ({
        id: String(item.id),
        title: String(item.title || "新会话"),
        createdAt: item.createdAt || new Date().toISOString(),
        updatedAt: item.updatedAt || item.createdAt || new Date().toISOString(),
        messages: Array.isArray(item.messages)
          ? item.messages
              .filter((message) => message && typeof message === "object" && message.role && !message.loading)
              .map((message) => ({
                role: String(message.role),
                content: String(message.content || ""),
                citations: Array.isArray(message.citations) ? message.citations : [],
                fallback: Boolean(message.fallback),
                loading: false,
                createdAt: message.createdAt || new Date().toISOString(),
              }))
          : [],
      }));
  }

  function saveConversations() {
    writeJSON(LOCAL_KEYS.conversations, state.conversations);
    writeJSON(LOCAL_KEYS.activeConversationId, state.activeConversationId);
  }

  function createConversation(title = "新会话") {
    return {
      id: uid("conv"),
      title,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: [],
    };
  }

  function getActiveConversation() {
    return state.conversations.find((conversation) => conversation.id === state.activeConversationId) || state.conversations[0] || null;
  }

  function ensureActiveConversation() {
    if (!state.conversations.length) {
      state.conversations.push(createConversation("新会话"));
    }
    if (!state.activeConversationId || !state.conversations.some((conversation) => conversation.id === state.activeConversationId)) {
      state.activeConversationId = state.conversations[0].id;
      saveConversations();
    }
  }

  function setActiveConversation(conversationId) {
    if (!state.conversations.some((conversation) => conversation.id === conversationId)) {
      return;
    }
    state.activeConversationId = conversationId;
    saveConversations();
    renderConversationList();
    renderChatConversation();
  }

  function renameConversationFromQuestion(conversation, question) {
    if (!conversation || conversation.title !== "新会话") {
      return;
    }
    const candidate = truncate(question.trim(), 16);
    conversation.title = candidate || "新会话";
  }

  function getCurrentDocumentName(documentId) {
    if (!documentId) {
      return "未知文档";
    }
    return state.documentMap.get(documentId)?.file_name || state.recentTasks.find((item) => item.document_id === documentId)?.file_name || shortId(documentId);
  }

  function normalizeVectorRecord(record) {
    const payload = record && typeof record === "object" && record.payload && typeof record.payload === "object" ? record.payload : record;
    return {
      id: String(record?.id || payload?.chunk_id || payload?.id || uid("chunk")),
      chunk_id: String(payload?.chunk_id || record?.chunk_id || record?.id || ""),
      document_id: String(payload?.document_id || record?.document_id || ""),
      page: Number(payload?.page || record?.page || 0),
      category: String(payload?.category || record?.category || "未分类"),
      text: String(payload?.text || record?.text || ""),
      source_span: String(payload?.source_span || record?.source_span || ""),
      file_name: getCurrentDocumentName(String(payload?.document_id || record?.document_id || "")),
    };
  }

  function inferFailedStepIndex(progress) {
    const value = Number(progress || 0);
    if (value >= 90) {
      return STEP_INDEX.get("upserting") ?? 4;
    }
    if (value >= 75) {
      return STEP_INDEX.get("embedding") ?? 3;
    }
    if (value >= 60) {
      return STEP_INDEX.get("chunking") ?? 2;
    }
    if (value >= 45) {
      return STEP_INDEX.get("cleaning") ?? 1;
    }
    return STEP_INDEX.get("parsing") ?? 0;
  }

  function getTaskDescriptor(task) {
    const status = String(task?.status || "pending");
    const step = String(task?.step || "pending");
    const progress = Number(task?.progress || 0);
    if (!task) {
      return {
        status,
        statusLabel: "pending / 待机",
        statusClass: "task-status-pending",
        stepLabel: "pending / 待机",
        progress: 0,
        currentIndex: -1,
        note: "等待任务启动。",
      };
    }
    if (status === "completed") {
      return {
        status,
        statusLabel: "completed / 已完成",
        statusClass: "task-status-completed",
        stepLabel: "completed / 完成",
        progress: 100,
        currentIndex: STEP_SEQUENCE.length - 1,
        note: "BUILD COMPLETE",
      };
    }
    if (status === "failed") {
      const currentIndex = inferFailedStepIndex(progress);
      return {
        status,
        statusLabel: "failed / 失败",
        statusClass: "task-status-failed",
        stepLabel: `${STEP_SEQUENCE[currentIndex].label} / 失败`,
        progress,
        currentIndex,
        note: "构建失败，当前步骤异常。",
      };
    }
    if (status === "running") {
      const currentIndex = STEP_INDEX.has(step) ? STEP_INDEX.get(step) : 0;
      return {
        status,
        statusLabel: "running / 进行中",
        statusClass: "task-status-running",
        stepLabel: STEP_SEQUENCE[currentIndex]?.label || step,
        progress,
        currentIndex,
        note: "正在处理，请保持连接。",
      };
    }
    return {
      status,
      statusLabel: "pending / 待机",
      statusClass: "task-status-pending",
      stepLabel: "pending / 待机",
      progress,
      currentIndex: -1,
      note: "等待任务启动。",
    };
  }

  function getTaskStepClass(task, index, descriptor) {
    if (!task) {
      return "";
    }
    if (descriptor.status === "completed" || (descriptor.status === "running" && index < descriptor.currentIndex) || (descriptor.status === "failed" && index < descriptor.currentIndex)) {
      return "is-done";
    }
    if (descriptor.status === "running" && index === descriptor.currentIndex) {
      return "is-active";
    }
    if (descriptor.status === "failed" && index === descriptor.currentIndex) {
      return "is-error";
    }
    return "";
  }

  function getTaskStepState(task, index, descriptor) {
    if (!task) {
      return "待机";
    }
    if (descriptor.status === "completed" || index < descriptor.currentIndex) {
      return "已通过";
    }
    if (descriptor.status === "running" && index === descriptor.currentIndex) {
      return "进行中";
    }
    if (descriptor.status === "failed" && index === descriptor.currentIndex) {
      return "异常";
    }
    return "待机";
  }

  function parseResponseError(body) {
    if (!body) {
      return "请求失败";
    }
    if (typeof body === "string") {
      const trimmed = body.trim();
      if (!trimmed) {
        return "请求失败";
      }
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed && typeof parsed === "object") {
          return String(parsed.detail || parsed.message || trimmed);
        }
      } catch {
        return trimmed;
      }
      return trimmed;
    }
    if (typeof body === "object") {
      return String(body.detail || body.message || JSON.stringify(body));
    }
    return "请求失败";
  }

  async function requestJson(url, options = {}) {
    const {
      method = "GET",
      headers = {},
      body,
      timeoutMs = 30000,
    } = options;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    const finalHeaders = new Headers(headers);
    let finalBody = body;
    if (body && typeof body === "object" && !(body instanceof FormData) && !(body instanceof Blob) && !(body instanceof URLSearchParams) && !(body instanceof ArrayBuffer)) {
      if (!finalHeaders.has("Content-Type")) {
        finalHeaders.set("Content-Type", "application/json");
      }
      finalBody = JSON.stringify(body);
    }
    try {
      const response = await fetch(url, {
        method,
        headers: finalHeaders,
        body: finalBody,
        signal: controller.signal,
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(parseResponseError(errorText));
      }
      if (response.status === 204) {
        return null;
      }
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        return await response.json();
      }
      const text = await response.text();
      try {
        return JSON.parse(text);
      } catch {
        return text;
      }
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("请求超时，请稍后重试。");
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function uploadWithProgress(file, onProgress) {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", API.upload);
      xhr.responseType = "json";
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && typeof onProgress === "function") {
          onProgress(event.loaded, event.total);
        }
      };
      xhr.onload = () => {
        const payload = xhr.response ?? safeParseJson(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(payload);
          return;
        }
        reject(new Error(parseResponseError(payload)));
      };
      xhr.onerror = () => reject(new Error("上传失败，请检查网络连接。"));
      xhr.onabort = () => reject(new Error("上传已取消。"));
      xhr.send(formData);
    });
  }

  function safeParseJson(text) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  function renderEmptyArts() {
    document.querySelectorAll(".empty-art").forEach((node) => {
      const kind = node.dataset.empty || "default";
      node.innerHTML = EMPTY_ARTS[kind] || EMPTY_ARTS.default;
    });
  }

  function renderUploadSelection(file) {
    state.selectedFile = file || null;
    if (file) {
      el.pickedFileName.textContent = `${file.name} · ${formatBytes(file.size)}`;
      el.uploadBtn.disabled = false;
    } else {
      el.pickedFileName.textContent = "尚未选择文件";
      el.uploadBtn.disabled = true;
    }
    el.uploadActionBar.classList.add("is-hidden");
    el.uploadResultBody.textContent = "尚未上传任何 PDF。";
    setUploadProgress(0, "等待文件接入");
  }

  function validatePdfFile(file) {
    if (!file) {
      return "请选择一个 PDF 文件。";
    }
    const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name || "");
    if (!isPdf) {
      return "仅支持 PDF 文件。";
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      return "文件超过 100MB，请先压缩后再上传。";
    }
    return "";
  }

  function renderTaskSnapshot(task) {
    state.currentTask = task || null;
    state.currentTaskId = task?.task_id || "";

    const descriptor = getTaskDescriptor(task);
    if (!task) {
      setTaskProgress(0, "请先上传并启动构建。");
      renderTaskSteps(null);
      setTaskStatusLine("等待任务信号。", "live");
      el.taskError.textContent = "暂无错误。";
      updateStats();
      renderRecentTaskLists();
      return;
    }

    const fileName = getCurrentDocumentName(task.document_id);
    setTaskProgress(Number(task.progress || 0), `${descriptor.stepLabel} · 进度 ${Number(task.progress || 0)}%`);
    renderTaskSteps(task);
    setTaskStatusLine(descriptor.note, descriptor.status === "failed" ? "error" : descriptor.status === "completed" ? "ok" : "live");
    el.taskError.textContent = task.error_message || "暂无错误。";

    upsertRecentTask({
      task_id: task.task_id,
      document_id: task.document_id,
      file_name: fileName,
      status: task.status,
      step: task.step,
      progress: task.progress,
      created_at: task.created_at,
      updated_at: task.updated_at,
      error_message: task.error_message,
      last_seen_at: new Date().toISOString(),
    });
    updateStats();
  }

  function renderTaskSteps(task) {
    const descriptor = getTaskDescriptor(task);
    const markup = STEP_SEQUENCE.map((step, index) => {
      const stateClass = getTaskStepClass(task, index, descriptor);
      const stateLabel = getTaskStepState(task, index, descriptor);
      return `
        <div class="task-step ${stateClass}">
          <div class="index">${index + 1}</div>
          <div>
            <div class="label">${escapeHtml(step.label)}</div>
            <div class="desc">${escapeHtml(step.description)}</div>
          </div>
          <div class="state">${escapeHtml(stateLabel)}</div>
        </div>
      `;
    }).join("");
    el.taskSteps.innerHTML = markup;
  }

  function renderRecentTaskLists() {
    renderRecentTasksInto(el.recentUploadList, 4);
    renderRecentTasksInto(el.recentTaskList, 8);
  }

  function renderRecentTasksInto(container, limit) {
    if (!container) {
      return;
    }
    const items = state.recentTasks.slice(0, limit);
    if (!items.length) {
      container.innerHTML = `
        <div class="mini-item">
          <div class="title">暂无最近任务</div>
          <div class="meta">上传 PDF 后，任务会自动写入本地历史。</div>
        </div>
      `;
      return;
    }
    container.innerHTML = items
      .map((task) => {
        const descriptor = getTaskDescriptor(task);
        const statusText = descriptor.statusLabel;
        const timeText = formatClock(task.updated_at || task.created_at);
        const progressText = `${Number(task.progress || 0)}% · ${STEP_SEQUENCE[descriptor.currentIndex] ? STEP_SEQUENCE[descriptor.currentIndex].label : descriptor.stepLabel}`;
        return `
          <button class="mini-item" type="button" data-task-id="${escapeHtml(task.task_id)}">
            <div class="row">
              <div class="title">${escapeHtml(task.file_name || "未命名文档")}</div>
              <span class="status-badge ${descriptor.statusClass}">${escapeHtml(statusText)}</span>
            </div>
            <div class="meta">${escapeHtml(progressText)} · ${escapeHtml(timeText)}</div>
          </button>
        `;
      })
      .join("");
  }

  function renderDocuments() {
    const items = [...state.documents].sort((a, b) => {
      const aTime = new Date(a.created_at || 0).getTime();
      const bTime = new Date(b.created_at || 0).getTime();
      return bTime - aTime;
    });
    el.documentCount.textContent = `${items.length} DOCUMENTS`;
    el.documentsEmpty.classList.toggle("is-hidden", items.length > 0);
    el.documentGrid.innerHTML = items
      .map((document) => {
        const statusClass = {
          uploaded: "status-uploaded",
          indexing: "status-indexing",
          indexed: "status-indexed",
          failed: "status-failed",
        }[document.status] || "status-uploaded";
        const recentTask = state.recentTasks.find((task) => task.document_id === document.document_id);
        return `
          <button class="doc-item" type="button" data-document-id="${escapeHtml(document.document_id)}">
            <div class="head">
              <h4>${escapeHtml(document.file_name)}</h4>
              <div class="doc-meta">
                <span>Doc ${escapeHtml(shortId(document.document_id))}</span>
                <span>${escapeHtml(formatDate(document.created_at))}</span>
              </div>
            </div>
            <div class="doc-meta">
              <span class="status-badge ${statusClass}">${escapeHtml(document.status)}</span>
              <span>${recentTask ? "已关联构建记录" : "未找到本地构建记录"}</span>
            </div>
            <div class="doc-actions">
              <span class="panel-kicker">${recentTask ? "点击查看构建协议" : "仅文档记录"}</span>
            </div>
          </button>
        `;
      })
      .join("");
    updateStats();
  }

  function renderVectorCategoryChips() {
    const categories = ["all", ...state.categories.filter(Boolean)];
    el.vectorCategoryChips.innerHTML = categories
      .map((category) => {
        const active = state.selectedCategory === category;
        const label = category === "all" ? "全部" : category;
        return `<button class="chip ${active ? "is-active" : ""}" type="button" data-category="${escapeHtml(category)}">${escapeHtml(label)}</button>`;
      })
      .join("");
  }

  function renderVectorRecords() {
    const search = state.vectorSearch.trim().toLowerCase();
    const records = state.vectorRecords
      .filter((record) => {
        const matchCategory = state.selectedCategory === "all" || record.category === state.selectedCategory;
        const matchSearch = !search || [record.text, record.category, record.source_span, record.document_id, record.file_name]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(search));
        return matchCategory && matchSearch;
      })
      .sort((a, b) => a.page - b.page || a.file_name.localeCompare(b.file_name, "zh-CN"));

    const total = state.vectorRecords.length;
    const filtered = records.length;
    el.vectorSummary.textContent = `${filtered} / ${total} RECORDS`;
    el.vectorLimitNote.textContent = total >= 1000 ? "数据量较大，仅显示前 1000 条。" : total ? `已加载 ${total} 条记录。` : "尚未加载向量库。";

    if (!records.length) {
      el.vectorRecordList.innerHTML = `
        <div class="empty-state compact">
          <div class="empty-art" data-empty="vectors"></div>
          <h3>暂无匹配分块</h3>
          <p>尝试切换类别，或者换一个更具体的关键词。</p>
        </div>
      `;
      renderEmptyArts();
      return;
    }

    el.vectorRecordList.innerHTML = records
      .map((record, index) => {
        const title = record.file_name || shortId(record.document_id);
        const snippet = truncate(record.text, 260);
        return `
          <details class="vector-card">
            <summary>
              <div class="summary-line">
                <span class="chip">EVIDENCE-${String(index + 1).padStart(2, "0")}</span>
                <span class="status-badge status-indexed">${escapeHtml(record.category || "未分类")}</span>
                <span class="status-badge">${escapeHtml(`P${record.page || "-"}`)}</span>
                <span class="vector-meta">${escapeHtml(title)}</span>
              </div>
              <p class="snippet">${escapeHtml(snippet)}</p>
            </summary>
            <div class="detail-block">
              <div class="vector-meta">
                <span>Doc ${escapeHtml(shortId(record.document_id))}</span>
                <span>${escapeHtml(record.source_span || "source_span 未提供")}</span>
              </div>
              <p class="snippet">${escapeHtml(record.text || "暂无原文片段。")}</p>
            </div>
          </details>
        `;
      })
      .join("");
  }

  function renderConversationList() {
    ensureActiveConversation();
    el.conversationList.innerHTML = state.conversations
      .map((conversation) => {
        const active = conversation.id === state.activeConversationId;
        return `
          <div class="conversation-item ${active ? "is-active" : ""}" tabindex="0" role="button" data-conversation-id="${escapeHtml(conversation.id)}">
            <div class="row">
              <div class="title">${escapeHtml(conversation.title || "新会话")}</div>
              <div class="actions">
                <button class="tiny-btn" type="button" data-action="delete-conversation" data-conversation-id="${escapeHtml(conversation.id)}">删除</button>
              </div>
            </div>
            <div class="meta">${conversation.messages.length} 条消息 · ${escapeHtml(formatDate(conversation.updatedAt || conversation.createdAt))}</div>
          </div>
        `;
      })
      .join("");
  }

  function renderChatConversation() {
    const conversation = getActiveConversation();
    if (!conversation) {
      el.chatStream.innerHTML = "";
      el.citationPanel.innerHTML = "";
      setChatStatus("等待输入问题。", "live");
      return;
    }
    const loadingAssistant = [...conversation.messages].reverse().find((message) => message.role === "assistant" && message.loading);
    if (loadingAssistant) {
      el.chatStream.innerHTML = conversation.messages
        .map((message) => renderChatMessage(message))
        .join("");
      el.citationPanel.innerHTML = `
        <div class="citation-title">证据引用</div>
        <div class="muted">正在等待模型返回结构化引用。</div>
      `;
      setChatStatus("正在接入本地神经中枢 OLLAMA…", "live");
      el.chatStream.scrollTop = el.chatStream.scrollHeight;
      return;
    }
    if (!conversation.messages.length) {
      el.chatStream.innerHTML = `
        <div class="message message-assistant">
          <div class="message-head">
            <span>NEON CORE</span>
            <span>${escapeHtml(formatClock(new Date().toISOString()))}</span>
          </div>
          <div class="bubble">
            <p>输入问题，系统会调用本地 Ollama 并返回可追溯引用。</p>
            <p>你可以把答案复制到别处，也可以切换会话继续追问。</p>
          </div>
        </div>
      `;
      el.citationPanel.innerHTML = `
        <div class="citation-title">证据引用</div>
        <div class="muted">等待首个回答。</div>
      `;
      setChatStatus("等待输入问题。", "live");
      return;
    }
    el.chatStream.innerHTML = conversation.messages
      .map((message) => renderChatMessage(message))
      .join("");
    const lastAssistant = [...conversation.messages].reverse().find((message) => message.role === "assistant" && !message.loading);
    if (lastAssistant) {
      renderCitationPanel(lastAssistant.citations || [], Boolean(lastAssistant.fallback));
      setChatStatus(lastAssistant.fallback ? "知识库中未找到相关依据。请先上传并构建文档。" : "回答已返回。", lastAssistant.fallback ? "error" : "ok");
    } else {
      el.citationPanel.innerHTML = `
        <div class="citation-title">证据引用</div>
        <div class="muted">等待首个回答。</div>
      `;
      setChatStatus("等待输入问题。", "live");
    }
    el.chatStream.scrollTop = el.chatStream.scrollHeight;
  }

  function renderChatMessage(message) {
    const createdAt = formatClock(message.createdAt || new Date().toISOString());
    const isUser = message.role === "user";
    const isLoading = Boolean(message.loading);
    const isAssistant = message.role === "assistant";
    let bubbleContent = "";
    if (isLoading) {
      bubbleContent = `
        <p>正在接入本地神经中枢 OLLAMA… <span class="loading-dots"><span></span><span></span><span></span></span></p>
      `;
    } else {
      bubbleContent = `<div class="bubble">${markdownToHtml(message.content || "")}</div>`;
    }
    const citationSummary = isAssistant && !isLoading ? renderMessageCitationSummary(message.citations || []) : "";
    return `
      <article class="message ${isUser ? "message-user" : "message-assistant"} ${isLoading ? "loading" : ""} ${message.fallback ? "message-warning" : ""}">
        <div class="message-head">
          <span>${isUser ? "USER" : isAssistant ? "ASSISTANT" : "SYSTEM"}</span>
          <span>${escapeHtml(createdAt)}</span>
        </div>
        ${bubbleContent}
        ${citationSummary}
      </article>
    `;
  }

  function renderMessageCitationSummary(citations) {
    const items = Array.isArray(citations) ? citations : [];
    if (!items.length) {
      return "";
    }
    return `
      <div class="message-citations">
        ${items
          .map((citation, index) => {
            const fileName = String(citation.file_name || "未知文档");
            const page = Number(citation.page || 0);
            const scoreValue = citation.score === null || citation.score === undefined ? Number.NaN : Number(citation.score);
            const scoreLabel = Number.isFinite(scoreValue) ? scoreValue.toFixed(3) : "-";
            return `
              <div class="message-citation">
                <div class="message-citation-head">
                  <span class="chip">引用 ${String(index + 1)}</span>
                  <span class="status-badge citation-score">检索得分 ${escapeHtml(scoreLabel)}</span>
                  <span class="status-badge">P${escapeHtml(page || "-")}</span>
                  <span class="status-badge">${escapeHtml(citation.category || "未分类")}</span>
                </div>
                <div class="message-citation-source">${escapeHtml(fileName)}</div>
                <div class="message-citation-text">${escapeHtml(truncate(citation.text || "暂无原文片段。", 180))}</div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function renderCitationPanel(citations, fallback) {
    if (!citations || !citations.length) {
      el.citationPanel.innerHTML = fallback
        ? `
          <div class="citation-title">证据引用</div>
          <div class="empty-state compact">
            <div class="empty-art" data-empty="documents"></div>
            <h3>知识库中未找到相关依据。</h3>
            <p>请先上传并构建文档，然后重新提问。</p>
          </div>
        `
        : `
          <div class="citation-title">证据引用</div>
          <div class="muted">本次回答未返回结构化引用。</div>
        `;
      renderEmptyArts();
      return;
    }
    el.citationPanel.innerHTML = `
      <div class="citation-title">证据引用</div>
      <div class="citation-list">
        ${citations
          .map((citation, index) => {
            const fileName = String(citation.file_name || "未知文档");
            const page = Number(citation.page || 0);
            const scoreValue = citation.score === null || citation.score === undefined ? Number.NaN : Number(citation.score);
            const hasScore = Number.isFinite(scoreValue);
            const scoreLabel = hasScore ? `检索得分 ${scoreValue.toFixed(3)}` : "检索得分 -";
            return `
              <details class="citation-card" open>
                <summary>
                  <div class="citation-head">
                    <span class="chip">EVIDENCE-${String(index + 1).padStart(2, "0")}</span>
                    <span class="status-badge status-indexed">${escapeHtml(citation.category || "未分类")}</span>
                    <span class="status-badge">P${escapeHtml(page || "-")}</span>
                    <span class="status-badge citation-score">${escapeHtml(scoreLabel)}</span>
                    <span class="citation-file">${escapeHtml(fileName)}</span>
                  </div>
                  <div class="citation-meta">
                    <span>Doc ${escapeHtml(shortId(citation.document_id))}</span>
                  </div>
                </summary>
                <div class="citation-text">${escapeHtml(citation.text || "暂无原文片段。")}</div>
              </details>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function markdownToHtml(markdown) {
    const source = String(markdown || "").replace(/\r\n/g, "\n");
    if (!source.trim()) {
      return "";
    }
    const lines = source.split("\n");
    const html = [];
    let paragraph = [];
    let listType = null;
    let codeFence = false;
    let codeLines = [];

    const flushParagraph = () => {
      if (!paragraph.length) {
        return;
      }
      html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    };

    const flushList = () => {
      if (!listType) {
        return;
      }
      html.push(`</${listType}>`);
      listType = null;
    };

    const flushCode = () => {
      if (!codeFence) {
        return;
      }
      html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      codeFence = false;
      codeLines = [];
    };

    for (const rawLine of lines) {
      const line = rawLine.trimEnd();
      const trimmed = line.trim();
      if (trimmed.startsWith("```")) {
        if (codeFence) {
          flushCode();
        } else {
          flushParagraph();
          flushList();
          codeFence = true;
        }
        continue;
      }

      if (codeFence) {
        codeLines.push(rawLine);
        continue;
      }

      if (!trimmed) {
        flushParagraph();
        flushList();
        continue;
      }

      const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
      if (headingMatch) {
        flushParagraph();
        flushList();
        const level = headingMatch[1].length;
        html.push(`<h${level}>${inlineMarkdown(headingMatch[2])}</h${level}>`);
        continue;
      }

      if (/^>\s+/.test(trimmed)) {
        flushParagraph();
        flushList();
        html.push(`<blockquote>${inlineMarkdown(trimmed.replace(/^>\s+/, ""))}</blockquote>`);
        continue;
      }

      const unorderedMatch = trimmed.match(/^[-*+]\s+(.*)$/);
      if (unorderedMatch) {
        flushParagraph();
        if (listType !== "ul") {
          flushList();
          html.push("<ul>");
          listType = "ul";
        }
        html.push(`<li>${inlineMarkdown(unorderedMatch[1])}</li>`);
        continue;
      }

      const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (orderedMatch) {
        flushParagraph();
        if (listType !== "ol") {
          flushList();
          html.push("<ol>");
          listType = "ol";
        }
        html.push(`<li>${inlineMarkdown(orderedMatch[1])}</li>`);
        continue;
      }

      paragraph.push(trimmed);
    }

    flushParagraph();
    flushList();
    flushCode();
    return html.join("");
  }

  function inlineMarkdown(text) {
    let output = escapeHtml(text);
    output = output.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    output = output.replace(/`([^`]+)`/g, "<code>$1</code>");
    output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    output = output.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return output;
  }

  function getLastAssistantAnswer() {
    const conversation = getActiveConversation();
    if (!conversation) {
      return "";
    }
    const assistantMessage = [...conversation.messages].reverse().find((message) => message.role === "assistant" && !message.loading);
    return assistantMessage?.content || "";
  }

  function updateStats() {
    el.statTask.textContent = String(state.recentTasks.length);
    el.statDoc.textContent = String(state.documents.length);
    el.statVec.textContent = String(state.vectorRecords.length);
  }

  function renderAllCountsAndLists() {
    renderRecentTaskLists();
    renderConversationList();
    renderChatConversation();
    renderTaskSteps(state.currentTask);
    updateStats();
  }

  async function checkHealth() {
    try {
      await requestJson(API.health, { timeoutMs: 10000 });
      setConnectionStatus("online", "API READY");
    } catch (error) {
      setConnectionStatus("warning", "API DEGRADED");
      console.warn(error);
    }
  }

  async function refreshDocuments() {
    try {
      const documents = await requestJson(API.documents, { timeoutMs: 15000 });
      state.documents = Array.isArray(documents) ? documents : [];
      state.documentMap = new Map(state.documents.map((item) => [String(item.document_id), item]));
      state.recentTasks = state.recentTasks.map((task) => normalizeRecentTask(task, state.documentMap));
      state.vectorRecords = state.vectorRecords.map((record) => ({
        ...record,
        file_name: getCurrentDocumentName(record.document_id),
      }));
      renderDocuments();
      renderRecentTaskLists();
      renderVectorRecords();
      updateStats();
      setConnectionStatus("online", "API READY");
    } catch (error) {
      setConnectionStatus("warning", "API DEGRADED");
      console.warn(error);
    }
  }

  async function refreshCategories() {
    try {
      const categories = await requestJson(API.categories, { timeoutMs: 15000 });
      state.categories = Array.isArray(categories) ? categories.map((category) => String(category)).filter(Boolean) : [];
      renderVectorCategoryChips();
      renderVectorRecords();
      setConnectionStatus("online", "API READY");
    } catch (error) {
      state.categories = [];
      renderVectorCategoryChips();
      renderVectorRecords();
      setConnectionStatus("warning", "API DEGRADED");
      console.warn(error);
    }
  }

  async function refreshVectors() {
    try {
      const rawRecords = await requestJson(API.vectors, { timeoutMs: 15000 });
      state.vectorRecords = Array.isArray(rawRecords) ? rawRecords.map(normalizeVectorRecord) : [];
      renderVectorRecords();
      updateStats();
      setConnectionStatus("online", "API READY");
    } catch (error) {
      setConnectionStatus("warning", "API DEGRADED");
      console.warn(error);
      if (!state.vectorRecords.length) {
        el.vectorRecordList.innerHTML = `
          <div class="empty-state compact">
            <div class="empty-art" data-empty="vectors"></div>
            <h3>向量库加载失败</h3>
            <p>${escapeHtml(error.message || "未知错误")}</p>
          </div>
        `;
        renderEmptyArts();
      }
    }
  }

  async function loadTask(taskId, { follow = true } = {}) {
    const trimmed = String(taskId || "").trim();
    if (!trimmed) {
      setTaskStatusLine("未找到可用任务。", "error");
      return null;
    }
    try {
      const task = await requestJson(API.task(trimmed), { timeoutMs: 15000 });
      renderTaskSnapshot(task);
      if (follow && !isTerminalTask(task.status)) {
        taskPoller.start(task.task_id);
      } else {
        taskPoller.stop();
      }
      state.currentTaskId = task.task_id;
      setConnectionStatus("online", "API READY");
      return task;
    } catch (error) {
      taskPoller.stop();
      setTaskStatusLine(error.message || "任务查询失败。", "error");
      setConnectionStatus("warning", "API DEGRADED");
      el.taskError.textContent = error.message || "任务查询失败。";
      return null;
    }
  }

  async function startTaskById(taskId) {
    const trimmed = String(taskId || "").trim();
    if (!trimmed) {
      setTaskStatusLine("未找到可用任务。", "error");
      return null;
    }
    try {
      await requestJson(API.startTask(trimmed), { method: "POST", timeoutMs: 15000 });
      setTaskStatusLine("任务已调度，正在启动构建。", "live");
      navigate("tasks");
      return await loadTask(trimmed, { follow: true });
    } catch (error) {
      setTaskStatusLine(error.message || "启动失败。", "error");
      setConnectionStatus("warning", "API DEGRADED");
      return null;
    }
  }

  function isTerminalTask(status) {
    return status === "completed" || status === "failed";
  }

  function openTaskFromDocument(documentId) {
    const task = state.recentTasks.find((item) => item.document_id === documentId);
    navigate("tasks");
    if (!task) {
      renderTaskSnapshot(null);
      setTaskStatusLine("未找到该文档对应的本地任务记录。", "error");
      return;
    }
    loadTask(task.task_id, { follow: true });
  }

  function openTaskFromRecent(taskId) {
    navigate("tasks");
    loadTask(taskId, { follow: true });
  }

  function setSelectedFile(file) {
    const validation = validatePdfFile(file);
    if (validation) {
      setUploadProgress(0, validation);
      state.selectedFile = null;
      el.pickedFileName.textContent = "尚未选择文件";
      el.uploadBtn.disabled = true;
      el.uploadActionBar.classList.add("is-hidden");
      el.uploadResultBody.textContent = validation;
      return false;
    }
    renderUploadSelection(file);
    return true;
  }

  async function uploadSelectedFile() {
    const file = state.selectedFile;
    const validation = validatePdfFile(file);
    if (validation) {
      setUploadProgress(0, validation);
      el.uploadResultBody.textContent = validation;
      return;
    }
    setConnectionStatus("online", "API READY");
    el.uploadBtn.disabled = true;
    el.pickFileBtn.disabled = true;
    setUploadProgress(0, "UPLOADING… 0%");
    el.uploadResultBody.textContent = "正在上传文件并创建任务…";
    try {
      const response = await uploadWithProgress(file, (loaded, total) => {
        const percent = total ? Math.round((loaded / total) * 100) : 0;
        setUploadProgress(percent, `UPLOADING… ${percent}%`);
      });
      const documentId = String(response.document_id || "");
      const taskId = String(response.task_id || "");
      const now = new Date().toISOString();
      upsertRecentTask({
        task_id: taskId,
        document_id: documentId,
        file_name: file.name,
        status: "pending",
        step: "pending",
        progress: 0,
        created_at: now,
        updated_at: now,
        last_seen_at: now,
      });
      el.uploadResultBody.innerHTML = `
        构建记录已创建。<br />
        <span class="muted">Document ID: ${escapeHtml(documentId)}</span><br />
        <span class="muted">你可以在构建协议中直接查看处理步骤。</span>
      `;
      el.uploadActionBar.classList.remove("is-hidden");
      el.uploadLaunchBtn.dataset.taskId = taskId;
      el.uploadLaterBtn.dataset.taskId = taskId;
      setUploadProgress(100, "接入完成，等待启动。");
      await refreshDocuments();
      renderRecentTaskLists();
      renderDocuments();
      renderVectorRecords();
      setTaskStatusLine("任务已创建，等待启动构建协议。", "ok");
    } catch (error) {
      setUploadProgress(0, error.message || "上传失败。");
      el.uploadResultBody.textContent = error.message || "上传失败。";
      setConnectionStatus("warning", "API DEGRADED");
    } finally {
      el.uploadBtn.disabled = false;
      el.pickFileBtn.disabled = false;
    }
  }

  function removeConversation(conversationId) {
    const conversation = state.conversations.find((item) => item.id === conversationId);
    if (!conversation) {
      return;
    }
    const confirmDelete = window.confirm(`删除会话「${conversation.title}」？`);
    if (!confirmDelete) {
      return;
    }
    state.conversations = state.conversations.filter((item) => item.id !== conversationId);
    if (!state.conversations.length) {
      state.conversations = [createConversation("新会话")];
    }
    if (!state.conversations.some((item) => item.id === state.activeConversationId)) {
      state.activeConversationId = state.conversations[0].id;
    }
    saveConversations();
    renderConversationList();
    renderChatConversation();
  }

  function newConversation() {
    const conversation = createConversation("新会话");
    state.conversations.unshift(conversation);
    state.activeConversationId = conversation.id;
    saveConversations();
    renderConversationList();
    renderChatConversation();
    el.chatInput.focus();
  }

  function clearCurrentConversation() {
    const conversation = getActiveConversation();
    if (!conversation) {
      return;
    }
    const shouldClear = window.confirm("清空当前会话的全部消息？");
    if (!shouldClear) {
      return;
    }
    conversation.messages = [];
    conversation.updatedAt = new Date().toISOString();
    saveConversations();
    renderConversationList();
    renderChatConversation();
    setChatStatus("会话已清空。", "live");
  }

  async function copyText(text) {
    const value = String(text || "");
    if (!value) {
      return false;
    }
    try {
      await navigator.clipboard.writeText(value);
      return true;
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.setAttribute("readonly", "true");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      document.body.removeChild(textarea);
      return copied;
    }
  }

  async function copyLastAnswer() {
    const answer = getLastAssistantAnswer();
    if (!answer) {
      setChatStatus("当前会话没有可复制的回答。", "error");
      return;
    }
    const copied = await copyText(answer);
    setChatStatus(copied ? "回答已复制。" : "复制失败，请手动选择文本。", copied ? "ok" : "error");
  }

  async function sendChatQuestion(question) {
    const normalizedQuestion = String(question || "").trim();
    if (!normalizedQuestion) {
      setChatStatus("请输入问题。", "error");
      return;
    }
    if (state.chatBusy) {
      return;
    }
    const conversation = getActiveConversation();
    if (!conversation) {
      return;
    }

    state.chatBusy = true;
    el.chatSendBtn.disabled = true;
    el.chatInput.disabled = true;
    setChatStatus("正在接入本地神经中枢 OLLAMA…", "live");

    conversation.messages.push({
      role: "user",
      content: normalizedQuestion,
      createdAt: new Date().toISOString(),
    });
    conversation.messages.push({
      role: "assistant",
      content: "",
      citations: [],
      fallback: false,
      loading: true,
      createdAt: new Date().toISOString(),
    });
    renameConversationFromQuestion(conversation, normalizedQuestion);
    conversation.updatedAt = new Date().toISOString();
    saveConversations();
    renderConversationList();
    renderChatConversation();

    try {
      const response = await requestJson(API.chat, {
        method: "POST",
        timeoutMs: 130000,
        body: { question: normalizedQuestion },
      });
      conversation.messages = conversation.messages.filter((message) => !message.loading);
      conversation.messages.push({
        role: "assistant",
        content: String(response.answer || ""),
        citations: Array.isArray(response.citations) ? response.citations : [],
        fallback: Boolean(response.fallback),
        loading: false,
        createdAt: new Date().toISOString(),
      });
      conversation.updatedAt = new Date().toISOString();
      saveConversations();
      renderConversationList();
      renderChatConversation();
      el.chatInput.value = "";
      setChatStatus(response.fallback ? "知识库中未找到相关依据。请先上传并构建文档。" : "回答已返回。", response.fallback ? "error" : "ok");
    } catch (error) {
      conversation.messages = conversation.messages.filter((message) => !message.loading);
      conversation.messages.push({
        role: "assistant",
        content: error.message || "问答失败。",
        citations: [],
        fallback: false,
        loading: false,
        createdAt: new Date().toISOString(),
      });
      conversation.updatedAt = new Date().toISOString();
      saveConversations();
      renderConversationList();
      renderChatConversation();
      setChatStatus(error.message || "问答失败。", "error");
      setConnectionStatus("warning", "API DEGRADED");
    } finally {
      state.chatBusy = false;
      el.chatSendBtn.disabled = false;
      el.chatInput.disabled = false;
      el.chatInput.focus();
    }
  }

  function openUploadTask(taskId, shouldStart) {
    if (!taskId) {
      return;
    }
    navigate("tasks");
    if (shouldStart) {
      startTaskById(taskId);
      return;
    }
    loadTask(taskId, { follow: false });
  }

  function bindEvents() {
    el.topNav.addEventListener("click", (event) => {
      const button = event.target.closest("[data-route]");
      if (!button) {
        return;
      }
      navigate(button.dataset.route);
    });

    window.addEventListener("hashchange", syncRouteFromHash);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        taskPoller.pause();
        return;
      }
      taskPoller.resume();
    });

    el.dropzone.addEventListener("click", (event) => {
      if (event.target.closest("button")) {
        return;
      }
      el.uploadFile.click();
    });

    el.dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      el.dropzone.classList.add("is-dragover");
    });
    el.dropzone.addEventListener("dragleave", () => {
      el.dropzone.classList.remove("is-dragover");
    });
    el.dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      el.dropzone.classList.remove("is-dragover");
      const file = event.dataTransfer?.files?.[0];
      if (file) {
        setSelectedFile(file);
      }
    });

    el.pickFileBtn.addEventListener("click", () => {
      el.uploadFile.click();
    });
    el.uploadFile.addEventListener("change", () => {
      setSelectedFile(el.uploadFile.files?.[0] || null);
    });
    el.uploadBtn.addEventListener("click", uploadSelectedFile);
    el.uploadLaunchBtn.addEventListener("click", () => {
      const taskId = el.uploadLaunchBtn.dataset.taskId;
      openUploadTask(taskId, true);
    });
    el.uploadLaterBtn.addEventListener("click", () => {
      const taskId = el.uploadLaterBtn.dataset.taskId;
      openUploadTask(taskId, false);
    });

    el.refreshRecentTasksBtn.addEventListener("click", renderRecentTaskLists);
    el.recentUploadList.addEventListener("click", onRecentTaskListClick);
    el.recentTaskList.addEventListener("click", onRecentTaskListClick);

    el.refreshDocumentsBtn.addEventListener("click", refreshDocuments);
    el.documentGrid.addEventListener("click", (event) => {
      const button = event.target.closest("[data-document-id]");
      if (!button) {
        return;
      }
      openTaskFromDocument(button.dataset.documentId);
    });

    el.refreshVectorsBtn.addEventListener("click", async () => {
      await Promise.allSettled([refreshCategories(), refreshVectors()]);
    });
    el.vectorSearchInput.addEventListener("input", () => {
      state.vectorSearch = el.vectorSearchInput.value;
      renderVectorRecords();
    });
    el.vectorCategoryChips.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-category]");
      if (!chip) {
        return;
      }
      state.selectedCategory = chip.dataset.category || "all";
      renderVectorCategoryChips();
      renderVectorRecords();
    });

    el.conversationList.addEventListener("click", (event) => {
      const deleteButton = event.target.closest("[data-action='delete-conversation']");
      if (deleteButton) {
        event.stopPropagation();
        removeConversation(deleteButton.dataset.conversationId);
        return;
      }
      const item = event.target.closest("[data-conversation-id]");
      if (!item) {
        return;
      }
      setActiveConversation(item.dataset.conversationId);
    });
    el.conversationList.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      const item = event.target.closest("[data-conversation-id]");
      if (!item) {
        return;
      }
      event.preventDefault();
      setActiveConversation(item.dataset.conversationId);
    });

    el.newConversationBtn.addEventListener("click", newConversation);
    el.clearConversationBtn.addEventListener("click", clearCurrentConversation);
    el.copyLastAnswerBtn.addEventListener("click", copyLastAnswer);
    el.chatForm.addEventListener("submit", (event) => {
      event.preventDefault();
      sendChatQuestion(el.chatInput.value);
    });
    el.chatInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.ctrlKey) {
        event.preventDefault();
        sendChatQuestion(el.chatInput.value);
      }
    });
  }

  function onRecentTaskListClick(event) {
    const item = event.target.closest("[data-task-id]");
    if (!item) {
      return;
    }
    openTaskFromRecent(item.dataset.taskId);
  }

  function onPolledTaskUpdate(task) {
    renderTaskSnapshot(task);
    setConnectionStatus("online", "API READY");
    if (isTerminalTask(task.status)) {
      taskPoller.stop();
    }
  }

  function onPolledTaskError(error, attempt, exhausted) {
    if (exhausted) {
      setTaskStatusLine(`信号丢失，轮询已停止。${error.message || ""}`.trim(), "error");
      setConnectionStatus("danger", "API LOST");
      return;
    }
    setTaskStatusLine(`信号丢失，正在重连…（${attempt}/3）`, "live");
    setConnectionStatus("warning", "RECONNECTING");
  }

  async function bootstrap() {
    renderEmptyArts();
    ensureActiveConversation();
    if (!window.location.hash) {
      window.history.replaceState(null, "", "#/upload");
    }
    syncRouteFromHash();
    bindEvents();
    renderRecentTaskLists();
    renderConversationList();
    renderChatConversation();
    renderTaskSnapshot(null);
    renderDocuments();
    renderVectorCategoryChips();
    renderVectorRecords();
    updateStats();
    setUploadProgress(0, "等待文件接入");
    setTaskProgress(0, "请先上传并启动构建。");
    renderUploadSelection(null);
    await Promise.allSettled([checkHealth(), refreshDocuments(), refreshCategories(), refreshVectors()]);
  }

  class TaskPoller {
    constructor(onUpdate, onError) {
      this.onUpdate = onUpdate;
      this.onError = onError;
      this.taskId = "";
      this.timer = null;
      this.active = false;
      this.hidden = false;
      this.retryCount = 0;
    }

    start(taskId) {
      this.stop();
      this.taskId = String(taskId || "").trim();
      if (!this.taskId) {
        return;
      }
      this.active = true;
      this.hidden = document.hidden;
      this.retryCount = 0;
      if (!this.hidden) {
        this.schedule(1000);
      }
    }

    stop() {
      if (this.timer) {
        window.clearTimeout(this.timer);
      }
      this.timer = null;
      this.active = false;
      this.hidden = false;
      this.retryCount = 0;
      this.taskId = "";
    }

    pause() {
      if (!this.active) {
        return;
      }
      this.hidden = true;
      if (this.timer) {
        window.clearTimeout(this.timer);
      }
      this.timer = null;
      if (state.currentTaskId) {
        setTaskStatusLine("页面失焦，轮询暂停。", "live");
      }
    }

    resume() {
      if (!this.active) {
        return;
      }
      this.hidden = false;
      this.schedule(0);
    }

    schedule(delay) {
      if (!this.active || this.hidden) {
        return;
      }
      if (this.timer) {
        window.clearTimeout(this.timer);
      }
      this.timer = window.setTimeout(() => this.tick(), delay);
    }

    async tick() {
      if (!this.active || this.hidden || !this.taskId) {
        return;
      }
      try {
        const task = await requestJson(API.task(this.taskId), { timeoutMs: 15000 });
        this.retryCount = 0;
        this.onUpdate(task);
        if (isTerminalTask(task.status)) {
          this.stop();
          return;
        }
        this.schedule(1000);
      } catch (error) {
        this.retryCount += 1;
        const exhausted = this.retryCount >= 3;
        this.onError(error, this.retryCount, exhausted);
        if (exhausted) {
          this.stop();
          return;
        }
        const delay = Math.min(1000 * (2 ** (this.retryCount - 1)), 8000);
        this.schedule(delay);
      }
    }
  }

  taskPoller = new TaskPoller(onPolledTaskUpdate, onPolledTaskError);

  bootstrap();
})();
