(function () {
  "use strict";

  const BRAND_ICON = '<img class="zhitu-mark" src="src/zhitu-mark.svg?v=20260916" alt="" aria-hidden="true" />';
  // Logged-in users can upload private materials for the current consultation.
  const ENABLE_FILE_UPLOADS = true;
  const SEND_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 4 16 8-16 8 3-8-3-8Zm3 8h13" /></svg>';
  const CHEVRON_ICON = '<svg class="chevron is-open" viewBox="0 0 20 20" aria-hidden="true"><path d="m6 8 4 4 4-4" /></svg>';
  const DELETE_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v5m4-5v5" /></svg>';

  const conversation = document.getElementById("conversation");
  const conversationScroll = document.getElementById("conversation-scroll");
  const emptyState = document.getElementById("empty-state");
  const form = document.getElementById("question-form");
  const input = document.getElementById("question-input");
  const sendButton = document.getElementById("send-button");
  const homeForm = document.getElementById("home-question-form");
  const homeInput = document.getElementById("home-question-input");
  const homeSubmit = document.getElementById("home-question-submit");
  const homeMaterialUpload = document.getElementById("home-material-upload");
  const homeMaterialSummary = document.getElementById("home-material-summary");
  const homeAccountButton = document.getElementById("home-account-button");
  const newChatButton = document.getElementById("new-chat-button");
  const newTaskCompact = document.getElementById("new-task-compact");
  const deleteTaskCompact = document.getElementById("delete-task-compact");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const thinkingToggleButton = document.getElementById("thinking-toggle-button");
  const fileInput = document.getElementById("file-input");
  const fileStatus = document.getElementById("file-status");
  const workspaceLabel = document.getElementById("workspace-label");
  const loginButton = document.getElementById("login-button");
  const sidebarAccountName = document.getElementById("sidebar-account-name");
  const sidebarUserAvatar = document.getElementById("sidebar-user-avatar");
  const authForm = document.getElementById("auth-form");
  const authSwitch = document.getElementById("auth-switch");
  const authTitle = document.getElementById("auth-title");
  const authSubmit = document.getElementById("auth-submit");
  const authError = document.getElementById("auth-error");
  const authEmailField = document.getElementById("auth-email-field");
  const authEmail = document.getElementById("auth-email");
  const authBackdrop = document.getElementById("auth-backdrop");
  const authDialog = document.getElementById("auth-dialog");
  const appShell = document.querySelector(".app-shell");
  const chatMain = document.querySelector(".chat-main");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebarBackdrop = document.getElementById("sidebar-backdrop");
  const accountMenu = document.getElementById("account-menu");
  let authReturnFocus = null;
  const settingsButton = document.getElementById("settings-button");
  const settingsBackdrop = document.getElementById("settings-backdrop");
  const settingsDialog = document.getElementById("settings-dialog");
  const settingsTabs = document.querySelectorAll("[data-settings-tab]");
  const settingsPanels = document.querySelectorAll("[data-settings-panel]");
  let settingsReturnFocus = null;
  const settingsClose = document.getElementById("settings-close");
  const settingsSave = document.getElementById("settings-save");
  const settingsStatus = document.getElementById("settings-status");
  const settingAutoExpand = document.getElementById("setting-auto-expand");
  const settingShowProcess = document.getElementById("setting-show-process");
  const settingAnswerDetailOptions = document.querySelectorAll('input[name="answer-detail"]');
  const retrievalModeButtons = document.querySelectorAll("[data-retrieval-mode]");
  const retrievalModeStatuses = document.querySelectorAll("[data-retrieval-mode-status]");
  const settingTheme = document.getElementById("setting-theme");
  const settingBackendUrl = document.getElementById("setting-backend-url");
  const activityList = document.getElementById("activity-list");
  const activityRefresh = document.getElementById("activity-refresh");
  const accountDeletionStatus = document.getElementById("account-deletion-status");
  const accountDeletionButton = document.getElementById("account-deletion-button");
  const settingsAccountName = document.getElementById("settings-account-name");
  const settingsAccountStatus = document.getElementById("settings-account-status");
  const settingsAccountUserId = document.getElementById("settings-account-user-id");
  const settingsAccountWorkspace = document.getElementById("settings-account-workspace");
  const settingsLogoutButton = document.getElementById("settings-logout-button");
  const appVersion = document.getElementById("app-version");
  const diagnosticsTriggers = document.querySelectorAll("[data-diagnostics-trigger]");
  const diagnosticsBackdrop = document.getElementById("diagnostics-backdrop");
  const diagnosticsClose = document.getElementById("diagnostics-close");
  const diagnosticsRefresh = document.getElementById("diagnostics-refresh");
  const diagnosticsCopy = document.getElementById("diagnostics-copy");
  const diagnosticsOutput = document.getElementById("diagnostics-output");
  const taskList = document.getElementById("task-list");
  const taskCount = document.getElementById("task-count");
  const taskSelect = document.getElementById("task-select");
  const workflowList = document.getElementById("workflow-list");
  const workflowTotal = document.getElementById("workflow-total");
  const routePages = document.querySelectorAll("[data-route-page]");
  const routeLinks = document.querySelectorAll("[data-route-target]");
  const topbarTitle = document.getElementById("topbar-title");
  const topbarSubtitle = document.getElementById("topbar-subtitle");
  const materialDrawer = document.getElementById("material-drawer");
  const materialDrawerToggle = document.getElementById("material-drawer-toggle");
  const materialDrawerResizeHandle = document.getElementById("material-drawer-resize-handle");
  const composerResizeHandle = document.getElementById("composer-resize-handle");
  const materialUploadAction = document.getElementById("material-upload-action");
  const materialList = document.getElementById("material-list");
  const materialCount = document.getElementById("material-count");
  const materialSessionLabel = document.getElementById("material-session-label");
  const settingsPageOpen = document.getElementById("settings-page-open");
  const settingsPageAccount = document.getElementById("settings-page-account");
  const settingsPreviewDetail = document.getElementById("settings-preview-detail");
  const settingsPreviewProcess = document.getElementById("settings-preview-process");
  const settingsPreviewTheme = document.getElementById("settings-preview-theme");
  const filesPageList = document.getElementById("files-page-list");
  const filesCount = document.getElementById("files-count");
  const authStatus = document.getElementById("auth-status");
  let currentRoute = "home";
  const MAX_TASKS = 20;
  const MAX_UPLOAD_FILES = 5;
  let taskNamespace = "guest";
  let tasks = [];
  let activeTaskId = "";
  let activeTask = null;
  let sessionId = "";
  let authenticated = false;
  let currentUser = null;
  // Invalidate pending requests whenever the displayed account changes.
  let identityVersion = 0;
  let authAttemptVersion = 0;
  let questionController = null;
  let authMode = "login";
  let authPanelOpen = false;
  let loading = false;
  let queuedQuestions = [];
  let conversationFollowLatest = true;
  const THEME_STORAGE_KEY = "lawrag.theme";
  const DELETED_SESSIONS_GLOBAL_KEY = "lawrag.deleted_sessions.global";
  let deepseekThinkingEnabled = false;
  let settingsSaveTimer = null;
  let settingsSaveVersion = 0;
  let backendUrlSaveTimer = null;
  let versionClickCount = 0;
  const DEFAULT_USER_SETTINGS = {
    include_web_default: false,
    auto_expand_professional: false,
    show_retrieval_process: true,
    answer_detail: "standard",
    enable_long_memory: false,
    theme: "warm_gold",
  };
  const RESIZE_STORAGE_KEY = "lawrag.layout.sizes";
  const RESIZE_DEFAULTS = {
    sidebar: 276,
    materialDrawer: 340,
  };
  const RESIZE_LIMITS = {
    sidebar: [220, 420],
    materialDrawer: [220, 520],
  };
  let materialRenderFrame = 0;
  let materialRenderSignature = "";
  let workspaceRefreshTimer = null;
  let workspaceRefreshInFlight = false;
  let workspaceRefreshDelay = 0;
  let userSettings = { ...DEFAULT_USER_SETTINGS };
  const COMPOSER_DEFAULT_SIZE = 178;
  const COMPOSER_LIMITS = [132, 300];
  const THEME_TOKENS = {
    government_green: {
      paper: "#F8FFFE",
      warm: "#F0FDFA",
      line: "#D1DCDB",
      ink: "#132F2C",
      muted: "#64748B",
      sage: "#0F766E",
      sageDeep: "#0D5D57",
      sageSoft: "#ECFDF8",
      gold: "#0F766E",
      goldSoft: "#E6F7F4",
      quoteBg: "#ECFDF8",
      quoteLine: "#0F766E",
      quoteText: "#0F766E",
      sidebarStart: "#0F766E",
      sidebarEnd: "#0D5D57",
      sidebarText: "#F4FFFD",
      sidebarMuted: "rgba(244,255,253,.66)",
      sidebarPanel: "rgba(255,255,255,.08)",
      sidebarPanelBorder: "rgba(255,255,255,.18)",
      sidebarActiveStart: "#7FCFC6",
      sidebarActiveEnd: "#4FAFA2",
      sidebarIcon: "#C9EFEB",
    },
    warm_gold: {
      shellBg: "#24231F",
      shellText: "#F5F1E8",
      shellMuted: "#BBB4A6",
      shellAccent: "#C7AD78",
      shellBorder: "#3F392E",
      onAccent: "#FCFAF5",
      paper: "#FCFAF5",
      warm: "#F5F1E8",
      line: "#E1DBCE",
      ink: "#302E28",
      muted: "#716C61",
      sage: "#806637",
      sageDeep: "#68512B",
      sageSoft: "#F1EADB",
      gold: "#A88A52",
      goldSoft: "#F5F1E8",
      quoteBg: "#F1EADB",
      quoteLine: "#A88A52",
      quoteText: "#68512B",
      sidebarStart: "#1E1D19",
      sidebarEnd: "#1E1D19",
      sidebarText: "#F5F1E8",
      sidebarMuted: "#BBB4A6",
      sidebarPanel: "rgba(168,138,82,.09)",
      sidebarPanelBorder: "rgba(168,138,82,.24)",
      sidebarActiveStart: "#514632",
      sidebarActiveEnd: "#403827",
      sidebarIcon: "#C7AD78",
    },
    indigo_ai: {
      paper: "#FBF9FF",
      warm: "#FAF5FF",
      line: "#EDE9FE",
      ink: "#1F2937",
      muted: "#9CA3AF",
      sage: "#7C3AED",
      sageDeep: "#6D28D9",
      sageSoft: "#F3E8FF",
      gold: "#7C3AED",
      goldSoft: "#FAF5FF",
      quoteBg: "#F3E8FF",
      quoteLine: "#7C3AED",
      quoteText: "#7C3AED",
      sidebarStart: "#7C3AED",
      sidebarEnd: "#6D28D9",
      sidebarText: "#FAF7FF",
      sidebarMuted: "rgba(250,247,255,.64)",
      sidebarPanel: "rgba(255,255,255,.08)",
      sidebarPanelBorder: "rgba(255,255,255,.18)",
      sidebarActiveStart: "#C4A7FF",
      sidebarActiveEnd: "#9D78F0",
      sidebarIcon: "#E4D4FF",
    },
    cool_gray: {
      shellBg: "#F8FAFC",
      shellText: "#1E293B",
      shellMuted: "#637080",
      shellAccent: "#475569",
      shellBorder: "#E2E8F0",
      onAccent: "#F8FAFC",
      paper: "#F8FAFC",
      warm: "#F1F5F9",
      line: "#E2E8F0",
      ink: "#1E293B",
      muted: "#637080",
      sage: "#475569",
      sageDeep: "#334155",
      sageSoft: "#F1F5F9",
      gold: "#475569",
      goldSoft: "#F8FAFC",
      quoteBg: "#F1F5F9",
      quoteLine: "#64748B",
      quoteText: "#334155",
      sidebarStart: "#F1F5F9",
      sidebarEnd: "#F1F5F9",
      sidebarText: "#1E293B",
      sidebarMuted: "#637080",
      sidebarPanel: "#E2E8F0",
      sidebarPanelBorder: "#CBD5E1",
      sidebarActiveStart: "#E2E8F0",
      sidebarActiveEnd: "#E2E8F0",
      sidebarIcon: "#475569",
    },
  };
  const WORKFLOW_STEPS = [
    { stage: "understanding", label: "问题理解", waiting: "等待提问", running: "正在理解问题", done: "已识别问题结构" },
    { stage: "retrieval", label: "多路检索", waiting: "等待检索", running: "正在检索材料", done: "多路召回已完成" },
    { stage: "ranking", label: "证据整理", waiting: "等待重排", running: "正在融合与重排", done: "融合、重排与校验完成" },
    { stage: "generation", label: "答案生成", waiting: "等待生成", running: "正在生成回答", done: "回答已生成" },
  ];

  const ROUTE_META = {
    home: { title: "知途 · 民事法律问答", subtitle: "了解法律，找到依法解决问题的路" },
    ask: { title: "知途 · 法律问答", subtitle: "依据事实与法律，提供参考分析和处理建议" },
    settings: { title: "偏好设置", subtitle: "调整回答方式、检索范围和工作区体验" },
  };
  const API_BASE_STORAGE_KEY = "lawrag.api_base_url";
  const DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:7294";
  const backendOriginCandidates = ["apiBase", "backend", "backendUrl"];

  function routeName(value) {
    return String(value || "").replace(/^#\/?/, "").split("?")[0].trim().toLowerCase();
  }

  function normalizeRoute(value) {
    const route = routeName(value);
    return Object.prototype.hasOwnProperty.call(ROUTE_META, route) ? route : "ask";
  }

  function locationOrigin() {
    const origin = String(window.location.origin || "").trim();
    if (!origin || origin === "null" || origin.startsWith("file:")) return "";
    return origin;
  }

  function normalizeBackendOrigin(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    const candidate = /^[a-zA-Z][a-zA-Z\d+\-.]*:/.test(raw) ? raw : `http://${raw}`;
    try {
      const url = new URL(candidate);
      if (!/^https?:$/.test(url.protocol)) return "";
      return url.origin;
    } catch (_error) {
      return "";
    }
  }

  function readBackendOriginOverride() {
    for (const key of backendOriginCandidates) {
      const fromQuery = normalizeBackendOrigin(new URLSearchParams(window.location.search).get(key));
      if (fromQuery) return fromQuery;
    }
    const injected = normalizeBackendOrigin(window.__LAW_RAG_API_BASE__ || window.__LAW_RAG_BACKEND_URL__ || "");
    if (injected) return injected;
    const currentLocation = locationOrigin();
    if (currentLocation) return currentLocation;
    const stored = normalizeBackendOrigin(readStorage(API_BASE_STORAGE_KEY));
    if (stored) return stored;
    return DEFAULT_BACKEND_ORIGIN;
  }

  function applyTheme(themeName = userSettings.theme) {
    const themeKey = Object.prototype.hasOwnProperty.call(THEME_TOKENS, themeName) ? themeName : DEFAULT_USER_SETTINGS.theme;
    const theme = THEME_TOKENS[themeKey];
    if (!theme) return themeKey;
    const root = document.documentElement.style;
    root.setProperty("--shell-bg", theme.shellBg || theme.paper);
    root.setProperty("--shell-text", theme.shellText || theme.ink);
    root.setProperty("--shell-muted", theme.shellMuted || theme.muted);
    root.setProperty("--shell-accent", theme.shellAccent || theme.sage);
    root.setProperty("--shell-border", theme.shellBorder || theme.line);
    root.setProperty("--paper", theme.paper);
    root.setProperty("--warm", theme.warm);
    root.setProperty("--line", theme.line);
    root.setProperty("--ink", theme.ink);
    root.setProperty("--muted", theme.muted);
    root.setProperty("--sage", theme.sage);
    root.setProperty("--sage-deep", theme.sageDeep);
    root.setProperty("--sage-soft", theme.sageSoft);
    root.setProperty("--gold", theme.gold);
    root.setProperty("--gold-soft", theme.goldSoft);
    root.setProperty("--quote-green-soft", theme.quoteBg);
    root.setProperty("--quote-green-line", theme.quoteLine);
    root.setProperty("--quote-green", theme.quoteText);
    root.setProperty("--text-strong", theme.ink);
    root.setProperty("--text-body", theme.ink);
    root.setProperty("--text-muted", theme.muted);
    root.setProperty("--text-accent", theme.sage);
    root.setProperty("--text-accent-strong", theme.sageDeep);
    root.setProperty("--text-accent-soft", theme.sageSoft);
    root.setProperty("--text-on-accent", theme.onAccent || theme.sidebarText);
    root.setProperty("--text-panel", theme.ink);
    root.setProperty("--text-panel-muted", theme.muted);
    root.setProperty("--text-input", theme.ink);
    root.setProperty("--text-input-placeholder", theme.muted);
    root.setProperty("--text-tag", theme.quoteLine);
    root.setProperty("--text-tag-bg", theme.quoteBg);
    root.setProperty("--text-tag-border", theme.quoteLine);
    root.setProperty("--text-answer-strong", theme.ink);
    root.setProperty("--text-answer-body", theme.ink);
    root.setProperty("--text-answer-note", theme.muted);
    root.setProperty("--text-answer-link", theme.quoteLine);
    root.setProperty("--text-answer-link-bg", theme.quoteBg);
    root.setProperty("--text-answer-link-border", theme.quoteLine);
    root.setProperty("--text-risk", theme.quoteLine);
    root.setProperty("--text-risk-bg", theme.quoteBg);
    root.setProperty("--text-table-head", theme.ink);
    root.setProperty("--text-table-head-bg", theme.goldSoft);
    root.setProperty("--sidebar-start", theme.sidebarStart);
    root.setProperty("--sidebar-end", theme.sidebarEnd);
    root.setProperty("--sidebar-text", theme.sidebarText);
    root.setProperty("--sidebar-muted", theme.sidebarMuted);
    root.setProperty("--sidebar-panel", theme.sidebarPanel);
    root.setProperty("--sidebar-panel-border", theme.sidebarPanelBorder);
    root.setProperty("--sidebar-active-start", theme.sidebarActiveStart);
    root.setProperty("--sidebar-active-end", theme.sidebarActiveEnd);
    root.setProperty("--sidebar-icon", theme.sidebarIcon);
    document.body.dataset.theme = themeKey;
    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) themeColor.content = theme.shellBg || theme.paper;
    if (settingTheme) settingTheme.value = themeKey;
    return themeKey;
  }

  let apiBaseUrl = readBackendOriginOverride();
  let configuredBackendUrl = "";

  function persistBackendOrigin(value) {
    const normalized = normalizeBackendOrigin(value);
    const currentLocation = locationOrigin();
    try {
      if (normalized && normalized !== currentLocation) window.localStorage?.setItem(API_BASE_STORAGE_KEY, normalized);
      else window.localStorage?.removeItem(API_BASE_STORAGE_KEY);
    } catch (_error) {
      /* localStorage may be unavailable */
    }
    return normalized;
  }

  async function loadDesktopBackendOrigin() {
    if (!window.lawragDesktop?.getConfig) return "";
    try {
      const config = await window.lawragDesktop.getConfig();
      const connectedUrl = normalizeBackendOrigin(config?.serverUrl);
      const savedUrl = normalizeBackendOrigin(config?.savedServerUrl);
      if (!connectedUrl && !savedUrl) return "";
      const currentLocation = locationOrigin();
      configuredBackendUrl = savedUrl || connectedUrl;
      apiBaseUrl = currentLocation || connectedUrl || savedUrl;
      persistBackendOrigin(apiBaseUrl);
      applyBackendOriginToForm();
      return apiBaseUrl;
    } catch (_error) {
      return "";
    }
  }

  function applyBackendOriginToForm() {
    if (!settingBackendUrl) return;
    const currentLocation = locationOrigin();
    const displayUrl = configuredBackendUrl || apiBaseUrl;
    settingBackendUrl.value = displayUrl === DEFAULT_BACKEND_ORIGIN || displayUrl === currentLocation ? "" : displayUrl;
  }

  async function saveBackendOriginFromForm({ quiet = false } = {}) {
    if (!settingBackendUrl) return apiBaseUrl;
    window.clearTimeout(backendUrlSaveTimer);
    const normalized = normalizeBackendOrigin(settingBackendUrl.value);
    configuredBackendUrl = normalized;
    apiBaseUrl = normalized || locationOrigin() || DEFAULT_BACKEND_ORIGIN;
    persistBackendOrigin(normalized ? apiBaseUrl : "");
    if (window.lawragDesktop?.applyServerUrl) {
      try {
        const result = await window.lawragDesktop.applyServerUrl(apiBaseUrl);
        const savedUrl = normalizeBackendOrigin(result?.savedServerUrl || apiBaseUrl);
        const connectedUrl = normalizeBackendOrigin(result?.serverUrl || savedUrl);
        configuredBackendUrl = savedUrl || configuredBackendUrl;
        apiBaseUrl = connectedUrl || savedUrl || apiBaseUrl;
        persistBackendOrigin(savedUrl || apiBaseUrl);
      } catch (error) {
        if (settingsStatus) {
          settingsStatus.classList.add("is-error");
          settingsStatus.textContent = friendlyErrorMessage(error, "服务地址保存失败");
        }
        return apiBaseUrl;
      }
    }
    applyBackendOriginToForm();
    if (!quiet && settingsStatus) {
      settingsStatus.classList.remove("is-error");
      settingsStatus.textContent = normalized ? `服务地址已保存：${apiBaseUrl}` : "已恢复默认服务地址";
    }
    return apiBaseUrl;
  }

  function renderSettingsPreview() {
    if (settingsPreviewDetail) settingsPreviewDetail.textContent = formatAnswerDetail(userSettings.answer_detail);
    if (settingsPreviewProcess) settingsPreviewProcess.textContent = userSettings.show_retrieval_process ? "显示" : "隐藏";
    if (settingsPreviewTheme) settingsPreviewTheme.textContent = formatThemeName(userSettings.theme);
  }

  function setMaterialDrawerExpanded(expanded) {
    const isExpanded = ENABLE_FILE_UPLOADS && Boolean(expanded);
    document.body.classList.toggle("material-drawer-expanded", isExpanded);
    if (materialDrawer) materialDrawer.classList.toggle("is-expanded", isExpanded);
    if (materialDrawerToggle) {
      materialDrawerToggle.setAttribute("aria-expanded", String(isExpanded));
      materialDrawerToggle.title = isExpanded ? "收起证据材料" : "展开证据材料";
      const icon = materialDrawerToggle.querySelector("strong");
      if (icon) icon.textContent = isExpanded ? "›" : "‹";
    }
  }

  function toggleMaterialDrawer() {
    setMaterialDrawerExpanded(!materialDrawer?.classList.contains("is-expanded"));
  }

  function closeMaterialDrawer() {
    setMaterialDrawerExpanded(false);
  }

  function renderMaterialItems(list, { compact = false } = {}) {
    if (!list.length) {
      return `<div class="empty-panel"><strong>本次咨询还没有证据材料</strong><span>添加合同、转账记录或聊天记录，可以补充案情并帮助分析证据价值。</span><button type="button" data-upload-files>添加证据材料</button></div>`;
    }
    return list.map((file) => {
      const name = file.file_name || file.source || "材料";
      const meta = workspaceFileMeta(file) || "已加入当前会话";
      const preview = workspaceFilePreview(file);
      const documentId = file.document_id || "";
      const previewHtml = preview
        ? `<p class="file-extraction-preview">识别预览：${escapeHtml(preview)}</p>`
        : `<p class="file-extraction-preview is-empty">暂无可预览的识别文本</p>`;
      return `<article class="material-item${compact ? " is-page-item" : ""}"><div class="file-type-mark">${escapeHtml(String(name).split(".").pop()?.slice(0, 4).toUpperCase() || "FILE")}</div><div class="material-item-main"><strong>${escapeHtml(name)}</strong><span>${escapeHtml(meta)}</span>${previewHtml}</div><button class="material-delete" type="button" data-document-id="${escapeHtml(documentId)}" data-file-name="${escapeHtml(name)}">删除</button></article>`;
    }).join("");
  }

  function syncMaterialList(files = []) {
    const list = Array.isArray(files) ? files : [];
    const signature = list.map((file) => `${file.document_id || file.source_id || file.file_name || ""}:${file.status || ""}:${file.extracted_char_count || 0}:${workspaceFilePreview(file)}`).join("|");
    if (signature === materialRenderSignature && materialList?.children?.length) return;
    materialRenderSignature = signature;
    if (materialCount) materialCount.textContent = formatNumber(list.length);
    if (materialSessionLabel) materialSessionLabel.textContent = `当前会话 · ${list.length} 份材料`;
    if (!materialList) return;
    const html = renderMaterialItems(list);
    const previousScrollTop = materialList.scrollTop;
    materialList.classList.add("is-rendering");
    if (materialRenderFrame) cancelAnimationFrame(materialRenderFrame);
    materialRenderFrame = requestAnimationFrame(() => {
      materialList.innerHTML = html;
      bindMaterialListActions(materialList);
      materialList.scrollTop = previousScrollTop;
      materialList.classList.remove("is-rendering");
    });
  }

  function bindMaterialListActions(root) {
    if (!root) return;
    root.querySelectorAll("[data-upload-files]").forEach((button) => button.addEventListener("click", openFilePicker));
    root.querySelectorAll(".material-delete").forEach((button) => button.addEventListener("click", () => deleteWorkspaceFile(button.dataset.documentId, button.dataset.fileName)));
  }

  function renderMaterialPanel(files = []) {
    syncMaterialList(files);
  }

  function renderFilesPage(files = []) {
    if (!filesPageList) return;
    const list = Array.isArray(files) ? files : [];
    if (filesCount) filesCount.textContent = formatNumber(list.length);
    filesPageList.innerHTML = `
      <div class="page-heading">
        <div><p class="eyebrow">事实与证据</p><h1>我的材料</h1><p>整理你上传的合同、聊天记录与证据。材料只在对应会话中使用。</p></div>
        <button class="secondary-action" type="button" data-upload-files>上传材料</button>
      </div>
      <div class="files-page-list${list.length ? "" : " is-empty"}">${renderMaterialItems(list, { compact: true })}</div>`;
    bindMaterialListActions(filesPageList);
  }

  function closeAccountMenu() {
    if (accountMenu) accountMenu.hidden = true;
    loginButton?.setAttribute("aria-expanded", "false");
  }

  function setSidebarOpen(open) {
    const mobile = window.matchMedia("(max-width: 820px)").matches;
    const wasOpen = document.body.classList.contains("sidebar-open");
    document.body.classList.toggle("sidebar-open", mobile && Boolean(open));
    if (chatMain) chatMain.inert = mobile && Boolean(open);
    if (sidebarBackdrop) sidebarBackdrop.hidden = !mobile || !open;
    const expanded = mobile ? Boolean(open) : !document.body.classList.contains("sidebar-collapsed");
    sidebarToggle?.setAttribute("aria-expanded", String(expanded));
    sidebarToggle?.setAttribute("aria-label", expanded ? "收起导航" : "展开导航");
    if (mobile && open && !wasOpen) newChatButton?.focus();
    else if (wasOpen && !open && document.activeElement?.closest?.("#main-sidebar")) sidebarToggle?.focus();
  }

  function navigate(route, { replace = false } = {}) {
    const requestedRoute = routeName(route);
    const nextRoute = normalizeRoute(route);
    setSidebarOpen(false);
    closeAccountMenu();
    const currentHash = String(window.location.hash || "");
    if (replace || requestedRoute !== nextRoute) window.history.replaceState({}, "", `#${nextRoute}`);
    else if (currentHash !== `#${nextRoute}`) window.history.pushState({}, "", `#${nextRoute}`);
    currentRoute = nextRoute;
    document.body.classList.toggle("is-landing", nextRoute === "home");
    document.body.classList.toggle("show-auth-panel", authPanelOpen && nextRoute === "home");
    routePages.forEach((page) => page.classList.toggle("is-active", page.dataset.routePage === nextRoute));
    document.querySelectorAll(".route-link").forEach((link) => {
      const active = link.dataset.routeTarget === nextRoute;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    document.querySelectorAll(".mobile-route-nav [data-route-target]").forEach((link) => link.classList.toggle("is-active", link.dataset.routeTarget === nextRoute));
    const meta = ROUTE_META[nextRoute];
    if (topbarTitle) topbarTitle.textContent = meta.title;
    if (topbarSubtitle) topbarSubtitle.textContent = meta.subtitle;
    if (nextRoute === "settings") renderSettingsPreview();
    if (nextRoute === "ask") requestAnimationFrame(() => input?.focus());
  }

  window.addEventListener("hashchange", () => navigate(window.location.hash));
  window.addEventListener("popstate", () => navigate(window.location.hash));

  function getOrCreateScope(key, prefix) {
    let existing = "";
    try { existing = window.localStorage?.getItem(key) || ""; } catch (_error) { /* storage may be restricted */ }
    if (existing) return existing;
    const value = `${prefix}_${globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`}`;
    try { window.localStorage?.setItem(key, value); } catch (_error) { /* memory-only session */ }
    return value;
  }

  function readStorage(key) {
    try { return window.localStorage?.getItem(key) || ""; } catch (_error) { return ""; }
  }

  function writeStorage(key, value) {
    try { window.localStorage?.setItem(key, value); } catch (_error) { /* cookie remains available */ }
  }

  function readDeletedSessions(namespace = taskNamespace) {
    const namespaces = Array.from(new Set([String(namespace || "guest").replace(/[^0-9A-Za-z_-]/g, "_"), "guest", "global"]));
    const deleted = new Set();
    namespaces.forEach((ns) => {
      try {
        const parsed = JSON.parse(readStorage(ns === "global" ? DELETED_SESSIONS_GLOBAL_KEY : deletedSessionsStorageKey(ns)) || "[]");
        (Array.isArray(parsed) ? parsed : []).forEach((value) => {
          const sessionId = String(value || "").trim();
          if (sessionId) deleted.add(sessionId);
        });
      } catch (_error) {
        /* ignore */
      }
    });
    return deleted;
  }

  function markSessionDeleted(session) {
    const sessionIdToDelete = String(session || "").trim();
    if (!sessionIdToDelete) return;
    const namespaces = Array.from(new Set([taskNamespace, "guest", "global"])).filter(Boolean);
    namespaces.forEach((ns) => {
      const deleted = readDeletedSessions(ns);
      deleted.add(sessionIdToDelete);
      writeStorage(ns === "global" ? DELETED_SESSIONS_GLOBAL_KEY : deletedSessionsStorageKey(ns), JSON.stringify(Array.from(deleted).slice(-MAX_TASKS * 2)));
    });
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, Number(value) || min));
  }

  function loadLayoutSizes() {
    let stored = {};
    try { stored = JSON.parse(readStorage(RESIZE_STORAGE_KEY) || "{}"); } catch (_error) { stored = {}; }
    const sizes = Object.fromEntries(Object.entries(RESIZE_DEFAULTS).map(([key, value]) => {
      const [min, max] = RESIZE_LIMITS[key];
      return [key, clamp(stored[key] ?? value, min, max)];
    }));
    sizes.composer = clamp(stored.composer ?? COMPOSER_DEFAULT_SIZE, ...COMPOSER_LIMITS);
    return sizes;
  }

  function saveLayoutSizes(sizes) {
    writeStorage(RESIZE_STORAGE_KEY, JSON.stringify(sizes));
  }

  function applyLayoutSizes(sizes = loadLayoutSizes()) {
    document.documentElement.style.setProperty("--sidebar-width", `${sizes.sidebar}px`);
    document.documentElement.style.setProperty("--material-drawer-width", `${sizes.materialDrawer}px`);
    document.documentElement.style.setProperty("--input-area-height", `${sizes.composer}px`);
  }

  function ensureResizeHandles() {
    if (document.querySelector('.resize-handle[data-resize-target="sidebar"]')) return;
    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "resize-handle";
    handle.dataset.resizeTarget = "sidebar";
    handle.dataset.resizeAxis = "x";
    handle.setAttribute("aria-label", "拖动调整左侧任务栏宽度");
    handle.title = "拖动调整左侧任务栏宽度";
    document.body.appendChild(handle);
  }

  function bindResizeHandle(handle, targetKey, updateSize) {
    if (!handle || handle.dataset.bound === "true") return;
    handle.dataset.bound = "true";
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const axis = handle.dataset.resizeAxis === "y" ? "y" : "x";
      const startPos = axis === "y" ? event.clientY : event.clientX;
      const start = loadLayoutSizes();
      handle.setPointerCapture?.(event.pointerId);
      handle.classList.add("is-dragging");
      document.body.classList.add("is-resizing");

      function update(nextEvent) {
        const next = updateSize(start, startPos, nextEvent, axis);
        applyLayoutSizes(next);
        handle._nextSizes = next;
      }

      function finish() {
        if (handle._nextSizes) saveLayoutSizes(handle._nextSizes);
        handle.classList.remove("is-dragging");
        document.body.classList.remove("is-resizing");
        window.removeEventListener("pointermove", update);
        window.removeEventListener("pointerup", finish);
        window.removeEventListener("pointercancel", finish);
      }

      window.addEventListener("pointermove", update);
      window.addEventListener("pointerup", finish);
      window.addEventListener("pointercancel", finish);
    });
  }

  function bindResizableLayout() {
    applyLayoutSizes();
    ensureResizeHandles();
    document.querySelectorAll('.resize-handle[data-resize-target="sidebar"]').forEach((handle) => {
      bindResizeHandle(handle, "sidebar", (start, startPos, nextEvent) => ({
        sidebar: clamp(start.sidebar + nextEvent.clientX - startPos, ...RESIZE_LIMITS.sidebar),
        materialDrawer: start.materialDrawer,
        composer: start.composer,
      }));
    });
    bindResizeHandle(materialDrawerResizeHandle, "materialDrawer", (start, startPos, nextEvent) => ({
      sidebar: start.sidebar,
      materialDrawer: clamp(start.materialDrawer + startPos - nextEvent.clientX, ...RESIZE_LIMITS.materialDrawer),
      composer: start.composer,
    }));
    bindResizeHandle(composerResizeHandle, "composer", (start, startPos, nextEvent) => ({
      sidebar: start.sidebar,
      materialDrawer: start.materialDrawer,
      composer: clamp(start.composer + nextEvent.clientY - startPos, ...COMPOSER_LIMITS),
    }));
  }

  function taskStorageKey(namespace) {
    return `lawrag.tasks.${namespace}`;
  }

  function activeTaskStorageKey(namespace) {
    return `lawrag.active_task.${namespace}`;
  }

  function activeTaskSnapshotStorageKey(namespace) {
    return `lawrag.active_task_snapshot.${namespace}`;
  }

  function deletedSessionsStorageKey(namespace) {
    return `lawrag.deleted_sessions.${namespace}`;
  }

  function settingsStorageKey(namespace) {
    return `lawrag.settings.${namespace}`;
  }

  function normalizeSettings(values = {}) {
    const next = { ...DEFAULT_USER_SETTINGS, ...(values || {}) };
    next.include_web_default = Boolean(next.include_web_default);
    next.auto_expand_professional = Boolean(next.auto_expand_professional);
    next.show_retrieval_process = Boolean(next.show_retrieval_process);
    next.enable_long_memory = false;
    next.answer_detail = ["concise", "standard", "detailed"].includes(next.answer_detail) ? next.answer_detail : "standard";
    next.theme = normalizeThemeKey(next.theme) || readThemePreference();
    return next;
  }

  function loadLocalSettings(namespace = taskNamespace) {
    let parsed = {};
    try { parsed = JSON.parse(readStorage(settingsStorageKey(namespace)) || "{}"); } catch (_error) { parsed = {}; }
    userSettings = normalizeSettings(parsed);
    applySettingsToForm();
  }

  function saveLocalSettings(namespace = taskNamespace) {
    writeStorage(settingsStorageKey(namespace), JSON.stringify(userSettings));
    if (userSettings?.theme) writeStorage(THEME_STORAGE_KEY, userSettings.theme);
  }

  function applySettingsToForm() {
    if (settingAutoExpand) settingAutoExpand.checked = userSettings.auto_expand_professional;
    if (settingShowProcess) settingShowProcess.checked = userSettings.show_retrieval_process;
    settingAnswerDetailOptions.forEach((option) => { option.checked = option.value === userSettings.answer_detail; });
    if (settingTheme) settingTheme.value = userSettings.theme;
    document.body.classList.toggle("hide-process", !userSettings.show_retrieval_process);
    applyTheme(userSettings.theme);
    renderSettingsPreview();
  }

  function readSettingsForm() {
    const selectedDetail = Array.from(settingAnswerDetailOptions).find((option) => option.checked)?.value;
    return normalizeSettings({
      include_web_default: false,
      auto_expand_professional: Boolean(settingAutoExpand?.checked),
      show_retrieval_process: Boolean(settingShowProcess?.checked),
      answer_detail: selectedDetail || "standard",
      theme: settingTheme?.value || DEFAULT_USER_SETTINGS.theme,
    });
  }

  function normalizeRetrievalMode(value) {
    return ["local", "auto", "force"].includes(value) ? value : "auto";
  }

  function retrievalModeDescription(mode, actualWeb = null) {
    if (mode === "local") return "仅知识库：不会访问网页，只使用法律知识库和本次咨询材料。";
    if (mode === "force") return "强制联网：本次咨询会同时检索网页资料，并对引用进行标注。";
    if (actualWeb === true) return "智能联网：系统判断本题需要时效信息，本次已补充网页检索。";
    if (actualWeb === false) return "智能联网：系统判断知识库足够，本次未访问网页。";
    return "智能联网：系统会按问题需要决定是否检索网页。";
  }

  function renderRetrievalMode(actualWeb = null) {
    const mode = normalizeRetrievalMode(activeTask?.retrievalMode);
    retrievalModeButtons.forEach((button) => {
      const selected = button.dataset.retrievalMode === mode;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    retrievalModeStatuses.forEach((status) => { status.textContent = retrievalModeDescription(mode, actualWeb); });
  }

  function setRetrievalMode(mode) {
    if (!activeTask) return;
    activeTask.retrievalMode = normalizeRetrievalMode(mode);
    renderRetrievalMode();
    touchTask(activeTask);
  }

  function makeTask(session = "") {
    const id = `task_${globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`}`;
    return {
      id,
      sessionId: session || `session_${globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`}`,
      title: "新对话",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
      retrievalMode: "auto",
    };
  }

  function readStoredTasks(namespace) {
    let parsed = [];
    try { parsed = JSON.parse(readStorage(taskStorageKey(namespace)) || "[]"); } catch (_error) { parsed = []; }
    if (!Array.isArray(parsed)) parsed = [];
    return parsed.filter((task) => task && task.id && task.sessionId).map((task) => ({
      ...task,
      title: String(task.title || "新对话").slice(0, 42),
      messages: Array.isArray(task.messages) ? task.messages.slice(-80) : [],
      retrievalMode: normalizeRetrievalMode(task.retrievalMode),
    }));
  }

  function normalizeStoredMessage(message = {}) {
    if (!message || typeof message !== "object") return null;
    if (message.role === "user") {
      return {
        role: "user",
        content: String(message.content || "").slice(0, 4000),
        createdAt: message.createdAt || Date.now(),
      };
    }
    if (message.role === "assistant") {
      const storedAnswer = message.answer && typeof message.answer === "object"
        ? message.answer
        : {
            answer: String(message.answer || message.content || message.text || ""),
            case_analysis: Array.isArray(message.case_analysis) ? message.case_analysis : [],
            key_issues: Array.isArray(message.key_issues) ? message.key_issues : [],
            evidence_analysis: Array.isArray(message.evidence_analysis) ? message.evidence_analysis : [],
            defense_arguments: Array.isArray(message.defense_arguments) ? message.defense_arguments : [],
            action_steps: Array.isArray(message.action_steps) ? message.action_steps : [],
            document_checklist: Array.isArray(message.document_checklist) ? message.document_checklist : [],
            questions_to_confirm: Array.isArray(message.questions_to_confirm) ? message.questions_to_confirm : [],
            legal_basis: Array.isArray(message.legal_basis) ? message.legal_basis : [],
            related_cases: Array.isArray(message.related_cases) ? message.related_cases : [],
            suggestions: Array.isArray(message.suggestions) ? message.suggestions : [],
            risk_notice: String(message.risk_notice || ""),
          };
      const answerObject = {
        ...storedAnswer,
        answer: String(storedAnswer.answer || message.content || message.text || ""),
      };
      return {
        role: "assistant",
        answer: answerObject,
        content: String(message.content || answerObject.answer || ""),
        question: String(message.question || ""),
        meta: message.meta || {},
        sourceCount: Number(message.sourceCount || 0),
        sourceFiles: Array.isArray(message.sourceFiles) ? message.sourceFiles : [],
        sources: Array.isArray(message.sources) ? message.sources : [],
        thinkingEnabled: Boolean(message.thinkingEnabled),
        turnId: String(message.turnId || ""),
        createdAt: message.createdAt || message.timestamp || Date.now(),
        solution: message.solution || null,
        solutionCacheKey: String(message.solutionCacheKey || ""),
      };
    }
    return null;
  }

  function compactSourceForStorage(source = {}) {
    return {
      index: Number(source.index || 0),
      title: String(source.title || "未命名来源").slice(0, 120),
      label: String(source.label || "检索来源").slice(0, 64),
      source_type: String(source.source_type || "unknown"),
      collection: String(source.collection || "").slice(0, 64),
      file_name: String(source.file_name || "").slice(0, 120),
      content: String(source.content || "").slice(0, 240),
      url: String(source.url || "").slice(0, 200),
      article_number: String(source.article_number || "").slice(0, 40),
      law_name: String(source.law_name || "").slice(0, 120),
      source_id: String(source.source_id || "").slice(0, 80),
    };
  }

  function compactSolutionForStorage(solution = null) {
    if (!solution || typeof solution !== "object") return null;
    return {
      title: String(solution.title || "本次对话解决方案").slice(0, 80),
      scope: String(solution.scope || "").slice(0, 240),
      markdown: String(solution.markdown || "").slice(0, 4000),
      sources: Array.isArray(solution.sources) ? solution.sources.slice(0, 8).map(compactSourceForStorage) : [],
      cache_key: String(solution.cache_key || "").slice(0, 120),
    };
  }

  function compactTaskForStorage(task = {}) {
    return {
      ...task,
      title: String(task.title || "新对话").slice(0, 42),
      messages: Array.isArray(task.messages)
        ? task.messages.slice(-80).map((message) => {
          const normalized = normalizeStoredMessage(message);
          if (!normalized) return null;
          if (normalized.role === "assistant") {
            const answerText = String(normalized.answer?.answer || normalized.content || "");
            return {
              role: "assistant",
              content: answerText.slice(0, 4000),
              answer: {
                answer: answerText.slice(0, 4000),
                case_analysis: Array.isArray(normalized.answer?.case_analysis) ? normalized.answer.case_analysis.slice(0, 5) : [],
                key_issues: Array.isArray(normalized.answer?.key_issues) ? normalized.answer.key_issues.slice(0, 3) : [],
                evidence_analysis: Array.isArray(normalized.answer?.evidence_analysis) ? normalized.answer.evidence_analysis.slice(0, 4) : [],
                defense_arguments: Array.isArray(normalized.answer?.defense_arguments) ? normalized.answer.defense_arguments.slice(0, 2) : [],
                action_steps: Array.isArray(normalized.answer?.action_steps) ? normalized.answer.action_steps.slice(0, 4) : [],
                document_checklist: Array.isArray(normalized.answer?.document_checklist) ? normalized.answer.document_checklist.slice(0, 5) : [],
                questions_to_confirm: Array.isArray(normalized.answer?.questions_to_confirm) ? normalized.answer.questions_to_confirm.slice(0, 3) : [],
                legal_basis: Array.isArray(normalized.answer?.legal_basis) ? normalized.answer.legal_basis.slice(0, 4) : [],
                related_cases: Array.isArray(normalized.answer?.related_cases) ? normalized.answer.related_cases.slice(0, 2) : [],
                suggestions: Array.isArray(normalized.answer?.suggestions) ? normalized.answer.suggestions.slice(0, 2) : [],
                risk_notice: String(normalized.answer?.risk_notice || "").slice(0, 1000),
              },
              question: String(normalized.question || "").slice(0, 4000),
              meta: { elapsed_ms: Number(normalized.meta?.elapsed_ms || 0) },
              sourceCount: Number(normalized.sourceCount || 0),
              sourceFiles: Array.isArray(normalized.sourceFiles) ? normalized.sourceFiles.slice(0, 6).map((item) => String(item || "").slice(0, 120)).filter(Boolean) : [],
              sources: Array.isArray(normalized.sources) ? normalized.sources.slice(0, 8).map(compactSourceForStorage) : [],
              thinkingEnabled: Boolean(normalized.thinkingEnabled),
              turnId: String(normalized.turnId || "").slice(0, 80),
              createdAt: Number(normalized.createdAt || Date.now()),
              solution: compactSolutionForStorage(normalized.solution),
              solutionCacheKey: String(normalized.solutionCacheKey || "").slice(0, 120),
            };
          }
          return normalized;
        }).filter(Boolean)
        : [],
    };
  }

  function findAssistantMessageIndex(task = {}, message = {}) {
    const messages = Array.isArray(task?.messages) ? task.messages : [];
    const turnId = String(message.turnId || "").trim();
    if (turnId) {
      for (let index = messages.length - 1; index >= 0; index -= 1) {
        const item = messages[index];
        if (item?.role === "assistant" && String(item.turnId || "").trim() === turnId) return index;
      }
    }
    const question = String(message.question || "").trim();
    if (question) {
      for (let index = messages.length - 1; index >= 0; index -= 1) {
        const item = messages[index];
        if (item?.role === "assistant" && String(item.question || "").trim() === question) return index;
      }
    }
    return -1;
  }

  function writeTaskMessage(task, message, { replace = false, touch = true } = {}) {
    if (!task || !message) return;
    const messages = Array.isArray(task.messages) ? task.messages.slice() : [];
    const index = message.role === "assistant" && replace ? findAssistantMessageIndex(task, message) : -1;
    if (index >= 0) {
      const previous = messages[index];
      const previousAnswer = String(previous?.answer?.answer || previous?.content || previous?.text || "").trim();
      const nextAnswer = String(message?.answer?.answer || message?.content || message?.text || "").trim();
      if (message.role === "assistant" && previousAnswer && !nextAnswer) return;
      messages[index] = message;
    }
    else messages.push(message);
    task.messages = messages.slice(-80);
    if (touch) touchTask(task);
    else {
      task.updatedAt = Date.now();
      saveTasks();
    }
  }

  function hasMeaningfulTask(tasks = []) {
    return Array.isArray(tasks) && tasks.some((task) => Array.isArray(task?.messages) && task.messages.length > 0);
  }

  function loadTaskState(namespace) {
    const normalizedNamespace = String(namespace || "guest").replace(/[^0-9A-Za-z_-]/g, "_");
    if (normalizedNamespace === "guest") {
      const task = makeTask();
      return {activeTaskId: task.id, tasks: [task]};
    }
    let tasks = readStoredTasks(normalizedNamespace);
    const activeNamespace = normalizedNamespace;
    if (!tasks.length) tasks = [makeTask()];
    const snapshotRaw = readStorage(activeTaskSnapshotStorageKey(activeNamespace));
    if (snapshotRaw) {
      try {
        const snapshot = JSON.parse(snapshotRaw);
        const snapshotTask = snapshot && snapshot.id && snapshot.sessionId ? {
          ...snapshot,
          title: String(snapshot.title || "新对话").slice(0, 42),
          messages: Array.isArray(snapshot.messages) ? snapshot.messages.map(normalizeStoredMessage).filter(Boolean).slice(-80) : [],
        } : null;
        if (snapshotTask?.messages?.length) {
          const matchIndex = tasks.findIndex((task) => task.id === snapshotTask.id || String(task.sessionId || "") === String(snapshotTask.sessionId || ""));
          if (matchIndex >= 0) {
            const current = tasks[matchIndex];
            tasks[matchIndex] = {
              ...current,
              ...snapshotTask,
              messages: mergeServerMessages(current.messages, snapshotTask.messages),
              updatedAt: Math.max(taskTimestamp(current.updatedAt), taskTimestamp(snapshotTask.updatedAt || snapshotTask.createdAt), Date.now()),
            };
          }
        }
      } catch (_error) {
        /* snapshot may be incomplete */
      }
    }
    const activeTaskId = readStorage(activeTaskStorageKey(activeNamespace)) || tasks[0].id;
    return {
      activeTaskId,
      tasks: tasks.slice(0, MAX_TASKS),
    };
  }

  // ============ 关键修复 1: saveTasks 添加日志 ============
  function saveTasks() {
    if (taskNamespace === "guest") return;
    try {
      const taskData = tasks.slice(0, MAX_TASKS).map((task) => compactTaskForStorage(task));
      const serialized = JSON.stringify(taskData);
      window.localStorage?.setItem(taskStorageKey(taskNamespace), serialized);
      window.localStorage?.setItem(activeTaskStorageKey(taskNamespace), activeTaskId);
      if (activeTask) window.localStorage?.setItem(activeTaskSnapshotStorageKey(taskNamespace), JSON.stringify(compactTaskForStorage(activeTask)));
      console.log('[saveTasks] 已保存', {
        namespace: taskNamespace,
        taskCount: taskData.length,
        activeTaskId: activeTaskId,
        messageCount: activeTask?.messages?.length || 0,
        lastMessage: activeTask?.messages?.[activeTask?.messages?.length - 1]?.content?.slice(0, 30) || '无'
      });
    } catch (_error) {
      console.warn('[saveTasks] 保存失败', _error);
    }
  }

  function taskTitle(task) {
    const firstQuestion = task.messages?.find((message) => message.role === "user")?.content;
    return String(firstQuestion || task.title || "新对话").replace(/\s+/g, " ").trim().slice(0, 30) || "新对话";
  }

  function renderTaskList() {
    if (taskCount) taskCount.textContent = String(tasks.length);
    if (taskList) {
      taskList.innerHTML = tasks.map((task) => `
        <div class="task-item${task.id === activeTaskId ? " is-active" : ""}" role="listitem">
          <button class="task-open" type="button" data-task-id="${escapeHtml(task.id)}" title="${escapeHtml(taskTitle(task))}">
            <span class="task-item-icon" aria-hidden="true">○</span><span class="task-item-title">${escapeHtml(taskTitle(task))}</span>
          </button>
          <button class="task-delete" type="button" data-delete-task-id="${escapeHtml(task.id)}" aria-label="删除对话：${escapeHtml(taskTitle(task))}" title="删除对话">${DELETE_ICON}</button>
        </div>`).join("");
      taskList.querySelectorAll("[data-task-id]").forEach((button) => {
        button.addEventListener("click", () => switchTask(button.dataset.taskId));
      });
      taskList.querySelectorAll("[data-delete-task-id]").forEach((button) => {
        button.addEventListener("click", () => deleteTask(button.dataset.deleteTaskId));
      });
    }
    if (taskSelect) {
      taskSelect.innerHTML = tasks.map((task) => `<option value="${escapeHtml(task.id)}">${escapeHtml(taskTitle(task))}</option>`).join("");
      taskSelect.value = activeTaskId;
    }
  }

  function clearConversationView() {
    conversation.innerHTML = `<section class="empty-state" id="empty-state">
      <div class="empty-mark" data-scale-icon></div>
      <p class="eyebrow">知途 · 从事实说起</p>
      <h1>知权利，明路径。</h1>
      <p class="empty-copy">描述事情的经过和你希望解决的问题。依据事实与法律，理解权利义务，梳理合法处理方式。</p>
      <div class="quick-prompts">
        <button type="button" data-prompt="离婚时夫妻共同财产如何分割？"><span>离婚时夫妻共同财产如何分割？</span><span>↗</span></button>
        <button type="button" data-prompt="公司拖欠工资，我应该准备哪些证据？"><span>公司拖欠工资，我应该准备哪些证据？</span><span>↗</span></button>
        <button type="button" data-prompt="借款到期不还，可以怎么依法追讨？"><span>借款到期不还，可以怎么依法追讨？</span><span>↗</span></button>
      </div>
    </section>`;
    conversation.classList.add("empty-conversation");
    conversation.querySelectorAll("[data-scale-icon]").forEach((element) => { element.innerHTML = BRAND_ICON; });
    conversation.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => submitQuestion(button.dataset.prompt));
    });
  }

  function activateTaskNamespace(namespace) {
    identityVersion += 1;
    window.clearTimeout(settingsSaveTimer);
    closeSettings();
    closeDiagnostics();
    if (diagnosticsOutput) diagnosticsOutput.textContent = "";
    if (activityList) activityList.textContent = "";
    questionController?.abort();
    questionController = null;
    queuedQuestions = [];
    setLoading(false);
    input.value = "";
    homeInput.value = "";
    homeSubmit.disabled = true;
    closeSourceDrawer();
    closeAccountMenu();
    const state = loadTaskState(namespace);
    taskNamespace = String(namespace || "guest").replace(/[^0-9A-Za-z_-]/g, "_");
    loadLocalSettings(taskNamespace);
    tasks = state.tasks;
    activeTaskId = state.activeTaskId || tasks[0].id;
    activeTask = tasks.find((task) => task.id === activeTaskId) || tasks[0];
    activeTaskId = activeTask.id;
    sessionId = activeTask.sessionId;
    saveTasks();
    renderTaskList();
    renderTaskMessages();
    renderRetrievalMode();
    renderWorkspaceFiles([]);
  }

  function clearActiveTaskSnapshot(namespace = taskNamespace) {
    try { window.localStorage?.removeItem(activeTaskSnapshotStorageKey(namespace)); } catch (_error) { /* ignore */ }
  }

  function renderTaskMessages() {
    resetWorkflow();
    clearConversationView();
    const messages = Array.isArray(activeTask?.messages) ? activeTask.messages.map((message) => normalizeStoredMessage(message)).filter(Boolean) : [];
    if (!messages.length) return;
    activeTask.messages = messages;
    messages.forEach((message) => {
      if (message.role === "user") {
        appendUserMessage(message.content, { persist: false, timestamp: message.createdAt });
      } else if (message.role === "assistant") {
        appendAssistantMessage(message.answer, message.meta, message.sourceCount, {
          persist: false,
          timestamp: message.createdAt,
          sourceFiles: message.sourceFiles,
          sources: message.sources,
          thinkingEnabled: message.thinkingEnabled,
          solution: message.solution,
        });
      }
    });
  }

  function taskTimestamp(value) {
    if (typeof value === "number") return Number.isFinite(value) ? value : 0;
    const parsed = Date.parse(String(value || ""));
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function sameStoredMessage(left, right) {
    if (!left || !right || left.role !== right.role) return false;
    if (left.role === "user") return String(left.content || "") === String(right.content || "");
    const leftQuestion = String(left.question || "").trim();
    const rightQuestion = String(right.question || "").trim();
    if (leftQuestion && rightQuestion) return leftQuestion === rightQuestion;
    return String(left.content || left.answer?.answer || "") === String(right.content || right.answer?.answer || "");
  }

  function preferStoredMessage(localMessage, serverMessage) {
    if (!localMessage) return serverMessage;
    if (!serverMessage) return localMessage;
    if (serverMessage.role !== "assistant") return localMessage;
    const localText = String(localMessage.content || localMessage.answer?.answer || "");
    const serverText = String(serverMessage.content || serverMessage.answer?.answer || "");
    if (serverText.length <= localText.length) return localMessage;
    return {
      ...localMessage,
      ...serverMessage,
      sources: serverMessage.sources?.length ? serverMessage.sources : localMessage.sources,
      sourceFiles: serverMessage.sourceFiles?.length ? serverMessage.sourceFiles : localMessage.sourceFiles,
      solution: localMessage.solution || serverMessage.solution || null,
    };
  }

  function mergeServerMessages(localMessages, serverMessages) {
    const local = (Array.isArray(localMessages) ? localMessages : []).map(normalizeStoredMessage).filter(Boolean);
    const server = (Array.isArray(serverMessages) ? serverMessages : []).map(normalizeStoredMessage).filter(Boolean);
    const usedLocal = new Set();
    const merged = server.map((serverMessage) => {
      const localIndex = local.findIndex((localMessage, index) => !usedLocal.has(index) && sameStoredMessage(localMessage, serverMessage));
      if (localIndex < 0) return serverMessage;
      usedLocal.add(localIndex);
      return preferStoredMessage(local[localIndex], serverMessage);
    });
    local.forEach((localMessage, index) => {
      if (!usedLocal.has(index) && !merged.some((message) => sameStoredMessage(message, localMessage))) merged.push(localMessage);
    });
    return merged.slice(-80);
  }

  function taskHasAssistantAnswer(task = {}) {
    return Array.isArray(task?.messages) && task.messages.some((message) => {
      if (message?.role !== "assistant") return false;
      const answerText = String(message.answer?.answer || message.content || message.text || "").trim();
      return Boolean(answerText);
    });
  }

  function removeStoredSessionFromNamespace(namespace, sessionIdToRemove) {
    const targetNamespace = String(namespace || "guest").replace(/[^0-9A-Za-z_-]/g, "_");
    const targetSessionId = String(sessionIdToRemove || "").trim();
    if (!targetSessionId) return;
    const storedTasks = readStoredTasks(targetNamespace).filter((task) => String(task.sessionId || "") !== targetSessionId);
    writeStorage(taskStorageKey(targetNamespace), JSON.stringify(storedTasks.slice(0, MAX_TASKS)));
    const activeId = readStorage(activeTaskStorageKey(targetNamespace));
    if (activeId === targetSessionId) {
      const nextActive = storedTasks[0]?.id || "";
      if (nextActive) writeStorage(activeTaskStorageKey(targetNamespace), nextActive);
      else window.localStorage?.removeItem(activeTaskStorageKey(targetNamespace));
    }
    const deleted = readDeletedSessions(targetNamespace);
    deleted.add(targetSessionId);
    writeStorage(deletedSessionsStorageKey(targetNamespace), JSON.stringify(Array.from(deleted).slice(-MAX_TASKS * 2)));
  }

  async function restoreServerConversationHistory() {
    if (!authenticated || !currentUser) return;
    const requestIdentity = identityVersion;
    try {
      const response = await fetch(backendUrl("/api/v1/legal/sessions?limit=20"), { credentials: "include", cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json().catch(() => null);
      if (requestIdentity !== identityVersion || !authenticated) return;
      const summaries = Array.isArray(payload?.data) ? payload.data : [];
      if (!summaries.length) return;

      const deletedSessions = readDeletedSessions();
      const bySession = new Map(tasks.map((task) => [String(task.sessionId), task]));
      const remoteTasks = [];
      summaries.forEach((summary) => {
        const remoteSessionId = String(summary?.session_id || "").trim();
        const remoteMessages = Array.isArray(summary?.messages) ? summary.messages : [];
        if (!remoteSessionId || deletedSessions.has(remoteSessionId) || !remoteMessages.length) return;
        let task = bySession.get(remoteSessionId);
        if (!task) {
          task = makeTask(remoteSessionId);
          task.title = String(summary.title || "新对话").slice(0, 42);
          task.createdAt = taskTimestamp(summary.messages?.[0]?.createdAt) || Date.now();
          task.updatedAt = taskTimestamp(summary.updated_at) || task.createdAt;
          task.messages = remoteMessages.map(normalizeStoredMessage).filter(Boolean).slice(-80);
          tasks.push(task);
          bySession.set(remoteSessionId, task);
        } else {
          const serverMessages = remoteMessages.map(normalizeStoredMessage).filter(Boolean).slice(-80);
          task.messages = mergeServerMessages(task.messages, serverMessages);
        }
        remoteTasks.push(task);
        if ((!task.title || task.title === "新对话") && summary.title) {
          task.title = String(summary.title).slice(0, 42);
        }
        task.updatedAt = Math.max(taskTimestamp(task.updatedAt), taskTimestamp(summary.updated_at));
      });

      const remoteSessionIds = new Set(remoteTasks.map((task) => String(task?.sessionId || "").trim()).filter(Boolean));
      const localDrafts = tasks.filter((task) => {
        const session = String(task?.sessionId || "").trim();
        return task && task.id && session && !remoteSessionIds.has(session) && Array.isArray(task.messages) && task.messages.length;
      });

      tasks = [...remoteTasks, ...localDrafts]
        .filter((task) => task && task.id && task.sessionId)
        .sort((left, right) => taskTimestamp(right.updatedAt) - taskTimestamp(left.updatedAt))
        .slice(0, MAX_TASKS);

      const activeSessionId = String(activeTask?.sessionId || sessionId || "").trim();
      const hasLocalDraft = loading || (Array.isArray(activeTask?.messages) && activeTask.messages.some((message) => message.role === "user"));
      let nextActive = tasks.find((task) => task.id === activeTaskId)
        || tasks.find((task) => String(task.sessionId || "").trim() === activeSessionId)
        || tasks[0]
        || null;
      if (!hasLocalDraft && !taskHasAssistantAnswer(nextActive)) {
        nextActive = tasks.find((task) => remoteSessionIds.has(String(task.sessionId || "").trim()) && taskHasAssistantAnswer(task))
          || tasks.find((task) => remoteSessionIds.has(String(task.sessionId || "").trim()))
          || tasks.find((task) => taskHasAssistantAnswer(task))
          || nextActive;
      }
      if (nextActive) {
        activeTask = nextActive;
        activeTaskId = nextActive.id;
        sessionId = nextActive.sessionId;
      }
      activeTask = tasks.find((task) => task.id === activeTaskId) || tasks[0];
      activeTaskId = activeTask.id;
      sessionId = activeTask.sessionId;
      saveTasks();
      renderTaskList();
      renderTaskMessages();
    } catch (error) {
      console.warn("服务端会话历史恢复失败，继续使用本机缓存", error);
    }
  }

  function touchTask(task = activeTask) {
    if (!task) return;
    task.updatedAt = Date.now();
    tasks = [task, ...tasks.filter((item) => item.id !== task.id)].slice(0, MAX_TASKS);
    renderTaskList();
    saveTasks();
  }

  function createTask() {
    clearActiveTaskSnapshot();
    activeTask = makeTask();
    activeTaskId = activeTask.id;
    sessionId = activeTask.sessionId;
    tasks = [activeTask, ...tasks.filter((task) => task.id !== activeTask.id)].slice(0, MAX_TASKS);
    saveTasks();
    renderTaskList();
    renderTaskMessages();
    renderRetrievalMode();
    renderWorkspaceFiles([]);
    if (authenticated) void refreshWorkspaceFiles();
    navigate("ask");
    input.focus();
  }

  function switchTask(taskId) {
    if (!taskId) return;
    const next = tasks.find((task) => task.id === taskId);
    if (!next) return;
    if (taskId === activeTaskId) {
      renderRetrievalMode();
      navigate("ask");
      input.focus();
      return;
    }
    clearActiveTaskSnapshot();
    activeTask = next;
    activeTaskId = next.id;
    sessionId = next.sessionId;
    saveTasks();
    renderTaskList();
    renderTaskMessages();
    renderRetrievalMode();
    renderWorkspaceFiles([]);
    if (authenticated) void refreshWorkspaceFiles();
    navigate("ask");
    input.focus();
  }

  function hideWorkspaceFileLocally(documentId) {
    const target = String(documentId || "").trim();
    if (!target) return;
    const current = Array.isArray(window._lawragWorkspaceFiles) ? window._lawragWorkspaceFiles : [];
    renderWorkspaceFiles(current.filter((file) => String(file?.document_id || "") !== target));
  }

  async function deleteTask(taskId) {
    if (loading || !taskId) return;
    const target = tasks.find((task) => task.id === taskId);
    if (!target) return;
    const wasActive = target.id === activeTaskId;
    if (wasActive) {
      try { window.localStorage?.removeItem(activeTaskSnapshotStorageKey(taskNamespace)); } catch (_error) { /* ignore */ }
    }
    tasks = tasks.filter((task) => task.id !== target.id);
    markSessionDeleted(target.sessionId);
    removeStoredSessionFromNamespace("guest", target.sessionId);
    if (taskNamespace !== "guest") removeStoredSessionFromNamespace(taskNamespace, target.sessionId);
    if (!tasks.length) tasks = [makeTask()];
    if (wasActive) {
      activeTask = tasks[0];
      activeTaskId = activeTask.id;
      sessionId = activeTask.sessionId;
    }
    saveTasks();
    renderTaskList();
    renderTaskMessages();
    renderWorkspaceFiles([]);
    if (fileStatus) {
      fileStatus.className = "file-status is-success";
      fileStatus.textContent = "对话已从本机移除，服务器记录会自动同步清理";
    }
    fetch(
      backendUrl(`/api/v1/legal/sessions/${encodeURIComponent(target.sessionId)}`),
      { method: "DELETE", credentials: "include" },
    ).then(async (response) => {
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        console.warn("对话服务器清理稍后重试", getErrorMessage(payload, `HTTP ${response.status}`));
      }
    }).catch((error) => console.warn("对话服务器清理稍后重试", error));
    if (authenticated) void refreshWorkspaceFiles();
    input.focus();
  }

  function backendUrl(path) {
    return `${apiBaseUrl}${path}`;
  }

  document.querySelectorAll("[data-scale-icon]").forEach((element) => {
    element.innerHTML = BRAND_ICON;
  });
  sendButton.innerHTML = SEND_ICON;

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatNumber(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? number.toLocaleString("zh-CN") : "0";
  }

  function userAvatarText(user = currentUser) {
    const name = String(user?.username || "").trim();
    if (!name) return "未";
    return name.slice(0, 1).toUpperCase();
  }

  function workspaceFilePreview(file) {
    const preview = String(file?.extraction_preview || "").trim();
    return preview.length > 180 ? `${preview.slice(0, 180)}…` : preview;
  }

  function workspaceFileMeta(file) {
    const parts = [];
    const mediaLabels = { text: "文本", image: "图片", audio: "音频", video: "视频", file: "文件" };
    if (file.media_type && mediaLabels[file.media_type]) parts.push(mediaLabels[file.media_type]);
    else if (file.document_type) parts.push(file.document_type);
    if (file.extraction_method === "multimodal") parts.push("多模态分析");
    else if (file.extraction_method) parts.push(file.extraction_method);
    if (file.multimodal_used) parts.push(file.multimodal_status === "success" ? "多模态成功" : "多模态待检查");
    if (file.ocr_used) parts.push(file.ocr_status === "success" ? "OCR 成功" : "OCR 待检查");
    if (file.extracted_char_count) parts.push(`${formatNumber(file.extracted_char_count)} 字`);
    if (file.status === "processing") parts.push("后台解析中");
    else if (file.status === "stored") parts.push("等待解析内容");
    else if (file.status === "failed") parts.push("解析失败");
    else if (file.status === "ready") parts.push("解析完成");
    else if (file.vector_index_status === "not_used") parts.push("MySQL 已保存");
    return parts.join(" · ");
  }

  function formatInline(value) {
    const escaped = escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
    const linkTokens = [];
    const linkPattern = "(?:https?:\\/\\/|\\/\\/|\\/|\\.\\.?\\/)[^\\s)]+";
    const withMarkdownLinks = escaped.replace(new RegExp(`\\[([^\\]]+)]\\((${linkPattern})\\)`, "g"), (_match, label, url) => {
      const token = `__LINK_TOKEN_${linkTokens.length}__`;
      const href = normalizeExternalUrl(url);
      if (!href) return label;
      linkTokens.push(`<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer noopener" referrerpolicy="no-referrer">${label}</a>`);
      return token;
    });
    const withAutoLinks = withMarkdownLinks.replace(/(^|[^\w/])((?:https?:\/\/|\/\/)[^\s<]+)/g, (match, prefix, url) => {
      const token = `__URL_TOKEN_${linkTokens.length}__`;
      const cleanUrl = normalizeExternalUrl(url);
      if (!cleanUrl) return match;
      linkTokens.push(`${prefix}<a href="${escapeHtml(cleanUrl)}" target="_blank" rel="noreferrer noopener" referrerpolicy="no-referrer">${escapeHtml(cleanUrl)}</a>`);
      return token;
    });
    return linkTokens.reduce((html, link, index) => html.replace(`__LINK_TOKEN_${index}__`, link).replace(`__URL_TOKEN_${index}__`, link), withAutoLinks);
  }

  function normalizeExternalUrl(value) {
    const raw = String(value || "").trim();
    const clean = raw
      .replaceAll("&amp;", "&")
      .replaceAll("&quot;", '"')
      .replaceAll("&#039;", "'")
      .replace(/[\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u3001)\]}>'"`]+$/g, "");
    if (!clean) return "";
    try {
      const candidate = clean.startsWith("//") ? `${window.location.protocol}${clean}` : clean;
      const url = new URL(candidate, window.location.origin);
      return /^https?:$/.test(url.protocol) ? url.href : "";
    } catch (_error) {
      return "";
    }
  }

  function escapeRegExp(value) {
    return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  const PUBLIC_COLLECTION_LABELS = {
    civil_code_articles: "民法典法条库",
    civil_interpretations: "民事司法解释库",
    civil_cases: "民事案例库",
    civil_elements: "裁判要素库",
    civil_evidence: "证据规则库",
    civil_processes: "办案流程库",
    civil_questions: "民法典问题模板库",
    civil_citations: "民法典引用关系库",
  };

  function sourceDisplayLabel(source = {}) {
    if (source.source_label) return String(source.source_label);
    if (source.source_type === "private" || source.source_type === "user_upload") return "用户上传材料";
    if (source.source_type === "web") return "网页资料";
    if (source.collection && PUBLIC_COLLECTION_LABELS[source.collection]) return PUBLIC_COLLECTION_LABELS[source.collection];
    if (source.source_type === "public") return "公共法律库";
    return "检索来源";
  }

  function sourceDisplayUrl(source = {}) {
    const direct = String(source?.url || source?.link || "").trim();
    if (direct) return normalizeExternalUrl(direct);
    const sourceId = String(source?.source_id || "").trim();
    if (source.source_type === "web" && /^https?:\/\//i.test(sourceId)) return normalizeExternalUrl(sourceId);
    return "";
  }

  function isCaseLikeSource(source = {}) {
    if (source.collection === "civil_cases") return true;
    const text = `${source.title || ""}\n${source.content || ""}`;
    return /(案例|判决书|裁定书|纠纷|争议案|民事判决|民事裁定|法院)/.test(text);
  }

  function extractReferenceLabels(source = {}) {
    const labels = new Set();
    extractArticleLabels(source).forEach((label) => labels.add(label));
    const title = String(source.title || "").trim();
    const isGenericTitle = /^(公共法律库|公共法律资料|检索来源|用户上传材料|网页资料)$/.test(title);
    const isCompactTitle = title.length >= 4 && title.length <= 34 && !/[。；;，,]/.test(title);
    if (title && !isGenericTitle && isCompactTitle) labels.add(title);
    [source.case_number, source.source_case_number, source.document_number].forEach((value) => {
      const text = String(value || "").trim();
      if (text) labels.add(text);
    });
    return [...labels].filter((label) => label.length >= 3);
  }

  function normalizeSourceList(sources = [], citations = []) {
    const rows = Array.isArray(sources) && sources.length ? sources : (Array.isArray(citations) ? citations : []);
    const citationRows = Array.isArray(citations) ? citations : [];
    return rows.slice(0, 8).map((source, index) => {
      const citation = citationRows.find((item) => item?.source_id && item.source_id === source?.source_id) || {};
      return {
        index,
        title: source?.title || source?.file_name || citation.title || "未命名来源",
        label: sourceDisplayLabel({ ...citation, ...source }),
        source_type: source?.source_type || citation.source_type || "unknown",
        collection: source?.collection || citation.collection || "",
        file_name: source?.file_name || citation.file_name || "",
        content: String(source?.content || source?.summary || source?.text || citation.content || "").trim(),
        url: sourceDisplayUrl({ ...citation, ...source }),
        article_number: source?.article_number || citation.article_number || "",
        law_name: source?.law_name || citation.law_name || "",
        source_id: source?.source_id || citation.source_id || "",
      };
    });
  }

  function normalizeStoredSources(sources = []) {
    return normalizeSourceList(sources).map((source) => ({
      ...source,
      content: source.content.slice(0, 3200),
    }));
  }

  function compactLine(value, limit = 72) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
  }

  function renderSourceStrip(sources = []) {
    const rows = normalizeSourceList(sources).filter((source) => source.source_type !== "unknown");
    if (!rows.length) return "";
    return `<div class="source-strip" aria-label="本次回答参考来源">${rows.slice(0, 8).map((source, index) => {
      const title = compactLine(source.title || source.file_name || source.label || `来源 ${index + 1}`, 28);
      const label = source.article_number ? `第${String(source.article_number).replace(/^第|条$/g, "")}条` : title;
      return `<button type="button" class="source-strip-link" data-source-index="${index}" title="查看来源：${escapeHtml(title)}"><span>来源${index + 1}</span>${escapeHtml(label)}</button>`;
    }).join("")}</div>`;
  }

  function sourcePreview(source, limit = 140) {
    return compactLine(source?.content || source?.title || "点击查看来源内容", limit);
  }

  function ensureSourceDrawer() {
    let backdrop = document.getElementById("source-drawer-backdrop");
    if (backdrop) return backdrop;
    backdrop = document.createElement("div");
    backdrop.className = "source-drawer-backdrop";
    backdrop.id = "source-drawer-backdrop";
    backdrop.hidden = true;
    backdrop.innerHTML = `
      <aside class="source-drawer" role="dialog" aria-modal="true" aria-labelledby="source-drawer-title">
        <div class="source-drawer-top">
          <div>
            <p class="eyebrow source-drawer-kicker">引用来源</p>
            <h2 id="source-drawer-title">来源详情</h2>
          </div>
          <div class="source-drawer-actions" hidden>
            <button type="button" data-solution-download="pdf">下载 PDF</button>
            <button type="button" data-solution-download="word">下载 Word</button>
          </div>
          <button class="source-drawer-close" type="button" aria-label="关闭面板">×</button>
        </div>
        <div class="source-drawer-meta"></div>
        <article class="source-drawer-content"></article>
      </aside>`;
    document.body.appendChild(backdrop);
    backdrop.addEventListener("click", (event) => {
      const downloadButton = event.target.closest("[data-solution-download]");
      if (downloadButton) {
        downloadLegalSolution(downloadButton.dataset.solutionDownload, backdrop._solutionMarkdown || "");
        return;
      }
      if (event.target === backdrop || event.target.closest(".source-drawer-close")) closeSourceDrawer();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !backdrop.hidden) closeSourceDrawer();
    });
    return backdrop;
  }

  function openSourceDrawer(source = {}) {
    const backdrop = ensureSourceDrawer();
    const title = backdrop.querySelector("#source-drawer-title");
    const kicker = backdrop.querySelector(".source-drawer-kicker");
    const actions = backdrop.querySelector(".source-drawer-actions");
    const meta = backdrop.querySelector(".source-drawer-meta");
    const content = backdrop.querySelector(".source-drawer-content");
    backdrop.classList.remove("is-solution-drawer");
    backdrop._solutionMarkdown = "";
    if (kicker) kicker.textContent = "引用来源";
    if (actions) actions.hidden = true;
    if (title) title.textContent = source.title || "来源详情";
    if (meta) {
      const sourceUrl = sourceDisplayUrl(source);
      meta.innerHTML = [source.label, source.file_name, sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer noopener" referrerpolicy="no-referrer">打开原网页</a>` : ""]
        .filter(Boolean)
        .map((item) => `<span>${typeof item === "string" && item.startsWith("<a ") ? item : escapeHtml(item)}</span>`)
        .join("");
    }
    if (content) content.innerHTML = renderMarkdown(source.content || "当前来源没有可展开的正文片段，只能以标题和检索标签辅助核对。");
    backdrop.hidden = false;
    document.body.classList.add("source-drawer-open");
  }

  function openSolutionDrawer(solution = {}) {
    const backdrop = ensureSourceDrawer();
    const title = backdrop.querySelector("#source-drawer-title");
    const kicker = backdrop.querySelector(".source-drawer-kicker");
    const actions = backdrop.querySelector(".source-drawer-actions");
    const meta = backdrop.querySelector(".source-drawer-meta");
    const content = backdrop.querySelector(".source-drawer-content");
    const markdown = String(solution.markdown || "暂时没有可查看的解决方案。");
    backdrop.classList.add("is-solution-drawer");
    backdrop._solutionMarkdown = markdown;
    if (kicker) kicker.textContent = "法律解决方案";
    if (actions) actions.hidden = false;
    if (title) title.textContent = solution.title || "本次对话解决方案";
    if (meta) {
      meta.innerHTML = "";
    }
    if (content) content.innerHTML = renderAnswerMarkdown(markdown, solution.sources || []);
    bindSourceInteractions(backdrop, solution.sources || []);
    backdrop.hidden = false;
    document.body.classList.add("source-drawer-open");
  }

  function closeSourceDrawer() {
    const backdrop = document.getElementById("source-drawer-backdrop");
    if (backdrop) backdrop.hidden = true;
    document.body.classList.remove("source-drawer-open");
  }

  function extractArticleLabels(source = {}) {
    const labels = new Set();
    const articleNumber = String(source.article_number || "").trim();
    if (articleNumber) labels.add(`第${articleNumber.replace(/^第|条$/g, "")}条`);
    const sourceIdMatch = String(source.source_id || "").match(/article[_:-]?(\d{1,5})/i);
    if (sourceIdMatch) labels.add(`第${sourceIdMatch[1]}条`);
    const text = `${source.title || ""}\n${source.content || ""}`;
    const matches = text.match(/第[一二三四五六七八九十百千万亿零〇两\d\s]+条/g) || [];
    matches.slice(0, 4).forEach((match) => labels.add(match.replace(/\s+/g, "")));
    const digitMatches = text.match(/(?:民法典|第)\s*(\d{1,5})\s*条/g) || [];
    digitMatches.slice(0, 4).forEach((match) => {
      const number = match.match(/\d{1,5}/)?.[0];
      if (number) labels.add(`第${number}条`);
    });
    return [...labels].filter((label) => label.length >= 3);
  }

  function linkLegalReferences(html, sources = []) {
    const references = [];
    const addReference = (label, index, url = "", exactCitation = false) => {
      const rawLabel = String(label || "");
      const renderedLabel = escapeHtml(rawLabel);
      if (!renderedLabel) return;
      references.push({ label: renderedLabel, index, url, exactCitation });
    };
    sources.forEach((source, index) => {
      if (!source || (source.source_type !== "public" && source.source_type !== "web")) return;
      extractReferenceLabels(source).forEach((label) => addReference(label, index, sourceDisplayUrl(source)));
    });
    sources.forEach((source, index) => {
      if (!source || (source.source_type !== "public" && source.source_type !== "web")) return;
      [`[${index + 1}]`, `【${index + 1}】`].forEach((label) => {
        addReference(label, index, "", true);
      });
    });
    const uniqueMap = new Map();
    references.forEach((item) => {
      const current = uniqueMap.get(item.label);
      if (!current || (!current.url && item.url)) uniqueMap.set(item.label, item);
    });
    const unique = [...uniqueMap.values()].sort((first, second) => second.label.length - first.label.length);
    if (!unique.length) return html;
    let inInteractiveTag = false;
    return String(html || "").split(/(<[^>]+>)/g).map((segment) => {
      if (segment.startsWith("<")) {
        if (/^<\/(a|button)\b/i.test(segment)) inInteractiveTag = false;
        else if (/^<(a|button)\b/i.test(segment)) inInteractiveTag = true;
        return segment;
      }
      if (inInteractiveTag) return segment;
      return unique.reduce((value, item) => value.replace(
        new RegExp(escapeRegExp(item.label), "g"),
        `<button type="button" class="source-link" data-source-index="${item.index}" title="查看对应来源内容">$&</button>`,
      ), segment);
    }).join("");
  }

  function normalizeAnswerMarkdown(value) {
    const labels = "先说结论|核心结论|结论|初步判断|实际争点|法律上怎么判断|现有证据怎么看|接下来最实际的动作|建议|处理建议|风险提示|关键点|关键结论|现在先做|需要补充|法律依据|注意|依据核对";
    return String(value || "暂无内容")
      .replace(new RegExp(`(^|\\n)(\\s*)(?!\\*\\*)(${labels})(?:[：:])\\s*(.*)(?=\\n|$)`, "g"), "$1$2**$3：** $4");
  }

  function renderAnswerMarkdown(value, sources = []) {
    return linkLegalReferences(renderMarkdown(normalizeAnswerMarkdown(value)), sources);
  }

  function normalizeAnswerPayload(answer, fallbackText = "") {
    const fallback = String(fallbackText || "").trim();
    if (answer && typeof answer === "object" && !Array.isArray(answer)) {
      const answerText = String(answer.answer || answer.content || answer.text || fallback || "").trim();
      return {
        ...answer,
        answer: answerText,
        case_analysis: Array.isArray(answer.case_analysis) ? answer.case_analysis : [],
        key_issues: Array.isArray(answer.key_issues) ? answer.key_issues : [],
        evidence_analysis: Array.isArray(answer.evidence_analysis) ? answer.evidence_analysis : [],
        defense_arguments: Array.isArray(answer.defense_arguments) ? answer.defense_arguments : [],
        action_steps: Array.isArray(answer.action_steps) ? answer.action_steps : [],
        document_checklist: Array.isArray(answer.document_checklist) ? answer.document_checklist : [],
        questions_to_confirm: Array.isArray(answer.questions_to_confirm) ? answer.questions_to_confirm : [],
        legal_basis: Array.isArray(answer.legal_basis) ? answer.legal_basis : [],
        related_cases: Array.isArray(answer.related_cases) ? answer.related_cases : [],
        suggestions: Array.isArray(answer.suggestions) ? answer.suggestions : [],
        risk_notice: String(answer.risk_notice || ""),
        citations: Array.isArray(answer.citations) ? answer.citations : [],
      };
    }
    return {
      answer: String(answer || fallback || "").trim(),
      case_analysis: [],
      key_issues: [],
      evidence_analysis: [],
      defense_arguments: [],
      action_steps: [],
      document_checklist: [],
      questions_to_confirm: [],
      legal_basis: [],
      related_cases: [],
      suggestions: [],
      risk_notice: "",
      citations: [],
    };
  }

  function dedupeAnswerItems(items = [], mainAnswer = "", limit = 5) {
    const seen = new Set();
    const bodyText = String(mainAnswer || "");
    return (Array.isArray(items) ? items : [])
      .map((item) => String(item || "").trim())
      .filter((item) => {
        if (!item || seen.has(item)) return false;
        seen.add(item);
        const compact = item.replace(/^(实际争点|现有证据怎么看|法律上怎么判断|类似情况的处理倾向|对方可能怎么反驳|接下来最实际的动作|下一步建议|证据清单|还要确认)[：:]/, "").trim();
        return compact.length < 16 || !bodyText.includes(compact.slice(0, Math.min(42, compact.length)));
      })
      .slice(0, limit);
  }

  function sectionMarkdown(title, items = [], { ordered = false } = {}) {
    const cleanItems = (Array.isArray(items) ? items : [])
      .map((item) => String(item || "").trim())
      .filter(Boolean);
    if (!cleanItems.length) return "";
    const body = cleanItems.map((item, index) => ordered ? `${index + 1}. ${item}` : `- ${item}`).join("\n");
    return `### ${title}\n${body}`;
  }

  function buildDisplayAnswerMarkdown(answer = {}, mainAnswer = "") {
    const parts = [String(mainAnswer || "").trim()];
    const mainText = String(mainAnswer || "").trim();
    const hasHeading = (pattern) => pattern.test(String(mainAnswer || ""));
    const isStructured = /^#{1,4}\s+/m.test(mainText) || /\n\s*(?:[-*]|\d+[.)])\s+/.test(mainText);
    const needsAssistiveSections = mainText.length < 360 || !isStructured;
    const caseAnalysis = dedupeAnswerItems(answer.case_analysis, mainAnswer, 5);
    const relatedCases = dedupeAnswerItems(answer.related_cases, mainAnswer, 4);
    const actionSteps = dedupeAnswerItems(answer.action_steps?.length ? answer.action_steps : answer.suggestions, mainAnswer, 4);
    const documentChecklist = dedupeAnswerItems(answer.document_checklist, mainAnswer, 6);
    const questionsToConfirm = dedupeAnswerItems(answer.questions_to_confirm, mainAnswer, 4);
    const riskNotice = String(answer.risk_notice || "").trim();
    if (needsAssistiveSections && caseAnalysis.length && !hasHeading(/实际|争点|证据|怎么判断|本案|分析/)) {
      parts.push(sectionMarkdown("补充分析", caseAnalysis));
    }
    if (relatedCases.length && !hasHeading(/相关案例|类案|案例/)) {
      parts.push(sectionMarkdown("相关案例", relatedCases));
    }
    if (needsAssistiveSections && actionSteps.length && !hasHeading(/下一步|处理建议|行动|建议/)) {
      parts.push(sectionMarkdown("可操作事项", actionSteps, { ordered: true }));
    }
    if (needsAssistiveSections && documentChecklist.length && !hasHeading(/证据|材料|清单/)) {
      parts.push(sectionMarkdown("可用材料", documentChecklist));
    }
    if (needsAssistiveSections && questionsToConfirm.length && !hasHeading(/补充|确认|核实/)) {
      parts.push(sectionMarkdown("待核实信息", questionsToConfirm));
    }
    if (riskNotice && !String(mainAnswer || "").includes(riskNotice.slice(0, 40))) {
      parts.push(`> ${riskNotice}`);
    }
    return parts.filter(Boolean).join("\n\n");
  }

  function renderInlineSolution(panel, solution = {}) {
    if (!panel) return;
    const sources = Array.isArray(solution.sources) ? solution.sources : [];
    const markdown = String(solution.markdown || "暂时没有可查看的解决方案。");
    panel._solutionMarkdown = markdown;
    panel.innerHTML = `
      <div class="solution-panel-top">
        <div><span>法律解决方案</span><strong>${escapeHtml(solution.title || "本次对话解决方案")}</strong></div>
        <div class="solution-panel-actions">
          <button type="button" data-solution-download="pdf">下载 PDF</button>
          <button type="button" data-solution-download="word">下载 Word</button>
        </div>
      </div>
      <div class="solution-panel-content">${renderAnswerMarkdown(markdown, sources)}</div>`;
    bindSourceInteractions(panel, sources);
    panel.querySelectorAll("[data-solution-download]").forEach((button) => {
      button.addEventListener("click", () => downloadLegalSolution(button.dataset.solutionDownload, panel._solutionMarkdown || ""));
    });
    panel.hidden = false;
  }

  function toggleInlineSolution(button, panel, solution = null) {
    if (!panel) return;
    if (solution) renderInlineSolution(panel, solution);
    else panel.hidden = !panel.hidden;
    if (button) button.textContent = panel.hidden ? "查看法律解决方案" : "收起法律解决方案";
    scrollToLatest();
  }

  function collectConversationTurns(extraAnswer = null) {
    const messages = activeTask?.messages || [];
    const turns = [];
    let currentQuestion = "";
    messages.forEach((message) => {
      if (message.role === "user") currentQuestion = String(message.content || "").trim();
      if (message.role === "assistant") {
        const answerObject = message.answer || {};
        const answerText = String(answerObject.answer || "").trim();
        if (currentQuestion || answerText) turns.push({ question: currentQuestion, answer: answerText, answerObject });
      }
    });
    if (extraAnswer) {
      const latestAnswer = String(extraAnswer.answer || "").trim();
      const alreadyIncluded = turns.some((turn) => turn.answerObject === extraAnswer || (latestAnswer && turn.answer === latestAnswer));
      if (!alreadyIncluded) {
        const latestQuestion = messages.filter((message) => message.role === "user").at(-1)?.content || "";
        turns.push({ question: String(latestQuestion || "").trim(), answer: latestAnswer, answerObject: extraAnswer });
      }
    }
    return turns;
  }

  function listFromTurns(turns, field, fallback = []) {
    const values = turns.flatMap((turn) => {
      const value = turn.answerObject?.[field];
      if (Array.isArray(value)) return value;
      return value ? [value] : [];
    });
    return Array.from(new Set([...values, ...(fallback || [])]))
      .map((item) => String(item || "").trim())
      .filter(Boolean);
  }

  function solutionGuideTitle(question) {
    const text = String(question || "").trim();
    if (["殴打", "打人", "受伤", "伤情", "正当防卫", "人身损害"].some((word) => text.includes(word))) return "人身损害（被殴打）维权指南";
    if (["离婚", "夫妻", "婚姻", "抚养", "彩礼"].some((word) => text.includes(word))) return "婚姻家庭纠纷处理指南";
    if (["工资", "劳动合同", "辞退", "裁员", "工伤", "加班"].some((word) => text.includes(word))) return "劳动争议维权指南";
    if (["借款", "欠款", "不还", "借条", "还款"].some((word) => text.includes(word))) return "借款纠纷追偿指南";
    if (["合同", "违约", "退款", "定金", "解除"].some((word) => text.includes(word))) return "合同纠纷处理指南";
    return "法律问题处理指南";
  }

  function buildLegalSolution(answer = {}, sources = []) {
    const allTurns = collectConversationTurns(answer);
    const latestQuestion = allTurns.at(-1)?.question || activeTask?.messages?.filter((message) => message.role === "user").at(-1)?.content || "本次法律咨询";
    const latestAnswer = String(answer.answer || allTurns.at(-1)?.answer || "").trim();
    const keyIssues = listFromTurns(allTurns, "key_issues", answer.key_issues).slice(0, 10);
    const evidenceAnalysis = listFromTurns(allTurns, "evidence_analysis", answer.evidence_analysis).slice(0, 10);
    const legalBasis = listFromTurns(allTurns, "legal_basis", answer.legal_basis).slice(0, 10);
    const actionSteps = listFromTurns(allTurns, "action_steps", answer.action_steps || answer.suggestions).slice(0, 10);
    const documents = listFromTurns(allTurns, "document_checklist", answer.document_checklist).slice(0, 10);
    const questions = listFromTurns(allTurns, "questions_to_confirm", answer.questions_to_confirm).slice(0, 8);
    const relatedCases = listFromTurns(allTurns, "related_cases", answer.related_cases).slice(0, 6);
    const risks = listFromTurns(allTurns, "defense_arguments", answer.defense_arguments)
      .concat(listFromTurns(allTurns, "risk_notice", [answer.risk_notice]))
      .slice(0, 8);
    const sourceLabels = Array.from(new Set(sources.slice(0, 6).map((source) => source.title || source.label).filter(Boolean)));
    const safeCell = (value, limit = 110) => compactLine(value, limit).replaceAll("|", "/") || "待补充";
    const firstAction = actionSteps[0] || "整理事实时间线和原始证据";
    const pathRows = [
      [safeCell(latestQuestion, 76), safeCell(keyIssues[0] || "依据现有事实和证据初步判断", 90), safeCell(questions[0] || "核对关键行为、责任主体和结果", 90)],
      [safeCell(evidenceAnalysis[0] || "现有证据", 76), safeCell("补充材料后确定救济路径", 90), safeCell(documents[0] || "保留原始文件和形成过程", 90)],
      [safeCell(risks[0] || "对方拒绝处理或存在期限", 76), safeCell("协商、投诉、调解、仲裁或诉讼", 90), safeCell("核对管辖、时效和程序材料", 90)],
    ];
    const pathTable = [
      "| 事实情况 | 可能路径 | 识别重点 |",
      "| --- | --- | --- |",
      ...pathRows.map((row) => `| ${row.join(" | ")} |`),
    ].join("\n");
    const claimItems = (keyIssues.length ? keyIssues : actionSteps).slice(0, 5);
    const evidenceItems = (documents.length ? documents : ["事实经过、时间、地点和相关人员记录", "合同、通知、付款凭证或其他权利义务文件", "聊天记录、录音录像、照片和平台记录", "能证明损失或行为后果的凭证"]).slice(0, 8);
    const immediateActions = (actionSteps.length ? actionSteps : ["整理事实时间线", "保留原始证据", "核对处理期限"]).slice(0, 3);
    const lines = [
      `# ${solutionGuideTitle(latestQuestion)}`,
      `*基于现有信息的初步方案*`,
      `## 1. 先给用户的结论`,
      latestAnswer || "当前信息还不足，需要先补齐事实经过和关键证据，再判断具体处理路径。",
      immediateActions.map((item, index) => `${index + 1}. **${item}**`).join("\n"),
      `## 2. 立即行动清单（优先级最高）`,
      `### 2.1 当前最先做的事\n**${firstAction}。**`,
      `### 2.2 证据清单\n${evidenceItems.map((item) => `- ${item}`).join("\n")}`,
      `### 2.3 取证必须合法\n不得盗取账号、非法监听、侵入住宅、威胁取证、伪造或剪辑证据。电子数据应保留来源、形成过程和完整上下文。`,
      `## 3. 识别责任路径\n${pathTable}`,
      relatedCases.length ? `## 4. 相关案例\n${relatedCases.map((item) => `- ${item}`).join("\n")}` : "",
      `## 5. 可主张的责任、请求或赔偿\n${(claimItems.length ? claimItems : ["根据责任路径主张返还、履行、赔偿或停止侵害"]).map((item) => `- ${item}`).join("\n")}\n\n具体金额、责任范围和处理结果，需要结合事实、证据、过错程度和实际损失计算，不能凭主观估算。`,
      `## 6. 起诉/申请/维权前的四项整理\n### 6.1 原告/申请人信息\n准备身份证明、联系方式、住所或经常居住地等基础信息。\n\n### 6.2 对方/被告信息\n尽量核实姓名、主体名称、联系方式和可送达地址；不完整时通过合法程序补齐。\n\n### 6.3 事实时间线\n按"日期 → 地点 → 行为 → 后果 → 沟通/处理 → 损失"记录，并附对应证据编号。\n\n### 6.4 请求与证据目录\n将每一项请求与合同、记录、票据、鉴定或其他证据逐项关联。`,
      `## 7. 管辖、立案与时效\n根据争议类型核对处理机关或法院、管辖连接点、立案材料和诉讼时效。正式行动前应核对当地现行要求，发现可能临近期限时及时咨询专业人士。`,
      `## 8. 明确禁止的报复性维权\n- 不要公开对方身份证、住址、手机号等个人信息；\n- 不要威胁、恐吓、围攻、私自扣押或骚扰对方；\n- 不要未经核实公开指控对方犯罪；\n- 不要删除、剪辑或伪造证据。`,
      `## 9. 面向用户的简明输出模板\n**初步判断：** 当前结论仍需结合完整事实和证据核验。\n\n**现在先做：** ${firstAction}。\n\n**如果仍有危险/紧急情形：** 先离开危险现场，联系当地紧急服务，不要单独与对方见面。\n\n**后续可主张：** 根据责任路径和可证明损失，主张返还、履行、赔偿、停止侵害或其他法定救济。\n\n**需要补充：** ${(questions.length ? questions : ["对方身份、发生时间地点、已有证据和实际损失"]).join("；")}。`,
      `## 10. 法律依据索引（用于检索标注）\n${(legalBasis.length ? legalBasis : sourceLabels).map((item) => `- ${item}`).join("\n") || "- 当前没有可稳定列出的参考来源"}\n\n法律和司法解释可能修订。正式行动前，应核对现行有效文本及当地主管机关要求。`,
      `本方案结合当前对话中的问题、回答和已检索来源。`,
      `\n本方案仅供法律信息参考，不替代律师结合完整证据作出的正式法律意见。`,
    ].filter(Boolean);
    return {
      title: solutionGuideTitle(latestQuestion),
      scope: `已整合 ${allTurns.length || 1} 轮对话`,
      generatedAt: formatTime(),
      markdown: lines.join("\n\n"),
      sources,
    };
  }

  async function generateLegalSolution(question, answer = {}, sources = [], fallbackSolution = null) {
    const response = await fetch(backendUrl("/api/v1/legal/solution"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        question: String(question || "").trim() || "本次法律咨询",
        answer,
        sources: normalizeStoredSources(sources).slice(0, 20),
        session_id: sessionId,
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(getErrorMessage(payload, "法律解决方案生成失败"));
    const solution = payload?.data || {};
    return {
      title: solution.title || fallbackSolution?.title || "本次对话解决方案",
      scope: solution.scope || "第二次模型调用生成",
      generatedAt: solution.generated_at || formatTime(),
      elapsedMs: solution.elapsed_ms,
      markdown: String(solution.markdown || fallbackSolution?.markdown || "暂时没有可查看的解决方案。"),
      sources,
    };
  }

  async function openGeneratedLegalSolution(button, question, answer = {}, sources = [], options = {}) {
    const requestIdentity = identityVersion;
    const fallbackSolution = buildLegalSolution(answer, sources);
    const originalText = button?.textContent || "查看法律解决方案";
    if (button) {
      button.disabled = true;
      button.textContent = "正在生成法律解决方案…";
    }
    try {
      const solution = await generateLegalSolution(question, answer, sources, fallbackSolution);
      if (requestIdentity !== identityVersion) return;
      if (options.inlinePanel) renderInlineSolution(options.inlinePanel, solution);
      else openSolutionDrawer(solution);
      if (button && options.inlinePanel) button.textContent = "收起法律解决方案";
      options.onGenerated?.(solution);
    } catch (error) {
      if (requestIdentity !== identityVersion) return;
      console.warn("法律解决方案后端生成失败，使用本地兜底方案", error);
      const solution = {
        ...fallbackSolution,
        scope: "后端方案生成失败，已使用本地兜底",
      };
      if (options.inlinePanel) renderInlineSolution(options.inlinePanel, solution);
      else openSolutionDrawer(solution);
      if (button && options.inlinePanel) button.textContent = "收起法律解决方案";
      options.onGenerated?.(solution);
    } finally {
      if (button) {
        button.disabled = false;
        if (!options.inlinePanel) button.textContent = originalText;
      }
    }
  }

  function downloadBlob(fileName, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function markdownToDocumentHtml(markdown) {
    return `<!doctype html><html><head><meta charset="utf-8"><title>法律解决方案</title><style>body{font-family:"Microsoft YaHei",sans-serif;line-height:1.9;color:#222;padding:32px;}h1,h2{font-family:"Songti SC",SimSun,serif;}li{margin:6px 0;}</style></head><body>${renderMarkdown(markdown)}</body></html>`;
  }

  async function downloadLegalSolution(format, markdown) {
    const content = String(markdown || "").trim();
    if (!content) return;
    const stamp = new Date().toISOString().slice(0, 10);
    if (format === "word") {
      downloadBlob(`法律解决方案-${stamp}.doc`, markdownToDocumentHtml(content), "application/msword;charset=utf-8");
      return;
    }
    try {
      const response = await fetch(backendUrl("/api/v1/legal/solution/pdf"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ markdown: content }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(getErrorMessage(payload, "PDF 生成失败"));
      }
      const pdfBlob = await response.blob();
      downloadBlob(`法律解决方案-${stamp}.pdf`, pdfBlob, "application/pdf");
    } catch (error) {
      window.alert(friendlyErrorMessage(error, "PDF 下载失败，请稍后重试"));
    }
  }

  function bindSourceInteractions(row, sources = []) {
    if (!row) return;
    row._sourceLinkSources = normalizeSourceList(sources);
    const openIndexedSource = (element, event) => {
      if (!element || !row.contains(element)) return;
      event?.preventDefault();
      event?.stopPropagation();
      const source = row._sourceLinkSources?.[Number(element.dataset.sourceIndex)];
      if (source) openSourceDrawer(source);
    };
    row.querySelectorAll("button[data-source-index]").forEach((element) => {
      if (element.dataset.sourceBound === "true") return;
      element.dataset.sourceBound = "true";
      element.title = "查看对应来源内容";
      element.addEventListener("click", (event) => openIndexedSource(element, event));
    });
    if (row._sourceLinkHandler) return;
    row._sourceLinkHandler = (event) => {
      const element = event.target.closest("[data-source-index]");
      if (!element || !row.contains(element)) return;
      openIndexedSource(element, event);
    };
    row.addEventListener("click", row._sourceLinkHandler);
  }

  function forGeneralUser(value) {
    return String(value || "")
      .replace(/(?:根据|依据)?\s*[（(]?\[?材料\s*\d+\]?[）)]?/g, "")
      .replace(/^[，、；：\s]+/, "")
      .replace(/([。！？；])\s*[，、；：]+/g, "$1")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  function renderMarkdown(value) {
    const normalized = String(value || "暂无内容")
      .replaceAll("\\r\\n", "\n")
      .replaceAll("\\n", "\n")
      .replaceAll("\\t", "\t");
    const lines = normalized.split(/\r?\n/);
    const output = [];
    let listType = "";

    function closeList() {
      if (listType) output.push(`</${listType}>`);
      listType = "";
    }

    function renderParagraph(text) {
      const priority = String(text || "").match(/^(?:\*\*)?(先说结论|核心结论|结论|初步判断|实际争点|法律上怎么判断|现有证据怎么看|接下来最实际的动作|最重要的是|建议|处理建议|注意|风险提示|关键点|关键结论|现在先做|需要补充|法律依据|依据核对)(?:\*\*)?[：:]\s*(.*)$/);
      if (!priority) return `<p>${formatInline(text)}</p>`;
      const label = priority[1];
      const body = priority[2];
      return `<p class="priority-paragraph"><strong class="priority-label">${escapeHtml(label)}</strong>${body ? ` ${formatInline(body)}` : ""}</p>`;
    }

    function tableCells(text) {
      const value = String(text || "").trim();
      if (!value.includes("|")) return [];
      return value.replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
    }

    function isTableDivider(cells) {
      return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
    }

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex];
      const trimmed = line.trim();
      const headerCells = tableCells(trimmed);
      const dividerCells = tableCells(lines[lineIndex + 1]);
      if (headerCells.length >= 2 && isTableDivider(dividerCells) && dividerCells.length === headerCells.length) {
        closeList();
        const rows = [];
        let nextIndex = lineIndex + 2;
        while (nextIndex < lines.length) {
          const rowCells = tableCells(lines[nextIndex]);
          if (rowCells.length !== headerCells.length || !rowCells.length) break;
          rows.push(rowCells);
          nextIndex += 1;
        }
        output.push(`<div class="solution-table-wrap"><table><thead><tr>${headerCells.map((cell) => `<th>${formatInline(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${formatInline(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        lineIndex = nextIndex - 1;
        continue;
      }
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        closeList();
        output.push("<hr />");
        continue;
      }
      const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
      const unordered = trimmed.match(/^[-*]\s+(.+)$/);
      const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);

      if (heading) {
        closeList();
        const level = Math.min(heading[1].length + 1, 4);
        const headingClass = /核心|结论|重点|建议|风险|注意|关键/.test(heading[2]) ? ' class="priority-heading"' : "";
        output.push(`<h${level}${headingClass}>${formatInline(heading[2])}</h${level}>`);
      } else if (unordered || ordered) {
        const nextType = unordered ? "ul" : "ol";
        if (listType !== nextType) {
          closeList();
          output.push(`<${nextType}>`);
          listType = nextType;
        }
        output.push(`<li>${formatInline((unordered || ordered)[1])}</li>`);
      } else if (!trimmed) {
        closeList();
      } else {
        closeList();
        output.push(renderParagraph(trimmed));
      }
    }
    closeList();
    return output.join("");
  }

  function formatTime(timestamp = Date.now()) {
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(new Date(timestamp));
  }

  function formatActivityTime(timestamp) {
    const value = Number(timestamp || 0);
    if (!value) return "";
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(new Date(value * 1000));
  }

  function formatDateTime(timestamp) {
    if (!timestamp) return "未知时间";
    const date = typeof timestamp === "string" && Number.isNaN(Number(timestamp))
      ? new Date(timestamp)
      : new Date(Number(timestamp) * 1000);
    if (Number.isNaN(date.getTime())) return "未知时间";
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(date);
  }

  function formatRemainingTime(seconds) {
    const value = Math.max(0, Number(seconds || 0));
    const days = Math.floor(value / 86400);
    const hours = Math.floor((value % 86400) / 3600);
    if (days > 0) return `${days} 天 ${hours} 小时`;
    const minutes = Math.ceil(value / 60);
    return `${Math.max(1, minutes)} 分钟`;
  }

  function friendlyErrorMessage(error, fallback = "操作失败") {
    const raw = String(error?.message || error || "");
    if (/not found/i.test(raw) || raw === "404") return "当前服务暂未开启这个接口";
    if (/failed to fetch|networkerror|load failed/i.test(raw)) return "无法连接到本地服务";
    return raw || fallback;
  }

  function getErrorMessage(payload, fallback = "操作失败") {
    return payload?.message || payload?.detail || fallback;
  }

  function formatEnabled(value) {
    return value ? "开启" : "关闭";
  }

  function formatStatusValue(value) {
    const text = String(value || "未知").toLowerCase();
    if (text === "ok" || text === "online") return "正常";
    if (text === "offline") return "离线";
    if (text === "error" || text === "failed") return "异常";
    if (text === "unknown") return "未知";
    return String(value || "未知");
  }

  function formatAnswerDetail(value) {
    return {
      concise: "简洁",
      standard: "标准",
      detailed: "详细",
    }[value] || "标准";
  }

  function formatThemeName(value) {
    return {
      warm_gold: "沉静暗金",
      cool_gray: "素雅纸白",
    }[value] || "沉静暗金";
  }

  function normalizeThemeKey(value) {
    // Keep the existing backend theme key; retired palettes use the new default.
    return ["warm_gold", "cool_gray"].includes(value) ? value : "";
  }

  function readThemePreference() {
    try {
      return normalizeThemeKey(readStorage(THEME_STORAGE_KEY)) || DEFAULT_USER_SETTINGS.theme;
    } catch (_error) {
      return DEFAULT_USER_SETTINGS.theme;
    }
  }

  function syncThinkingToggle() {
    if (thinkingToggleButton) {
      thinkingToggleButton.classList.remove("is-enabled");
      thinkingToggleButton.removeAttribute("aria-pressed");
      thinkingToggleButton.title = "添加证据材料";
      thinkingToggleButton.setAttribute("aria-label", "添加证据材料");
    }
    if (fileStatus && !fileStatus.textContent.trim()) {
      fileStatus.className = "file-status";
      fileStatus.textContent = "材料用于补充案情和证据线索";
    }
  }

  function formatHealthStatus(health = {}) {
    if (health.error) return `服务异常：${health.error}`;
    const parts = [
      `整体状态：${formatStatusValue(health.status)}`,
      `知识库检索：${formatStatusValue(health.rag)}`,
      `大模型：${health.llm || "未知"} / ${health.llm_model || "未知"}`,
      `思考提示：${formatEnabled(health.deepseek_thinking)}`,
      `向量库：${formatStatusValue(health.milvus)}`,
      `网页检索：${formatStatusValue(health.web)}`,
      `索引数量：${health.indexed_records ?? "未知"}`,
    ];
    return parts.join("\n");
  }

  function formatDiagnostics(data) {
    const settings = normalizeSettings(data.settings || {});
    const user = data.user;
    const deletion = data.account_deletion || user?.account_deletion || {};
    const activity = Array.isArray(data.recent_activity) ? data.recent_activity : [];
    const api = data.api_status || {};
    const lines = [
      "问题查询",
      "",
      "一、当前页面",
      `页面地址：${data.page_url || ""}`,
      `后端地址：${data.backend_url || ""}`,
      `版本：${data.version || "未知"}`,
      "",
      "二、登录状态",
      `是否登录：${data.authenticated ? "已登录" : "未登录"}`,
      `用户名：${user?.username || "无"}`,
      `工作区：${user?.workspace_id || "无"}`,
      `当前会话：${data.active_session_id || "无"}`,
      `注销状态：${deletion.status === "pending_deletion" ? "审核中" : "正常"}`,
      deletion.status === "pending_deletion" ? `注销生效时间：${formatDateTime(deletion.deletion_scheduled_at)}` : "",
      "",
      "三、用户设置",
      `默认联网检索：${formatEnabled(settings.include_web_default)}`,
      `长期记忆：${formatEnabled(settings.enable_long_memory)}`,
      `自动展开辅助信息：${formatEnabled(settings.auto_expand_professional)}`,
      `显示检索过程：${formatEnabled(settings.show_retrieval_process)}`,
      `回答详细程度：${formatAnswerDetail(settings.answer_detail)}`,
      "会话记忆范围：当前完整对话窗口",
      "",
      "四、服务状态",
      formatHealthStatus(data.health),
      "",
      "五、接口检查",
      `设置接口：${api.settings || "未检查"}`,
      `长期记忆接口：${api.memory || "未检查"}`,
      `操作记录接口：${api.activity || "未检查"}`,
      `账号注销接口：${api.deletion || "未检查"}`,
      "",
      "六、最近操作",
    ];
    if (!activity.length) {
      lines.push("暂无可显示的操作记录。");
    } else {
      activity.slice(0, 10).forEach((record, index) => {
        lines.push(`${index + 1}. ${record.title || record.action || "操作"} - ${record.detail || "无详情"} - ${formatActivityTime(record.created_at) || "无时间"}`);
      });
    }
    return lines.join("\n");
  }

  function renderActivity(records = []) {
    if (!activityList) return;
    if (!authenticated) {
      activityList.textContent = "登录后可查看最近操作记录。";
      return;
    }
    if (!records.length) {
      activityList.textContent = "暂无操作记录。";
      return;
    }
    activityList.innerHTML = records.map((record) => `
      <div class="activity-item">
        <strong>${escapeHtml(record.title || record.action || "操作记录")}</strong>
        <span>${escapeHtml(record.detail || "")}</span>
        <small>${escapeHtml(formatActivityTime(record.created_at))}</small>
      </div>`).join("");
  }

  async function loadUserSettings() {
    const requestIdentity = identityVersion;
    const requestVersion = settingsSaveVersion;
    if (!authenticated) {
      loadLocalSettings("guest");
      return;
    }
    try {
      const response = await fetch(backendUrl("/api/v1/user/settings"), { credentials: "include" });
      if (!response.ok) throw new Error("设置加载失败");
      const payload = await response.json();
      if (requestIdentity !== identityVersion || requestVersion !== settingsSaveVersion || !authenticated) return;
      const pendingTheme = normalizeThemeKey(readStorage(`${settingsStorageKey(taskNamespace)}.pending_theme`));
      userSettings = normalizeSettings({ ...userSettings, ...(payload.data || {}), ...(pendingTheme ? { theme: pendingTheme } : {}), enable_long_memory: false });
      saveLocalSettings(taskNamespace);
      applySettingsToForm();
      if (pendingTheme) queueUserSettingsSave();
    } catch (_error) {
      if (requestIdentity !== identityVersion || requestVersion !== settingsSaveVersion) return;
      loadLocalSettings(taskNamespace);
    }
  }

  async function saveUserSettings() {
    const requestIdentity = identityVersion;
    const requestVersion = ++settingsSaveVersion;
    window.clearTimeout(settingsSaveTimer);
    userSettings = readSettingsForm();
    const requestTheme = userSettings.theme;
    applySettingsToForm();
    saveLocalSettings(taskNamespace);
    if (settingsStatus) {
      settingsStatus.classList.remove("is-error");
      settingsStatus.textContent = "正在保存…";
    }
    try {
      if (authenticated) {
        const { enable_long_memory: _enableLongMemory, ...profileSettings } = userSettings;
        const response = await fetch(backendUrl("/api/v1/user/settings"), {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify(profileSettings),
        });
        const payload = await response.json().catch(() => null);
        if (requestIdentity !== identityVersion || requestVersion !== settingsSaveVersion) return;
        if (!response.ok) throw new Error(getErrorMessage(payload, "设置保存失败"));
        userSettings = normalizeSettings({ ...userSettings, ...(payload.data || {}), enable_long_memory: false });
        const pendingKey = `${settingsStorageKey(taskNamespace)}.pending_theme`;
        if (userSettings.theme === requestTheme && readStorage(pendingKey) === requestTheme) writeStorage(pendingKey, "");
      }
      saveLocalSettings(taskNamespace);
      applySettingsToForm();
      if (settingsStatus) settingsStatus.textContent = "已保存";
      await loadActivity();
    } catch (error) {
      if (requestIdentity !== identityVersion || requestVersion !== settingsSaveVersion) return;
      saveLocalSettings(taskNamespace);
      applySettingsToForm();
      if (settingsStatus) {
        const message = friendlyErrorMessage(error, "设置保存失败");
        const canUseLocalFallback = /接口|服务|连接/.test(message);
        settingsStatus.classList.toggle("is-error", !canUseLocalFallback);
        settingsStatus.textContent = canUseLocalFallback ? `已保存到本机；${message}` : message;
      }
    }
  }

  function queueUserSettingsSave() {
    const previousTheme = userSettings.theme;
    settingsSaveVersion += 1;
    userSettings = readSettingsForm();
    if (userSettings.theme !== previousTheme) writeStorage(`${settingsStorageKey(taskNamespace)}.pending_theme`, userSettings.theme);
    applySettingsToForm();
    saveLocalSettings(taskNamespace);
    if (settingsStatus) {
      settingsStatus.classList.remove("is-error");
      settingsStatus.textContent = "正在自动保存…";
    }
    window.clearTimeout(settingsSaveTimer);
    settingsSaveTimer = window.setTimeout(saveUserSettings, 350);
  }

  async function loadActivity() {
    const requestIdentity = identityVersion;
    if (!authenticated) {
      renderActivity([]);
      return [];
    }
    try {
      const response = await fetch(backendUrl("/api/v1/user/activity?limit=20"), { credentials: "include" });
      if (!response.ok) throw new Error(response.status === 404 ? "404" : "操作记录加载失败");
      const payload = await response.json();
      if (requestIdentity !== identityVersion || !authenticated) return [];
      const records = Array.isArray(payload.data) ? payload.data : [];
      renderActivity(records);
      return records;
    } catch (error) {
      if (requestIdentity !== identityVersion) return [];
      if (activityList) activityList.textContent = `${friendlyErrorMessage(error, "操作记录加载失败")}，不影响设置保存。`;
      return [];
    }
  }

  function renderAccountSummary() {
    document.body.dataset.authState = authenticated ? "authenticated" : "guest";
    const displayName = authenticated && currentUser?.username ? currentUser.username : "未登录";
    if (homeAccountButton) homeAccountButton.textContent = authenticated ? "我的账户" : "登录 / 注册";
    if (workspaceLabel) workspaceLabel.textContent = authenticated ? `${displayName} · 个人咨询记录` : "未登录 · 公共法律库";
    if (sidebarAccountName) sidebarAccountName.textContent = authenticated ? displayName : "登录 / 注册";
    document.getElementById("personal-history").hidden = !authenticated;
    document.getElementById("guest-history-hint").hidden = authenticated;
    document.getElementById("home-session-state").textContent = authenticated ? "个人咨询 · 民事法律问答" : "游客体验 · 民事法律问答";
    document.getElementById("home-greeting").textContent = authenticated ? `${displayName}，说说你的事情，我们一起梳理下一步。` : "说说你的事情，一起找到依法解决问题的路。";
    document.getElementById("account-menu-name").textContent = displayName;
    if (authStatus) authStatus.textContent = authenticated ? "个人咨询记录" : "登录后保存与继续咨询";
    if (taskSelect) taskSelect.hidden = !authenticated;
    if (deleteTaskCompact) deleteTaskCompact.hidden = !authenticated;
    if (sidebarUserAvatar) sidebarUserAvatar.textContent = authenticated ? userAvatarText(currentUser) : "未";
    if (loginButton) loginButton.setAttribute("aria-label", authenticated ? "打开账户菜单" : "打开账户登录");
    if (settingsAccountName) settingsAccountName.textContent = displayName;
    if (settingsAccountStatus) settingsAccountStatus.textContent = authenticated ? "当前账号已登录，可同步个人咨询记录和设置。" : "登录后可以同步个人咨询记录和设置。";
    if (settingsAccountUserId) settingsAccountUserId.textContent = authenticated && currentUser?.user_id ? currentUser.user_id : "—";
    if (settingsAccountWorkspace) settingsAccountWorkspace.textContent = authenticated && currentUser?.workspace_id ? currentUser.workspace_id : "公共法律库";
    if (settingsLogoutButton) {
      settingsLogoutButton.disabled = !authenticated;
      settingsLogoutButton.textContent = authenticated ? "退出登录" : "未登录";
    }
  }

  function renderAccountDeletionStatus(data = currentUser?.account_deletion || {}) {
    if (!accountDeletionStatus || !accountDeletionButton) return;
    if (!authenticated) {
      accountDeletionStatus.textContent = "登录后可以申请注销账号。";
      accountDeletionButton.textContent = "申请注销";
      accountDeletionButton.disabled = true;
      accountDeletionButton.dataset.mode = "request";
      return;
    }
    const pending = data?.status === "pending_deletion";
    if (pending) {
      accountDeletionStatus.textContent = `注销审核中，将于 ${formatDateTime(data.deletion_scheduled_at)} 后生效。剩余 ${formatRemainingTime(data.seconds_remaining)}，期间可以取消。`;
      accountDeletionButton.textContent = "取消注销";
      accountDeletionButton.disabled = !data.can_cancel;
      accountDeletionButton.dataset.mode = "cancel";
      return;
    }
    accountDeletionStatus.textContent = "当前账号正常。申请注销后会进入 3 天审核期，审核期内可以取消。";
    accountDeletionButton.textContent = "申请注销";
    accountDeletionButton.disabled = false;
    accountDeletionButton.dataset.mode = "request";
  }

  async function loadAccountDeletionStatus() {
    const requestIdentity = identityVersion;
    if (!authenticated) {
      renderAccountDeletionStatus();
      return null;
    }
    try {
      const response = await fetch(backendUrl("/api/v1/user/deletion-request"), { credentials: "include" });
      const payload = await response.json().catch(() => null);
      if (requestIdentity !== identityVersion || !authenticated) return null;
      if (!response.ok) throw new Error(getErrorMessage(payload, "注销状态加载失败"));
      currentUser.account_deletion = payload.data || {};
      renderAccountDeletionStatus(currentUser.account_deletion);
      return currentUser.account_deletion;
    } catch (error) {
      if (requestIdentity !== identityVersion) return null;
      if (accountDeletionStatus) accountDeletionStatus.textContent = friendlyErrorMessage(error, "注销状态加载失败");
      if (accountDeletionButton) accountDeletionButton.disabled = true;
      return null;
    }
  }

  async function handleAccountDeletionAction() {
    const requestIdentity = identityVersion;
    if (!authenticated || !accountDeletionButton) {
      openAuth("login");
      return;
    }
    const mode = accountDeletionButton.dataset.mode || "request";
    const requesting = mode === "request";
    if (requesting && !window.confirm("确定申请注销账号吗？账号会进入 3 天审核期，期间可以取消注销。")) return;
    accountDeletionButton.disabled = true;
    if (accountDeletionStatus) accountDeletionStatus.textContent = requesting ? "正在提交注销申请…" : "正在取消注销申请…";
    try {
      const response = await fetch(backendUrl("/api/v1/user/deletion-request"), {
        method: requesting ? "POST" : "DELETE",
        credentials: "include",
      });
      const payload = await response.json().catch(() => null);
      if (requestIdentity !== identityVersion || !authenticated) return;
      if (!response.ok) throw new Error(getErrorMessage(payload, requesting ? "注销申请失败" : "取消注销失败"));
      currentUser.account_deletion = payload.data || {};
      renderAccountDeletionStatus(currentUser.account_deletion);
      await loadActivity();
    } catch (error) {
      if (requestIdentity !== identityVersion) return;
      if (accountDeletionStatus) accountDeletionStatus.textContent = friendlyErrorMessage(error, requesting ? "注销申请失败" : "取消注销失败");
      if (accountDeletionButton) accountDeletionButton.disabled = false;
    }
  }

  function selectSettingsTab(tabName = "preferences") {
    const selected = tabName === "account" ? "account" : "preferences";
    settingsTabs.forEach((tab) => {
      const active = tab.dataset.settingsTab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    settingsPanels.forEach((panel) => { panel.hidden = panel.dataset.settingsPanel !== selected; });
    if (settingsSave) settingsSave.hidden = selected !== "preferences";
  }

  async function openSettings(tabName = "preferences") {
    const requestIdentity = identityVersion;
    if (!settingsBackdrop) return;
    closeAccountMenu();
    setSidebarOpen(false);
    if (settingsBackdrop.hidden) {
      settingsReturnFocus = document.activeElement?.closest?.("#account-menu") ? loginButton : document.activeElement;
    }
    settingsBackdrop.hidden = false;
    appShell.inert = true;
    selectSettingsTab(tabName);
    settingsClose?.focus();
    applyBackendOriginToForm();
    applySettingsToForm();
    if (settingsStatus) settingsStatus.textContent = authenticated ? "" : "访客设置只保存在当前浏览器。";
    renderAccountDeletionStatus();
    if (activityList) activityList.textContent = authenticated ? "正在加载操作记录…" : "登录后可查看最近操作记录。";
    await loadAccountDeletionStatus();
    if (requestIdentity !== identityVersion) return;
    await loadActivity();
  }

  function closeSettings() {
    const wasOpen = settingsBackdrop && !settingsBackdrop.hidden;
    if (settingsBackdrop) settingsBackdrop.hidden = true;
    appShell.inert = !authBackdrop.hidden;
    if (wasOpen && authBackdrop.hidden && settingsReturnFocus?.isConnected) settingsReturnFocus.focus();
  }

  async function buildDiagnostics() {
    const health = await fetch(backendUrl("/health")).then((response) => response.json()).catch((error) => ({ error: error.message }));
    const records = await loadActivity();
    const settingsStatusCode = await fetch(backendUrl("/api/v1/user/settings"), { credentials: "include" })
      .then((response) => response.ok ? "正常" : `不可用（${response.status}）`)
      .catch((error) => `异常：${friendlyErrorMessage(error)}`);
    const activityStatusCode = await fetch(backendUrl("/api/v1/user/activity?limit=1"), { credentials: "include" })
      .then((response) => response.ok ? "正常" : `不可用（${response.status}）`)
      .catch((error) => `异常：${friendlyErrorMessage(error)}`);
    const deletionStatusCode = await fetch(backendUrl("/api/v1/user/deletion-request"), { credentials: "include" })
      .then((response) => response.ok ? "正常" : `不可用（${response.status}）`)
      .catch((error) => `异常：${friendlyErrorMessage(error)}`);
    const memoryStatusCode = await fetch(backendUrl("/api/v1/memory/settings"), { credentials: "include" })
      .then((response) => response.ok ? "正常" : `不可用（${response.status}）`)
      .catch((error) => `异常：${friendlyErrorMessage(error)}`);
    return {
      page_url: window.location.href,
      backend_url: apiBaseUrl,
      version: appVersion?.textContent || "",
      authenticated,
      user: currentUser ? { user_id: currentUser.user_id, username: currentUser.username, workspace_id: currentUser.workspace_id } : null,
      account_deletion: currentUser?.account_deletion || null,
      active_session_id: sessionId,
      settings: userSettings,
      health,
      api_status: {
        settings: settingsStatusCode,
        activity: activityStatusCode,
        deletion: deletionStatusCode,
        memory: memoryStatusCode,
      },
      recent_activity: records.slice(0, 20),
    };
  }

  async function refreshDiagnostics() {
    const requestIdentity = identityVersion;
    if (!diagnosticsOutput) return;
    diagnosticsOutput.textContent = "正在读取诊断信息…";
    const data = await buildDiagnostics();
    if (requestIdentity !== identityVersion) return;
    diagnosticsOutput.textContent = formatDiagnostics(data);
  }

  async function openDiagnostics() {
    if (!diagnosticsBackdrop) return;
    diagnosticsBackdrop.hidden = false;
    await refreshDiagnostics();
  }

  function closeDiagnostics() {
    if (diagnosticsBackdrop) diagnosticsBackdrop.hidden = true;
  }

  function resetVersionClickHint() {
    versionClickCount = 0;
    diagnosticsTriggers.forEach((trigger) => {
      window.clearTimeout(trigger._resetTimer);
      trigger.removeAttribute("data-clicks");
      trigger.title = "版本信息，连续点击 5 次打开问题查询";
      trigger.setAttribute("aria-label", "版本信息，连续点击 5 次打开问题查询");
    });
  }

  function handleVersionClick(event) {
    const trigger = event.currentTarget;
    if (!trigger) return;
    versionClickCount += 1;
    const remaining = Math.max(0, 5 - versionClickCount);
    diagnosticsTriggers.forEach((item) => {
      item.dataset.clicks = String(versionClickCount);
      item.title = remaining ? `再点 ${remaining} 次打开问题查询` : "打开问题查询";
      item.setAttribute("aria-label", item.title);
    });
    window.clearTimeout(trigger._resetTimer);
    trigger._resetTimer = window.setTimeout(resetVersionClickHint, 5000);
    if (settingsStatus && !settingsBackdrop?.hidden && remaining > 0) {
      settingsStatus.classList.remove("is-error");
      settingsStatus.textContent = `再点 ${remaining} 次打开问题查询`;
    }
    if (versionClickCount >= 5) {
      resetVersionClickHint();
      closeSettings();
      openDiagnostics();
    }
  }

  function setServiceStatus(status) {
    statusDot.className = `status-dot ${status}`;
    statusText.textContent = status === "online" ? "问答服务在线" : status === "offline" ? "问答服务离线" : "正在连接服务";
  }

  function setLoading(nextLoading) {
    loading = nextLoading;
    document.body.classList.toggle("is-loading", loading);
    input.disabled = false;
    input.placeholder = loading ? "正在回答，你可以继续输入下一问…" : "请描述你遇到的法律问题…";
    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.disabled = false;
    });
    routeLinks.forEach((button) => {
      button.disabled = false;
    });
    document.querySelectorAll(".mobile-route-nav button").forEach((button) => {
      button.disabled = false;
    });
    newChatButton.disabled = false;
    if (newTaskCompact) newTaskCompact.disabled = false;
    if (deleteTaskCompact) deleteTaskCompact.disabled = false;
    if (taskSelect) taskSelect.disabled = false;
    if (taskList) {
      taskList.querySelectorAll("button").forEach((button) => {
        button.disabled = false;
      });
    }
    sendButton.disabled = !input.value.trim();
    if (thinkingToggleButton) thinkingToggleButton.disabled = !ENABLE_FILE_UPLOADS;
  }

  function formatWorkflowDuration(ms) {
    const value = Number(ms);
    if (!Number.isFinite(value) || value < 0) return "";
    const seconds = value / 1000;
    if (seconds < 10) return `${seconds.toFixed(2)}s`;
    if (seconds < 100) return `${seconds.toFixed(1)}s`;
    return `${Math.round(seconds)}s`;
  }

  function createWorkflowState() {
    return Object.fromEntries(WORKFLOW_STEPS.map(({ stage }) => [stage, { status: "waiting", elapsed_ms: null, message: "" }]));
  }

  function renderWorkflowItems(state = workflowState) {
    if (!workflowList) return;
    workflowList.innerHTML = WORKFLOW_STEPS.map((step, index) => {
      const item = state?.[step.stage] || {};
      const status = item.status || "waiting";
      const detail = item.message || (status === "running" ? step.running : status === "completed" ? step.done : step.waiting);
      const duration = status === "completed"
        ? formatWorkflowDuration(item.elapsed_ms)
        : status === "running"
          ? "进行中"
          : "";
      return `<li class="workflow-step is-${status}" data-workflow-stage="${escapeHtml(step.stage)}"><span>${String(index + 1).padStart(2, "0")}</span><div class="workflow-step-body"><strong>${escapeHtml(step.label)}</strong><small data-workflow-detail>${escapeHtml(detail)}</small></div><em class="workflow-duration" data-workflow-duration>${escapeHtml(duration)}</em></li>`;
    }).join("");
  }

  function updateWorkflowTotal(elapsedMs) {
    if (!workflowTotal) return;
    if (Number.isFinite(Number(elapsedMs))) {
      workflowTotal.textContent = `总耗时 ${formatWorkflowDuration(elapsedMs)}`;
    } else {
      workflowTotal.textContent = "等待提问";
    }
  }

  let workflowState = createWorkflowState();

  function resetWorkflow() {
    workflowState = createWorkflowState();
    renderWorkflowItems(workflowState);
    updateWorkflowTotal();
  }

  function updateWorkflow(stage, status, data = {}) {
    if (!stage || !Object.prototype.hasOwnProperty.call(workflowState, stage)) return;
    const currentIndex = WORKFLOW_STEPS.findIndex((item) => item.stage === stage);
    workflowState[stage] = {
      ...workflowState[stage],
      ...data,
      status,
    };
    if (currentIndex >= 0 && status === "completed") {
      for (let index = 0; index < currentIndex; index += 1) {
        const previousStage = WORKFLOW_STEPS[index].stage;
        workflowState[previousStage] = {
          ...workflowState[previousStage],
          status: workflowState[previousStage].status === "error" ? "error" : "completed",
          message: workflowState[previousStage].message || WORKFLOW_STEPS[index].done,
        };
      }
    }
    renderWorkflowItems(workflowState);
    if (stage === "generation" && status === "completed" && Number.isFinite(Number(data.elapsed_ms))) {
      updateWorkflowTotal(data.elapsed_ms);
    }
  }

  function revealConversation() {
    conversation.querySelector("#empty-state")?.remove();
    conversation.classList.remove("empty-conversation");
  }

  function isConversationNearBottom(threshold = 48) {
    if (!conversationScroll) return true;
    const distance = conversationScroll.scrollHeight - conversationScroll.scrollTop - conversationScroll.clientHeight;
    return distance <= threshold;
  }

  function syncConversationFollowLatest() {
    conversationFollowLatest = isConversationNearBottom();
  }

  function scrollToLatest() {
    if (!conversationScroll || !conversationFollowLatest) return;
    requestAnimationFrame(() => {
      if (!conversationFollowLatest) return;
      conversationScroll.scrollTo({ top: conversationScroll.scrollHeight, behavior: "auto" });
    });
  }

  function appendUserMessage(content, options = {}) {
    revealConversation();
    const row = document.createElement("article");
    row.className = "message-row user-row";
    row.innerHTML = `
      <div class="message-stack user-stack">
        <div class="user-bubble">${escapeHtml(content)}</div>
        <div class="message-meta user-meta">${formatTime(options.timestamp)}</div>
      </div>
      <div class="avatar user-avatar" aria-hidden="true">我</div>`;
    const shouldRender = options.render !== false;
    const targetTask = options.task || activeTask;
    if (shouldRender) conversation.appendChild(row);
    if (options.persist !== false && targetTask) {
      const wasFirstUserMessage = !targetTask.messages?.filter((message) => message.role === "user").length;
      targetTask.messages = [...(targetTask.messages || []), { role: "user", content: String(content || "").slice(0, 4000), createdAt: Date.now() }].slice(-80);
      if (wasFirstUserMessage || targetTask.title === "新对话") targetTask.title = String(content || "");
      touchTask(targetTask);
    }
    if (shouldRender) scrollToLatest();
  }

  // ============ 关键修复 2: appendAssistantMessage 添加显式保存 ============
  function appendAssistantMessage(answer, meta, sourceCount, options = {}) {
    answer = normalizeAnswerPayload(answer, options.fallbackAnswer || options.fallbackText || "");
    meta = meta || {};
    options = options || {};
    const targetTask = options.task || activeTask;
    const shouldRender = options.render !== false;
    const thinkingEnabled = options.thinkingEnabled !== undefined ? Boolean(options.thinkingEnabled) : deepseekThinkingEnabled;
    const createdAt = options.timestamp || Date.now();
    let solutionCache = options.solution ? { ...options.solution } : null;
    const answerList = (field, limit = 16) => Array.from(new Set(
      (Array.isArray(answer[field]) ? answer[field] : [])
        .map((item) => String(item || "").trim())
        .filter(Boolean),
    )).slice(0, limit);
    const sourceFiles = Array.from(new Set(
      (options.sources || []).filter((source) => ["user_upload", "private"].includes(source?.source_type))
        .map((source) => source.file_name || source.title || source.source || "用户材料")
        .concat(options.sourceFiles || [])
        .map((name) => String(name || "").trim())
        .filter(Boolean),
    )).slice(0, 6);
    if (shouldRender) revealConversation();
    const storedSteps = Array.isArray(answer.action_steps) && answer.action_steps.length
      ? answer.action_steps
      : (Array.isArray(answer.suggestions) ? answer.suggestions : []);
    const actionSteps = Array.from(new Set(
      storedSteps
        .map((step) => String(step || "").trim())
        .filter(Boolean),
    )).slice(0, 4);
    const keyIssues = answerList("key_issues", 3);
    const evidenceAnalysis = answerList("evidence_analysis", 4);
    const defenseArguments = answerList("defense_arguments", 2);
    const documentChecklist = answerList("document_checklist", 5);
    const questionsToConfirm = answerList("questions_to_confirm", 3);
    const caseAnalysis = answerList("case_analysis", 5);
    const legalBasis = answerList("legal_basis", 4);
    const relatedCases = answerList("related_cases", 2);
    const suggestions = answerList("suggestions", 2);
    const citations = Array.isArray(answer.citations) ? answer.citations.slice(0, 8) : [];
    const normalizedSources = normalizeSourceList(options.sources || [], citations);
    const mainAnswer = String(answer.answer || "").trim();
    const displayAnswer = buildDisplayAnswerMarkdown(answer, mainAnswer || suggestions.join("\n") || answer.risk_notice || "暂时没有生成可用答案。");
    const mainBody = renderAnswerMarkdown(displayAnswer, normalizedSources);
    const sourceStrip = renderSourceStrip(normalizedSources);
    const row = document.createElement("article");
    row.className = "message-row assistant-row";
    row.innerHTML = `
      <div class="avatar assistant-avatar" aria-hidden="true">${BRAND_ICON}</div>
      <div class="message-stack">
        <div class="message-label">知途助手</div>
        <div class="assistant-card">
          <div class="friendly-answer markdown-body answer-body">${mainBody}</div>
        </div>
        ${sourceStrip}
        ${sourceFiles.length ? `<div class="workspace-evidence" title="本次回答实际使用的用户上传材料"><span>已使用上传材料</span>${sourceFiles.map((name) => `<code>${escapeHtml(name)}</code>`).join("")}</div>` : ""}
        <div class="message-meta">
          <span>${formatTime(options.timestamp)}</span>
          ${sourceCount ? `<span>参考 ${sourceCount} 条材料</span>` : ""}
          ${meta.elapsed_ms ? `<span>${(Number(meta.elapsed_ms) / 1000).toFixed(1)} 秒</span>` : ""}
        </div>
        <div class="solution-actions">
          <button class="solution-open" type="button" aria-label="打开本轮对话生成的法律解决方案">查看法律解决方案</button>
        </div>
        <section class="solution-panel markdown-body" hidden></section>
      </div>`;

    if (shouldRender) {
      conversation.appendChild(row);
    }

    bindSourceInteractions(row, normalizedSources);
    const solutionAnswer = {
      ...answer,
      case_analysis: caseAnalysis,
      key_issues: keyIssues,
      evidence_analysis: evidenceAnalysis,
      defense_arguments: defenseArguments,
      action_steps: actionSteps,
      document_checklist: documentChecklist,
      questions_to_confirm: questionsToConfirm,
      legal_basis: legalBasis,
      related_cases: relatedCases,
      suggestions,
    };
    const solutionQuestion = options.question || targetTask?.messages?.filter((message) => message.role === "user").at(-1)?.content || "本次法律咨询";
    row.querySelector(".solution-open")?.addEventListener("click", (event) => {
      const panel = row.querySelector(".solution-panel");
      if (solutionCache) {
        toggleInlineSolution(event.currentTarget, panel, panel?.hidden ? solutionCache : null);
        return;
      }
      openGeneratedLegalSolution(event.currentTarget, solutionQuestion, solutionAnswer, normalizedSources, {
        inlinePanel: panel,
        onGenerated: (solution) => {
          solutionCache = solution;
          if (persistedMessage) {
            persistedMessage.solution = solution;
            touchTask(targetTask);
          }
        },
      });
    });

    const persistedMessage = options.persist !== false && targetTask ? {
      role: "assistant",
      content: String(answer.answer || ""),
      answer: {
        answer: String(answer.answer || ""),
        case_analysis: caseAnalysis,
        key_issues: keyIssues,
        evidence_analysis: evidenceAnalysis,
        defense_arguments: defenseArguments,
        action_steps: actionSteps,
        document_checklist: documentChecklist,
        questions_to_confirm: questionsToConfirm,
        legal_basis: legalBasis,
        related_cases: relatedCases,
        risk_notice: String(answer.risk_notice || ""),
        suggestions,
      },
      question: String(options.question || ""),
      meta: { elapsed_ms: meta.elapsed_ms },
      sourceCount: Number(sourceCount || 0),
      sourceFiles,
      sources: normalizeStoredSources(normalizedSources),
      thinkingEnabled,
      createdAt,
      solution: solutionCache,
      solutionCacheKey: solutionCache?.cache_key || options.solutionCacheKey || "",
      turnId: String(options.turnId || ""),
    } : null;
    if (options.persist !== false && targetTask) {
      writeTaskMessage(targetTask, persistedMessage, { replace: Boolean(options.upsert) });
      // ✅ 显式保存
      saveTasks();
    }
    if (shouldRender) scrollToLatest();
    if (options.returnElement === true) return row;
  }

  function appendStreamingAssistantMessage(options = {}) {
    const thinkingEnabled = options.thinkingEnabled !== undefined ? Boolean(options.thinkingEnabled) : deepseekThinkingEnabled;
    revealConversation();
    const row = document.createElement("article");
    row.className = "message-row assistant-row";
    row._thinkingEnabled = thinkingEnabled;
    row._persistTask = options.task || activeTask;
    row._persistQuestion = String(options.question || "");
    row._turnId = String(options.turnId || "");
    row._streamSourceCount = Number(options.sourceCount || 0);
    row._streamSources = Array.isArray(options.sources) ? options.sources : [];
    row.innerHTML = `
      <div class="avatar assistant-avatar" aria-hidden="true">${BRAND_ICON}</div>
      <div class="message-stack">
        <div class="message-label">知途助手</div>
        <div class="assistant-card streaming-card">
          ${thinkingEnabled ? `<div class="thinking-inline" aria-live="polite"><span class="thinking-dot"></span><span>思考中</span></div>` : ""}
          <div class="friendly-answer markdown-body streaming-answer" hidden></div>
        </div>
        <div class="message-meta"><span>${formatTime()}</span><span class="streaming-status">正在生成</span></div>
      </div>`;
    conversation.appendChild(row);
    bindSourceInteractions(row, normalizeSourceList(options.sources || [], options.citations || []));
    scrollToLatest();
    return row;
  }

  function updateStreamingThinking(row, thinking) {
    if (!row || row._thinkingEnabled === false) return;
    const inline = row.querySelector(".thinking-inline span:last-child");
    if (inline) inline.textContent = "思考中";
  }

  function completeStreamingThinking(row) {
    const inline = row?.querySelector(".thinking-inline");
    if (inline) inline.hidden = true;
  }

  function updateStreamingAssistantMessage(row, answer, meta) {
    if (!row) return;
    answer = normalizeAnswerPayload(answer, row._streamedAnswerText || "");
    const preview = String(
      answer?.answer || "",
    );
    if (preview.trim()) row._streamedAnswerText = preview;
    const displayPreview = row._streamCompleted ? buildDisplayAnswerMarkdown(answer || {}, preview) : preview;
    const answerElement = row.querySelector(".streaming-answer");
    if (answerElement) {
      answerElement.hidden = !preview.trim();
      if (preview.trim()) completeStreamingThinking(row);
      answerElement.innerHTML = renderAnswerMarkdown(displayPreview, row._streamSources || []);
    }
    const status = row.querySelector(".streaming-status");
    if (status && !row._streamCompleted) status.textContent = meta?.elapsed_ms ? `${(meta.elapsed_ms / 1000).toFixed(1)} 秒` : "正在生成";
    scrollToLatest();
  }

  function persistAssistantMessage(answer, meta, sourceCount, options = {}) {
    appendAssistantMessage(answer, meta, sourceCount, { ...options, render: false });
  }

  // ============ 关键修复 3: finishStreamingAssistantMessage 添加显式保存 ============
  function finishStreamingAssistantMessage(row, answer, meta, sourceCount, options = {}) {
    if (!row) return false;
    answer = normalizeAnswerPayload(answer, row._streamedAnswerText || options.fallbackAnswer || "");
    if (row._streamFrame) {
      cancelAnimationFrame(row._streamFrame);
      row._streamFrame = null;
    }
    row._streamCompleted = true;
    const normalizedSources = normalizeSourceList(options.sources || [], answer?.citations || []);
    row._streamSources = normalizedSources;
    bindSourceInteractions(row, normalizedSources);
    updateStreamingAssistantMessage(row, answer, meta);
    const status = row.querySelector(".streaming-status");
    if (status) status.textContent = meta?.elapsed_ms ? `${(Number(meta.elapsed_ms) / 1000).toFixed(1)} 秒` : "已完成";
    const messageMeta = row.querySelector(".message-meta");
    if (messageMeta && sourceCount && !messageMeta.querySelector(".source-count")) {
      messageMeta.insertAdjacentHTML("beforeend", `<span class="source-count">参考 ${Number(sourceCount || 0)} 条材料</span>`);
    }
    const messageStack = row.querySelector(".message-stack");
    if (messageStack && normalizedSources.length && !messageStack.querySelector(".source-strip")) {
      const card = messageStack.querySelector(".assistant-card");
      card?.insertAdjacentHTML("afterend", renderSourceStrip(normalizedSources));
      bindSourceInteractions(row, normalizedSources);
    }
    if (messageStack && !messageStack.querySelector(".solution-actions")) {
      messageStack.insertAdjacentHTML("beforeend", `<div class="solution-actions"><button class="solution-open" type="button" aria-label="打开本轮对话生成的法律解决方案">查看法律解决方案</button></div><section class="solution-panel markdown-body" hidden></section>`);
      const solutionQuestion = options.question || options.task?.messages?.filter((message) => message.role === "user").at(-1)?.content || "本次法律咨询";
      row.querySelector(".solution-open")?.addEventListener("click", (event) => {
        const panel = row.querySelector(".solution-panel");
        if (row._solutionCache) {
          openSolutionDrawer(row._solutionCache);
          toggleInlineSolution(event.currentTarget, panel, panel?.hidden ? row._solutionCache : null);
          return;
        }
        openGeneratedLegalSolution(event.currentTarget, solutionQuestion, answer, normalizedSources, {
          inlinePanel: panel,
          onGenerated: (solution) => {
            row._solutionCache = solution;
            if (options.task) {
              const taskMessage = [...(options.task.messages || [])].reverse().find((message) => message.role === "assistant" && message.question === String(solutionQuestion || ""));
              if (taskMessage) {
                taskMessage.solution = solution;
                taskMessage.solutionCacheKey = solution.cache_key || taskMessage.solutionCacheKey || "";
                touchTask(options.task);
              }
            }
          },
        });
      });
    }
    if (options.solution) row._solutionCache = options.solution;
    persistAssistantMessage(answer, meta, sourceCount, options);
    // ✅ 显式保存
    if (options.task) {
      saveTasks();
    }
    scrollToLatest();
    return true;
  }

  // 流式过程中只更新当前 DOM，最终 complete 后再保存，避免历史里出现半成品消息。
  function queueStreamingAssistantUpdate(row, answer) {
    if (!row) return;
    row._streamAnswer = normalizeAnswerPayload(answer, row._streamedAnswerText || "");
    if (row._streamAnswer.answer) row._streamedAnswerText = row._streamAnswer.answer;
    if (row._streamFrame) return;
    row._streamFrame = requestAnimationFrame(() => {
      row._streamFrame = null;
      if (row._streamCompleted) return;
      updateStreamingAssistantMessage(row, row._streamAnswer, null);
    });
  }

  function updateLoadingStage(message, stage = "") {
    const loading = document.getElementById("loading-message");
    const text = loading?.querySelector(".loading-card p");
    if (text && message) text.textContent = message;
    if (stage) updateWorkflow(stage, "running", { message });
  }

  async function consumeSse(response, onEvent) {
    if (!response.body) throw new Error("浏览器不支持流式响应");
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let pending = "";

    function dispatch(block) {
      const lines = block.split(/\r?\n/);
      const eventName = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
      const dataText = lines
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!dataText) return;
      try {
        onEvent(eventName, JSON.parse(dataText));
      } catch (error) {
        if (error instanceof SyntaxError) return;
        throw error;
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value || new Uint8Array(), { stream: !done });
      let separator = pending.match(/\r?\n\r?\n/);
      while (separator?.index !== undefined) {
        dispatch(pending.slice(0, separator.index));
        pending = pending.slice(separator.index + separator[0].length);
        separator = pending.match(/\r?\n\r?\n/);
      }
      if (done) {
        if (pending.trim()) dispatch(pending);
        break;
      }
    }
  }

  function appendLoadingMessage() {
    revealConversation();
    const row = document.createElement("article");
    row.className = "message-row assistant-row";
    row.id = "loading-message";
    row.setAttribute("aria-live", "polite");
    row.innerHTML = `
      <div class="avatar assistant-avatar" aria-hidden="true">${BRAND_ICON}</div>
      <div class="message-stack">
        <div class="message-label">知途助手</div>
        <div class="loading-card">
          <div class="thinking-dots"><span></span><span></span><span></span></div>
          <div><strong>正在审慎分析</strong><p>检索案例、融合排序并生成回答，可能需要片刻。</p></div>
        </div>
      </div>`;
    conversation.appendChild(row);
    scrollToLatest();
  }

  function renderWorkspaceFiles(files) {
    const list = Array.isArray(files) ? files : [];
    window._lawragWorkspaceFiles = list;
    renderHomeMaterialSummary(list);
    renderMaterialPanel(list);
    renderFilesPage(list);
    updateWorkspaceFileStatusSummary(list);
    scheduleWorkspaceStatusRefresh(list);
  }

  function renderHomeMaterialSummary(files = []) {
    if (!homeMaterialSummary) return;
    const list = Array.isArray(files) ? files : [];
    homeMaterialSummary.hidden = !list.length;
    homeMaterialSummary.innerHTML = list.length
      ? list.slice(0, 4).map((file) => {
          const name = file.file_name || file.name || "证据材料";
          const state = workspaceFileMeta(file) || "已加入本次咨询";
          return `<span class="home-material-chip"><strong>${escapeHtml(name)}</strong><span>${escapeHtml(state)}</span></span>`;
        }).join("") + (list.length > 4 ? `<span class="home-material-chip"><strong>另有 ${list.length - 4} 份材料</strong></span>` : "")
      : "";
  }

  function updateWorkspaceFileStatusSummary(files = window._lawragWorkspaceFiles || []) {
    if (!fileStatus || !authenticated) return;
    const list = Array.isArray(files) ? files : [];
    if (!list.length) {
      fileStatus.className = "file-status";
      fileStatus.textContent = "材料用于补充案情和证据线索";
      return;
    }
    const processing = list.filter((file) => file.status === "processing").length;
    const stored = list.filter((file) => file.status === "stored").length;
    const failed = list.filter((file) => file.status === "failed").length;
    const ready = list.filter((file) => file.status === "ready").length;
    const extractedChars = list.reduce((total, file) => total + Number(file.extracted_char_count || 0), 0);
    if (processing || stored) {
      fileStatus.className = "file-status";
      fileStatus.textContent = `${list.length} 份材料，${processing} 份解析中，${stored} 份等待解析内容，${ready} 份已完成；已抽取 ${formatNumber(extractedChars)} 字`;
      return;
    }
    if (failed) {
      fileStatus.className = ready ? "file-status" : "file-status is-error";
      fileStatus.textContent = `${ready} 份材料解析完成，${failed} 份解析失败；已抽取 ${formatNumber(extractedChars)} 字`;
      return;
    }
    fileStatus.className = "file-status is-success";
    fileStatus.textContent = `${ready || list.length} 份证据材料已就绪，可用于补充本次案情；已识别 ${formatNumber(extractedChars)} 字`;
  }

  function hasPendingWorkspaceFiles(files = window._lawragWorkspaceFiles || []) {
    return Array.isArray(files) && files.some((file) => ["processing", "stored"].includes(String(file?.status || "")));
  }

  function stopWorkspaceStatusRefresh() {
    if (workspaceRefreshTimer) window.clearTimeout(workspaceRefreshTimer);
    workspaceRefreshTimer = null;
    workspaceRefreshDelay = 0;
  }

  function scheduleWorkspaceStatusRefresh(files = window._lawragWorkspaceFiles || []) {
    if (!ENABLE_FILE_UPLOADS) return;
    if (!authenticated || !sessionId || !hasPendingWorkspaceFiles(files)) {
      stopWorkspaceStatusRefresh();
      return;
    }
    if (workspaceRefreshTimer || workspaceRefreshInFlight) return;
    workspaceRefreshDelay = workspaceRefreshDelay ? Math.min(workspaceRefreshDelay + 1500, 12000) : 2500;
    workspaceRefreshTimer = window.setTimeout(async () => {
      workspaceRefreshTimer = null;
      await refreshWorkspaceFiles({ quiet: true });
    }, workspaceRefreshDelay);
  }

  async function deleteWorkspaceFile(documentId, fileName = "") {
    const target = String(documentId || "").trim();
    if (!target || !authenticated) return;
    hideWorkspaceFileLocally(target);
    fileStatus.className = "file-status is-success";
    fileStatus.textContent = "证据材料已从当前咨询移除，原文件、解析内容和检索数据会同步清理";
    try {
      const response = await fetch(backendUrl(`/api/v1/legal/files/${encodeURIComponent(target)}?session_id=${encodeURIComponent(sessionId)}`), { method: "DELETE", credentials: "include" });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        console.warn("材料服务器清理稍后重试", getErrorMessage(payload, `HTTP ${response.status}`));
      }
      await refreshWorkspaceFiles();
    } catch (error) {
      console.warn("材料服务器清理稍后重试", error);
    }
  }

  async function refreshWorkspaceFiles({ quiet = false } = {}) {
    if (!ENABLE_FILE_UPLOADS) return;
    if (workspaceRefreshInFlight) return;
    workspaceRefreshInFlight = true;
    try {
      if (!quiet) {
        materialDrawer?.classList.add("is-refreshing");
        if (materialSessionLabel) materialSessionLabel.textContent = "正在同步材料状态…";
      }
      const response = await fetch(backendUrl(`/api/v1/legal/files?session_id=${encodeURIComponent(sessionId)}`), { credentials: "include" });
      if (!response.ok) {
        if (workspaceLabel) workspaceLabel.textContent = authenticated ? "账户材料加载失败" : "未登录 · 请先登录";
        return;
      }
      const payload = await response.json();
      renderWorkspaceFiles(payload.data || []);
      const count = (payload.data || []).length;
      if (materialSessionLabel) materialSessionLabel.textContent = `当前会话 · ${count} 份材料`;
      if (workspaceLabel) workspaceLabel.textContent = `${currentUser?.username || "账户"} · ${count} 份材料`;
    } catch (_error) {
      if (workspaceLabel) workspaceLabel.textContent = authenticated ? "账户材料加载失败" : "未登录 · 请先登录";
    } finally {
      workspaceRefreshInFlight = false;
      if (!quiet) materialDrawer?.classList.remove("is-refreshing");
      scheduleWorkspaceStatusRefresh();
    }
  }

  function openFilePicker() {
    if (!ENABLE_FILE_UPLOADS) return;
    if (!authenticated) {
      openAuth("login");
      return;
    }
    if (fileInput) fileInput.value = "";
    fileInput?.click();
  }

  function bindFilePickerTrigger(button) {
    button?.addEventListener("click", openFilePicker);
  }

  async function uploadWorkspaceFiles(fileList) {
    if (!ENABLE_FILE_UPLOADS) return;
    const files = Array.from(fileList || []);
    if (!files.length || !fileStatus) return;
    if (!authenticated) {
      fileStatus.className = "file-status is-error";
      fileStatus.textContent = "请先登录，再上传个人材料";
      openAuth("login");
      return;
    }
    let uploadPassword = "";
    if (files.length > MAX_UPLOAD_FILES) {
      uploadPassword = (window.prompt(`本次选择了 ${files.length} 个文件，超过 ${MAX_UPLOAD_FILES} 个需要输入上传密码：`) || "").trim();
      if (!uploadPassword) {
        fileStatus.className = "file-status is-error";
        fileStatus.textContent = `本次上传已取消：超过 ${MAX_UPLOAD_FILES} 个文件需要密码`;
        fileInput.value = "";
        return;
      }
    }
    fileStatus.className = "file-status";
    fileStatus.textContent = `正在处理 ${files.length} 份材料…`;
    renderHomeMaterialSummary(files.map((file) => ({file_name: file.name, status: "processing"})));
    if (materialSessionLabel) materialSessionLabel.textContent = `正在同步上传状态…`;
    materialDrawer?.classList.add("is-uploading");
    await new Promise(requestAnimationFrame);
    let extractedChars = 0;
    const failedUploads = [];
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file, file.name));
      if (uploadPassword) formData.append("upload_password", uploadPassword);
      const response = await fetch(backendUrl(`/api/v1/legal/files/batch?session_id=${encodeURIComponent(sessionId)}`), { method: "POST", credentials: "include", body: formData });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(getErrorMessage(payload, "材料已提交，服务端稍后处理"));
      const results = Array.isArray(payload?.data?.files) ? payload.data.files : [];
      for (const result of results) {
        if (result.status === "failed") {
          failedUploads.push(`${result.file_name || "文件"}：已保存，稍后重试`);
          continue;
        }
        extractedChars += Number(result.extracted_char_count || 0);
      }
      const completed = Number(payload?.data?.uploaded_count || results.filter((item) => ["ready", "stored", "processing"].includes(item.status)).length);
      const processingCount = Number(payload?.data?.processing_count || results.filter((item) => item.status === "processing").length);
      const storedCount = Number(payload?.data?.stored_count || results.filter((item) => item.status === "stored").length);
      fileStatus.className = "file-status is-success";
      if (completed && failedUploads.length) {
        fileStatus.textContent = `${completed} 份材料已保存，${failedUploads.length} 份暂未完成解析；其中 ${processingCount} 份后台解析中，已抽取 ${formatNumber(extractedChars)} 字。稍后处理：${failedUploads.join("；")}`;
      } else if (completed) {
        if (processingCount) {
          fileStatus.textContent = `${completed} 份材料已保存，其中 ${processingCount} 份正在后台解析；已抽取 ${formatNumber(extractedChars)} 字`;
        } else {
          fileStatus.textContent = storedCount
            ? `${completed} 份材料已保存，其中 ${storedCount} 份等待解析内容；已抽取 ${formatNumber(extractedChars)} 字`
            : `${completed} 份证据材料已就绪，可用于补充本次案情；已识别 ${formatNumber(extractedChars)} 字`;
        }
      } else {
        fileStatus.textContent = "材料已提交，正在等待服务端保存和解析";
      }
      await refreshWorkspaceFiles();
    } catch (error) {
      console.warn("材料上传稍后同步", error);
      fileStatus.className = "file-status is-error";
      fileStatus.textContent = friendlyErrorMessage(error, "材料上传失败，请确认已登录、服务地址正确，并稍后重试");
    } finally {
      materialDrawer?.classList.remove("is-uploading");
      fileInput.value = "";
    }
  }

  function openAuth(mode = "login") {
    closeSettings();
    closeAccountMenu();
    setSidebarOpen(false);
    if (authBackdrop.hidden) authReturnFocus = document.activeElement;
    authMode = mode;
    authPanelOpen = true;
    authBackdrop.hidden = false;
    appShell.inert = true;
    document.body.classList.add("auth-modal-open");
    const registering = mode === "register";
    authTitle.textContent = registering ? "创建知途账户" : "登录知途";
    authSubmit.textContent = registering ? "创建账户" : "登录账户";
    authSwitch.textContent = registering ? "已有账户？登录" : "没有账户？注册";
    authEmailField.hidden = !registering;
    authEmail.disabled = !registering;
    authEmail.required = registering;
    if (!registering) authEmail.value = "";
    document.getElementById("auth-password").autocomplete = registering ? "new-password" : "current-password";
    authError.textContent = "";
    document.getElementById("auth-username")?.focus();
  }

  function closeAuth() {
    authPanelOpen = false;
    authBackdrop.hidden = true;
    appShell.inert = false;
    document.body.classList.remove("auth-modal-open");
    if (authReturnFocus?.isConnected) authReturnFocus.focus();
  }

  async function loadCurrentUser() {
    const requestIdentity = identityVersion;
    const requestAttempt = authAttemptVersion;
    if (authStatus) authStatus.textContent = "正在恢复登录状态…";
    if (workspaceLabel) workspaceLabel.textContent = "正在恢复登录状态";
    try {
      const response = await fetch(backendUrl("/api/v1/auth/me"), { credentials: "include" });
      if (!response.ok) throw new Error("登录已过期");
      const user = (await response.json()).data;
      if (requestIdentity !== identityVersion || requestAttempt !== authAttemptVersion) return;
      currentUser = user;
      authenticated = true;
      const userNamespace = `user_${currentUser.user_id}`;
      activateTaskNamespace(userNamespace);
      renderAccountSummary();
      renderAccountDeletionStatus(currentUser.account_deletion);
      try {
        await restoreServerConversationHistory();
        await loadUserSettings();
        await refreshWorkspaceFiles();
      } catch (syncError) {
        console.warn("登录后的工作区同步失败，已保留登录状态", syncError);
      }
      syncThinkingToggle();
    } catch (_error) {
      if (requestIdentity !== identityVersion || requestAttempt !== authAttemptVersion) return;
      authenticated = false;
      currentUser = null;
      renderAccountSummary();
      if (workspaceLabel) workspaceLabel.textContent = "未登录 · 公共法律库";
      renderAccountDeletionStatus();
    }
  }

  async function submitAuth(event) {
    event.preventDefault();
    const username = document.getElementById("auth-username")?.value.trim() || "";
    const password = document.getElementById("auth-password")?.value || "";
    const requestBody = { username, password };
    if (authMode === "register") {
      requestBody.email = authEmail?.value.trim() || "";
    }
    authSubmit.disabled = true;
    authError.textContent = "";
    const requestIdentity = identityVersion;
    const requestAttempt = ++authAttemptVersion;
    try {
      const response = await fetch(backendUrl(`/api/v1/auth/${authMode}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(requestBody),
      });
      const payload = await response.json().catch(() => null);
      if (requestIdentity !== identityVersion || requestAttempt !== authAttemptVersion) return;
      if (!response.ok) throw new Error(getErrorMessage(payload, "账户操作失败"));
      currentUser = payload.data;
      authenticated = true;
      activateTaskNamespace(`user_${currentUser.user_id}`);
      renderAccountSummary();
      renderAccountDeletionStatus(currentUser.account_deletion);
      closeAuth();
      navigate("home");
      try {
        await restoreServerConversationHistory();
        await loadUserSettings();
        await refreshWorkspaceFiles();
      } catch (syncError) {
        console.warn("登录后的工作区同步失败，已保留登录状态", syncError);
        if (authStatus) authStatus.textContent = `已登录：${currentUser.username}，部分工作区数据稍后同步`;
      }
      syncThinkingToggle();
    } catch (error) {
      authError.textContent = error.message || "账户操作失败";
    } finally {
      authSubmit.disabled = false;
    }
  }

  async function logout() {
    const requestAttempt = ++authAttemptVersion;
    const requestIdentity = identityVersion;
    try {
      const response = await fetch(backendUrl("/api/v1/auth/logout"), { method: "POST", credentials: "include" });
      if (requestIdentity !== identityVersion || requestAttempt !== authAttemptVersion) return;
      if (!response.ok) throw new Error("退出请求失败");
    } catch (_error) {
      if (requestIdentity !== identityVersion || requestAttempt !== authAttemptVersion) return;
      if (authStatus) authStatus.textContent = "退出失败，请检查连接后重试";
      return;
    }
    closeSettings();
    closeAccountMenu();
    authenticated = false;
    currentUser = null;
    activateTaskNamespace("guest");
    loadLocalSettings("guest");
    renderAccountSummary();
    if (workspaceLabel) workspaceLabel.textContent = "未登录 · 公共法律库";
    renderAccountDeletionStatus();
    renderWorkspaceFiles([]);
    syncThinkingToggle();
    navigate("home");
  }

  function fallbackClientAnswer(question, error) {
    const message = friendlyErrorMessage(error || new Error("请求中断"));
    const text = String(question || "").trim();
    const isFightDefense = /打|动手|反击|正当防卫|互殴|争执|伤/.test(text);
    const answer = isFightDefense
      ? `本次请求在传输阶段中断了（${message}），但不要按"无答案"处理。就你描述的争执和反击场景，核心是判断对方先动手时侵害是否正在进行、你的反击是否为了制止侵害、反击强度是否明显超过必要限度。\n\n你现在应立即固定证据：报警并完整说明对方先动手和你被迫反击的经过，保留监控、证人、录音录像、聊天记录，及时就医并保存病历、诊断证明、费用票据和伤情照片。做笔录时，如果内容没有写清"对方先动手、自己是被迫反击"，可以要求补充或更正，不要签署与你陈述不一致的笔录。\n\n后续维权可以围绕三件事推进：申请调取监控和证人材料，配合伤情鉴定；如被认定为互殴或处罚不当，在期限内考虑行政复议或行政诉讼；如自己受伤或财产受损，可结合证据向对方主张民事赔偿。`
      : `本次请求在传输阶段中断了（${message}），但不要按"无答案"处理。针对"${text || "这个问题"}"，你可以先按通用法律处理框架推进：明确事实经过、确认双方身份和权利义务关系、整理证据材料、核对是否存在期限要求，再决定协商、投诉、调解、仲裁或诉讼路径。\n\n现在最重要的是固定证据：保留合同、聊天记录、付款凭证、通知函、照片视频、证人信息、平台记录或行政机关材料；如果已经发生损失，同步整理损失金额、计算方式和票据。\n\n服务恢复后建议重新提问，以便系统结合本地知识库和可用来源生成带引用的完整答案。`;
    return {
      answer,
      case_analysis: isFightDefense
        ? [
            "实际争点：这类案件最关键不是谁吃亏更多，而是对方先动手时侵害是否正在进行，以及你的反击是否仍在制止侵害范围内。",
            "现有证据怎么看：监控、证人、接处警记录、病历和伤情照片会直接影响是否能证明对方先动手以及你的反击限度。",
            "法律上怎么判断：如果反击发生在不法侵害正在进行时，且强度没有明显超过必要限度，可以主张正当防卫；否则可能被认定为互殴或防卫过当。",
            "接下来最实际的动作：先报警、就医、保存证据和核对笔录，再根据警方认定结果决定是否申请复议、诉讼或索赔。",
          ]
        : [
            "实际争点：先确认请求权基础、关键事实和对方可能抗辩。",
            "现有证据怎么看：优先整理能证明事实发生、责任归属和损失金额的材料。",
            "法律上怎么判断：不同案由会影响管辖、时效、举证责任和处理路径。",
            "接下来最实际的动作：先固定证据和期限，再选择协商、投诉、调解、仲裁或诉讼。",
          ],
      action_steps: isFightDefense ? ["报警并固定现场证据。", "就医并保存病历票据。", "申请调取监控和寻找证人。", "核对笔录，不准确时要求补充。"] : ["梳理事实时间线。", "整理合同、沟通和付款等证据。", "确认是否存在时效或办理期限。", "服务恢复后重新提问获取带引用答案。"],
      document_checklist: isFightDefense ? ["接处警记录", "监控或现场视频", "证人联系方式", "病历和伤情照片", "聊天记录或其他沟通证据"] : ["合同或协议", "聊天记录或通知函", "付款和损失凭证", "照片视频或平台记录", "对方身份和联系方式"],
      questions_to_confirm: isFightDefense ? ["对方动手时有没有监控或证人？", "你反击时对方是否仍在持续攻击？", "双方伤情和警方处理结果是什么？"] : ["事情发生的准确时间和地点是什么？", "你现在已有哪几类证据？", "希望对方承担什么具体责任？"],
      risk_notice: "这是请求中断后的前端兜底提示，不替代后端检索结果；恢复服务后建议重新提问获取带引用的完整答案。",
    };
  }

  // ============ 关键修复 5: submitQuestion 中 complete 事件添加保存 ============
  function queueNextQuestion(requestIdentity) {
    const nextQuestion = queuedQuestions.shift();
    if (nextQuestion) window.setTimeout(() => {
      if (requestIdentity === identityVersion) submitQuestion(nextQuestion);
    }, 0);
  }

  async function submitQuestion(rawQuestion) {
    const question = String(rawQuestion || "").trim();
    if (!question) return;
    if (loading) {
      queuedQuestions.push(question);
      input.value = "";
      input.placeholder = `已加入发送队列（${queuedQuestions.length} 条），你还可以继续输入…`;
      sendButton.disabled = true;
      return;
    }

    const requestTask = activeTask;
    const requestIdentity = identityVersion;
    questionController = new AbortController();
    const requestTaskId = activeTaskId;
    const requestSessionId = sessionId;
    const requestTurnId = globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`;
    navigate("ask");
    appendUserMessage(question, { task: requestTask });
    input.value = "";
    setLoading(true);
    appendLoadingMessage();

    try {
      const response = await fetch(backendUrl("/api/v1/legal/ask/stream"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        credentials: "include",
        signal: questionController.signal,
        body: JSON.stringify({
          query: question,
          include_web: normalizeRetrievalMode(requestTask?.retrievalMode) === "force",
          retrieval_mode: normalizeRetrievalMode(requestTask?.retrievalMode),
          answer_detail: userSettings.answer_detail || "standard",
          thinking_enabled: deepseekThinkingEnabled,
          debug: false,
          session_id: requestSessionId,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(getErrorMessage(payload, `请求失败（${response.status}）`));
      }
      let streamingRow = null;
      let streamedAnswer = "";
      let completedPayload = null;
      const ensureRequestStreamingRow = () => {
        if (!streamingRow || !streamingRow.isConnected) {
          streamingRow = appendStreamingAssistantMessage({ task: requestTask, question, turnId: requestTurnId, thinkingEnabled: deepseekThinkingEnabled });
          if (streamedAnswer) queueStreamingAssistantUpdate(streamingRow, { answer: streamedAnswer });
        }
        return streamingRow;
      };
      await consumeSse(response, (eventName, data) => {
        if (requestIdentity !== identityVersion) return;
        const shouldRenderStream = activeTaskId === requestTaskId && currentRoute === "ask";
        if (eventName === "status") {
          if (shouldRenderStream) updateLoadingStage(data.message, data.stage);
        } else if (eventName === "step") {
          if (shouldRenderStream) {
            updateWorkflow(data.stage, data.status || "running", data);
            if (data.stage === "retrieval" && typeof data.web_used === "boolean") renderRetrievalMode(data.web_used);
          }
        } else if (eventName === "thinking") {
          if (!shouldRenderStream) return;
          document.getElementById("loading-message")?.remove();
          streamingRow = ensureRequestStreamingRow();
          updateStreamingThinking(streamingRow, data);
          if (String(data?.status || "") === "completed" && streamedAnswer) {
            queueStreamingAssistantUpdate(streamingRow, { answer: streamedAnswer });
          }
        } else if (eventName === "chunk") {
          if (!shouldRenderStream) return;
          document.getElementById("loading-message")?.remove();
          streamingRow = ensureRequestStreamingRow();
          streamedAnswer += String(data.delta || "");
          queueStreamingAssistantUpdate(streamingRow, { answer: streamedAnswer });
        } else if (eventName === "complete") {
        completedPayload = data;

        const completedData = data?.data || data;

        if (completedData?.meta?.elapsed_ms !== undefined) {
          updateWorkflowTotal(completedData.meta.elapsed_ms);
        }

        /*
        * ========================================
        * 关键修复：
        *
        * complete 一到，就立刻把 assistant
        * 真正写进当前 task.messages。
        *
        * 不能只 saveTasks()，
        * 因为原来的 saveTasks() 执行时，
        * assistant 消息还没有加入 requestTask。
        * ========================================
        */

        const completedAnswer = normalizeAnswerPayload(
          completedData?.answer || {},
          streamedAnswer
        );

        const completedMeta =
          completedData?.meta ||
          data?.meta ||
          {};

        const completedSources =
          Array.isArray(completedData?.sources)
            ? completedData.sources
            : [];

        persistAssistantMessage(
          completedAnswer,
          completedMeta,
          completedSources.length,
          {
            question,
            sources: completedSources,
            thinkingEnabled: deepseekThinkingEnabled,
            task: requestTask,
            turnId: requestTurnId,
            upsert: true,
            fallbackAnswer: streamedAnswer,
          }
        );
      } else if (eventName === "error") {
          if (!completedPayload && !streamedAnswer.trim()) throw new Error(data.message || "流式问答失败");
        }
      });
      if (requestIdentity !== identityVersion) return;
      if (!completedPayload) throw new Error("流式响应未正常结束");
      const resultData = completedPayload.data || completedPayload;
      const finalAnswer = normalizeAnswerPayload(resultData.answer || {}, streamedAnswer);
      const finalMeta = resultData.meta || completedPayload.meta;
      const finalSourceCount = resultData.sources?.length || 0;
      const finalOptions = {
        question,
        sources: resultData.sources || [],
        thinkingEnabled: deepseekThinkingEnabled,
        task: requestTask,
        turnId: requestTurnId,
        upsert: true,
        fallbackAnswer: streamedAnswer,
        render: activeTaskId === requestTaskId && currentRoute === "ask",
      };
      if (streamingRow) streamingRow._streamSources = normalizeSourceList(finalOptions.sources, finalAnswer.citations || []);
      if (finalOptions.render && !finishStreamingAssistantMessage(streamingRow, finalAnswer, finalMeta, finalSourceCount, finalOptions)) {
        appendAssistantMessage(finalAnswer, finalMeta, finalSourceCount, finalOptions);
      } else if (!finalOptions.render) {
        appendAssistantMessage(finalAnswer, finalMeta, finalSourceCount, finalOptions);
      }
      setServiceStatus("online");
    } catch (error) {
      if (requestIdentity !== identityVersion) return;
      appendAssistantMessage(
        fallbackClientAnswer(question, error),
        null,
        0,
        { task: requestTask, question, turnId: requestTurnId, upsert: true, render: activeTaskId === requestTaskId && currentRoute === "ask" },
      );
      setServiceStatus("offline");
    } finally {
      if (requestIdentity !== identityVersion) return;
      questionController = null;
      if (activeTaskId === requestTaskId && currentRoute === "ask") {
        document.getElementById("loading-message")?.remove();
        input.focus();
      }
      setLoading(false);
      queueNextQuestion(requestIdentity);
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion(input.value);
  });
  homeForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = homeInput.value.trim();
    if (!question) return;
    homeInput.value = "";
    homeSubmit.disabled = true;
    navigate("ask");
    submitQuestion(question);
  });
  homeInput?.addEventListener("input", () => { homeSubmit.disabled = !homeInput.value.trim(); });
  document.querySelectorAll("[data-home-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      homeInput.value = button.dataset.homePrompt;
      homeSubmit.disabled = false;
      homeInput.focus();
    });
  });
  homeAccountButton?.addEventListener("click", () => currentUser ? openSettings("account") : openAuth("login"));
  document.getElementById("auth-close-button")?.addEventListener("click", closeAuth);
  authBackdrop?.addEventListener("click", (event) => { if (event.target === authBackdrop) closeAuth(); });
  document.addEventListener("keydown", (event) => {
    if (authBackdrop.hidden) return;
    if (event.key === "Escape") { event.preventDefault(); closeAuth(); return; }
    if (event.key !== "Tab") return;
    const controls = Array.from(authDialog.querySelectorAll("button:not(:disabled), input:not(:disabled)"))
      .filter((element) => element.getClientRects().length);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  });
  conversationScroll?.addEventListener("scroll", () => {
    conversationFollowLatest = isConversationNearBottom();
  }, { passive: true });
  routeLinks.forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault();
    navigate(link.dataset.routeTarget);
  }));
  bindFilePickerTrigger(homeMaterialUpload);
  retrievalModeButtons.forEach((button) => button.addEventListener("click", () => setRetrievalMode(button.dataset.retrievalMode)));
  bindFilePickerTrigger(materialUploadAction);
  materialDrawerToggle?.addEventListener("click", toggleMaterialDrawer);
  document.getElementById("material-drawer-close")?.addEventListener("click", closeMaterialDrawer);
  settingsPageOpen?.addEventListener("click", () => openSettings());
  settingsPageAccount?.addEventListener("click", () => openSettings("account"));
  input.addEventListener("input", () => {
    sendButton.disabled = !input.value.trim();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitQuestion(input.value);
    }
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => submitQuestion(button.dataset.prompt));
  });
  document.querySelectorAll("[data-auth-mode]").forEach((button) => {
    button.addEventListener("click", () => openAuth(button.dataset.authMode || "register"));
  });
  newChatButton.addEventListener("click", createTask);
  newTaskCompact?.addEventListener("click", createTask);
  deleteTaskCompact?.addEventListener("click", () => deleteTask(activeTaskId));
  taskSelect?.addEventListener("change", () => switchTask(taskSelect.value));

  fileInput?.addEventListener("change", () => uploadWorkspaceFiles(fileInput.files));
  bindFilePickerTrigger(thinkingToggleButton);
  const isFileDrag = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
  form?.addEventListener("dragover", (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = ENABLE_FILE_UPLOADS ? "copy" : "none";
  });
  form?.addEventListener("drop", (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    uploadWorkspaceFiles(event.dataTransfer.files);
  });
  document.addEventListener("dragover", (event) => {
    if (isFileDrag(event)) event.preventDefault();
  });
  document.addEventListener("drop", (event) => {
    if (isFileDrag(event) && event.target !== form && !form?.contains(event.target)) event.preventDefault();
  });
  loginButton?.addEventListener("click", () => {
    if (!authenticated) { openAuth("login"); return; }
    accountMenu.hidden = !accountMenu.hidden;
    loginButton.setAttribute("aria-expanded", String(!accountMenu.hidden));
  });
  sidebarToggle?.addEventListener("click", () => {
    if (window.matchMedia("(max-width: 820px)").matches) {
      document.body.classList.remove("sidebar-collapsed");
      setSidebarOpen(!document.body.classList.contains("sidebar-open"));
    } else {
      document.body.classList.toggle("sidebar-collapsed");
      setSidebarOpen(false);
    }
    closeAccountMenu();
  });
  sidebarBackdrop?.addEventListener("click", () => { setSidebarOpen(false); sidebarToggle.focus(); });
  window.addEventListener("resize", () => {
    if (window.matchMedia("(max-width: 820px)").matches) document.body.classList.remove("sidebar-collapsed");
    setSidebarOpen(false);
  });
  document.getElementById("account-menu-settings")?.addEventListener("click", () => { closeAccountMenu(); openSettings(); });
  document.getElementById("account-menu-account")?.addEventListener("click", () => openSettings("account"));
  document.getElementById("account-menu-logout")?.addEventListener("click", logout);
  document.addEventListener("click", (event) => {
    if (!accountMenu.hidden && !event.target.closest(".sidebar-account-panel")) closeAccountMenu();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !authBackdrop.hidden || !settingsBackdrop.hidden) return;
    if (!accountMenu.hidden) { closeAccountMenu(); loginButton.focus(); }
    if (document.body.classList.contains("sidebar-open")) { setSidebarOpen(false); sidebarToggle.focus(); }
  });
  settingsLogoutButton?.addEventListener("click", logout);
  settingsButton?.addEventListener("click", () => openSettings());
  settingsClose?.addEventListener("click", closeSettings);
  settingsSave?.addEventListener("click", saveUserSettings);
  settingsTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectSettingsTab(tab.dataset.settingsTab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const direction = ["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1;
      const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? settingsTabs.length - 1 : (index + direction + settingsTabs.length) % settingsTabs.length;
      selectSettingsTab(settingsTabs[nextIndex].dataset.settingsTab);
      settingsTabs[nextIndex].focus();
    });
  });
  document.addEventListener("keydown", (event) => {
    if (settingsBackdrop.hidden || !authBackdrop.hidden || !diagnosticsBackdrop.hidden) return;
    if (event.key === "Escape") { event.preventDefault(); closeSettings(); return; }
    if (event.key !== "Tab") return;
    const controls = Array.from(settingsDialog.querySelectorAll("button:not(:disabled), input:not(:disabled), select:not(:disabled), summary"))
      .filter((element) => element.tabIndex >= 0 && element.getClientRects().length);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  });
  [settingAutoExpand, settingShowProcess, ...settingAnswerDetailOptions, settingTheme]
    .filter(Boolean)
    .forEach((control) => control.addEventListener("change", queueUserSettingsSave));
  accountDeletionButton?.addEventListener("click", handleAccountDeletionAction);
  activityRefresh?.addEventListener("click", loadActivity);
  settingsBackdrop?.addEventListener("click", (event) => { if (event.target === settingsBackdrop) closeSettings(); });
  diagnosticsClose?.addEventListener("click", closeDiagnostics);
  diagnosticsRefresh?.addEventListener("click", refreshDiagnostics);
  diagnosticsCopy?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(diagnosticsOutput?.textContent || "");
    } catch (_error) {
      /* clipboard permissions can be unavailable */
    }
  });
  diagnosticsBackdrop?.addEventListener("click", (event) => { if (event.target === diagnosticsBackdrop) closeDiagnostics(); });
  resetVersionClickHint();
  diagnosticsTriggers.forEach((trigger) => trigger.addEventListener("click", handleVersionClick));
  authSwitch?.addEventListener("click", () => openAuth(authMode === "login" ? "register" : "login"));
  authForm?.addEventListener("submit", submitAuth);
  settingBackendUrl?.addEventListener("input", () => {
    window.clearTimeout(backendUrlSaveTimer);
    if (settingsStatus) {
      settingsStatus.classList.remove("is-error");
      settingsStatus.textContent = "正在保存服务地址…";
    }
    backendUrlSaveTimer = window.setTimeout(() => saveBackendOriginFromForm(), 300);
  });
  settingBackendUrl?.addEventListener("change", () => saveBackendOriginFromForm());
  syncThinkingToggle();
  bindResizableLayout();
  setMaterialDrawerExpanded(false);
  renderWorkspaceFiles([]);
  renderAccountSummary();
  activateTaskNamespace("guest");
  applyBackendOriginToForm();
  renderSettingsPreview();
  navigate(window.location.hash || "#home", { replace: !window.location.hash });
  loadDesktopBackendOrigin().finally(() => loadCurrentUser());

  fetch(backendUrl("/health"))
    .then(async (response) => {
      if (!response.ok) throw new Error("offline");
      const payload = await response.json().catch(() => null);
      setServiceStatus("online");
    })
    .catch(() => setServiceStatus("offline"));
})();
