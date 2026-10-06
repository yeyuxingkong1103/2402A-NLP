const fs = require("fs");
const os = require("os");
const path = require("path");

const { app, BrowserWindow, ipcMain, shell } = require("electron");

let mainWindow = null;
let currentServerUrl = defaultServerUrl();
const DESKTOP_BUILD_ID = "20260903-entry-route-v3";

function assetPath(name) {
  return path.join(__dirname, "assets", name);
}

function configFilePath() {
  return path.join(app.getPath("userData"), "config.json");
}

function legacyConfigFilePath() {
  return path.join(app.getPath("appData"), "law-rag-desktop-client", "config.json");
}

function parseCliServerUrl() {
  const entry = process.argv.find((value) => value.startsWith("--server="));
  if (entry) return entry.slice("--server=".length);
  const fallback = process.argv.find((value) => value.startsWith("--url="));
  return fallback ? fallback.slice("--url=".length) : "";
}

function normalizeServerUrl(value) {
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

function isPrivateIpv4(address) {
  return /^(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})$/.test(address);
}

function detectLanAddress() {
  const entries = [];
  const interfaces = os.networkInterfaces();
  for (const network of Object.values(interfaces)) {
    for (const item of network || []) {
      if (item && item.family === "IPv4" && !item.internal) {
        entries.push(item.address);
      }
    }
  }
  return entries.find(isPrivateIpv4) || entries[0] || "127.0.0.1";
}

function defaultServerUrl() {
  return normalizeServerUrl(process.env.LAWRAG_SERVER_URL) || normalizeServerUrl(readBundledConfig().serverUrl) || "";
}

function readBundledConfig() {
  try {
    const parsed = JSON.parse(fs.readFileSync(path.join(__dirname, "default-server.json"), "utf8"));
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (_error) {
    return {};
  }
}

function readSavedConfig() {
  try {
    const parsed = JSON.parse(fs.readFileSync(configFilePath(), "utf8"));
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (_error) {
    try {
      const legacyParsed = JSON.parse(fs.readFileSync(legacyConfigFilePath(), "utf8"));
      const legacyConfig = legacyParsed && typeof legacyParsed === "object" ? legacyParsed : {};
      if (legacyConfig.serverUrl) {
        writeSavedConfig(legacyConfig);
      }
      return legacyConfig;
    } catch (_legacyError) {
      return {};
    }
  }
}

function writeSavedConfig(nextConfig) {
  try {
    fs.mkdirSync(path.dirname(configFilePath()), { recursive: true });
    fs.writeFileSync(configFilePath(), JSON.stringify(nextConfig, null, 2), "utf8");
  } catch (_error) {
    /* ignore local config write failures */
  }
}

function resolveStartupUrl() {
  return (
    normalizeServerUrl(parseCliServerUrl()) ||
    normalizeServerUrl(process.env.LAWRAG_SERVER_URL) ||
    normalizeServerUrl(readSavedConfig().serverUrl) ||
    normalizeServerUrl(readBundledConfig().serverUrl) ||
    defaultServerUrl()
  );
}

function loadOfflinePage(reason = "") {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const options = reason ? { query: { reason } } : {};
  mainWindow.loadFile(path.join(__dirname, "offline.html"), options).catch(() => {
    /* keep the current window content if even the offline page cannot load */
  });
}

async function isServerReachable(url, timeoutMs = 5000) {
  const target = normalizeServerUrl(url);
  if (!target) return false;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const healthUrl = new URL("/health", target);
    const response = await fetch(healthUrl, { signal: controller.signal, cache: "no-store" });
    return response.ok;
  } catch (_error) {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function reachableStartupUrl(url) {
  const target = normalizeServerUrl(url);
  if (await isServerReachable(target)) return target;

  const bundledUrl = defaultServerUrl();
  if (bundledUrl && bundledUrl !== target && await isServerReachable(bundledUrl, 2500)) {
    return bundledUrl;
  }

  const localUrl = "http://127.0.0.1:7294";
  if (target !== localUrl && await isServerReachable(localUrl, 2500)) {
    return localUrl;
  }
  return "";
}

async function clearRendererCache() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  try {
    await mainWindow.webContents.session.clearCache();
    await mainWindow.webContents.session.clearStorageData({ storages: ["serviceworkers", "cachestorage"] });
  } catch (_error) {
    /* cache cleanup is best effort */
  }
}

function versionedServerUrl(url) {
  try {
    const target = new URL(url);
    target.searchParams.set("desktopBuild", DESKTOP_BUILD_ID);
    return target.toString();
  } catch (_error) {
    return url;
  }
}

function attachWindowGuards(win) {
  win.webContents.setWindowOpenHandler(({ url }) => {
    const target = normalizeServerUrl(url);
    if (target && target === currentServerUrl) {
      if (!win.isDestroyed()) win.focus();
      return { action: "deny" };
    }
    shell.openExternal(url).catch(() => {});
    return { action: "deny" };
  });

  win.webContents.on("will-navigate", (event, url) => {
    const target = normalizeServerUrl(url);
    if (target && target === currentServerUrl) return;
    if (/^https?:/i.test(url)) {
      event.preventDefault();
      shell.openExternal(url).catch(() => {});
    }
  });

  win.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL, isMainFrame) => {
    if (!isMainFrame) return;
    if (normalizeServerUrl(validatedURL) !== currentServerUrl) return;
    loadOfflinePage(errorDescription || `加载失败 (${errorCode})`);
  });
}

async function loadServerUrl(url) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const reachableUrl = await reachableStartupUrl(url);
  if (!reachableUrl) {
    loadOfflinePage(`无法连接到 ${url}`);
    return;
  }

  currentServerUrl = reachableUrl;
  try {
    await clearRendererCache();
    await mainWindow.loadURL(versionedServerUrl(reachableUrl));
  } catch (_error) {
    loadOfflinePage("无法连接到后端服务");
  }
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#263a32",
    show: false,
    title: "法衡",
    icon: assetPath("faheng.ico"),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  attachWindowGuards(mainWindow);
  mainWindow.once("ready-to-show", () => {
    if (!mainWindow.isDestroyed()) mainWindow.show();
  });
  setTimeout(() => {
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) {
      mainWindow.show();
    }
  }, 3000);
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  await loadServerUrl(currentServerUrl);
}

ipcMain.handle("desktop:get-config", () => ({
  configFile: configFilePath(),
  legacyConfigFile: legacyConfigFilePath(),
  savedServerUrl: normalizeServerUrl(readSavedConfig().serverUrl),
  serverUrl: currentServerUrl,
}));

ipcMain.handle("desktop:apply-server-url", async (_event, value) => {
  const nextUrl = normalizeServerUrl(value) || defaultServerUrl();
  currentServerUrl = nextUrl;
  writeSavedConfig({ serverUrl: nextUrl });
  await loadServerUrl(nextUrl);
  return { ok: true, serverUrl: currentServerUrl, savedServerUrl: nextUrl };
});

ipcMain.handle("desktop:reload", async () => {
  await loadServerUrl(currentServerUrl);
  return { ok: true };
});

ipcMain.handle("desktop:open-external", async (_event, url) => {
  await shell.openExternal(String(url || "").trim());
  return { ok: true };
});

app.setAppUserModelId("com.faheng.desktop");

app.whenReady().then(async () => {
  currentServerUrl = resolveStartupUrl();
  await createWindow();

  app.on("activate", async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      currentServerUrl = resolveStartupUrl();
      await createWindow();
    }
  });
});

app.on("second-instance", () => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
