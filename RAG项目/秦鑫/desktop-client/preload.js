const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("lawragDesktop", {
  applyServerUrl: (value) => ipcRenderer.invoke("desktop:apply-server-url", value),
  getConfig: () => ipcRenderer.invoke("desktop:get-config"),
  openExternal: (value) => ipcRenderer.invoke("desktop:open-external", value),
  reloadApp: () => ipcRenderer.invoke("desktop:reload"),
});
