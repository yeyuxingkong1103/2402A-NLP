import AppConfig from "../config.js"
import { getErrorMessage } from "../api/http.js"
import { createExport, claimExportPassword, deleteAccount, downloadExport, listMemories, updateMemoryPreferences } from "../api/user-api.js"

const userId = localStorage.getItem("user_id") || AppConfig.devUserId

const memoryList = document.querySelector("#memory-list")
const status = document.querySelector("#profile-status")
const toggle = document.querySelector("#memory-toggle")
const exportButton = document.querySelector("#export-button")
const deleteButton = document.querySelector("#delete-account-button")

function setBusy(button, busy, busyText) {
  button.disabled = busy
  if (busy) {
    button.dataset.originalText = button.textContent
    button.textContent = busyText
    return
  }
  button.textContent = button.dataset.originalText || button.textContent
}

function renderMemories(memories = []) {
  memoryList.replaceChildren()
  if (!memories.length) {
    const empty = document.createElement("p")
    empty.textContent = "暂无长期记忆。"
    memoryList.append(empty)
    return
  }
  memories.forEach((memory) => {
    const item = document.createElement("p")
    item.textContent = memory.summary || "已保存一条脱敏记忆"
    memoryList.append(item)
  })
}

async function loadMemories() {
  status.textContent = "正在加载长期记忆…"
  const result = await listMemories(userId)
  renderMemories(result.memories || [])
  status.textContent = ""
}

toggle.addEventListener("change", async () => {
  try {
    toggle.disabled = true
    await updateMemoryPreferences(userId, toggle.checked)
    status.textContent = toggle.checked ? "已启用长期记忆" : "已关闭新增记忆提取"
  } catch (error) {
    toggle.checked = !toggle.checked
    status.textContent = getErrorMessage(error, "记忆设置保存失败，请稍后重试。")
  } finally {
    toggle.disabled = false
  }
})

exportButton.addEventListener("click", async () => {
  try {
    status.textContent = "正在准备导出文件…"
    setBusy(exportButton, true, "导出中…")
    const job = await createExport(userId)
    const downloaded = await downloadExport(userId, job.id)
    const password = await claimExportPassword(userId, job.id)
    const file = new Blob([JSON.stringify(downloaded.encrypted_zip)], { type: "application/json" })
    const link = document.createElement("a")
    link.href = URL.createObjectURL(file)
    link.download = `${job.id}.encrypted.json`
    link.click()
    URL.revokeObjectURL(link.href)
    status.textContent = `导出已下载，解密口令：${password.zip_password}`
  } catch (error) {
    status.textContent = getErrorMessage(error, "导出失败或已达到下载限制，请稍后重试。")
  } finally {
    setBusy(exportButton, false)
  }
})

deleteButton.addEventListener("click", async () => {
  if (!window.confirm("确定注销账号并删除个人数据吗？")) return
  try {
    setBusy(deleteButton, true, "注销中…")
    await deleteAccount(userId)
    localStorage.clear()
    window.location.href = "./login.html"
  } catch (error) {
    status.textContent = getErrorMessage(error, "账号注销失败，请稍后重试。")
    setBusy(deleteButton, false)
  }
})

loadMemories().catch((error) => { status.textContent = getErrorMessage(error, "记忆加载失败。") })
