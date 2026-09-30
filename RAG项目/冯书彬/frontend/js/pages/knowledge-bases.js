import { getErrorMessage } from "../api/http.js"
import { listMaterials, publishMaterial, reviewMaterial, uploadMaterial } from "../api/knowledge-base-api.js"

const list = document.querySelector("#material-list")
const error = document.querySelector("#knowledge-error")
const uploadForm = document.querySelector("#upload-form")
const fileInput = document.querySelector("#material-file")
const fileName = document.querySelector("#file-name")
const titleInput = document.querySelector("#material-title")
const publisherInput = document.querySelector("#material-publisher")
const typeInput = document.querySelector("#material-type")
const contentInput = document.querySelector("#material-content")
const uploadButton = document.querySelector("#upload-button")
const uploadStatus = document.querySelector("#upload-status")
const refreshButton = document.querySelector("#refresh-button")

function tokenRoles() {
  const storedRoles = localStorage.getItem("roles")
  if (storedRoles) {
    try {
      const roles = JSON.parse(storedRoles)
      if (Array.isArray(roles)) return roles
    } catch (_error) {
      return storedRoles.split(",").map((role) => role.trim()).filter(Boolean)
    }
  }
  const token = localStorage.getItem("access_token")
  if (!token) return ["super_admin", "content_reviewer"]
  try {
    const rawPayload = token.split(".")[1]
    if (!rawPayload) return ["super_admin", "content_reviewer"]
    const base64 = rawPayload.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(rawPayload.length / 4) * 4, "=")
    const payload = JSON.parse(atob(base64))
    return Array.isArray(payload.roles) ? payload.roles : ["super_admin", "content_reviewer"]
  } catch (_error) {
    return ["super_admin", "content_reviewer"]
  }
}

const roles = tokenRoles()
const canReview = roles.includes("content_reviewer") || roles.includes("super_admin")
const canPublish = roles.includes("super_admin")

function setBusy(button, busy, busyText) {
  button.disabled = busy
  if (busy) {
    button.dataset.originalText = button.textContent
    button.textContent = busyText
    return
  }
  button.textContent = button.dataset.originalText || button.textContent
}

function statusLabel(status) {
  const labels = {
    pending_review: "待审核",
    reviewed: "已审核",
    rejected: "已拒绝",
    published: "已发布",
    deprecated: "已废止",
  }
  return labels[status] || status || "待审核"
}

function materialTitle(material) {
  const source = material.source_url || ""
  if (!source.startsWith("local-upload://")) return material.title || material.id
  return source.replace("local-upload://", "").replace(/^[0-9a-f-]{36}-/, "")
}

function renderMaterial(material) {
  const item = document.createElement("article")
  item.className = "material-card"
  item.dataset.testid = "material-item"

  const content = document.createElement("div")
  const title = document.createElement("h3")
  title.textContent = materialTitle(material)
  const meta = document.createElement("p")
  meta.textContent = `${material.publisher || "未知来源"} · ${material.material_type || "law"}`
  content.append(title, meta)

  const status = document.createElement("strong")
  status.className = `material-status status-${material.status || "pending_review"}`
  status.dataset.testid = "material-status"
  status.textContent = statusLabel(material.status)

  const actions = document.createElement("div")
  actions.className = "material-actions"

  const reviewButton = document.createElement("button")
  reviewButton.className = "button secondary small"
  reviewButton.dataset.testid = "review-material-button"
  reviewButton.textContent = "审核通过"
  reviewButton.addEventListener("click", async () => {
    try {
      error.textContent = ""
      setBusy(reviewButton, true, "审核中…")
      const updated = await reviewMaterial(material.id, "approved", "本地上传材料已核验")
      status.textContent = statusLabel(updated.status)
      status.className = `material-status status-${updated.status}`
      await loadMaterials()
    } catch (caught) {
      error.textContent = getErrorMessage(caught, "审核失败，请检查当前角色权限。")
    } finally {
      setBusy(reviewButton, false)
    }
  })
  reviewButton.hidden = !canReview || material.status !== "pending_review"

  const publishButton = document.createElement("button")
  publishButton.className = "button primary small"
  publishButton.textContent = "发布"
  publishButton.dataset.testid = "publish-material-button"
  publishButton.hidden = !canPublish || material.status !== "reviewed"
  publishButton.addEventListener("click", async () => {
    try {
      error.textContent = ""
      setBusy(publishButton, true, "发布中…")
      const updated = await publishMaterial(material.id)
      status.textContent = statusLabel(updated.status)
      status.className = `material-status status-${updated.status}`
      await loadMaterials()
    } catch (caught) {
      error.textContent = getErrorMessage(caught, "发布失败，仅超级管理员可以发布材料。")
    } finally {
      setBusy(publishButton, false)
    }
  })

  actions.append(reviewButton, publishButton)
  item.append(content, status, actions)
  return item
}

async function loadMaterials() {
  error.textContent = "正在加载知识库材料…"
  const result = await listMaterials()
  const materials = result.materials || result
  list.replaceChildren()
  if (!materials.length) {
    const empty = document.createElement("p")
    empty.className = "empty-state"
    empty.textContent = "暂无材料。请先上传一个本地文件。"
    list.append(empty)
  } else {
    materials.forEach((material) => list.append(renderMaterial(material)))
  }
  error.textContent = ""
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0]
  if (!file) return
  fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`
  if (!titleInput.value.trim()) titleInput.value = file.name.replace(/\.[^.]+$/, "")
  try {
    contentInput.value = await file.text()
  } catch (_error) {
    uploadStatus.textContent = "文件读取失败，请直接粘贴正文。"
  }
})

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault()
  const payload = {
    title: titleInput.value.trim(),
    publisher: publisherInput.value.trim() || "本地上传",
    material_type: typeInput.value,
    content: contentInput.value.trim(),
  }
  if (!payload.content) {
    uploadStatus.textContent = "请先选择文件或填写材料正文。"
    return
  }
  try {
    setBusy(uploadButton, true, "上传中…")
    uploadStatus.textContent = "正在提交到后端知识库接口…"
    const result = await uploadMaterial(payload)
    uploadStatus.textContent = `上传成功，材料状态：${statusLabel(result.material.status)}`
    uploadForm.reset()
    fileName.textContent = "支持 txt、md、html、json、csv，浏览器读取内容后提交。"
    publisherInput.value = "本地上传"
    await loadMaterials()
  } catch (caught) {
    uploadStatus.textContent = getErrorMessage(caught, "上传失败，请稍后重试。")
  } finally {
    setBusy(uploadButton, false)
  }
})

refreshButton.addEventListener("click", () => loadMaterials().catch((caught) => { error.textContent = getErrorMessage(caught, "知识库加载失败。") }))

loadMaterials().catch((caught) => { error.textContent = getErrorMessage(caught, "知识库加载失败，请检查权限。") })
