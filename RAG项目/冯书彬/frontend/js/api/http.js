import AppConfig from "../config.js"

class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
  }
}

function isLocalBackend() {
  try {
    const url = new URL(AppConfig.apiBaseUrl, window.location.href)
    return ["127.0.0.1", "localhost"].includes(url.hostname)
  } catch (_error) {
    return false
  }
}

function currentUserId() {
  return localStorage.getItem("user_id") || AppConfig.devUserId
}

async function readError(response) {
  const contentType = response.headers.get("content-type") || ""
  if (contentType.includes("application/json")) {
    const payload = await response.json().catch(() => ({}))
    return payload.detail || payload.message || payload.error || "请求失败"
  }
  const text = await response.text().catch(() => "")
  return text || "请求失败"
}

async function request(path, options = {}) {
  const token = localStorage.getItem("access_token")
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) }
  if (token) headers.Authorization = `Bearer ${token}`
  if (!token && isLocalBackend()) headers["X-Dev-User-Id"] = currentUserId()

  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), options.timeout || AppConfig.requestTimeout)
  try {
    const response = await fetch(`${AppConfig.apiBaseUrl}${path}`, { ...options, headers, signal: controller.signal })
    if (response.status === 401) {
      throw new ApiError("当前接口需要登录；页面可继续预览。", { status: 401 })
    }
    if (!response.ok) {
      const detail = await readError(response)
      throw new ApiError(detail, { status: response.status, detail })
    }
    return response
  } catch (error) {
    if (error.name === "AbortError") throw new ApiError("请求超时，请稍后重试。")
    throw error
  } finally {
    window.clearTimeout(timeout)
  }
}

async function requestJson(path, options = {}) {
  const response = await request(path, options)
  if (response.status === 204) return null
  return response.json()
}

function getErrorMessage(error, fallback = "请求失败，请稍后重试。") {
  return error instanceof ApiError ? error.message : fallback
}

export { ApiError, getErrorMessage, request, requestJson }
