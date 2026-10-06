import AppConfig from "../config.js"
import { getErrorMessage } from "../api/http.js"
import { sendMessage, regenerateMessage } from "../api/chat-api.js"
import { renderCitations } from "../components/citation-list.js"
import { renderDisclaimer } from "../components/disclaimer.js"
import { renderFeedback } from "../components/message-feedback.js"

const HISTORY_KEY = "chat_conversations"

const messageList = document.querySelector("#message-list")
const form = document.querySelector("#chat-form")
const input = document.querySelector("#message-input")
const sendButton = document.querySelector('[data-testid="send-button"]')
const error = document.querySelector("#chat-error")
const emptyState = document.querySelector("#empty-state")
const conversationList = document.querySelector("#conversation-list")
const newChatButton = document.querySelector("#new-chat-button")
const clearHistoryButton = document.querySelector("#clear-history-button")
const uploadLinkButton = document.querySelector("#upload-link-button")
const userId = localStorage.getItem("user_id") || AppConfig.devUserId
let conversationId = new URLSearchParams(window.location.search).get("conversation_id") || `web-${crypto.randomUUID()}`
let activeConversation = loadConversation(conversationId)

renderDisclaimer(document.querySelector("#disclaimer"))
renderHistory()
renderConversation()

function setButtonBusy(button, busy, busyText) {
  button.disabled = busy
  if (busy) {
    button.dataset.originalText = button.textContent
    button.textContent = busyText
    return
  }
  button.textContent = button.dataset.originalText || button.textContent
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]")
  } catch (_error) {
    return []
  }
}

function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 30)))
}

function loadConversation(id) {
  const existing = loadHistory().find((item) => item.id === id)
  return existing || { id, title: "新的法律咨询", messages: [], updatedAt: Date.now() }
}

function persistConversation() {
  activeConversation.updatedAt = Date.now()
  const history = loadHistory().filter((item) => item.id !== activeConversation.id)
  saveHistory([activeConversation, ...history])
  renderHistory()
}

function renderHistory() {
  const history = loadHistory()
  conversationList.replaceChildren()
  if (!history.length) {
    const empty = document.createElement("p")
    empty.className = "history-empty"
    empty.textContent = "暂无历史对话"
    conversationList.append(empty)
    return
  }
  history.forEach((conversation) => {
    const item = document.createElement("button")
    item.type = "button"
    item.className = conversation.id === conversationId ? "conversation-item active" : "conversation-item"
    item.innerHTML = `<span>${escapeHtml(conversation.title)}</span><small>${formatTime(conversation.updatedAt)}</small>`
    item.addEventListener("click", () => switchConversation(conversation.id))
    conversationList.append(item)
  })
}

function renderConversation() {
  messageList.replaceChildren()
  activeConversation.messages.forEach((message) => {
    if (message.role === "user") appendUserMessage(message.text, false)
    if (message.role === "assistant") appendAssistantMessage(message.result, false)
  })
  updateEmptyState()
  scrollToBottom()
}

function switchConversation(id) {
  conversationId = id
  activeConversation = loadConversation(id)
  window.history.replaceState({}, "", `?conversation_id=${encodeURIComponent(id)}`)
  renderHistory()
  renderConversation()
}

function startNewConversation() {
  conversationId = `web-${crypto.randomUUID()}`
  activeConversation = { id: conversationId, title: "新的法律咨询", messages: [], updatedAt: Date.now() }
  window.history.replaceState({}, "", `?conversation_id=${encodeURIComponent(conversationId)}`)
  renderHistory()
  renderConversation()
  input.focus()
}

function updateTitle(text) {
  if (activeConversation.title !== "新的法律咨询") return
  activeConversation.title = text.length > 18 ? `${text.slice(0, 18)}…` : text
}

function updateEmptyState() {
  const hasMessages = activeConversation.messages.length > 0
  emptyState.hidden = hasMessages
  messageList.hidden = !hasMessages
}

function scrollToBottom() {
  messageList.scrollTop = messageList.scrollHeight
}

function appendUserMessage(text, shouldPersist = true) {
  const wrapper = document.createElement("article")
  wrapper.className = "message user-message"
  wrapper.dataset.role = "user"
  wrapper.textContent = text
  messageList.append(wrapper)
  if (shouldPersist) {
    updateTitle(text)
    activeConversation.messages.push({ role: "user", text })
    persistConversation()
  }
  updateEmptyState()
  scrollToBottom()
}

function appendAssistantMessage(result, shouldPersist = true) {
  const wrapper = document.createElement("article")
  wrapper.className = "message assistant-message"
  wrapper.dataset.testid = "assistant-message"
  wrapper.dataset.state = result.status || "answered"

  const avatar = document.createElement("div")
  avatar.className = "message-avatar"
  avatar.textContent = "法"

  const body = document.createElement("div")
  body.className = "message-body"
  const answer = document.createElement("p")
  answer.textContent = result.answer || statusText(result.status)
  const citations = document.createElement("ul")
  renderCitations(citations, result.citations)
  const feedback = document.createElement("div")
  feedback.className = "message-feedback-wrap"
  if (result.message_id) renderFeedback(feedback, result.message_id)
  body.append(answer, citations, feedback)

  if (result.reason && result.status !== "answered") {
    const reason = document.createElement("small")
    reason.textContent = `状态：${result.reason}`
    body.append(reason)
  }

  wrapper.append(avatar, body)
  messageList.append(wrapper)
  if (shouldPersist) {
    activeConversation.messages.push({ role: "assistant", result })
    persistConversation()
  }
  updateEmptyState()
  scrollToBottom()
  return wrapper
}

function statusText(status) {
  if (status === "insufficient_basis") return "当前材料依据不足，暂不能给出可靠回答。"
  if (status === "emergency_guidance") return "检测到紧急风险，请优先联系当地紧急救助渠道。"
  if (status === "llm_unavailable") return "模型服务暂不可用，请稍后重试。"
  if (status === "answer_rejected") return "回答未通过引用校验，暂不展示。"
  return "暂无回答。"
}

async function handleRegenerate(wrapper, messageId, button) {
  try {
    error.textContent = ""
    setButtonBusy(button, true, "重新生成中…")
    const result = await regenerateMessage(messageId, userId)
    wrapper.dataset.state = result.status || "answered"
    wrapper.querySelector("p").textContent = result.answer || statusText(result.status)
    renderCitations(wrapper.querySelector("ul"), result.citations)
  } catch (caught) {
    error.textContent = getErrorMessage(caught, "重新生成失败，请稍后重试。")
  } finally {
    setButtonBusy(button, false)
  }
}

async function submitMessage(text) {
  if (!text) return
  try {
    error.textContent = ""
    input.value = ""
    input.disabled = true
    setButtonBusy(sendButton, true, "发送中…")
    appendUserMessage(text)
    const result = await sendMessage(conversationId, userId, text)
    const wrapper = appendAssistantMessage(result)
    const regenerate = document.createElement("button")
    regenerate.className = "inline-action"
    regenerate.textContent = "重新生成"
    regenerate.dataset.testid = "regenerate-button"
    regenerate.addEventListener("click", () => handleRegenerate(wrapper, result.message_id, regenerate))
    wrapper.querySelector(".message-body").append(regenerate)
  } catch (caught) {
    error.textContent = getErrorMessage(caught, "发送失败，请稍后重试。")
  } finally {
    input.disabled = false
    setButtonBusy(sendButton, false)
    input.focus()
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]))
}

function formatTime(value) {
  if (!value) return "刚刚"
  const date = new Date(value)
  return `${String(date.getMonth() + 1).padStart(2, "0")}/${String(date.getDate()).padStart(2, "0")}`
}

form.addEventListener("submit", (event) => {
  event.preventDefault()
  submitMessage(input.value.trim())
})

input.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return
  event.preventDefault()
  submitMessage(input.value.trim())
})

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt
    input.focus()
  })
})

newChatButton.addEventListener("click", startNewConversation)
clearHistoryButton.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY)
  startNewConversation()
})
uploadLinkButton.addEventListener("click", () => { window.location.href = "./knowledge-bases.html" })
