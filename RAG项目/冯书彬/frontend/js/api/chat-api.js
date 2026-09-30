import { requestJson } from "./http.js"

async function sendMessage(conversationId, userId, text) {
  return requestJson(`/chat/conversations/${encodeURIComponent(conversationId)}/messages`, { method: "POST", body: JSON.stringify({ user_id: userId, text }) })
}

async function regenerateMessage(messageId, userId) {
  return requestJson(`/chat/messages/${encodeURIComponent(messageId)}/regenerate`, { method: "POST", body: JSON.stringify({ user_id: userId }) })
}

export { sendMessage, regenerateMessage }
