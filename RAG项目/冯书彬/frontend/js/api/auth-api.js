import { requestJson } from "./http.js"

async function requestLoginCode(phone, clientId = "browser") {
  return requestJson("/auth/code", { method: "POST", body: JSON.stringify({ phone, client_id: clientId }) })
}

async function loginWithCode(phone, code, clientId = "browser") {
  return requestJson("/auth/login", { method: "POST", body: JSON.stringify({ phone, code, client_id: clientId }) })
}

export { requestLoginCode, loginWithCode }
