import { getErrorMessage } from "../api/http.js"
import { loginWithCode, requestLoginCode } from "../api/auth-api.js"

const form = document.querySelector("#login-form")
const phoneInput = document.querySelector("#phone-input")
const codeInput = document.querySelector("#code-input")
const codeButton = document.querySelector("#request-code-button")
const loginButton = document.querySelector('[data-testid="login-button"]')
const codeStatus = document.querySelector("#code-status")
const errorMessage = document.querySelector("#error-message")

function showError(message) {
  errorMessage.textContent = message
}

function setBusy(button, busy, busyText) {
  button.disabled = busy
  if (busy) {
    button.dataset.originalText = button.textContent
    button.textContent = busyText
    return
  }
  button.textContent = button.dataset.originalText || button.textContent
}

codeButton.addEventListener("click", async () => {
  try {
    showError("")
    codeStatus.textContent = ""
    setBusy(codeButton, true, "发送中…")
    const result = await requestLoginCode(phoneInput.value.trim())
    codeStatus.textContent = result.sent === false ? "验证码已准备，请输入测试验证码。" : "验证码已发送，请查看短信。"
  } catch (error) {
    showError(getErrorMessage(error, "获取验证码失败，请稍后重试。"))
  } finally {
    setBusy(codeButton, false)
  }
})

form.addEventListener("submit", async (event) => {
  event.preventDefault()
  try {
    showError("")
    setBusy(loginButton, true, "登录中…")
    const tokens = await loginWithCode(phoneInput.value.trim(), codeInput.value.trim())
    localStorage.setItem("access_token", tokens.access_token)
    localStorage.setItem("refresh_token", tokens.refresh_token)
    localStorage.setItem("user_id", tokens.user_id)
    window.location.href = "./chat.html"
  } catch (error) {
    showError(getErrorMessage(error, "登录失败，请检查手机号和验证码。"))
  } finally {
    setBusy(loginButton, false)
  }
})
