import { requestJson } from "../api/http.js"

function renderFeedback(container, messageId) {
  container.replaceChildren()
  const up = document.createElement("button")
  up.textContent = "有帮助"
  up.dataset.testid = "feedback-up"
  const down = document.createElement("button")
  down.textContent = "需改进"
  down.dataset.testid = "feedback-down"
  const submit = async (rating) => {
    await requestJson("/feedback", { method: "POST", body: JSON.stringify({ message_id: messageId, rating }) })
    container.textContent = "感谢反馈"
  }
  up.addEventListener("click", () => submit("up"))
  down.addEventListener("click", () => submit("down"))
  container.append(up, down)
}

export { renderFeedback }
