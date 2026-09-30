import { expect, test } from "@playwright/test"

const runRealRagE2e = process.env.RUN_REAL_RAG_E2E === "1"
const realRagQuery = process.env.REAL_RAG_QUERY || "我在上海，已婚，有孩子，想离婚并争取抚养权，无危险，抚养权如何判断？"
const expectedAnswerText = process.env.REAL_RAG_EXPECT_TEXT || "抚养"
const expectedCitationText = process.env.REAL_RAG_EXPECT_CITATION || ""

test.skip(!runRealRagE2e, "真实知识库 RAG E2E 需要先启动 8010 后端、Milvus、MySQL、Redis、本地模型和已索引知识库")

test("real backend answers legal question with RAG citations", async ({ page }) => {
  await page.goto("http://127.0.0.1:8010/")
  await page.evaluate(() => {
    localStorage.removeItem("access_token")
    localStorage.setItem("user_id", "real-rag-e2e-user")
    localStorage.removeItem("chat_conversations")
  })
  await page.reload()

  await page.fill('[data-testid="chat-input"]', realRagQuery)
  await page.click('[data-testid="send-button"]')

  const assistant = page.locator('[data-testid="assistant-message"]').last()
  await expect(assistant).toBeVisible({ timeout: 60_000 })
  await expect(assistant).toHaveAttribute("data-state", "answered", { timeout: 60_000 })
  await expect(assistant).toContainText(expectedAnswerText)

  const citations = assistant.locator('[data-testid="citation-list"]')
  await expect(citations).toBeVisible()
  if (expectedCitationText) await expect(citations).toContainText(expectedCitationText)
  await expect(page.locator("#chat-error")).toHaveText("")
})
