import { expect, test } from "@playwright/test"

const runRealE2e = process.env.RUN_REAL_E2E === "1"

test.skip(!runRealE2e, "真实后端 E2E 需要先启动 8010 后端、Redis、Celery、MySQL、Milvus 和本地模型")

test("real backend chat page handles emergency guidance without mocks", async ({ page }) => {
  await page.goto("http://127.0.0.1:8010/")
  await page.evaluate(() => {
    localStorage.removeItem("access_token")
    localStorage.setItem("user_id", "real-e2e-user")
    localStorage.removeItem("chat_conversations")
  })
  await page.reload()

  await page.fill('[data-testid="chat-input"]', "我现在正在遭受家暴威胁，孩子也有危险，应该怎么办？")
  await page.click('[data-testid="send-button"]')

  const assistant = page.locator('[data-testid="assistant-message"]').last()
  await expect(assistant).toBeVisible({ timeout: 15_000 })
  await expect(assistant).toHaveAttribute("data-state", "emergency_guidance")
  await expect(assistant).toContainText("110")
  await expect(assistant).toContainText("12348")
  await expect(page.locator("#chat-error")).toHaveText("")
})
