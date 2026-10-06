import { expect, test } from "@playwright/test"

test("chat sends authorized request, regenerates answer and submits feedback", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("access_token", "test-token")
    localStorage.setItem("user_id", "user-1")
  })
  await page.route("**/api/v1/chat/conversations/**/messages", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe("Bearer test-token")
    expect(await request.postDataJSON()).toMatchObject({ user_id: "user-1", text: "离婚时孩子抚养权一般怎么判断？" })
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "answered",
        conversation_id: "c1",
        message_id: "m1",
        answer: "初步判断需要结合具体事实。",
        citations: [{ title: "中华人民共和国民法典", location: "婚姻家庭编" }],
      }),
    })
  })
  await page.route("**/api/v1/chat/messages/m1/regenerate", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe("Bearer test-token")
    expect(await request.postDataJSON()).toMatchObject({ user_id: "user-1" })
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "answered", conversation_id: "c1", message_id: "m1", answer: "重新生成后的回答。", citations: [] }),
    })
  })
  await page.route("**/api/v1/feedback", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe("Bearer test-token")
    expect(await request.postDataJSON()).toMatchObject({ message_id: "m1", rating: "up" })
    await route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify({ id: "fb-1", message_id: "m1", rating: "up", risk_level: "normal" }) })
  })

  await page.goto("/frontend/pages/chat.html")
  await page.fill('[data-testid="chat-input"]', "离婚时孩子抚养权一般怎么判断？")
  await page.click('[data-testid="send-button"]')
  await expect(page.locator('[data-testid="assistant-message"]')).toContainText("初步判断")
  await expect(page.locator('[data-testid="citation-list"]')).toBeVisible()
  await page.click('[data-testid="regenerate-button"]')
  await expect(page.locator('[data-testid="assistant-message"]')).toContainText("重新生成后的回答")
  await page.click('[data-testid="feedback-up"]')
  await expect(page.locator('[data-testid="assistant-message"]')).toContainText("感谢反馈")
})
